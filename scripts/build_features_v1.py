#!/usr/bin/env python3
"""Phase 1, step 1: EPS-derived features.

Builds data/features_v1.parquet -- one row per (ticker, period_end), the whole
of data/eps_panel.parquet carried through unchanged plus the features below.

This step is deliberately narrow. Only quantities derivable from EPS already in
the panel are computed: no prices, no new data sources, no model, no split.

    eps_growth_yoy          (eps_diluted - eps_year_ago_adjusted)
                            / abs(eps_year_ago_adjusted)
    label_lag_1..4          the label from the 1-4 prior quarters
    growth_streak           consecutive PRIOR quarters with label == 1, counted
                            backwards from this row, capped at 8
    eps_growth_yoy_lag_1..4 eps_growth_yoy from the 1-4 prior quarters
    growth_acceleration     eps_growth_yoy - eps_growth_yoy_lag_1
    quarters_available      how many prior quarters exist for this ticker

THE LEAKAGE RULE
----------------
Every lagged value must come from a row whose prediction_date is strictly
EARLIER than this row's prediction_date.

The panel is stored sorted by (ticker, period_end), and it is tempting to lag
with groupby("ticker").shift(k) on that order. This module does not, because
period_end order and prediction_date order are not the same thing: a late
filing can invert them, and a positional shift on period_end order would then
hand a row a "prior" quarter that in fact became public later. That is a
look-ahead which nothing downstream would reveal.

So rank is established by prediction_date within ticker, and lags are attached
by an explicit merge on that rank -- never by a shift, and never by period_end.
The result is then CHECKED rather than trusted: assert_no_lag_leakage compares
every lag's carried prediction_date against the row's own, on the built file,
and the check is itself fire-tested against an injected violation before use.

The panel is a fixed, read-only input. This script never writes to it.

Run:  python scripts/build_features_v1.py
"""

from __future__ import annotations

import os
import sys

import numpy as np
import pandas as pd

PANEL_PARQUET = "data/eps_panel.parquet"
FEATURES_PARQUET = "data/features_v1.parquet"

# Lags exposed as feature columns.
FEATURE_LAGS = (1, 2, 3, 4)

# growth_streak counts backwards up to this many quarters, so the lag machinery
# has to reach this far even though lags 5-8 are not exposed as features.
STREAK_CAP = 8
ALL_LAGS = tuple(range(1, STREAK_CAP + 1))

# Values carried from a prior row. prediction_date rides along so the leakage
# rule can be re-verified on the built file rather than taken on faith.
CARRIED_COLUMNS = ("prediction_date", "label_yoy", "eps_growth_yoy")

# Thresholds are searched over this many quantiles of each feature. Fine enough
# to find the useful cut, coarse enough to stay quick on ~24k rows.
THRESHOLD_GRID = 256

FEATURE_COLUMNS = (
    ["eps_growth_yoy"]
    + [f"eps_growth_yoy_lag_{k}" for k in FEATURE_LAGS]
    + ["growth_acceleration"]
    + [f"label_lag_{k}" for k in FEATURE_LAGS]
    + ["growth_streak", "quarters_available"]
)

# Features computed from THIS quarter's eps_diluted -- that is, from the same two
# numbers that define label_yoy.
#
# They are legitimately KNOWABLE at prediction_date: the filing is already
# public, which is what prediction_date means. They are still not legal
# PREDICTORS of label_yoy, because sign(eps_growth_yoy) reproduces the label
# exactly, and growth_acceleration carries eps_growth_yoy as a term. A model
# given either one scores ~100% and has learned nothing.
#
# They are built anyway: eps_growth_yoy is what the lags are lags OF, and both
# are asked for by the feature spec. The identity is verified at build time by
# report_target_identity rather than left as a warning nobody re-checks.
CONTEMPORANEOUS_COLUMNS = ("eps_growth_yoy", "growth_acceleration")

# Derived only from strictly earlier quarters. This is the set a model of
# label_yoy may actually draw on.
PRIOR_ONLY_COLUMNS = tuple(
    name for name in FEATURE_COLUMNS if name not in CONTEMPORANEOUS_COLUMNS
)


class LeakageError(AssertionError):
    """A lag was sourced from a row that was not strictly earlier."""


def section(title: str) -> None:
    print("\n" + "=" * 78)
    print(title)
    print("=" * 78)


def fail(message: str) -> None:
    print(f"\n*** FATAL: {message}", file=sys.stderr)
    sys.exit(1)


# --------------------------------------------------------------------------
# Stage 1 -- load and check preconditions
# --------------------------------------------------------------------------


def load_panel() -> pd.DataFrame:
    if not os.path.exists(PANEL_PARQUET):
        fail(f"{PANEL_PARQUET} not found. Run scripts/build_eps_panel.py first.")

    panel = pd.read_parquet(PANEL_PARQUET)
    print(f"panel rows              : {len(panel)}")
    print(f"tickers                 : {panel['ticker'].nunique()}")
    print(f"period_end span         : "
          f"{panel['period_end'].min().date()} .. {panel['period_end'].max().date()}")
    return panel


def check_preconditions(panel: pd.DataFrame) -> None:
    """What the ranking needs to be well defined before anything is built."""
    if panel["prediction_date"].isna().any():
        fail(f"{int(panel['prediction_date'].isna().sum())} rows have a null "
             f"prediction_date; the lag ordering would be undefined for them.")

    # A tie means two quarters of the same ticker became public on the same day.
    # Rank would then make one "prior" to the other at an EQUAL prediction_date,
    # which is not strictly earlier -- assert_no_lag_leakage would reject it. It
    # is caught here so the failure names the cause rather than the symptom.
    tied = panel.duplicated(subset=["ticker", "prediction_date"], keep=False)
    if int(tied.sum()):
        offenders = panel.loc[tied, ["ticker", "period_end", "prediction_date"]]
        print(offenders.head(20).to_string(index=False), file=sys.stderr)
        fail(f"{int(tied.sum())} rows share a (ticker, prediction_date) with "
             f"another row. Strict ordering is impossible for them.")

    print("precondition            : no null prediction_date, no (ticker, "
          "prediction_date) ties")


# --------------------------------------------------------------------------
# Stage 2 -- rank by prediction_date, attach lags by merge
# --------------------------------------------------------------------------


def attach_prediction_rank(panel: pd.DataFrame) -> pd.DataFrame:
    """Number each ticker's rows 0,1,2,... in PREDICTION_DATE order.

    This ordering, not the panel's period_end ordering, is what "prior quarter"
    means for the rest of this module.
    """
    ranked = panel.sort_values(["ticker", "prediction_date"]).copy()
    ranked["pred_rank"] = ranked.groupby("ticker").cumcount()

    # Rank counts the rows strictly before this one, which is exactly the
    # definition of quarters_available.
    ranked["quarters_available"] = ranked["pred_rank"]

    return ranked


def attach_lag(ranked: pd.DataFrame, k: int, columns=CARRIED_COLUMNS) -> pd.DataFrame:
    """Join each row to the row k places EARLIER in prediction_date order.

    An explicit merge on (ticker, pred_rank), not a shift: the join key states
    which row is being pulled in, so a re-sort of the frame cannot silently
    change the answer.
    """
    source = ranked[["ticker", "pred_rank", *columns]].copy()

    # The row at rank r serves the row at rank r + k.
    source["pred_rank"] = source["pred_rank"] + k
    source = source.rename(columns={name: f"{name}_lag_{k}" for name in columns})

    return ranked.merge(source, on=["ticker", "pred_rank"], how="left")


# --------------------------------------------------------------------------
# Stage 3 -- features
# --------------------------------------------------------------------------


def attach_eps_growth(ranked: pd.DataFrame) -> tuple:
    """eps_growth_yoy, with a zero year-ago base left NULL rather than infinite.

    Growth from a zero base is not a finite quantity. Dividing anyway yields
    +/-inf, which is not a missing-value marker: it survives correlations and
    means silently and poisons them. Null is already the panel's convention for
    a quantity that cannot be computed honestly, so it is used here too, and the
    count is reported separately from ordinary missingness.
    """
    ranked = ranked.copy()
    base = ranked["eps_year_ago_adjusted"]

    zero_base = base.notna() & (base == 0)
    computable = base.notna() & (base != 0)

    growth = (ranked["eps_diluted"] - base) / base.abs()
    ranked["eps_growth_yoy"] = growth.where(computable)

    return ranked, int(zero_base.sum()), int(base.isna().sum())


def attach_growth_streak(ranked: pd.DataFrame) -> pd.DataFrame:
    """Consecutive PRIOR quarters with label == 1, counted backwards, capped at 8.

    The current row's own label is never consulted -- that is the target. The
    walk stops at the first prior quarter that is not a confirmed 1, which
    includes a NULL label: an unresolvable split or a missing year-ago match is
    not evidence of a beat, so it breaks the streak rather than being skipped
    over as if the quarter did not exist.

    A row with no prior quarters gets 0. So does a row whose previous quarter
    missed; quarters_available is what distinguishes the two.
    """
    ranked = ranked.copy()

    streak = np.zeros(len(ranked), dtype=int)
    still_running = np.ones(len(ranked), dtype=bool)

    for k in ALL_LAGS:
        # label_yoy is a nullable Int64, so "== 1" yields NA (not False) on a
        # null label. fillna(False) is what turns "not known to be a beat" into
        # the intended break in the streak.
        beat = (ranked[f"label_yoy_lag_{k}"] == 1).fillna(False).to_numpy(dtype=bool)
        still_running &= beat
        streak += still_running

    ranked["growth_streak"] = streak
    return ranked


def build_features(panel: pd.DataFrame) -> tuple:
    section("STAGE 2 -- build features")

    ranked = attach_prediction_rank(panel)
    ranked, zero_base, no_year_ago = attach_eps_growth(ranked)
    print(f"eps_growth_yoy computed : {int(ranked['eps_growth_yoy'].notna().sum())}")
    print(f"    null, no year-ago   : {no_year_ago}")
    print(f"    null, zero base     : {zero_base}")

    for k in ALL_LAGS:
        ranked = attach_lag(ranked, k)
    print(f"lags attached           : {ALL_LAGS[0]}..{ALL_LAGS[-1]} by "
          f"(ticker, pred_rank) merge on prediction_date order")

    ranked = attach_growth_streak(ranked)

    # Exposed lag columns take their public names; lags 5-8 stay internal to the
    # streak walk and are dropped below.
    for k in FEATURE_LAGS:
        ranked[f"label_lag_{k}"] = ranked[f"label_yoy_lag_{k}"]

    ranked["growth_acceleration"] = (
        ranked["eps_growth_yoy"] - ranked["eps_growth_yoy_lag_1"]
    )

    return ranked, zero_base


def assemble_output(ranked: pd.DataFrame, panel_columns: list) -> pd.DataFrame:
    """Panel columns unchanged, then features, then the leakage audit trail."""
    audit_columns = [f"prediction_date_lag_{k}" for k in FEATURE_LAGS]

    output = ranked[panel_columns + FEATURE_COLUMNS + audit_columns].copy()
    return output.sort_values(["ticker", "period_end"]).reset_index(drop=True)


# --------------------------------------------------------------------------
# Stage 4 -- the leakage assertion, and the fire test that proves it works
# --------------------------------------------------------------------------


def assert_no_lag_leakage(frame: pd.DataFrame, lags=FEATURE_LAGS) -> None:
    """Every lag must carry a prediction_date STRICTLY earlier than the row's own.

    Raises LeakageError naming the offending rows. Runs on the built frame, so
    it checks what was actually produced rather than what was intended.
    """
    for k in lags:
        carried = f"prediction_date_lag_{k}"
        if carried not in frame.columns:
            raise LeakageError(f"{carried} missing; the lag cannot be verified.")

        present = frame[carried].notna()

        # Strictly earlier: an equal prediction_date is a violation too.
        violating = present & (frame[carried] >= frame["prediction_date"])

        if bool(violating.any()):
            offenders = frame.loc[
                violating, ["ticker", "period_end", "prediction_date", carried]
            ]
            raise LeakageError(
                f"lag {k}: {int(violating.sum())} rows carry a prediction_date "
                f"that is not strictly earlier than their own.\n"
                f"{offenders.head(10).to_string(index=False)}"
            )

        # A feature column may not be populated where its source row is absent.
        for name in (f"label_lag_{k}", f"eps_growth_yoy_lag_{k}"):
            orphaned = frame[name].notna() & ~present
            if bool(orphaned.any()):
                raise LeakageError(
                    f"{name}: {int(orphaned.sum())} rows hold a value with no "
                    f"source row behind it."
                )


def fire_test_leakage_assertion(frame: pd.DataFrame) -> None:
    """Prove the assertion fails on data it is supposed to reject.

    An assertion that has only ever seen clean data is not evidence of anything;
    it may be checking the wrong column, or nothing at all. Each injection below
    is a violation the rule exists to catch, and each must be caught.
    """
    section("FIRE TEST -- inject violations, require the assertion to catch them")

    usable = frame["prediction_date_lag_1"].notna()
    if not bool(usable.any()):
        fail("No row carries a lag_1; the fire test cannot run.")

    target = frame.index[usable][0]

    injections = {
        "lag prediction_date EQUAL to the row's own (not strictly earlier)":
            ("prediction_date_lag_1", frame.loc[target, "prediction_date"]),
        "lag prediction_date LATER than the row's own (a true look-ahead)":
            ("prediction_date_lag_1",
             frame.loc[target, "prediction_date"] + pd.Timedelta(days=90)),
    }

    for description, (column, value) in injections.items():
        corrupted = frame.copy()
        corrupted.loc[target, column] = value

        try:
            assert_no_lag_leakage(corrupted)
        except LeakageError:
            print(f"PASS  caught: {description}")
        else:
            fail(f"the leakage assertion did NOT catch: {description}. "
                 f"It cannot be trusted on the real data either.")

    # A feature value whose source row does not exist is the other failure mode.
    corrupted = frame.copy()
    orphan = corrupted.index[corrupted["prediction_date_lag_1"].isna()]
    if len(orphan):
        corrupted.loc[orphan[0], "label_lag_1"] = 1
        try:
            assert_no_lag_leakage(corrupted)
        except LeakageError:
            print("PASS  caught: a lag value with no source row behind it")
        else:
            fail("the leakage assertion did NOT catch an orphaned lag value.")

    # And the clean frame must still pass, or the test proves nothing.
    assert_no_lag_leakage(frame)
    print("PASS  the real built frame passes the same assertion")


def verify_features(frame: pd.DataFrame, panel: pd.DataFrame) -> None:
    section("ASSERTIONS")

    if len(frame) != len(panel):
        fail(f"feature table has {len(frame)} rows, panel has {len(panel)}.")
    print(f"PASS  row count matches the panel ({len(frame)})")

    duplicated = frame.duplicated(subset=["ticker", "period_end"], keep=False)
    if int(duplicated.sum()):
        fail(f"{int(duplicated.sum())} duplicate (ticker, period_end) rows.")
    print("PASS  one row per (ticker, period_end)")

    merged = frame.merge(
        panel[["ticker", "period_end", "eps_diluted", "label_yoy"]],
        on=["ticker", "period_end"],
        how="inner",
        suffixes=("", "_panel"),
    )
    if len(merged) != len(panel):
        fail(f"join back to the panel matched {len(merged)} of {len(panel)} rows.")
    if not merged["eps_diluted"].equals(merged["eps_diluted_panel"]):
        fail("eps_diluted was altered relative to the panel.")
    if not merged["label_yoy"].astype("Float64").equals(
        merged["label_yoy_panel"].astype("Float64")
    ):
        fail("label_yoy was altered relative to the panel.")
    print("PASS  every panel row is present and unmodified")

    assert_no_lag_leakage(frame)
    print(f"PASS  every lag {FEATURE_LAGS[0]}..{FEATURE_LAGS[-1]} comes from a "
          f"strictly earlier prediction_date")

    available = frame["quarters_available"]
    for k in FEATURE_LAGS:
        # A lag can only exist where at least k prior quarters do.
        impossible = frame[f"prediction_date_lag_{k}"].notna() & (available < k)
        if bool(impossible.any()):
            fail(f"lag {k} present on {int(impossible.sum())} rows with fewer "
                 f"than {k} prior quarters.")
    print("PASS  no lag exists where quarters_available says it cannot")

    streak = frame["growth_streak"]
    if not bool(((streak >= 0) & (streak <= STREAK_CAP)).all()):
        fail(f"growth_streak outside 0..{STREAK_CAP}.")
    if bool((streak > available).any()):
        fail(f"{int((streak > available).sum())} rows have a streak longer than "
             f"their available history.")
    print(f"PASS  growth_streak within 0..{STREAK_CAP} and within history")

    infinite = {
        name: int(np.isinf(frame[name].astype("float64")).sum())
        for name in FEATURE_COLUMNS
        if frame[name].dtype.kind in "fc" or str(frame[name].dtype) == "Float64"
    }
    if any(infinite.values()):
        fail(f"infinite values present: "
             f"{ {k: v for k, v in infinite.items() if v} }")
    print("PASS  no infinite values in any feature")


# --------------------------------------------------------------------------
# Reporting
# --------------------------------------------------------------------------


def best_threshold_accuracy(values: pd.Series, labels: pd.Series) -> tuple:
    """Best in-sample accuracy from a single cut on one feature.

    Both directions are tried, since a feature can be informative pointing
    either way. The threshold is chosen on the same rows it is scored on, so
    this is an OPTIMISTIC ceiling, not an estimate of held-out performance --
    it exists to rank features against the majority-class baseline, nothing more.
    """
    usable = values.notna() & labels.notna()
    if int(usable.sum()) == 0:
        return float("nan"), float("nan"), float("nan"), 0

    x = values[usable].astype("float64").to_numpy()
    y = labels[usable].astype("float64").to_numpy()

    baseline = max(y.mean(), 1 - y.mean())

    quantiles = np.linspace(0.0, 1.0, THRESHOLD_GRID)
    candidates = np.unique(np.quantile(x, quantiles))
    if len(candidates) < 2:
        return baseline, float("nan"), baseline, int(usable.sum())

    best_accuracy = -1.0
    best_cut = float("nan")
    for cut in candidates:
        above = x > cut
        # ">" predicts 1, and its mirror "<=" predicts 1.
        for accuracy in ((above == y).mean(), (~above == y).mean()):
            if accuracy > best_accuracy:
                best_accuracy = float(accuracy)
                best_cut = float(cut)

    return best_accuracy, best_cut, baseline, int(usable.sum())


def rank_correlation(values: pd.Series, labels: pd.Series) -> float:
    """Spearman, computed as Pearson on ranks so scipy is not a dependency.

    Worth having alongside Pearson because these growth features are violently
    heavy-tailed -- a handful of rows with a near-zero year-ago base produce
    growth in the millions, and a linear correlation on that is dominated by
    them. Rank correlation is what actually reflects monotone association here.
    """
    usable = values.notna() & labels.notna()
    if int(usable.sum()) < 2:
        return float("nan")

    x = values[usable].astype("float64").rank()
    y = labels[usable].astype("float64").rank()
    return float(x.corr(y))


def report_target_identity(frame: pd.DataFrame) -> None:
    """State, with the arithmetic checked, which features restate the label.

    label_yoy is 1 exactly when eps_diluted > eps_year_ago_adjusted, and
    eps_growth_yoy is (eps_diluted - eps_year_ago_adjusted) / |...|. The sign of
    one IS the other. This is not a look-ahead -- both are public on
    prediction_date -- but a model handed the feature is reading the answer.
    """
    section("TARGET IDENTITY -- features that restate the label")

    labels = frame["label_yoy"]
    growth = frame["eps_growth_yoy"]
    usable = growth.notna() & labels.notna()

    reproduced = int((
        (growth[usable] > 0).astype("int64") == labels[usable].astype("int64")
    ).sum())
    total = int(usable.sum())

    print(f"sign(eps_growth_yoy) reproduces label_yoy on {reproduced}/{total} rows")

    if reproduced == total:
        print("\n    eps_growth_yoy IS the label, in continuous form.")
        print("    growth_acceleration = eps_growth_yoy - eps_growth_yoy_lag_1,")
        print("    so it carries that same term and is equally disqualified.")
        print("\n    Both are knowable at prediction_date and both are built as")
        print("    specified -- eps_growth_yoy is what the lags are lags of --")
        print("    but NEITHER may be used to predict label_yoy.")
    else:
        print(f"\n    !!! expected an exact identity; {total - reproduced} rows "
              f"disagree. Investigate before using either column.")

    print(f"\nnot legal predictors of label_yoy : {list(CONTEMPORANEOUS_COLUMNS)}")
    print(f"safe to model on                  : {list(PRIOR_ONLY_COLUMNS)}")


def report(frame: pd.DataFrame, zero_base: int) -> None:
    labels = frame["label_yoy"]

    section("COVERAGE")
    print(f"rows                    : {len(frame)}")
    print(f"tickers                 : {frame['ticker'].nunique()}")
    print(f"rows with a label       : {int(labels.notna().sum())}")

    complete_lags = np.ones(len(frame), dtype=bool)
    for k in FEATURE_LAGS:
        complete_lags &= frame[f"label_lag_{k}"].notna().to_numpy()
    print(f"\nrows with label_lag_1..4 all present : {int(complete_lags.sum())} "
          f"({complete_lags.mean():.1%})")
    print(f"    of those, also labelled          : "
          f"{int((complete_lags & labels.notna()).sum())}")

    # The requirement as it would actually bite: a trainable row needs its own
    # label as well as the four lags.
    growth_lags = np.ones(len(frame), dtype=bool)
    for k in FEATURE_LAGS:
        growth_lags &= frame[f"eps_growth_yoy_lag_{k}"].notna().to_numpy()
    print(f"rows with eps_growth_yoy_lag_1..4 all present : {int(growth_lags.sum())} "
          f"({growth_lags.mean():.1%})")

    every_feature = np.ones(len(frame), dtype=bool)
    for name in FEATURE_COLUMNS:
        every_feature &= frame[name].notna().to_numpy()
    print(f"rows with EVERY feature present               : "
          f"{int(every_feature.sum())} ({every_feature.mean():.1%})")
    print(f"    and a label                               : "
          f"{int((every_feature & labels.notna()).sum())}")

    section("NULL COUNTS PER FEATURE")
    print(f"{'feature':28s} {'nulls':>8s} {'pct':>8s}   note")
    for name in FEATURE_COLUMNS:
        nulls = int(frame[name].isna().sum())
        note = ""
        if name == "eps_growth_yoy":
            note = (f"{zero_base} of these are a zero year-ago base "
                    f"(nulled, not infinite)")
        if name in ("growth_streak", "quarters_available"):
            note = "never null by construction"
        print(f"{name:28s} {nulls:8d} {nulls / len(frame):8.1%}   {note}")

    section("growth_streak DISTRIBUTION")
    counts = frame["growth_streak"].value_counts().sort_index()
    print(f"{'streak':>7s} {'rows':>7s} {'pct':>8s}  {'label rate':>11s}")
    for value, count in counts.items():
        group = labels[frame["growth_streak"] == value].dropna()
        rate = f"{group.mean():.1%}" if len(group) else "n/a"
        bar = "#" * min(int(count / len(frame) * 200), 60)
        print(f"{value:7d} {count:7d} {count / len(frame):8.1%}  {rate:>11s}  {bar}")

    # A streak of 0 means "no beat last quarter" OR "no history at all"; the two
    # are worth separating before anyone reads meaning into that bucket.
    no_history = int(((frame["growth_streak"] == 0) &
                      (frame["quarters_available"] == 0)).sum())
    print(f"\n    streak == 0 with no prior quarters at all: {no_history}")
    print(f"    streak == 0 after a prior quarter        : "
          f"{int((frame['growth_streak'] == 0).sum()) - no_history}")

    section("FEATURE vs LABEL")
    labelled = labels.notna()
    baseline = labels[labelled].astype("float64").mean()
    majority = max(baseline, 1 - baseline)
    print(f"labelled rows           : {int(labelled.sum())}")
    print(f"positive rate           : {baseline:.1%}")
    print(f"majority-class baseline : {majority:.1%}   <-- the number to beat")
    print("\nThresholds below are chosen on the same rows they are scored on.")
    print("They are an in-sample CEILING, not held-out performance.\n")

    print("Pearson is reported because it was asked for, but these growth")
    print("features are heavy-tailed enough that it is near-meaningless on them;")
    print("read the Spearman column instead. A row marked (=label) restates the")
    print("target and its accuracy is arithmetic, not signal.\n")

    print(f"{'feature':28s} {'n':>7s} {'pearson':>8s} {'spearman':>9s} "
          f"{'acc':>8s} {'base':>7s} {'lift':>7s}  {'cut':>10s}")
    for name in FEATURE_COLUMNS:
        values = frame[name].astype("Float64")
        usable = values.notna() & labelled

        if int(usable.sum()) < 2:
            print(f"{name:28s} {int(usable.sum()):7d}  {'n/a':>7s}")
            continue

        pearson = (
            values[usable].astype("float64")
            .corr(labels[usable].astype("float64"))
        )
        spearman = rank_correlation(values, labels)
        accuracy, cut, feature_baseline, count = best_threshold_accuracy(
            values, labels
        )
        flag = "  (=label)" if name in CONTEMPORANEOUS_COLUMNS else ""
        print(f"{name:28s} {count:7d} {pearson:8.3f} {spearman:9.3f} "
              f"{accuracy:8.1%} {feature_baseline:7.1%} "
              f"{accuracy - feature_baseline:+7.1%}  {cut:10.3f}{flag}")

    section("IF lag_1..4 ARE ALL REQUIRED")
    survivors = frame.loc[complete_lags]
    print(f"rows surviving          : {len(survivors)} of {len(frame)} "
          f"({len(survivors) / len(frame):.1%})")
    print(f"    with a label        : {int(survivors['label_yoy'].notna().sum())}")
    print(f"    tickers             : {survivors['ticker'].nunique()} of "
          f"{frame['ticker'].nunique()}")
    if len(survivors):
        print(f"    period_end span     : "
              f"{survivors['period_end'].min().date()} .. "
              f"{survivors['period_end'].max().date()}")
        survivor_labels = survivors["label_yoy"].dropna()
        if len(survivor_labels):
            print(f"    positive rate       : {survivor_labels.mean():.1%}  "
                  f"(vs {baseline:.1%} on the full panel)")
        removed = survivors["is_removed_name"].mean()
        print(f"    removed names       : {removed:.1%}  "
              f"(vs {frame['is_removed_name'].mean():.1%} on the full panel)")

    # Requiring four lags drops the early quarters of every ticker, which is a
    # systematic loss, not a random one. Worth seeing before anyone treats the
    # surviving set as representative.
    lost = frame.loc[~complete_lags]
    if len(lost):
        print(f"\n    rows lost           : {len(lost)}")
        print(f"    median quarters_available among lost rows: "
              f"{int(lost['quarters_available'].median())}")
        by_year = lost.groupby(lost["period_end"].dt.year).size()
        print(f"    lost rows by year   : "
              f"{ {int(y): int(c) for y, c in by_year.items()} }")


# --------------------------------------------------------------------------
# Entry point
# --------------------------------------------------------------------------


def main() -> None:
    section("STAGE 1 -- load the panel (read-only)")
    panel = load_panel()
    check_preconditions(panel)
    panel_columns = list(panel.columns)

    ranked, zero_base = build_features(panel)
    frame = assemble_output(ranked, panel_columns)

    fire_test_leakage_assertion(frame)
    verify_features(frame, panel)
    report_target_identity(frame)
    report(frame, zero_base)

    section("OUTPUT")
    os.makedirs("data", exist_ok=True)
    frame.to_parquet(FEATURES_PARQUET, index=False)
    print(f"features -> {FEATURES_PARQUET}  ({len(frame)} rows, "
          f"{len(frame.columns)} columns)")
    print(f"    panel columns carried through : {len(panel_columns)}")
    print(f"    features added                : {len(FEATURE_COLUMNS)}")
    print(f"    lag audit columns             : {len(FEATURE_LAGS)}")

    # Re-read and re-assert: the guarantee that matters is the one that holds on
    # the file other steps will actually load, not on the frame in memory.
    reloaded = pd.read_parquet(FEATURES_PARQUET)
    assert_no_lag_leakage(reloaded)
    print("\nPASS  leakage assertion re-verified on the written file")


if __name__ == "__main__":
    main()
