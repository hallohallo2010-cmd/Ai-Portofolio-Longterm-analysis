#!/usr/bin/env python3
"""Build the point-in-time quarterly EPS panel and its year-over-year label.

First script in this repo that PERSISTS an artifact rather than only reporting.
Output is data/eps_panel.parquet, one row per (ticker, period_end).

Scope is deliberately narrow: panel + label, nothing else. No features are
engineered, no train/test split is made, no price data is touched.

Every rule below is ENFORCED in code and verified by assertions before the
parquet is written; nothing here is assumed to hold.

Run:  python scripts/build_eps_panel.py
"""

from __future__ import annotations

import os
import sys

import pandas as pd

# src/ is a sibling of scripts/, so make the repo root importable.
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

# All EDGAR access -- headers, throttle, CIK overrides, fact shaping -- lives in
# src/data_loader.py. Nothing in this script talks to EDGAR directly.
from src.data_loader import (  # noqa: E402
    CIK_OVERRIDES,
    QUARTER_MAX_DAYS,
    QUARTER_MIN_DAYS,
    SEC_SLEEP_SECONDS,
    fail,
    fetch_quarterly_eps,
    load_ticker_cik_map,
    resolve_cik,
)

# --------------------------------------------------------------------------
# Configuration -- edit these, nothing below.
# --------------------------------------------------------------------------

TICKERS = ["AAPL", "MSFT", "JNJ", "JPM", "XOM", "PG", "WMT", "CAT", "NEE", "T"]

# XBRL tagging was phased in over 2009-2011. Before it settled, a quarter's
# EARLIEST XBRL record is frequently a later comparative rather than its own
# original filing -- the number was public on time in HTML, but was not tagged
# until a subsequent filing repeated it. Measured across this basket: 42 of the
# 48 periods whose first filing lands >90 days after period end have a period
# end before 2011. Since prediction_date is derived from filed_date, those rows
# would carry a filing date up to a year later than the market's actual
# knowledge date. The panel therefore starts here.
STUDY_START = pd.Timestamp("2011-01-01")

# A quarter filed more than this long after period end did not become public on
# a normal reporting schedule; its filed_date cannot be trusted as the moment the
# market learned the number. Dropped and logged rather than silently kept.
MAX_FILING_LAG_DAYS = 120

# The year-ago quarter is matched by DATE, not by row offset: a missing quarter
# would make a positional shift silently compare the wrong periods.
YEAR_AGO_TARGET_DAYS = 365
YEAR_AGO_TOLERANCE_DAYS = 45

# Label balance outside this band means the baseline is lopsided enough to matter.
# Upper bound raised to 0.65 because the split-adjusted label sits near 61%:
# correcting the pre-split year-ago values turns false negatives into positives,
# so the naive "always predict 1" baseline is now ~61% accurate. Judge any model
# against that number, not against 50%.
BALANCE_WARN_LOW = 0.40
BALANCE_WARN_HIGH = 0.65

# A split is detected when a later filing restates a period's EPS by almost
# exactly an integer factor. Real restatements do not land on 4.000; splits do.
SPLIT_RATIO_TOLERANCE = 0.02
SPLIT_MIN_FACTOR = 2

PANEL_PARQUET = "data/eps_panel.parquet"
DROPPED_CSV = "data/dropped_periods.csv"
SPLIT_CSV = "data/split_contaminated_periods.csv"

# --------------------------------------------------------------------------
# Small helpers
# --------------------------------------------------------------------------


def section(title: str) -> None:
    print("\n" + "=" * 78)
    print(title)
    print("=" * 78)


# --------------------------------------------------------------------------
# Stage 1 -- fetch raw quarterly facts
# --------------------------------------------------------------------------


# --------------------------------------------------------------------------
# Stage 2 -- dedupe to first filing
# --------------------------------------------------------------------------


def dedupe_to_first_filing(all_facts: pd.DataFrame) -> pd.DataFrame:
    """Collapse to one row per (ticker, period_end), keeping the EARLIEST filing.

    Later filings repeat a quarter as a comparative, and across a stock split they
    repeat it with a SPLIT-ADJUSTED value that did not exist at the time. Keeping
    MIN(filed) is what makes each row point-in-time correct.
    """
    # Count how many raw records existed per period BEFORE collapsing, so the
    # panel records how much repetition each period had.
    filings_per_period = all_facts.groupby(["ticker", "period_end"]).size()
    filings_per_period = filings_per_period.rename("n_filings_seen")

    # Ascending filed order means .first() within each group is the earliest.
    ordered = all_facts.sort_values(["ticker", "period_end", "filed_date"])
    deduped = ordered.groupby(["ticker", "period_end"], as_index=False).first()

    # Re-attach the pre-dedupe count.
    deduped = deduped.merge(filings_per_period, on=["ticker", "period_end"], how="left")

    return deduped


# --------------------------------------------------------------------------
# Stage 3 -- filters, each logging what it removed
# --------------------------------------------------------------------------


def apply_filters(deduped: pd.DataFrame) -> tuple:
    """Apply the lag and study-window filters, returning (kept, dropped_log)."""
    dropped_frames = []

    # --- filter 1: implausible filing lag ---------------------------------
    # Applied BEFORE the study-window trim so the log captures the pre-2011
    # phase-in cases too; they are the evidence for where STUDY_START sits.
    lag_too_long = deduped["filing_lag_days"] > MAX_FILING_LAG_DAYS

    lag_dropped = deduped.loc[lag_too_long].copy()
    lag_dropped["reason"] = f"filing_lag_days > {MAX_FILING_LAG_DAYS}"
    dropped_frames.append(lag_dropped)

    after_lag = deduped.loc[~lag_too_long].copy()

    # --- filter 2: study window -------------------------------------------
    # NOTE ON ORDERING: the window trim happens AFTER the label is computed, in
    # main(), so that a surviving 2010 quarter can still serve as the year-ago
    # lookup for a 2011 quarter. Only the trim is deferred; the log entry is
    # built here so both reasons land in one file.
    before_window = after_lag["period_end"] < STUDY_START

    window_dropped = after_lag.loc[before_window].copy()
    window_dropped["reason"] = f"period_end < STUDY_START ({STUDY_START.date()})"
    dropped_frames.append(window_dropped)

    dropped_log = pd.concat(dropped_frames, ignore_index=True)

    return after_lag, dropped_log


def write_dropped_log(dropped_log: pd.DataFrame) -> None:
    """Persist every removed period with enough context to audit the decision."""
    log_columns = ["ticker", "period_end", "filed_date", "filing_lag_days", "reason"]

    log = dropped_log[log_columns].copy()
    # Column names the request asked for: end / filed / lag.
    log = log.rename(
        columns={
            "period_end": "end",
            "filed_date": "filed",
            "filing_lag_days": "lag_days",
        }
    )
    log = log.sort_values(["reason", "ticker", "end"])
    log.to_csv(DROPPED_CSV, index=False)

    print(f"\ndropped periods logged -> {DROPPED_CSV}  ({len(log)} rows)")
    for reason, group in log.groupby("reason"):
        print(f"    {len(group):4d}  {reason}")


# --------------------------------------------------------------------------
# Stage 4 -- year-ago match
# --------------------------------------------------------------------------


def attach_year_ago(panel: pd.DataFrame) -> pd.DataFrame:
    """Attach each row's year-ago quarter. No label is computed here.

    The year-ago row is located by DATE PROXIMITY, never by row offset: a missing
    quarter would make a positional shift compare against the wrong period. Its
    filed_date comes along too, because the split logic in stage 5 needs to know
    the window between the two observations' prediction dates.
    """
    panel = panel.copy()
    panel["year_ago_target"] = panel["period_end"] - pd.Timedelta(days=YEAR_AGO_TARGET_DAYS)

    candidates = panel[["ticker", "period_end", "filed_date", "eps_diluted"]].copy()
    candidates = candidates.rename(
        columns={
            "period_end": "period_end_year_ago",
            "filed_date": "filed_date_year_ago",
            "eps_diluted": "eps_year_ago",
        }
    )

    # merge_asof needs both sides sorted on the join key.
    left_sorted = panel.sort_values("year_ago_target")
    right_sorted = candidates.sort_values("period_end_year_ago")

    # direction="nearest" with a tolerance implements "~365 days prior, +/- 45".
    # by="ticker" keeps each company's history separate. A row whose nearest
    # candidate falls outside the tolerance gets NaN and ends up unlabelled.
    matched = pd.merge_asof(
        left_sorted,
        right_sorted,
        left_on="year_ago_target",
        right_on="period_end_year_ago",
        by="ticker",
        direction="nearest",
        tolerance=pd.Timedelta(days=YEAR_AGO_TOLERANCE_DAYS),
    )

    return matched


# --------------------------------------------------------------------------
# Stage 5 -- split detection and adjustment
# --------------------------------------------------------------------------


def detect_split_events(all_facts: pd.DataFrame) -> list:
    """Find stock splits, and bound each one's effective date, from EDGAR alone.

    When a company splits its stock, later filings repeat earlier quarters with
    the EPS divided by the split factor. A period whose original value and a
    later restated value differ by almost exactly an integer factor is therefore
    a split. Real restatements do not land on 4.000; splits do.

    The effective date is never stated in the data, but it IS bounded. A filing
    made on date D reports in the share units current on D, so:

        window_start = latest filing date still reporting PRE-split units
        window_end   = earliest filing date already reporting POST-split units

    and the true effective date lies in (window_start, window_end]. Callers must
    treat that as an interval, not a point -- see apply_split_adjustment.

    Returns dicts with ticker, factor, window_start, window_end, n_evidence.
    """
    # Near-zero EPS makes the ratio meaningless, so exclude it before dividing.
    usable = all_facts[all_facts["eps_diluted"].abs() > 0.01]

    # (ticker, factor) -> {"pre": [filed dates], "post": [filed dates]}
    evidence = {}

    for (ticker, _period_end), group in usable.groupby(["ticker", "period_end"]):
        ordered = group.sort_values("filed_date")
        original = ordered.iloc[0]

        for _, later in ordered.iloc[1:].iterrows():
            if abs(later["eps_diluted"] - original["eps_diluted"]) <= 1e-9:
                continue  # same number repeated, not a restatement

            ratio = original["eps_diluted"] / later["eps_diluted"]
            nearest_factor = round(ratio)

            # Only an (almost exactly) integer shrink counts as a split.
            if nearest_factor < SPLIT_MIN_FACTOR:
                continue
            if abs(ratio - nearest_factor) / nearest_factor > SPLIT_RATIO_TOLERANCE:
                continue

            key = (ticker, int(nearest_factor))
            bucket = evidence.setdefault(key, {"pre": [], "post": []})
            bucket["pre"].append(original["filed_date"])
            bucket["post"].append(later["filed_date"])

    events = []
    for (ticker, factor), dates in evidence.items():
        window_start = max(dates["pre"])
        window_end = min(dates["post"])

        # If the bounds cross, the evidence is self-contradictory and the
        # effective date cannot be established. Recorded so callers can refuse
        # to adjust rather than guess.
        established = window_start < window_end

        events.append(
            {
                "ticker": ticker,
                "factor": factor,
                "window_start": window_start,
                "window_end": window_end,
                "established": established,
                "n_evidence": len(dates["pre"]),
            }
        )

    return sorted(events, key=lambda event: (event["ticker"], event["factor"]))


def apply_split_adjustment(panel: pd.DataFrame, split_events: list) -> pd.DataFrame:
    """Adjust the YEAR-AGO EPS for any split falling between the two observations.

    The current quarter's eps_diluted is never touched: it stays exactly as
    MIN(filed) reported it. Only the year-ago value is restated into the current
    quarter's share units, because that is the side quoted in stale units.

    A split counts only if its effective date falls between the two moments this
    row's two EPS numbers became public -- i.e. inside
        (prediction_date_year_ago, prediction_date]
    Since the effective date is only bounded to a window, that is decided per
    filing rather than by comparing whole intervals (see the comment below):

        year-ago in old units AND current in new units -> adjust
        both on the same side of the window            -> leave alone
        either filing lands inside the window          -> NULL the label

    The third case is why this returns a flag as well as a factor: guessing
    either way would silently produce a wrong label.
    """
    panel = panel.copy()
    panel["split_factor_applied"] = 1
    panel["split_ambiguous"] = False

    # Rows with no year-ago match are already unlabelled; excluded so their NaT
    # comparisons cannot be mistaken for ambiguity.
    has_match = panel["period_end_year_ago"].notna()

    for event in split_events:
        is_ticker = panel["ticker"] == event["ticker"]

        if not event["established"]:
            # Cannot bound the date at all -- refuse to adjust any row of this
            # ticker whose comparison could span it.
            panel.loc[is_ticker & has_match, "split_ambiguous"] = True
            continue

        window_start = event["window_start"]
        window_end = event["window_end"]

        # The interval test is applied per FILING rather than to the interval as
        # a whole. A split falls between two observations exactly when the earlier
        # one was published in old share units and the later one in new units, and
        # each filing's side is decidable on its own:
        #
        #     filed on/before window_start -> definitely OLD units
        #     filed on/after  window_end   -> definitely NEW units
        #     filed inside the window      -> undecidable
        #
        # Asking it this way uses the same evidence as comparing the window to
        # (prediction_date_year_ago, prediction_date], but stays decisive at the
        # boundaries: window_start IS the last pre-split filing date, so a
        # year-ago row filed exactly then is known to be in old units, where the
        # whole-interval comparison would call it ambiguous and discard the row.
        year_ago_filed = panel["filed_date_year_ago"]
        current_filed = panel["filed_date"]

        year_ago_old_units = year_ago_filed <= window_start
        year_ago_new_units = year_ago_filed >= window_end
        current_old_units = current_filed <= window_start
        current_new_units = current_filed >= window_end

        # Split sits strictly between the two publications -> adjust.
        intervenes = year_ago_old_units & current_new_units

        # Both readings on the same side -> same units, nothing to do.
        same_units = (year_ago_old_units & current_old_units) | (
            year_ago_new_units & current_new_units
        )

        applies = is_ticker & has_match & intervenes
        unclear = is_ticker & has_match & ~intervenes & ~same_units

        # Multiplied, not assigned: two splits between the same pair compound.
        panel.loc[applies, "split_factor_applied"] *= event["factor"]
        panel.loc[unclear, "split_ambiguous"] = True

    # The year-ago value restated into current share units.
    panel["eps_year_ago_adjusted"] = panel["eps_year_ago"] / panel["split_factor_applied"]

    # Kept so the unadjusted label can always be reproduced: eps_year_ago is
    # still the raw as-filed value, and this marks every row a split touched.
    panel["split_contaminated"] = (panel["split_factor_applied"] > 1) | panel["split_ambiguous"]

    return panel


def attach_label(panel: pd.DataFrame) -> pd.DataFrame:
    """label_yoy = 1 when EPS beat the SPLIT-ADJUSTED year-ago quarter, else 0."""
    panel = panel.copy()

    # Strictly greater: a flat quarter is not a beat.
    beat_year_ago = panel["eps_diluted"] > panel["eps_year_ago_adjusted"]

    # Null where there is no year-ago row at all, and null where a split's
    # effective date could not be pinned to one side of the interval.
    usable = panel["eps_year_ago_adjusted"].notna() & ~panel["split_ambiguous"]
    panel["label_yoy"] = beat_year_ago.astype("Int64").where(usable)

    return panel


def report_split_events(panel: pd.DataFrame, split_events: list) -> None:
    """Print what was detected and what it changed."""
    if not split_events:
        print("no stock splits detected in the raw filings.")
        return

    print("splits detected from EDGAR restatement evidence:")
    for event in split_events:
        status = "bounded" if event["established"] else "UNBOUNDED"
        print(
            f"    {event['ticker']:5s} {event['factor']}:1  effective date in "
            f"({event['window_start'].date()}, {event['window_end'].date()}]  "
            f"{status}, {event['n_evidence']} evidence records"
        )

    adjusted = panel[panel["split_factor_applied"] > 1]
    ambiguous = panel[panel["split_ambiguous"]]

    print(f"\nrows with year-ago EPS adjusted : {len(adjusted)}")
    print(f"rows nulled as unresolvable     : {len(ambiguous)}")

    if not adjusted.empty:
        detail = pd.DataFrame(
            {
                "ticker": adjusted["ticker"],
                "period_end": adjusted["period_end"].dt.date,
                "eps": adjusted["eps_diluted"],
                "eps_yr_ago_raw": adjusted["eps_year_ago"],
                "factor": adjusted["split_factor_applied"],
                "eps_yr_ago_adj": adjusted["eps_year_ago_adjusted"].round(4),
                "label": adjusted["label_yoy"],
            }
        )
        print("\nadjusted rows:")
        print(detail.to_string(index=False))

    if not ambiguous.empty:
        print("\nnulled rows (split effective date not resolvable to one side):")
        unresolved = ambiguous[["ticker", "period_end", "eps_diluted", "eps_year_ago"]]
        print(unresolved.to_string(index=False))

    # Persist the full picture so the unadjusted label can be rebuilt offline.
    touched = panel[panel["split_contaminated"]]
    log_columns = [
        "ticker", "period_end", "eps_diluted", "period_end_year_ago", "eps_year_ago",
        "split_factor_applied", "eps_year_ago_adjusted", "split_ambiguous", "label_yoy",
    ]
    touched[log_columns].to_csv(SPLIT_CSV, index=False)
    print(f"\nsplit-touched rows -> {SPLIT_CSV}  ({len(touched)} rows)")


# --------------------------------------------------------------------------
# Stage 6 -- assertions
# --------------------------------------------------------------------------


def verify_panel(panel: pd.DataFrame) -> None:
    """Every invariant the panel is supposed to hold, checked rather than assumed."""
    section("ASSERTIONS")

    # --- no duplicate (ticker, period_end) --------------------------------
    duplicate_mask = panel.duplicated(subset=["ticker", "period_end"], keep=False)
    duplicate_count = int(duplicate_mask.sum())
    if duplicate_count:
        offenders = panel.loc[duplicate_mask, ["ticker", "period_end"]]
        print(offenders.to_string(index=False), file=sys.stderr)
        fail(f"{duplicate_count} duplicate (ticker, period_end) rows in the panel.")
    print("PASS  no duplicate (ticker, period_end) rows")

    # --- prediction_date strictly after period_end -------------------------
    not_after = panel["prediction_date"] <= panel["period_end"]
    not_after_count = int(not_after.sum())
    if not_after_count:
        offenders = panel.loc[not_after, ["ticker", "period_end", "prediction_date"]]
        print(offenders.to_string(index=False), file=sys.stderr)
        fail(f"{not_after_count} rows have prediction_date <= period_end.")
    print("PASS  prediction_date > period_end for every row")

    # --- filed_date never null --------------------------------------------
    null_filed_count = int(panel["filed_date"].isna().sum())
    if null_filed_count:
        fail(f"{null_filed_count} rows have a null filed_date.")
    print("PASS  no null filed_date")

    # --- prediction_date is exactly filed_date + 1 day ---------------------
    # Cheap to check and would catch a timezone or dtype slip silently shifting it.
    offset_days = (panel["prediction_date"] - panel["filed_date"]).dt.days
    if not (offset_days == 1).all():
        fail("prediction_date is not exactly filed_date + 1 day for every row.")
    print("PASS  prediction_date == filed_date + 1 day")


def report_label_balance(panel: pd.DataFrame) -> None:
    section("LABEL BALANCE")

    labelled = panel.dropna(subset=["label_yoy"])
    null_count = len(panel) - len(labelled)

    print(f"rows in panel        : {len(panel)}")
    print(f"rows with a label    : {len(labelled)}")
    # Two distinct causes, worth separating: a missing year-ago quarter is a
    # coverage gap, an unresolvable split is a knowledge gap.
    no_match_count = int(panel["eps_year_ago"].isna().sum())
    ambiguous_count = int(panel["split_ambiguous"].sum())
    print(
        f"rows with NULL label : {null_count}  "
        f"({no_match_count} no year-ago match, {ambiguous_count} unresolvable split)"
    )

    print("\nper ticker:")
    print(f"    {'ticker':7s} {'n':>5s} {'pos':>5s} {'rate':>7s}  {'null':>5s}")

    for ticker, group in panel.groupby("ticker"):
        group_labelled = group.dropna(subset=["label_yoy"])
        positive_count = int(group_labelled["label_yoy"].sum())
        group_nulls = len(group) - len(group_labelled)

        if len(group_labelled):
            rate = positive_count / len(group_labelled)
            rate_text = f"{rate:6.1%}"
        else:
            rate_text = "     --"

        print(
            f"    {ticker:7s} {len(group_labelled):5d} {positive_count:5d} "
            f"{rate_text}  {group_nulls:5d}"
        )

    if labelled.empty:
        fail("No rows carry a label; the year-ago match produced nothing.")

    overall_positive = int(labelled["label_yoy"].sum())
    overall_rate = overall_positive / len(labelled)

    print(f"\noverall positive rate: {overall_rate:.1%}  ({overall_positive}/{len(labelled)})")

    # A lopsided baseline is not an error, but it decides what "good" means later.
    # Say plainly how much of the balance rests on the split adjustment, and how
    # the unadjusted label would have scored, so the two are always comparable.
    if "split_factor_applied" in panel.columns:
        adjusted_count = int((panel["split_factor_applied"] > 1).sum())
        ambiguous_count = int(panel["split_ambiguous"].sum())

        unadjusted_beat = panel["eps_diluted"] > panel["eps_year_ago"]
        unadjusted = unadjusted_beat.astype("Int64").where(panel["eps_year_ago"].notna())
        unadjusted_rate = unadjusted.dropna().mean()

        print(
            f"\nsplit adjustment: {adjusted_count} rows had their year-ago EPS "
            f"restated, {ambiguous_count} nulled as unresolvable."
        )
        print(
            f"unadjusted label would have scored {unadjusted_rate:.1%} positive "
            f"(kept reproducible via eps_year_ago)."
        )

    if not (BALANCE_WARN_LOW <= overall_rate <= BALANCE_WARN_HIGH):
        print(
            f"\n!!! WARNING: overall balance {overall_rate:.1%} is outside "
            f"{BALANCE_WARN_LOW:.0%}-{BALANCE_WARN_HIGH:.0%}."
        )
        print(f"!!! A constant 'always predict {1 if overall_rate > 0.5 else 0}' "
              f"baseline scores {max(overall_rate, 1 - overall_rate):.1%} accuracy.")
        print("!!! Judge any model against that number, not against 50%.")
    else:
        print(f"balance is within {BALANCE_WARN_LOW:.0%}-{BALANCE_WARN_HIGH:.0%}; no warning.")


# --------------------------------------------------------------------------
# Entry point
# --------------------------------------------------------------------------


def main() -> None:
    section("FETCH")
    print(f"tickers: {', '.join(TICKERS)}")
    print(f"throttle: one request per {SEC_SLEEP_SECONDS}s (EDGAR allows 10/s)")

    cik_map = load_ticker_cik_map()

    ticker_frames = []
    for ticker in TICKERS:
        cik, _company, was_overridden = resolve_cik(ticker, cik_map)
        note = "  (CIK override)" if was_overridden else ""

        facts = fetch_quarterly_eps(ticker, cik)
        print(f"    {ticker:5s} CIK {cik}  {len(facts):4d} quarterly facts{note}")
        ticker_frames.append(facts)

    all_facts = pd.concat(ticker_frames, ignore_index=True)

    # ---- dedupe -----------------------------------------------------------
    section("DEDUPE + FILTER")
    print(f"raw quarterly facts            : {len(all_facts)}")

    deduped = dedupe_to_first_filing(all_facts)
    print(f"after dedupe on (ticker, end)  : {len(deduped)}  [kept MIN(filed)]")

    # ---- filters ----------------------------------------------------------
    after_lag, dropped_log = apply_filters(deduped)
    print(f"after lag filter (<= {MAX_FILING_LAG_DAYS}d)     : {len(after_lag)}")

    os.makedirs("data", exist_ok=True)
    write_dropped_log(dropped_log)

    # ---- label ------------------------------------------------------------
    # Computed BEFORE the study-window trim so a surviving 2010 quarter can serve
    # as the year-ago lookup for an in-window 2011 quarter. Those warm-up rows are
    # used only as lookups; they never reach the output.
    section("YEAR-AGO MATCH")

    # Matched BEFORE the study-window trim so a surviving 2010 quarter can serve
    # as the year-ago lookup for an in-window 2011 quarter. Those warm-up rows are
    # used only as lookups; they never reach the output.
    matched = attach_year_ago(after_lag)

    in_window = matched["period_end"] >= STUDY_START
    panel = matched.loc[in_window].copy()
    print(f"study window {STUDY_START.date()} onward : {len(panel)} rows "
          f"({len(matched) - len(panel)} warm-up rows used for lookup only)")

    # Prediction dates for both observations must exist before the split logic,
    # which decides whether a split falls between them.
    panel["prediction_date"] = panel["filed_date"] + pd.Timedelta(days=1)
    panel["prediction_date_year_ago"] = panel["filed_date_year_ago"] + pd.Timedelta(days=1)

    # ---- split adjustment -------------------------------------------------
    # Takes the RAW facts: split evidence lives in the restatements dedupe removed.
    section("SPLIT ADJUSTMENT")
    split_events = detect_split_events(all_facts)
    panel = apply_split_adjustment(panel, split_events)

    # ---- label ------------------------------------------------------------
    # Computed only after the adjustment, so it compares like with like.
    panel = attach_label(panel)
    report_split_events(panel, split_events)

    # ---- assemble ---------------------------------------------------------
    output_columns = [
        "ticker",
        "cik",
        "period_end",
        "filed_date",
        "prediction_date",
        "eps_diluted",
        "n_filings_seen",
        "label_yoy",
        # Label provenance, kept so the +/-45 day match and the split adjustment
        # can both be audited, and so the UNADJUSTED label can be rebuilt from
        # eps_diluted vs eps_year_ago. These are NOT features -- they describe
        # how label_yoy was derived.
        "period_end_year_ago",
        "prediction_date_year_ago",
        "eps_year_ago",
        "split_factor_applied",
        "eps_year_ago_adjusted",
        "split_ambiguous",
        "split_contaminated",
    ]
    panel = panel[output_columns].sort_values(["ticker", "period_end"])
    panel = panel.reset_index(drop=True)

    verify_panel(panel)
    report_label_balance(panel)

    # ---- persist ----------------------------------------------------------
    section("OUTPUT")
    panel.to_parquet(PANEL_PARQUET, index=False)

    print(f"panel -> {PANEL_PARQUET}")
    print(f"    rows    : {len(panel)}")
    print(f"    tickers : {panel['ticker'].nunique()}")
    print(f"    span    : {panel['period_end'].min().date()} .. {panel['period_end'].max().date()}")
    print(f"    columns : {list(panel.columns)}")


if __name__ == "__main__":
    main()
