#!/usr/bin/env python3
"""Phase 3: score a committed forward-test prediction file. SCORING ONLY.

This is the other half of scripts/run_forward.py, and it is a separate script
on purpose. run_forward predicts and never scores; this scores and never
predicts. Nothing here fits a model, chooses a threshold, or reads
config/frozen_model.json for anything but a consistency check. The predicted
probabilities are taken from the committed file exactly as they were written.

    python scripts/score_forward.py data/live_predictions_2026-09-04.csv

THE TWO SCRIPTS MUST NEVER RUN IN THE SAME PASS
-----------------------------------------------
A forward test is evidence because a prediction was recorded before its outcome
existed. The record is the git commit: its timestamp is what a later reader can
check. If both halves ran together, there would be no interval to point at and
no commit standing between the prediction and the filing -- the file would just
be a description of things already known.

So this script refuses to score a prediction file that is not committed, or
whose working-tree contents differ from what was committed. And it does not
take the timestamp on trust: STAGE 4 finds the commit that ADDED the file and
asserts that it predates every filing it scores against. If that fails, the
predictions may still be interesting, but they are not a forward test, and the
script says so rather than printing a number.

WHAT IT MATCHES ON
------------------
For each predicted ticker, the scored quarter is the FIRST one filed after the
cut -- not the row whose period_end happens to equal expected_period_end. The
prediction was "this ticker's next report", and the next report is whichever one
arrived first. expected_period_end was recorded as an estimate and is used here
only to flag where the estimate missed, never to select.

The selection rule is written as `prediction_date > cut`, which is the exact
complement of run_forward's history rule (`prediction_date <= cut`). Stating it
that way rather than as `filed_date > cut` matters: prediction_date is
filed_date + 1, so a quarter filed ON the cut date belongs to the scored set,
and a `filed_date > cut` test would silently drop it and report the ticker as
still unfiled.

WHAT IT DOES NOT DO
-------------------
It does not re-gate on index membership. The prediction was committed for a
ticker that was a constituent at the cut; dropping it afterwards because the
company has since left would be deciding what to score after the outcome
existed, which is the selection this whole design exists to avoid. Membership at
the actual filing date is reported as information and nothing more.

It does not modify the prediction file or the panel. Both are hashed at the
start and re-checked at the end.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys

import numpy as np
import pandas as pd

# Metrics only. No estimator, no transformer, no pipeline -- there is nothing to
# fit here, and importing something that could fit would make that a convention
# rather than a fact about this file.
from sklearn.metrics import accuracy_score, brier_score_loss, log_loss

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

from src.data_loader import (  # noqa: E402
    SEC_SLEEP_SECONDS,
    describe_spans,
    fetch_quarterly_eps_spanned,
    load_recovered_ciks,
    load_ticker_cik_map,
    resolve_ticker_spans,
)
from src.index_membership import (  # noqa: E402
    build_intervals,
    is_member_at,
    load_tables,
    normalise_changes,
)

# The same construction rules the panel and the forward run used, imported
# rather than restated. A scored quarter that reached its label by a different
# route than a training quarter would not be the same measurement.
from scripts.build_eps_panel import (  # noqa: E402
    MAX_FILING_LAG_DAYS,
    STUDY_START,
    apply_lag_filter,
    apply_split_adjustment,
    attach_label,
    attach_year_ago,
    dedupe_to_first_filing,
    detect_split_events,
)

CONFIG_PATH = "config/frozen_model.json"
PANEL_PARQUET = "data/eps_panel.parquet"
SCORED_TEMPLATE = "data/scored_{cut}.csv"
FACT_CACHE_TEMPLATE = "data/score_cache_{scored_at}"

LABEL = "label_yoy"
CALIBRATION_DECILES = 10

# Header fields the prediction file must carry. Scoring against a file missing
# any of them would mean supplying the missing piece here, and the constant
# baseline in particular has to come from the prediction file's own training
# window -- never from the period being scored, which would be the baseline
# using the answer.
REQUIRED_HEADER = (
    "cut_date",
    "code_sha",
    "working_tree_clean",
    "config",
    "model",
    "decision_threshold",
    "panel_sha256",
    "training_rows_labelled",
    "training_positive_rate",
)


def section(title: str) -> None:
    print("\n" + "=" * 78)
    print(title)
    print("=" * 78)


def stop(message: str) -> None:
    """Anything that would make this something other than a scored forward test."""
    print(f"\n*** STOPPED: {message}", file=sys.stderr)
    print("*** Nothing was scored and no output was written.", file=sys.stderr)
    sys.exit(1)


def file_digest(path: str) -> str:
    with open(path, "rb") as handle:
        return hashlib.sha256(handle.read()).hexdigest()


def git(*args: str) -> tuple:
    """Run a git command; return (ok, stdout)."""
    try:
        result = subprocess.run(["git", *args], capture_output=True, text=True)
        return result.returncode == 0, result.stdout.strip()
    except FileNotFoundError:
        return False, ""


# --------------------------------------------------------------------------
# Stage 0 -- read the prediction file and everything it claims about itself
# --------------------------------------------------------------------------


def parse_header(path: str) -> dict:
    """Pull the `#` provenance block out of a prediction file.

    Values may carry a trailing parenthetical gloss, so callers that need a
    scalar take the first whitespace token.
    """
    header = {}
    with open(path) as handle:
        for line in handle:
            if not line.startswith("#"):
                break
            body = line[1:].strip()
            if ":" not in body:
                continue
            key, _sep, value = body.partition(":")
            key = key.strip()
            if key and key == key.lower() and " " not in key.strip():
                header[key] = value.strip()
    return header


def token(header: dict, key: str) -> str:
    return header[key].split()[0]


def load_predictions(path: str) -> tuple:
    section("STAGE 0 -- the prediction file")

    if not os.path.exists(path):
        stop(f"{path} not found.")

    header = parse_header(path)
    missing = [key for key in REQUIRED_HEADER if key not in header]
    if missing:
        stop(f"{path} is missing header fields {missing}. It was not written by "
             f"scripts/run_forward.py, and the constant baseline cannot be "
             f"reconstructed without them.")

    predictions = pd.read_csv(path, comment="#")
    if predictions.empty:
        stop(f"{path} contains no prediction rows.")

    for column in ("ticker", "predicted_probability", "predicted_label",
                   "expected_period_end"):
        if column not in predictions.columns:
            stop(f"{path} has no {column} column.")

    for column in ("actual_label", "label_yoy", "outcome", "correct"):
        if column in predictions.columns:
            stop(f"{path} already carries an outcome column ({column}). A "
                 f"prediction file must not contain what it is scored against.")

    duplicated = predictions["ticker"].duplicated()
    if bool(duplicated.any()):
        stop(f"{path} has {int(duplicated.sum())} duplicate tickers; each "
             f"prediction must name one ticker once.")

    cut = pd.Timestamp(token(header, "cut_date"))
    train_rate = float(token(header, "training_positive_rate"))
    threshold = float(token(header, "decision_threshold"))
    digest = file_digest(path)

    print(f"prediction file         : {path}")
    print(f"    sha256              : {digest[:16]}...  (re-checked at exit)")
    print(f"cut date                : {cut.date()}")
    print(f"predictions             : {len(predictions)} tickers")
    print(f"model                   : {header['model']}")
    print(f"decision threshold      : {threshold}")
    print(f"training rows           : {token(header, 'training_rows_labelled')}")
    print(f"training positive rate  : {train_rate:.6f}  "
          f"(the constant baseline's probability)")
    print(f"generated at            : {header.get('produced_at_utc', '?')}")
    print(f"generated from code     : {token(header, 'code_sha')[:12]}  "
          f"(clean tree: {token(header, 'working_tree_clean')})")

    return predictions, header, cut, train_rate, threshold, digest


def check_consistency(header: dict) -> None:
    """The pre-registration and the panel must be what they were at prediction time.

    Both feed the label this script computes -- the panel supplies the year-ago
    quarter, the config supplies the threshold the predicted_label used -- so if
    either moved since the predictions were written, the thing being scored is
    not the thing that was predicted.
    """
    if not os.path.exists(CONFIG_PATH):
        stop(f"{CONFIG_PATH} not found.")

    with open(CONFIG_PATH) as handle:
        config = json.load(handle)

    claimed = header["config"]
    if f"revision {config['revision']}" not in claimed:
        stop(f"the prediction file was written against '{claimed}', but "
             f"{CONFIG_PATH} is now revision {config['revision']}. The "
             f"pre-registration changed after the predictions were made.")

    if not os.path.exists(PANEL_PARQUET):
        stop(f"{PANEL_PARQUET} not found; the year-ago lookup needs it.")

    panel_digest = file_digest(PANEL_PARQUET)
    if panel_digest != token(header, "panel_sha256"):
        stop(f"{PANEL_PARQUET} has changed since the predictions were written "
             f"(recorded {token(header, 'panel_sha256')[:16]}..., now "
             f"{panel_digest[:16]}...). The year-ago comparisons this script "
             f"would compute are not the ones the predictions assumed.")

    print(f"config revision         : {config['revision']}  (matches the file)")
    print(f"panel sha256            : {panel_digest[:16]}...  (matches the file)")

    return panel_digest


def locate_commit(path: str, cut: pd.Timestamp, header: dict) -> pd.Timestamp:
    """Find the commit that ADDED the prediction file, and sanity-check it.

    The returned timestamp is the committer date, not the author date: author
    dates are trivially set by --date, while the committer date is what the
    commit was actually written with. Neither is unforgeable, and this script
    does not pretend otherwise -- it checks what a reader of the repository can
    check.
    """
    ok, _ = git("ls-files", "--error-unmatch", path)
    if not ok:
        stop(f"{path} is not tracked by git. An uncommitted prediction file has "
             f"no timestamp, and a forward test is the timestamp.")

    ok, diff = git("diff", "--name-only", "HEAD", "--", path)
    if not ok:
        stop("could not compare the prediction file against HEAD.")
    if diff:
        stop(f"{path} has uncommitted changes. What would be scored is not what "
             f"was committed.")

    ok, log = git("log", "--diff-filter=A", "--format=%H%x09%cI", "--", path)
    if not ok or not log:
        stop(f"could not find the commit that added {path}.")

    # git log is newest-first, so the last line is the original addition.
    sha, _tab, iso = log.splitlines()[-1].partition("\t")
    committed = pd.Timestamp(iso).tz_convert("UTC").tz_localize(None)

    # The code that produced the file must predate the commit that recorded it.
    code_sha = token(header, "code_sha")
    ok, _ = git("merge-base", "--is-ancestor", code_sha, sha)
    if not ok:
        stop(f"the code sha recorded in the file ({code_sha[:12]}) is not an "
             f"ancestor of the commit that added it ({sha[:12]}). The file does "
             f"not descend from the code it claims to have come from.")

    lag = (committed.normalize() - cut).days
    print(f"added by commit         : {sha[:12]}  at {committed} UTC")
    print(f"    cut -> commit lag   : {lag} day(s)")
    if lag > 1:
        # Not fatal: run_forward's own STAGE 6 already refuses to predict a
        # ticker whose next quarter was filed by the time it ran. But a long lag
        # means the file sat uncommitted while the world moved, and a reader
        # should see that rather than have it averaged into a headline number.
        print(f"    NOTE the file was committed {lag} days after its cut date. "
              f"The cut is what\n         the model was fitted to; the commit is "
              f"what a reader can verify. A\n         long gap weakens the "
              f"second without changing the first.")

    return committed


# --------------------------------------------------------------------------
# Stage 1 -- fetch current filings
# --------------------------------------------------------------------------


def fetch_facts(tickers: list, scored_at: pd.Timestamp) -> pd.DataFrame:
    section("STAGE 1 -- fetch current filings")

    cache_dir = FACT_CACHE_TEMPLATE.format(scored_at=scored_at.date())
    os.makedirs(cache_dir, exist_ok=True)

    cik_map = load_ticker_cik_map()
    recovered = load_recovered_ciks()

    print(f"tickers to fetch        : {len(tickers)}  "
          f"(throttle floor ~{len(tickers) * SEC_SLEEP_SECONDS / 60:.0f} min)")
    print(f"cache                   : {cache_dir}  (keyed by scoring date)")

    frames = []
    unresolved = []
    empty = []

    for position, ticker in enumerate(sorted(tickers), start=1):
        spans, route = resolve_ticker_spans(ticker, cik_map, recovered)
        if not spans:
            unresolved.append(ticker)
            continue

        cache_path = os.path.join(cache_dir, f"{ticker}.parquet")
        if os.path.exists(cache_path):
            facts = pd.read_parquet(cache_path)
        else:
            facts = fetch_quarterly_eps_spanned(ticker, spans)
            if facts is not None:
                facts.to_parquet(cache_path, index=False)

        if facts is None or facts.empty:
            empty.append(ticker)
            continue

        facts = facts.copy()
        facts["cik_span_used"] = describe_spans(spans)
        facts["resolution_route"] = route
        frames.append(facts)

        if position % 100 == 0:
            print(f"    {position}/{len(tickers)}  ({len(frames)} with facts)")

    if not frames:
        stop("no predicted ticker returned any EPS facts.")

    all_facts = pd.concat(frames, ignore_index=True)
    print(f"\nresolved with facts     : {len(frames)}")
    print(f"resolved but no facts   : {len(empty)}")
    print(f"unresolved              : {len(unresolved)}")
    print(f"raw quarterly facts     : {len(all_facts)}")

    return all_facts


# --------------------------------------------------------------------------
# Stage 2 -- build labelled quarters under the panel's rules
# --------------------------------------------------------------------------


def build_quarters(all_facts: pd.DataFrame, panel: pd.DataFrame) -> pd.DataFrame:
    """Apply the panel's construction rules to the fetched facts.

    MIN(filed), the 120-day lag filter, the STUDY_START floor, the +/-45 day
    year-ago match and the split adjustment -- all imported. The panel is a
    read-only lookup: it supplies year-ago candidates and is never written.
    """
    section("STAGE 2 -- rebuild quarters under the panel's rules")

    deduped = dedupe_to_first_filing(all_facts)
    print(f"after dedupe MIN(filed) : {len(deduped)}")

    after_lag, lag_dropped = apply_lag_filter(deduped)
    print(f"after lag filter (<={MAX_FILING_LAG_DAYS}d): {len(after_lag)}  "
          f"({len(lag_dropped)} dropped)")

    after_lag = after_lag.copy()
    after_lag["prediction_date"] = after_lag["filed_date"] + pd.Timedelta(days=1)

    carry = ["ticker", "cik", "cik_span_used", "period_end", "filed_date",
             "prediction_date", "eps_diluted", "n_filings_seen",
             "filing_lag_days"]
    # Panel rows first, so the panel's committed value wins for any quarter both
    # sources hold. Both are MIN(filed) and should agree; where they do not, the
    # panel's is the one the model was trained against.
    combined = pd.concat(
        [panel[carry], after_lag[carry]], ignore_index=True
    ).drop_duplicates(subset=["ticker", "period_end"], keep="first")

    matched = attach_year_ago(combined.sort_values(["ticker", "period_end"]))

    before_floor = len(matched)
    matched = matched.loc[matched["period_end"] >= STUDY_START].copy()
    print(f"STUDY_START floor       : {len(matched)} kept, "
          f"{before_floor - len(matched)} pre-{STUDY_START.year} rows dropped")

    matched["prediction_date_year_ago"] = (
        matched["filed_date_year_ago"] + pd.Timedelta(days=1)
    )

    split_events = detect_split_events(all_facts)
    established = [event for event in split_events if event["established"]]
    print(f"splits detected         : {len(split_events)} "
          f"({len(established)} bounded)")

    matched = apply_split_adjustment(matched, split_events)
    matched = attach_label(matched)

    print(f"year-ago EPS adjusted   : "
          f"{int((matched['split_factor_applied'] > 1).sum())} rows")
    print(f"nulled as unresolvable  : "
          f"{int(matched['split_ambiguous'].sum())} rows")

    return matched


# --------------------------------------------------------------------------
# Stage 3 -- resolve each prediction to its first quarter filed after the cut
# --------------------------------------------------------------------------


def resolve_outcomes(predictions: pd.DataFrame, quarters: pd.DataFrame,
                     cut: pd.Timestamp) -> pd.DataFrame:
    """Attach, to each prediction, the first quarter that ticker filed after the cut.

    Selection is on prediction_date > cut -- the exact complement of the history
    rule run_forward used -- and then on the earliest filed_date. Never on
    expected_period_end, which was an estimate.
    """
    section("STAGE 3 -- resolve predictions to filed quarters")

    after_cut = quarters.loc[quarters["prediction_date"] > cut]
    after_cut = after_cut.sort_values(["ticker", "filed_date", "period_end"])

    # drop_duplicates, NOT groupby().first(): groupby's first() takes the first
    # non-null value PER COLUMN, so a row with a missing year-ago would silently
    # borrow the next quarter's eps_year_ago and the label would be computed
    # across two different quarters. This takes the first ROW.
    first = after_cut.drop_duplicates(subset="ticker", keep="first")

    resolved = predictions.merge(
        first[["ticker", "period_end", "filed_date", "prediction_date",
               "eps_diluted", "period_end_year_ago", "eps_year_ago",
               "split_factor_applied", "eps_year_ago_adjusted",
               "split_ambiguous", "filing_lag_days", LABEL]],
        on="ticker", how="left", suffixes=("", "_actual"),
    )

    resolved = resolved.rename(columns={
        "period_end": "actual_period_end",
        "filed_date": "actual_filed_date",
        "prediction_date": "actual_prediction_date",
        "eps_diluted": "actual_eps_diluted",
        LABEL: "actual_label",
    })

    filed = resolved["actual_filed_date"].notna()
    scored = filed & resolved["actual_label"].notna()

    resolved["status"] = np.where(
        ~filed, "not_yet_filed",
        np.where(scored, "scored", "filed_but_unlabelled"),
    )

    print(f"predictions             : {len(resolved)}")
    print(f"    filed since the cut : {int(filed.sum())}")
    print(f"    scored (labelled)   : {int(scored.sum())}")
    print(f"    filed, no label     : {int((filed & ~scored).sum())}  "
          f"(no year-ago match, or an unresolvable split)")
    print(f"    not yet filed       : {int((~filed).sum())}")

    # The estimate was never used to select; this is only how far it missed.
    hit = resolved.loc[filed].copy()
    if len(hit):
        gap = (pd.to_datetime(hit["actual_period_end"])
               - pd.to_datetime(hit["expected_period_end"])).dt.days
        print(f"\nexpected_period_end vs actual (estimate only, not used to match):")
        print(f"    exact               : {int((gap == 0).sum())} of {len(hit)}")
        print(f"    within 7 days       : {int((gap.abs() <= 7).sum())}")
        print(f"    median |error|      : {int(gap.abs().median())} days")
        print(f"    worst               : {int(gap.abs().max())} days")

    return resolved


# --------------------------------------------------------------------------
# Stage 4 -- the assertion the whole test rests on
# --------------------------------------------------------------------------


def assert_commit_precedes_filings(resolved: pd.DataFrame,
                                   committed: pd.Timestamp,
                                   path: str) -> None:
    """The prediction commit must predate every filing being scored against.

    This is the forward test. Everything else is arithmetic that would be just
    as easy to produce after the fact, so it is verified rather than assumed.

    Filing dates carry day granularity, so the check is on calendar days: the
    commit must fall on a strictly earlier UTC day than every scored filing. A
    commit and a filing on the same day cannot be ordered from this data, and
    unorderable is treated as failing -- the burden is on the evidence.
    """
    section("STAGE 4 -- the prediction commit must predate every filing")

    scored = resolved.loc[resolved["status"] == "scored"]
    if scored.empty:
        print("no scored rows yet; nothing to check against.")
        return

    filed = pd.to_datetime(scored["actual_filed_date"])
    commit_day = committed.normalize()
    earliest = filed.min()

    violations = scored.loc[filed <= commit_day]
    if len(violations):
        names = ", ".join(
            f"{row.ticker} (filed {pd.Timestamp(row.actual_filed_date).date()})"
            for row in violations.head(10).itertuples()
        )
        stop(
            f"{len(violations)} scored quarters were filed on or before the day "
            f"{path} was committed ({commit_day.date()}): {names}"
            f"{' ...' if len(violations) > 10 else ''}. Those outcomes existed, "
            f"or may have existed, when the prediction was recorded, so this is "
            f"not a forward test for them. Scoring the rest in isolation would "
            f"be choosing which rows count after seeing them, so nothing is "
            f"scored."
        )

    print(f"prediction committed    : {commit_day.date()}")
    print(f"earliest scored filing  : {earliest.date()}  "
          f"({(earliest - commit_day).days} days later)")
    print(f"latest scored filing    : {filed.max().date()}")
    print(f"\nPASS  all {len(scored)} scored filings postdate the prediction commit")


# --------------------------------------------------------------------------
# Stage 5 -- score
# --------------------------------------------------------------------------


def score(resolved: pd.DataFrame, train_rate: float, threshold: float) -> dict:
    """Model against the constant baseline, on the labelled rows only.

    The baseline predicts class 1 always, at the prediction file's OWN training
    positive rate. It may never use the rate of the period being scored -- that
    would score the baseline using the answer -- which is why the rate is read
    from the file rather than computed here.
    """
    section("STAGE 5 -- model vs the constant baseline")

    scored = resolved.loc[resolved["status"] == "scored"].copy()
    if scored.empty:
        print("nothing labelled yet; no metrics.")
        return {}

    y_true = scored["actual_label"].astype("int64").to_numpy()
    proba = scored["predicted_probability"].astype("float64").to_numpy()
    label = scored["predicted_label"].astype("int64").to_numpy()

    constant_proba = np.full(len(y_true), train_rate, dtype="float64")
    constant_label = np.ones(len(y_true), dtype="int64")

    result = {
        "n_predicted": int(len(resolved)),
        "n_scored": int(len(scored)),
        "n_not_yet_filed": int((resolved["status"] == "not_yet_filed").sum()),
        "n_filed_unlabelled": int((resolved["status"] == "filed_but_unlabelled").sum()),
        "actual_positive_rate": float(y_true.mean()),
        "constant_probability": float(train_rate),
        "model_log_loss": float(log_loss(y_true, proba, labels=[0, 1])),
        "constant_log_loss": float(log_loss(y_true, constant_proba, labels=[0, 1])),
        "model_accuracy": float(accuracy_score(y_true, label)),
        "constant_accuracy": float(accuracy_score(y_true, constant_label)),
        "model_brier": float(brier_score_loss(y_true, proba)),
        "predicted_positive_rate": float(label.mean()),
        "decision_threshold": float(threshold),
    }
    result["log_loss_delta"] = result["model_log_loss"] - result["constant_log_loss"]
    result["accuracy_delta"] = result["model_accuracy"] - result["constant_accuracy"]
    result["passes_log_loss"] = bool(result["log_loss_delta"] < 0)
    result["passes_accuracy"] = bool(result["accuracy_delta"] > 0)
    result["passes_both"] = bool(result["passes_log_loss"] and result["passes_accuracy"])

    print(f"scored rows             : {result['n_scored']} of "
          f"{result['n_predicted']} predicted")
    print(f"actual positive rate    : {result['actual_positive_rate']:.1%}")
    print(f"predicted positive rate : {result['predicted_positive_rate']:.1%}")
    print()
    print(f"{'':24s} {'model':>10s} {'constant':>10s} {'delta':>10s}")
    print(f"{'log-loss':24s} {result['model_log_loss']:10.4f} "
          f"{result['constant_log_loss']:10.4f} {result['log_loss_delta']:+10.4f}")
    print(f"{'accuracy':24s} {result['model_accuracy']:10.1%} "
          f"{result['constant_accuracy']:10.1%} {result['accuracy_delta']:+10.2%}")
    print(f"{'Brier':24s} {result['model_brier']:10.4f}")
    print(f"\nconstant baseline probability {train_rate:.6f}, from the prediction")
    print(f"file's own training window -- never from the period being scored.")

    print(f"\nlog-loss below constant : {'PASS' if result['passes_log_loss'] else 'FAIL'}")
    print(f"accuracy above constant : {'PASS' if result['passes_accuracy'] else 'FAIL'}")
    print(f"both (log-loss primary) : {'PASS' if result['passes_both'] else 'FAIL'}")

    if result["n_not_yet_filed"]:
        print(f"\nNOTE {result['n_not_yet_filed']} predictions are still unfiled. "
              f"This is a PARTIAL score.\n     The names that report first are "
              f"not a random sample of the universe, so\n     treat these "
              f"numbers as provisional until the file is fully resolved.")

    return result


def report_calibration(resolved: pd.DataFrame) -> pd.DataFrame:
    section("STAGE 6 -- calibration")

    scored = resolved.loc[resolved["status"] == "scored"].copy()
    if len(scored) < CALIBRATION_DECILES * 2:
        print(f"only {len(scored)} scored rows; too few for a decile table.")
        return pd.DataFrame()

    scored["actual_label"] = scored["actual_label"].astype("int64")
    scored["decile"] = pd.qcut(scored["predicted_probability"],
                               CALIBRATION_DECILES, labels=False,
                               duplicates="drop")

    table = scored.groupby("decile").agg(
        n=("actual_label", "size"),
        predicted=("predicted_probability", "mean"),
        actual=("actual_label", "mean"),
        p_min=("predicted_probability", "min"),
        p_max=("predicted_probability", "max"),
    ).reset_index()
    table["gap"] = table["actual"] - table["predicted"]

    print(f"{'decile':>7s} {'n':>6s} {'range':>17s} {'predicted':>10s} "
          f"{'actual':>8s} {'gap':>8s}")
    for _index, row in table.iterrows():
        print(f"{int(row['decile']) + 1:7d} {int(row['n']):6d} "
              f"[{row['p_min']:.3f},{row['p_max']:.3f}] {row['predicted']:10.3f} "
              f"{row['actual']:8.3f} {row['gap']:+8.3f}  "
              f"{'.' * int(row['actual'] * 40)}|")

    ece = float((table["n"] * table["gap"].abs()).sum() / table["n"].sum())
    print(f"\n    expected calibration error (ECE) : {ece:.4f}")
    print(f"    mean predicted / mean actual     : "
          f"{scored['predicted_probability'].mean():.4f} / "
          f"{scored['actual_label'].mean():.4f}")
    rises = int((table["actual"].diff().dropna() > 0).sum())
    print(f"    deciles where actual rate rises  : {rises} of {len(table) - 1}")
    print(f"\n    Holdout reference (data/holdout_calibration.csv): ECE 0.0378,")
    print(f"    decile 10 predicted 0.824 against 0.736 realized. The model was")
    print(f"    overconfident exactly where it was most confident; check whether")
    print(f"    that reproduces here before trusting any top-slice ranking.")

    return table


def report_membership(resolved: pd.DataFrame) -> None:
    """Membership at the actual filing date -- reported, never acted on.

    A prediction was committed for a ticker that was a constituent at the cut.
    If the company has since left the index, that is information about the
    universe, not grounds to drop the row: deciding which committed predictions
    count, after their outcomes exist, is exactly the selection this design
    exists to avoid. So this prints a count and changes nothing.

    Membership also comes from the pinned 2025-05-27 revision, which cannot see
    changes after that date, so a "still a member" answer here is weaker the
    further the cut sits from the pin.
    """
    section("MEMBERSHIP AT THE FILING DATE (information only)")

    scored = resolved.loc[resolved["status"] == "scored"]
    if scored.empty:
        print("no scored rows.")
        return

    # This section is information only, so it must not be able to kill a scoring
    # run. load_tables() aborts via src.data_loader.fail() -- correctly, for the
    # panel build, where substituting today's constituent list would be a
    # survivorship error. Here there is nothing to substitute and nothing
    # downstream depends on the answer, so a failure degrades to a note.
    try:
        current, changes_raw, provenance = load_tables()
        intervals, _metadata = build_intervals(
            current, normalise_changes(changes_raw), STUDY_START
        )
    except SystemExit:
        print("membership lookup unavailable; skipping. Nothing that is scored")
        print("depends on it -- the rows are scored as committed either way.")
        return

    still = [
        is_member_at(intervals, ticker, moment)
        for ticker, moment in zip(scored["ticker"],
                                  pd.to_datetime(scored["actual_prediction_date"]))
    ]
    outside = int(len(still) - sum(still))

    print(f"provenance              : {provenance}")
    print(f"scored rows             : {len(scored)}")
    print(f"    member at filing    : {sum(still)}")
    print(f"    not a member        : {outside}")
    if outside:
        print("")
        print(f"{outside} scored quarters belong to tickers the pinned revision")
        print("does not place in the index at their filing date. They are scored")
        print("anyway: dropping a committed prediction because the company later")
        print("left would be choosing which rows count after the outcome existed.")


# --------------------------------------------------------------------------
# Stage 7 -- write
# --------------------------------------------------------------------------


def write_scored(resolved: pd.DataFrame, calibration: pd.DataFrame,
                 result: dict, header: dict, cut: pd.Timestamp,
                 committed: pd.Timestamp, scored_at_utc: pd.Timestamp,
                 predictions_path: str, predictions_digest: str,
                 output_path: str) -> None:
    section("STAGE 7 -- write")

    output = resolved.copy()
    is_scored = output["status"] == "scored"

    # Carried so the decile table in the log is reconstructible from the file.
    output["decile"] = pd.NA
    if not calibration.empty:
        output.loc[is_scored, "decile"] = pd.qcut(
            output.loc[is_scored, "predicted_probability"],
            CALIBRATION_DECILES, labels=False, duplicates="drop",
        )

    output["correct"] = pd.NA
    output.loc[is_scored, "correct"] = (
        output.loc[is_scored, "predicted_label"]
        == output.loc[is_scored, "actual_label"]
    )

    lines = [
        "Scored forward-test results. Scoring only -- nothing was refit.",
        "",
        f"scored_prediction_file   : {predictions_path}",
        f"prediction_file_sha256   : {predictions_digest}",
        f"prediction_committed_utc : {committed}",
        f"cut_date                 : {cut.date()}",
        f"scored_at_utc            : {scored_at_utc.isoformat()}",
        f"scoring_code_sha         : {git('rev-parse', 'HEAD')[1]}",
        f"config                   : {header['config']}",
        f"model                    : {header['model']}",
        f"decision_threshold       : {header['decision_threshold']}",
        f"constant_probability     : {result.get('constant_probability', 'n/a')}",
        "",
        "RESOLUTION",
        f"  predicted              : {result.get('n_predicted', len(resolved))}",
        f"  scored                 : {result.get('n_scored', 0)}",
        f"  filed but unlabelled   : {result.get('n_filed_unlabelled', 0)}",
        f"  not yet filed          : {result.get('n_not_yet_filed', 0)}",
        "",
        "RESULT",
        f"  actual positive rate   : {result.get('actual_positive_rate', 'n/a')}",
        f"  model log-loss         : {result.get('model_log_loss', 'n/a')}",
        f"  constant log-loss      : {result.get('constant_log_loss', 'n/a')}",
        f"  model accuracy         : {result.get('model_accuracy', 'n/a')}",
        f"  constant accuracy      : {result.get('constant_accuracy', 'n/a')}",
        f"  model Brier            : {result.get('model_brier', 'n/a')}",
        f"  passes both conditions : {result.get('passes_both', 'n/a')}",
        "",
        "HOW THE OUTCOME WAS MATCHED",
        "  Each row is the FIRST quarter its ticker filed after the cut, selected",
        "  on prediction_date > cut_date -- the exact complement of the history",
        "  rule run_forward used. expected_period_end was never used to match.",
        "  Labels are built with the panel's own rules: MIN(filed), the 120-day",
        "  filing-lag filter, the +/-45 day year-ago match, split adjustment from",
        "  restatement evidence, and the STUDY_START floor.",
        "",
        "WHAT WAS NOT DONE",
        "  Nothing was refit and no threshold was chosen; predicted_probability",
        "  and predicted_label are copied from the prediction file unchanged.",
        "  Predictions were NOT re-gated on index membership -- dropping a",
        "  committed prediction because the company later left the index would be",
        "  selecting rows after the outcome existed.",
        "",
    ]

    if result.get("n_not_yet_filed"):
        lines += [
            "PARTIAL",
            f"  {result['n_not_yet_filed']} predictions are still unfiled, so this",
            "  is a partial score. Early reporters are not a random sample of the",
            "  universe; treat these numbers as provisional until fully resolved.",
            "",
        ]

    os.makedirs("data", exist_ok=True)
    with open(output_path, "w") as handle:
        for line in lines:
            handle.write(f"# {line}\n" if line else "#\n")
        output.to_csv(handle, index=False)

    print(f"scored -> {output_path}  ({len(output)} rows)")


# --------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Score a committed forward-test prediction file. "
                    "Scores only; never predicts and never refits."
    )
    parser.add_argument("predictions", help="path to data/live_predictions_<cut>.csv")
    parser.add_argument("--rescore", action="store_true",
                        help="overwrite an existing scored file (for re-scoring "
                             "once more of the predictions have resolved)")
    args = parser.parse_args()

    scored_at_utc = pd.Timestamp.now("UTC")
    scored_at = scored_at_utc.normalize().tz_localize(None)

    predictions, header, cut, train_rate, threshold, digest = load_predictions(
        args.predictions
    )
    panel_digest = check_consistency(header)
    committed = locate_commit(args.predictions, cut, header)

    output_path = SCORED_TEMPLATE.format(cut=cut.date())
    if os.path.exists(output_path) and not args.rescore:
        stop(f"{output_path} already exists. Pass --rescore to overwrite it, "
             f"which is the right move only when more predictions have resolved "
             f"since -- git keeps the previous version either way.")

    panel = pd.read_parquet(PANEL_PARQUET)
    all_facts = fetch_facts(predictions["ticker"].tolist(), scored_at)
    quarters = build_quarters(all_facts, panel)
    resolved = resolve_outcomes(predictions, quarters, cut)

    assert_commit_precedes_filings(resolved, committed, args.predictions)

    result = score(resolved, train_rate, threshold)
    calibration = report_calibration(resolved)

    report_membership(resolved)

    write_scored(resolved, calibration, result, header, cut, committed,
                 scored_at_utc, args.predictions, digest, output_path)

    section("READ-ONLY CHECKS")
    if file_digest(args.predictions) != digest:
        stop(f"{args.predictions} changed during the run.")
    print(f"PASS  {args.predictions} is byte-identical to the start of the run")
    if file_digest(PANEL_PARQUET) != panel_digest:
        stop(f"{PANEL_PARQUET} changed during the run.")
    print(f"PASS  {PANEL_PARQUET} is byte-identical to the start of the run")

    section("DONE -- SCORING ONLY")
    print("Nothing was refit and no prediction was altered.")


if __name__ == "__main__":
    main()
