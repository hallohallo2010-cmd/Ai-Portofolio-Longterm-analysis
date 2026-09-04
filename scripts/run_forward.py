#!/usr/bin/env python3
"""Phase 3: the live forward test -- PREDICT ONLY.

The 2022-2025 holdout is spent. It has been read, so no further evaluation
against it is out-of-sample. The only test left that can still be honest is one
run against quarters that did not exist when the model was frozen.

This script produces that test's predictions and NOTHING ELSE. It never scores,
never computes an outcome, and never writes a label. Scoring is a separate
script and a separate commit, made later, once the predicted quarters have
actually been filed. That separation is the experiment: a prediction committed
before the filing exists is evidence; the same prediction produced afterwards is
a re-description of history, and no amount of care in the code recovers the
difference.

WHAT IS PREDICTED, AND WHY IT IS NOT THE NEWLY FILED QUARTER
------------------------------------------------------------
The obvious reading of "run the pipeline forward on new filings" is: fetch the
quarters filed since the panel ends, build their features, predict them. That
reading does not work, and it is worth being explicit about why.

label_yoy compares a quarter's diluted EPS against the SAME QUARTER ONE YEAR
EARLIER. The year-ago quarter is already in the panel. So the moment a new
filing lands, both sides of the comparison are in hand and the label is
computable immediately -- there is no waiting period. Predicting a quarter you
have just downloaded is therefore predicting something already determined, by a
process that has already touched the number that determines it.

So this script predicts the quarter that has NOT BEEN FILED YET. For each index
member, it builds the ten frozen features -- all of which are lagged, and none
of which need the quarter's own EPS -- for that ticker's NEXT quarterly report,
and commits a probability for it. The outcome becomes knowable only when the
company files, which is weeks away and outside anyone's control here.

That is also what makes the run-time assertion in STAGE 6 satisfiable rather
than decorative: a predicted row has no eps_diluted, so its label cannot be
resolved, and the script proves this instead of asserting it by construction.

WHAT THE INCREMENTAL FETCH IS FOR
---------------------------------
Fresh filings are still fetched, and still matter -- they EXTEND THE HISTORY the
features are computed from, and the training window the model is refit on. A
quarter filed after the panel was built is a legitimate new input; it is simply
an input, not a prediction target.

THE FORWARD UNIVERSE IS FROZEN, AND DRIFTS
------------------------------------------
Index membership comes from pinned Wikipedia revision 1292523673 (2025-05-27),
the last revision verified to still carry the added/removed changes table. It
cannot know about index changes after that date, and the live page is NEVER
consulted as a fallback -- substituting today's constituent list is precisely
the survivorship error the membership gate exists to prevent.

The consequence is stated rather than hidden: the forward universe is the S&P
500 as it stood on 2025-05-27, and real index turnover is roughly 20-25 names a
year, so the universe drifts from the real index at about that rate. Every
prediction file carries this note in its header. It is a known limitation of the
forward test, not a defect to be worked around.

Run:  python scripts/run_forward.py                # cut = today
      python scripts/run_forward.py --as-of DATE   # cut = DATE (never future)
"""

from __future__ import annotations

import argparse
import hashlib
import os
import subprocess
import sys

import numpy as np
import pandas as pd

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
    MEMBERSHIP_OPEN_END,
    WIKIPEDIA_CHANGES_OLDID,
    build_intervals,
    is_member_at,
    load_tables,
    normalise_changes,
)

# The panel-construction rules are IMPORTED, never restated. A forward quarter
# that reached the feature table by a different route than a panel quarter would
# not be the same kind of observation, and the model was frozen on the panel's
# kind.
from scripts.build_eps_panel import (  # noqa: E402
    MAX_FILING_LAG_DAYS,
    STUDY_START,
    apply_lag_filter,
    apply_split_adjustment,
    attach_label,
    attach_year_ago,
    dedupe_to_first_filing,
    detect_split_events,
    gate_on_membership,
)
from scripts.build_features_v1 import (  # noqa: E402
    ALL_LAGS,
    FEATURE_LAGS,
    LeakageError,
    assert_no_lag_leakage,
    attach_eps_growth,
    attach_growth_streak,
    attach_lag,
    attach_prediction_rank,
    check_preconditions,
)

# The pipeline is built by the same function the holdout used, from the same
# config file. Rebuilding it here would be a second definition of a frozen
# object, which is the one thing a freeze is supposed to make impossible.
from scripts.run_holdout import build_pipeline, load_config  # noqa: E402

PANEL_PARQUET = "data/eps_panel.parquet"
PREDICTIONS_TEMPLATE = "data/live_predictions_{cut}.csv"

# Fetched facts are cached per run so re-running the same cut does not re-hit
# EDGAR ~500 times. Keyed BY CUT DATE: a later run must not silently reuse an
# earlier run's view of EDGAR, because the whole point is what was filed by when.
FACT_CACHE_TEMPLATE = "data/forward_cache_{cut}"

LABEL = "label_yoy"

# A pending row needs a prediction_date for the lag machinery to rank it, but it
# does not have one -- the quarter is not filed. The cut plus one day is used as
# a sentinel: strictly later than every history row (all <= cut), so the row
# sorts last and its lags can only reach backwards. It is NOT a forecast of the
# filing date and is never written to the output under that name.
PENDING_RANK_OFFSET = pd.Timedelta(days=1)

# Quarters of history consulted when estimating where the next period_end falls.
QUARTER_GAP_LOOKBACK = 8
QUARTER_GAP_FALLBACK_DAYS = 91

# A ticker with fewer prior quarters than this cannot fill lag 4, so every one of
# its rank-transformed growth features would be imputed and the prediction would
# be a statement about the training medians rather than about the company.
MIN_HISTORY_QUARTERS = 4

# A ticker that has not filed in this long has missed a reporting cycle, and its
# "next quarter" is a quarter nobody is going to file.
#
# The bound is arithmetic, not taste. A quarter ends ~92 days after the previous
# one, and the panel accepts a filing lag of up to MAX_FILING_LAG_DAYS (120), so
# the longest gap a company on a normal schedule can leave between consecutive
# prediction_dates is about 92 + 120 = 212 days. Past that it is not slow, it has
# stopped -- delisted, acquired, or gone private.
#
# This matters MORE than it looks, and it is the frozen universe's sharpest edge.
# The pinned revision still lists as current any company acquired after
# 2025-05-27, so without this guard the run would emit confident predictions for
# names that will never report again. Those rows could never be scored, and the
# rows that COULD be scored would then be the survivors of continued filing --
# reintroducing, inside the forward test, exactly the survivorship selection the
# membership gate exists to prevent.
MAX_REPORTING_GAP_DAYS = 220

# If fewer than this fraction of eligible members end up with a pending quarter,
# the run is not measuring what it claims to. At any real cut, essentially every
# current constituent has filed within MAX_REPORTING_GAP_DAYS, so a large stale
# count means the incremental fetch under-delivered -- a failed or empty EDGAR
# pull, a cache from an earlier cut, or a cut far ahead of the filings on hand.
# Emitting a prediction file for the surviving handful would look like a result
# and would not be one.
MIN_PREDICTED_FRACTION = 0.5

# Real S&P 500 turnover, used only in the documentation the output carries.
INDEX_TURNOVER_NAMES_PER_YEAR = "20-25"


def section(title: str) -> None:
    print("\n" + "=" * 78)
    print(title)
    print("=" * 78)


def stop(message: str) -> None:
    """Anything that would make the forward test something other than a forward
    test. Nothing is written when this fires."""
    print(f"\n*** STOPPED: {message}", file=sys.stderr)
    print("*** No prediction file was written.", file=sys.stderr)
    sys.exit(1)


def file_digest(path: str) -> str:
    with open(path, "rb") as handle:
        return hashlib.sha256(handle.read()).hexdigest()


def git_sha() -> str:
    """The commit these predictions were produced at, recorded in the output.

    A forward prediction is only as good as the evidence of when it was made, so
    the code state is part of the record.
    """
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True, text=True, check=True,
        )
        return result.stdout.strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        return "unknown"


def git_is_clean() -> bool:
    try:
        result = subprocess.run(
            ["git", "status", "--porcelain"],
            capture_output=True, text=True, check=True,
        )
        return not result.stdout.strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        return False


# --------------------------------------------------------------------------
# Stage 0 -- preflight
# --------------------------------------------------------------------------


def parse_cut(raw: str | None) -> pd.Timestamp:
    """The cut is the moment the prediction is made. Everything at or before it
    is knowable; everything after it is what the test is about."""
    today = pd.Timestamp.utcnow().normalize().tz_localize(None)

    if raw is None:
        return today

    cut = pd.Timestamp(raw).normalize()
    if cut > today:
        stop(f"--as-of {cut.date()} is in the future. A prediction cannot be "
             f"made as of a date that has not happened.")
    if cut < today:
        # Backdating is allowed for a rehearsal, but it is the one input that
        # could turn this into a re-description of history, so it is announced
        # loudly and the STAGE 6 assertion is what actually polices it.
        print(f"NOTE  cut is backdated to {cut.date()} (today is {today.date()}). "
              f"Any ticker whose next quarter was already filed by now will be "
              f"REFUSED in STAGE 6, not silently predicted.")
    return cut


def preflight(cut: pd.Timestamp) -> tuple:
    section("STAGE 0 -- preflight")

    if not os.path.exists(PANEL_PARQUET):
        stop(f"{PANEL_PARQUET} not found.")

    output_path = PREDICTIONS_TEMPLATE.format(cut=cut.date())
    if os.path.exists(output_path):
        # Re-rolling a prediction for a cut that already has one is how a
        # forward test quietly becomes a search over predictions.
        stop(f"{output_path} already exists. A cut date gets ONE prediction "
             f"file; delete it deliberately if it was a rehearsal.")

    if cut > MEMBERSHIP_OPEN_END:
        # is_member_at would return False for every ticker and the run would
        # produce an empty file that looked like a legitimate result.
        stop(f"cut {cut.date()} is past MEMBERSHIP_OPEN_END "
             f"({MEMBERSHIP_OPEN_END.date()}), where every membership interval "
             f"closes. The pinned revision cannot gate this date; extending the "
             f"forward test past it is a decision about the universe, not a "
             f"parameter to nudge.")

    config = load_config()
    features = config["features"]["columns"]
    identities = config["features"]["excluded_target_identities"]["columns"]

    contaminated = [name for name in identities if name in features]
    if contaminated:
        stop(f"target identity columns present in the frozen feature list: "
             f"{contaminated}")

    panel = pd.read_parquet(PANEL_PARQUET)
    digest = file_digest(PANEL_PARQUET)

    print(f"cut (prediction made as of): {cut.date()}")
    print(f"config revision            : {config['revision']}")
    print(f"frozen features            : {len(features)}")
    print(f"excluded identities        : {identities}")
    print(f"panel rows                 : {len(panel)}  ({panel['ticker'].nunique()} tickers)")
    print(f"panel prediction_date       : {panel['prediction_date'].min().date()} .. "
          f"{panel['prediction_date'].max().date()}")
    print(f"panel sha256               : {digest[:16]}...  (read-only; verified at exit)")
    print(f"code sha                   : {git_sha()[:12]}  "
          f"(working tree {'clean' if git_is_clean() else 'DIRTY'})")

    if panel["prediction_date"].max() > cut:
        stop(f"the panel already contains rows with prediction_date after the "
             f"cut ({panel['prediction_date'].max().date()} > {cut.date()}). "
             f"There is no forward window left to predict.")

    return config, panel, digest, output_path


# --------------------------------------------------------------------------
# Stage 1 -- membership, pinned and frozen
# --------------------------------------------------------------------------


def load_membership(cut: pd.Timestamp) -> tuple:
    section("STAGE 1 -- index membership (pinned, never live)")

    current, changes_raw, provenance = load_tables()
    changes = normalise_changes(changes_raw)
    intervals, membership = build_intervals(current, changes, STUDY_START)

    members = sorted(
        ticker for ticker in intervals if is_member_at(intervals, ticker, cut)
    )

    print(f"provenance              : {provenance}")
    print(f"ever-constituents       : {len(intervals)}")
    print(f"members at {cut.date()}    : {len(members)}")
    print()
    print(f"FROZEN UNIVERSE: membership is the index as of 2025-05-27 (pinned")
    print(f"revision {WIKIPEDIA_CHANGES_OLDID}). Index changes after that date are")
    print(f"invisible to this run, and real turnover is ~{INDEX_TURNOVER_NAMES_PER_YEAR} "
          f"names/year, so the")
    print(f"forward universe drifts from the real index at about that rate.")
    print(f"The live page is never consulted -- doing so would substitute today's")
    print(f"survivors for the constituent list of the time.")

    return intervals, membership, members, provenance


# --------------------------------------------------------------------------
# Stage 2 -- incremental fetch
# --------------------------------------------------------------------------


def fetch_new_facts(members: list, cut: pd.Timestamp) -> pd.DataFrame:
    """Fetch quarterly EPS facts for the tickers this run could predict.

    EDGAR's companyconcept endpoint has no "filed since" parameter, so a
    per-ticker fetch necessarily returns that filer's whole history. The run is
    incremental in what it KEEPS, not in what it asks for -- and in the universe
    it asks about: only tickers that are index members at the cut, rather than
    all 716 ever-constituents.

    Returned UNDEDUPED and UNFILTERED, because split detection needs to see the
    repeated filings of a period, and STAGE 6 needs to see filings dated after
    the cut in order to refuse them.
    """
    section("STAGE 2 -- incremental fetch")

    cache_dir = FACT_CACHE_TEMPLATE.format(cut=cut.date())
    os.makedirs(cache_dir, exist_ok=True)

    cik_map = load_ticker_cik_map()
    recovered = load_recovered_ciks()

    estimate = len(members) * SEC_SLEEP_SECONDS / 60
    print(f"tickers to fetch        : {len(members)}  "
          f"(throttle floor ~{estimate:.0f} min)")
    print(f"cache                   : {cache_dir}  (keyed by cut date)")

    frames = []
    unresolved = []
    empty = []

    for position, ticker in enumerate(members, start=1):
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
            print(f"    {position}/{len(members)}  ({len(frames)} with facts)")

    if not frames:
        stop("no member ticker returned any EPS facts.")

    all_facts = pd.concat(frames, ignore_index=True)

    print(f"\nresolved with facts     : {len(frames)}")
    print(f"resolved but no facts   : {len(empty)}")
    print(f"unresolved              : {len(unresolved)}")
    print(f"raw quarterly facts     : {len(all_facts)}")

    return all_facts


# --------------------------------------------------------------------------
# Stage 3 -- extend the history, without touching the panel
# --------------------------------------------------------------------------


def build_new_history(all_facts: pd.DataFrame, panel: pd.DataFrame,
                      intervals: dict, membership: pd.DataFrame,
                      cut: pd.Timestamp) -> tuple:
    """Turn freshly fetched facts into history rows shaped exactly like panel rows.

    Every rule is the panel's own, imported rather than restated: MIN(filed) for
    point-in-time correctness, the 120-day filing-lag filter, the +/-45 day
    year-ago match, split adjustment from restatement evidence, and the
    membership gate at prediction_date.

    The panel is used as a READ-ONLY lookup and is never modified: the year-ago
    match for a new quarter needs the panel's history as candidates, so the two
    are concatenated for the match and the panel's own rows are then discarded
    from the result, keeping their committed values untouched.
    """
    section("STAGE 3 -- extend history (panel read-only)")

    deduped = dedupe_to_first_filing(all_facts)
    print(f"after dedupe MIN(filed) : {len(deduped)}")

    after_lag, lag_dropped = apply_lag_filter(deduped)
    print(f"after lag filter (<={MAX_FILING_LAG_DAYS}d): {len(after_lag)}  "
          f"({len(lag_dropped)} dropped)")

    after_lag = after_lag.copy()
    after_lag["prediction_date"] = after_lag["filed_date"] + pd.Timedelta(days=1)

    # Anything the panel already carries is the panel's, not this run's. Matching
    # on (ticker, period_end) rather than on a date threshold means a quarter the
    # panel already committed cannot be re-derived here with a different value.
    known = set(zip(panel["ticker"], panel["period_end"]))
    is_known = [
        (ticker, period_end) in known
        for ticker, period_end in zip(after_lag["ticker"], after_lag["period_end"])
    ]
    fresh = after_lag.loc[~np.array(is_known)].copy()

    # Filings dated after the cut are held back from the history -- they are not
    # knowable at the moment the prediction is made -- but they are RETAINED and
    # returned, because STAGE 6 must see them to refuse the tickers they belong to.
    after_cut = fresh.loc[fresh["prediction_date"] > cut].copy()
    fresh = fresh.loc[fresh["prediction_date"] <= cut].copy()

    print(f"quarters not in panel   : {len(fresh) + len(after_cut)}")
    print(f"    knowable by the cut : {len(fresh)}")
    print(f"    filed after the cut : {len(after_cut)}  (withheld; see STAGE 6)")

    if fresh.empty:
        print("\nNOTE  no new quarters became knowable since the panel was built. "
              "The\n      forward run proceeds on the panel's history alone, which is "
              "valid --\n      it just means no company has reported since.")

    # ---- year-ago match, against panel history as candidates ----------------
    panel_columns = list(panel.columns)
    carry = ["ticker", "cik", "cik_span_used", "period_end", "filed_date",
             "prediction_date", "eps_diluted", "n_filings_seen",
             "filing_lag_days"]
    combined = pd.concat(
        [panel[carry], fresh[carry]], ignore_index=True
    ).sort_values(["ticker", "period_end"])

    matched = attach_year_ago(combined)

    # Keep only this run's rows; the panel's rows keep their committed columns.
    fresh_keys = set(zip(fresh["ticker"], fresh["period_end"]))
    is_fresh = [
        (ticker, period_end) in fresh_keys
        for ticker, period_end in zip(matched["ticker"], matched["period_end"])
    ]
    new_rows = matched.loc[np.array(is_fresh)].copy()

    if not new_rows.empty:
        new_rows["prediction_date_year_ago"] = (
            new_rows["filed_date_year_ago"] + pd.Timedelta(days=1)
        )

        split_events = detect_split_events(all_facts)
        established = [event for event in split_events if event["established"]]
        print(f"splits detected         : {len(split_events)} "
              f"({len(established)} bounded)")

        new_rows = apply_split_adjustment(new_rows, split_events)
        new_rows = attach_label(new_rows)

        before_gate = len(new_rows)
        new_rows, gate_dropped = gate_on_membership(new_rows, intervals)
        print(f"membership gate         : {len(new_rows)} kept, "
              f"{len(gate_dropped)} dropped of {before_gate}")

        flags = membership.set_index("ticker")
        new_rows["is_removed_name"] = (
            new_rows["ticker"].map(flags["is_removed_name"]).fillna(False)
        )
        new_rows = new_rows[panel_columns]
    else:
        new_rows = panel.iloc[0:0].copy()

    history = pd.concat([panel, new_rows], ignore_index=True)
    history = history.sort_values(["ticker", "period_end"]).reset_index(drop=True)

    print(f"\nhistory for features    : {len(history)} rows "
          f"({len(panel)} panel + {len(new_rows)} new)")
    labelled = int(history[LABEL].notna().sum())
    print(f"    labelled            : {labelled}")

    return history, new_rows, after_cut


# --------------------------------------------------------------------------
# Stage 4 -- the pending rows: one per member, the quarter not yet filed
# --------------------------------------------------------------------------


def build_pending(history: pd.DataFrame, members: list,
                  cut: pd.Timestamp) -> pd.DataFrame:
    """One row per member ticker, standing for its NEXT quarterly report.

    The row carries no EPS, no year-ago comparison and no label, because the
    filing does not exist yet. It carries a sentinel prediction_date of cut + 1
    day purely so the lag machinery can rank it last within its ticker; that
    sentinel is never presented as an estimate of the filing date.

    expected_period_end IS an estimate -- the ticker's own median quarter length
    added to its last known period_end -- and is labelled as one in the output.
    It exists so the scorer can sanity-check which quarter it is matching, not
    as a claim about the calendar.
    """
    section("STAGE 4 -- pending quarters")

    records = []
    skipped_thin = []
    skipped_stale = []
    skipped_no_history = []

    by_ticker = {
        ticker: group.sort_values("prediction_date")
        for ticker, group in history.groupby("ticker", sort=False)
    }

    for ticker in members:
        group = by_ticker.get(ticker)
        if group is None or group.empty:
            skipped_no_history.append(ticker)
            continue

        if len(group) < MIN_HISTORY_QUARTERS:
            skipped_thin.append(ticker)
            continue

        silence = (cut - group.iloc[-1]["prediction_date"]).days
        if silence > MAX_REPORTING_GAP_DAYS:
            skipped_stale.append((ticker, silence))
            continue

        recent = group.tail(QUARTER_GAP_LOOKBACK)
        gaps = recent["period_end"].diff().dt.days.dropna()
        gap = int(gaps.median()) if len(gaps) else QUARTER_GAP_FALLBACK_DAYS

        last = group.iloc[-1]
        records.append({
            "ticker": ticker,
            "cik": last["cik"],
            "cik_span_used": last["cik_span_used"],
            "period_end": last["period_end"] + pd.Timedelta(days=gap),
            "filed_date": pd.NaT,
            "prediction_date": cut + PENDING_RANK_OFFSET,
            "eps_diluted": np.nan,
            "n_filings_seen": 0,
            LABEL: pd.NA,
            "in_index_at_prediction": True,
            "is_removed_name": bool(last["is_removed_name"]),
            "period_end_year_ago": pd.NaT,
            "prediction_date_year_ago": pd.NaT,
            "eps_year_ago": np.nan,
            "split_factor_applied": 1,
            "eps_year_ago_adjusted": np.nan,
            "split_ambiguous": False,
            "split_contaminated": False,
            "filing_lag_days": 0,
            "last_known_period_end": last["period_end"],
            "last_known_prediction_date": last["prediction_date"],
            "quarter_gap_days_used": gap,
        })

    pending = pd.DataFrame(records)

    print(f"members at cut          : {len(members)}")
    print(f"pending rows built      : {len(pending)}")
    print(f"    skipped, no history : {len(skipped_no_history)}")
    print(f"    skipped, < {MIN_HISTORY_QUARTERS} quarters: {len(skipped_thin)}"
          f"  (every growth lag would be imputed)")
    print(f"    skipped, silent > {MAX_REPORTING_GAP_DAYS}d: {len(skipped_stale)}"
          f"  (missed a reporting cycle; will not file)")
    if skipped_stale:
        worst = sorted(skipped_stale, key=lambda pair: -pair[1])[:8]
        print("        " + ", ".join(f"{t} ({days}d)" for t, days in worst)
              + (" ..." if len(skipped_stale) > 8 else ""))

    if pending.empty:
        stop("no member ticker has enough history to predict a next quarter.")

    eligible = len(members) - len(skipped_no_history)
    covered = len(pending) / eligible if eligible else 0.0
    print(f"coverage                : {covered:.1%} of eligible members")
    if covered < MIN_PREDICTED_FRACTION:
        stop(
            f"only {len(pending)} of {eligible} eligible members ({covered:.1%}) "
            f"have a predictable next quarter, below the "
            f"{MIN_PREDICTED_FRACTION:.0%} floor. Most were skipped as silent "
            f"for more than {MAX_REPORTING_GAP_DAYS} days, which at a real cut "
            f"means the incremental fetch did not deliver recent filings -- "
            f"check that EDGAR was reached and that "
            f"{FACT_CACHE_TEMPLATE.format(cut=cut.date())} is not a stale cache."
        )

    return pending


# --------------------------------------------------------------------------
# Stage 5 -- the ten frozen features
# --------------------------------------------------------------------------


def build_features(history: pd.DataFrame, pending: pd.DataFrame,
                   config: dict) -> tuple:
    """Compute the frozen feature set over history + pending.

    Returns (pending_with_features, history_with_features). Both come out of ONE
    ranked frame, which is the point: a pending row's growth_streak, lags and
    quarters_available are produced by the same imported code, in the same pass,
    as the training rows the model is fitted on.

    Recomputing over the whole history rather than splicing onto the committed
    data/features_v1.parquet is deliberate. The fresh quarters change every
    downstream rank within their ticker, so the committed table is stale by
    exactly the rows this run added.
    """
    section("STAGE 5 -- frozen features")

    panel_columns = list(history.columns)
    extras = [name for name in pending.columns if name not in panel_columns]

    frame = pd.concat([history, pending[panel_columns + extras]], ignore_index=True)

    # concat against rows carrying pd.NA can widen a nullable column to object,
    # and attach_growth_streak's "== 1" then yields object rather than a
    # nullable boolean. Restated explicitly rather than left to inference.
    frame[LABEL] = frame[LABEL].astype("Int64")
    for name in ("period_end", "filed_date", "prediction_date",
                 "period_end_year_ago", "prediction_date_year_ago"):
        frame[name] = pd.to_datetime(frame[name])
    for name in ("in_index_at_prediction", "split_ambiguous", "split_contaminated"):
        frame[name] = frame[name].astype(bool)

    check_preconditions(frame)

    ranked = attach_prediction_rank(frame)
    ranked, zero_base, no_year_ago = attach_eps_growth(ranked)

    for k in ALL_LAGS:
        ranked = attach_lag(ranked, k)

    ranked = attach_growth_streak(ranked)
    for k in FEATURE_LAGS:
        ranked[f"label_lag_{k}"] = ranked[f"label_yoy_lag_{k}"]

    # Built because build_features_v1 builds it, then never used: the config
    # excludes it as a target identity. Present so this frame is the same shape
    # as the one the model was selected and frozen on.
    ranked["growth_acceleration"] = (
        ranked["eps_growth_yoy"] - ranked["eps_growth_yoy_lag_1"]
    )

    try:
        assert_no_lag_leakage(ranked)
    except LeakageError as error:
        stop(f"leakage assertion failed on the forward frame: {error}")
    print("PASS  every lag carries a strictly earlier prediction_date")

    features = config["features"]["columns"]
    missing = [name for name in features if name not in ranked.columns]
    if missing:
        stop(f"frozen features missing after build: {missing}")

    identities = config["features"]["excluded_target_identities"]["columns"]
    leaked = [name for name in identities if name in features]
    if leaked:
        stop(f"target identity columns reached the frozen feature list: {leaked}")

    # quarter_gap_days_used exists only on pending rows, so it is the marker that
    # separates them -- not a date comparison, which would be one edit away from
    # quietly reclassifying a history row as a prediction.
    is_pending = ranked["quarter_gap_days_used"].notna()
    built = ranked.loc[is_pending].copy()
    history_features = ranked.loc[~is_pending].copy()

    print(f"ranked frame            : {len(ranked)} rows")
    print(f"    history             : {len(history_features)}")
    print(f"    pending             : {len(built)}")
    print(f"eps_growth_yoy null, no year-ago: {no_year_ago}, zero base: {zero_base}")
    print()
    print("feature presence on the predicted rows:")
    for name in features:
        filled = int(built[name].notna().sum())
        print(f"    {name:24s} {filled:5d}/{len(built)}")

    return built, history_features


# --------------------------------------------------------------------------
# Stage 6 -- the assertion that makes this a forward test
# --------------------------------------------------------------------------


def assert_no_resolvable_label(built: pd.DataFrame, after_cut: pd.DataFrame,
                               history: pd.DataFrame, cut: pd.Timestamp) -> None:
    """Refuse to predict anything whose actual label could already be worked out.

    label_yoy compares a quarter against the same quarter a year earlier, and the
    year-ago quarter is already in the panel. So a label becomes resolvable the
    instant the new filing exists -- there is no lag to hide behind. Four ways
    that could happen, each checked rather than assumed:

      1. The predicted row carries an EPS figure.
      2. It already carries a label.
      3. A filing for that ticker is dated after the cut, so the quarter has been
         reported and this run held the number in memory even though it withheld
         it from the history. That is what a backdated cut looks like from the
         inside.
      4. The predicted quarter is not actually later than the ticker's history.

    A failure here is not a bug to route around. It means the run would be
    describing something already settled, and the honest response is to stop.
    """
    section("STAGE 6 -- no predicted row may have a resolvable label")

    with_eps = built.loc[built["eps_diluted"].notna()]
    if len(with_eps):
        stop(f"{len(with_eps)} predicted rows carry an eps_diluted value, so "
             f"their label is already computable against the panel's year-ago "
             f"quarter.")
    print(f"PASS  no predicted row carries an EPS figure ({len(built)} rows)")

    with_label = built.loc[built[LABEL].notna()]
    if len(with_label):
        stop(f"{len(with_label)} predicted rows already carry a label.")
    print("PASS  no predicted row carries a label")

    if not after_cut.empty:
        already_filed = sorted(set(after_cut["ticker"]) & set(built["ticker"]))
        if already_filed:
            stop(
                f"{len(already_filed)} predicted "
                f"{'ticker has' if len(already_filed) == 1 else 'tickers have'} "
                f"a quarterly filing dated after the cut ({cut.date()}), so the "
                f"next quarter is already reported and its label is resolvable: "
                f"{', '.join(already_filed[:12])}"
                f"{' ...' if len(already_filed) > 12 else ''}. "
                f"Move the cut forward rather than predicting them."
            )
    print("PASS  no predicted ticker has a filing dated after the cut")

    last_known = history.groupby("ticker")["period_end"].max()
    stale = built.loc[built["period_end"] <= built["ticker"].map(last_known)]
    if len(stale):
        stop(f"{len(stale)} predicted rows sit at or before their ticker's last "
             f"known period_end; they are not future quarters.")
    print("PASS  every predicted quarter is later than its ticker's history")

    earliest_predicted = built["prediction_date"].min()
    latest_known = history["prediction_date"].max()
    if not latest_known < earliest_predicted:
        stop(f"history reaches {latest_known.date()}, which is not strictly "
             f"before the first predicted row.")
    print(f"PASS  all history ({latest_known.date()}) precedes every predicted row")


# --------------------------------------------------------------------------
# Stage 7 -- refit the frozen config to the cut, and predict
# --------------------------------------------------------------------------


def predict(built: pd.DataFrame, history_features: pd.DataFrame, config: dict,
            cut: pd.Timestamp) -> tuple:
    """Refit the frozen pipeline on everything knowable at the cut, then predict.

    This is the holdout's walk-forward rule advanced one step: train on every
    labelled row with prediction_date at or before the cut, predict rows that are
    strictly later. Nothing about the model is chosen here -- the config supplies
    the estimator, its parameters, the transform and the threshold, and
    build_pipeline is the same function the holdout ran.
    """
    section("STAGE 7 -- refit to the cut and predict")

    features = config["features"]["columns"]
    threshold = config["model"]["decision_threshold"]

    train = history_features.loc[
        history_features[LABEL].notna()
        & (history_features["prediction_date"] <= cut)
    ]

    if train.empty:
        stop("the training window is empty.")
    if not train["prediction_date"].max() < built["prediction_date"].min():
        stop("the training window is not strictly before the predicted rows.")

    pipeline = build_pipeline(config)

    # The config's dtype_cast: float64 with NaN for missing. Cast exactly as the
    # holdout did -- a nullable Int64 lag column reaching the ColumnTransformer's
    # passthrough carries pd.NA, which sklearn refuses rather than silently
    # coercing.
    pipeline.fit(train[features].astype("float64"),
                 train[LABEL].astype("int64").to_numpy())

    probabilities = pipeline.predict_proba(built[features].astype("float64"))[:, 1]

    print(f"training rows (labelled): {len(train)}")
    print(f"    window              : prediction_date <= {cut.date()}")
    print(f"    last training row   : {train['prediction_date'].max().date()}")
    print(f"    positive rate       : {train[LABEL].astype('int64').mean():.6f}")
    print(f"predicted rows          : {len(built)}")
    print(f"    mean probability    : {probabilities.mean():.4f}")
    print(f"    predicted beats     : {int((probabilities >= threshold).sum())} "
          f"({(probabilities >= threshold).mean():.1%})")

    return probabilities, train


# --------------------------------------------------------------------------
# Stage 8 -- write
# --------------------------------------------------------------------------


def write_predictions(built: pd.DataFrame, probabilities, train: pd.DataFrame,
                      config: dict, cut: pd.Timestamp, provenance: str,
                      panel_digest: str, output_path: str) -> None:
    """Write the predictions, with the run's provenance inside the file.

    The header is `#`-prefixed so the file stays an ordinary CSV
    (`pd.read_csv(path, comment="#")`), and it carries the frozen-universe note
    every prediction file is required to state.
    """
    section("STAGE 8 -- write")

    threshold = config["model"]["decision_threshold"]
    features = config["features"]["columns"]

    output = pd.DataFrame({
        "cut_date": cut.date().isoformat(),
        "ticker": built["ticker"].to_numpy(),
        "cik": built["cik"].to_numpy(),
        "expected_period_end": built["period_end"].dt.date.astype(str).to_numpy(),
        "last_known_period_end":
            built["last_known_period_end"].dt.date.astype(str).to_numpy(),
        "last_known_prediction_date":
            built["last_known_prediction_date"].dt.date.astype(str).to_numpy(),
        "quarter_gap_days_used": built["quarter_gap_days_used"].to_numpy(),
        "predicted_probability": probabilities,
        "predicted_label": (probabilities >= threshold).astype(int),
    })

    # The ten features ride along so a prediction can be audited against the
    # frozen config without re-fetching EDGAR.
    for name in features:
        output[name] = built[name].to_numpy()

    output = output.sort_values("ticker").reset_index(drop=True)

    header = [
        "Live forward-test predictions. PREDICTIONS ONLY -- no outcome, no score.",
        "",
        f"cut_date                 : {cut.date()}  (predictions made as of this date)",
        f"produced_at_utc          : {pd.Timestamp.utcnow().isoformat()}",
        f"code_sha                 : {git_sha()}",
        f"working_tree_clean       : {git_is_clean()}",
        f"config                   : config/frozen_model.json revision {config['revision']}",
        f"model                    : {config['model']['variant_name']}",
        f"decision_threshold       : {threshold}",
        f"panel_sha256             : {panel_digest}",
        f"training_rows_labelled   : {len(train)}",
        f"training_window          : prediction_date <= {cut.date()}",
        f"training_positive_rate   : {train[LABEL].astype('int64').mean():.6f}",
        "",
        "WHAT IS PREDICTED",
        "  Each row is a ticker's NEXT quarterly EPS report -- a quarter that had",
        "  NOT been filed as of cut_date. The label is label_yoy: whether diluted",
        "  EPS beats the same quarter one year earlier, split-adjusted.",
        "",
        "  expected_period_end is an ESTIMATE: the ticker's median quarter length",
        "  added to its last known period_end. It is recorded so the scorer can",
        "  check which quarter it matched, and is not a claim about the filing",
        "  calendar. The scorer should match on ticker and first filing after",
        "  cut_date, using expected_period_end only as a sanity check.",
        "",
        "FROZEN UNIVERSE -- READ THIS BEFORE USING THESE ROWS",
        f"  Membership comes from {provenance},",
        "  pinned at 2025-05-27 -- the last revision carrying the added/removed",
        "  changes table. Index changes after that date are invisible to this run.",
        f"  Real S&P 500 turnover is ~{INDEX_TURNOVER_NAMES_PER_YEAR} names a year, so this universe drifts from",
        "  the real index at roughly that rate, and the drift compounds with every",
        "  quarter the forward test runs. The live Wikipedia page is NEVER used as",
        "  a fallback: substituting today's constituent list would reintroduce the",
        "  survivorship bias the membership gate exists to prevent.",
        "",
        "SCORING",
        "  This file is committed BEFORE the outcomes exist, and that timestamp is",
        "  the experiment. Scoring is a separate script and a separate commit.",
        "  Nothing in this file may be recomputed once outcomes are known.",
        "",
    ]

    os.makedirs("data", exist_ok=True)
    with open(output_path, "w") as handle:
        for line in header:
            handle.write(f"# {line}\n" if line else "#\n")
        output.to_csv(handle, index=False)

    print(f"predictions -> {output_path}  ({len(output)} rows)")
    print(f"columns     : {list(output.columns)}")


# --------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Live forward test: predict the next unfiled quarter. "
                    "Predicts only; never scores."
    )
    parser.add_argument("--as-of", dest="as_of", default=None,
                        help="cut date (YYYY-MM-DD); defaults to today (UTC)")
    args = parser.parse_args()

    cut = parse_cut(args.as_of)
    config, panel, panel_digest, output_path = preflight(cut)

    intervals, membership, members, provenance = load_membership(cut)
    all_facts = fetch_new_facts(members, cut)
    history, _new_rows, after_cut = build_new_history(
        all_facts, panel, intervals, membership, cut
    )

    pending = build_pending(history, members, cut)
    built, history_features = build_features(history, pending, config)

    assert_no_resolvable_label(built, after_cut, history, cut)
    probabilities, train = predict(built, history_features, config, cut)

    write_predictions(built, probabilities, train, config, cut, provenance,
                      panel_digest, output_path)

    section("PANEL READ-ONLY CHECK")
    if file_digest(PANEL_PARQUET) != panel_digest:
        stop(f"{PANEL_PARQUET} changed during the run. The panel is read-only.")
    print(f"PASS  {PANEL_PARQUET} is byte-identical to the start of the run")

    section("DONE -- PREDICTIONS ONLY")
    print("Nothing was scored, and no outcome was computed.")
    print()
    print("Commit this file NOW, before the quarters it predicts are filed. The")
    print("commit timestamp is what makes this a test rather than a re-description")
    print("of history. Scoring is a separate script and a separate commit.")


if __name__ == "__main__":
    main()
