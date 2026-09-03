#!/usr/bin/env python3
"""Phase 0 deliverable: the full S&P 500 point-in-time EPS panel.

Builds data/eps_panel.parquet -- one row per (ticker, period_end) for every
ever-constituent of the index between 2011 and 2025 that resolves to a filer.

Scope is still panel + label. No features are engineered and no split is made.

What makes a row admissible:
  - its EPS value is the FIRST one filed for that period (point-in-time correct)
  - it became public on a normal reporting schedule (filing lag <= 120 days)
  - the year-ago comparison is in the same share units (splits adjusted)
  - and the ticker was actually IN the index on the prediction date

That last rule is what separates this from a survivors-only panel: a company's
2013 quarters do not belong here if it joined the index in 2016, and a company
that left in 2015 must still contribute the quarters it was a member for.

Run:  python scripts/build_eps_panel.py
"""

from __future__ import annotations

import os
import sys

import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

from src.data_loader import (  # noqa: E402
    SEC_SLEEP_SECONDS,
    describe_spans,
    fail,
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

# --------------------------------------------------------------------------
# Configuration
# --------------------------------------------------------------------------

# XBRL tagging was phased in over 2009-2011. Before it settled, a quarter's
# EARLIEST XBRL record is frequently a later comparative rather than its own
# original filing -- the number was public on time in HTML, but was not tagged
# until a subsequent filing repeated it. Measured across the basket: 42 of the
# 48 periods whose first filing lands >90 days after period end have a period
# end before 2011. Since prediction_date is derived from filed_date, those rows
# would carry a filing date up to a year later than the market's actual
# knowledge date. The panel therefore starts here.
STUDY_START = pd.Timestamp("2011-01-01")
WINDOW_END = pd.Timestamp("2025-12-31")

# A quarter filed more than this long after period end did not become public on
# a normal reporting schedule; its filed_date cannot be trusted as the moment the
# market learned the number. Dropped and logged rather than silently kept.
MAX_FILING_LAG_DAYS = 120

# The year-ago quarter is matched by DATE, not by row offset: a missing quarter
# would make a positional shift silently compare the wrong periods.
YEAR_AGO_TARGET_DAYS = 365
YEAR_AGO_TOLERANCE_DAYS = 45

# A split is detected when a later filing restates a period's EPS by almost
# exactly an integer factor. Real restatements do not land on 4.000; splits do.
SPLIT_RATIO_TOLERANCE = 0.02
SPLIT_MIN_FACTOR = 2

# Label balance outside this band means the baseline is lopsided enough to matter.
# The split-adjusted label sat near 61% on the ten-name basket, so the naive
# "always predict 1" baseline is ~61% there. Judge a model against that, not 50%.
BALANCE_WARN_LOW = 0.40
BALANCE_WARN_HIGH = 0.65

# Tickers contributing fewer than this many quarters are reported; a name with a
# handful of rows cannot support a per-ticker view of anything.
MIN_QUARTERS_REPORTED = 8

PANEL_PARQUET = "data/eps_panel.parquet"
DROPPED_CSV = "data/dropped_periods.csv"
UNRESOLVED_CSV = "data/unresolved_tickers.csv"

# Fetched facts are cached per CIK so a rebuild does not re-hit EDGAR ~900 times.
# Delete data/fact_cache/ to force a refresh.
FACT_CACHE_DIR = "data/fact_cache"
USE_FACT_CACHE = True


def section(title: str) -> None:
    print("\n" + "=" * 78)
    print(title)
    print("=" * 78)


# --------------------------------------------------------------------------
# Stage 1 -- universe and facts
# --------------------------------------------------------------------------


def fetch_universe_facts(intervals: dict) -> tuple:
    """Fetch quarterly EPS for every ever-constituent that resolves."""
    section("STAGE 1 -- resolve universe and fetch facts")

    cik_map = load_ticker_cik_map()
    recovered = load_recovered_ciks()
    print(f"ticker map entries      : {len(cik_map)}")
    print(f"name-recovered CIKs     : {len(recovered)}")
    print(f"ever-constituents       : {len(intervals)}")

    os.makedirs(FACT_CACHE_DIR, exist_ok=True)

    frames = []
    routes = {}
    unresolved = []
    empty = []

    tickers = sorted(intervals)
    estimate = len(tickers) * SEC_SLEEP_SECONDS / 60
    print(f"\nfetching (throttle floor ~{estimate:.0f} min, cache {'on' if USE_FACT_CACHE else 'off'})\n")

    for position, ticker in enumerate(tickers, start=1):
        spans, route = resolve_ticker_spans(ticker, cik_map, recovered)
        routes[ticker] = (route, describe_spans(spans))

        if not spans:
            unresolved.append({"ticker": ticker, "reason": "no_cik_from_any_source"})
            continue

        cache_path = os.path.join(FACT_CACHE_DIR, f"{ticker}.parquet")
        if USE_FACT_CACHE and os.path.exists(cache_path):
            facts = pd.read_parquet(cache_path)
        else:
            facts = fetch_quarterly_eps_spanned(ticker, spans)
            if facts is not None:
                facts.to_parquet(cache_path, index=False)

        if facts is None or facts.empty:
            empty.append({"ticker": ticker, "reason": "no_eps_facts"})
            continue

        facts = facts.copy()
        facts["cik_span_used"] = describe_spans(spans)
        facts["resolution_route"] = route
        frames.append(facts)

        if position % 100 == 0:
            print(f"    {position}/{len(tickers)}  ({len(frames)} with facts)")

    if not frames:
        fail("No ticker in the universe returned any EPS facts.")

    all_facts = pd.concat(frames, ignore_index=True)

    print(f"\nresolved with facts     : {len(frames)}")
    print(f"resolved but no facts   : {len(empty)}")
    print(f"unresolved              : {len(unresolved)}")
    print(f"raw quarterly facts     : {len(all_facts)}")

    route_counts = pd.Series({t: r for t, (r, _s) in routes.items()}).value_counts()
    print("\nresolution route:")
    for route, count in route_counts.items():
        print(f"    {route:18s} {count:5d}")

    os.makedirs("data", exist_ok=True)
    pd.DataFrame(unresolved + empty).to_csv(UNRESOLVED_CSV, index=False)
    print(f"\nunresolved/empty logged -> {UNRESOLVED_CSV}")

    return all_facts, routes


# --------------------------------------------------------------------------
# Stage 2 -- dedupe to first filing
# --------------------------------------------------------------------------


def dedupe_to_first_filing(all_facts: pd.DataFrame) -> pd.DataFrame:
    """Collapse to one row per (ticker, period_end), keeping the EARLIEST filing.

    Later filings repeat a quarter as a comparative, and across a stock split they
    repeat it with a SPLIT-ADJUSTED value that did not exist at the time. Keeping
    MIN(filed) is what makes each row point-in-time correct.
    """
    filings_per_period = all_facts.groupby(["ticker", "period_end"]).size()
    filings_per_period = filings_per_period.rename("n_filings_seen")

    ordered = all_facts.sort_values(["ticker", "period_end", "filed_date"])
    deduped = ordered.groupby(["ticker", "period_end"], as_index=False).first()

    return deduped.merge(filings_per_period, on=["ticker", "period_end"], how="left")


# --------------------------------------------------------------------------
# Stage 3 -- filters
# --------------------------------------------------------------------------


def apply_lag_filter(deduped: pd.DataFrame) -> tuple:
    """Drop periods whose first filing came too late to trust its date."""
    lag_too_long = deduped["filing_lag_days"] > MAX_FILING_LAG_DAYS

    dropped = deduped.loc[lag_too_long].copy()
    dropped["reason"] = f"filing_lag_days > {MAX_FILING_LAG_DAYS}"

    return deduped.loc[~lag_too_long].copy(), dropped


# --------------------------------------------------------------------------
# Stage 4 -- year-ago match
# --------------------------------------------------------------------------


def attach_year_ago(panel: pd.DataFrame) -> pd.DataFrame:
    """Attach each row's year-ago quarter by DATE proximity, never row offset."""
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

    left_sorted = panel.sort_values("year_ago_target")
    right_sorted = candidates.sort_values("period_end_year_ago")

    # direction="nearest" with a tolerance implements "~365 days prior, +/- 45".
    return pd.merge_asof(
        left_sorted,
        right_sorted,
        left_on="year_ago_target",
        right_on="period_end_year_ago",
        by="ticker",
        direction="nearest",
        tolerance=pd.Timedelta(days=YEAR_AGO_TOLERANCE_DAYS),
    )


# --------------------------------------------------------------------------
# Stage 5 -- splits
# --------------------------------------------------------------------------


def detect_split_events(all_facts: pd.DataFrame) -> list:
    """Find splits, and bound each effective date, from EDGAR restatements alone.

    A period whose original value and a later restated value differ by almost
    exactly an integer factor is a split. The effective date is never stated but
    IS bounded: a filing made on date D reports in the units current on D, so the
    split falls in (last pre-split filing, first post-split filing].
    """
    usable = all_facts[all_facts["eps_diluted"].abs() > 0.01]
    evidence = {}

    for (ticker, _period_end), group in usable.groupby(["ticker", "period_end"]):
        ordered = group.sort_values("filed_date")
        original = ordered.iloc[0]

        for _index, later in ordered.iloc[1:].iterrows():
            if abs(later["eps_diluted"] - original["eps_diluted"]) <= 1e-9:
                continue

            ratio = original["eps_diluted"] / later["eps_diluted"]
            factor = round(ratio)

            if factor < SPLIT_MIN_FACTOR:
                continue
            if abs(ratio - factor) / factor > SPLIT_RATIO_TOLERANCE:
                continue

            bucket = evidence.setdefault((ticker, int(factor)), {"pre": [], "post": []})
            bucket["pre"].append(original["filed_date"])
            bucket["post"].append(later["filed_date"])

    events = []
    for (ticker, factor), dates in evidence.items():
        window_start = max(dates["pre"])
        window_end = min(dates["post"])
        events.append(
            {
                "ticker": ticker,
                "factor": factor,
                "window_start": window_start,
                "window_end": window_end,
                # Crossed bounds mean the evidence contradicts itself.
                "established": window_start < window_end,
            }
        )

    return events


def apply_split_adjustment(panel: pd.DataFrame, split_events: list) -> pd.DataFrame:
    """Adjust the YEAR-AGO EPS only; eps_diluted is never touched.

    Each of the two filings is placed on its own side of the split window:
    on/before window_start is old units, on/after window_end is new units, and a
    filing landing inside the window is undecidable -- those rows are flagged and
    their label nulled rather than guessed.
    """
    panel = panel.copy()
    panel["split_factor_applied"] = 1
    panel["split_ambiguous"] = False

    has_match = panel["period_end_year_ago"].notna()
    by_ticker = {}
    for event in split_events:
        by_ticker.setdefault(event["ticker"], []).append(event)

    for ticker, events in by_ticker.items():
        is_ticker = panel["ticker"] == ticker
        if not is_ticker.any():
            continue

        for event in events:
            if not event["established"]:
                panel.loc[is_ticker & has_match, "split_ambiguous"] = True
                continue

            year_ago_filed = panel["filed_date_year_ago"]
            current_filed = panel["filed_date"]

            year_ago_old = year_ago_filed <= event["window_start"]
            year_ago_new = year_ago_filed >= event["window_end"]
            current_old = current_filed <= event["window_start"]
            current_new = current_filed >= event["window_end"]

            intervenes = year_ago_old & current_new
            same_units = (year_ago_old & current_old) | (year_ago_new & current_new)

            applies = is_ticker & has_match & intervenes
            unclear = is_ticker & has_match & ~intervenes & ~same_units

            # Multiplied, not assigned: two splits between the same pair compound.
            panel.loc[applies, "split_factor_applied"] *= event["factor"]
            panel.loc[unclear, "split_ambiguous"] = True

    panel["eps_year_ago_adjusted"] = panel["eps_year_ago"] / panel["split_factor_applied"]
    panel["split_contaminated"] = (panel["split_factor_applied"] > 1) | panel["split_ambiguous"]

    return panel


def attach_label(panel: pd.DataFrame) -> pd.DataFrame:
    """label_yoy = 1 when EPS beat the SPLIT-ADJUSTED year-ago quarter, else 0."""
    panel = panel.copy()

    # Strictly greater: a flat quarter is not a beat.
    beat = panel["eps_diluted"] > panel["eps_year_ago_adjusted"]

    # Null where there is no year-ago row, and where a split could not be pinned.
    usable = panel["eps_year_ago_adjusted"].notna() & ~panel["split_ambiguous"]
    panel["label_yoy"] = beat.astype("Int64").where(usable)

    return panel


# --------------------------------------------------------------------------
# Stage 6 -- membership gating
# --------------------------------------------------------------------------


def gate_on_membership(panel: pd.DataFrame, intervals: dict) -> tuple:
    """Keep only rows whose ticker was IN the index on the prediction date.

    This is what makes the panel tradeable. An observation is admissible only if
    a portfolio could have acted on it: the company had to be a constituent at
    the moment the number became public. Gating on prediction_date rather than
    period_end is deliberate -- prediction_date is when the decision is taken.
    """
    panel = panel.copy()
    panel["in_index_at_prediction"] = [
        is_member_at(intervals, ticker, moment)
        for ticker, moment in zip(panel["ticker"], panel["prediction_date"])
    ]

    outside = panel[~panel["in_index_at_prediction"]].copy()
    outside["reason"] = "not_in_index_at_prediction_date"

    return panel[panel["in_index_at_prediction"]].copy(), outside


# --------------------------------------------------------------------------
# Stage 7 -- assertions
# --------------------------------------------------------------------------


def verify_panel(panel: pd.DataFrame) -> None:
    """Every invariant the panel is supposed to hold, checked rather than assumed."""
    section("ASSERTIONS")

    duplicate_mask = panel.duplicated(subset=["ticker", "period_end"], keep=False)
    if int(duplicate_mask.sum()):
        offenders = panel.loc[duplicate_mask, ["ticker", "period_end"]]
        print(offenders.head(20).to_string(index=False), file=sys.stderr)
        fail(f"{int(duplicate_mask.sum())} duplicate (ticker, period_end) rows.")
    print("PASS  no duplicate (ticker, period_end) rows")

    not_after = panel["prediction_date"] <= panel["period_end"]
    if int(not_after.sum()):
        fail(f"{int(not_after.sum())} rows have prediction_date <= period_end.")
    print("PASS  prediction_date > period_end for every row")

    if int(panel["filed_date"].isna().sum()):
        fail(f"{int(panel['filed_date'].isna().sum())} rows have a null filed_date.")
    print("PASS  no null filed_date")

    offset = (panel["prediction_date"] - panel["filed_date"]).dt.days
    if not (offset == 1).all():
        fail("prediction_date is not exactly filed_date + 1 day for every row.")
    print("PASS  prediction_date == filed_date + 1 day")

    # The membership gate is the new invariant: no row may survive that the
    # index did not actually contain when the number became public.
    if not panel["in_index_at_prediction"].all():
        outside = int((~panel["in_index_at_prediction"]).sum())
        fail(f"{outside} output rows are not in the index at prediction_date.")
    print("PASS  every row is in the index at prediction_date")

    if (panel["filing_lag_days"] > MAX_FILING_LAG_DAYS).any():
        fail(f"rows survived with filing_lag_days > {MAX_FILING_LAG_DAYS}.")
    print(f"PASS  no row with filing lag > {MAX_FILING_LAG_DAYS} days")

    if (panel["period_end"] < STUDY_START).any():
        fail(f"rows survived with period_end before {STUDY_START.date()}.")
    print(f"PASS  no row before STUDY_START ({STUDY_START.date()})")


# --------------------------------------------------------------------------
# Reporting
# --------------------------------------------------------------------------


def report_panel(panel: pd.DataFrame, dropped: pd.DataFrame) -> None:
    section("PANEL SHAPE")

    print(f"rows                    : {len(panel)}")
    print(f"tickers                 : {panel['ticker'].nunique()}")
    print(f"period_end span         : {panel['period_end'].min().date()} .. {panel['period_end'].max().date()}")

    print("\nrows per year (by period_end):")
    per_year = panel.groupby(panel["period_end"].dt.year).size()
    for year, count in per_year.items():
        print(f"    {year}  {count:5d}  {'#' * min(count // 10, 60)}")

    # --- per-ticker depth --------------------------------------------------
    quarters = panel.groupby("ticker").size()
    thin = quarters[quarters < MIN_QUARTERS_REPORTED]
    print(f"\ntickers with fewer than {MIN_QUARTERS_REPORTED} quarters: {len(thin)} of {len(quarters)}")
    print(f"    median quarters per ticker: {int(quarters.median())}")
    print(f"    rows held by thin tickers : {int(thin.sum())} ({thin.sum() / len(panel):.1%})")

    section("LABEL BALANCE")

    labelled = panel.dropna(subset=["label_yoy"])
    null_count = len(panel) - len(labelled)
    no_match = int(panel["eps_year_ago"].isna().sum())
    ambiguous = int(panel["split_ambiguous"].sum())

    print(f"rows with a label       : {len(labelled)}")
    print(f"rows with NULL label    : {null_count}  "
          f"({no_match} no year-ago match, {ambiguous} unresolvable split)")

    if labelled.empty:
        fail("No rows carry a label.")

    overall_rate = labelled["label_yoy"].mean()
    print(f"\noverall positive rate   : {overall_rate:.1%}  "
          f"({int(labelled['label_yoy'].sum())}/{len(labelled)})")

    # --- the survivorship comparison ---------------------------------------
    # If removed names beat less often, a survivors-only panel would have taught
    # the model an optimism that the real index never had.
    print("\nby index status:")
    print(f"    {'group':16s} {'rows':>7s} {'labelled':>9s} {'pos':>7s} {'rate':>8s}")
    for is_removed, group in labelled.groupby("is_removed_name"):
        label = "removed names" if is_removed else "never removed"
        rate = group["label_yoy"].mean()
        print(f"    {label:16s} {len(group):7d} {len(group):9d} "
              f"{int(group['label_yoy'].sum()):7d} {rate:8.1%}")

    rates = labelled.groupby("is_removed_name")["label_yoy"].mean()
    if len(rates) == 2:
        gap = rates.get(True, float("nan")) - rates.get(False, float("nan"))
        print(f"\n    removed minus never-removed: {gap:+.1%}")

    if not (BALANCE_WARN_LOW <= overall_rate <= BALANCE_WARN_HIGH):
        print(f"\n!!! WARNING: overall balance {overall_rate:.1%} is outside "
              f"{BALANCE_WARN_LOW:.0%}-{BALANCE_WARN_HIGH:.0%}.")
        print(f"!!! A constant baseline scores {max(overall_rate, 1 - overall_rate):.1%}.")
    else:
        print(f"\nbalance within {BALANCE_WARN_LOW:.0%}-{BALANCE_WARN_HIGH:.0%}; no warning.")

    section("DROPS")
    print(f"total dropped rows      : {len(dropped)}")
    for reason, group in dropped.groupby("reason"):
        print(f"    {len(group):6d}  {reason}")


# --------------------------------------------------------------------------
# Entry point
# --------------------------------------------------------------------------


def main() -> None:
    section("STAGE 0 -- index membership")
    current, changes_raw, provenance = load_tables()
    print(f"provenance              : {provenance}")

    changes = normalise_changes(changes_raw)
    in_window = changes[changes["date"].between(STUDY_START, WINDOW_END)]
    print(f"changes in window       : {len(in_window)}")

    intervals, membership = build_intervals(current, changes, STUDY_START)
    print(f"tickers with membership : {len(intervals)}")
    print(f"    still in index      : {int(membership['still_in_index'].sum())}")
    print(f"    removed at any point: {int(membership['is_removed_name'].sum())}")
    print(f"    multi-stint tickers : {int((membership['n_stints'] > 1).sum())}")

    all_facts, _routes = fetch_universe_facts(intervals)

    # ---- dedupe + filter --------------------------------------------------
    section("STAGE 2 -- dedupe and filter")
    deduped = dedupe_to_first_filing(all_facts)
    print(f"after dedupe on (ticker, end) : {len(deduped)}  [kept MIN(filed)]")

    after_lag, lag_dropped = apply_lag_filter(deduped)
    print(f"after lag filter (<= {MAX_FILING_LAG_DAYS}d)      : {len(after_lag)}")

    # ---- label ------------------------------------------------------------
    # Matched BEFORE the window trim so a surviving 2010 quarter can serve as the
    # year-ago lookup for an in-window 2011 quarter; warm-up rows never reach the
    # output.
    section("STAGE 3 -- year-ago match, splits, label")
    matched = attach_year_ago(after_lag)

    in_study = matched["period_end"] >= STUDY_START
    panel = matched.loc[in_study].copy()
    window_dropped = matched.loc[~in_study].copy()
    window_dropped["reason"] = f"period_end < STUDY_START ({STUDY_START.date()})"
    print(f"study window {STUDY_START.date()} onward: {len(panel)} rows "
          f"({len(window_dropped)} warm-up rows used for lookup only)")

    panel["prediction_date"] = panel["filed_date"] + pd.Timedelta(days=1)
    panel["prediction_date_year_ago"] = panel["filed_date_year_ago"] + pd.Timedelta(days=1)

    split_events = detect_split_events(all_facts)
    established = [event for event in split_events if event["established"]]
    print(f"splits detected         : {len(split_events)} ({len(established)} bounded)")

    panel = apply_split_adjustment(panel, split_events)
    panel = attach_label(panel)
    print(f"year-ago EPS adjusted   : {int((panel['split_factor_applied'] > 1).sum())} rows")
    print(f"nulled as unresolvable  : {int(panel['split_ambiguous'].sum())} rows")

    # ---- membership gating ------------------------------------------------
    section("STAGE 4 -- membership gating")
    before_gate = len(panel)
    panel, gate_dropped = gate_on_membership(panel, intervals)
    print(f"rows before gate        : {before_gate}")
    print(f"rows in index at prediction_date: {len(panel)}")
    print(f"dropped (not a member)  : {len(gate_dropped)}  ({len(gate_dropped) / before_gate:.1%})")

    # ---- assemble ---------------------------------------------------------
    flags = membership.set_index("ticker")
    panel["is_removed_name"] = panel["ticker"].map(flags["is_removed_name"]).fillna(False)

    output_columns = [
        "ticker",
        "cik",
        "cik_span_used",
        "period_end",
        "filed_date",
        "prediction_date",
        "eps_diluted",
        "n_filings_seen",
        "label_yoy",
        "in_index_at_prediction",
        "is_removed_name",
        # Label provenance: lets the +/-45 day match and the split adjustment be
        # audited, and the UNADJUSTED label rebuilt. Not features.
        "period_end_year_ago",
        "prediction_date_year_ago",
        "eps_year_ago",
        "split_factor_applied",
        "eps_year_ago_adjusted",
        "split_ambiguous",
        "split_contaminated",
        "filing_lag_days",
    ]
    panel = panel[output_columns].sort_values(["ticker", "period_end"]).reset_index(drop=True)

    verify_panel(panel)

    # ---- drops ------------------------------------------------------------
    log_columns = ["ticker", "period_end", "filed_date", "filing_lag_days", "reason"]
    dropped = pd.concat(
        [frame[log_columns] for frame in (lag_dropped, window_dropped, gate_dropped)],
        ignore_index=True,
    )
    dropped = dropped.rename(
        columns={"period_end": "end", "filed_date": "filed", "filing_lag_days": "lag_days"}
    )
    os.makedirs("data", exist_ok=True)
    dropped.sort_values(["reason", "ticker", "end"]).to_csv(DROPPED_CSV, index=False)

    report_panel(panel, dropped)

    section("OUTPUT")
    panel.to_parquet(PANEL_PARQUET, index=False)
    print(f"panel   -> {PANEL_PARQUET}  ({len(panel)} rows)")
    print(f"drops   -> {DROPPED_CSV}  ({len(dropped)} rows)")
    print(f"columns : {list(panel.columns)}")


if __name__ == "__main__":
    main()
