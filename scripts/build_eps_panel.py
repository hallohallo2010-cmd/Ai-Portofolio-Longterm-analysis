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
import time

import pandas as pd
import requests

# --------------------------------------------------------------------------
# Configuration -- edit these, nothing below.
# --------------------------------------------------------------------------

TICKERS = ["AAPL", "MSFT", "JNJ", "JPM", "XOM", "PG", "WMT", "CAT", "NEE", "T"]

# SEC's ticker->CIK map points at the CURRENT registrant for a ticker. When a
# company reorganizes, the ticker is repointed to the new holding-company CIK,
# which carries only post-reorganization filings -- decades of history sit under
# the old CIK and are silently invisible. Verified: the map sends XOM to CIK
# 2115436 ("ExxonMobil Holdings Corp", 4 EPS records from 2025-06-30), while CIK
# 34088 ("Exxon Mobil Corporation") holds 224 records back to 2007-12-31.
CIK_OVERRIDES = {
    "XOM": "0000034088",  # historical Exxon Mobil Corporation filer
}

# SEC EDGAR blocks requests that do not identify a real contact.
# REPLACE THIS with your own address or the script will refuse to call EDGAR.
SEC_CONTACT_EMAIL = "REPLACE_ME@example.com"
SEC_APP_NAME = "earnings-surprise-research"

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
BALANCE_WARN_LOW = 0.40
BALANCE_WARN_HIGH = 0.60

# EDGAR's fair-access limit is 10 requests/second; one per 0.5s is 2/s.
SEC_SLEEP_SECONDS = 0.5

# A quarterly XBRL fact covers a ~3 month duration. Fiscal quarters are ragged
# (13 weeks, 4-4-5 calendars, 52/53-week years), so accept a generous window.
QUARTER_MIN_DAYS = 60
QUARTER_MAX_DAYS = 110

# A split is detected when a later filing restates a period's EPS by almost
# exactly an integer factor. Real restatements do not land on 4.000; splits do.
SPLIT_RATIO_TOLERANCE = 0.02
SPLIT_MIN_FACTOR = 2

PANEL_PARQUET = "data/eps_panel.parquet"
DROPPED_CSV = "data/dropped_periods.csv"
SPLIT_CSV = "data/split_contaminated_periods.csv"

SEC_TICKER_MAP_URL = "https://www.sec.gov/files/company_tickers.json"
SEC_CONCEPT_URL = (
    "https://data.sec.gov/api/xbrl/companyconcept/"
    "CIK{cik}/us-gaap/EarningsPerShareDiluted.json"
)
SEC_TIMEOUT_SECONDS = 30


# --------------------------------------------------------------------------
# Small helpers
# --------------------------------------------------------------------------


def fail(message: str) -> None:
    """Abort loudly with a non-zero exit. All hard failures funnel through here."""
    print(f"\n*** FATAL: {message}", file=sys.stderr)
    sys.exit(1)


def section(title: str) -> None:
    print("\n" + "=" * 78)
    print(title)
    print("=" * 78)


def sec_headers() -> dict:
    """Headers EDGAR requires. A missing/blank User-Agent gets a 403."""
    if "REPLACE_ME" in SEC_CONTACT_EMAIL:
        fail(
            "SEC_CONTACT_EMAIL is still the placeholder. EDGAR requires a real "
            "contact address in the User-Agent header; edit the constant at the "
            "top of this script before running."
        )
    return {
        "User-Agent": f"{SEC_APP_NAME} ({SEC_CONTACT_EMAIL})",
        "Accept-Encoding": "gzip, deflate",
        "Accept": "application/json",
    }


def sec_get(url: str) -> requests.Response:
    """Single throttled EDGAR GET. Every outbound call goes through here so the
    rate limit cannot be bypassed by adding a call site later."""
    time.sleep(SEC_SLEEP_SECONDS)  # sleep BEFORE, so bursts cannot slip through
    return requests.get(url, headers=sec_headers(), timeout=SEC_TIMEOUT_SECONDS)


# --------------------------------------------------------------------------
# Stage 1 -- fetch raw quarterly facts
# --------------------------------------------------------------------------


def load_ticker_cik_map() -> dict:
    """Fetch SEC's ticker->CIK map once and index it by upper-case ticker."""
    response = sec_get(SEC_TICKER_MAP_URL)
    if response.status_code != 200:
        fail(
            f"SEC ticker map request failed with HTTP {response.status_code}. "
            f"A 403 here almost always means the User-Agent was rejected."
        )

    raw_map = response.json()
    if not raw_map:
        fail("SEC ticker map came back empty.")

    indexed = {}
    for entry in raw_map.values():
        symbol = entry.get("ticker", "").upper()
        # CIKs arrive as ints; the API path needs them padded to 10 digits.
        indexed[symbol] = str(entry["cik_str"]).zfill(10)

    return indexed


def fetch_quarterly_facts(ticker: str, cik: str) -> pd.DataFrame:
    """Return every quarterly-duration diluted-EPS fact for one CIK, undeduped."""
    response = sec_get(SEC_CONCEPT_URL.format(cik=cik))

    if response.status_code == 403:
        fail("EDGAR returned 403 Forbidden -- the User-Agent header was rejected.")
    if response.status_code == 404:
        fail(f"EDGAR returned 404 for {ticker} (CIK {cik}); concept not reported.")
    if response.status_code != 200:
        fail(f"EDGAR request for {ticker} failed with HTTP {response.status_code}.")

    units = response.json().get("units", {})
    if not units:
        fail(f"EDGAR response for {ticker} (CIK {cik}) contained no 'units' block.")

    unit_key = "USD/shares" if "USD/shares" in units else sorted(units)[0]
    raw_records = units[unit_key]
    if not raw_records:
        fail(f"EDGAR returned an empty record list for {ticker} under '{unit_key}'.")

    facts = pd.DataFrame(raw_records)
    for required_field in ("start", "end", "filed", "val"):
        if required_field not in facts.columns:
            fail(f"{ticker}: EDGAR records lack the '{required_field}' field entirely.")

    end_dates = pd.to_datetime(facts["end"], errors="coerce")
    start_dates = pd.to_datetime(facts["start"], errors="coerce")

    # Records with no 'start' are instantaneous facts; ~365 day durations are
    # annual EPS. Only ~quarter-length durations are quarterly EPS.
    duration_days = (end_dates - start_dates).dt.days
    is_quarterly = duration_days.between(QUARTER_MIN_DAYS, QUARTER_MAX_DAYS)

    quarterly = facts.loc[is_quarterly].copy()
    if quarterly.empty:
        fail(f"{ticker}: no quarterly-duration EPS facts found.")

    quarterly["ticker"] = ticker
    quarterly["cik"] = cik
    quarterly["period_end"] = end_dates.loc[is_quarterly]
    quarterly["filed_date"] = pd.to_datetime(quarterly["filed"], errors="coerce")
    quarterly["eps_diluted"] = quarterly["val"]
    quarterly["filing_lag_days"] = (quarterly["filed_date"] - quarterly["period_end"]).dt.days

    return quarterly


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
# Stage 4 -- year-over-year label
# --------------------------------------------------------------------------


def attach_yoy_label(panel: pd.DataFrame) -> pd.DataFrame:
    """Label 1 when EPS beat the same quarter a year earlier, else 0.

    The year-ago row is located by DATE PROXIMITY, never by row offset: a missing
    quarter would make a positional shift compare against the wrong period.
    """
    # The date each row is looking for: roughly one year before its period end.
    panel = panel.copy()
    panel["year_ago_target"] = panel["period_end"] - pd.Timedelta(days=YEAR_AGO_TARGET_DAYS)

    # Right-hand side is the same panel, offering its period_end and EPS as the
    # candidate year-ago observation.
    candidates = panel[["ticker", "period_end", "eps_diluted"]].copy()
    candidates = candidates.rename(
        columns={"period_end": "period_end_year_ago", "eps_diluted": "eps_year_ago"}
    )

    # merge_asof needs both sides sorted on the join key.
    left_sorted = panel.sort_values("year_ago_target")
    right_sorted = candidates.sort_values("period_end_year_ago")

    # direction="nearest" with a tolerance implements "~365 days prior, +/- 45".
    # by="ticker" keeps each company's history separate. A row whose nearest
    # candidate falls outside the tolerance gets NaN, which becomes a null label.
    matched = pd.merge_asof(
        left_sorted,
        right_sorted,
        left_on="year_ago_target",
        right_on="period_end_year_ago",
        by="ticker",
        direction="nearest",
        tolerance=pd.Timedelta(days=YEAR_AGO_TOLERANCE_DAYS),
    )

    # Strictly greater: a flat quarter is not a beat.
    beat_year_ago = matched["eps_diluted"] > matched["eps_year_ago"]

    # Where no year-ago row matched, the comparison is meaningless -> null, not 0.
    has_match = matched["eps_year_ago"].notna()
    matched["label_yoy"] = beat_year_ago.astype("Int64").where(has_match)

    return matched


# --------------------------------------------------------------------------
# Stage 5 -- stock-split contamination check
# --------------------------------------------------------------------------


def detect_split_events(all_facts: pd.DataFrame) -> list:
    """Find stock splits using EDGAR's own restatement evidence.

    When a company splits its stock, later filings repeat earlier quarters with
    the EPS divided by the split factor. So a period whose original value and a
    later restated value differ by almost exactly an integer factor is a split.
    This needs no external split calendar and no hardcoded dates.

    Returns (ticker, factor, last_pre_split_period_end) tuples, where the last
    element is the newest period still ORIGINALLY reported in pre-split units.
    """
    events = []

    # Near-zero EPS makes the ratio meaningless, so exclude it before dividing.
    usable = all_facts[all_facts["eps_diluted"].abs() > 0.01]

    for (ticker, period_end), group in usable.groupby(["ticker", "period_end"]):
        ordered = group.sort_values("filed_date")
        original_value = ordered.iloc[0]["eps_diluted"]

        for _, later in ordered.iloc[1:].iterrows():
            if abs(later["eps_diluted"] - original_value) <= 1e-9:
                continue  # same number repeated, not a restatement

            ratio = original_value / later["eps_diluted"]
            nearest_factor = round(ratio)

            # Only an (almost exactly) integer shrink counts as a split.
            if nearest_factor < SPLIT_MIN_FACTOR:
                continue
            if abs(ratio - nearest_factor) / nearest_factor > SPLIT_RATIO_TOLERANCE:
                continue

            events.append((ticker, int(nearest_factor), period_end))

    # Collapse to one boundary per (ticker, factor): the LATEST period still
    # originally reported pre-split marks where the unit change takes effect.
    boundaries = {}
    for ticker, factor, period_end in events:
        key = (ticker, factor)
        if key not in boundaries or period_end > boundaries[key]:
            boundaries[key] = period_end

    return [(ticker, factor, boundary) for (ticker, factor), boundary in boundaries.items()]


def flag_split_contamination(panel: pd.DataFrame, split_events: list) -> pd.DataFrame:
    """Mark rows whose year-ago EPS is quoted in pre-split shares.

    The panel keeps MIN(filed), which is point-in-time correct but means a row
    straddling a split compares post-split EPS against a PRE-split year-ago value
    -- different units. The label is left exactly as specified; these rows are
    flagged, logged, and warned about so the distortion is visible.
    """
    panel = panel.copy()
    panel["split_contaminated"] = False

    contaminated_frames = []

    for ticker, factor, boundary in split_events:
        # Contaminated: this period is post-split, its year-ago period is not.
        is_ticker = panel["ticker"] == ticker
        after_split = panel["period_end"] > boundary
        year_ago_before_split = panel["period_end_year_ago"] <= boundary

        affected = is_ticker & after_split & year_ago_before_split
        panel.loc[affected, "split_contaminated"] = True

        if not affected.any():
            continue

        detail = panel.loc[affected].copy()
        detail["split_factor"] = factor
        detail["split_boundary_period"] = boundary
        # What the label WOULD be if the year-ago value were put in current units.
        detail["eps_year_ago_adjusted"] = detail["eps_year_ago"] / factor
        detail["label_if_split_adjusted"] = (
            detail["eps_diluted"] > detail["eps_year_ago_adjusted"]
        ).astype(int)
        contaminated_frames.append(detail)

    if not contaminated_frames:
        print("no split-contaminated periods detected.")
        return panel

    contaminated = pd.concat(contaminated_frames, ignore_index=True)
    would_flip = contaminated["label_if_split_adjusted"] != contaminated["label_yoy"].astype(int)

    log_columns = [
        "ticker", "period_end", "eps_diluted", "period_end_year_ago", "eps_year_ago",
        "split_factor", "eps_year_ago_adjusted", "label_yoy", "label_if_split_adjusted",
    ]
    log = contaminated[log_columns].sort_values(["ticker", "period_end"])
    log.to_csv(SPLIT_CSV, index=False)

    print(f"\n!!! WARNING: {len(contaminated)} rows compare against a PRE-SPLIT year-ago EPS.")
    print(f"!!! Splits detected from EDGAR restatements: "
          f"{', '.join(f'{t} {f}:1' for t, f, _ in sorted(split_events))}")
    print(f"!!! {int(would_flip.sum())} of those labels are WRONG as a result "
          f"(the year-ago value is in different share units).")
    print(f"!!! label_yoy is left as specified; rows are flagged in the "
          f"'split_contaminated' column.")
    print(f"!!! Full detail -> {SPLIT_CSV}")

    return panel


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
    print(f"rows with NULL label : {null_count}  (no year-ago match within tolerance)")

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
    # If split contamination is present, report what the balance would be without
    # it -- the raw rate can sit inside the band only because of the distortion.
    if "split_contaminated" in panel.columns and panel["split_contaminated"].any():
        contaminated_count = int(panel["split_contaminated"].sum())
        print(
            f"\nnote: {contaminated_count} rows are split-contaminated (see SPLIT CHECK "
            f"above);\n      the rate above includes their distorted labels."
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
        if ticker.upper() not in cik_map:
            fail(f"Ticker {ticker} not found in the SEC ticker->CIK map.")

        # An override wins over the map; see CIK_OVERRIDES for why this exists.
        cik = CIK_OVERRIDES.get(ticker.upper(), cik_map[ticker.upper()])
        note = "  (CIK override)" if ticker.upper() in CIK_OVERRIDES else ""

        facts = fetch_quarterly_facts(ticker, cik)
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
    section("LABEL")
    labelled = attach_yoy_label(after_lag)

    in_window = labelled["period_end"] >= STUDY_START
    panel = labelled.loc[in_window].copy()
    print(f"study window {STUDY_START.date()} onward : {len(panel)} rows "
          f"({len(labelled) - len(panel)} warm-up rows used for lookup only)")

    # ---- assemble ---------------------------------------------------------
    panel["prediction_date"] = panel["filed_date"] + pd.Timedelta(days=1)

    output_columns = [
        "ticker",
        "cik",
        "period_end",
        "filed_date",
        "prediction_date",
        "eps_diluted",
        "n_filings_seen",
        "label_yoy",
        # Label provenance, kept so the +/-45 day match can be audited. These are
        # NOT features -- they describe how label_yoy was derived.
        "period_end_year_ago",
        "eps_year_ago",
    ]
    panel = panel[output_columns].sort_values(["ticker", "period_end"])
    panel = panel.reset_index(drop=True)

    # ---- split contamination ---------------------------------------------
    # Runs on the assembled panel because it needs period_end_year_ago, but takes
    # the RAW facts since split evidence lives in the restatements we deduped away.
    section("SPLIT CHECK")
    split_events = detect_split_events(all_facts)
    panel = flag_split_contamination(panel, split_events)

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
