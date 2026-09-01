#!/usr/bin/env python3
"""EDGAR-only audit of quarterly diluted EPS across ten tickers.

Extends scripts/explore_one_ticker.py: the yfinance portion is dropped entirely,
and the SEC EDGAR XBRL portion runs across a basket instead of one symbol.

This INSPECTS ONLY. It builds no features and no labels. Its job is to establish
what the point-in-time EPS record actually looks like before anything is modelled
on it -- in particular whether MIN(filed) per period is a trustworthy "first
public moment", since filed date + 1 day is the intended prediction date.

Run:  python scripts/explore_ten_tickers.py
"""

from __future__ import annotations

import sys
import time
from collections import defaultdict

import pandas as pd
import requests

# --------------------------------------------------------------------------
# Configuration -- edit these, nothing below.
# --------------------------------------------------------------------------

TICKERS = ["AAPL", "MSFT", "JNJ", "JPM", "XOM", "PG", "WMT", "CAT", "NEE", "T"]

# SEC's ticker->CIK map points at the CURRENT registrant for a ticker. When a
# company reorganizes, the ticker is repointed to the new holding-company CIK,
# which carries only post-reorganization filings -- the decades of history sit
# under the old CIK and are silently invisible. Verified example: as of this
# writing the map sends XOM to CIK 2115436 ("ExxonMobil Holdings Corp", 4 EPS
# records from 2025-06-30), while CIK 34088 ("Exxon Mobil Corporation") holds
# 224 records back to 2007-12-31. Override such tickers here.
CIK_OVERRIDES = {
    "XOM": "0000034088",  # historical Exxon Mobil Corporation filer
}

# SEC EDGAR blocks requests that do not identify a real contact.
# REPLACE THIS with your own address or the script will refuse to call EDGAR.
SEC_CONTACT_EMAIL = "REPLACE_ME@example.com"
SEC_APP_NAME = "earnings-surprise-research"

# EDGAR's fair-access limit is 10 requests/second. One request per 0.5s is 2/s,
# comfortably under, and this script makes only ~11 calls in total.
SEC_SLEEP_SECONDS = 0.5

# A quarterly XBRL fact covers a ~3 month duration. Fiscal quarters are ragged
# (13 weeks, 4-4-5 calendars, 52/53-week years), so accept a generous window.
QUARTER_MIN_DAYS = 60
QUARTER_MAX_DAYS = 110

# Two lag populations are expected: the original filing (~30-45 days after period
# end) and the same quarter restated as a comparative in a later filing (~1 year).
# These bounds only classify for reporting; nothing is filtered on them.
ORIGINAL_LAG_MAX_DAYS = 90

# Consecutive sorted lag values further apart than this start a new cluster.
LAG_CLUSTER_GAP_DAYS = 20

# Two EPS values are treated as different beyond this; EPS is quoted to cents.
EPS_EQUALITY_TOLERANCE = 1e-9

# XBRL tagging was phased in 2009-2011. Before it settled, a quarter's earliest
# XBRL record is frequently a later comparative rather than its original filing --
# the number was public on time, just not tagged. Used only to split the Q2 report.
XBRL_ADOPTION_SETTLED_YEAR = 2011

# A ticker whose period count falls below this fraction of the basket median is
# flagged as suspiciously thin -- usually a wrong-CIK resolution, not a real gap.
THIN_COVERAGE_FRACTION = 0.5

OUTPUT_CSV = "data/edgar_audit.csv"
RECORDS_CSV = "data/edgar_records.csv"

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
    """Abort loudly. Every empty/missing source funnels through here."""
    print(f"\n*** FATAL: {message}", file=sys.stderr)
    sys.exit(1)


def section(title: str) -> None:
    """Visual separator so the report stays scannable."""
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
        # EDGAR's documented format: application name followed by a contact address.
        "User-Agent": f"{SEC_APP_NAME} ({SEC_CONTACT_EMAIL})",
        # EDGAR serves gzip; asking for it explicitly avoids oversized transfers.
        "Accept-Encoding": "gzip, deflate",
        "Accept": "application/json",
    }


def sec_get(url: str) -> requests.Response:
    """Single throttled EDGAR GET. Every outbound call goes through here so the
    rate limit cannot be bypassed by adding a call site later."""
    time.sleep(SEC_SLEEP_SECONDS)  # sleep BEFORE, so bursts cannot slip through
    return requests.get(url, headers=sec_headers(), timeout=SEC_TIMEOUT_SECONDS)


def cluster_values(values: list, gap_threshold: int) -> list:
    """Group sorted numbers into runs separated by gaps larger than the threshold.

    Used to collapse hundreds of filing lags into the handful of distinct clusters
    that actually exist, rather than printing every value.
    """
    if not values:
        return []

    ordered = sorted(values)
    clusters = [[ordered[0]]]

    for value in ordered[1:]:
        # A jump bigger than the threshold means a genuinely separate population.
        if value - clusters[-1][-1] > gap_threshold:
            clusters.append([value])
        else:
            clusters[-1].append(value)

    return clusters


# --------------------------------------------------------------------------
# EDGAR fetch
# --------------------------------------------------------------------------


def load_ticker_cik_map() -> dict:
    """Fetch SEC's ticker->CIK map once and index it by upper-case ticker."""
    response = sec_get(SEC_TICKER_MAP_URL)
    if response.status_code != 200:
        fail(
            f"SEC ticker map request failed with HTTP {response.status_code}. "
            f"A 403 here almost always means the User-Agent was rejected."
        )

    # The payload is a dict keyed by row number: {"0": {cik_str, ticker, title}, ...}
    raw_map = response.json()
    if not raw_map:
        fail("SEC ticker map came back empty.")

    indexed = {}
    for entry in raw_map.values():
        symbol = entry.get("ticker", "").upper()
        # CIKs arrive as ints; the API path needs them padded to 10 digits.
        indexed[symbol] = (str(entry["cik_str"]).zfill(10), entry.get("title", "?"))

    return indexed


def fetch_quarterly_eps(ticker: str, cik: str) -> pd.DataFrame:
    """Return the quarterly-duration diluted-EPS facts for one CIK."""
    response = sec_get(SEC_CONCEPT_URL.format(cik=cik))

    if response.status_code == 403:
        fail("EDGAR returned 403 Forbidden -- the User-Agent header was rejected.")
    if response.status_code == 404:
        fail(
            f"EDGAR returned 404 for {ticker} (CIK {cik}). This company does not "
            f"report us-gaap:EarningsPerShareDiluted under that CIK."
        )
    if response.status_code != 200:
        fail(f"EDGAR request for {ticker} failed with HTTP {response.status_code}.")

    payload = response.json()

    # Facts are grouped by unit of measure; EPS is reported in USD per share.
    units = payload.get("units", {})
    if not units:
        fail(f"EDGAR response for {ticker} (CIK {cik}) contained no 'units' block.")

    unit_key = "USD/shares" if "USD/shares" in units else sorted(units)[0]
    if unit_key != "USD/shares":
        print(f"  NOTE: expected unit 'USD/shares', using '{unit_key}'.")

    raw_records = units[unit_key]
    if not raw_records:
        fail(f"EDGAR returned an empty record list for {ticker} under '{unit_key}'.")

    facts = pd.DataFrame(raw_records)
    facts.attrs["raw_record_count"] = len(facts)

    # 'end' is the period end; 'filed' is when the number became public. The gap
    # between them is the entire reason this source is used, so both must exist.
    for required_field in ("start", "end", "filed", "val"):
        if required_field not in facts.columns:
            fail(f"{ticker}: EDGAR records lack the '{required_field}' field entirely.")

    end_dates = pd.to_datetime(facts["end"], errors="coerce")
    filed_dates = pd.to_datetime(facts["filed"], errors="coerce")
    start_dates = pd.to_datetime(facts["start"], errors="coerce")

    # Records with no 'start' are instantaneous facts; ~365 day durations are annual
    # EPS. Only ~quarter-length durations are quarterly EPS.
    duration_days = (end_dates - start_dates).dt.days
    is_quarterly = duration_days.between(QUARTER_MIN_DAYS, QUARTER_MAX_DAYS)

    quarterly = facts.loc[is_quarterly].copy()
    if quarterly.empty:
        fail(
            f"{ticker}: no records had a {QUARTER_MIN_DAYS}-{QUARTER_MAX_DAYS} day "
            f"duration, so no quarterly EPS facts were found."
        )

    quarterly["ticker"] = ticker
    quarterly["period_end"] = end_dates.loc[is_quarterly]
    quarterly["filed_date"] = filed_dates.loc[is_quarterly]
    quarterly["filing_lag_days"] = (quarterly["filed_date"] - quarterly["period_end"]).dt.days

    # Missing-date counts are computed on the QUARTERLY subset, which is what the
    # rest of the audit consumes; the raw count is carried alongside for reporting.
    quarterly.attrs["raw_record_count"] = len(facts)
    quarterly.attrs["missing_end"] = int(quarterly["period_end"].isna().sum())
    quarterly.attrs["missing_filed"] = int(quarterly["filed_date"].isna().sum())

    return quarterly


# --------------------------------------------------------------------------
# Per-ticker report
# --------------------------------------------------------------------------


def audit_one_ticker(ticker: str, company: str, cik: str, quarterly: pd.DataFrame) -> dict:
    """Print the per-ticker block and return its summary row."""
    section(f"{ticker} -- {company}  (CIK {cik})")

    raw_count = quarterly.attrs["raw_record_count"]
    quarterly_count = len(quarterly)
    unique_periods = quarterly["period_end"].nunique()

    print(f"raw records (all durations) : {raw_count}")
    print(f"quarterly records           : {quarterly_count}")
    print(f"unique periods (dedupe 'end'): {unique_periods}")
    print(f"records per period (mean)   : {quarterly_count / unique_periods:.2f}")

    # --- both dates present ------------------------------------------------
    missing_end = quarterly.attrs["missing_end"]
    missing_filed = quarterly.attrs["missing_filed"]
    both_dates_present = (missing_end == 0) and (missing_filed == 0)
    print(
        f"every record has end+filed  : {'YES' if both_dates_present else 'NO'}"
        f"  (missing end={missing_end}, missing filed={missing_filed})"
    )

    print(f"earliest period end         : {quarterly['period_end'].min().date()}")
    print(f"latest period end           : {quarterly['period_end'].max().date()}")

    # --- filing-lag clusters ------------------------------------------------
    lag_values = quarterly["filing_lag_days"].dropna().tolist()
    lag_clusters = cluster_values(lag_values, LAG_CLUSTER_GAP_DAYS)

    print(f"\nfiling-lag clusters ({len(lag_clusters)} distinct):")
    cluster_labels = []
    for cluster in lag_clusters:
        low = min(cluster)
        high = max(cluster)
        # Classify by the lower bound: a cluster starting inside the original band
        # is first-filing behaviour, anything later is a comparative/restatement.
        kind = "original" if low <= ORIGINAL_LAG_MAX_DAYS else "restated/comparative"
        print(f"    {low:5d}-{high:5d} days  n={len(cluster):4d}  [{kind}]")
        cluster_labels.append(f"{low}-{high}(n={len(cluster)})")

    return {
        "ticker": ticker,
        "company": company,
        "cik": cik,
        "raw_records": raw_count,
        "quarterly_records": quarterly_count,
        "unique_periods": unique_periods,
        "earliest_period_end": quarterly["period_end"].min().date(),
        "latest_period_end": quarterly["period_end"].max().date(),
        "all_have_end_and_filed": both_dates_present,
        "n_lag_clusters": len(lag_clusters),
        "lag_clusters": " | ".join(cluster_labels),
    }


# --------------------------------------------------------------------------
# Anomaly detection
# --------------------------------------------------------------------------


def find_anomalies(all_records: pd.DataFrame) -> dict:
    """Scan every (ticker, period) group for the four flagged conditions."""
    single_filing = []
    three_plus = []
    value_mismatches = []
    late_first_filing = []

    grouped = all_records.groupby(["ticker", "period_end"], sort=True)

    for (ticker, period_end), group in grouped:
        filing_count = len(group)

        if filing_count == 1:
            single_filing.append((ticker, period_end, group.iloc[0]["filing_lag_days"]))
        if filing_count >= 3:
            three_plus.append((ticker, period_end, filing_count))

        # Order by filed date so row 0 is the earliest public appearance.
        ordered = group.sort_values("filed_date")
        first_row = ordered.iloc[0]

        # Q2 check: is the EARLIEST-filed record actually a timely original filing?
        # If its lag is large, this period never had a prompt first filing, and
        # MIN(filed) would not mean "the moment the market learned the number".
        if first_row["filing_lag_days"] > ORIGINAL_LAG_MAX_DAYS:
            late_first_filing.append((ticker, period_end, first_row["filing_lag_days"]))

        # Most important check: did a later filing report a DIFFERENT number for
        # the same period? If so, the "original" value and the "restated" value
        # disagree and only the original was knowable at prediction time.
        original_value = first_row["val"]
        differing = ordered[(ordered["val"] - original_value).abs() > EPS_EQUALITY_TOLERANCE]
        if not differing.empty:
            value_mismatches.append((ticker, period_end, ordered))

    return {
        "single_filing": single_filing,
        "three_plus": three_plus,
        "value_mismatches": value_mismatches,
        "late_first_filing": late_first_filing,
    }


def find_thin_coverage(summary_rows: list) -> list:
    """Flag tickers whose history is far shorter than the basket's typical depth.

    A ticker resolving to the wrong CIK looks exactly like a ticker with genuinely
    short history, and it silently destroys the common study window (Q3). Comparing
    against the basket median catches it without hardcoding a date.
    """
    period_counts = [row["unique_periods"] for row in summary_rows]
    median_periods = sorted(period_counts)[len(period_counts) // 2]
    threshold = median_periods * THIN_COVERAGE_FRACTION

    thin = []
    for row in summary_rows:
        if row["unique_periods"] < threshold:
            thin.append((row["ticker"], row["unique_periods"], median_periods))

    return thin


def print_anomalies(anomalies: dict) -> None:
    section("ANOMALIES")

    # --- periods with only one filing --------------------------------------
    single = anomalies["single_filing"]
    print(f"[A] Periods with only ONE filing: {len(single)}")
    for ticker, period_end, lag in single[:20]:
        print(f"      {ticker:5s} {period_end.date()}  lag={int(lag)}d")
    if len(single) > 20:
        print(f"      ... and {len(single) - 20} more")

    # --- periods with three or more ----------------------------------------
    many = anomalies["three_plus"]
    print(f"\n[B] Periods with THREE OR MORE filings: {len(many)}")
    for ticker, period_end, count in many[:20]:
        print(f"      {ticker:5s} {period_end.date()}  {count} filings")
    if len(many) > 20:
        print(f"      ... and {len(many) - 20} more")

    # --- earliest filing that is not timely --------------------------------
    late = anomalies["late_first_filing"]
    print(
        f"\n[C] Periods whose EARLIEST filing is later than "
        f"{ORIGINAL_LAG_MAX_DAYS}d after period end: {len(late)}"
    )
    for ticker, period_end, lag in late[:20]:
        print(f"      {ticker:5s} {period_end.date()}  first filed {int(lag)}d after end")
    if len(late) > 20:
        print(f"      ... and {len(late) - 20} more")

    # --- restatement changed the number ------------------------------------
    mismatches = anomalies["value_mismatches"]
    print(f"\n[D] Periods where a later filing reported a DIFFERENT EPS: {len(mismatches)}")
    print("    (printed in full -- these are the cases that break naive dedupe)")

    for ticker, period_end, ordered in mismatches:
        print(f"\n    --- {ticker}  period_end={period_end.date()} ---")
        detail = pd.DataFrame(
            {
                "filed": ordered["filed_date"].dt.date,
                "lag_days": ordered["filing_lag_days"],
                "eps": ordered["val"],
                "fy": ordered["fy"],
                "fp": ordered["fp"],
                "form": ordered["form"],
                "accession": ordered["accn"],
            }
        )
        print(detail.to_string(index=False))

    # --- thin coverage (printed last so it does not split block [D]) --------
    thin = anomalies["thin_coverage"]
    print(f"\n[E] Tickers with suspiciously THIN coverage: {len(thin)}")
    if thin:
        print("    (a wrong-CIK resolution looks identical to genuinely short history)")
    for ticker, count, median_count in thin:
        print(f"      {ticker:5s} {count} periods vs basket median {median_count}")


# --------------------------------------------------------------------------
# Cross-ticker summary
# --------------------------------------------------------------------------


def print_cross_ticker_summary(
    all_records: pd.DataFrame, summary_rows: list, anomalies: dict
) -> None:
    section("CROSS-TICKER SUMMARY")

    summary = pd.DataFrame(summary_rows)

    # ---- Q1: does the two-cluster pattern hold for all ten? ---------------
    print("Q1. Does the two-filings-per-period pattern (original + restated) hold")
    print("    for ALL 10 tickers, or do some differ?\n")

    for row in summary_rows:
        # Two clusters is the expected shape; anything else is worth naming.
        shape = "as expected" if row["n_lag_clusters"] == 2 else "DIFFERS"
        ratio = row["quarterly_records"] / row["unique_periods"]
        print(
            f"    {row['ticker']:5s} clusters={row['n_lag_clusters']}  "
            f"records/period={ratio:.2f}  [{shape}]   {row['lag_clusters']}"
        )

    conforming = sum(1 for row in summary_rows if row["n_lag_clusters"] == 2)
    print(f"\n    -> {conforming} of {len(summary_rows)} tickers show exactly two lag clusters.")

    # ---- Q2: is MIN(filed) always the original filing? --------------------
    print("\nQ2. Is MIN(filed) per 'end' always the ORIGINAL filing?\n")

    late = anomalies["late_first_filing"]
    if not late:
        print(
            f"    YES for all 10. Every (ticker, period) group's earliest filing "
            f"lands within {ORIGINAL_LAG_MAX_DAYS} days of period end, so MIN(filed)"
        )
        print("    always picks a timely original rather than a later comparative.")
    else:
        offenders = sorted({ticker for ticker, _, _ in late})
        print(f"    NO -- {len(late)} periods across {len(offenders)} ticker(s) break it:")
        print(f"    affected tickers: {', '.join(offenders)}")

        # Whether these cluster in the XBRL-adoption era or persist decides if this
        # is merely a start-date problem or an ongoing hazard for the whole sample.
        late_years = [period_end.year for _, period_end, _ in late]
        year_counts = defaultdict(int)
        for year in late_years:
            year_counts[year] += 1

        breakdown = ", ".join(f"{y}:{year_counts[y]}" for y in sorted(year_counts))
        print(f"    by period_end year: {breakdown}")

        recent = [year for year in late_years if year >= XBRL_ADOPTION_SETTLED_YEAR]
        print(
            f"    {len(late) - len(recent)} fall before {XBRL_ADOPTION_SETTLED_YEAR} "
            f"(XBRL phase-in: a pre-2011 quarter's FIRST XBRL record is often a later"
        )
        print("    comparative, even though the number was public on time in HTML).")
        print(
            f"    But {len(recent)} occur from {XBRL_ADOPTION_SETTLED_YEAR} onward, so this "
            f"is a live hazard, not just a start-date artifact."
        )
        print("    See anomaly block [C] above for the full list.")

    # A separate, sharper caveat: MIN(filed) picks the right ROW, but if the value
    # was later revised, that row's VALUE is not what later data would show.
    mismatch_count = len(anomalies["value_mismatches"])
    print(
        f"\n    Caveat: {mismatch_count} periods had a later filing report a different"
    )
    print("    EPS value. MIN(filed) still gives the point-in-time-correct row, but")
    print("    it will NOT match a restated/split-adjusted figure. See block [D].")

    # ---- Q3: common study start ------------------------------------------
    print("\nQ3. Earliest period end that ALL 10 tickers have data for?\n")

    # The common start is the LATEST of the per-ticker earliest periods: every
    # ticker has data at or after this date, none has data covering all before it.
    earliest_per_ticker = all_records.groupby("ticker")["period_end"].min()
    common_start = earliest_per_ticker.max()
    binding_ticker = earliest_per_ticker.idxmax()

    ordered_starts = earliest_per_ticker.sort_values()
    for ticker, first_end in ordered_starts.items():
        marker = "  <-- binding constraint" if ticker == binding_ticker else ""
        print(f"    {ticker:5s} earliest period end {first_end.date()}{marker}")

    print(f"\n    -> Common coverage across ALL {len(earliest_per_ticker)}: period end "
          f"{common_start.date()} ({binding_ticker}), study start year {common_start.year}")

    # A single thin-coverage ticker sets the common start for the whole basket, so
    # report the answer both ways rather than letting one artifact decide it.
    thin_tickers = [ticker for ticker, _, _ in anomalies["thin_coverage"]]
    if thin_tickers:
        healthy = earliest_per_ticker.drop(labels=thin_tickers)
        healthy_start = healthy.max()
        print(
            f"\n    NOTE: {', '.join(thin_tickers)} flagged as thin coverage (block [E]) "
            f"and is\n    single-handedly setting that date."
        )
        print(
            f"    -> Excluding it: common coverage starts {healthy_start.date()} "
            f"({healthy.idxmax()}), study start year {healthy_start.year}"
        )
        print("    -> Resolve the thin ticker's CIK before trusting either number.")


# --------------------------------------------------------------------------
# Entry point
# --------------------------------------------------------------------------


def main() -> None:
    print(f"EDGAR audit of {len(TICKERS)} tickers -- inspection only, nothing is merged.")
    print(f"Throttle: one request per {SEC_SLEEP_SECONDS}s (EDGAR allows 10/s).")

    cik_map = load_ticker_cik_map()

    summary_rows = []
    per_ticker_frames = []

    for ticker in TICKERS:
        if ticker.upper() not in cik_map:
            fail(f"Ticker {ticker} not found in the SEC ticker->CIK map.")

        cik, company = cik_map[ticker.upper()]

        # An override wins over the map; see CIK_OVERRIDES for why this exists.
        if ticker.upper() in CIK_OVERRIDES:
            override_cik = CIK_OVERRIDES[ticker.upper()]
            print(f"\nNOTE: {ticker} CIK overridden {cik} -> {override_cik} (see CIK_OVERRIDES).")
            cik = override_cik

        quarterly = fetch_quarterly_eps(ticker, cik)

        summary_rows.append(audit_one_ticker(ticker, company, cik, quarterly))
        per_ticker_frames.append(quarterly)

    all_records = pd.concat(per_ticker_frames, ignore_index=True)

    anomalies = find_anomalies(all_records)
    anomalies["thin_coverage"] = find_thin_coverage(summary_rows)
    print_anomalies(anomalies)
    print_cross_ticker_summary(all_records, summary_rows, anomalies)

    # ---- persist ----------------------------------------------------------
    section("OUTPUT")

    # data/ is gitignored, so these stay local.
    import os

    os.makedirs("data", exist_ok=True)

    summary = pd.DataFrame(summary_rows)
    summary.to_csv(OUTPUT_CSV, index=False)
    print(f"per-ticker audit  -> {OUTPUT_CSV}  ({len(summary)} rows)")

    # Record-level detail, so the anomaly cases can be inspected directly.
    detail_columns = [
        "ticker", "period_end", "filed_date", "filing_lag_days",
        "val", "fy", "fp", "form", "accn",
    ]
    detail = all_records[detail_columns].sort_values(["ticker", "period_end", "filed_date"])
    detail.to_csv(RECORDS_CSV, index=False)
    print(f"record-level detail -> {RECORDS_CSV}  ({len(detail)} rows)")


if __name__ == "__main__":
    main()
