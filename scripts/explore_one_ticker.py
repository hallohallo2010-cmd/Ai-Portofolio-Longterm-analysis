#!/usr/bin/env python3
"""Exploratory data pull for a SINGLE ticker.

Purpose: look at what the raw sources actually return before any modelling
decisions get made. This script INSPECTS ONLY -- it builds no features, no
labels, and merges nothing. Every source is reported on independently.

Sources touched:
  1. yfinance daily OHLCV
  2. yfinance earnings dates
  3. SEC EDGAR XBRL company concept (EarningsPerShareDiluted)

Run:  python scripts/explore_one_ticker.py
"""

from __future__ import annotations

import sys
from datetime import date

import numpy as np
import pandas as pd
import requests
import yfinance as yf

# --------------------------------------------------------------------------
# Configuration -- edit these, nothing below.
# --------------------------------------------------------------------------

TICKER = "AAPL"
START_DATE = "2005-01-01"
END_DATE = date.today().isoformat()  # yfinance end is exclusive; today is fine

# SEC EDGAR blocks requests that do not identify a real contact. Their fair-access
# policy asks for "Sample Company Name AdminContact@example.com" style values.
# REPLACE THIS with your own address or the script will refuse to call EDGAR.
SEC_CONTACT_EMAIL = "REPLACE_ME@example.com"
SEC_APP_NAME = "earnings-surprise-research"

# A "gap" is more than this many business days between two consecutive sessions.
# Normal weekend = 1 business-day step, so 5 comfortably clears holiday weeks.
MAX_GAP_TRADING_DAYS = 5

# yfinance caps how far back earnings dates go; ask for more than we expect.
EARNINGS_DATES_LIMIT = 200

# A quarterly XBRL fact covers a ~3 month duration. Fiscal quarters are ragged
# (13 weeks, 4-4-5 calendars, 52/53-week years), so accept a generous window.
QUARTER_MIN_DAYS = 60
QUARTER_MAX_DAYS = 110

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
    """Visual separator so the four blocks of output stay distinguishable."""
    print("\n" + "=" * 74)
    print(title)
    print("=" * 74)


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


# --------------------------------------------------------------------------
# 1. Daily OHLCV
# --------------------------------------------------------------------------


def pull_price_history() -> pd.DataFrame:
    section(f"[1/4] yfinance daily OHLCV -- {TICKER}  ({START_DATE} -> {END_DATE})")

    # auto_adjust=False keeps raw OHLC alongside a separate 'Adj Close' column,
    # so we can see what the source gives rather than a pre-adjusted view.
    # progress=False suppresses the download bar that would clutter this report.
    prices = yf.download(
        TICKER,
        start=START_DATE,
        end=END_DATE,
        auto_adjust=False,
        progress=False,
    )

    if prices is None or prices.empty:
        fail(
            f"yfinance returned no price rows for {TICKER}. Check the symbol, the "
            f"date range, and network access to Yahoo Finance."
        )

    # Recent yfinance versions return MultiIndex columns (field, ticker) even for a
    # single symbol. Flatten to the field level so printing stays readable.
    if isinstance(prices.columns, pd.MultiIndex):
        prices.columns = prices.columns.get_level_values(0)

    print(f"shape          : {prices.shape}  (rows, columns)")
    print(f"columns        : {list(prices.columns)}")
    print(f"first date     : {prices.index[0].date()}")
    print(f"last date      : {prices.index[-1].date()}")

    # --- gap detection -----------------------------------------------------
    # The index contains only trading days, so consecutive rows are normally one
    # business day apart. Counting business days between them (rather than calendar
    # days) means weekends do not register as gaps. Market holidays still add ~1
    # each, which is why the threshold is 5 rather than 2.
    session_days = prices.index.normalize().values.astype("datetime64[D]")
    business_day_steps = np.busday_count(session_days[:-1], session_days[1:])

    gap_positions = np.flatnonzero(business_day_steps > MAX_GAP_TRADING_DAYS)

    print(
        f"\ngaps > {MAX_GAP_TRADING_DAYS} business days between sessions: "
        f"{len(gap_positions)}"
    )
    for position in gap_positions:
        # position indexes the *step*, so the gap spans rows [position, position+1].
        gap_start = prices.index[position].date()
        gap_end = prices.index[position + 1].date()
        step_size = business_day_steps[position]
        print(f"    {gap_start} -> {gap_end}   ({step_size} business days)")

    return prices


# --------------------------------------------------------------------------
# 2. yfinance earnings dates
# --------------------------------------------------------------------------


def pull_earnings_dates() -> pd.DataFrame:
    section(f"[2/4] yfinance earnings dates -- {TICKER}")

    ticker_handle = yf.Ticker(TICKER)

    # This endpoint is scraped rather than official and breaks periodically, so
    # catch anything and report it as a source failure instead of a traceback.
    try:
        earnings = ticker_handle.get_earnings_dates(limit=EARNINGS_DATES_LIMIT)
    except Exception as error:  # noqa: BLE001 - any failure here is a source failure
        fail(f"yfinance get_earnings_dates() raised {type(error).__name__}: {error}")

    if earnings is None or earnings.empty:
        fail(
            f"yfinance returned no earnings dates for {TICKER}. This endpoint is "
            f"unofficial and is often empty or rate-limited."
        )

    print(f"rows           : {len(earnings)}")
    print(f"columns        : {list(earnings.columns)}")
    print(f"index name     : {earnings.index.name}")

    # Index holds the announcement timestamps; min/max give the covered range.
    # Rows include FUTURE scheduled dates, so the max may be ahead of today.
    print(f"earliest date  : {earnings.index.min()}")
    print(f"latest date    : {earnings.index.max()}")

    # How much of the range is history vs. forward-looking schedule.
    now_in_index_tz = pd.Timestamp.now(tz=earnings.index.tz)
    future_row_count = (earnings.index > now_in_index_tz).sum()
    print(f"future-dated   : {future_row_count} of {len(earnings)} rows")

    print("\nmost recent 5 rows:")
    print(earnings.head(5).to_string())

    return earnings


# --------------------------------------------------------------------------
# 3. SEC EDGAR XBRL company facts
# --------------------------------------------------------------------------


def lookup_cik(ticker: str) -> tuple[str, str]:
    """Resolve a ticker to its zero-padded 10-digit CIK via SEC's public map."""
    response = requests.get(
        SEC_TICKER_MAP_URL, headers=sec_headers(), timeout=SEC_TIMEOUT_SECONDS
    )
    if response.status_code != 200:
        fail(
            f"SEC ticker map request failed with HTTP {response.status_code}. "
            f"A 403 here almost always means the User-Agent was rejected."
        )

    # The payload is a dict keyed by row number: {"0": {cik_str, ticker, title}, ...}
    ticker_map = response.json()

    for entry in ticker_map.values():
        if entry.get("ticker", "").upper() == ticker.upper():
            # CIKs arrive as ints; the API path needs them padded to 10 digits.
            padded_cik = str(entry["cik_str"]).zfill(10)
            return padded_cik, entry.get("title", "<unknown>")

    fail(f"Ticker {ticker} not found in the SEC ticker->CIK map ({len(ticker_map)} entries).")


def pull_sec_eps() -> pd.DataFrame:
    section(f"[3/4] SEC EDGAR XBRL -- EarningsPerShareDiluted for {TICKER}")

    cik, company_name = lookup_cik(TICKER)
    print(f"resolved CIK   : {cik}  ({company_name})")

    url = SEC_CONCEPT_URL.format(cik=cik)
    print(f"endpoint       : {url}")

    response = requests.get(url, headers=sec_headers(), timeout=SEC_TIMEOUT_SECONDS)

    if response.status_code == 403:
        fail("EDGAR returned 403 Forbidden -- the User-Agent header was rejected.")
    if response.status_code == 404:
        fail(
            f"EDGAR returned 404 for CIK {cik}. This company does not report the "
            f"us-gaap:EarningsPerShareDiluted concept under that CIK."
        )
    if response.status_code != 200:
        fail(f"EDGAR request failed with HTTP {response.status_code}.")

    payload = response.json()

    # Facts are grouped by unit of measure; EPS is reported in USD per share.
    units = payload.get("units", {})
    if not units:
        fail(f"EDGAR response for CIK {cik} contained no 'units' block.")

    unit_key = "USD/shares" if "USD/shares" in units else sorted(units)[0]
    if unit_key != "USD/shares":
        print(f"NOTE: expected unit 'USD/shares', using '{unit_key}' instead.")

    raw_records = units[unit_key]
    if not raw_records:
        fail(f"EDGAR returned an empty record list under unit '{unit_key}'.")

    facts = pd.DataFrame(raw_records)
    print(f"\nraw records    : {len(facts)}  (all durations, all filings)")
    print(f"record keys    : {list(facts.columns)}")

    # --- required-field check ---------------------------------------------
    # 'end' is the period end; 'filed' is when the number became public. The gap
    # between them is the whole reason this source is worth using, so both must exist.
    for required_field in ("end", "filed", "val"):
        if required_field not in facts.columns:
            fail(f"EDGAR records are missing the '{required_field}' field entirely.")

    end_dates = pd.to_datetime(facts["end"], errors="coerce")
    filed_dates = pd.to_datetime(facts["filed"], errors="coerce")

    missing_end_count = int(end_dates.isna().sum())
    missing_filed_count = int(filed_dates.isna().sum())

    print(f"missing 'end'  : {missing_end_count}")
    print(f"missing 'filed': {missing_filed_count}")
    if missing_end_count or missing_filed_count:
        fail(
            "Some EDGAR records lack an 'end' or 'filed' date; the point-in-time "
            "guarantee this source is used for does not hold."
        )
    print("OK: every record carries BOTH an 'end' date and a 'filed' date.")

    # --- isolate quarterly durations ---------------------------------------
    # Records with no 'start' are instantaneous facts; duration facts of ~1 year are
    # annual EPS. Only ~quarter-length durations are quarterly EPS.
    if "start" not in facts.columns:
        fail("EDGAR records have no 'start' field, so quarterly facts cannot be identified.")

    start_dates = pd.to_datetime(facts["start"], errors="coerce")
    duration_days = (end_dates - start_dates).dt.days
    is_quarterly = duration_days.between(QUARTER_MIN_DAYS, QUARTER_MAX_DAYS)

    quarterly = facts.loc[is_quarterly].copy()
    if quarterly.empty:
        fail(
            f"No records had a {QUARTER_MIN_DAYS}-{QUARTER_MAX_DAYS} day duration, "
            f"so no quarterly EPS facts were found."
        )

    # Attach parsed dates so downstream printing does not re-parse strings.
    quarterly["period_end"] = end_dates.loc[is_quarterly]
    quarterly["filed_date"] = filed_dates.loc[is_quarterly]

    # fy/fp label the FILING that carried the fact, not the period the fact covers.
    # A 10-Q filed in FY2025 restates the year-ago quarter, so period end 2024-06-29
    # shows up tagged both '2024-Q3' and '2025-Q3'. Do not treat fy/fp as the
    # period's own identity -- 'end' is the only trustworthy period key here.
    fiscal_year = quarterly["fy"].astype("Int64").astype(str)
    quarterly["fiscal_period"] = fiscal_year + "-" + quarterly["fp"].astype(str)

    print(f"\nquarterly EPS records: {len(quarterly)}")

    # The same fiscal quarter is restated across later filings (a 10-Q number
    # reappears in the next year's comparative 10-Q), so records > unique periods.
    unique_period_count = quarterly["period_end"].nunique()
    print(f"unique period ends   : {unique_period_count}  (duplicates = restatements/comparatives)")

    earliest_period = quarterly.loc[quarterly["period_end"].idxmin()]
    latest_period = quarterly.loc[quarterly["period_end"].idxmax()]
    print(
        f"earliest fiscal period: {earliest_period['fiscal_period']} "
        f"(period end {earliest_period['period_end'].date()})"
    )
    print(
        f"latest fiscal period  : {latest_period['fiscal_period']} "
        f"(period end {latest_period['period_end'].date()})"
    )

    return quarterly


# --------------------------------------------------------------------------
# 4. Sample table
# --------------------------------------------------------------------------


def print_eps_sample(quarterly: pd.DataFrame, row_count: int = 12) -> None:
    section(f"[4/4] Sample of quarterly EPS records (most recent {row_count})")

    sample = quarterly.sort_values(["period_end", "filed_date"])
    sample = sample.tail(row_count)

    # Build a narrow view with just the four requested columns, renamed for reading.
    table = pd.DataFrame(
        {
            "fiscal_period": sample["fiscal_period"],
            "period_end": sample["period_end"].dt.date,
            "filed": sample["filed_date"].dt.date,
            "eps_diluted": sample["val"],
        }
    )

    # Filing lag is the practical takeaway: how long after period end the number
    # was actually public. Anything built on this data must respect that lag.
    table["filing_lag_days"] = (sample["filed_date"] - sample["period_end"]).dt.days

    print(table.to_string(index=False))

    # Filing lag splits into two clusters and the distinction matters enormously:
    #   ~30-45 days  = the ORIGINAL filing, the first moment the number was public
    #   ~400 days    = the same quarter repeated as a comparative in a later filing
    # Only the first-filed record is safe as a point-in-time observation. Anything
    # picking the max/last row per period silently imports a year of hindsight.
    first_filed_lag = table["filing_lag_days"].min()
    last_filed_lag = table["filing_lag_days"].max()
    print(
        f"\nfiling lag in this sample: {first_filed_lag} to {last_filed_lag} days. "
        f"Large lags are restatements of an older quarter, not slow filings."
    )


# --------------------------------------------------------------------------
# Entry point
# --------------------------------------------------------------------------


def main() -> None:
    print(f"Exploratory pull for {TICKER} -- inspection only, nothing is merged or saved.")

    pull_price_history()
    pull_earnings_dates()
    quarterly = pull_sec_eps()
    print_eps_sample(quarterly)

    section("SUMMARY")

    # "Usable" here means: quarterly duration, an EPS value, and both dates present.
    usable = quarterly.dropna(subset=["period_end", "filed_date", "val"])
    earliest_usable_year = int(usable["period_end"].min().year)

    print(f"ticker                        : {TICKER}")
    print(f"earliest year with usable EPS : {earliest_usable_year}")
    print(f"filed dates present           : YES (all {len(usable)} usable records)")
    print(
        "\nNotes: EDGAR coverage starts with XBRL adoption (~2009 for large filers), "
        "so EPS history is usually shorter than the price history above. Records "
        "outnumber unique periods because later filings repeat earlier quarters."
    )


if __name__ == "__main__":
    main()
