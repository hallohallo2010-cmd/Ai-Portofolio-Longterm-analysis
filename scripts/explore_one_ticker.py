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

import os
import sys
from datetime import date

import numpy as np
import pandas as pd
import yfinance as yf

# src/ is a sibling of scripts/, so make the repo root importable.
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

# All EDGAR access lives in src/data_loader.py; only the yfinance steps below
# are local to this script.
from src.data_loader import (  # noqa: E402
    QUARTER_MAX_DAYS,
    QUARTER_MIN_DAYS,
    fail,
    fetch_quarterly_eps,
    load_ticker_cik_map,
    resolve_cik,
)

# --------------------------------------------------------------------------
# Configuration -- edit these, nothing below.
# --------------------------------------------------------------------------

TICKER = "AAPL"
START_DATE = "2005-01-01"
END_DATE = date.today().isoformat()  # yfinance end is exclusive; today is fine

# A "gap" is more than this many business days between two consecutive sessions.
# Normal weekend = 1 business-day step, so 5 comfortably clears holiday weeks.
MAX_GAP_TRADING_DAYS = 5

# yfinance caps how far back earnings dates go; ask for more than we expect.
EARNINGS_DATES_LIMIT = 200

# --------------------------------------------------------------------------
# Small helpers
# --------------------------------------------------------------------------


def section(title: str) -> None:
    """Visual separator so the four blocks of output stay distinguishable."""
    print("\n" + "=" * 72)
    print(title)
    print("=" * 72)


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


def pull_sec_eps() -> pd.DataFrame:
    section(f"[3/4] SEC EDGAR XBRL -- EarningsPerShareDiluted for {TICKER}")

    # CIK resolution, throttling, headers and fact shaping all come from the
    # loader; this script only reports on what comes back.
    cik_map = load_ticker_cik_map()
    cik, company_name, was_overridden = resolve_cik(TICKER, cik_map)

    print(f"resolved CIK   : {cik}  ({company_name})")
    if was_overridden:
        print("               (CIK override applied -- see CIK_OVERRIDES in the loader)")

    facts = fetch_quarterly_eps(TICKER, cik)

    print(f"\nraw records    : {facts.attrs['raw_record_count']}  (all durations, all filings)")
    print(f"unit           : {facts.attrs['unit_key']}")
    print(f"record keys    : {list(facts.columns)}")

    # --- required-field check ---------------------------------------------
    # 'end' is the period end; 'filed' is when the number became public. The gap
    # between them is the whole reason this source is worth using, so both must exist.
    for required_field in ("end", "filed", "val"):
        if required_field not in facts.columns:
            fail(f"EDGAR records are missing the '{required_field}' field entirely.")

    missing_end_count = facts.attrs["missing_end"]
    missing_filed_count = facts.attrs["missing_filed"]

    print(f"missing 'end'  : {missing_end_count}")
    print(f"missing 'filed': {missing_filed_count}")
    if missing_end_count or missing_filed_count:
        fail(
            "Some EDGAR records lack an 'end' or 'filed' date; the point-in-time "
            "guarantee this source is used for does not hold."
        )
    print("OK: every record carries BOTH an 'end' date and a 'filed' date.")

    # The loader already restricted to quarterly durations; annual and
    # instantaneous facts never reach this point.
    quarterly = facts.copy()
    quarterly["period_end"] = quarterly["period_end"]
    quarterly["filed_date"] = quarterly["filed_date"]

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
