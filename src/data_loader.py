"""Single source of truth for SEC EDGAR access.

Everything that talks to EDGAR lives here: the companyconcept API call, the
User-Agent header EDGAR requires, the request throttle, the ticker->CIK
resolution (including overrides), and the thin-coverage guard.

Scripts must not build EDGAR URLs, set headers, or sleep for rate limiting on
their own -- importing from this module is what keeps those consistent.

This module fetches and shapes raw facts. It applies no study window, no dedupe,
and no labelling; those are pipeline decisions and belong to the scripts.
"""

from __future__ import annotations

import sys
import time

import pandas as pd
import requests

# --------------------------------------------------------------------------
# Contact identity -- EDGAR rejects requests that do not carry one.
# --------------------------------------------------------------------------

# REPLACE THIS with your own address. Every script that imports this module
# inherits it, and each will refuse to call EDGAR while it is the placeholder.
SEC_CONTACT_EMAIL = "REPLACE_ME@example.com"
SEC_APP_NAME = "earnings-surprise-research"

# --------------------------------------------------------------------------
# CIK resolution
# --------------------------------------------------------------------------

# SEC's ticker->CIK map points at the CURRENT registrant for a ticker. When a
# company reorganizes, the ticker is repointed to the new holding-company CIK,
# which carries only post-reorganization filings -- decades of history sit under
# the old CIK and are silently invisible. Verified: the map sends XOM to CIK
# 2115436 ("ExxonMobil Holdings Corp", 4 EPS records from 2025-06-30), while CIK
# 34088 ("Exxon Mobil Corporation") holds 224 records back to 2007-12-31.
CIK_OVERRIDES = {
    "XOM": "0000034088",  # historical Exxon Mobil Corporation filer
}

# --------------------------------------------------------------------------
# Request behaviour
# --------------------------------------------------------------------------

# EDGAR's fair-access limit is 10 requests/second; one per 0.5s is 2/s.
SEC_SLEEP_SECONDS = 0.5
SEC_TIMEOUT_SECONDS = 30

SEC_TICKER_MAP_URL = "https://www.sec.gov/files/company_tickers.json"
SEC_CONCEPT_URL = (
    "https://data.sec.gov/api/xbrl/companyconcept/"
    "CIK{cik}/us-gaap/{concept}.json"
)
DEFAULT_CONCEPT = "EarningsPerShareDiluted"

# --------------------------------------------------------------------------
# Fact shaping
# --------------------------------------------------------------------------

# A quarterly XBRL fact covers a ~3 month duration. Fiscal quarters are ragged
# (13 weeks, 4-4-5 calendars, 52/53-week years), so accept a generous window.
QUARTER_MIN_DAYS = 60
QUARTER_MAX_DAYS = 110

# A ticker whose period count falls below this fraction of the basket median is
# flagged as suspiciously thin -- usually a wrong-CIK resolution, not a real gap.
THIN_COVERAGE_FRACTION = 0.5


# --------------------------------------------------------------------------
# Shared failure path
# --------------------------------------------------------------------------


def fail(message: str) -> None:
    """Abort loudly with a non-zero exit. All hard failures funnel through here."""
    print(f"\n*** FATAL: {message}", file=sys.stderr)
    sys.exit(1)


# --------------------------------------------------------------------------
# HTTP
# --------------------------------------------------------------------------


def sec_headers() -> dict:
    """Headers EDGAR requires. A missing/blank User-Agent gets a 403."""
    if "REPLACE_ME" in SEC_CONTACT_EMAIL:
        fail(
            "SEC_CONTACT_EMAIL is still the placeholder. EDGAR requires a real "
            "contact address in the User-Agent header; edit the constant at the "
            "top of src/data_loader.py before running."
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


# --------------------------------------------------------------------------
# Ticker -> CIK
# --------------------------------------------------------------------------


def load_ticker_cik_map() -> dict:
    """Fetch SEC's ticker->CIK map once, indexed by upper-case ticker.

    Values are (cik, company_title) so callers that want the registrant name do
    not need a second request.
    """
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
        indexed[symbol] = (str(entry["cik_str"]).zfill(10), entry.get("title", "?"))

    return indexed


def resolve_cik(ticker: str, cik_map: dict) -> tuple:
    """Return (cik, company_title, was_overridden) for one ticker.

    An entry in CIK_OVERRIDES beats the SEC map; see that constant for why.
    """
    symbol = ticker.upper()
    if symbol not in cik_map:
        fail(f"Ticker {ticker} not found in the SEC ticker->CIK map.")

    mapped_cik, company = cik_map[symbol]

    if symbol in CIK_OVERRIDES:
        return CIK_OVERRIDES[symbol], company, True

    return mapped_cik, company, False


# --------------------------------------------------------------------------
# Facts
# --------------------------------------------------------------------------


def fetch_quarterly_eps(ticker: str, cik: str, concept: str = DEFAULT_CONCEPT) -> pd.DataFrame:
    """Return every quarterly-duration EPS fact for one CIK, UNDEDUPED.

    Each row is one XBRL fact as filed. A period appears once per filing that
    reported it, so duplicates on (ticker, period_end) are expected and are the
    caller's to resolve.

    Columns are the raw EDGAR fields (start, end, val, accn, fy, fp, form, filed,
    frame) plus derived: ticker, cik, period_end, filed_date, eps_diluted,
    filing_lag_days. Frame-level counts are carried in .attrs.
    """
    response = sec_get(SEC_CONCEPT_URL.format(cik=cik, concept=concept))

    if response.status_code == 403:
        fail("EDGAR returned 403 Forbidden -- the User-Agent header was rejected.")
    if response.status_code == 404:
        fail(
            f"EDGAR returned 404 for {ticker} (CIK {cik}). This company does not "
            f"report us-gaap:{concept} under that CIK."
        )
    if response.status_code != 200:
        fail(f"EDGAR request for {ticker} failed with HTTP {response.status_code}.")

    units = response.json().get("units", {})
    if not units:
        fail(f"EDGAR response for {ticker} (CIK {cik}) contained no 'units' block.")

    # EPS is reported in USD per share; fall back rather than assume.
    unit_key = "USD/shares" if "USD/shares" in units else sorted(units)[0]

    raw_records = units[unit_key]
    if not raw_records:
        fail(f"EDGAR returned an empty record list for {ticker} under '{unit_key}'.")

    facts = pd.DataFrame(raw_records)

    # 'end' is the period end; 'filed' is when the number became public. The gap
    # between them is the entire reason this source is used, so both must exist.
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
        fail(
            f"{ticker}: no records had a {QUARTER_MIN_DAYS}-{QUARTER_MAX_DAYS} day "
            f"duration, so no quarterly EPS facts were found."
        )

    quarterly["ticker"] = ticker
    quarterly["cik"] = cik
    quarterly["period_end"] = end_dates.loc[is_quarterly]
    quarterly["filed_date"] = pd.to_datetime(quarterly["filed"], errors="coerce")
    quarterly["eps_diluted"] = quarterly["val"]
    quarterly["filing_lag_days"] = (
        quarterly["filed_date"] - quarterly["period_end"]
    ).dt.days

    quarterly.attrs["raw_record_count"] = len(facts)
    quarterly.attrs["unit_key"] = unit_key
    quarterly.attrs["missing_end"] = int(quarterly["period_end"].isna().sum())
    quarterly.attrs["missing_filed"] = int(quarterly["filed_date"].isna().sum())

    return quarterly


# --------------------------------------------------------------------------
# Coverage guard
# --------------------------------------------------------------------------


def find_thin_coverage(period_counts: dict) -> list:
    """Flag tickers whose history is far shorter than the basket's typical depth.

    A ticker resolving to the wrong CIK looks exactly like a ticker with genuinely
    short history, and it silently destroys any common study window. Comparing
    against the basket median catches it without hardcoding a date.

    Takes {ticker: unique_period_count} and returns (ticker, count, median) tuples.
    """
    if not period_counts:
        return []

    counts = sorted(period_counts.values())
    median_periods = counts[len(counts) // 2]
    threshold = median_periods * THIN_COVERAGE_FRACTION

    thin = []
    for ticker, count in period_counts.items():
        if count < threshold:
            thin.append((ticker, count, median_periods))

    return thin
