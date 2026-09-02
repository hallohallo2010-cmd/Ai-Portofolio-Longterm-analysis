#!/usr/bin/env python3
"""S&P 500 ever-constituent census: how much of the index can EDGAR actually serve?

This resolves the HISTORICAL constituent set -- every ticker in the index at any
point in the study window, including those since removed -- to CIKs, then asks
EDGAR only for record COUNTS. It does not build a panel. The question it answers
is whether a universe is worth committing to before any of that work happens.

Why historical membership matters: censusing today's 503 members would measure
only the survivors. Companies drop out of the index because they were acquired,
shrank, or failed, and those are exactly the observations a model must see to
avoid learning that stocks only ever go up.

Run:  python scripts/build_universe.py
"""

from __future__ import annotations

import io
import os
import sys
import time

import pandas as pd
import requests

# src/ is a sibling of scripts/, so make the repo root importable.
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

# All EDGAR access -- headers, throttle, CIK overrides -- lives in the loader.
from src.data_loader import (  # noqa: E402
    SEC_SLEEP_SECONDS,
    fail,
    load_ticker_cik_map,
    try_fetch_quarterly_eps,
    try_resolve_cik,
)

# --------------------------------------------------------------------------
# Study window
# --------------------------------------------------------------------------

WINDOW_START = pd.Timestamp("2011-01-01")
WINDOW_END = pd.Timestamp("2025-12-31")

# --------------------------------------------------------------------------
# Constituent history source
# --------------------------------------------------------------------------

WIKIPEDIA_PAGE = "List of S&P 500 companies"
WIKIPEDIA_URL = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
WIKIPEDIA_OLDID_URL = "https://en.wikipedia.org/w/index.php"

# The live page carried a "Selected changes to the list of S&P 500 components"
# table until mid-2025; it has since been removed, leaving only the current
# constituent table. Without the changes table there is NO removed-constituent
# history, and a census built on the current table alone is survivorship-biased
# by construction. This revision is the pinned fallback: the last one verified to
# still contain the changes table (2025-05-27, 372 change rows, 308 of them
# inside the study window). The script tries the live page first and only falls
# back to this, reporting which source it used.
WIKIPEDIA_CHANGES_OLDID = "1292523673"

# Wikipedia rate-limits aggressively from shared IPs; unrelated to EDGAR's limit.
WIKI_MAX_ATTEMPTS = 5
WIKI_BACKOFF_SECONDS = 15
WIKI_TIMEOUT_SECONDS = 40
WIKI_USER_AGENT = "earnings-surprise-research (see SEC_CONTACT_EMAIL in src/data_loader.py)"

# Minimum index changes inside the window before the history is called usable.
# Real S&P 500 turnover is roughly 20-25 names a year; anything far below that
# means the table is a stub and the reconstruction would be silently incomplete.
MIN_CHANGES_IN_WINDOW = 150

# --------------------------------------------------------------------------
# Coverage thresholds
# --------------------------------------------------------------------------

# Fewer than this many quarterly records cannot support a year-over-year panel
# across the window (15 years of quarters would be ~60 periods before dedupe).
THIN_RECORD_THRESHOLD = 20

# The XOM failure mode: SEC's map repoints a ticker at a successor registrant
# whose filings only start at the reorganization, so EDGAR history begins long
# after the company was already in the index. A gap this large between joining
# the index and the first available filing is not a real gap -- it is the wrong
# entity. Two years is well beyond any legitimate reporting delay.
REMAP_LAG_DAYS = 730

OUTPUT_CSV = "data/universe_audit.csv"


# --------------------------------------------------------------------------
# Helpers
# --------------------------------------------------------------------------


def section(title: str) -> None:
    print("\n" + "=" * 78)
    print(title)
    print("=" * 78)


def wiki_get(params: dict | None = None, url: str = WIKIPEDIA_URL) -> str:
    """Fetch a Wikipedia page, backing off on the 429s shared IPs attract."""
    session = requests.Session()
    session.headers.update({"User-Agent": WIKI_USER_AGENT})

    for attempt in range(WIKI_MAX_ATTEMPTS):
        response = session.get(url, params=params, timeout=WIKI_TIMEOUT_SECONDS)
        if response.status_code == 200:
            return response.text

        # Linear backoff: Wikipedia's limiter clears on a timescale of seconds,
        # so exponential waits would idle far longer than necessary.
        wait_seconds = WIKI_BACKOFF_SECONDS * (attempt + 1)
        print(f"    Wikipedia HTTP {response.status_code}; retrying in {wait_seconds}s")
        time.sleep(wait_seconds)

    fail(
        f"Wikipedia returned HTTP {response.status_code} after "
        f"{WIKI_MAX_ATTEMPTS} attempts. Cannot source constituent history."
    )


def find_changes_table(tables: list) -> pd.DataFrame | None:
    """Pick out the added/removed changes table from a page's parsed tables.

    Identified by its column structure rather than its position, so a new table
    appearing on the page does not silently shift the selection.
    """
    for table in tables:
        flattened = [" ".join(str(part) for part in col) if isinstance(col, tuple) else str(col)
                     for col in table.columns]
        joined = " ".join(flattened).lower()

        if "added" in joined and "removed" in joined and "date" in joined:
            return table

    return None


# --------------------------------------------------------------------------
# Stage 1 -- constituent history
# --------------------------------------------------------------------------


def load_constituent_tables() -> tuple:
    """Return (current_table, changes_table, provenance) for the index.

    Tries the live page first, as it is the authoritative current state. Falls
    back to the pinned revision only for the changes table, which the live page
    no longer carries.
    """
    section("STAGE 1 -- constituent history")

    print(f"source: live {WIKIPEDIA_PAGE}")
    live_html = wiki_get()
    live_tables = pd.read_html(io.StringIO(live_html))
    print(f"    parsed {len(live_tables)} tables from the live page")

    current = live_tables[0]
    changes = find_changes_table(live_tables)

    if changes is not None:
        print("    changes table found on the live page")
        return current, changes, "live page"

    print("    NO changes table on the live page (the section was removed).")
    print(f"    falling back to pinned revision oldid={WIKIPEDIA_CHANGES_OLDID}")

    pinned_html = wiki_get(params={"oldid": WIKIPEDIA_CHANGES_OLDID}, url=WIKIPEDIA_OLDID_URL)
    pinned_tables = pd.read_html(io.StringIO(pinned_html))
    changes = find_changes_table(pinned_tables)

    if changes is None:
        fail(
            f"The pinned revision {WIKIPEDIA_CHANGES_OLDID} no longer yields a "
            f"changes table either. Constituent history cannot be sourced from "
            f"Wikipedia; do not substitute today's list."
        )

    provenance = f"current: live page; changes: pinned revision {WIKIPEDIA_CHANGES_OLDID}"
    return current, changes, provenance


def normalise_changes(changes: pd.DataFrame) -> pd.DataFrame:
    """Flatten the changes table to date / added_ticker / removed_ticker."""
    # The table has a two-level header (Added->Ticker, Removed->Ticker, ...).
    flat = changes.copy()
    flat.columns = [
        "_".join(str(part) for part in col).lower() if isinstance(col, tuple) else str(col).lower()
        for col in flat.columns
    ]

    def pick(*needles):
        for column in flat.columns:
            if all(needle in column for needle in needles):
                return column
        return None

    date_column = pick("date")
    added_column = pick("added", "ticker")
    removed_column = pick("removed", "ticker")

    if not all([date_column, added_column, removed_column]):
        fail(f"Changes table columns not recognised: {list(flat.columns)}")

    tidy = pd.DataFrame(
        {
            "date": pd.to_datetime(flat[date_column], errors="coerce", format="mixed"),
            "added": flat[added_column].astype(str).str.strip(),
            "removed": flat[removed_column].astype(str).str.strip(),
        }
    )

    # Wikipedia writes an empty cell as the string "nan" once astype(str) runs.
    tidy["added"] = tidy["added"].replace({"nan": None, "": None})
    tidy["removed"] = tidy["removed"].replace({"nan": None, "": None})

    return tidy


def assess_history(changes: pd.DataFrame) -> pd.DataFrame:
    """Check the changes table really covers the window, and stop if it does not."""
    parsed = int(changes["date"].notna().sum())
    print(f"\nchanges rows            : {len(changes)}  ({parsed} with a parseable date)")

    dated = changes.dropna(subset=["date"])
    if dated.empty:
        fail("No change rows carried a parseable date; history is unusable.")

    print(f"changes date range      : {dated['date'].min().date()} .. {dated['date'].max().date()}")

    in_window = dated[dated["date"].between(WINDOW_START, WINDOW_END)]
    print(f"changes in {WINDOW_START.year}-{WINDOW_END.year}     : {len(in_window)}")

    per_year = in_window.groupby(in_window["date"].dt.year).size()
    print("\nchanges per year:")
    for year, count in per_year.items():
        print(f"    {year}  {count:3d}  {'#' * count}")

    # A table that does not span the window cannot reconstruct membership, and
    # falling back to today's list would silently substitute survivors.
    covered_years = set(per_year.index)
    expected_years = set(range(WINDOW_START.year, WINDOW_END.year + 1))
    missing_years = sorted(expected_years - covered_years)

    if len(in_window) < MIN_CHANGES_IN_WINDOW:
        fail(
            f"Only {len(in_window)} index changes inside {WINDOW_START.year}-"
            f"{WINDOW_END.year}, below the {MIN_CHANGES_IN_WINDOW} needed to trust "
            f"the reconstruction. Reporting coverage and stopping rather than "
            f"substituting today's constituent list."
        )

    if missing_years:
        print(f"\nNOTE: no recorded changes in {missing_years} -- thin but not disqualifying.")

    print(f"\nhistory accepted: {len(in_window)} changes spanning the window.")
    return in_window


def build_membership(current: pd.DataFrame, in_window: pd.DataFrame) -> pd.DataFrame:
    """Reconstruct first/last seen in the index for every ever-constituent."""
    current_tickers = set(current["Symbol"].astype(str).str.strip())

    # Earliest add and latest removal per ticker, from the changes table.
    added_events = in_window.dropna(subset=["added"]).groupby("added")["date"].min()
    removed_events = in_window.dropna(subset=["removed"]).groupby("removed")["date"].max()

    universe = current_tickers | set(added_events.index) | set(removed_events.index)
    universe.discard("")

    # "Date added" on the current table dates today's members, including ones
    # that joined long before the changes table's window.
    current_added = {}
    if "Date added" in current.columns:
        parsed_added = pd.to_datetime(current["Date added"], errors="coerce", format="mixed")
        for symbol, added_date in zip(current["Symbol"].astype(str).str.strip(), parsed_added):
            if pd.notna(added_date):
                current_added[symbol] = added_date

    rows = []
    for ticker in sorted(universe):
        is_current = ticker in current_tickers

        # Prefer the current table's own "Date added"; fall back to the changes
        # table; otherwise the ticker was already a member when the window opened
        # and its true join date is left-censored.
        first_seen = current_added.get(ticker)
        if first_seen is None and ticker in added_events.index:
            first_seen = added_events[ticker]

        left_censored = first_seen is None or first_seen < WINDOW_START
        if left_censored:
            # Conservative: treat membership as starting when the window opens.
            first_seen = WINDOW_START

        if is_current:
            last_seen = pd.NaT  # still in the index
        elif ticker in removed_events.index:
            last_seen = removed_events[ticker]
        else:
            last_seen = pd.NaT

        rows.append(
            {
                "ticker": ticker,
                "first_seen_in_index": first_seen,
                "last_seen_in_index": last_seen,
                "still_in_index": is_current,
                "first_seen_left_censored": left_censored,
            }
        )

    membership = pd.DataFrame(rows)

    print(f"\ncurrent constituents    : {len(current_tickers)}")
    print(f"ever-added in window    : {len(added_events)}")
    print(f"ever-removed in window  : {len(removed_events)}")
    print(f"EVER-CONSTITUENT total  : {len(membership)}")

    departed = membership[~membership["still_in_index"]]
    print(
        f"removed and not current : {len(departed)}  "
        f"<- the names a survivors-only census would miss"
    )

    return membership


# --------------------------------------------------------------------------
# Stage 2 -- CIK resolution
# --------------------------------------------------------------------------


def resolve_universe(membership: pd.DataFrame) -> pd.DataFrame:
    section("STAGE 2 -- ticker -> CIK resolution")

    cik_map = load_ticker_cik_map()
    print(f"SEC ticker map entries  : {len(cik_map)}")

    ciks = []
    overridden = []
    for ticker in membership["ticker"]:
        cik, _company, was_overridden = try_resolve_cik(ticker, cik_map)
        ciks.append(cik)
        overridden.append(was_overridden)

    membership = membership.copy()
    membership["cik"] = ciks
    membership["cik_overridden"] = overridden

    resolved = membership["cik"].notna()
    print(f"resolved to a CIK       : {int(resolved.sum())} / {len(membership)}")
    print(f"unresolved              : {int((~resolved).sum())}")

    # Unresolved names are overwhelmingly the departed ones: SEC's map lists
    # CURRENT registrants, so an acquired or delisted company is simply absent.
    departed_unresolved = (~resolved) & (~membership["still_in_index"])
    current_unresolved = (~resolved) & membership["still_in_index"]
    print(f"    of which departed   : {int(departed_unresolved.sum())}")
    print(f"    of which current    : {int(current_unresolved.sum())}")

    return membership


# --------------------------------------------------------------------------
# Stage 3 -- EDGAR record census (counts only)
# --------------------------------------------------------------------------


def census_eps_records(membership: pd.DataFrame) -> pd.DataFrame:
    section("STAGE 3 -- EDGAR EPS record census")

    to_fetch = membership[membership["cik"].notna()]
    estimate_minutes = len(to_fetch) * SEC_SLEEP_SECONDS / 60
    print(f"fetching counts for {len(to_fetch)} CIKs at {SEC_SLEEP_SECONDS}s each")
    print(f"estimated floor: ~{estimate_minutes:.0f} min of throttle alone\n")

    record_counts = {}
    earliest_ends = {}
    latest_ends = {}
    failure_reasons = {}

    for position, (_index, row) in enumerate(to_fetch.iterrows(), start=1):
        ticker = row["ticker"]

        # Non-fatal: a barren CIK is a finding to record, not a reason to abort.
        facts, reason = try_fetch_quarterly_eps(ticker, row["cik"])

        if facts is None:
            record_counts[ticker] = 0
            failure_reasons[ticker] = reason
        else:
            record_counts[ticker] = len(facts)
            earliest_ends[ticker] = facts["period_end"].min()
            latest_ends[ticker] = facts["period_end"].max()

        if position % 50 == 0:
            print(f"    {position}/{len(to_fetch)} fetched")

    census = membership.copy()
    census["n_eps_records"] = census["ticker"].map(record_counts)
    census["earliest_period_end"] = census["ticker"].map(earliest_ends)
    census["latest_period_end"] = census["ticker"].map(latest_ends)
    census["failure_reason"] = census["ticker"].map(failure_reasons)

    return census


# --------------------------------------------------------------------------
# Stage 4 -- flags
# --------------------------------------------------------------------------


def assign_flags(census: pd.DataFrame) -> pd.DataFrame:
    """One primary flag per ticker, most specific diagnosis winning."""
    census = census.copy()

    no_cik = census["cik"].isna()
    no_records = (~no_cik) & (census["n_eps_records"].fillna(0) == 0)

    # The remap signature: EDGAR history starts long after index membership did.
    # Measured against first_seen_in_index, which is floored at the window start
    # for left-censored members, so the comparison is conservative.
    history_starts_late = (
        census["earliest_period_end"] - census["first_seen_in_index"]
    ).dt.days > REMAP_LAG_DAYS
    suspected_remap = (~no_cik) & (~no_records) & history_starts_late.fillna(False)

    thin = (
        (~no_cik)
        & (~no_records)
        & (~suspected_remap)
        & (census["n_eps_records"] < THIN_RECORD_THRESHOLD)
    )

    # Priority: unresolvable, then empty, then the specific remap diagnosis,
    # then plain thinness. A remap is usually also thin; naming it as a remap is
    # more useful because it is fixable with a CIK override.
    flag = pd.Series("OK", index=census.index)
    flag[thin] = "THIN"
    flag[suspected_remap] = "SUSPECTED_REMAP"
    flag[no_records] = "NO_EPS_RECORDS"
    flag[no_cik] = "NO_CIK"
    census["flag"] = flag

    return census


# --------------------------------------------------------------------------
# Reporting
# --------------------------------------------------------------------------


def report_census(census: pd.DataFrame, provenance: str) -> None:
    section("COVERAGE SUMMARY")

    total = len(census)
    resolved = int(census["cik"].notna().sum())
    zero_records = int((census["flag"] == "NO_EPS_RECORDS").sum())
    thin = int((census["n_eps_records"].fillna(0) < THIN_RECORD_THRESHOLD).sum())
    usable = int((census["flag"] == "OK").sum())

    print(f"constituent history source : {provenance}")
    print(f"ever-constituents          : {total}")
    print(f"resolved to a CIK          : {resolved}  ({resolved / total:.1%})")
    print(f"returned zero EPS records  : {zero_records}")
    print(f"fewer than {THIN_RECORD_THRESHOLD} records      : {thin}")
    print(f"USABLE (flag OK)           : {usable}  ({usable / total:.1%})")

    print("\nflag breakdown:")
    for flag, count in census["flag"].value_counts().items():
        print(f"    {flag:18s} {count:5d}")

    print("\nusable split by index status:")
    for still_in, group in census.groupby("still_in_index"):
        label = "current members" if still_in else "departed"
        group_ok = int((group["flag"] == "OK").sum())
        print(f"    {label:16s} {group_ok:4d} usable / {len(group):4d}  ({group_ok / len(group):.1%})")

    # --- earliest period_end distribution ---------------------------------
    print("\ndistribution of earliest available period_end:")
    with_history = census.dropna(subset=["earliest_period_end"])

    if with_history.empty:
        print("    none")
    else:
        by_year = with_history.groupby(with_history["earliest_period_end"].dt.year).size()
        for year, count in by_year.items():
            bar = "#" * min(count // 5, 60)
            print(f"    {year}  {count:4d}  {bar}")

        reach_window = int((with_history["earliest_period_end"] <= WINDOW_START).sum())
        print(
            f"\n    tickers whose history reaches {WINDOW_START.date()} or earlier: "
            f"{reach_window} of {len(with_history)}"
        )

    # --- remap suspects ---------------------------------------------------
    remaps = census[census["flag"] == "SUSPECTED_REMAP"]
    print(f"\nSUSPECTED_REMAP tickers ({len(remaps)}) -- the XOM failure mode:")
    print("    (in the index for years, but EDGAR history starts much later;")
    print("     each needs a CIK_OVERRIDES entry pointing at the historical filer)")

    if not remaps.empty:
        detail = pd.DataFrame(
            {
                "ticker": remaps["ticker"],
                "cik": remaps["cik"],
                "in_index_from": remaps["first_seen_in_index"].dt.date,
                "eps_from": remaps["earliest_period_end"].dt.date,
                "n_records": remaps["n_eps_records"],
                "gap_years": (
                    (remaps["earliest_period_end"] - remaps["first_seen_in_index"]).dt.days / 365
                ).round(1),
            }
        )
        detail = detail.sort_values("gap_years", ascending=False)
        print(detail.head(40).to_string(index=False))
        if len(detail) > 40:
            print(f"    ... and {len(detail) - 40} more (full list in the CSV)")


def write_audit(census: pd.DataFrame) -> None:
    section("OUTPUT")

    output_columns = [
        "ticker",
        "cik",
        "first_seen_in_index",
        "last_seen_in_index",
        "n_eps_records",
        "earliest_period_end",
        "flag",
    ]

    audit = census[output_columns].sort_values("ticker")
    os.makedirs("data", exist_ok=True)
    audit.to_csv(OUTPUT_CSV, index=False)

    print(f"universe audit -> {OUTPUT_CSV}  ({len(audit)} rows)")
    print(f"columns: {output_columns}")


# --------------------------------------------------------------------------
# Entry point
# --------------------------------------------------------------------------


def main() -> None:
    print("S&P 500 ever-constituent census -- counts only, no panel is built.")

    current, changes_raw, provenance = load_constituent_tables()
    changes = normalise_changes(changes_raw)
    in_window = assess_history(changes)
    membership = build_membership(current, in_window)

    membership = resolve_universe(membership)
    census = census_eps_records(membership)
    census = assign_flags(census)

    report_census(census, provenance)
    write_audit(census)


if __name__ == "__main__":
    main()
