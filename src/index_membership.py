"""S&P 500 membership reconstruction: who was in the index, and when.

The panel needs more than a list of ever-constituents. It needs to know, for any
given date, whether a ticker was actually IN the index then -- otherwise a
company's 2013 quarters end up in a panel it only joined in 2016, and the
backtest trades names nobody could have held.

Membership is reconstructed from Wikipedia's added/removed changes table. That
table was removed from the live page in mid-2025, so the live page is tried
first (it is authoritative for current membership) and the changes table comes
from a pinned revision. Provenance is reported, never silently substituted.
"""

from __future__ import annotations

import io
import time

import pandas as pd
import requests

from src.data_loader import fail

WIKIPEDIA_URL = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
WIKIPEDIA_OLDID_URL = "https://en.wikipedia.org/w/index.php"

# Last revision verified to still contain the changes table (2025-05-27):
# 372 change rows, 308 of them inside the study window.
WIKIPEDIA_CHANGES_OLDID = "1292523673"

WIKI_USER_AGENT = "earnings-surprise-research (see SEC_CONTACT_EMAIL in src/data_loader.py)"
WIKI_MAX_ATTEMPTS = 5
WIKI_BACKOFF_SECONDS = 15
WIKI_TIMEOUT_SECONDS = 40

# Real S&P 500 turnover is ~20-25 names a year; far below that means the table
# is a stub and the reconstruction would be silently incomplete.
MIN_CHANGES_IN_WINDOW = 150

# A membership interval that is still open ends here rather than at "today", so
# the panel's notion of current membership matches the pinned snapshot.
MEMBERSHIP_OPEN_END = pd.Timestamp("2026-12-31")


def _wiki_get(params: dict | None = None, url: str = WIKIPEDIA_URL) -> str:
    """Fetch a Wikipedia page, backing off on the 429s shared IPs attract."""
    session = requests.Session()
    session.headers.update({"User-Agent": WIKI_USER_AGENT})

    response = None
    for attempt in range(WIKI_MAX_ATTEMPTS):
        response = session.get(url, params=params, timeout=WIKI_TIMEOUT_SECONDS)
        if response.status_code == 200:
            return response.text
        # Linear backoff: the limiter clears in seconds, not minutes.
        time.sleep(WIKI_BACKOFF_SECONDS * (attempt + 1))

    status = response.status_code if response is not None else "no response"
    fail(f"Wikipedia returned {status} after {WIKI_MAX_ATTEMPTS} attempts.")


def _find_changes_table(tables: list):
    """Identify the changes table by column structure, not position."""
    for table in tables:
        flattened = [
            " ".join(str(part) for part in col) if isinstance(col, tuple) else str(col)
            for col in table.columns
        ]
        joined = " ".join(flattened).lower()
        if "added" in joined and "removed" in joined and "date" in joined:
            return table
    return None


def load_tables() -> tuple:
    """Return (current_table, changes_table, provenance).

    Both tables come from the SAME pinned revision. Mixing a live current table
    with a pinned changes table leaves a blind spot for every index change
    between the revision and today, which previously misclassified 21 departed
    tickers as current.
    """
    pinned_html = _wiki_get(params={"oldid": WIKIPEDIA_CHANGES_OLDID}, url=WIKIPEDIA_OLDID_URL)
    tables = pd.read_html(io.StringIO(pinned_html))

    changes = _find_changes_table(tables)
    if changes is None:
        fail(
            f"Pinned revision {WIKIPEDIA_CHANGES_OLDID} no longer yields a changes "
            f"table. Membership history cannot be sourced; do not substitute "
            f"today's constituent list."
        )

    provenance = f"Wikipedia pinned revision {WIKIPEDIA_CHANGES_OLDID} (current + changes)"
    return tables[0], changes, provenance


def normalise_changes(changes: pd.DataFrame) -> pd.DataFrame:
    """Flatten the two-level changes header to date / added / removed."""
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
    # astype(str) turns empty cells into the literal string "nan".
    tidy["added"] = tidy["added"].replace({"nan": None, "": None})
    tidy["removed"] = tidy["removed"].replace({"nan": None, "": None})

    return tidy.dropna(subset=["date"])


def build_intervals(current: pd.DataFrame, changes: pd.DataFrame, window_start) -> tuple:
    """Reconstruct membership intervals per ticker.

    Returns ({ticker: [(start, end), ...]}, metadata DataFrame). A ticker can
    hold several disjoint intervals -- companies do leave and rejoin -- so a
    single first/last pair would wrongly cover the gap between two stints.
    """
    if len(changes) < MIN_CHANGES_IN_WINDOW:
        fail(
            f"Only {len(changes)} index changes available, below the "
            f"{MIN_CHANGES_IN_WINDOW} needed to trust the reconstruction."
        )

    current_tickers = {
        symbol.strip()
        for symbol in current["Symbol"].astype(str)
        if isinstance(symbol, str) and symbol.strip() and symbol.strip().lower() != "nan"
    }

    # Date added on the current table dates today's members, including those
    # that joined long before the changes table's coverage begins.
    current_added = {}
    if "Date added" in current.columns:
        parsed = pd.to_datetime(current["Date added"], errors="coerce", format="mixed")
        for symbol, added in zip(current["Symbol"].astype(str).str.strip(), parsed):
            if pd.notna(added):
                current_added[symbol] = added

    def clean_ticker(value):
        """Only a non-empty string is a ticker.

        Guarding on truthiness alone is a trap: Series.replace(..., None) leaves
        float NaN in an object column, and NaN is TRUTHY, so an empty cell would
        sail through and become a ticker key.
        """
        if not isinstance(value, str):
            return None
        cleaned = value.strip()
        return cleaned or None

    events = {}
    for _index, row in changes.iterrows():
        added = clean_ticker(row["added"])
        removed = clean_ticker(row["removed"])
        if added:
            events.setdefault(added, []).append((row["date"], "add"))
        if removed:
            events.setdefault(removed, []).append((row["date"], "remove"))

    universe = set(events) | current_tickers
    universe.discard("")

    intervals = {}
    metadata = []

    for ticker in sorted(universe):
        ticker_events = sorted(events.get(ticker, []))

        # A ticker whose first recorded event is a REMOVAL was already a member
        # before the changes table starts: its membership is left-censored.
        left_censored = bool(ticker_events) and ticker_events[0][1] == "remove"
        if not ticker_events and ticker in current_tickers:
            left_censored = current_added.get(ticker, window_start) < window_start

        spans = []
        open_start = None

        if left_censored:
            # Member from before the window; opened at the earliest date we model.
            open_start = min(window_start, ticker_events[0][0]) if ticker_events else window_start

        for date, kind in ticker_events:
            if kind == "add":
                if open_start is None:
                    open_start = date
            else:  # remove
                if open_start is not None:
                    spans.append((open_start, date))
                    open_start = None

        if open_start is None and ticker in current_tickers and not ticker_events:
            open_start = current_added.get(ticker, window_start)

        if open_start is not None:
            # Still open: either currently a member, or removed after the
            # changes table's coverage ends.
            end = MEMBERSHIP_OPEN_END if ticker in current_tickers else MEMBERSHIP_OPEN_END
            spans.append((open_start, end))

        if not spans:
            continue

        intervals[ticker] = spans
        metadata.append(
            {
                "ticker": ticker,
                "first_seen_in_index": min(start for start, _end in spans),
                "last_seen_in_index": (
                    pd.NaT if ticker in current_tickers else max(end for _start, end in spans)
                ),
                "still_in_index": ticker in current_tickers,
                # "Removed" means it left the index at some point, whether or not
                # it later rejoined -- the survivorship-relevant distinction.
                "is_removed_name": any(kind == "remove" for _date, kind in ticker_events),
                "n_stints": len(spans),
                "left_censored": left_censored,
            }
        )

    return intervals, pd.DataFrame(metadata)


def is_member_at(intervals: dict, ticker: str, moment) -> bool:
    """Was this ticker in the index at this moment?"""
    spans = intervals.get(ticker)
    if not spans or pd.isna(moment):
        return False

    for start, end in spans:
        if start <= moment <= end:
            return True

    return False
