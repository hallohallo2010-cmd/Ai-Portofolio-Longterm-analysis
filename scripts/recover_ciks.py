#!/usr/bin/env python3
"""Recover CIKs that ticker-based resolution misses, by company NAME.

Two failure modes leave the census blind, and both are resolution problems
rather than data problems:

  1. NO_CIK -- SEC's company_tickers.json lists only CURRENT registrants, so an
     acquired or delisted company has no entry at all. Its filings still exist
     under a CIK; only the ticker route to them is gone.

  2. SUSPECTED_REMAP -- the ticker resolves, but to a successor registrant whose
     filings begin at a reorganization. The historical filer is a separate CIK.

Both are addressed here with https://www.sec.gov/Archives/edgar/cik-lookup-data.txt,
SEC's full historical name->CIK list, which includes defunct filers.

This script PROPOSES and VERIFIES; it does not silently adopt anything. Fuzzy
name matches are reported separately and never auto-accepted.

Run:  python scripts/recover_ciks.py
"""

from __future__ import annotations

import difflib
import io
import os
import re
import sys
import time

import pandas as pd
import requests

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

from src.data_loader import (  # noqa: E402
    fail,
    try_fetch_quarterly_eps,
)

# --------------------------------------------------------------------------
# Configuration
# --------------------------------------------------------------------------

CIK_LOOKUP_URL = "https://www.sec.gov/Archives/edgar/cik-lookup-data.txt"
CIK_LOOKUP_CACHE = "data/cik-lookup-data.txt"

WIKIPEDIA_OLDID_URL = "https://en.wikipedia.org/w/index.php"
WIKIPEDIA_CHANGES_OLDID = "1292523673"
WIKI_USER_AGENT = "earnings-surprise-research (see SEC_CONTACT_EMAIL in src/data_loader.py)"

UNIVERSE_CSV = "data/universe_audit.csv"
RECOVERY_CSV = "data/cik_recovery.csv"

# A fuzzy match below this similarity is not even worth showing a human.
FUZZY_CUTOFF = 0.90

# Corporate-form words carry no identifying information and differ freely
# between SEC's registration name and an index listing's display name.
NAME_NOISE_WORDS = {
    "THE", "INC", "INCORPORATED", "CORP", "CORPORATION", "CO", "COMPANY",
    "COMPANIES", "PLC", "LTD", "LIMITED", "LLC", "LP", "HOLDING", "HOLDINGS",
    "GROUP", "SA", "NV", "AG", "CLASS", "COM", "CMN", "NEW", "DEL",
}

TOP_N_TO_PRINT = 40


def section(title: str) -> None:
    print("\n" + "=" * 78)
    print(title)
    print("=" * 78)


def normalise_name(name: str) -> str:
    """Aggressively reduce a company name to its identifying core.

    SEC registration names and index display names differ in punctuation, case,
    corporate form, and trailing qualifiers. Stripping all of that is what makes
    the two joinable at all.
    """
    if not isinstance(name, str):
        return ""

    text = name.upper()
    # Ampersand and slash separate words in one source and join them in another.
    text = text.replace("&", " AND ").replace("/", " ")
    # Drop anything parenthesised: "(Class A)", "(The)" and similar qualifiers.
    text = re.sub(r"\([^)]*\)", " ", text)
    # Everything that is not a letter, digit or space is noise.
    text = re.sub(r"[^A-Z0-9 ]", " ", text)

    words = [word for word in text.split() if word and word not in NAME_NOISE_WORDS]
    return " ".join(words)


# --------------------------------------------------------------------------
# Sources
# --------------------------------------------------------------------------


def load_cik_lookup() -> pd.DataFrame:
    """Fetch (and cache) SEC's full historical name->CIK list."""
    section("SOURCE -- SEC historical CIK lookup")

    if os.path.exists(CIK_LOOKUP_CACHE):
        print(f"using cached {CIK_LOOKUP_CACHE}")
        raw = open(CIK_LOOKUP_CACHE, encoding="latin-1").read()
    else:
        print(f"fetching {CIK_LOOKUP_URL}")
        response = requests.get(
            CIK_LOOKUP_URL,
            headers={"User-Agent": WIKI_USER_AGENT},
            timeout=300,
        )
        if response.status_code != 200:
            fail(f"CIK lookup file returned HTTP {response.status_code}.")
        # The file is latin-1, not UTF-8; decoding as UTF-8 raises on some names.
        raw = response.content.decode("latin-1")
        os.makedirs("data", exist_ok=True)
        open(CIK_LOOKUP_CACHE, "w", encoding="latin-1").write(raw)
        print(f"cached to {CIK_LOOKUP_CACHE}  ({len(response.content) / 1e6:.1f} MB)")

    # Format is "COMPANY NAME:0000123456:" -- one line per registered name, and
    # a company appears once per name it has ever filed under.
    rows = []
    for line in raw.split("\n"):
        line = line.strip()
        if not line:
            continue
        parts = line.split(":")
        if len(parts) < 2 or not parts[1]:
            continue
        rows.append((parts[0], parts[1]))

    lookup = pd.DataFrame(rows, columns=["name", "cik"])
    lookup["norm"] = lookup["name"].map(normalise_name)
    lookup = lookup[lookup["norm"] != ""]

    print(f"rows parsed             : {len(lookup)}")
    print(f"distinct CIKs           : {lookup['cik'].nunique()}")
    print(f"distinct normalised names: {lookup['norm'].nunique()}")

    return lookup


def load_index_names() -> dict:
    """Map ticker -> company name, from both Wikipedia tables at the pinned revision."""
    session = requests.Session()
    session.headers.update({"User-Agent": WIKI_USER_AGENT})

    for attempt in range(5):
        response = session.get(
            WIKIPEDIA_OLDID_URL, params={"oldid": WIKIPEDIA_CHANGES_OLDID}, timeout=40
        )
        if response.status_code == 200:
            break
        time.sleep(15 * (attempt + 1))
    else:
        fail("Could not fetch the pinned Wikipedia revision for company names.")

    current, changes = pd.read_html(io.StringIO(response.text))

    names = {}
    for symbol, security in zip(
        current["Symbol"].astype(str).str.strip(), current["Security"].astype(str)
    ):
        names[symbol] = security

    # The changes table names each added/removed company alongside its ticker --
    # the only place a DEPARTED constituent's name survives.
    changes.columns = [
        "_".join(str(part) for part in col).lower() if isinstance(col, tuple) else str(col).lower()
        for col in changes.columns
    ]

    def column(*needles):
        for candidate in changes.columns:
            if all(needle in candidate for needle in needles):
                return candidate
        return None

    for ticker_col, name_col in [
        (column("added", "ticker"), column("added", "security")),
        (column("removed", "ticker"), column("removed", "security")),
    ]:
        if ticker_col is None or name_col is None:
            continue
        for ticker, security in zip(changes[ticker_col], changes[name_col]):
            ticker = str(ticker).strip()
            if ticker and ticker != "nan" and ticker not in names:
                names[ticker] = str(security)

    return names


# --------------------------------------------------------------------------
# Matching
# --------------------------------------------------------------------------


def match_names(targets: pd.DataFrame, lookup: pd.DataFrame) -> pd.DataFrame:
    """Join index company names to SEC registrant names.

    Returns one row per target with match_kind in {exact, exact_ambiguous,
    fuzzy, none}. Fuzzy matches are surfaced for human review, never adopted.
    """
    # Group the lookup by normalised name so an exact hit yields all its CIKs.
    by_norm = lookup.groupby("norm")["cik"].unique()
    norm_index = list(by_norm.index)

    results = []
    for _position, row in targets.iterrows():
        normalised = normalise_name(row["company_name"])

        if not normalised:
            results.append({**row, "match_kind": "none", "match_name": None,
                            "candidate_ciks": None, "similarity": None})
            continue

        if normalised in by_norm.index:
            ciks = list(by_norm[normalised])
            # One name can map to several CIKs (re-registrations, subsidiaries).
            kind = "exact" if len(ciks) == 1 else "exact_ambiguous"
            results.append({**row, "match_kind": kind, "match_name": normalised,
                            "candidate_ciks": ciks, "similarity": 1.0})
            continue

        close = difflib.get_close_matches(normalised, norm_index, n=1, cutoff=FUZZY_CUTOFF)
        if close:
            best = close[0]
            similarity = difflib.SequenceMatcher(None, normalised, best).ratio()
            results.append({**row, "match_kind": "fuzzy", "match_name": best,
                            "candidate_ciks": list(by_norm[best]),
                            "similarity": round(similarity, 3)})
        else:
            results.append({**row, "match_kind": "none", "match_name": None,
                            "candidate_ciks": None, "similarity": None})

    return pd.DataFrame(results)


def probe_candidates(candidate_ciks: list, ticker: str) -> tuple:
    """Fetch EPS counts for candidate CIKs; return the one with the most history.

    Counts are the evidence: a name match that yields no EPS facts has not
    recovered anything usable, and among several candidates the real operating
    filer is the one that actually reported quarterly EPS.
    """
    best = (None, 0, None, None)  # cik, count, earliest, latest

    for cik in candidate_ciks[:4]:  # cap: ambiguous names can list many shells
        padded = str(cik).zfill(10)
        facts, _reason = try_fetch_quarterly_eps(ticker, padded)

        if facts is None:
            continue
        if len(facts) > best[1]:
            best = (padded, len(facts), facts["period_end"].min(), facts["period_end"].max())

    return best


# --------------------------------------------------------------------------
# Task 1 -- recover NO_CIK tickers by name
# --------------------------------------------------------------------------


def recover_no_cik(universe: pd.DataFrame, lookup: pd.DataFrame, names: dict) -> pd.DataFrame:
    section("TASK 1 -- recover NO_CIK tickers by company name")

    missing = universe[universe["flag"] == "NO_CIK"].copy()
    missing["company_name"] = missing["ticker"].map(names)

    named = missing["company_name"].notna()
    print(f"NO_CIK tickers            : {len(missing)}")
    print(f"  with a company name     : {int(named.sum())}")
    print(f"  with NO name available  : {int((~named).sum())}  (cannot be matched at all)")

    targets = missing.loc[named, ["ticker", "company_name", "first_seen_in_index",
                                  "last_seen_in_index"]]
    matched = match_names(targets, lookup)

    print("\nmatch kinds:")
    for kind, count in matched["match_kind"].value_counts().items():
        print(f"    {kind:18s} {count:4d}")

    # --- verify by fetching EPS counts ------------------------------------
    # Only exact matches are probed. A fuzzy match is a suggestion for a human,
    # and spending requests on it would imply a confidence this does not have.
    confident = matched[matched["match_kind"].isin(["exact", "exact_ambiguous"])]
    print(f"\nprobing EPS for {len(confident)} exact/ambiguous matches...")

    resolved_cik = {}
    record_count = {}
    earliest = {}

    for position, (_index, row) in enumerate(confident.iterrows(), start=1):
        cik, count, first_end, _last_end = probe_candidates(row["candidate_ciks"], row["ticker"])
        if cik is not None:
            resolved_cik[row["ticker"]] = cik
            record_count[row["ticker"]] = count
            earliest[row["ticker"]] = first_end
        if position % 25 == 0:
            print(f"    {position}/{len(confident)} probed")

    matched["recovered_cik"] = matched["ticker"].map(resolved_cik)
    matched["n_eps_records"] = matched["ticker"].map(record_count)
    matched["earliest_period_end"] = matched["ticker"].map(earliest)

    # --- outcome ----------------------------------------------------------
    with_cik = matched["recovered_cik"].notna()
    with_history = with_cik & (matched["n_eps_records"].fillna(0) > 0)

    print(f"\nRESULT of {len(missing)} NO_CIK tickers:")
    print(f"    resolved to a CIK        : {int(with_cik.sum())}")
    print(f"    of those, real EPS history: {int(with_history.sum())}")
    print(f"    fuzzy (needs review)     : {int((matched['match_kind'] == 'fuzzy').sum())}")
    print(f"    no name match at all     : {int((matched['match_kind'] == 'none').sum())}")
    unrecoverable = len(missing) - int(with_history.sum())
    print(f"    STILL UNRECOVERABLE      : {unrecoverable}")

    # --- eyeball table ----------------------------------------------------
    # Ranked by index tenure: the longest-tenured names matter most, since they
    # contribute the most observations if recovered.
    tenure = matched.copy()
    tenure["first_seen_in_index"] = pd.to_datetime(tenure["first_seen_in_index"])
    tenure["last_seen_in_index"] = pd.to_datetime(tenure["last_seen_in_index"])
    tenure["tenure_days"] = (
        tenure["last_seen_in_index"] - tenure["first_seen_in_index"]
    ).dt.days

    top = tenure.sort_values("tenure_days", ascending=False).head(TOP_N_TO_PRINT)

    print(f"\nTOP {TOP_N_TO_PRINT} BY INDEX TENURE -- every proposed name->CIK pair:")
    display = pd.DataFrame(
        {
            "ticker": top["ticker"],
            "index_name": top["company_name"].str.slice(0, 28),
            "matched_sec_name": top["match_name"].fillna("-").str.slice(0, 28),
            "kind": top["match_kind"],
            "sim": top["similarity"].fillna(0).round(2),
            "cik": top["recovered_cik"].fillna("-"),
            "n_eps": top["n_eps_records"].fillna(0).astype(int),
            "eps_from": top["earliest_period_end"].dt.date.astype(str).replace("NaT", "-"),
            "tenure_yr": (top["tenure_days"] / 365).round(1),
        }
    )
    print(display.to_string(index=False))

    fuzzy = matched[matched["match_kind"] == "fuzzy"]
    if not fuzzy.empty:
        print(f"\nFUZZY MATCHES ({len(fuzzy)}) -- NOT adopted, review before use:")
        review = pd.DataFrame(
            {
                "ticker": fuzzy["ticker"],
                "index_name": fuzzy["company_name"].str.slice(0, 34),
                "closest_sec_name": fuzzy["match_name"].str.slice(0, 34),
                "similarity": fuzzy["similarity"],
            }
        )
        print(review.sort_values("similarity", ascending=False).to_string(index=False))

    return matched


# --------------------------------------------------------------------------
# Task 2 -- historical filers for SUSPECTED_REMAP tickers
# --------------------------------------------------------------------------


def recover_remaps(universe: pd.DataFrame, lookup: pd.DataFrame, names: dict) -> pd.DataFrame:
    section("TASK 2 -- historical filers for SUSPECTED_REMAP tickers")

    remaps = universe[universe["flag"] == "SUSPECTED_REMAP"].copy()
    remaps["company_name"] = remaps["ticker"].map(names)
    print(f"SUSPECTED_REMAP tickers: {len(remaps)}")

    by_norm = lookup.groupby("norm")["cik"].unique()

    rows = []
    for _index, row in remaps.iterrows():
        ticker = row["ticker"]
        normalised = normalise_name(row.get("company_name") or "")

        # Every CIK SEC has ever registered under this company's name. The
        # successor is among them; so, usually, is the historical filer.
        candidates = list(by_norm[normalised]) if normalised in by_norm.index else []

        # Exclude the CIK the ticker already resolves to -- that is the successor
        # whose short history triggered the flag in the first place.
        current_cik = str(row["cik"]).split(".")[0].zfill(10) if pd.notna(row["cik"]) else None
        alternatives = [c for c in candidates if str(c).zfill(10) != current_cik]

        best_cik, best_count, best_first, best_last = probe_candidates(alternatives, ticker)

        rows.append(
            {
                "ticker": ticker,
                "company_name": row.get("company_name"),
                "old_cik": current_cik,
                "old_n_records": row["n_eps_records"],
                "old_earliest": row["earliest_period_end"],
                "n_candidates": len(alternatives),
                "new_cik": best_cik,
                "new_n_records": best_count if best_cik else None,
                "new_earliest": best_first,
                "new_latest": best_last,
            }
        )

    proposals = pd.DataFrame(rows)
    proposals["old_earliest"] = pd.to_datetime(proposals["old_earliest"])

    # An override is only worth adopting if it actually reaches further back.
    improved = (
        proposals["new_cik"].notna()
        & proposals["new_earliest"].notna()
        & (proposals["new_earliest"] < proposals["old_earliest"])
    )
    proposals["adopt"] = improved

    print(f"\nBEFORE / AFTER (adopt = the alternative reaches further back):")
    display = pd.DataFrame(
        {
            "ticker": proposals["ticker"],
            "old_cik": proposals["old_cik"],
            "old_n": proposals["old_n_records"].fillna(0).astype(int),
            "old_from": proposals["old_earliest"].dt.date.astype(str),
            "new_cik": proposals["new_cik"].fillna("-"),
            "new_n": proposals["new_n_records"].fillna(0).astype(int),
            "new_from": proposals["new_earliest"].dt.date.astype(str).replace("NaT", "-"),
            "adopt": proposals["adopt"],
        }
    )
    print(display.to_string(index=False))

    adopted = proposals[proposals["adopt"]]
    print(f"\nadoptable overrides: {len(adopted)} of {len(proposals)}")

    if not adopted.empty:
        print("\npaste into CIK_OVERRIDES in src/data_loader.py:")
        for _index, row in adopted.iterrows():
            gain = (row["old_earliest"] - row["new_earliest"]).days / 365
            print(f'    "{row["ticker"]}": "{row["new_cik"]}",'
                  f'  # {row["company_name"]}: +{gain:.1f}yr history')

    return proposals


# --------------------------------------------------------------------------
# Entry point
# --------------------------------------------------------------------------


def main() -> None:
    if not os.path.exists(UNIVERSE_CSV):
        fail(f"{UNIVERSE_CSV} not found -- run scripts/build_universe.py first.")

    universe = pd.read_csv(
        UNIVERSE_CSV,
        parse_dates=["first_seen_in_index", "last_seen_in_index", "earliest_period_end"],
    )

    lookup = load_cik_lookup()
    names = load_index_names()
    print(f"company names available : {len(names)}")

    recovered = recover_no_cik(universe, lookup, names)
    proposals = recover_remaps(universe, lookup, names)

    section("OUTPUT")
    os.makedirs("data", exist_ok=True)

    recovered_out = recovered[
        ["ticker", "company_name", "match_kind", "match_name", "similarity",
         "recovered_cik", "n_eps_records", "earliest_period_end"]
    ]
    recovered_out.to_csv(RECOVERY_CSV, index=False)
    print(f"name-recovery detail -> {RECOVERY_CSV}  ({len(recovered_out)} rows)")

    proposals.to_csv("data/remap_proposals.csv", index=False)
    print(f"remap proposals      -> data/remap_proposals.csv  ({len(proposals)} rows)")


if __name__ == "__main__":
    main()
