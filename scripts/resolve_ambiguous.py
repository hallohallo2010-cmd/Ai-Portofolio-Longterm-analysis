#!/usr/bin/env python3
"""Resolve ambiguous and fuzzy name->CIK matches by filing-period overlap.

A company name can be registered by several CIKs (re-registrations, subsidiaries,
unrelated firms sharing a name). Name similarity alone cannot separate them. What
can: whether a candidate's filings actually cover the years the ticker was in the
index. A subsidiary that filed for two years in the 1990s is not the S&P 500
constituent of 2011-2018, however well its name matches.

Selection rule, applied in order and reported per ticker:
    a) drop candidates with zero EPS records
    b) keep candidates whose period_end range overlaps the index tenure;
       if exactly one overlaps, take it
    c) if several overlap, take the longest overlap
    d) if none overlap, mark UNRESOLVED_AMBIGUOUS and leave it out

Every candidate and every decision is printed. Nothing here is meant to be
trusted without reading the table.

Run:  python scripts/resolve_ambiguous.py
"""

from __future__ import annotations

import io
import os
import sys
import time

import pandas as pd
import requests

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

from src.data_loader import fail, try_fetch_quarterly_eps  # noqa: E402
from scripts.recover_ciks import (  # noqa: E402
    CIK_LOOKUP_CACHE,
    WIKIPEDIA_CHANGES_OLDID,
    WIKIPEDIA_OLDID_URL,
    WIKI_USER_AGENT,
    normalise_name,
    section,
)

RECOVERY_CSV = "data/cik_recovery.csv"
UNIVERSE_CSV = "data/universe_audit.csv"
DECISIONS_CSV = "data/ambiguous_decisions.csv"

# Fuzzy matches the user cleared for overlap-testing. Everything else stays out.
FUZZY_TO_TEST = ["NFX", "FII", "LLTC", "WIN"]

# A current constituent has no removal date; its tenure runs to the census date.
TENURE_OPEN_END = pd.Timestamp("2025-12-31")


def load_lookup_index():
    """normalised name -> array of CIKs, from the cached SEC registrant list."""
    if not os.path.exists(CIK_LOOKUP_CACHE):
        fail(f"{CIK_LOOKUP_CACHE} missing -- run scripts/recover_ciks.py first.")

    rows = []
    for line in open(CIK_LOOKUP_CACHE, encoding="latin-1"):
        line = line.strip()
        if not line:
            continue
        parts = line.split(":")
        if len(parts) < 2 or not parts[1]:
            continue
        rows.append((parts[0], parts[1]))

    lookup = pd.DataFrame(rows, columns=["name", "cik"])
    lookup["norm"] = lookup["name"].map(normalise_name)
    return lookup.groupby("norm")["cik"].unique(), lookup


def overlap_days(range_start, range_end, tenure_start, tenure_end) -> int:
    """Days shared by a filer's period range and the ticker's index tenure."""
    if pd.isna(range_start) or pd.isna(range_end):
        return 0

    latest_start = max(range_start, tenure_start)
    earliest_end = min(range_end, tenure_end)
    return max((earliest_end - latest_start).days, 0)


def resolve(targets: pd.DataFrame, by_norm, lookup) -> pd.DataFrame:
    """Probe every candidate for every target and apply the selection rule."""
    # Name lookup for display: which SEC registration name each CIK carries.
    cik_to_name = lookup.drop_duplicates("cik").set_index("cik")["name"].to_dict()

    decisions = []

    for _index, row in targets.iterrows():
        ticker = row["ticker"]
        normalised = normalise_name(row["company_name"])
        candidates = list(by_norm[normalised]) if normalised in by_norm.index else []

        tenure_start = row["tenure_start"]
        tenure_end = row["tenure_end"]

        # --- probe EVERY candidate, no cap ---------------------------------
        probed = []
        for cik in candidates:
            padded = str(cik).zfill(10)
            facts, reason = try_fetch_quarterly_eps(ticker, padded)

            if facts is None:
                probed.append(
                    {"cik": padded, "name": cik_to_name.get(cik, "?"), "n": 0,
                     "first": pd.NaT, "last": pd.NaT, "overlap": 0, "reason": reason}
                )
                continue

            first_end = facts["period_end"].min()
            last_end = facts["period_end"].max()
            probed.append(
                {
                    "cik": padded,
                    "name": cik_to_name.get(cik, "?"),
                    "n": len(facts),
                    "first": first_end,
                    "last": last_end,
                    "overlap": overlap_days(first_end, last_end, tenure_start, tenure_end),
                    "reason": None,
                }
            )

        # --- rule (a): drop empties ----------------------------------------
        alive = [candidate for candidate in probed if candidate["n"] > 0]

        # --- rules (b)/(c): overlap with index tenure ----------------------
        overlapping = [candidate for candidate in alive if candidate["overlap"] > 0]

        if len(overlapping) == 1:
            chosen, rule = overlapping[0], "b_single_overlap"
        elif len(overlapping) > 1:
            # Longest overlap wins: the filer that was reporting for the most of
            # the time the ticker was actually in the index.
            chosen = max(overlapping, key=lambda candidate: candidate["overlap"])
            rule = "c_longest_overlap"
        else:
            chosen, rule = None, "d_unresolved_ambiguous"

        decisions.append(
            {
                "ticker": ticker,
                "company_name": row["company_name"],
                "tenure_start": tenure_start,
                "tenure_end": tenure_end,
                "n_candidates": len(candidates),
                "n_with_records": len(alive),
                "n_overlapping": len(overlapping),
                "chosen_cik": chosen["cik"] if chosen else None,
                "chosen_n": chosen["n"] if chosen else 0,
                "chosen_first": chosen["first"] if chosen else pd.NaT,
                "chosen_last": chosen["last"] if chosen else pd.NaT,
                "rule": rule,
                "candidates": probed,
            }
        )

    return pd.DataFrame(decisions)


def print_decisions(decisions: pd.DataFrame, heading: str) -> None:
    section(heading)

    for _index, row in decisions.iterrows():
        tenure = f"{row['tenure_start'].date()} .. {row['tenure_end'].date()}"
        print(f"\n{row['ticker']}  ({row['company_name']})   index tenure: {tenure}")

        for candidate in row["candidates"]:
            marker = " <== CHOSEN" if candidate["cik"] == row["chosen_cik"] else ""
            period = (
                f"{candidate['first'].date()}..{candidate['last'].date()}"
                if pd.notna(candidate["first"]) else "-"
            )
            overlap_years = candidate["overlap"] / 365
            print(
                f"    {candidate['cik']}  n={candidate['n']:>4d}  {period:<24s} "
                f"overlap={overlap_years:5.1f}yr  {candidate['name'][:34]:<34s}{marker}"
            )

        print(f"    -> rule {row['rule']}")


def main() -> None:
    if not os.path.exists(RECOVERY_CSV):
        fail(f"{RECOVERY_CSV} missing -- run scripts/recover_ciks.py first.")

    recovery = pd.read_csv(RECOVERY_CSV)
    universe = pd.read_csv(
        UNIVERSE_CSV, parse_dates=["first_seen_in_index", "last_seen_in_index"]
    )

    tenure = universe.set_index("ticker")
    by_norm, lookup = load_lookup_index()

    def build_targets(subset: pd.DataFrame) -> pd.DataFrame:
        rows = []
        for _index, row in subset.iterrows():
            ticker = row["ticker"]
            if ticker not in tenure.index:
                continue
            start = tenure.loc[ticker, "first_seen_in_index"]
            end = tenure.loc[ticker, "last_seen_in_index"]
            rows.append(
                {
                    "ticker": ticker,
                    "company_name": row["company_name"],
                    "tenure_start": start,
                    # An open tenure (still in the index) runs to the census date.
                    "tenure_end": end if pd.notna(end) else TENURE_OPEN_END,
                }
            )
        return pd.DataFrame(rows)

    # ---- task 1: the ambiguous set ---------------------------------------
    ambiguous = recovery[recovery["match_kind"] == "exact_ambiguous"]
    targets = build_targets(ambiguous)
    print(f"resolving {len(targets)} ambiguous tickers (probing every candidate)")

    decisions = resolve(targets, by_norm, lookup)
    print_decisions(decisions, "TASK 1 -- ambiguous name matches: every candidate, every decision")

    section("TASK 1 SUMMARY")
    print(decisions["rule"].value_counts().to_string())
    resolved = decisions[decisions["chosen_cik"].notna()]
    print(f"\nresolved: {len(resolved)} of {len(decisions)}")
    print(f"UNRESOLVED_AMBIGUOUS: {int((decisions['rule'] == 'd_unresolved_ambiguous').sum())}")

    # ---- task 2: the four testable fuzzy matches -------------------------
    fuzzy = recovery[recovery["ticker"].isin(FUZZY_TO_TEST)]
    fuzzy_targets = build_targets(fuzzy)
    fuzzy_decisions = resolve(fuzzy_targets, by_norm, lookup)

    # A fuzzy name never matches the lookup index exactly, so probe its closest
    # SEC name directly instead.
    for position, row in fuzzy.iterrows():
        if row["ticker"] in set(fuzzy_decisions["ticker"]):
            existing = fuzzy_decisions[fuzzy_decisions["ticker"] == row["ticker"]]
            if existing.iloc[0]["n_candidates"] > 0:
                continue
        match_name = row["match_name"]
        if not isinstance(match_name, str) or match_name not in by_norm.index:
            continue
        extra = build_targets(pd.DataFrame([row]))
        extra["company_name"] = match_name
        more = resolve(extra, by_norm, lookup)
        fuzzy_decisions = pd.concat([fuzzy_decisions, more], ignore_index=True)

    print_decisions(fuzzy_decisions, "TASK 2 -- fuzzy matches, accepted only on tenure overlap")

    section("TASK 2 SUMMARY")
    for _index, row in fuzzy_decisions.iterrows():
        verdict = "ACCEPT" if row["chosen_cik"] else "REJECT (no tenure overlap)"
        print(f"    {row['ticker']:6s} {verdict}  cik={row['chosen_cik'] or '-'}  rule={row['rule']}")

    # ---- persist ---------------------------------------------------------
    section("OUTPUT")
    flat = decisions.drop(columns=["candidates"])
    flat_fuzzy = fuzzy_decisions.drop(columns=["candidates"])
    combined = pd.concat([flat.assign(source="ambiguous"),
                          flat_fuzzy.assign(source="fuzzy")], ignore_index=True)
    combined.to_csv(DECISIONS_CSV, index=False)
    print(f"decisions -> {DECISIONS_CSV}  ({len(combined)} rows)")


if __name__ == "__main__":
    main()
