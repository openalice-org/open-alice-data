#!/usr/bin/env python3
"""
build_domain_shares.py — port the Sprint 6 SPECTER2 domain probabilities into
per-person domain *shares* and merge them into people_viz.json.

The shares feed the Sprint 7 "Map / Domains" barycentric (ternary) view. We
re-use the probabilities already computed in `people_taxonomy.json` rather than
recomputing SPECTER2 (cheap join, no GPU).

For each person we read `domain_probs_specter` (a dict over
computing/communication/sensing/other), DROP "other", and renormalize the three
remaining reals so they sum to 1. Output fields merged by OpenAlex `id`:
    sensing_share, communication_share, computing_share : float (sum == 1)

Fallbacks (so the frontend auto-disables the Domains lens for that person):
    * taxonomy file missing            -> warn, leave all three shares null
    * person has no taxonomy entry     -> shares null
    * domain_top_specter == "unknown"  -> shares null
    * probs missing / sum to zero      -> shares null

Usage:
    python3 build_domain_shares.py
    python3 build_domain_shares.py --in app/data/people_viz.json --out app/data/people_viz.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

# ----------------------------------------------------------------------------
# Paths
# ----------------------------------------------------------------------------
SPRINT11 = Path(__file__).resolve().parent
ROOT = SPRINT11.parent
DEFAULT_VIZ = SPRINT11 / "data" / "people_viz.json"
# NOTE: the original source hardcoded this to a sibling sprint directory;
# Sprint 8 produces its own people_taxonomy.json (via this dir's
# build_taxonomy_data.py + build_domain_specter.py), so this must point at
# this same directory instead.
TAXONOMY = SPRINT11 / "data" / "people_taxonomy.json"

# The three domains kept for the ternary; "other" is dropped before renorm.
DOMAINS = ("sensing", "communication", "computing")


def load_json(path: Path) -> dict:
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def shares_for(entry: dict) -> dict | None:
    """
    Compute renormalized sensing/communication/computing shares for one
    taxonomy entry, or None if there is no usable evidence.
    """
    if not entry:
        return None
    if entry.get("domain_top_specter") == "unknown":
        return None
    probs = entry.get("domain_probs_specter")
    if not isinstance(probs, dict):
        return None
    reals = [float(probs.get(d, 0.0) or 0.0) for d in DOMAINS]
    total = sum(reals)
    if total <= 0:
        return None
    return {
        f"{d}_share": v / total
        for d, v in zip(DOMAINS, reals)
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--in", dest="inp", default=str(DEFAULT_VIZ),
                    help="people_viz.json to read")
    ap.add_argument("--out", dest="out", default=None,
                    help="output path (default: in place)")
    args = ap.parse_args()

    in_path = Path(args.inp)
    out_path = Path(args.out) if args.out else in_path

    data = load_json(in_path)
    people = data["people"]

    tax_by_id: dict = {}
    if TAXONOMY.exists():
        tax = load_json(TAXONOMY)
        tax_by_id = {p["id"]: p for p in tax.get("people", []) if p.get("id")}
    else:
        print(f"[shares] WARNING: taxonomy file missing ({TAXONOMY}); "
              "leaving all shares null (Domains lens will disable).",
              file=sys.stderr)

    n_with = 0
    n_null = 0
    for p in people:
        shares = shares_for(tax_by_id.get(p.get("id")))
        if shares is None:
            p["sensing_share"] = None
            p["communication_share"] = None
            p["computing_share"] = None
            n_null += 1
        else:
            p["sensing_share"] = shares["sensing_share"]
            p["communication_share"] = shares["communication_share"]
            p["computing_share"] = shares["computing_share"]
            n_with += 1

    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

    print(f"[shares] {len(people)} people: {n_with} with shares, "
          f"{n_null} null -> {out_path}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
