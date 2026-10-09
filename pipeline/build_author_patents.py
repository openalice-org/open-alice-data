"""Export per-author filtered (SE-inventor / uni-applicant) patent lists to
outputs/author_patents.json, using only cached EPO data (no API calls).

Reuses fetch_patents.fetch_patents(skip_fetch=True) + filter_patents.filter_patents
exactly as fetch_epo_patents.py's stages 3-4, so the output matches the counts
already baked into author_summary.csv's patent_kth_filtered_count.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from urllib.parse import quote

from src.fetch_patents import fetch_patents
from src.filter_patents import filter_patents
from src.io_authors import load_authors


def espacenet_link(jurisdiction: str, doc_number: str, kind: str) -> str:
    pn = f"{jurisdiction}{doc_number}{kind or ''}"
    return f"https://worldwide.espacenet.com/patent/search?q={quote('pn=' + pn)}"


def main() -> None:
    root = Path(__file__).resolve().parent
    universities = json.loads((root / "universities.json").read_text())

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--university", required=True, choices=sorted(universities.keys()))
    parser.add_argument("--since-year", type=int, default=2000)
    args = parser.parse_args()

    uni = universities[args.university]
    tier_value = f"core_{uni['tier_prefix']}_quantum_author"

    authors_path = root / "data" / args.university / "authors_v2.csv"

    # fetch_epo_patents.py always caches under data/<uni>/caches, even for kth
    # (its papers/authors CSVs live in SPRINT 1, but its EPO cache does not) —
    # match that so this reads the same cache the live refetch just populated.
    caches_dir = root / "data" / args.university / "caches"
    outputs_dir = root / "data" / args.university / "outputs"

    authors = load_authors(authors_path, args.since_year, tier_value)
    if not authors:
        print(f"[{args.university}] no authors, skipping")
        return

    patent_records, author_links, biblio_map = fetch_patents(
        authors, None, caches_dir, args.since_year, None, skip_fetch=True
    )
    filtered, _rejected = filter_patents(patent_records, biblio_map, outputs_dir, uni["applicant_regex"])
    filtered_ids = {fp["epo_id"] for fp in filtered}

    kind_by_id = {p["epo_id"]: p.get("kind", "") for p in patent_records}

    by_author: dict[str, dict] = {}
    for link in author_links:
        epo_id = link["patent_epo_id"]
        if epo_id not in filtered_ids:
            continue
        aid = link["author_openalex_id"]
        entry = by_author.setdefault(aid, {"author_name": link["author_name"], "patents": {}})
        if epo_id in entry["patents"]:
            continue
        title_parts = link.get("patent_title") or []
        title = ""
        for t in title_parts:
            title = (t.get("text", "") if isinstance(t, dict) else str(t)).strip()
            if title:
                break
        jurisdiction = link.get("patent_jurisdiction", "")
        doc_number = epo_id[len(jurisdiction):] if jurisdiction and epo_id.startswith(jurisdiction) else epo_id
        kind = kind_by_id.get(epo_id, "")
        entry["patents"][epo_id] = {
            "epo_id": epo_id,
            "jurisdiction": jurisdiction,
            "doc_number": doc_number,
            "kind": kind,
            "year": link.get("patent_year", 0),
            "title": title,
            "link": espacenet_link(jurisdiction, doc_number, kind),
        }

    out = {
        aid: {
            "author_name": entry["author_name"],
            "patents": sorted(entry["patents"].values(), key=lambda p: -(p["year"] or 0)),
        }
        for aid, entry in by_author.items()
    }

    out_path = outputs_dir / "author_patents.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(out, ensure_ascii=False, indent=1))
    n_patents = sum(len(v["patents"]) for v in out.values())
    print(f"[{args.university}] {len(out)} authors, {n_patents} filtered patents -> {out_path}")


if __name__ == "__main__":
    main()
