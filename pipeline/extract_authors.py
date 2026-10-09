"""Author extraction + tiering from a university's classified works corpus."""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from pathlib import Path
from typing import Any

ACCEPTED_LABELS = {"core_quantum", "likely_quantum"}


def tier_for_author(quantum_work_count: int, tier_prefix: str) -> str:
    """Tier boundaries: >=3 quantum works -> core, >=1 -> probable, else weak.

    (The exact original KTH tier-boundary rule is unrecoverable — see design
    doc §Reconstructed KTH pipeline. This replicates the tier *shape*
    confirmed in the data: 3 tiers, core is the smallest/highest-confidence
    bucket, weak is name-only/low-evidence.)
    """
    if quantum_work_count >= 3:
        return f"core_{tier_prefix}_quantum_author"
    if quantum_work_count >= 1:
        return f"probable_{tier_prefix}_quantum_author"
    return f"weak_{tier_prefix}_affiliation"


def aggregate_authors(
    works: list[dict[str, Any]],
    author_objects: dict[str, dict[str, Any]],
    tier_prefix: str,
    institution_id: str,
) -> list[dict[str, Any]]:
    """Aggregate authors from classified works, counting a work only toward
    authors whose OWN authorship record lists `institution_id` among their
    institutions for that work — NOT every co-author on the work. A work is
    retrieved because at least one author is affiliated with the target
    university; without this per-authorship filter, every international
    co-author on a large-collaboration paper (common in physics) would be
    misattributed as a "university author," wildly inflating the pool with
    unrelated people and their (unrelated) patents downstream.
    """
    per_author: dict[str, dict[str, Any]] = {}

    for work in works:
        if work.get("classification_label") not in ACCEPTED_LABELS:
            continue
        year = work.get("publication_year", 0) or 0
        primary_topic = (work.get("primary_topic") or {}).get("display_name", "") or ""
        keyword_names = [kw.get("display_name", "") for kw in (work.get("keywords") or [])]

        for authorship in work.get("authorships", []) or []:
            own_institution_ids = {inst.get("id") for inst in (authorship.get("institutions") or [])}
            if institution_id not in own_institution_ids:
                continue
            author = authorship.get("author") or {}
            aid = author.get("id")
            if not aid:
                continue
            rec = per_author.setdefault(aid, {
                "author_openalex_id": aid,
                "author_name": author.get("display_name", "") or "",
                "orcid": author.get("orcid", "") or "",
                "quantum_work_count": 0,
                "years": [],
                "topics": Counter(),
                "keywords": Counter(),
                "example_work_ids": [],
                "example_titles": [],
            })
            # prefer the canonical full Author object's display_name if we fetched one
            full_author = author_objects.get(aid)
            if full_author and full_author.get("display_name"):
                rec["author_name"] = full_author["display_name"]

            rec["quantum_work_count"] += 1
            if year:
                rec["years"].append(year)
            if primary_topic:
                rec["topics"][primary_topic] += 1
            for kw in keyword_names:
                if kw:
                    rec["keywords"][kw] += 1
            if len(rec["example_work_ids"]) < 5:
                rec["example_work_ids"].append(work.get("id", "") or "")
                rec["example_titles"].append(work.get("display_name", "") or "")

    rows = []
    for rec in per_author.values():
        years = rec["years"]
        rows.append({
            "author_openalex_id": rec["author_openalex_id"],
            "author_name": rec["author_name"],
            "orcid": rec["orcid"],
            "source": "seed",
            "kth_author_tier": tier_for_author(rec["quantum_work_count"], tier_prefix),
            "quantum_work_count": rec["quantum_work_count"],
            "kth_lineage_quantum_work_count": 0,
            "first_quantum_year": min(years) if years else "",
            "last_quantum_year": max(years) if years else "",
            "top_primary_topics": ";;".join(t for t, _ in rec["topics"].most_common(3)),
            "top_keywords": ";;".join(k for k, _ in rec["keywords"].most_common(5)),
            "example_work_ids": ";;".join(rec["example_work_ids"]),
            "example_titles": ";;".join(rec["example_titles"]),
        })
    return rows


def write_authors_csv(rows: list[dict[str, Any]], path: Path) -> None:
    fieldnames = [
        "author_openalex_id", "author_name", "orcid", "source", "kth_author_tier",
        "quantum_work_count", "kth_lineage_quantum_work_count",
        "first_quantum_year", "last_quantum_year",
        "top_primary_topics", "top_keywords", "example_work_ids", "example_titles",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(rows)


def main() -> None:
    root = Path(__file__).resolve().parent
    universities = json.loads((root / "universities.json").read_text())

    parser = argparse.ArgumentParser(description="Extract + tier authors from a university's classified corpus")
    parser.add_argument("--university", required=True, choices=sorted(universities.keys()))
    args = parser.parse_args()

    tier_prefix = universities[args.university]["tier_prefix"]
    institution_id = universities[args.university]["openalex_institution_id"]

    data_dir = root / "data" / args.university
    works = [json.loads(line) for line in (data_dir / "quantum_classified_works_specter_reviewed.jsonl").open() if line.strip()]
    author_cache_path = data_dir / "author_cache.jsonl"
    author_objects = (
        {
            row["id"]: row
            for row in (json.loads(line) for line in author_cache_path.open() if line.strip())
            if row.get("id")
        }
        if author_cache_path.exists()
        else {}
    )

    rows = aggregate_authors(works, author_objects, tier_prefix, institution_id)
    write_authors_csv(rows, data_dir / "authors_v2.csv")
    print(f"Wrote {len(rows)} authors to {data_dir / 'authors_v2.csv'}")


_TEST_INSTITUTION_ID = "https://openalex.org/I66862912"  # Chalmers, for tests only


def _test_aggregate_authors():
    works = [
        {
            "id": "W1", "display_name": "Paper 1", "publication_year": 2020,
            "classification_label": "core_quantum",
            "primary_topic": {"display_name": "Quantum Computing"},
            "keywords": [{"display_name": "Qubit"}],
            "authorships": [{
                "author": {"id": "https://openalex.org/A1", "display_name": "A. Test", "orcid": ""},
                "institutions": [{"id": _TEST_INSTITUTION_ID}],
            }],
        },
        {
            "id": "W2", "display_name": "Paper 2", "publication_year": 2021,
            "classification_label": "reject",
            "primary_topic": {}, "keywords": [],
            "authorships": [{
                "author": {"id": "https://openalex.org/A1", "display_name": "A. Test", "orcid": ""},
                "institutions": [{"id": _TEST_INSTITUTION_ID}],
            }],
        },
    ]
    rows = aggregate_authors(works, {}, "chalmers", _TEST_INSTITUTION_ID)
    assert len(rows) == 1, rows
    assert rows[0]["quantum_work_count"] == 1, rows[0]  # W2 is rejected, not counted
    assert rows[0]["kth_author_tier"] == "probable_chalmers_quantum_author", rows[0]
    print("aggregate_authors test passed")


def _test_aggregate_authors_excludes_unaffiliated_coauthors():
    """A large-collaboration paper: only the co-author actually affiliated
    with the target institution should be counted, not every co-author."""
    works = [
        {
            "id": "W1", "display_name": "Collaboration paper", "publication_year": 2022,
            "classification_label": "core_quantum",
            "primary_topic": {"display_name": "Quantum Materials"},
            "keywords": [],
            "authorships": [
                {
                    "author": {"id": "https://openalex.org/A_LOCAL", "display_name": "Local Author", "orcid": ""},
                    "institutions": [{"id": _TEST_INSTITUTION_ID}],
                },
                {
                    "author": {"id": "https://openalex.org/A_FOREIGN", "display_name": "Foreign Coauthor", "orcid": ""},
                    "institutions": [{"id": "https://openalex.org/I_SOME_OTHER_INSTITUTION"}],
                },
            ],
        },
    ]
    rows = aggregate_authors(works, {}, "chalmers", _TEST_INSTITUTION_ID)
    ids = {r["author_openalex_id"] for r in rows}
    assert ids == {"https://openalex.org/A_LOCAL"}, ids
    print("aggregate_authors excludes-unaffiliated-coauthors test passed")


if __name__ == "__main__":
    import sys

    if "--university" in sys.argv:
        main()
    else:
        assert tier_for_author(5, "chalmers") == "core_chalmers_quantum_author"
        assert tier_for_author(1, "lund") == "probable_lund_quantum_author"
        assert tier_for_author(0, "chalmers") == "weak_chalmers_affiliation"
        _test_aggregate_authors()
        _test_aggregate_authors_excludes_unaffiliated_coauthors()
        print("extract_authors smoke tests passed")
