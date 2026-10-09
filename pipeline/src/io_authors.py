"""Author dataclass and CSV loader (tier value parameterized per university)."""

from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass
class Author:
    openalex_id: str
    name: str
    tier: str
    work_count: int
    first_year: int
    last_year: int


def _as_int(value: Any, default: int = 0) -> int:
    try:
        return int(value) if value not in (None, "") else default
    except (TypeError, ValueError):
        return default


def load_authors(path: Path, since_year: int, tier_value: str) -> list[Author]:
    """Parse a university's quantum authors CSV, keeping only the given tier value active since since_year."""
    authors: list[Author] = []
    with path.open(encoding="utf-8", newline="") as f:
        for row in csv.DictReader(f):
            if row.get("kth_author_tier") != tier_value:
                continue
            if _as_int(row.get("last_quantum_year")) < since_year:
                continue
            authors.append(
                Author(
                    openalex_id=row.get("author_openalex_id", ""),
                    name=row.get("author_name", ""),
                    tier=row.get("kth_author_tier", ""),
                    work_count=_as_int(row.get("quantum_work_count")),
                    first_year=_as_int(row.get("first_quantum_year")),
                    last_year=_as_int(row.get("last_quantum_year")),
                )
            )
    return authors


if __name__ == "__main__":
    import csv as _csv
    import tempfile

    with tempfile.TemporaryDirectory() as td:
        path = Path(td) / "authors.csv"
        with path.open("w", newline="") as f:
            w = _csv.DictWriter(f, fieldnames=[
                "author_openalex_id", "author_name", "kth_author_tier",
                "quantum_work_count", "first_quantum_year", "last_quantum_year",
            ])
            w.writeheader()
            w.writerow({
                "author_openalex_id": "A1", "author_name": "Test Author",
                "kth_author_tier": "core_chalmers_quantum_author",
                "quantum_work_count": "5", "first_quantum_year": "2018", "last_quantum_year": "2022",
            })
            w.writerow({
                "author_openalex_id": "A2", "author_name": "Other Tier Author",
                "kth_author_tier": "weak_chalmers_affiliation",
                "quantum_work_count": "1", "first_quantum_year": "2019", "last_quantum_year": "2019",
            })
        authors = load_authors(path, since_year=2000, tier_value="core_chalmers_quantum_author")
        assert len(authors) == 1, authors
        assert authors[0].name == "Test Author"
    print("io_authors (parameterized) smoke test passed")
