"""Paper dataclass and JSONL loader."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass
class Paper:
    openalex_id: str
    doi: str
    title: str
    year: int
    label: str
    keywords: list[str]
    topics: list[str]
    abstract: str
    authors: list[str]


def _normalize_doi(doi: str | None) -> str:
    if not doi:
        return ""
    doi = doi.strip()
    doi = re.sub(r"^https?://(dx\.)?doi\.org/", "", doi, flags=re.I)
    doi = re.sub(r"^doi:\s*", "", doi, flags=re.I)
    return doi.strip().lower()


def _reconstruct_abstract(index: dict[str, list[int]] | None) -> str:
    if not index:
        return ""
    pairs: list[tuple[int, str]] = []
    for token, positions in index.items():
        for pos in positions:
            pairs.append((pos, token))
    return " ".join(t for _, t in sorted(pairs))


def _as_int(value: Any, default: int = 0) -> int:
    try:
        return int(value) if value not in (None, "") else default
    except (TypeError, ValueError):
        return default


def load_papers(path: Path, since_year: int) -> list[Paper]:
    """Parse OpenAlex works JSONL into Paper objects, filtering by year and reject label."""
    papers: list[Paper] = []
    if not path.exists():
        return papers
    with path.open(encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            raw = json.loads(line)
            doi = _normalize_doi(raw.get("doi"))
            if not doi:
                continue
            year = _as_int(raw.get("publication_year"))
            if year < since_year:
                continue
            if raw.get("classification_label") == "reject":
                continue
            keywords = [
                item.get("display_name", "")
                for item in raw.get("keywords", []) or []
                if item.get("display_name")
            ]
            topics = [
                item.get("display_name", "")
                for item in raw.get("topics", []) or []
                if item.get("display_name")
            ]
            authors = [
                a.get("author", {}).get("display_name", "")
                for a in raw.get("authorships", []) or []
                if a.get("author", {}).get("display_name")
            ]
            papers.append(
                Paper(
                    openalex_id=raw.get("id", ""),
                    doi=doi,
                    title=raw.get("display_name", "") or "",
                    year=year,
                    label=raw.get("classification_label", "") or "",
                    keywords=keywords,
                    topics=topics,
                    abstract=_reconstruct_abstract(raw.get("abstract_inverted_index")),
                    authors=authors,
                )
            )
    return papers


if __name__ == "__main__":
    # Smoke test: verify load_papers parses without error
    from pathlib import Path

    sprint2 = Path(__file__).resolve().parent.parent.parent / "SPRINT 2" / "kth_quantum_classified_works_specter_reviewed.jsonl"
    if sprint2.exists():
        papers = load_papers(sprint2, since_year=2010)
        assert len(papers) > 0, "Expected at least one paper"
        assert all(isinstance(p, Paper) for p in papers), "Expected Paper objects"
        print(f"io_papers smoke test passed: loaded {len(papers)} papers")
    else:
        print("SPRINT 2 works file not found; skipping smoke test")
