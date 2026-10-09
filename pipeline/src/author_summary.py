"""Per-author paper / patent / quantum-aligned counts."""

from __future__ import annotations

import csv
import json
import time
from pathlib import Path
from typing import Any

import requests


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    out: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            if line.strip():
                out.append(json.loads(line))
    return out


def _append_jsonl(path: Path, item: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(item, ensure_ascii=False) + "\n")


def fetch_openalex_totals(
    author_ids: list[str],
    cache_path: Path,
    delay: float = 0.15,
    mailto: str | None = None,
    api_key: str | None = None,
) -> dict[str, int]:
    """Fetch `works_count` (total papers) per OpenAlex author. Cached to JSONL.

    Returns {openalex_id: works_count}.
    """
    cache_rows = _load_jsonl(cache_path)
    cache = {row["author_openalex_id"]: row for row in cache_rows}

    out: dict[str, int] = {row["author_openalex_id"]: int(row.get("works_count", 0)) for row in cache_rows}

    headers: dict[str, str] = {}
    if mailto:
        headers["User-Agent"] = f"qsip-pipeline (mailto:{mailto})"

    missing = [a for a in author_ids if a and a not in cache]
    if not missing:
        return out

    valid_missing = [
        aid for aid in missing
        if aid.rsplit("/", 1)[-1].startswith("A")
    ]
    print(
        f"[author_summary] fetching works_count for {len(valid_missing)} authors "
        "from OpenAlex in batches of 100 …",
        flush=True,
    )
    for offset in range(0, len(valid_missing), 100):
        batch = valid_missing[offset : offset + 100]
        short_ids = [aid.rsplit("/", 1)[-1] for aid in batch]
        params = {
            "filter": f"openalex:{'|'.join(short_ids)}",
            "per-page": 100,
        }
        if api_key:
            params["api_key"] = api_key
        elif mailto:
            params["mailto"] = mailto
        try:
            r = requests.get(
                "https://api.openalex.org/authors",
                params=params,
                headers=headers,
                timeout=30,
            )
            r.raise_for_status()
            results = r.json().get("results", [])
        except Exception as exc:
            end = min(offset + len(batch), len(valid_missing))
            print(
                f"  [{offset + 1}-{end}/{len(valid_missing)}] batch fetch failed: "
                f"{str(exc)[:120]}",
                flush=True,
            )
            continue
        for data in results:
            aid = data.get("id", "")
            if not aid:
                continue
            works = int(data.get("works_count", 0) or 0)
            row = {"author_openalex_id": aid, "works_count": works}
            _append_jsonl(cache_path, row)
            out[aid] = works
        end = min(offset + len(batch), len(valid_missing))
        print(f"  [{offset + 1}-{end}/{len(valid_missing)}] cached", flush=True)
        time.sleep(delay)
    return out


def build_author_summary(
    authors: list[Any],
    authors_csv: Path,
    papers: list[Any],
    paper_doc_clusters: dict[str, set[int]],
    paper_doc_authors: dict[str, list[str]],
    filtered_patents: list[dict[str, Any]],
    patent_doc_clusters: dict[str, set[int]],
    author_patent_links: list[dict[str, Any]],
    epo_search_cache_path: Path,
    openalex_totals_cache: Path,
    mailto: str | None = None,
    api_key: str | None = None,
    skip_openalex: bool = False,
    patent_family_by_id: dict[str, str] | None = None,
) -> list[dict[str, Any]]:
    """Build per-author summary row list.

    Columns:
      author_name
      openalex_id
      paper_quantum_count          (from authors CSV `quantum_work_count`)
      paper_total_count            (OpenAlex `works_count`; 0 if skipped)
      patent_total_count           (raw EPO search hits for that inventor)
      patent_kth_filtered_count    (passed SE inventor / KTH applicant filter,
                                    deduped to one count per EPO patent FAMILY —
                                    see patent_family_by_id below)
      patent_quantum_aligned_count (same family dedup, AND cluster set ∩
                                    author's paper clusters)

    `patent_family_by_id` maps an EPO publication id (e.g. "EP1202478") to its
    INPADOC/DOCDB family id (EPO biblio's `@family-id`). The same invention is
    routinely published as several distinct EPO ids — a national filing plus
    its PCT/WO application, or the same application in multiple offices — each
    of which shows up as its own row in `filtered_patents`/`author_patent_links`.
    Without this, `patent_kth_filtered_count` counts publications, not
    inventions, and inflates an author's patent count (~6% too high on the
    Nordic corpus). Ids missing from the map (fetched before family ids were
    cached, or a lookup miss) fall back to treating the id as its own
    single-member family, matching the pre-dedup behavior for that one id.
    """
    family_of = (lambda pid: (patent_family_by_id or {}).get(pid, pid))
    # Author CSV → metadata (restrict to the exact working-set OpenAlex IDs).
    # Names are not identifiers: OpenAlex can contain split profiles with the
    # same display name, and keying here by name let a low-evidence duplicate
    # overwrite the core author's totals.
    working_ids = {a.openalex_id for a in authors}
    authors_meta: dict[str, dict[str, Any]] = {}
    with authors_csv.open(encoding="utf-8") as f:
        for row in csv.DictReader(f):
            author_id = row.get("author_openalex_id", "")
            if author_id in working_ids:
                authors_meta[author_id] = row

    # Per-author quantum paper clusters (from this run's filtered paper docs)
    author_to_clusters: dict[str, set[int]] = {}
    for doc_id, author_list in paper_doc_authors.items():
        clusters = paper_doc_clusters.get(doc_id, set())
        for name in author_list:
            author_to_clusters.setdefault(name, set()).update(clusters)

    # EPO search cache → per-author total patent hits
    epo_search_rows = _load_jsonl(epo_search_cache_path)
    patent_total_by_author: dict[str, int] = {}
    for row in epo_search_rows:
        name = row.get("author_name", "")
        # Prefer 'total'; fall back to len(refs)
        total = row.get("total", -1)
        if total is None or total < 0:
            total = len(row.get("refs", []))
        patent_total_by_author[name] = max(patent_total_by_author.get(name, 0), int(total))

    # Patents per author (post-fetch, may include duplicates across authors)
    author_to_patents: dict[str, set[str]] = {}
    for link in author_patent_links:
        author_to_patents.setdefault(link["author_name"], set()).add(link["patent_epo_id"])

    filtered_ids = {fp["epo_id"] for fp in filtered_patents}

    # OpenAlex totals
    if skip_openalex:
        openalex_totals: dict[str, int] = {}
    else:
        ids = [
            author_id
            for author_id in authors_meta
        ]
        openalex_totals = fetch_openalex_totals(ids, openalex_totals_cache, mailto=mailto, api_key=api_key)

    rows: list[dict[str, Any]] = []
    for oa_id, meta in authors_meta.items():
        name = meta.get("author_name", "")
        quantum_papers = int(meta.get("quantum_work_count", 0) or 0)
        total_papers = openalex_totals.get(oa_id, 0)
        total_patents = patent_total_by_author.get(name, 0)
        author_patent_ids = author_to_patents.get(name, set())
        kth_patents = author_patent_ids & filtered_ids
        kth_filtered_count = len({family_of(pid) for pid in kth_patents})

        author_clusters = author_to_clusters.get(name, set())
        aligned_families: set[str] = set()
        for pid in kth_patents:
            patent_doc_id = f"patent:{pid}"
            patent_clusters = patent_doc_clusters.get(patent_doc_id, set())
            if patent_clusters & author_clusters:
                aligned_families.add(family_of(pid))
        aligned = len(aligned_families)

        rows.append({
            "author_name": name,
            "openalex_id": oa_id,
            "paper_quantum_count": quantum_papers,
            "paper_total_count": total_papers,
            "patent_total_count": total_patents,
            "patent_kth_filtered_count": kth_filtered_count,
            "patent_quantum_aligned_count": aligned,
        })

    rows.sort(key=lambda r: (-r["patent_quantum_aligned_count"],
                              -r["patent_kth_filtered_count"],
                              -r["paper_quantum_count"]))
    return rows


def write_author_summary_csv(rows: list[dict[str, Any]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "author_name",
        "openalex_id",
        "paper_quantum_count",
        "paper_total_count",
        "patent_total_count",
        "patent_kth_filtered_count",
        "patent_quantum_aligned_count",
    ]
    with path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(rows)
