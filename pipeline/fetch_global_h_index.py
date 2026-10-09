#!/usr/bin/env python3
"""Cache OpenAlex global h-index values and merge them into frontend datasets."""

from __future__ import annotations

import argparse
import json
import os
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import requests

OPENALEX_AUTHORS_URL = "https://api.openalex.org/authors"
DEFAULT_MAILTO = os.environ.get("OPENALEX_MAILTO", "")
BATCH_SIZE = 100


def load_jsonl(path: Path) -> dict[str, dict[str, Any]]:
    rows: dict[str, dict[str, Any]] = {}
    if not path.exists():
        return rows
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                row = json.loads(line)
                if row.get("id"):
                    rows[row["id"]] = row
    return rows


def append_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def fetch_batch(author_ids: list[str], mailto: str, api_key: str = "") -> list[dict[str, Any]]:
    short_ids = [author_id.rsplit("/", 1)[-1] for author_id in author_ids]
    params = {
        "filter": f"openalex:{'|'.join(short_ids)}",
        "per-page": BATCH_SIZE,
    }
    if api_key:
        params["api_key"] = api_key
    else:
        params["mailto"] = mailto
    headers = {"User-Agent": f"qsip-pipeline (mailto:{mailto})"}
    for attempt in range(3):
        response = requests.get(
            OPENALEX_AUTHORS_URL, params=params, headers=headers, timeout=30
        )
        if response.status_code == 429 and attempt < 2:
            # Retry-After on an exhausted polite-pool quota can be hours;
            # cap the wait so a rate limit never hangs the pipeline.
            time.sleep(min(float(response.headers.get("Retry-After", 2 ** attempt)), 30))
            continue
        response.raise_for_status()
        return response.json().get("results", [])
    return []


def atomic_write_json(path: Path, data: dict[str, Any], indent: int) -> None:
    temp_path = path.with_suffix(path.suffix + ".tmp")
    temp_path.write_text(json.dumps(data, ensure_ascii=False, indent=indent), encoding="utf-8")
    temp_path.replace(path)


def enrich_dataset(path: Path, stats: dict[str, dict[str, Any]]) -> tuple[int, int]:
    raw = path.read_text(encoding="utf-8")
    data = json.loads(raw)
    first_indented_line = next((line for line in raw.splitlines()[1:] if line.strip()), "  ")
    indent = max(1, len(first_indented_line) - len(first_indented_line.lstrip()))
    enriched = 0
    missing = 0
    for person in data.get("people", []):
        person["quantum_h_index"] = int(person.get("h_index", 0) or 0)
        row = stats.get(person.get("id", ""))
        if row is None:
            person["global_h_index"] = None
            missing += 1
        else:
            person["global_h_index"] = int(row.get("global_h_index", 0) or 0)
            enriched += 1

    size_modes = data.setdefault("meta", {}).setdefault("size_modes", {})
    size_modes["quantum_h_index"] = "h-index from accepted quantum-corpus works"
    size_modes["global_h_index"] = "global h-index reported by OpenAlex"
    atomic_write_json(path, data, indent)
    return enriched, missing


def main() -> None:
    root = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--viz", type=Path, default=root / "data" / "people_viz.json")
    parser.add_argument("--taxonomy", type=Path, default=root / "data" / "people_taxonomy.json")
    parser.add_argument("--cache", type=Path, default=root / "caches" / "openalex_global_h_index.jsonl")
    parser.add_argument("--mailto", default=DEFAULT_MAILTO)
    parser.add_argument("--api-key", default=os.environ.get("OPENALEX_API_KEY", ""))  # from the environment, never a command-line flag
    parser.add_argument("--refresh", action="store_true")
    args = parser.parse_args()

    datasets = [path for path in (args.viz, args.taxonomy) if path.exists()]
    author_ids: set[str] = set()
    for path in datasets:
        data = json.loads(path.read_text(encoding="utf-8"))
        author_ids.update(person.get("id", "") for person in data.get("people", []))
    author_ids.discard("")

    cache = load_jsonl(args.cache)
    missing_ids = sorted(author_ids if args.refresh else author_ids - cache.keys())
    print(f"[global-h] {len(author_ids) - len(missing_ids)}/{len(author_ids)} cached; {len(missing_ids)} to fetch")

    for offset in range(0, len(missing_ids), BATCH_SIZE):
        batch = missing_ids[offset : offset + BATCH_SIZE]
        fetched_at = datetime.now(timezone.utc).isoformat()
        rows = []
        for author in fetch_batch(batch, args.mailto, args.api_key):
            row = {
                "id": author["id"],
                "global_h_index": int((author.get("summary_stats") or {}).get("h_index", 0) or 0),
                "works_count": int(author.get("works_count", 0) or 0),
                "fetched_at": fetched_at,
            }
            rows.append(row)
            cache[row["id"]] = row
        append_jsonl(args.cache, rows)
        end = min(offset + len(batch), len(missing_ids))
        print(f"[global-h] {offset + 1}-{end}/{len(missing_ids)} fetched")
        time.sleep(0.1)

    for path in datasets:
        enriched, missing = enrich_dataset(path, cache)
        print(f"[global-h] {path}: {enriched} enriched, {missing} unavailable")


if __name__ == "__main__":
    main()
