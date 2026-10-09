"""3-stage EPO patent fetch pipeline with caching."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from src.epo_client import EPOClient, _as_int, _clean_space, _flatten_text, _nested_values, _preferred_lang_text


def _normalize_name_for_epo(name: str) -> str:
    name = _clean_space(name)
    if not name or " " not in name or name.isupper():
        return name
    parts = name.split()
    return f"{parts[-1]} {' '.join(parts[:-1])}"


def load_jsonl_cache(path: Path, key_field: str) -> dict[str, dict[str, Any]]:
    cache: dict[str, dict[str, Any]] = {}
    if not path.exists():
        return cache
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            item = json.loads(line)
            k = str(item.get(key_field, ""))
            if k:
                cache[k] = item
    return cache


def append_jsonl(path: Path, item: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(item, ensure_ascii=False) + "\n")


def fetch_patents(
    authors: list[Any],
    client: EPOClient | None,
    caches_dir: Path,
    since_year: int,
    max_authors: int | None = None,
    skip_fetch: bool = False,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, dict[str, Any]]]:
    """Fetch patents for authors through 3-stage cache pipeline.

    Stages:
      1. author search  → caches/epo_author_search.jsonl
      2. patent text    → caches/epo_patent_text.jsonl
      3. patent biblio  → caches/epo_patent_biblio.jsonl

    Returns (patent_records, author_patent_links, biblio_map).
    """
    search_cache_path = caches_dir / "epo_author_search.jsonl"
    text_cache_path = caches_dir / "epo_patent_text.jsonl"
    biblio_cache_path = caches_dir / "epo_patent_biblio.jsonl"

    search_cache = load_jsonl_cache(search_cache_path, "author_name")
    text_cache = load_jsonl_cache(text_cache_path, "epo_id")
    biblio_cache = load_jsonl_cache(biblio_cache_path, "epo_id")
    # Normalize: cache file stores {"epo_id": ..., "biblio": {...}}; extract raw biblio
    for epo_id, item in list(biblio_cache.items()):
        if "biblio" in item:
            biblio_cache[epo_id] = item["biblio"]

    selected = authors[:max_authors] if max_authors is not None else authors
    print(f"EPO: searching {len(selected)} core authors (since {since_year}) …", flush=True)

    all_patents: dict[str, dict[str, Any]] = {}
    author_links: list[dict[str, Any]] = []
    failures: list[str] = []

    for idx, author in enumerate(selected, 1):
        print(f"  [{idx}/{len(selected)}] {author.name}", flush=True)
        epo_name = _normalize_name_for_epo(author.name)

        # --- Search phase ---
        if author.name in search_cache:
            item = search_cache[author.name]
        else:
            if skip_fetch or client is None:
                continue
            try:
                total, refs = client.search_by_inventor(epo_name, since_year)
                error = ""
            except Exception as exc:
                total, refs, error = -1, [], str(exc)
                print(f"    search error: {error[:120]}", flush=True)
                failures.append(f"search:{author.name}")
            item = {
                "author_name": author.name,
                "epo_name": epo_name,
                "total": total,
                "refs": refs,
            }
            if not error:
                append_jsonl(search_cache_path, item)
                search_cache[author.name] = item

        refs = item.get("refs", [])
        if not refs:
            continue

        # Filter by year post-fetch
        refs = [r for r in refs if not r.get("date") or int(r["date"][:4] or 0) >= since_year]
        if not refs:
            print(f"    0 patents after year filter (since {since_year})", flush=True)
            continue

        print(f"    {len(refs)} patents found (total={item.get('total', '?')})", flush=True)

        # --- Biblio + text fetch phase ---
        for ref in refs:
            country = ref.get("country", "")
            doc_number = ref.get("doc_number", "")
            kind = ref.get("kind", "")
            date_str = ref.get("date", "")
            if not country or not doc_number:
                continue
            epo_id = f"{country}{doc_number}"

            # --- Fetch biblio (needed for filtering, independent of text) ---
            if epo_id in biblio_cache:
                biblio = biblio_cache[epo_id]
            elif not skip_fetch and client is not None:
                try:
                    biblio = client.fetch_biblio(country, doc_number)
                except Exception as exc:
                    print(f"    biblio fetch error for {epo_id}: {str(exc)[:120]}", flush=True)
                    biblio = {}
                    failures.append(f"biblio:{epo_id}")
                if biblio:
                    append_jsonl(biblio_cache_path, {"epo_id": epo_id, "biblio": biblio})
                    biblio_cache[epo_id] = biblio
            else:
                biblio = {}

            # --- Fetch patent text ---
            if epo_id in text_cache:
                patent_record = text_cache[epo_id]
            elif not skip_fetch and client is not None:
                try:
                    abstract = client.fetch_abstract(country, doc_number)
                    claims = client.fetch_claims(country, doc_number)
                except Exception as exc:
                    print(f"    text fetch error for {epo_id}: {str(exc)[:120]}", flush=True)
                    failures.append(f"text:{epo_id}")
                    continue

                title_candidates = _nested_values(biblio, {"invention-title", "invention_title", "title"})
                title = ""
                for tc in title_candidates:
                    t = _preferred_lang_text(tc).strip()
                    if t:
                        title = t
                        break

                year = _as_int(date_str[:4]) if date_str else 0

                patent_record = {
                    "epo_id": epo_id,
                    "doc_number": doc_number,
                    "jurisdiction": country,
                    "kind": kind,
                    "year_published": year,
                    "title": [{"lang": "en", "text": title}] if title else [],
                    "abstract": abstract,
                    "claims_text": claims,
                }
                append_jsonl(text_cache_path, patent_record)
                text_cache[epo_id] = patent_record
            else:
                continue  # skip_fetch and not in cache

            # Skip patents with no text
            has_abstract = bool(str(patent_record.get("abstract", "")).strip())
            has_claims = bool(str(patent_record.get("claims_text", "")).strip())
            if not has_abstract and not has_claims:
                continue

            all_patents[epo_id] = patent_record
            author_links.append({
                "author_name": author.name,
                "author_openalex_id": author.openalex_id,
                "patent_epo_id": epo_id,
                "patent_year": patent_record.get("year_published", 0),
                "patent_title": patent_record.get("title", []),
                "patent_jurisdiction": country,
            })

    if failures:
        sample = ", ".join(failures[:8])
        raise RuntimeError(
            f"EPO retrieval incomplete: {len(failures)} failed request(s) ({sample}). "
            "Successful cache entries were retained; rerun to resume before publishing."
        )

    print(f"EPO: {len(all_patents)} unique patents with text fetched.", flush=True)
    return list(all_patents.values()), author_links, biblio_cache


if __name__ == "__main__":
    # Verify cache helpers work
    import tempfile

    with tempfile.TemporaryDirectory() as td:
        tdir = Path(td)
        cache = load_jsonl_cache(tdir / "empty.jsonl", "key")
        assert cache == {}, "Empty cache should be empty dict"

        append_jsonl(tdir / "test.jsonl", {"key": "a", "val": 1})
        cache = load_jsonl_cache(tdir / "test.jsonl", "key")
        assert cache == {"a": {"key": "a", "val": 1}}, "Cache load mismatch"

        print("fetch_patents cache helpers smoke test passed")
