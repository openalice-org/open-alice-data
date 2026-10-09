"""Per-university EPO patent fetch + filter + author-summary orchestrator.

Trimmed copy of SPRINT 1/run.py: keeps stages 1 (load papers), 2 (load
authors), 3 (fetch patents), 4 (filter patents), 5 (tokenize), 6 (cluster —
needed for author_summary's patent_quantum_aligned_count), and 9 (author
summary). Drops stages 7-8 (paper<->patent correlation matrices/heatmaps),
which Sprint 8's people-map does not consume.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from dotenv import load_dotenv

from src.author_summary import build_author_summary, write_author_summary_csv
from src.cluster import cluster_documents
from src.epo_client import EPOClient
from src.fetch_patents import fetch_patents
from src.filter_patents import filter_patents
from src.io_authors import load_authors
from src.io_papers import load_papers
from src.tokenize import keyword_tokens, tokenize

load_dotenv(Path(__file__).resolve().parent.parent / ".env")


def main() -> None:
    root = Path(__file__).resolve().parent
    universities = json.loads((root / "universities.json").read_text())

    parser = argparse.ArgumentParser(description="EPO patent fetch/filter/summary for one university")
    parser.add_argument("--university", required=True, choices=sorted(universities.keys()))
    parser.add_argument("--since-year", type=int, default=2000)
    parser.add_argument("--max-authors", type=int, default=None)
    parser.add_argument("--skip-fetch", action="store_true")
    parser.add_argument(
        "--skip-patents",
        action="store_true",
        help="Build paper totals without loading or fetching patent data",
    )
    parser.add_argument("--skip-openalex", action="store_true")
    parser.add_argument("--mailto", type=str, default=os.environ.get("OPENALEX_MAILTO", ""))
    parser.add_argument("--api-key", type=str, default=os.environ.get("OPENALEX_API_KEY", ""))  # from the environment, never a command-line flag
    args = parser.parse_args()

    uni = universities[args.university]
    tier_value = f"core_{uni['tier_prefix']}_quantum_author"

    papers_path = root / "data" / args.university / "quantum_classified_works_specter_reviewed.jsonl"
    authors_path = root / "data" / args.university / "authors_v2.csv"

    caches_dir = root / "data" / args.university / "caches"
    outputs_dir = root / "data" / args.university / "outputs"

    print(f"=== {uni['label']}: Stage 1: Load papers ===")
    papers = load_papers(papers_path, args.since_year)
    print(f"Loaded {len(papers)} papers since {args.since_year}")

    print(f"=== {uni['label']}: Stage 2: Load authors (tier={tier_value}) ===")
    authors = load_authors(authors_path, args.since_year, tier_value)
    print(f"Loaded {len(authors)} core authors")
    if args.max_authors is not None:
        authors = authors[: args.max_authors]
    if not authors:
        print("ERROR: no authors selected, nothing to do")
        return

    print(f"=== {uni['label']}: Stage 3: Fetch patents ===")
    client: EPOClient | None = None
    if not args.skip_fetch:
        key = os.getenv("EPO_CONSUMER_KEY", "")
        secret = os.getenv("EPO_CONSUMER_SECRET_KEY", "")
        if not key or not secret:
            print("ERROR: EPO credentials not found in environment")
            return
        client = EPOClient(key, secret)
    if args.skip_patents:
        patent_records, author_links, biblio_map = [], [], {}
    else:
        patent_records, author_links, biblio_map = fetch_patents(
            authors, client, caches_dir, args.since_year, args.max_authors, args.skip_fetch
        )
    print(f"Fetched {len(patent_records)} unique patents")

    print(f"=== {uni['label']}: Stage 4: Filter patents ===")
    filtered, rejected = filter_patents(patent_records, biblio_map, outputs_dir, uni["applicant_regex"])
    print(f"Filtered: {len(filtered)}, rejected: {len(rejected)}")

    print(f"=== {uni['label']}: Stage 5+6: Tokenize + cluster ===")
    docs: list[dict] = []
    paper_doc_ids: set[str] = set()
    patent_doc_ids: set[str] = set()
    for paper in papers:
        tokens = tokenize(paper.title) + tokenize(paper.abstract) + keyword_tokens(paper.keywords) + keyword_tokens(paper.topics)
        doc_id = f"paper:{paper.doi}"
        docs.append({"doc_id": doc_id, "tokens": tokens})
        paper_doc_ids.add(doc_id)

    patent_record_by_id = {p["epo_id"]: p for p in patent_records}
    for fp in filtered:
        epo_id = fp["epo_id"]
        original = patent_record_by_id.get(epo_id, {})
        tokens: list[str] = []
        title = original.get("title", [])
        if isinstance(title, list):
            for t in title:
                tokens.extend(tokenize(t.get("text", "") if isinstance(t, dict) else str(t)))
        tokens.extend(tokenize(original.get("abstract", "")))
        tokens.extend(tokenize(original.get("claims_text", "")))
        doc_id = f"patent:{epo_id}"
        docs.append({"doc_id": doc_id, "tokens": tokens})
        patent_doc_ids.add(doc_id)

    cluster_result = cluster_documents(docs, max_features=500, k_range=range(5, 41, 5)) if docs else None

    print(f"=== {uni['label']}: Stage 9: Author summary ===")
    paper_doc_authors = {f"paper:{p.doi}": p.authors for p in papers}
    doc_clusters = cluster_result.doc_clusters if cluster_result else {}
    # EPO's own family id (INPADOC/DOCDB), so the same invention published
    # under several EPO ids (a national filing + its PCT/WO application, or
    # the same application in multiple offices) counts once per author
    # instead of once per publication — see build_author_summary's docstring.
    patent_family_by_id = {
        epo_id: str(biblio.get("@family-id") or epo_id) for epo_id, biblio in biblio_map.items()
    }
    summary_rows = build_author_summary(
        authors=authors,
        authors_csv=authors_path,
        papers=papers,
        paper_doc_clusters=doc_clusters,
        paper_doc_authors=paper_doc_authors,
        filtered_patents=filtered,
        patent_doc_clusters=doc_clusters,
        author_patent_links=author_links,
        epo_search_cache_path=caches_dir / "epo_author_search.jsonl",
        openalex_totals_cache=caches_dir / "openalex_author_totals.jsonl",
        mailto=args.mailto,
        api_key=args.api_key,
        skip_openalex=args.skip_openalex,
        patent_family_by_id=patent_family_by_id,
    )
    write_author_summary_csv(summary_rows, outputs_dir / "author_summary.csv")
    print(f"Wrote {outputs_dir / 'author_summary.csv'} ({len(summary_rows)} authors)")


if __name__ == "__main__":
    main()
