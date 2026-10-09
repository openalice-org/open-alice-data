"""Recover prior in-scope records hidden by current OpenAlex metadata regressions."""

from __future__ import annotations

import csv
import datetime as dt
import json
import os
import re
import unicodedata

from utils.abstract_utils import reconstruct_abstract
from utils.openalex_utils import build_works_url, fetch_page, file_sha256
from utils.quantum_topic_groups import (
    BROAD_RELATED,
    EXPLICIT_QUANTUM,
    INTRINSICALLY_QUANTUM,
    candidate_group,
)
from utils.settings import settings

SCOPE_CONFIG = "config/sweden_quantum_scope.json"
FILTER_CONFIG = "config/sweden_quantum_paper_filter.json"
TOPIC_CANDIDATES = "data/sweden_quantum_topic_candidates.csv"
DERIVED_KEYWORDS = "data/sweden_quantum_derived_keywords.csv"
CANDIDATE_UNION = "data/sweden_quantum_candidate_union.jsonl"
CANDIDATE_MANIFEST = "data/sweden_quantum_candidate_union_manifest.json"
HISTORICAL_WORKS = "data/all_universities_works.jsonl"
OUTPUT = "data/sweden_quantum_source_reconciliations.jsonl"
AUDIT_OUTPUT = "data/sweden_quantum_source_reconciliation_audit.csv"
MANIFEST_OUTPUT = "data/sweden_quantum_source_reconciliations_manifest.json"

SELECT = ",".join(
    [
        "id",
        "doi",
        "ids",
        "display_name",
        "publication_year",
        "publication_date",
        "type",
        "language",
        "authorships",
        "primary_topic",
        "topics",
        "keywords",
        "abstract_inverted_index",
        "cited_by_count",
        "is_retracted",
        "primary_location",
        "locations",
        "open_access",
        "updated_date",
    ]
)

AUDIT_FIELDS = (
    "historical_work_id",
    "resolved_work_id",
    "doi",
    "title",
    "status",
    "regression_reasons",
    "restored_fields",
    "matched_routes",
)


def load_json(path):
    with open(path, encoding="utf-8") as handle:
        return json.load(handle)


def short_id(value):
    return str(value or "").rsplit("/", 1)[-1]


def normalize_doi(value):
    return re.sub(
        r"^https?://(?:dx\.)?doi\.org/",
        "",
        str(value or "").strip(),
        flags=re.IGNORECASE,
    ).casefold()


def normalize_title(value):
    text = unicodedata.normalize("NFKD", str(value or "")).casefold()
    text = "".join(
        character for character in text if not unicodedata.combining(character)
    )
    return " ".join(re.findall(r"[a-z0-9]+", text))


def scope_country_codes(scope):
    """The country codes behind the scope's two kinds of affiliation evidence.

    `primary` is the publication-time authorship country claim and `audit` is the
    country code of a matched authorship institution, so both are read here rather
    than assuming one country or one kind of evidence.
    """
    evidence = scope.get("country_evidence") or {}

    def codes(section):
        raw = (evidence.get(section) or {}).get("filter_value") or ""
        parsed = frozenset(
            part.strip().upper() for part in raw.split("|") if part.strip()
        )
        if not parsed:
            raise ValueError(
                f"Scope must define country_evidence.{section}.filter_value"
            )
        return parsed

    return codes("primary"), codes("audit")


def has_country(work, authorship_codes, institution_codes):
    return any(
        any(
            str(code).upper() in authorship_codes
            for code in authorship.get("countries") or []
        )
        or any(
            str(institution.get("country_code") or "").upper() in institution_codes
            for institution in authorship.get("institutions") or []
            if isinstance(institution, dict)
        )
        for authorship in work.get("authorships") or []
        if isinstance(authorship, dict)
    )


def _record_score(work, authorship_codes, institution_codes):
    return (
        has_country(work, authorship_codes, institution_codes),
        isinstance(work.get("publication_year"), int),
        bool(work.get("abstract_inverted_index")),
        len(work.get("topics") or []),
        len(work.get("keywords") or []),
        work.get("updated_date") or "",
    )


def read_historical_in_scope(
    path, start_year, end_year, authorship_codes, institution_codes
):
    records = {}
    with open(path, encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            try:
                work = json.loads(line)
            except json.JSONDecodeError as exc:
                raise RuntimeError(
                    f"Invalid JSON in {path} line {line_number}"
                ) from exc
            work_id = short_id(work.get("id"))
            year = work.get("publication_year")
            if (
                not work_id
                or not has_country(work, authorship_codes, institution_codes)
                or not isinstance(year, int)
            ):
                continue
            if not start_year <= year <= end_year:
                continue
            if work_id not in records or _record_score(
                work, authorship_codes, institution_codes
            ) > _record_score(records[work_id], authorship_codes, institution_codes):
                records[work_id] = work
    return records


def read_union_index(path):
    ids = set()
    dois = {}
    titles = {}
    with open(path, encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            try:
                work = json.loads(line)
            except json.JSONDecodeError as exc:
                raise RuntimeError(
                    f"Invalid JSON in {path} line {line_number}"
                ) from exc
            work_id = short_id(work.get("id"))
            if not work_id or work_id in ids:
                raise RuntimeError("Candidate union IDs must be unique and non-empty")
            ids.add(work_id)
            doi = normalize_doi(work.get("doi"))
            title = normalize_title(work.get("display_name"))
            if doi:
                dois.setdefault(doi, set()).add(work_id)
            if title:
                titles.setdefault(title, set()).add(work_id)
    return ids, dois, titles


def _validated_results(data, context):
    results = data.get("results")
    if not isinstance(results, list):
        raise RuntimeError(f"OpenAlex returned no results list for {context}")
    return results


def fetch_by_ids(work_ids, api_key, fetch_json=fetch_page, batch_size=100, mailto=""):
    found = {}
    work_ids = sorted(set(work_ids))
    for offset in range(0, len(work_ids), batch_size):
        batch = work_ids[offset : offset + batch_size]
        url = build_works_url(
            api_key,
            mailto=mailto,
            filter="openalex_id:" + "|".join(batch),
            per_page=100,
            select=SELECT,
        )
        for work in _validated_results(fetch_json(url), "historical ID batch"):
            work_id = short_id(work.get("id"))
            if not work_id or work_id in found:
                raise RuntimeError("OpenAlex returned a missing or duplicate work ID")
            found[work_id] = work
    return found


def fetch_reconciliation_candidates(
    historical,
    api_key,
    authorship_codes,
    institution_codes,
    fetch_json=fetch_page,
    mailto="",
):
    candidates = []
    doi = normalize_doi(historical.get("doi"))
    if doi:
        url = build_works_url(
            api_key, mailto=mailto, filter=f"doi:{doi}", per_page=25, select=SELECT
        )
        candidates.extend(_validated_results(fetch_json(url), f"DOI {doi}"))
    title = historical.get("display_name") or ""
    if title:
        url = build_works_url(
            api_key, mailto=mailto, search=title, per_page=25, select=SELECT
        )
        candidates.extend(_validated_results(fetch_json(url), f"title {title!r}"))
    unique = {}
    for work in candidates:
        work_id = short_id(work.get("id"))
        if work_id:
            unique[work_id] = work
    exact_doi = [
        work
        for work in unique.values()
        if doi and normalize_doi(work.get("doi")) == doi
    ]
    exact_title = [
        work
        for work in unique.values()
        if normalize_title(work.get("display_name")) == normalize_title(title)
    ]
    pool = exact_doi or exact_title
    return sorted(
        pool,
        key=lambda work: _record_score(work, authorship_codes, institution_codes),
        reverse=True,
    )


def load_route_vocabulary(topic_path, keyword_path, keyword_batch_size):
    topic_groups = {}
    with open(topic_path, encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            if row.get("taxonomy_gate") != "blocked":
                topic_groups[row["topic_id"]] = candidate_group(row)

    keywords_by_tier = {"high_confidence": [], "title_confirmed": []}
    with open(keyword_path, encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            tier = row.get("selection_tier")
            if row.get("selected") == "true" and tier in keywords_by_tier:
                keywords_by_tier[tier].append(row["keyword_id"])

    keyword_routes = {}
    for tier, prefix in (
        ("high_confidence", "derived_keywords"),
        ("title_confirmed", "derived_keywords_title_confirmed"),
    ):
        for offset in range(0, len(keywords_by_tier[tier]), keyword_batch_size):
            route_id = f"{prefix}_{offset // keyword_batch_size + 1:03d}"
            for keyword_id in sorted(keywords_by_tier[tier])[
                offset : offset + keyword_batch_size
            ]:
                keyword_routes[keyword_id] = (tier, route_id)
    return topic_groups, keyword_routes


def annotate_routes(work, topic_groups, keyword_routes):
    topic_ids = {
        short_id(topic.get("id"))
        for topic in work.get("topics") or []
        if isinstance(topic, dict) and topic.get("id")
    }
    keyword_ids = {
        short_id(keyword.get("id"))
        for keyword in work.get("keywords") or []
        if isinstance(keyword, dict) and keyword.get("id")
    }
    grouped_topics = {
        group: sorted(
            topic_id for topic_id in topic_ids if topic_groups.get(topic_id) == group
        )
        for group in (EXPLICIT_QUANTUM, INTRINSICALLY_QUANTUM, BROAD_RELATED)
    }
    high_keywords = sorted(
        keyword_id
        for keyword_id in keyword_ids
        if keyword_routes.get(keyword_id, (None,))[0] == "high_confidence"
    )
    title_keywords = sorted(
        keyword_id
        for keyword_id in keyword_ids
        if keyword_routes.get(keyword_id, (None,))[0] == "title_confirmed"
    )
    route_ids = [f"topic_{topic_id}" for topic_id in topic_ids & set(topic_groups)]
    route_ids.extend(
        keyword_routes[keyword_id][1] for keyword_id in high_keywords + title_keywords
    )
    route_ids = sorted(set(route_ids))
    if not route_ids:
        return None
    return {
        **work,
        "retrieval_sources": ["openalex_source_reconciliation"],
        "retrieval_route_ids": route_ids,
        "matched_topic_ids": sorted(topic_ids & set(topic_groups)),
        "matched_explicit_quantum_topic_ids": grouped_topics[EXPLICIT_QUANTUM],
        "matched_intrinsically_quantum_topic_ids": grouped_topics[
            INTRINSICALLY_QUANTUM
        ],
        "matched_broad_related_topic_ids": grouped_topics[BROAD_RELATED],
        "matched_derived_keyword_ids": high_keywords,
        "matched_title_confirmed_derived_keyword_ids": title_keywords,
    }


def has_independent_paper_evidence(work, term_patterns, container_patterns):
    title = work.get("display_name") or ""
    if any(pattern.search(title) for pattern in container_patterns):
        return False
    abstract = reconstruct_abstract(work.get("abstract_inverted_index"))
    return any(
        pattern.search(title) or pattern.search(abstract) for pattern in term_patterns
    )


def reconcile_sources(
    api_key,
    scope_path=SCOPE_CONFIG,
    filter_config_path=FILTER_CONFIG,
    topic_path=TOPIC_CANDIDATES,
    keyword_path=DERIVED_KEYWORDS,
    union_path=CANDIDATE_UNION,
    union_manifest_path=CANDIDATE_MANIFEST,
    historical_path=HISTORICAL_WORKS,
    output_path=OUTPUT,
    audit_path=AUDIT_OUTPUT,
    manifest_path=MANIFEST_OUTPUT,
    fetch_json=fetch_page,
    mailto="",
):
    scope = load_json(scope_path)
    production = scope["year_windows"]["production"]
    authorship_codes, institution_codes = scope_country_codes(scope)
    union_manifest = load_json(union_manifest_path)
    if (union_manifest.get("artifact") or {}).get("sha256") != file_sha256(union_path):
        raise RuntimeError("Candidate union hash does not match its manifest")
    if union_manifest.get("schema_version") != 3:
        raise RuntimeError(
            "Source reconciliation requires the two-tier candidate union"
        )

    union_ids, union_dois, union_titles = read_union_index(union_path)
    historical_available = bool(historical_path) and os.path.exists(historical_path)
    if historical_available:
        historical = read_historical_in_scope(
            historical_path,
            production["start"],
            production["end"],
            authorship_codes,
            institution_codes,
        )
    else:
        # The historical corpus is an optional import from an earlier crawl. Without it
        # there is no prior record to compare against, so the step reconciles nothing
        # rather than failing the pipeline.
        print(f"No historical corpus at {historical_path}; reconciling nothing.")
        historical = {}
    missing = {
        work_id: work
        for work_id, work in historical.items()
        if work_id not in union_ids
    }
    current = fetch_by_ids(missing, api_key, fetch_json, mailto=mailto)
    topic_groups, keyword_routes = load_route_vocabulary(topic_path, keyword_path, 100)
    filter_config = load_json(filter_config_path)
    term_patterns = [
        re.compile(pattern, flags=re.IGNORECASE)
        for pattern in (
            filter_config["paper_level_term_patterns"]
            + filter_config.get("paper_level_only_term_patterns", [])
        )
    ]
    container_patterns = [
        re.compile(pattern, flags=re.IGNORECASE)
        for pattern in filter_config["container_title_patterns"]
    ]

    restored = []
    audit_rows = []
    for historical_id, old_work in sorted(missing.items()):
        current_work = current.get(historical_id)
        regression_reasons = []
        if current_work is None:
            regression_reasons.append("current OpenAlex ID disappeared")
        else:
            if not has_country(current_work, authorship_codes, institution_codes):
                regression_reasons.append(
                    "current record lost in-scope affiliation country"
                )
            year = current_work.get("publication_year")
            if not isinstance(year, int):
                regression_reasons.append("current record lost publication year")
            elif not production["start"] <= year <= production["end"]:
                regression_reasons.append("current publication year moved out of scope")
        if not regression_reasons:
            continue

        alternatives = fetch_reconciliation_candidates(
            old_work,
            api_key,
            authorship_codes,
            institution_codes,
            fetch_json,
            mailto=mailto,
        )
        resolved = alternatives[0] if alternatives else current_work or old_work
        resolved_id = short_id(resolved.get("id")) or historical_id
        doi = normalize_doi(resolved.get("doi") or old_work.get("doi"))
        title = normalize_title(
            resolved.get("display_name") or old_work.get("display_name")
        )
        already_present = (
            resolved_id in union_ids
            or bool(doi and doi in union_dois)
            or bool(title and title in union_titles)
        )
        restored_fields = []
        if not already_present:
            resolved = dict(resolved)
            if not has_country(resolved, authorship_codes, institution_codes):
                resolved["authorships"] = old_work.get("authorships") or []
                restored_fields.append("authorships")
            if not isinstance(resolved.get("publication_year"), int):
                resolved["publication_year"] = old_work.get("publication_year")
                resolved["publication_date"] = old_work.get("publication_date")
                restored_fields.extend(["publication_year", "publication_date"])
            annotated = annotate_routes(resolved, topic_groups, keyword_routes)
            independent_evidence = has_independent_paper_evidence(
                resolved, term_patterns, container_patterns
            )
            if annotated and independent_evidence:
                annotated["source_reconciliation"] = {
                    "historical_work_id": historical_id,
                    "resolved_work_id": resolved_id,
                    "regression_reasons": regression_reasons,
                    "restored_fields": restored_fields,
                    "historical_source": historical_path,
                }
                restored.append(annotated)
                status = "restored_candidate"
            elif not annotated:
                status = "regression_found_but_no_current_route"
            else:
                status = "regression_found_but_no_independent_paper_evidence"
        else:
            annotated = None
            status = "canonical_or_duplicate_already_present"
        audit_rows.append(
            {
                "historical_work_id": historical_id,
                "resolved_work_id": resolved_id,
                "doi": resolved.get("doi") or old_work.get("doi") or "",
                "title": resolved.get("display_name")
                or old_work.get("display_name")
                or "",
                "status": status,
                "regression_reasons": " | ".join(regression_reasons),
                "restored_fields": " | ".join(restored_fields),
                "matched_routes": " | ".join(
                    (annotated or {}).get("retrieval_route_ids") or []
                ),
            }
        )

    restored.sort(key=lambda work: work["id"])
    for path in (output_path, audit_path, manifest_path):
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    temporary = {
        path: path + ".tmp" for path in (output_path, audit_path, manifest_path)
    }
    try:
        with open(temporary[output_path], "w", encoding="utf-8") as handle:
            handle.writelines(
                json.dumps(work, ensure_ascii=False, sort_keys=True) + "\n"
                for work in restored
            )
        with open(temporary[audit_path], "w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=AUDIT_FIELDS)
            writer.writeheader()
            writer.writerows(audit_rows)
        manifest = {
            "schema_version": 1,
            "created_at": dt.datetime.now(dt.timezone.utc).isoformat(),
            "method": "historical_id_then_doi_and_normalized_title_source_reconciliation",
            "inputs": {
                path: file_sha256(path)
                for path in (
                    scope_path,
                    filter_config_path,
                    topic_path,
                    keyword_path,
                    union_path,
                    union_manifest_path,
                )
                + ((historical_path,) if historical_available else ())
            },
            "historical_source": historical_path if historical_available else "",
            "counts": {
                "historical_in_scope_unique_works": len(historical),
                "historical_works_missing_from_candidate_union": len(missing),
                "source_regressions_found": len(audit_rows),
                "restored_candidates": len(restored),
                "canonical_or_duplicate_already_present": sum(
                    row["status"] == "canonical_or_duplicate_already_present"
                    for row in audit_rows
                ),
                "regressions_without_current_route": sum(
                    row["status"] == "regression_found_but_no_current_route"
                    for row in audit_rows
                ),
                "regressions_without_independent_paper_evidence": sum(
                    row["status"]
                    == "regression_found_but_no_independent_paper_evidence"
                    for row in audit_rows
                ),
            },
            "artifacts": {
                "reconciled_candidates": {
                    "path": output_path,
                    "work_count": len(restored),
                    "sha256": file_sha256(temporary[output_path]),
                },
                "audit": {
                    "path": audit_path,
                    "row_count": len(audit_rows),
                    "sha256": file_sha256(temporary[audit_path]),
                },
            },
        }
        with open(temporary[manifest_path], "w", encoding="utf-8") as handle:
            json.dump(manifest, handle, ensure_ascii=False, indent=2, sort_keys=True)
            handle.write("\n")
        for path in (output_path, audit_path, manifest_path):
            os.replace(temporary[path], path)
        return manifest
    except Exception:
        for path in temporary.values():
            try:
                os.remove(path)
            except FileNotFoundError:
                pass
        raise


def main():
    os.chdir(os.path.dirname(os.path.abspath(__file__)) + "/..")
    api_key = settings.openalex_api_key
    mailto = settings.openalex_mailto
    if not api_key and not mailto:
        raise SystemExit(
            "Set OPENALEX_API_KEY or OPENALEX_MAILTO before reconciling OpenAlex "
            "sources."
        )
    manifest = reconcile_sources(api_key, mailto=mailto)
    counts = manifest["counts"]
    print(f"Source regressions found: {counts['source_regressions_found']}")
    print(f"Reconciled candidates:    {counts['restored_candidates']}")
    print(
        "Canonical duplicates already present: "
        f"{counts['canonical_or_duplicate_already_present']}"
    )


if __name__ == "__main__":
    main()
