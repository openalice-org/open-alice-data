"""Download and deduplicate the 1950–2026 Swedish quantum candidate union."""

import collections
import csv
import datetime as dt
import hashlib
import json
import os
import re

from utils.openalex_utils import (
    BASE_URL,
    fetch_page,
    file_sha256,
    join_filters,
    paginate,
    work_count,
)
from utils.quantum_topic_groups import (
    BROAD_RELATED,
    EXPLICIT_QUANTUM,
    INTRINSICALLY_QUANTUM,
    candidate_group,
)
from utils.settings import settings

SCOPE_CONFIG = "config/sweden_quantum_scope.json"
KEYWORD_CONFIG = "config/sweden_quantum_keyword_derivation.json"
PAPER_FILTER_CONFIG = "config/sweden_quantum_paper_filter.json"
CANDIDATE_FILE = "data/sweden_quantum_topic_candidates.csv"
CANDIDATE_MANIFEST = "data/sweden_quantum_topic_candidates_manifest.json"
DERIVED_KEYWORDS = "data/sweden_quantum_derived_keywords.csv"
DERIVED_KEYWORDS_MANIFEST = "data/sweden_quantum_derived_keywords_manifest.json"
OUTPUT = "data/sweden_quantum_candidate_union.jsonl"
MANIFEST = "data/sweden_quantum_candidate_union_manifest.json"
BATCH_DIR = "data/sweden_quantum_candidate_routes"

DERIVED_KEYWORD_GROUP = "derived_quantum_keyword"
TITLE_CONFIRMED_DERIVED_KEYWORD_GROUP = "derived_quantum_keyword_title_confirmed"
KEYWORD_TIERS = ("high_confidence", "title_confirmed")
DERIVED_KEYWORD_FIELDS = (
    "keyword_id",
    "keyword_name",
    "explicit_seed_work_count",
    "topic_universe_work_count",
    "explicit_share_pct",
    "explicit_topic_count",
    "explicit_topic_ids",
    "name_matches_quantum_pattern",
    "identifier_matches_quantum_pattern",
    "selection_tier",
    "selected",
    "decision",
)

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


def load_json(path):
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def short_id(value):
    return (value or "").rsplit("/", 1)[-1]


def _validate_keyword_config(config):
    if config.get("schema_version") != 2:
        raise ValueError("Unsupported keyword-derivation schema_version")
    if config.get("seed_topic_group") != EXPLICIT_QUANTUM:
        raise ValueError("Keyword derivation must use explicit_quantum seed topics")
    if config.get("source_field") != "keywords":
        raise ValueError("Keyword derivation must use OpenAlex paper keywords")
    tiers = config.get("selection_tiers") or {}
    if tuple(tiers) != KEYWORD_TIERS:
        raise ValueError(
            "Keyword derivation must define high_confidence and title_confirmed tiers"
        )
    expected_rules = {
        "high_confidence": "automatic",
        "title_confirmed": "title_quantum_pattern_required",
    }
    for tier_name, thresholds in tiers.items():
        if not isinstance(thresholds.get("minimum_explicit_seed_works"), int) or (
            thresholds["minimum_explicit_seed_works"] < 1
        ):
            raise ValueError(
                f"{tier_name} minimum_explicit_seed_works must be a positive integer"
            )
        if not isinstance(thresholds.get("minimum_explicit_topics"), int) or (
            thresholds["minimum_explicit_topics"] < 1
        ):
            raise ValueError(
                f"{tier_name} minimum_explicit_topics must be a positive integer"
            )
        explicit_share = thresholds.get("minimum_explicit_share_of_topic_universe")
        if not isinstance(explicit_share, (int, float)) or not 0 <= explicit_share <= 1:
            raise ValueError(
                f"{tier_name} minimum_explicit_share_of_topic_universe must be 0–1"
            )
        if thresholds.get("require_quantum_pattern_match_in_display_name") is not True:
            raise ValueError(
                "Derived keyword names must require a quantum-pattern match"
            )
        if thresholds.get("require_quantum_pattern_match_in_identifier") is not True:
            raise ValueError("Derived keyword IDs must require a quantum-pattern match")
        if thresholds.get("inclusion_rule") != expected_rules[tier_name]:
            raise ValueError(f"Unexpected inclusion rule for {tier_name}")
    high = tiers["high_confidence"]
    lower = tiers["title_confirmed"]
    for key in (
        "minimum_explicit_seed_works",
        "minimum_explicit_topics",
        "minimum_explicit_share_of_topic_universe",
    ):
        if lower[key] > high[key]:
            raise ValueError("title_confirmed thresholds cannot exceed high_confidence")
    batch_size = config.get("keyword_ids_per_route")
    if not isinstance(batch_size, int) or not 1 <= batch_size <= 100:
        raise ValueError("keyword_ids_per_route must be between 1 and 100")


def _validate_filter_config(config):
    if config.get("schema_version") != 2:
        raise ValueError("Unsupported paper-filter schema_version")
    patterns = config.get("paper_level_term_patterns") or []
    if not patterns or any(
        not isinstance(pattern, str) or not pattern for pattern in patterns
    ):
        raise ValueError("Paper filter must provide quantum-specific term patterns")
    for pattern in patterns:
        re.compile(pattern, flags=re.IGNORECASE)
    for pattern in config.get("paper_level_only_term_patterns") or []:
        re.compile(pattern, flags=re.IGNORECASE)


def load_verified_inputs(
    scope_path=SCOPE_CONFIG,
    keyword_config_path=KEYWORD_CONFIG,
    filter_config_path=PAPER_FILTER_CONFIG,
    candidate_path=CANDIDATE_FILE,
    candidate_manifest_path=CANDIDATE_MANIFEST,
):
    scope = load_json(scope_path)
    if scope.get("schema_version") != 1:
        raise ValueError("Unsupported scope schema_version")
    production = (scope.get("year_windows") or {}).get("production") or {}
    if (
        not isinstance(production.get("start"), int)
        or not isinstance(production.get("end"), int)
        or production["start"] > production["end"]
    ):
        raise ValueError(
            "Scope production must define a valid integer start/end window"
        )
    country = (scope.get("country_evidence") or {}).get("primary") or {}
    if not country.get("filter_field") or not country.get("filter_value"):
        raise ValueError("Scope must define the primary country filter")

    keyword_config = load_json(keyword_config_path)
    _validate_keyword_config(keyword_config)
    filter_config = load_json(filter_config_path)
    _validate_filter_config(filter_config)

    candidate_manifest = load_json(candidate_manifest_path)
    if candidate_manifest.get("output_sha256") != file_sha256(candidate_path):
        raise RuntimeError("Topic candidate CSV hash does not match its manifest")
    with open(candidate_path, encoding="utf-8", newline="") as f:
        candidates = list(csv.DictReader(f))
    topic_ids = [candidate.get("topic_id") for candidate in candidates]
    if any(not topic_id for topic_id in topic_ids) or len(topic_ids) != len(
        set(topic_ids)
    ):
        raise RuntimeError("Topic candidate IDs must be unique and non-empty")
    return scope, keyword_config, filter_config, candidates, candidate_manifest


def _base_filter(scope):
    production = scope["year_windows"]["production"]
    country = scope["country_evidence"]["primary"]
    return join_filters(
        f"{country['filter_field']}:{country['filter_value']}",
        f"publication_year:{production['start']}-{production['end']}",
    )


def build_topic_routes(scope, candidates):
    base_filter = _base_filter(scope)
    routes = []
    for candidate in candidates:
        if candidate.get("taxonomy_gate") == "blocked":
            continue
        topic_id = candidate["topic_id"]
        routes.append(
            {
                "route_id": f"topic_{topic_id}",
                "query_channel": "topic",
                "query_name": candidate.get("topic_name") or topic_id,
                "filter": join_filters(base_filter, f"topics.id:{topic_id}"),
                "search": "",
                "search_parameter": "",
                "topic_id": topic_id,
                "candidate_group": candidate_group(candidate),
            }
        )
    return routes


def _keyword_routes_for_tier(
    scope,
    selected_keywords,
    batch_size,
    tier_name,
    route_prefix,
    query_channel,
    candidate_group_name,
):
    keyword_ids = sorted(
        row["keyword_id"]
        for row in selected_keywords
        if row["selection_tier"] == tier_name
    )
    routes = []
    for offset in range(0, len(keyword_ids), batch_size):
        batch = keyword_ids[offset : offset + batch_size]
        route_number = offset // batch_size + 1
        routes.append(
            {
                "route_id": f"{route_prefix}_{route_number:03d}",
                "query_channel": query_channel,
                "query_name": f"Derived quantum paper keywords, batch {route_number}",
                "filter": join_filters(
                    _base_filter(scope), "keywords.id:" + "|".join(batch)
                ),
                "search": "",
                "search_parameter": "",
                "topic_id": "",
                "keyword_ids": batch,
                "candidate_group": candidate_group_name,
            }
        )
    return routes


def build_keyword_routes(scope, selected_keywords, batch_size):
    # Keep the original high-confidence route ID stable so its verified batch remains
    # reusable when the conditional tier is added.
    return _keyword_routes_for_tier(
        scope,
        selected_keywords,
        batch_size,
        "high_confidence",
        "derived_keywords",
        "derived_keyword",
        DERIVED_KEYWORD_GROUP,
    ) + _keyword_routes_for_tier(
        scope,
        selected_keywords,
        batch_size,
        "title_confirmed",
        "derived_keywords_title_confirmed",
        "derived_keyword_title_confirmed",
        TITLE_CONFIRMED_DERIVED_KEYWORD_GROUP,
    )


def _validate_route_ids(routes):
    route_ids = [route["route_id"] for route in routes]
    if len(route_ids) != len(set(route_ids)):
        raise RuntimeError("Route IDs must be globally unique")
    if any(not re.fullmatch(r"[A-Za-z0-9_-]+", route_id) for route_id in route_ids):
        raise RuntimeError(
            "Route IDs may contain only letters, digits, underscores, and hyphens"
        )


def _read_valid_part(part_path, manifest_path, route, expected_count, select):
    try:
        manifest = load_json(manifest_path)
        actual_hash = file_sha256(part_path)
    except (FileNotFoundError, json.JSONDecodeError, OSError):
        return None
    if (
        manifest.get("route") != route
        or manifest.get("expected_count") != expected_count
        or manifest.get("select") != select
        or manifest.get("output_sha256") != actual_hash
    ):
        return None
    return manifest


def _write_route_part(
    route,
    expected_count,
    select,
    api_key,
    part_path,
    part_manifest_path,
    fetch_json,
    mailto="",
):
    part_tmp = part_path + ".tmp"
    manifest_tmp = part_manifest_path + ".tmp"
    try:
        downloaded = 0
        with open(part_tmp, "w", encoding="utf-8") as out:
            for work in paginate(
                route["filter"],
                select,
                api_key,
                label=route["route_id"],
                search=route["search"] or None,
                search_parameter=route["search_parameter"] or "search",
                fetch_json=fetch_json,
                mailto=mailto,
            ):
                out.write(json.dumps(work, ensure_ascii=False, sort_keys=True) + "\n")
                downloaded += 1
        if downloaded != expected_count:
            raise RuntimeError(
                f"{route['route_id']}: counted {expected_count} but retrieved {downloaded}"
            )
        manifest = {
            "schema_version": 1,
            "created_at": dt.datetime.now(dt.timezone.utc).isoformat(),
            "route": route,
            "select": select,
            "expected_count": expected_count,
            "retrieved_count": downloaded,
            "output": part_path,
            "output_sha256": file_sha256(part_tmp),
        }
        with open(manifest_tmp, "w", encoding="utf-8") as f:
            json.dump(manifest, f, ensure_ascii=False, indent=2, sort_keys=True)
            f.write("\n")
        os.replace(part_tmp, part_path)
        os.replace(manifest_tmp, part_manifest_path)
        return manifest
    except Exception:
        for path in (part_tmp, manifest_tmp):
            try:
                os.remove(path)
            except FileNotFoundError:
                pass
        raise


def download_routes(routes, api_key, batch_dir, fetch_json=fetch_page, mailto=""):
    _validate_route_ids(routes)
    manifests = []
    for route in routes:
        expected_count = work_count(
            route["filter"],
            api_key,
            search=route["search"] or None,
            search_parameter=route["search_parameter"] or "search",
            fetch_json=fetch_json,
            mailto=mailto,
        )
        part_path = os.path.join(batch_dir, route["route_id"] + ".jsonl")
        part_manifest_path = os.path.join(
            batch_dir, route["route_id"] + ".manifest.json"
        )
        saved = None
        if os.path.exists(part_path):
            saved = _read_valid_part(
                part_path, part_manifest_path, route, expected_count, SELECT
            )
        if saved:
            print(
                f"  {route['route_id']}: reusing verified {expected_count}-work route"
            )
            manifests.append(saved)
        else:
            manifests.append(
                _write_route_part(
                    route,
                    expected_count,
                    SELECT,
                    api_key,
                    part_path,
                    part_manifest_path,
                    fetch_json,
                    mailto=mailto,
                )
            )
    return manifests


def _topic_route_sources(topic_part_manifests):
    sources = [
        {
            "route_id": part["route"]["route_id"],
            "candidate_group": part["route"]["candidate_group"],
            "work_count": part["retrieved_count"],
            "output_sha256": part["output_sha256"],
        }
        for part in topic_part_manifests
    ]
    sources.sort(key=lambda row: row["route_id"])
    snapshot = hashlib.sha256(
        json.dumps(sources, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    return sources, snapshot


def _keyword_identifier_text(keyword_id):
    """Convert an OpenAlex keyword slug into text suitable for term matching."""
    return " ".join(re.findall(r"[a-z0-9]+", (keyword_id or "").casefold()))


def _read_cached_keyword_inventory(
    output_path,
    manifest_path,
    source_snapshot_id,
    keyword_config_path,
    filter_config_path,
):
    try:
        manifest = load_json(manifest_path)
        if (
            manifest.get("schema_version") != 2
            or manifest.get("source_topic_route_snapshot_id") != source_snapshot_id
            or manifest.get("keyword_derivation_config") != keyword_config_path
            or manifest.get("keyword_derivation_config_sha256")
            != file_sha256(keyword_config_path)
            or manifest.get("paper_filter_config") != filter_config_path
            or manifest.get("paper_filter_config_sha256")
            != file_sha256(filter_config_path)
            or (manifest.get("artifact") or {}).get("path") != output_path
            or (manifest.get("artifact") or {}).get("sha256")
            != file_sha256(output_path)
        ):
            return None, None
        with open(output_path, encoding="utf-8", newline="") as f:
            rows = list(csv.DictReader(f))
    except (FileNotFoundError, json.JSONDecodeError, OSError):
        return None, None
    selected = [row for row in rows if row.get("selected") == "true"]
    if len(rows) != (manifest.get("artifact") or {}).get("row_count"):
        return None, None
    if len(selected) != manifest.get("selected_keyword_count"):
        return None, None
    tier_counts = {
        tier_name: sum(row.get("selection_tier") == tier_name for row in selected)
        for tier_name in KEYWORD_TIERS
    }
    if tier_counts != manifest.get("selected_keyword_count_by_tier"):
        return None, None
    return selected, manifest


def derive_keyword_inventory(
    topic_part_manifests,
    keyword_config,
    filter_config,
    keyword_config_path=KEYWORD_CONFIG,
    filter_config_path=PAPER_FILTER_CONFIG,
    output_path=DERIVED_KEYWORDS,
    manifest_path=DERIVED_KEYWORDS_MANIFEST,
):
    source_routes, source_snapshot_id = _topic_route_sources(topic_part_manifests)
    cached_rows, cached_manifest = _read_cached_keyword_inventory(
        output_path,
        manifest_path,
        source_snapshot_id,
        keyword_config_path,
        filter_config_path,
    )
    if cached_rows is not None:
        print(
            f"  derived keywords: reusing {len(cached_rows)} selected keywords "
            "from verified topic routes"
        )
        return cached_rows, cached_manifest

    universe_works = collections.defaultdict(set)
    explicit_works = collections.defaultdict(set)
    explicit_topics = collections.defaultdict(set)
    name_works = collections.defaultdict(lambda: collections.defaultdict(set))
    for part in topic_part_manifests:
        path = part["output"]
        if file_sha256(path) != part.get("output_sha256"):
            raise RuntimeError(f"Topic route changed before keyword derivation: {path}")
        route = part["route"]
        is_explicit = route["candidate_group"] == EXPLICIT_QUANTUM
        with open(path, encoding="utf-8") as f:
            for line_number, line in enumerate(f, start=1):
                try:
                    work = json.loads(line)
                except json.JSONDecodeError as exc:
                    raise RuntimeError(
                        f"Invalid JSON in {path} line {line_number}"
                    ) from exc
                work_id = work.get("id")
                if not work_id:
                    raise RuntimeError(f"Work without ID in {path}")
                work_keywords = {}
                for keyword in work.get("keywords") or []:
                    if not isinstance(keyword, dict):
                        continue
                    keyword_id = short_id(keyword.get("id"))
                    keyword_name = (keyword.get("display_name") or "").strip()
                    if keyword_id and keyword_name:
                        work_keywords[keyword_id] = keyword_name
                for keyword_id, keyword_name in work_keywords.items():
                    name_works[keyword_id][keyword_name].add(work_id)
                    universe_works[keyword_id].add(work_id)
                    if is_explicit:
                        explicit_works[keyword_id].add(work_id)
                        explicit_topics[keyword_id].add(route["topic_id"])

    selection_tiers = keyword_config["selection_tiers"]
    patterns = [
        re.compile(pattern, flags=re.IGNORECASE)
        for pattern in filter_config["paper_level_term_patterns"]
    ]
    names = {
        keyword_id: sorted(
            variants,
            key=lambda name: (-len(variants[name]), name.casefold(), name),
        )[0]
        for keyword_id, variants in name_works.items()
    }
    rows = []
    for keyword_id, seed_work_ids in explicit_works.items():
        seed_count = len(seed_work_ids)
        universe_count = len(universe_works[keyword_id])
        topic_count = len(explicit_topics[keyword_id])
        explicit_share = seed_count / universe_count if universe_count else 0
        name_pattern_match = any(
            pattern.search(names[keyword_id]) for pattern in patterns
        )
        identifier_text = _keyword_identifier_text(keyword_id)
        identifier_pattern_match = any(
            pattern.search(identifier_text) for pattern in patterns
        )
        failures_by_tier = {}
        for tier_name, thresholds in selection_tiers.items():
            failures = []
            if not name_pattern_match:
                failures.append("no quantum-specific pattern in keyword name")
            if not identifier_pattern_match:
                failures.append("no quantum-specific pattern in keyword identifier")
            if seed_count < thresholds["minimum_explicit_seed_works"]:
                failures.append("below minimum explicit seed-work count")
            if topic_count < thresholds["minimum_explicit_topics"]:
                failures.append("below minimum explicit-topic coverage")
            if explicit_share < thresholds["minimum_explicit_share_of_topic_universe"]:
                failures.append("below minimum explicit share of topic universe")
            failures_by_tier[tier_name] = failures
        selection_tier = next(
            (
                tier_name
                for tier_name in KEYWORD_TIERS
                if not failures_by_tier[tier_name]
            ),
            "",
        )
        selected = bool(selection_tier)
        decision_failures = failures_by_tier["title_confirmed"]
        rows.append(
            {
                "keyword_id": keyword_id,
                "keyword_name": names[keyword_id],
                "explicit_seed_work_count": seed_count,
                "topic_universe_work_count": universe_count,
                "explicit_share_pct": f"{100 * explicit_share:.1f}",
                "explicit_topic_count": topic_count,
                "explicit_topic_ids": " | ".join(sorted(explicit_topics[keyword_id])),
                "name_matches_quantum_pattern": str(name_pattern_match).lower(),
                "identifier_matches_quantum_pattern": str(
                    identifier_pattern_match
                ).lower(),
                "selection_tier": selection_tier,
                "selected": str(selected).lower(),
                "decision": (
                    f"selected:{selection_tier}"
                    if selected
                    else "; ".join(decision_failures)
                ),
            }
        )
    rows.sort(
        key=lambda row: (
            KEYWORD_TIERS.index(row["selection_tier"])
            if row["selection_tier"] in KEYWORD_TIERS
            else len(KEYWORD_TIERS),
            -int(row["explicit_seed_work_count"]),
            row["keyword_name"].casefold(),
        )
    )
    selected_rows = [row for row in rows if row["selected"] == "true"]
    if not selected_rows:
        raise RuntimeError("Keyword derivation selected no supplemental keyword routes")

    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    os.makedirs(os.path.dirname(manifest_path) or ".", exist_ok=True)
    output_tmp = output_path + ".tmp"
    manifest_tmp = manifest_path + ".tmp"
    try:
        with open(output_tmp, "w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=DERIVED_KEYWORD_FIELDS)
            writer.writeheader()
            writer.writerows(rows)
        manifest = {
            "schema_version": 2,
            "created_at": dt.datetime.now(dt.timezone.utc).isoformat(),
            "method": "paper_keywords_derived_from_explicit_quantum_topic_works",
            "source_topic_route_snapshot_id": source_snapshot_id,
            "source_topic_routes": source_routes,
            "keyword_derivation_config": keyword_config_path,
            "keyword_derivation_config_sha256": file_sha256(keyword_config_path),
            "paper_filter_config": filter_config_path,
            "paper_filter_config_sha256": file_sha256(filter_config_path),
            "explicit_topic_count": sum(
                source["candidate_group"] == EXPLICIT_QUANTUM
                for source in source_routes
            ),
            "observed_explicit_keyword_count": len(rows),
            "selected_keyword_count": len(selected_rows),
            "selected_keyword_count_by_tier": {
                tier_name: sum(
                    row["selection_tier"] == tier_name for row in selected_rows
                )
                for tier_name in KEYWORD_TIERS
            },
            "artifact": {
                "path": output_path,
                "row_count": len(rows),
                "sha256": file_sha256(output_tmp),
            },
        }
        with open(manifest_tmp, "w", encoding="utf-8") as f:
            json.dump(manifest, f, ensure_ascii=False, indent=2, sort_keys=True)
            f.write("\n")
        os.replace(output_tmp, output_path)
        os.replace(manifest_tmp, manifest_path)
        return selected_rows, manifest
    except Exception:
        for path in (output_tmp, manifest_tmp):
            try:
                os.remove(path)
            except FileNotFoundError:
                pass
        raise


def _preferred_work(current, incoming):
    if (incoming.get("updated_date") or "") > (current.get("updated_date") or ""):
        return incoming
    return current


def _annotate_work(work, route_hits):
    topic_hits = [hit for hit in route_hits if hit["query_channel"] == "topic"]
    keyword_hits = [
        hit for hit in route_hits if hit["query_channel"] == "derived_keyword"
    ]
    title_confirmed_keyword_hits = [
        hit
        for hit in route_hits
        if hit["query_channel"] == "derived_keyword_title_confirmed"
    ]
    matched_derived_keyword_ids = sorted(
        {
            keyword_id
            for hit in keyword_hits
            for keyword_id in hit["matched_keyword_ids"]
        }
    )
    matched_title_confirmed_keyword_ids = sorted(
        {
            keyword_id
            for hit in title_confirmed_keyword_hits
            for keyword_id in hit["matched_keyword_ids"]
        }
    )
    if keyword_hits and not matched_derived_keyword_ids:
        raise RuntimeError(
            f"Derived-keyword route returned {work.get('id')} without a selected keyword"
        )
    if title_confirmed_keyword_hits and not matched_title_confirmed_keyword_ids:
        raise RuntimeError(
            f"Conditional derived-keyword route returned {work.get('id')} without "
            "a selected keyword"
        )
    return {
        **work,
        "retrieval_sources": sorted(
            {
                "openalex_topic" if topic_hits else "",
                "openalex_derived_keyword"
                if keyword_hits or title_confirmed_keyword_hits
                else "",
            }
            - {""}
        ),
        "retrieval_route_ids": sorted(hit["route_id"] for hit in route_hits),
        "matched_topic_ids": sorted(hit["topic_id"] for hit in topic_hits),
        "matched_explicit_quantum_topic_ids": sorted(
            hit["topic_id"]
            for hit in topic_hits
            if hit["candidate_group"] == EXPLICIT_QUANTUM
        ),
        "matched_intrinsically_quantum_topic_ids": sorted(
            hit["topic_id"]
            for hit in topic_hits
            if hit["candidate_group"] == INTRINSICALLY_QUANTUM
        ),
        "matched_broad_related_topic_ids": sorted(
            hit["topic_id"]
            for hit in topic_hits
            if hit["candidate_group"] == BROAD_RELATED
        ),
        "matched_derived_keyword_ids": matched_derived_keyword_ids,
        "matched_title_confirmed_derived_keyword_ids": (
            matched_title_confirmed_keyword_ids
        ),
    }


def _write_candidate_union(part_manifests, output_tmp):
    """Combine the route parts into one annotated, id-sorted JSONL.

    Memory stays flat: pass 1 streams every part and keeps only a small index per
    work (part, byte offset, line digest, updated_date) plus its route hits; pass 2
    seeks back to each chosen record in id order. Selection rules are unchanged from
    the old in-memory dict: the first copy wins unless a later copy has a strictly
    newer updated_date, and a copy whose content differs counts as a metadata variant.
    Parts are written with sorted keys, so identical works have identical lines.
    """
    index = {}  # work_id -> (part_idx, offset, line digest, updated_date)
    hits = {}  # work_id -> [(part_idx, matched keyword ids or None)]
    metadata_variant_count = 0
    for part_idx, part_manifest in enumerate(part_manifests):
        part_path = part_manifest["output"]
        if file_sha256(part_path) != part_manifest.get("output_sha256"):
            raise RuntimeError(f"Route batch hash changed before combine: {part_path}")
        route = part_manifest["route"]
        derived = route["query_channel"] in {
            "derived_keyword",
            "derived_keyword_title_confirmed",
        }
        route_keyword_ids = set(route["keyword_ids"]) if derived else set()
        with open(part_path, "rb") as f:
            offset = 0
            for line_number, line in enumerate(f, start=1):
                line_offset = offset
                offset += len(line)
                try:
                    work = json.loads(line)
                except json.JSONDecodeError as exc:
                    raise RuntimeError(
                        f"Invalid JSON in {part_path} line {line_number}"
                    ) from exc
                work_id = work.get("id")
                if not work_id:
                    raise RuntimeError(f"Work without ID in {part_path}")
                digest = hashlib.blake2b(line.rstrip(b"\n"), digest_size=16).digest()
                updated = work.get("updated_date") or ""
                known = index.get(work_id)
                if known is None:
                    index[work_id] = (part_idx, line_offset, digest, updated)
                elif known[2] != digest:
                    metadata_variant_count += 1
                    if updated > known[3]:
                        index[work_id] = (part_idx, line_offset, digest, updated)
                matched = None
                if derived:
                    work_keyword_ids = {
                        short_id(keyword.get("id"))
                        for keyword in work.get("keywords") or []
                        if isinstance(keyword, dict) and keyword.get("id")
                    }
                    matched = tuple(sorted(work_keyword_ids & route_keyword_ids))
                    if not matched:
                        raise RuntimeError(
                            f"Derived-keyword route returned {work_id} without a "
                            "selected keyword"
                        )
                entry = (part_idx, matched)
                work_hits = hits.setdefault(work_id, [])
                if entry not in work_hits:
                    work_hits.append(entry)

    handles = {}
    try:
        with open(output_tmp, "w", encoding="utf-8") as out:
            for work_id in sorted(index):
                part_idx, offset, _, _ = index[work_id]
                handle = handles.get(part_idx)
                if handle is None:
                    if len(handles) >= 200:
                        for open_handle in handles.values():
                            open_handle.close()
                        handles.clear()
                    handle = handles[part_idx] = open(
                        part_manifests[part_idx]["output"], "rb"
                    )
                handle.seek(offset)
                work = json.loads(handle.readline())
                route_hits = []
                for hit_part_idx, matched in hits[work_id]:
                    route = part_manifests[hit_part_idx]["route"]
                    route_hits.append(
                        route
                        if matched is None
                        else {**route, "matched_keyword_ids": list(matched)}
                    )
                annotated = _annotate_work(work, route_hits)
                out.write(json.dumps(annotated, ensure_ascii=False, sort_keys=True) + "\n")
    finally:
        for open_handle in handles.values():
            open_handle.close()
    return len(index), metadata_variant_count


def download_candidate_union(
    api_key,
    scope_path=SCOPE_CONFIG,
    keyword_config_path=KEYWORD_CONFIG,
    filter_config_path=PAPER_FILTER_CONFIG,
    candidate_path=CANDIDATE_FILE,
    candidate_manifest_path=CANDIDATE_MANIFEST,
    derived_keywords_path=DERIVED_KEYWORDS,
    derived_keywords_manifest_path=DERIVED_KEYWORDS_MANIFEST,
    output_path=OUTPUT,
    manifest_path=MANIFEST,
    batch_dir=BATCH_DIR,
    fetch_json=fetch_page,
    mailto="",
):
    (
        scope,
        keyword_config,
        filter_config,
        candidates,
        candidate_manifest,
    ) = load_verified_inputs(
        scope_path,
        keyword_config_path,
        filter_config_path,
        candidate_path,
        candidate_manifest_path,
    )
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    os.makedirs(os.path.dirname(manifest_path) or ".", exist_ok=True)
    os.makedirs(batch_dir, exist_ok=True)

    topic_routes = build_topic_routes(scope, candidates)
    topic_part_manifests = download_routes(
        topic_routes, api_key, batch_dir, fetch_json, mailto=mailto
    )
    selected_keywords, keyword_manifest = derive_keyword_inventory(
        topic_part_manifests,
        keyword_config,
        filter_config,
        keyword_config_path,
        filter_config_path,
        derived_keywords_path,
        derived_keywords_manifest_path,
    )
    keyword_routes = build_keyword_routes(
        scope, selected_keywords, keyword_config["keyword_ids_per_route"]
    )
    keyword_part_manifests = download_routes(
        keyword_routes, api_key, batch_dir, fetch_json, mailto=mailto
    )
    routes = topic_routes + keyword_routes
    _validate_route_ids(routes)
    part_manifests = topic_part_manifests + keyword_part_manifests

    output_tmp = output_path + ".tmp"
    manifest_tmp = manifest_path + ".tmp"
    try:
        unique_work_count, metadata_variant_count = _write_candidate_union(
            part_manifests, output_tmp
        )
        manifest = {
            "schema_version": 3,
            "created_at": dt.datetime.now(dt.timezone.utc).isoformat(),
            "endpoint": BASE_URL,
            "api_authentication": (
                "OPENALEX_API_KEY supplied at runtime; value omitted"
                if api_key
                else "OPENALEX_MAILTO polite pool supplied at runtime; value omitted"
            ),
            "scope_config": scope_path,
            "scope_config_sha256": file_sha256(scope_path),
            "keyword_derivation_config": keyword_config_path,
            "keyword_derivation_config_sha256": file_sha256(keyword_config_path),
            "paper_filter_config": filter_config_path,
            "paper_filter_config_sha256": file_sha256(filter_config_path),
            "derived_keywords": derived_keywords_path,
            "derived_keywords_sha256": file_sha256(derived_keywords_path),
            "derived_keywords_manifest": derived_keywords_manifest_path,
            "derived_keywords_manifest_sha256": file_sha256(
                derived_keywords_manifest_path
            ),
            "derived_keyword_source_snapshot_id": keyword_manifest[
                "source_topic_route_snapshot_id"
            ],
            "topic_candidates": candidate_path,
            "topic_candidates_sha256": file_sha256(candidate_path),
            "topic_candidates_snapshot_id": candidate_manifest.get("input_snapshot_id"),
            "production_years": scope["year_windows"]["production"],
            "country_rule": scope["country_evidence"]["primary"],
            "route_count": len(routes),
            "topic_route_count": len(topic_routes),
            "derived_keyword_route_count": len(keyword_routes),
            "selected_derived_keyword_count": len(selected_keywords),
            "selected_derived_keyword_count_by_tier": keyword_manifest[
                "selected_keyword_count_by_tier"
            ],
            "derived_keyword_route_count_by_tier": {
                "high_confidence": sum(
                    route["query_channel"] == "derived_keyword"
                    for route in keyword_routes
                ),
                "title_confirmed": sum(
                    route["query_channel"] == "derived_keyword_title_confirmed"
                    for route in keyword_routes
                ),
            },
            "summed_route_work_count": sum(
                part["expected_count"] for part in part_manifests
            ),
            "unique_candidate_work_count": unique_work_count,
            "metadata_variant_count": metadata_variant_count,
            "batch_dir": batch_dir,
            "routes": part_manifests,
            "artifact": {
                "path": output_path,
                "work_count": unique_work_count,
                "sha256": file_sha256(output_tmp),
            },
        }
        with open(manifest_tmp, "w", encoding="utf-8") as f:
            json.dump(manifest, f, ensure_ascii=False, indent=2, sort_keys=True)
            f.write("\n")
        os.replace(output_tmp, output_path)
        os.replace(manifest_tmp, manifest_path)
        return manifest
    except Exception:
        for path in (output_tmp, manifest_tmp):
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
            "Set OPENALEX_API_KEY or OPENALEX_MAILTO before downloading Swedish "
            "candidates."
        )
    manifest = download_candidate_union(api_key, mailto=mailto)
    print(
        f"Saved {manifest['unique_candidate_work_count']} unique candidate works "
        f"from {manifest['topic_route_count']} topic routes and "
        f"{manifest['selected_derived_keyword_count']} derived paper keywords to {OUTPUT}"
    )
    print(f"Derived keyword audit: {DERIVED_KEYWORDS}")
    print(f"Manifest: {MANIFEST}")
    print(f"Resumable route batches: {BATCH_DIR}")


if __name__ == "__main__":
    main()
