"""Build the high-precision explicitly-quantum core set."""

import csv
import datetime as dt
import hashlib
import json
import os
import re

from utils.abstract_utils import reconstruct_abstract
from utils.config_digest import json_digest
from utils.openalex_utils import file_sha256
from utils.quantum_topic_groups import EXPLICIT_QUANTUM, candidate_group

SCOPE_CONFIG = "config/sweden_quantum_scope.json"
CORE_CONFIG = "config/sweden_quantum_core_and_outliers.json"
TOPIC_CANDIDATES = "data/sweden_quantum_topic_candidates.csv"
CANDIDATE_INPUT = "data/sweden_quantum_candidate_union.jsonl"
CANDIDATE_MANIFEST = "data/sweden_quantum_candidate_union_manifest.json"

CORE_OUTPUT = "data/sweden_quantum_core_works.jsonl"
FEATURES_OUTPUT = "data/sweden_quantum_candidate_features.jsonl"
SUMMARY_OUTPUT = "data/sweden_quantum_core_summary.csv"
MANIFEST_OUTPUT = "data/sweden_quantum_core_manifest.json"

SUMMARY_FIELDS = ("group", "label", "core_works", "candidate_works", "core_share")


def load_json(path):
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def short_id(value):
    return (value or "").rsplit("/", 1)[-1]


def validate_config(config):
    if config.get("schema_version") != 1:
        raise ValueError("Unsupported core/outlier schema_version")
    rule = config.get("core_rule") or {}
    if rule.get("require_explicit_quantum_primary_topic") is not True:
        raise ValueError("The core rule must require an explicit-quantum primary topic")
    score = rule.get("minimum_primary_topic_score")
    if not isinstance(score, (int, float)) or not 0.0 <= score <= 1.0:
        raise ValueError("minimum_primary_topic_score must be a fraction")
    if not rule.get("core_term_patterns"):
        raise ValueError("The core rule must define core_term_patterns")
    fields = set(rule.get("core_term_fields") or [])
    if not fields or not fields <= {"title", "abstract"}:
        raise ValueError("Core term fields must be title and/or abstract only")
    excluded = rule.get("excluded_work_types")
    if not isinstance(excluded, list) or any(
        not isinstance(item, str) for item in excluded
    ):
        raise ValueError("excluded_work_types must be a list of OpenAlex work types")
    for pattern in rule["core_term_patterns"] + list(
        rule.get("ambiguous_term_patterns") or []
    ):
        re.compile(pattern, flags=re.IGNORECASE)


def verify_candidate_union(input_path, manifest_path):
    manifest = load_json(manifest_path)
    artifact = manifest.get("artifact") or {}
    if artifact.get("path") != input_path:
        raise RuntimeError("Candidate-union manifest points to an unexpected input")
    if artifact.get("sha256") != file_sha256(input_path):
        raise RuntimeError("Candidate-union input hash does not match its manifest")
    return manifest


def explicit_quantum_topic_ids(path):
    with open(path, newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    if not rows:
        raise RuntimeError("Topic candidate inventory is empty")
    ids = {row["topic_id"] for row in rows if candidate_group(row) == EXPLICIT_QUANTUM}
    if not ids:
        raise RuntimeError("Topic inventory contains no explicit-quantum topics")
    names = {row["topic_id"]: row["topic_name"] for row in rows}
    return ids, names


def compile_patterns(patterns):
    return [re.compile(pattern, flags=re.IGNORECASE) for pattern in patterns]


def matched_terms(text, patterns):
    return sorted(
        {
            match.group(0).casefold()
            for pattern in patterns
            for match in pattern.finditer(text)
        },
        key=str.casefold,
    )


def core_decision(row, explicit_ids, rule, core_patterns, ambiguous_patterns):
    """Return (is_core, reasons) for one slim candidate row."""
    reasons = []
    if row["type"] in set(rule["excluded_work_types"]):
        reasons.append(f"work type {row['type']} carries no science of its own")
    if row["primary_topic_id"] not in explicit_ids:
        reasons.append("primary topic is not an explicit quantum topic")
    score = row["primary_topic_score"]
    if not isinstance(score, (int, float)):
        reasons.append("primary topic has no score")
    elif score < rule["minimum_primary_topic_score"]:
        reasons.append(
            f"primary topic score {score:.3f} is below "
            f"{rule['minimum_primary_topic_score']}"
        )
    own_text = "\n".join(row[field] for field in sorted(rule["core_term_fields"]))
    terms = matched_terms(own_text, core_patterns)
    if not terms:
        reasons.append("no explicit quantum term in the work's own title or abstract")
    else:
        ambiguous = matched_terms(own_text, ambiguous_patterns)
        remaining = [
            term for term in terms if not any(term in phrase for phrase in ambiguous)
        ]
        if ambiguous and not remaining:
            reasons.append(
                "the only quantum evidence is " + ", ".join(sorted(set(ambiguous)))
            )
    is_core = not reasons
    if is_core:
        reasons = [
            f"work type {row['type']} reports its own science",
            "explicit quantum primary topic",
            "primary topic score at or above the configured minimum",
            "explicit quantum term in the work's own "
            + "/".join(sorted(rule["core_term_fields"])),
        ]
    return is_core, reasons, terms


def slim_row(work, explicit_ids):
    primary = work.get("primary_topic") or {}
    topics = []
    for topic in work.get("topics") or []:
        if not isinstance(topic, dict):
            continue
        topic_id = short_id(topic.get("id"))
        score = topic.get("score")
        if topic_id and isinstance(score, (int, float)):
            topics.append([topic_id, round(float(score), 6)])
    return {
        "work_id": short_id(work.get("id")),
        "publication_year": work.get("publication_year"),
        "title": work.get("display_name") or "",
        "abstract": reconstruct_abstract(work.get("abstract_inverted_index")),
        "keywords": [
            keyword.get("display_name") or ""
            for keyword in work.get("keywords") or []
            if isinstance(keyword, dict)
        ],
        "primary_topic_id": short_id(primary.get("id")),
        "primary_topic_name": primary.get("display_name") or "",
        "primary_topic_score": primary.get("score"),
        "primary_topic_domain": ((primary.get("domain") or {}).get("display_name"))
        or "",
        "primary_topic_field": ((primary.get("field") or {}).get("display_name")) or "",
        "topics": topics,
        "matched_explicit_quantum_topic_ids": work.get(
            "matched_explicit_quantum_topic_ids"
        )
        or [],
        "matched_broad_related_topic_ids": work.get("matched_broad_related_topic_ids")
        or [],
        "matched_derived_keyword_ids": work.get("matched_derived_keyword_ids") or [],
        "retrieval_sources": work.get("retrieval_sources") or [],
        "type": work.get("type") or "",
        "language": work.get("language") or "",
        "cited_by_count": work.get("cited_by_count"),
        "is_retracted": bool(work.get("is_retracted")),
        "has_abstract": bool(work.get("abstract_inverted_index")),
        "has_explicit_quantum_primary_topic": short_id(primary.get("id"))
        in explicit_ids,
    }


def iter_works(path):
    with open(path, encoding="utf-8") as f:
        for line_number, line in enumerate(f, start=1):
            try:
                yield json.loads(line)
            except json.JSONDecodeError as exc:
                raise RuntimeError(
                    f"Invalid JSON in {path} line {line_number}"
                ) from exc


def write_summary(path, counters, topic_names):
    rows = []
    core_by_topic, candidates_by_topic, core_by_year, candidates_by_year = counters
    for topic_id, count in sorted(core_by_topic.items(), key=lambda item: -item[1]):
        rows.append(
            {
                "group": "primary_topic",
                "label": f"{topic_id} {topic_names.get(topic_id, '')}".strip(),
                "core_works": count,
                "candidate_works": candidates_by_topic.get(topic_id, 0),
                "core_share": round(
                    count / max(candidates_by_topic.get(topic_id, 1), 1), 4
                ),
            }
        )
    for year in sorted(candidates_by_year):
        rows.append(
            {
                "group": "publication_year",
                "label": str(year),
                "core_works": core_by_year.get(year, 0),
                "candidate_works": candidates_by_year[year],
                "core_share": round(
                    core_by_year.get(year, 0) / candidates_by_year[year], 4
                ),
            }
        )
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=SUMMARY_FIELDS)
        writer.writeheader()
        writer.writerows(rows)


def build_core(
    input_path=CANDIDATE_INPUT,
    manifest_path=CANDIDATE_MANIFEST,
    scope_path=SCOPE_CONFIG,
    config_path=CORE_CONFIG,
    topics_path=TOPIC_CANDIDATES,
    core_path=CORE_OUTPUT,
    features_path=FEATURES_OUTPUT,
    summary_path=SUMMARY_OUTPUT,
    manifest_output=MANIFEST_OUTPUT,
):
    scope = load_json(scope_path)
    config = load_json(config_path)
    validate_config(config)
    union_manifest = verify_candidate_union(input_path, manifest_path)
    explicit_ids, topic_names = explicit_quantum_topic_ids(topics_path)
    rule = config["core_rule"]
    core_patterns = compile_patterns(rule["core_term_patterns"])
    ambiguous_patterns = compile_patterns(rule.get("ambiguous_term_patterns") or [])
    production = scope["year_windows"]["production"]

    counts = {
        "candidate_works": 0,
        "outside_production_window": 0,
        "core_works": 0,
        "core_without_abstract": 0,
        "rejected_no_explicit_primary_topic": 0,
        "rejected_low_primary_topic_score": 0,
        "rejected_no_own_text_term": 0,
        "rejected_ambiguous_term_only": 0,
        "diagnostic_any_explicit_topic_and_own_text": 0,
    }
    core_by_topic = {}
    candidates_by_topic = {}
    core_by_year = {}
    candidates_by_year = {}
    # Hashes only the identifiers and texts that an embedding step would read, so a
    # changed core rule does not needlessly invalidate expensive GPU embeddings.
    text_hash = hashlib.sha256()

    tmp = {
        path: path + ".tmp"
        for path in (core_path, features_path, summary_path, manifest_output)
    }
    for path in tmp:
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    try:
        with (
            open(tmp[core_path], "w", encoding="utf-8") as core_file,
            open(tmp[features_path], "w", encoding="utf-8") as features_file,
        ):
            for work in iter_works(input_path):
                year = work.get("publication_year")
                if not isinstance(year, int) or not (
                    production["start"] <= year <= production["end"]
                ):
                    counts["outside_production_window"] += 1
                    continue
                counts["candidate_works"] += 1
                row = slim_row(work, explicit_ids)
                if not row["work_id"]:
                    raise RuntimeError("Candidate work is missing an OpenAlex ID")
                is_core, reasons, terms = core_decision(
                    row, explicit_ids, rule, core_patterns, ambiguous_patterns
                )
                text_hash.update(
                    json.dumps(
                        [row["work_id"], row["title"], row["abstract"]],
                        ensure_ascii=False,
                    ).encode("utf-8")
                )
                row["is_core"] = is_core
                row["core_reasons"] = reasons
                row["core_terms"] = terms
                features_file.write(json.dumps(row, ensure_ascii=False) + "\n")

                candidates_by_year[year] = candidates_by_year.get(year, 0) + 1
                topic_id = row["primary_topic_id"] or "(none)"
                candidates_by_topic[topic_id] = candidates_by_topic.get(topic_id, 0) + 1
                if row["matched_explicit_quantum_topic_ids"] and terms:
                    counts["diagnostic_any_explicit_topic_and_own_text"] += 1
                if is_core:
                    counts["core_works"] += 1
                    if not row["has_abstract"]:
                        counts["core_without_abstract"] += 1
                    core_by_year[year] = core_by_year.get(year, 0) + 1
                    core_by_topic[topic_id] = core_by_topic.get(topic_id, 0) + 1
                    core_file.write(
                        json.dumps(
                            {
                                **work,
                                "is_core": True,
                                "core_reasons": reasons,
                                "core_terms": terms,
                            },
                            ensure_ascii=False,
                            sort_keys=True,
                        )
                        + "\n"
                    )
                    continue
                for reason in reasons:
                    if reason.startswith("primary topic is not"):
                        counts["rejected_no_explicit_primary_topic"] += 1
                    elif reason.startswith("primary topic score") or reason.startswith(
                        "primary topic has no score"
                    ):
                        counts["rejected_low_primary_topic_score"] += 1
                    elif reason.startswith("no explicit quantum term"):
                        counts["rejected_no_own_text_term"] += 1
                    elif reason.startswith("the only quantum evidence"):
                        counts["rejected_ambiguous_term_only"] += 1

        write_summary(
            tmp[summary_path],
            (core_by_topic, candidates_by_topic, core_by_year, candidates_by_year),
            topic_names,
        )
        manifest = {
            "schema_version": 1,
            "created_at": dt.datetime.now(dt.timezone.utc).isoformat(),
            "core_rule": rule,
            "explicit_quantum_topic_ids": sorted(explicit_ids),
            "input": {
                "path": input_path,
                "sha256": file_sha256(input_path),
                "source_manifest": manifest_path,
                "source_manifest_sha256": file_sha256(manifest_path),
                "source_manifest_created_at": union_manifest.get("created_at"),
            },
            "scope_config": scope_path,
            "scope_config_sha256": file_sha256(scope_path),
            "core_config": config_path,
            "core_config_sha256": file_sha256(config_path),
            "core_rule_sha256": json_digest(rule),
            "topic_inventory": topics_path,
            "topic_inventory_sha256": file_sha256(topics_path),
            "production_years": production,
            "counts": counts,
            "artifacts": {
                "core_works": {
                    "path": core_path,
                    "work_count": counts["core_works"],
                    "sha256": file_sha256(tmp[core_path]),
                },
                "candidate_features": {
                    "path": features_path,
                    "row_count": counts["candidate_works"],
                    "sha256": file_sha256(tmp[features_path]),
                    "candidate_text_sha256": text_hash.hexdigest(),
                },
                "core_summary": {
                    "path": summary_path,
                    "sha256": file_sha256(tmp[summary_path]),
                },
            },
        }
        with open(tmp[manifest_output], "w", encoding="utf-8") as f:
            json.dump(manifest, f, ensure_ascii=False, indent=2, sort_keys=True)
            f.write("\n")
        for path in (core_path, features_path, summary_path, manifest_output):
            os.replace(tmp[path], path)
        return manifest
    except Exception:
        for path in tmp.values():
            try:
                os.remove(path)
            except FileNotFoundError:
                pass
        raise


def main():
    os.chdir(os.path.dirname(os.path.abspath(__file__)) + "/..")
    manifest = build_core()
    counts = manifest["counts"]
    print(f"Candidate works:      {counts['candidate_works']}")
    print(f"Core works:           {counts['core_works']}")
    print(f"  without abstract:   {counts['core_without_abstract']}")
    print("Rejection evidence (a work can fail several checks):")
    print(
        f"  no explicit primary topic:   {counts['rejected_no_explicit_primary_topic']}"
    )
    print(
        f"  low primary topic score:     {counts['rejected_low_primary_topic_score']}"
    )
    print(f"  no term in title/abstract:   {counts['rejected_no_own_text_term']}")
    print(f"  ambiguous term only:         {counts['rejected_ambiguous_term_only']}")
    print(
        "Diagnostic wider core (any explicit topic + own text term): "
        f"{counts['diagnostic_any_explicit_topic_and_own_text']}"
    )
    print(f"Core:     {CORE_OUTPUT}")
    print(f"Features: {FEATURES_OUTPUT}")
    print(f"Summary:  {SUMMARY_OUTPUT}")
    print(f"Manifest: {MANIFEST_OUTPUT}")


if __name__ == "__main__":
    main()
