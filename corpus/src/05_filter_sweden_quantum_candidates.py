"""Apply the lightweight topic-guided inclusion rule to candidate works."""

import csv
import datetime as dt
import json
import os
import re

from utils.abstract_utils import reconstruct_abstract
from utils.openalex_utils import file_sha256
from utils.quantum_topic_groups import (
    BROAD_RELATED,
    EXPLICIT_QUANTUM,
    INTRINSICALLY_QUANTUM,
)

SCOPE_CONFIG = "config/sweden_quantum_scope.json"
FILTER_CONFIG = "config/sweden_quantum_paper_filter.json"
KEYWORD_CONFIG = "config/sweden_quantum_keyword_derivation.json"
RECORD_GATES_CONFIG = "config/sweden_quantum_record_gates.json"
CANDIDATE_INPUT = "data/sweden_quantum_candidate_union.jsonl"
CANDIDATE_MANIFEST = "data/sweden_quantum_candidate_union_manifest.json"
RECONCILIATION_INPUT = "data/sweden_quantum_source_reconciliations.jsonl"
RECONCILIATION_MANIFEST = "data/sweden_quantum_source_reconciliations_manifest.json"
OUTPUT = "data/sweden_quantum_works.jsonl"
AUDIT_OUTPUT = "data/sweden_quantum_candidate_audit.csv"
MANIFEST_OUTPUT = "data/sweden_quantum_works_manifest.json"

AUDIT_FIELDS = (
    "work_id",
    "doi",
    "publication_year",
    "title",
    "included",
    "inclusion_reasons",
    "language",
    "work_type",
    "paper_level_quantum_terms",
    "title_level_quantum_terms",
    "strong_title_abstract_quantum_terms",
    "quantum_keyword_evidence_count",
    "primary_topic_domain",
    "keyword_only_blocked_primary_domain",
    "keyword_only_domain_exception_passed",
    "container_title",
    "matched_explicit_quantum_topic_ids",
    "matched_intrinsically_quantum_topic_ids",
    "matched_broad_related_topic_ids",
    "matched_derived_keyword_ids",
    "matched_title_confirmed_derived_keyword_ids",
)


def load_json(path):
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def short_id(value):
    return (value or "").rsplit("/", 1)[-1]


def validate_record_gates(gates):
    """The gates decide what kind of record can enter the corpus at all."""
    if not isinstance(gates, dict):
        raise ValueError("Record gates must be a configuration object")
    if gates.get("schema_version") != 1:
        raise ValueError("Unsupported record-gate schema_version")
    languages = gates.get("accepted_languages")
    if not languages or any(
        not isinstance(code, str) or code != code.lower() or not code
        for code in languages
    ):
        raise ValueError(
            "Record gates must accept at least one lowercase language code"
        )
    if not isinstance(gates.get("accept_missing_language"), bool):
        raise ValueError("Record gates must say whether a missing language is accepted")
    work_types = gates.get("excluded_work_types")
    if not isinstance(work_types, list) or any(
        not isinstance(name, str) or not name for name in work_types
    ):
        raise ValueError("Record-gate excluded_work_types must be a list of work types")


def validate_configs(scope, config, gates=None):
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
    policy = scope.get("inclusion_policy") or {}
    if policy.get("manual_review_required") is not False:
        raise ValueError("Topic-guided inclusion must not require manual review")
    if policy.get("trained_classifier_required") is not False:
        raise ValueError("Topic-guided inclusion must not require a trained classifier")
    if config.get("schema_version") != 2:
        raise ValueError("Unsupported paper-filter schema_version")
    automatic = config.get("automatic_acceptance") or {}
    if automatic != {
        EXPLICIT_QUANTUM: True,
        INTRINSICALLY_QUANTUM: True,
        "derived_quantum_keyword": True,
    }:
        raise ValueError("Paper-filter automatic acceptance rules are incomplete")
    required_groups = set(config.get("paper_term_required_for") or [])
    if required_groups != {BROAD_RELATED}:
        raise ValueError("Paper-filter broad topic groups are incomplete")
    conditional = (config.get("conditional_acceptance") or {}).get(
        "derived_quantum_keyword_title_confirmed"
    ) or {}
    if conditional != {
        "required_field": "title",
        "required_evidence": "paper_level_quantum_term",
        "reject_container_titles": True,
    }:
        raise ValueError("Conditional derived-keyword acceptance rule is incomplete")
    safeguard = config.get("derived_keyword_only_domain_safeguard") or {}
    if set(safeguard.get("domains") or []) != {"Social Sciences"}:
        raise ValueError(
            "Derived-keyword-only safeguarded domains must contain Social Sciences"
        )
    minimum_keywords = safeguard.get("minimum_quantum_keyword_count_for_exception")
    if not isinstance(minimum_keywords, int) or minimum_keywords < 2:
        raise ValueError("Domain exception must require at least two quantum keywords")
    patterns = (config.get("paper_level_term_patterns") or []) + (
        config.get("paper_level_only_term_patterns") or []
    )
    if not patterns or any(
        not isinstance(pattern, str) or not pattern for pattern in patterns
    ):
        raise ValueError("Paper filter must define non-empty term patterns")
    for pattern in patterns:
        re.compile(pattern, flags=re.IGNORECASE)
    for key in (
        "strong_title_abstract_patterns_for_exception",
        "container_title_patterns",
    ):
        values = safeguard.get(key) if key.startswith("strong_") else config.get(key)
        if not values or any(
            not isinstance(pattern, str) or not pattern for pattern in values
        ):
            raise ValueError(f"Paper filter must define non-empty {key}")
        for pattern in values:
            re.compile(pattern, flags=re.IGNORECASE)
    if gates is not None:
        validate_record_gates(gates)


def compile_patterns(config):
    return [
        re.compile(pattern, flags=re.IGNORECASE)
        for pattern in (
            config["paper_level_term_patterns"]
            + config.get("paper_level_only_term_patterns", [])
        )
    ]


def compile_domain_exception_patterns(config):
    safeguard = config["derived_keyword_only_domain_safeguard"]
    return [
        re.compile(pattern, flags=re.IGNORECASE)
        for pattern in safeguard["strong_title_abstract_patterns_for_exception"]
    ]


def compile_container_patterns(config):
    return [
        re.compile(pattern, flags=re.IGNORECASE)
        for pattern in config["container_title_patterns"]
    ]


def terms_in_text(text, patterns):
    return sorted(
        {
            match.group(0).casefold()
            for pattern in patterns
            for match in pattern.finditer(text or "")
        },
        key=str.casefold,
    )


def paper_level_terms(work, patterns):
    title = work.get("display_name") or ""
    abstract = reconstruct_abstract(work.get("abstract_inverted_index"))
    keywords = "\n".join(
        keyword.get("display_name") or ""
        for keyword in work.get("keywords") or []
        if isinstance(keyword, dict)
    )
    text = "\n".join((title, abstract, keywords))
    return terms_in_text(text, patterns)


def quantum_keyword_ids(work, patterns):
    matches = []
    for keyword in work.get("keywords") or []:
        if not isinstance(keyword, dict):
            continue
        keyword_id = short_id(keyword.get("id"))
        keyword_name = keyword.get("display_name") or ""
        identifier_text = " ".join(re.findall(r"[a-z0-9]+", keyword_id.casefold()))
        if (
            keyword_id
            and any(pattern.search(keyword_name) for pattern in patterns)
            and (any(pattern.search(identifier_text) for pattern in patterns))
        ):
            matches.append(keyword_id)
    return sorted(set(matches))


def iter_candidates(path):
    with open(path, encoding="utf-8") as f:
        for line_number, line in enumerate(f, start=1):
            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                raise RuntimeError(
                    f"Invalid JSON in {path} line {line_number}"
                ) from exc
            yield record


def primary_topic_domain(work):
    return (
        ((work.get("primary_topic") or {}).get("domain") or {}).get("display_name")
    ) or ""


def language_gate_failure(work, gates):
    """Why the recorded publication language keeps this work out, or an empty string."""
    if not gates:
        return ""
    language = (work.get("language") or "").lower()
    if not language:
        if gates["accept_missing_language"]:
            return ""
        return "no publication language recorded"
    if language in set(gates["accepted_languages"]):
        return ""
    return f"publication language {language} is not accepted"


def work_type_gate_failure(work, gates):
    """Why the OpenAlex work type keeps this work out, or an empty string."""
    if not gates:
        return ""
    work_type = (work.get("type") or "").lower()
    if work_type and work_type in set(gates["excluded_work_types"]):
        return f"work type {work_type} is not a paper"
    return ""


def inclusion_decision(
    work,
    patterns,
    domain_safeguard=None,
    strong_title_abstract_patterns=(),
    container_patterns=(),
    gates=None,
):
    explicit_topics = work.get("matched_explicit_quantum_topic_ids") or []
    intrinsic_topics = work.get("matched_intrinsically_quantum_topic_ids") or []
    broad_topics = work.get("matched_broad_related_topic_ids") or []
    derived_keywords = work.get("matched_derived_keyword_ids") or []
    title_confirmed_keywords = (
        work.get("matched_title_confirmed_derived_keyword_ids") or []
    )
    if not (
        explicit_topics
        or intrinsic_topics
        or broad_topics
        or derived_keywords
        or title_confirmed_keywords
    ):
        raise RuntimeError(f"Candidate {work.get('id')} has no retrieval provenance")
    title = work.get("display_name") or ""
    abstract = reconstruct_abstract(work.get("abstract_inverted_index"))
    terms = paper_level_terms(work, patterns)
    title_terms = terms_in_text(title, patterns)
    strong_terms = terms_in_text(
        "\n".join((title, abstract)), strong_title_abstract_patterns
    )
    quantum_keyword_matches = quantum_keyword_ids(work, patterns)
    container_title = any(pattern.search(title) for pattern in container_patterns)
    domain = primary_topic_domain(work)
    all_derived_keywords = bool(derived_keywords or title_confirmed_keywords)
    keyword_only = all_derived_keywords and not (
        explicit_topics or intrinsic_topics or broad_topics
    )
    safeguard = domain_safeguard or {}
    safeguarded_domain = keyword_only and domain in set(safeguard.get("domains") or [])
    domain_exception_passed = bool(
        safeguarded_domain
        and strong_terms
        and len(quantum_keyword_matches)
        >= safeguard.get("minimum_quantum_keyword_count_for_exception", 2)
    )
    keyword_only_blocked = safeguarded_domain and not domain_exception_passed
    accepted_derived_keyword = bool(derived_keywords) and not keyword_only_blocked
    accepted_title_confirmed_keyword = bool(
        title_confirmed_keywords
        and title_terms
        and not container_title
        and not keyword_only_blocked
    )
    reasons = []
    if explicit_topics:
        reasons.append("matched an explicit quantum topic")
    if intrinsic_topics:
        reasons.append("matched an intrinsically quantum topic")
    if accepted_derived_keyword:
        reasons.append(
            "matched a high-confidence OpenAlex paper keyword derived from "
            "explicit quantum topics"
        )
    if accepted_title_confirmed_keyword:
        reasons.append(
            "lower-confidence derived keyword confirmed independently by the paper title"
        )
    if domain_exception_passed:
        reasons.append(
            f"derived-keyword-only {domain} assignment overridden by a strong "
            "title/abstract signal and multiple quantum paper keywords"
        )
    if keyword_only_blocked:
        reasons.append(
            "derived-keyword-only work failed the multi-signal safeguard for "
            f"primary topic domain {domain}"
        )
    if title_confirmed_keywords and not title_terms:
        reasons.append(
            "lower-confidence derived keyword lacked an independent title-level "
            "quantum term"
        )
    if title_confirmed_keywords and container_title:
        reasons.append(
            "lower-confidence derived keyword matched a journal/program container title"
        )
    if terms and broad_topics:
        reasons.append(
            "broad-related topic route confirmed by a paper-level quantum term"
        )
    rejected_language = language_gate_failure(work, gates)
    rejected_work_type = work_type_gate_failure(work, gates)
    gate_failures = [
        reason for reason in (rejected_language, rejected_work_type) if reason
    ]
    reasons.extend(gate_failures)
    included = (
        bool(
            explicit_topics
            or intrinsic_topics
            or accepted_derived_keyword
            or accepted_title_confirmed_keyword
            or (terms and broad_topics)
        )
        and not gate_failures
    )
    if not included and broad_topics and not terms:
        reasons.append(
            "matched only a broad-related topic and had no paper-level quantum term"
        )
    return {
        "quantum_included": included,
        "quantum_inclusion_reasons": reasons,
        "paper_level_quantum_terms": terms,
        "title_level_quantum_terms": title_terms,
        "strong_title_abstract_quantum_terms": strong_terms,
        "quantum_keyword_evidence_ids": quantum_keyword_matches,
        "primary_topic_domain": domain,
        "keyword_only_blocked_primary_domain": keyword_only_blocked,
        "keyword_only_domain_exception_passed": domain_exception_passed,
        "container_title": container_title,
        "record_gate_failures": gate_failures,
        "rejected_language": bool(rejected_language),
        "rejected_work_type": bool(rejected_work_type),
        "language": work.get("language") or "",
        "work_type": work.get("type") or "",
    }


def _verify_source(
    input_path,
    source_manifest_path,
    scope_path,
    filter_config_path,
    keyword_config_path=KEYWORD_CONFIG,
):
    manifest = load_json(source_manifest_path)
    if manifest.get("schema_version") != 3:
        raise RuntimeError("Candidate union does not use two-tier schema version 3")
    artifact = manifest.get("artifact") or {}
    if artifact.get("path") != input_path:
        raise RuntimeError("Candidate-union manifest points to an unexpected input")
    if artifact.get("sha256") != file_sha256(input_path):
        raise RuntimeError("Candidate-union input hash does not match its manifest")
    if manifest.get("scope_config") != scope_path:
        raise RuntimeError(
            "Candidate union points to an unexpected scope configuration"
        )
    if manifest.get("scope_config_sha256") != file_sha256(scope_path):
        raise RuntimeError("Candidate union was built from a different scope")
    if manifest.get("paper_filter_config") != filter_config_path or (
        manifest.get("paper_filter_config_sha256") != file_sha256(filter_config_path)
    ):
        raise RuntimeError(
            "Candidate union was built from different paper-term patterns"
        )
    if manifest.get("keyword_derivation_config") != keyword_config_path or (
        manifest.get("keyword_derivation_config_sha256")
        != file_sha256(keyword_config_path)
    ):
        raise RuntimeError("Candidate union was built from different keyword rules")


def _verify_reconciliation(
    reconciliation_path,
    reconciliation_manifest_path,
    input_path,
    source_manifest_path,
):
    manifest = load_json(reconciliation_manifest_path)
    if manifest.get("schema_version") != 1:
        raise RuntimeError("Unsupported source-reconciliation schema version")
    artifact = (manifest.get("artifacts") or {}).get("reconciled_candidates") or {}
    if artifact.get("path") != reconciliation_path or (
        artifact.get("sha256") != file_sha256(reconciliation_path)
    ):
        raise RuntimeError("Source-reconciliation artifact does not match its manifest")
    inputs = manifest.get("inputs") or {}
    if inputs.get(input_path) != file_sha256(input_path) or (
        inputs.get(source_manifest_path) != file_sha256(source_manifest_path)
    ):
        raise RuntimeError(
            "Source reconciliation was built from a different candidate union"
        )
    return manifest


def _audit_row(work, result):
    return {
        "work_id": short_id(work.get("id")),
        "doi": work.get("doi") or "",
        "publication_year": work.get("publication_year") or "",
        "title": work.get("display_name") or "",
        "included": str(result["quantum_included"]).lower(),
        "inclusion_reasons": " | ".join(result["quantum_inclusion_reasons"]),
        "language": result["language"],
        "work_type": result["work_type"],
        "paper_level_quantum_terms": " | ".join(result["paper_level_quantum_terms"]),
        "title_level_quantum_terms": " | ".join(result["title_level_quantum_terms"]),
        "strong_title_abstract_quantum_terms": " | ".join(
            result["strong_title_abstract_quantum_terms"]
        ),
        "quantum_keyword_evidence_count": len(result["quantum_keyword_evidence_ids"]),
        "primary_topic_domain": result["primary_topic_domain"],
        "keyword_only_blocked_primary_domain": str(
            result["keyword_only_blocked_primary_domain"]
        ).lower(),
        "keyword_only_domain_exception_passed": str(
            result["keyword_only_domain_exception_passed"]
        ).lower(),
        "container_title": str(result["container_title"]).lower(),
        "matched_explicit_quantum_topic_ids": " | ".join(
            work.get("matched_explicit_quantum_topic_ids") or []
        ),
        "matched_intrinsically_quantum_topic_ids": " | ".join(
            work.get("matched_intrinsically_quantum_topic_ids") or []
        ),
        "matched_broad_related_topic_ids": " | ".join(
            work.get("matched_broad_related_topic_ids") or []
        ),
        "matched_derived_keyword_ids": " | ".join(
            work.get("matched_derived_keyword_ids") or []
        ),
        "matched_title_confirmed_derived_keyword_ids": " | ".join(
            work.get("matched_title_confirmed_derived_keyword_ids") or []
        ),
    }


def filter_candidates(
    input_path,
    source_manifest_path,
    output_path,
    audit_path,
    manifest_path,
    run_kind="production",
    scope_path=SCOPE_CONFIG,
    filter_config_path=FILTER_CONFIG,
    keyword_config_path=KEYWORD_CONFIG,
    gates_config_path=RECORD_GATES_CONFIG,
    reconciliation_path=None,
    reconciliation_manifest_path=None,
):
    scope = load_json(scope_path)
    config = load_json(filter_config_path)
    gates = load_json(gates_config_path)
    validate_configs(scope, config, gates)
    if run_kind != "production":
        raise ValueError("Only the production keyword-derived workflow is supported")
    _verify_source(
        input_path,
        source_manifest_path,
        scope_path,
        filter_config_path,
        keyword_config_path,
    )
    reconciliation_manifest = None
    if bool(reconciliation_path) != bool(reconciliation_manifest_path):
        raise ValueError("Provide both reconciliation path and manifest, or neither")
    if reconciliation_path:
        reconciliation_manifest = _verify_reconciliation(
            reconciliation_path,
            reconciliation_manifest_path,
            input_path,
            source_manifest_path,
        )
    patterns = compile_patterns(config)
    domain_safeguard = config["derived_keyword_only_domain_safeguard"]
    strong_patterns = compile_domain_exception_patterns(config)
    container_patterns = compile_container_patterns(config)
    production = scope["year_windows"]["production"]
    seen_ids = set()
    candidate_count = 0
    included_count = 0
    excluded_count = 0
    excluded_broad_topic_only_count = 0
    excluded_language_count = 0
    excluded_work_type_count = 0
    excluded_keyword_only_blocked_domain_count = 0
    excluded_title_confirmed_without_title_count = 0
    included_keyword_only_domain_exception_count = 0
    outside_window_count = 0
    reconciled_candidate_count = 0
    reconciled_included_count = 0
    inclusion_channels = {
        EXPLICIT_QUANTUM: 0,
        INTRINSICALLY_QUANTUM: 0,
        "derived_keyword": 0,
        "title_confirmed_derived_keyword": 0,
        "broad_related_paper_term_confirmation": 0,
    }

    for path in (output_path, audit_path, manifest_path):
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    tmp = {
        output_path: output_path + ".tmp",
        audit_path: audit_path + ".tmp",
        manifest_path: manifest_path + ".tmp",
    }
    try:
        with (
            open(tmp[output_path], "w", encoding="utf-8") as out,
            open(tmp[audit_path], "w", encoding="utf-8", newline="") as audit_file,
        ):
            writer = csv.DictWriter(audit_file, fieldnames=AUDIT_FIELDS)
            writer.writeheader()
            candidate_sources = [(input_path, False)]
            if reconciliation_path:
                candidate_sources.append((reconciliation_path, True))
            for candidate_source, is_reconciled in candidate_sources:
                for work in iter_candidates(candidate_source):
                    work_id = work.get("id")
                    if not work_id or work_id in seen_ids:
                        raise RuntimeError(
                            "Candidate work IDs must be unique and non-empty"
                        )
                    seen_ids.add(work_id)
                    year = work.get("publication_year")
                    if not isinstance(year, int) or not (
                        production["start"] <= year <= production["end"]
                    ):
                        outside_window_count += 1
                        continue
                    candidate_count += 1
                    if is_reconciled:
                        reconciled_candidate_count += 1
                    result = inclusion_decision(
                        work,
                        patterns,
                        domain_safeguard,
                        strong_patterns,
                        container_patterns,
                        gates,
                    )
                    writer.writerow(_audit_row(work, result))
                    if not result["quantum_included"]:
                        excluded_count += 1
                        if result["rejected_language"]:
                            excluded_language_count += 1
                        if result["rejected_work_type"]:
                            excluded_work_type_count += 1
                        # The counters below attribute an exclusion to the topic and
                        # keyword rule, which never got to speak for a gated record.
                        if result["record_gate_failures"]:
                            continue
                        if result["keyword_only_blocked_primary_domain"]:
                            excluded_keyword_only_blocked_domain_count += 1
                        if (
                            work.get("matched_title_confirmed_derived_keyword_ids")
                            and not result["title_level_quantum_terms"]
                        ):
                            excluded_title_confirmed_without_title_count += 1
                        if not result[
                            "keyword_only_blocked_primary_domain"
                        ] and not work.get(
                            "matched_title_confirmed_derived_keyword_ids"
                        ):
                            excluded_broad_topic_only_count += 1
                        continue
                    included_count += 1
                    if is_reconciled:
                        reconciled_included_count += 1
                    if result["keyword_only_domain_exception_passed"]:
                        included_keyword_only_domain_exception_count += 1
                    if work.get("matched_explicit_quantum_topic_ids"):
                        inclusion_channels[EXPLICIT_QUANTUM] += 1
                    if work.get("matched_intrinsically_quantum_topic_ids"):
                        inclusion_channels[INTRINSICALLY_QUANTUM] += 1
                    if work.get("matched_derived_keyword_ids"):
                        inclusion_channels["derived_keyword"] += 1
                    if (
                        work.get("matched_title_confirmed_derived_keyword_ids")
                        and result["title_level_quantum_terms"]
                    ):
                        inclusion_channels["title_confirmed_derived_keyword"] += 1
                    if result["paper_level_quantum_terms"] and work.get(
                        "matched_broad_related_topic_ids"
                    ):
                        inclusion_channels["broad_related_paper_term_confirmation"] += 1
                    included = {
                        **work,
                        **result,
                    }
                    out.write(
                        json.dumps(included, ensure_ascii=False, sort_keys=True) + "\n"
                    )

        manifest = {
            "schema_version": 3,
            "created_at": dt.datetime.now(dt.timezone.utc).isoformat(),
            "run_kind": run_kind,
            "method": "topic_and_derived_keyword_deterministic_inclusion_no_classifier",
            "definition": config["definition"],
            "production_years": production,
            "input": {
                "path": input_path,
                "sha256": file_sha256(input_path),
                "source_manifest": source_manifest_path,
                "source_manifest_sha256": file_sha256(source_manifest_path),
                "source_reconciliation": reconciliation_path or "",
                "source_reconciliation_sha256": (
                    file_sha256(reconciliation_path) if reconciliation_path else ""
                ),
                "source_reconciliation_manifest": (reconciliation_manifest_path or ""),
                "source_reconciliation_manifest_sha256": (
                    file_sha256(reconciliation_manifest_path)
                    if reconciliation_manifest_path
                    else ""
                ),
            },
            "scope_config": scope_path,
            "scope_config_sha256": file_sha256(scope_path),
            "filter_config": filter_config_path,
            "filter_config_sha256": file_sha256(filter_config_path),
            "keyword_derivation_config": keyword_config_path,
            "keyword_derivation_config_sha256": file_sha256(keyword_config_path),
            "record_gates_config": gates_config_path,
            "record_gates_config_sha256": file_sha256(gates_config_path),
            "counts": {
                "candidate_works": candidate_count,
                "included_works": included_count,
                "excluded_works": excluded_count,
                "excluded_broad_topic_only": excluded_broad_topic_only_count,
                "excluded_non_accepted_language": excluded_language_count,
                "excluded_non_paper_work_type": excluded_work_type_count,
                "excluded_keyword_only_blocked_domain": (
                    excluded_keyword_only_blocked_domain_count
                ),
                "excluded_title_confirmed_keyword_without_title_evidence": (
                    excluded_title_confirmed_without_title_count
                ),
                "included_keyword_only_domain_exception": (
                    included_keyword_only_domain_exception_count
                ),
                "outside_production_window": outside_window_count,
                "reconciled_candidates": reconciled_candidate_count,
                "reconciled_included_works": reconciled_included_count,
                "inclusion_channels_overlap": inclusion_channels,
            },
            "artifacts": {
                "included_works": {
                    "path": output_path,
                    "work_count": included_count,
                    "sha256": file_sha256(tmp[output_path]),
                },
                "candidate_audit": {
                    "path": audit_path,
                    "row_count": candidate_count,
                    "sha256": file_sha256(tmp[audit_path]),
                },
            },
        }
        with open(tmp[manifest_path], "w", encoding="utf-8") as f:
            json.dump(manifest, f, ensure_ascii=False, indent=2, sort_keys=True)
            f.write("\n")
        for path in (output_path, audit_path, manifest_path):
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
    manifest = filter_candidates(
        CANDIDATE_INPUT,
        CANDIDATE_MANIFEST,
        OUTPUT,
        AUDIT_OUTPUT,
        MANIFEST_OUTPUT,
        reconciliation_path=RECONCILIATION_INPUT,
        reconciliation_manifest_path=RECONCILIATION_MANIFEST,
    )
    counts = manifest["counts"]
    print(f"Candidate works: {counts['candidate_works']}")
    print(f"Included works:  {counts['included_works']}")
    print(f"Excluded works:   {counts['excluded_works']}")
    print(f"  broad-topic-only: {counts['excluded_broad_topic_only']}")
    print(
        "  keyword-only domain safeguard: "
        f"{counts['excluded_keyword_only_blocked_domain']}"
    )
    print(
        "  lower-tier without title evidence: "
        f"{counts['excluded_title_confirmed_keyword_without_title_evidence']}"
    )
    print(f"  language not accepted: {counts['excluded_non_accepted_language']}")
    print(f"  work type is not a paper: {counts['excluded_non_paper_work_type']}")
    print(f"Corpus: {manifest['artifacts']['included_works']['path']}")
    print(f"Audit: {manifest['artifacts']['candidate_audit']['path']}")
    print(f"Manifest: {MANIFEST_OUTPUT}")


if __name__ == "__main__":
    main()
