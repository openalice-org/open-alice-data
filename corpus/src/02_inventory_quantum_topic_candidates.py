"""Create a review-only quantum topic inventory from an OpenAlex snapshot."""

import collections
import csv
import datetime as dt
import hashlib
import json
import os
import re
import unicodedata

SNAPSHOT = "data/openalex_topics_snapshot.jsonl"
SNAPSHOT_MANIFEST = "data/openalex_topics_snapshot_manifest.json"
CONFIG = "config/sweden_quantum_discovery_terms.json"
OUTPUT = "data/sweden_quantum_topic_candidates.csv"
MANIFEST = "data/sweden_quantum_topic_candidates_manifest.json"

CSV_FIELDS = (
    "topic_id",
    "topic_url",
    "topic_name",
    "match_strengths",
    "matched_name_terms",
    "matched_keyword_terms",
    "domain_id",
    "domain_name",
    "field_id",
    "field_name",
    "subfield_id",
    "subfield_name",
    "taxonomy_gate",
    "taxonomy_gate_reason",
    "works_count",
    "topic_keywords",
    "topic_description",
)

GATE_ORDER = {"eligible": 0, "conditional": 1, "blocked": 2}


def file_sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            digest.update(chunk)
    return digest.hexdigest()


def short_id(value):
    return (value or "").rsplit("/", 1)[-1]


def normalize_text(value):
    value = unicodedata.normalize("NFKC", value or "").lower()
    value = value.replace("-", " ").replace("–", " ").replace("—", " ")
    return " ".join(re.findall(r"[a-z0-9]+", value))


def contains_term(text, term):
    normalized_text = f" {normalize_text(text)} "
    normalized_term = f" {normalize_text(term)} "
    return normalized_term != "  " and normalized_term in normalized_text


def validate_config(config):
    if config.get("schema_version") != 1:
        raise ValueError("Unsupported discovery-term schema_version")

    groups = config.get("term_groups") or {}
    if set(groups) != {"direct", "related"}:
        raise ValueError("term_groups must contain exactly direct and related")
    all_terms = [term for terms in groups.values() for term in terms]
    normalized_terms = [normalize_text(term) for term in all_terms]
    if any(not term for term in normalized_terms):
        raise ValueError("Discovery terms must not be empty")
    if len(normalized_terms) != len(set(normalized_terms)):
        raise ValueError("Discovery terms must be unique after normalization")

    policy = config.get("taxonomy_policy") or {}
    required = {"eligible_domains", "conditional_domains", "blocked_domains"}
    if set(policy) != required:
        raise ValueError(f"taxonomy_policy must contain exactly {sorted(required)}")
    domains = [domain for key in required for domain in policy[key]]
    if len(domains) != len(set(domains)):
        raise ValueError("A domain may appear in only one taxonomy gate")


def load_config(path):
    with open(path, encoding="utf-8") as f:
        config = json.load(f)
    validate_config(config)
    return config


def topic_keyword_names(topic):
    """Topic keywords as plain strings. OpenAlex now returns {display_name, id, score}
    objects here; it used to return bare strings."""
    return [
        keyword.get("display_name") or "" if isinstance(keyword, dict) else keyword
        for keyword in topic.get("keywords") or []
    ]


def topic_matches(topic, config):
    name = topic.get("display_name") or ""
    keywords = topic_keyword_names(topic)
    name_matches = collections.defaultdict(set)
    keyword_matches = collections.defaultdict(set)

    for strength, terms in config["term_groups"].items():
        for term in terms:
            if contains_term(name, term):
                name_matches[strength].add(term)
            if any(contains_term(keyword, term) for keyword in keywords):
                keyword_matches[strength].add(term)

    return name_matches, keyword_matches


def taxonomy_gate(topic, config):
    domain = (topic.get("domain") or {}).get("display_name") or ""
    policy = config["taxonomy_policy"]
    for gate, key in (
        ("eligible", "eligible_domains"),
        ("conditional", "conditional_domains"),
        ("blocked", "blocked_domains"),
    ):
        if domain in policy[key]:
            return gate, f"domain {domain!r} is {gate} under the review policy"
    raise ValueError(f"Topic has an unconfigured OpenAlex domain: {domain!r}")


def candidate_row(topic, config):
    name_matches, keyword_matches = topic_matches(topic, config)
    if not name_matches and not keyword_matches:
        return None

    gate, reason = taxonomy_gate(topic, config)
    strengths = sorted(set(name_matches) | set(keyword_matches))
    name_terms = sorted({term for terms in name_matches.values() for term in terms})
    keyword_terms = sorted(
        {term for terms in keyword_matches.values() for term in terms}
    )
    domain = topic.get("domain") or {}
    field = topic.get("field") or {}
    subfield = topic.get("subfield") or {}

    return {
        "topic_id": short_id(topic.get("id")),
        "topic_url": topic.get("id") or "",
        "topic_name": topic.get("display_name") or "",
        "match_strengths": "|".join(strengths),
        "matched_name_terms": "|".join(name_terms),
        "matched_keyword_terms": "|".join(keyword_terms),
        "domain_id": short_id(domain.get("id")),
        "domain_name": domain.get("display_name") or "",
        "field_id": short_id(field.get("id")),
        "field_name": field.get("display_name") or "",
        "subfield_id": short_id(subfield.get("id")),
        "subfield_name": subfield.get("display_name") or "",
        "taxonomy_gate": gate,
        "taxonomy_gate_reason": reason,
        "works_count": topic.get("works_count", ""),
        "topic_keywords": " | ".join(topic_keyword_names(topic)),
        "topic_description": topic.get("description") or "",
    }


def read_verified_snapshot(snapshot_path, snapshot_manifest_path):
    with open(snapshot_manifest_path, encoding="utf-8") as f:
        manifest = json.load(f)
    actual_hash = file_sha256(snapshot_path)
    if actual_hash != manifest.get("output_sha256"):
        raise RuntimeError("Topic snapshot hash does not match its manifest")

    topics = []
    with open(snapshot_path, encoding="utf-8") as f:
        for line_number, line in enumerate(f, start=1):
            try:
                topics.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise RuntimeError(
                    f"Invalid JSON on topic snapshot line {line_number}"
                ) from exc

    if len(topics) != manifest.get("topic_count"):
        raise RuntimeError(
            f"Topic snapshot contains {len(topics)} rows; "
            f"manifest records {manifest.get('topic_count')}"
        )
    return topics, manifest


def write_candidate_inventory(
    snapshot_path=SNAPSHOT,
    snapshot_manifest_path=SNAPSHOT_MANIFEST,
    config_path=CONFIG,
    output=OUTPUT,
    manifest_path=MANIFEST,
):
    config = load_config(config_path)
    topics, snapshot_manifest = read_verified_snapshot(
        snapshot_path, snapshot_manifest_path
    )
    rows = [row for topic in topics if (row := candidate_row(topic, config))]
    rows.sort(
        key=lambda row: (
            GATE_ORDER[row["taxonomy_gate"]],
            "direct" not in row["match_strengths"].split("|"),
            row["topic_name"].casefold(),
            row["topic_id"],
        )
    )

    output_tmp = output + ".tmp"
    manifest_tmp = manifest_path + ".tmp"
    os.makedirs(os.path.dirname(output) or ".", exist_ok=True)
    os.makedirs(os.path.dirname(manifest_path) or ".", exist_ok=True)
    try:
        with open(output_tmp, "w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=CSV_FIELDS)
            writer.writeheader()
            writer.writerows(rows)

        output_hash = file_sha256(output_tmp)
        gate_counts = collections.Counter(row["taxonomy_gate"] for row in rows)
        strength_counts = collections.Counter(
            strength for row in rows for strength in row["match_strengths"].split("|")
        )
        manifest = {
            "schema_version": 1,
            "created_at": dt.datetime.now(dt.timezone.utc).isoformat(),
            "input_snapshot": snapshot_path,
            "input_snapshot_id": snapshot_manifest.get("taxonomy_snapshot_id"),
            "input_topic_count": snapshot_manifest.get("topic_count"),
            "discovery_config": config_path,
            "discovery_config_sha256": file_sha256(config_path),
            "candidate_count": len(rows),
            "gate_counts": dict(sorted(gate_counts.items())),
            "match_strength_counts": dict(sorted(strength_counts.items())),
            "output": output,
            "output_sha256": output_hash,
        }
        with open(manifest_tmp, "w", encoding="utf-8") as f:
            json.dump(manifest, f, ensure_ascii=False, indent=2, sort_keys=True)
            f.write("\n")

        os.replace(output_tmp, output)
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
    result = write_candidate_inventory()
    print(f"Wrote {result['candidate_count']} review candidates to {OUTPUT}")
    print(f"Taxonomy gates: {result['gate_counts']}")
    print("No topics were approved or added to retrieval.")


if __name__ == "__main__":
    main()
