#!/usr/bin/env python3
"""Rescue pass, stage R1: choose which excluded candidates deserve a second look.

WHY THE RESCUE PASS EXISTS (see docs/METHODOLOGY.md for the full evidence)
    Corpus step 05 admits a work retrieved only through a *broad-related* topic when its
    title, abstract or OpenAlex keywords contain a literal quantum term (`quantum`, `qubit`,
    `entangl…`, …). Between our earlier build and the current one OpenAlex replaced the
    `keywords` field (generic concepts such as "Quantum mechanics" became paper-specific
    phrases such as "exciton-polariton condensates") and re-tagged many topics. A clearly
    quantum paper ("Josephson scanning tunneling microscopy") may therefore no longer contain
    any literal quantum term, and the rule drops it. A blind LLM audit found about a quarter
    of these rule-excluded works to be core quantum research.

WHAT THIS STAGE DOES
    It reads the candidate audit written by corpus step 05 and selects the candidates that
    were excluded by the *content rule only*. It deliberately leaves out:
      * record-gate failures (language, work type): intentional scope decisions, not evidence;
      * the keyword-only domain safeguard: an intentional precision guard;
      * works without a usable abstract (< MIN_ABSTRACT_CHARS): there is not enough text for
        a semantic judgement, so they stay excluded (conservative).
    The selected works are written, with only the fields the later stages need, to
    corpus/data/rescue_pool.jsonl, plus a manifest with counts and a content hash.

Usage:
    python rescue_pool.py                  # corpus/data/* -> corpus/data/rescue_pool.jsonl
    python rescue_pool.py --limit 1000     # quick dry run of the selection logic
"""

import argparse
import collections
import csv
import hashlib
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from build_domain_specter import reconstruct_abstract  # noqa: E402

CORPUS_DATA = HERE.parent / "corpus" / "data"
CORPUS_CONFIG = HERE.parent / "corpus" / "config"
MIN_ABSTRACT_CHARS = 200


def full_id(work_id):
    return work_id if work_id.startswith("http") else f"https://openalex.org/{work_id}"


def gate_failure(row, gates):
    """Mirror of corpus step 05's record gates (language, work type), from the audit row."""
    language = row["language"]
    accepted = gates.get("accepted_languages")
    if language:
        if accepted is not None and language not in accepted:
            return "language"
    elif not gates.get("accept_missing_language", True):
        return "language"
    if row["work_type"] and row["work_type"] in set(gates.get("excluded_work_types") or []):
        return "work type"
    return None


def pool_status(row, gates):
    """None if the excluded work is in the rescue pool, otherwise why it is left out."""
    if row["included"] == "true":
        return "already included"
    if gate_failure(row, gates):
        return "record gate (intentional scope decision)"
    if row["keyword_only_blocked_primary_domain"] == "true":
        return "keyword-only domain safeguard (intentional)"
    return None


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--audit", type=Path, default=CORPUS_DATA / "sweden_quantum_candidate_audit.csv")
    ap.add_argument("--union", type=Path, default=CORPUS_DATA / "sweden_quantum_candidate_union.jsonl")
    ap.add_argument("--gates", type=Path, default=CORPUS_CONFIG / "sweden_quantum_record_gates.json")
    ap.add_argument("--out", type=Path, default=CORPUS_DATA / "rescue_pool.jsonl")
    ap.add_argument("--limit", type=int, default=None, help="stop after N pool works (dry run)")
    args = ap.parse_args()

    gates = json.loads(args.gates.read_text(encoding="utf-8"))
    status = {}
    with open(args.audit, encoding="utf-8") as f:
        for row in csv.DictReader(f):
            status[full_id(row["work_id"])] = pool_status(row, gates)
    counts = collections.Counter("pool" if s is None else s for s in status.values())

    kept = collections.Counter()
    sha = hashlib.sha256()
    tmp = args.out.with_suffix(".jsonl.tmp")
    with open(args.union, encoding="utf-8") as src, open(tmp, "w", encoding="utf-8") as out:
        for line in src:
            work = json.loads(line)
            wid = work["id"]
            if status.get(wid, "not in audit") is not None:
                continue
            title = (work.get("display_name") or "").strip()
            abstract = reconstruct_abstract(work.get("abstract_inverted_index"))
            if not title or len(abstract) < MIN_ABSTRACT_CHARS:
                kept["pool candidate without usable abstract (stays excluded)"] += 1
                continue
            record = json.dumps(
                {"id": wid, "display_name": title, "abstract": abstract,
                 "publication_year": work.get("publication_year")},
                ensure_ascii=False,
            ) + "\n"
            out.write(record)
            sha.update(record.encode("utf-8"))
            kept["rescue pool"] += 1
            if args.limit and kept["rescue pool"] >= args.limit:
                break
    tmp.replace(args.out)

    manifest = {
        "schema_version": 1,
        "stage": "R1 rescue pool",
        "min_abstract_chars": MIN_ABSTRACT_CHARS,
        "audit_rows_by_status": dict(counts),
        "pool_size": kept["rescue pool"],
        "excluded_content_rule_without_abstract": kept["pool candidate without usable abstract (stays excluded)"],
        "output": str(args.out),
        "output_sha256": sha.hexdigest(),
        "dry_run_limit": args.limit,
    }
    args.out.with_name("rescue_pool_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
