#!/usr/bin/env python3
"""Split KTH.Tools.QuantumAuthors' combined corpus into per-university files.

Reads data/sweden_quantum_works.jsonl (one line per accepted quantum work,
country-scoped SE|NO|DK|IS|FI, quantum_included already true for every row)
and, for each institution in universities.json, writes every work with a
matching authorship institution id into
data/<university>/quantum_classified_works_specter_reviewed.jsonl — the same
filename the old fetch_openalex_corpus.py + specter_review_corpus.py stages
used to produce, so extract_authors.py and downstream scripts need no
changes. A work co-authored across two tracked universities is written to
both.

classification_label is set to "core_quantum" for every row (this corpus
has no likely_quantum/review/reject tier — quantum_included is already the
final accept decision) so ACCEPTED_LABELS checks downstream keep working.

Run once after the KTH tool has produced its output, before extract_authors.py:
  python split_nordic_corpus.py
"""

import json
import pathlib
import sys

SPRINT13 = pathlib.Path(__file__).resolve().parent
DEFAULT_INPUT = (SPRINT13.parent / "corpus" / "data"
                 / "sweden_quantum_works.jsonl")
UNIVERSITIES_PATH = SPRINT13 / "universities.json"
OUT_FILENAME = "quantum_classified_works_specter_reviewed.jsonl"


def main():
    input_path = pathlib.Path(sys.argv[1]) if len(sys.argv) > 1 else DEFAULT_INPUT
    if not input_path.exists():
        sys.exit(f"Input not found: {input_path}")

    universities = json.loads(UNIVERSITIES_PATH.read_text(encoding="utf-8"))
    id_to_key = {}
    for key, u in universities.items():
        id_to_key[u["openalex_institution_id"]] = key
        for alias_id in u.get("openalex_institution_id_aliases", []):
            id_to_key[alias_id] = key

    out_files = {}
    for key in universities:
        out_dir = SPRINT13 / "data" / key
        out_dir.mkdir(parents=True, exist_ok=True)
        out_files[key] = open(out_dir / OUT_FILENAME, "w", encoding="utf-8")

    counts = {key: 0 for key in universities}
    n_in = n_included = n_written = 0
    try:
        with open(input_path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                n_in += 1
                work = json.loads(line)
                if not work.get("quantum_included"):
                    continue
                n_included += 1
                work["classification_label"] = "core_quantum"

                matched_keys = set()
                for authorship in work.get("authorships") or []:
                    for inst in authorship.get("institutions") or []:
                        key = id_to_key.get(inst.get("id"))
                        if key:
                            matched_keys.add(key)

                out_line = json.dumps(work, ensure_ascii=False) + "\n"
                for key in matched_keys:
                    out_files[key].write(out_line)
                    counts[key] += 1
                    n_written += 1

                if n_in % 500_000 == 0:
                    print(f"[progress] {n_in} lines read, {n_included} included, "
                          f"{n_written} university writes", file=sys.stderr)
    finally:
        for fh in out_files.values():
            fh.close()

    print(f"\nRead {n_in} lines, {n_included} quantum_included, "
          f"{n_written} (work, university) writes")
    for key in sorted(universities):
        print(f"  {key:<12} {counts[key]}")


if __name__ == "__main__":
    main()
