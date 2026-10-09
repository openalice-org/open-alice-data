"""Merge per-university people_viz JSON files into one, tagging each person's university."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


_EVIDENCE_FIELD = "quantum_papers"


def merge_people_viz(datasets: dict[str, dict[str, Any]]) -> dict[str, Any]:
    """datasets: {university_label: parsed_people_viz_json}.

    Cross-university dedup: the same OpenAlex author `id` can legitimately
    appear in more than one university's dataset (real co-affiliations are
    common among KTH/Chalmers/Lund). The frontend's D3 render keys strictly
    on `id`, so leaving duplicates in silently collapses them (only one
    renders) rather than erroring loudly. To avoid that, when an `id`
    appears in multiple universities we keep exactly ONE record here — the
    one with the higher `quantum_papers` value (the person's confirmed
    quantum-work count for that university's corpus) for that university.
    Ties are broken deterministically by university label sort order
    (alphabetical: Chalmers < KTH < Lund), NOT by dict/set iteration order.
    The dropped duplicate's fields are discarded outright — this does not
    merge/combine fields from both records, it just picks a winner.
    """
    all_topics: set[str] = set()
    all_institutions: set[str] = set()
    n_papers_scanned = 0
    best_by_id: dict[str, dict[str, Any]] = {}
    no_id_people: list[dict[str, Any]] = []

    for label in sorted(datasets.keys()):
        data = datasets[label]
        for person in data.get("people", []):
            tagged = dict(person)
            tagged["university"] = label
            pid = tagged.get("id")
            if pid is None:
                # No `id` to dedup on at all -> can't collide, keep as-is.
                no_id_people.append(tagged)
                continue
            existing = best_by_id.get(pid)
            if existing is None:
                best_by_id[pid] = tagged
            else:
                new_evidence = tagged.get(_EVIDENCE_FIELD, 0) or 0
                existing_evidence = existing.get(_EVIDENCE_FIELD, 0) or 0
                if new_evidence > existing_evidence:
                    best_by_id[pid] = tagged
                # tie or lower evidence: keep `existing`, which already won
                # the alphabetical-label ordering by virtue of being
                # inserted first (datasets are iterated in sorted order).
                # This also correctly handles a person appearing in 3+
                # universities: each subsequent candidate is compared
                # against the current best-so-far, one at a time.
        all_topics.update(data.get("filters", {}).get("topics", []))
        all_institutions.update(data.get("filters", {}).get("institutions", []))
        n_papers_scanned += data.get("meta", {}).get("n_papers_scanned", 0)

    all_people: list[dict[str, Any]] = list(best_by_id.values()) + no_id_people

    first_meta = next(iter(datasets.values())).get("meta", {}) if datasets else {}
    merged_meta = dict(first_meta)
    merged_meta["n_people"] = len(all_people)
    merged_meta["n_papers_scanned"] = n_papers_scanned
    merged_meta["universities"] = sorted(datasets.keys())

    return {
        "meta": merged_meta,
        "filters": {
            "topics": sorted(all_topics),
            "institutions": sorted(all_institutions),
            "aggregations": ["person", "institution", "country"],
        },
        "people": all_people,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Merge per-university people_viz JSON files")
    parser.add_argument(
        "--dataset",
        action="append",
        default=[],
        metavar="LABEL=PATH",
        help="One per-university people_viz JSON, e.g. --dataset 'KTH=/tmp/kth.json'",
    )
    parser.add_argument(
        "--auto-from-config",
        action="store_true",
        help="Load datasets automatically from universities.json + data/<key>/people_viz_<key>.json",
    )
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    root = Path(__file__).resolve().parent
    datasets: dict[str, dict[str, Any]] = {}

    if args.auto_from_config:
        universities = json.loads((root / "universities.json").read_text())
        for key, uni in universities.items():
            path = root / "data" / key / f"people_viz_{key}.json"
            if not path.exists():
                raise FileNotFoundError(f"Dataset for {uni['label']} not found: {path}")
            datasets[uni["label"]] = json.loads(path.read_text())

    for item in args.dataset:
        if "=" not in item:
            raise ValueError(f"--dataset must be LABEL=PATH, got: {item}")
        label, raw_path = item.split("=", 1)
        datasets[label] = json.loads(Path(raw_path).read_text())

    if not datasets:
        raise ValueError("Provide at least one --dataset or use --auto-from-config")

    merged = merge_people_viz(datasets)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(merged, ensure_ascii=False, indent=1))
    print(f"Merged {merged['meta']['n_people']} people from {len(datasets)} universities into {args.out}")


if __name__ == "__main__":
    import sys

    if "--dataset" in sys.argv or "--auto-from-config" in sys.argv:
        main()
    else:
        kth = {
            "meta": {"n_papers_scanned": 100, "method": "heuristic"},
            "filters": {"topics": ["Quantum Computing"], "institutions": ["KTH"]},
            "people": [{"id": "A1", "name": "Alice", "institution": "KTH"}],
        }
        chalmers = {
            "meta": {"n_papers_scanned": 50},
            "filters": {"topics": ["Quantum Sensing"], "institutions": ["Chalmers"]},
            "people": [{"id": "A2", "name": "Bob", "institution": "Chalmers"}],
        }
        merged = merge_people_viz({"KTH": kth, "Chalmers": chalmers})
        assert merged["meta"]["n_people"] == 2, merged["meta"]
        assert merged["meta"]["n_papers_scanned"] == 150, merged["meta"]
        assert sorted(merged["filters"]["topics"]) == ["Quantum Computing", "Quantum Sensing"]
        by_id = {p["id"]: p for p in merged["people"]}
        assert by_id["A1"]["university"] == "KTH"
        assert by_id["A2"]["university"] == "Chalmers"
        print("merge_people_viz smoke test passed")

        # --- cross-university dedup smoke test ---
        # Same OpenAlex id ("A_DUP") appears in both KTH and Chalmers, with
        # different evidence (quantum_papers). Only the higher-evidence
        # record should survive, tagged with its own university.
        kth_dup = {
            "meta": {"n_papers_scanned": 10},
            "filters": {"topics": [], "institutions": ["KTH"]},
            "people": [
                {"id": "A1", "name": "Alice", "quantum_papers": 5},
                {"id": "A_DUP", "name": "Dup Person", "quantum_papers": 32},
            ],
        }
        chalmers_dup = {
            "meta": {"n_papers_scanned": 5},
            "filters": {"topics": [], "institutions": ["Chalmers"]},
            "people": [
                {"id": "A_DUP", "name": "Dup Person", "quantum_papers": 4},
            ],
        }
        merged_dup = merge_people_viz({"KTH": kth_dup, "Chalmers": chalmers_dup})
        assert merged_dup["meta"]["n_people"] == 2, merged_dup["meta"]
        dup_by_id = {p["id"]: p for p in merged_dup["people"]}
        assert len(dup_by_id) == 2, dup_by_id
        assert dup_by_id["A_DUP"]["university"] == "KTH", dup_by_id["A_DUP"]
        assert dup_by_id["A_DUP"]["quantum_papers"] == 32, dup_by_id["A_DUP"]
        # only one A_DUP record total (not silently duplicated)
        assert sum(1 for p in merged_dup["people"] if p["id"] == "A_DUP") == 1

        # tie-break: equal evidence -> alphabetically-first label wins (Chalmers < KTH)
        kth_tie = {
            "meta": {"n_papers_scanned": 1},
            "filters": {"topics": [], "institutions": ["KTH"]},
            "people": [{"id": "A_TIE", "name": "Tie Person", "quantum_papers": 7}],
        }
        chalmers_tie = {
            "meta": {"n_papers_scanned": 1},
            "filters": {"topics": [], "institutions": ["Chalmers"]},
            "people": [{"id": "A_TIE", "name": "Tie Person", "quantum_papers": 7}],
        }
        merged_tie = merge_people_viz({"KTH": kth_tie, "Chalmers": chalmers_tie})
        tie_by_id = {p["id"]: p for p in merged_tie["people"]}
        assert tie_by_id["A_TIE"]["university"] == "Chalmers", tie_by_id["A_TIE"]

        print("merge_people_viz cross-university dedup smoke test passed")
