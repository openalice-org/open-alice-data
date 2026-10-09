"""Per-person top co-authors and top co-institutions WITHIN THE QUANTUM CORPUS.

The frontend (open-alice) wants to show, per tracked researcher, who they
actually collaborate with and which institutions they actually collaborate
with — restricted to the quantum-classified corpus, not their whole career
(a person's people_viz.json total_papers/h_index etc. cover everything they
ever published, which is the wrong denominator for a "quantum collaboration
network").

Source data is per-university, not per-paper: data/<university>/
quantum_classified_works_specter_reviewed.jsonl, one file per Nordic
institution folder under data/. A paper co-authored across tracked
institutions is duplicated once per institution folder (same pattern as
build_paper_counts.py's all_universities_works.jsonl), so this script dedupes
by OpenAlex work id before building any edges — otherwise a single
multi-institution paper would be double- or triple-counted.

Tracked people = data/people_viz.json's id set (the working set the frontend
actually renders as nodes). For each paper, every tracked author A gets:
  - a coauthor edge to every OTHER author B on that paper, but only emitted
    in the final output when B is ALSO tracked (we only draw links between
    people the frontend can actually show as a node);
  - a co-institution edge to every institution display_name attached to any
    OTHER author's authorship on that paper (tracked or not — an external
    collaborator's institution is still a useful, nameable signal even when
    the person themselves isn't tracked), excluding A's own institution(s).

Output: data/collaborations.json, keyed by tracked person OpenAlex id, with
top 8 coauthors (by shared quantum-paper count) and top 8 co-institutions.
People with zero edges of either kind are omitted entirely, not emitted with
empty lists — keeps the frontend from having to special-case empty arrays.
"""

from __future__ import annotations

import argparse
import glob
import json
from collections import Counter, defaultdict
from pathlib import Path

TOP_N = 8


def load_tracked_people(people_viz_path: Path) -> dict[str, str]:
    """Maps tracked OpenAlex author id -> display name."""
    data = json.loads(people_viz_path.read_text())
    return {p["id"]: p["name"] for p in data.get("people", []) if p.get("id")}


def iter_quantum_papers(pattern: str):
    """Streams distinct (by work id) quantum-included paper records across all
    per-university jsonl files, deduping across files as they're consumed."""
    seen_ids: set[str] = set()
    files = sorted(glob.glob(pattern))
    for i, path in enumerate(files, 1):
        print(f"  [{i}/{len(files)}] {path}")
        with open(path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                rec = json.loads(line)
                if rec.get("quantum_included") is not True:
                    continue
                pid = rec.get("id")
                if not pid or pid in seen_ids:
                    continue
                seen_ids.add(pid)
                yield rec


def build_collaborations(pattern: str, tracked: dict[str, str]) -> dict:
    coauthor_counts: dict[str, Counter] = defaultdict(Counter)
    institution_counts: dict[str, Counter] = defaultdict(Counter)
    total_papers = 0

    for rec in iter_quantum_papers(pattern):
        total_papers += 1
        authorships = rec.get("authorships", []) or []

        # Per-authorship (author_id, [institution display_names]) for this paper.
        author_institutions: list[tuple[str, list[str]]] = []
        for ship in authorships:
            author = ship.get("author") or {}
            author_id = author.get("id")
            if not author_id:
                continue
            insts = [
                inst.get("display_name")
                for inst in (ship.get("institutions") or [])
                if inst.get("display_name")
            ]
            author_institutions.append((author_id, insts))

        for a_id, a_insts in author_institutions:
            if a_id not in tracked:
                continue
            a_inst_set = set(a_insts)
            for b_id, b_insts in author_institutions:
                if b_id == a_id:
                    continue
                if b_id in tracked:
                    coauthor_counts[a_id][b_id] += 1
                for inst_name in b_insts:
                    if inst_name in a_inst_set:
                        continue
                    institution_counts[a_id][inst_name] += 1

    result: dict = {}
    people_with_coauthors = 0
    people_with_institutions = 0

    all_person_ids = set(coauthor_counts) | set(institution_counts)
    for person_id in all_person_ids:
        entry: dict = {}

        coauthors = coauthor_counts.get(person_id)
        if coauthors:
            top = coauthors.most_common(TOP_N)
            entry["top_coauthors"] = [
                {"id": b_id, "name": tracked.get(b_id, ""), "papers": count}
                for b_id, count in top
            ]
            people_with_coauthors += 1

        institutions = institution_counts.get(person_id)
        if institutions:
            top = institutions.most_common(TOP_N)
            entry["top_institutions"] = [
                {"name": name, "papers": count} for name, count in top
            ]
            people_with_institutions += 1

        if entry:
            result[person_id] = entry

    return {
        "collaborations": result,
        "total_papers": total_papers,
        "people_with_coauthors": people_with_coauthors,
        "people_with_institutions": people_with_institutions,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build per-person top co-authors and top co-institutions within the quantum corpus"
    )
    parser.add_argument(
        "--works-glob",
        default="data/*/quantum_classified_works_specter_reviewed.jsonl",
        help="glob pattern for per-university quantum-classified works jsonl files",
    )
    parser.add_argument("--people-viz", type=Path, default=Path("data/people_viz.json"))
    parser.add_argument("--out", type=Path, default=Path("data/collaborations.json"))
    parser.add_argument(
        "--copy-to",
        type=Path,
        default=None,
        help="second location to write the same output (frontend app's public data dir)",
    )
    args = parser.parse_args()

    root = Path(__file__).resolve().parent
    tracked = load_tracked_people(args.people_viz)
    print(f"Loaded {len(tracked)} tracked people from {args.people_viz}")

    print("Streaming quantum-classified works...")
    stats = build_collaborations(str(root / args.works_glob) if not Path(args.works_glob).is_absolute() else args.works_glob, tracked)

    out_data = stats["collaborations"]
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(out_data, indent=2, sort_keys=True))
    print(f"Wrote {len(out_data)} person entries to {args.out}")

    if args.copy_to:  # optional second destination; publishing is done by `make publish`
        args.copy_to.parent.mkdir(parents=True, exist_ok=True)
        args.copy_to.write_text(json.dumps(out_data, indent=2, sort_keys=True))
        print(f"Copied to {args.copy_to}")

    print()
    print("Summary")
    print(f"  Total distinct quantum papers processed: {stats['total_papers']}")
    print(f"  Tracked people with >=1 coauthor edge:    {stats['people_with_coauthors']}")
    print(f"  Tracked people with >=1 institution edge: {stats['people_with_institutions']}")
    print()
    print("  Example entries:")
    for person_id, entry in list(out_data.items())[:2]:
        name = tracked.get(person_id, "?")
        print(f"    {name} ({person_id}):")
        print(f"      {json.dumps(entry, indent=6)}")


if __name__ == "__main__":
    main()
