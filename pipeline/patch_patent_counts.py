"""Refresh total_patents/quantum_patents in data/people_viz.json from freshly
rebuilt outputs/author_summary.csv, after an overnight live EPO gap-fill.

Only touches those two fields (same matching precedence as
build_people_data.py's load_summary: openalex_id, else normalized name) —
paper counts, axis scores, and everything else from the last full
build_people_data.py run are left untouched.
"""

from __future__ import annotations

import csv
import json
import re
from pathlib import Path

root = Path(__file__).resolve().parent
universities = json.loads((root / "universities.json").read_text())


def norm_name(name: str) -> str:
    if not name:
        return ""
    n = name.strip().lower().replace(".", " ")
    n = re.sub(r"\s+", " ", n)
    return n.strip()


def load_summary(patents_csv: Path):
    by_id, by_name = {}, {}
    if not patents_csv.exists():
        return by_id, by_name
    with patents_csv.open(encoding="utf-8") as f:
        for row in csv.DictReader(f):
            def gi(k):
                try:
                    return int(float(row.get(k) or 0))
                except ValueError:
                    return 0
            rec = {
                "total_patents": gi("patent_kth_filtered_count"),
                "quantum_patents": gi("patent_quantum_aligned_count"),
            }
            aid = (row.get("openalex_id") or "").strip()
            if aid:
                by_id[aid] = rec
            nm = norm_name(row.get("author_name", ""))
            if nm:
                by_name.setdefault(nm, rec)
    return by_id, by_name


viz_path = root / "data" / "people_viz.json"
viz = json.loads(viz_path.read_text())

by_id_all: dict[str, dict] = {}
by_name_all: dict[str, dict] = {}
for key in universities:
    by_id, by_name = load_summary(root / "data" / key / "outputs" / "author_summary.csv")
    by_id_all.update(by_id)
    for nm, rec in by_name.items():
        by_name_all.setdefault(nm, rec)

n_changed = 0
for person in viz["people"]:
    summ = by_id_all.get(person["id"]) or by_name_all.get(norm_name(person["name"]))
    if not summ:
        continue
    if (person.get("total_patents", 0) != summ["total_patents"]
            or person.get("quantum_patents", 0) != summ["quantum_patents"]):
        n_changed += 1
    person["total_patents"] = summ["total_patents"]
    person["quantum_patents"] = summ["quantum_patents"]

viz_path.write_text(json.dumps(viz, ensure_ascii=False, indent=1))
print(f"[patch] {n_changed} people had total_patents/quantum_patents change -> {viz_path}")
