"""Inject per-author filtered patent lists (from outputs/author_patents.json,
built by build_author_patents.py) into data/people_viz.json as `patents: [...]`.

Run after build_author_patents.py has been (re)run for every university.
"""

from __future__ import annotations

import json
from pathlib import Path

root = Path(__file__).resolve().parent
universities = json.loads((root / "universities.json").read_text())

viz_path = root / "data" / "people_viz.json"
viz = json.loads(viz_path.read_text())

by_id: dict[str, list[dict]] = {}
for key in universities:
    path = root / "data" / key / "outputs" / "author_patents.json"
    if not path.exists():
        continue
    entries = json.loads(path.read_text())
    for aid, entry in entries.items():
        by_id[aid] = entry["patents"]

n_matched = 0
n_with_patents = 0
for person in viz["people"]:
    patents = by_id.get(person["id"], [])
    person["patents"] = patents
    if person["id"] in by_id:
        n_matched += 1
    if patents:
        n_with_patents += 1

viz_path.write_text(json.dumps(viz, ensure_ascii=False, indent=1))
print(f"[patch] {n_matched} people matched to an author_patents.json record, "
      f"{n_with_patents} carry >=1 filtered patent -> {viz_path}")
