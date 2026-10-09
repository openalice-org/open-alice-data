"""Build a cross-author patent index so the frontend can show, for any patent
that lists 2+ Nordic researchers as inventors, who it is shared with.

Reads each university's data/<uni>/outputs/author_patents.json (the same
per-author filtered patent lists build_author_patents.py produces and
patch_people_patents.py injects into data/people_viz.json), groups patents
by epo_id across ALL people, and keeps only the epo_ids shared by 2+ distinct
people. Results are restricted to people who are actually present in
data/people_viz.json (the frontend's node set).

This script is read-only w.r.t. every existing file: it only *writes* the new
outputs below.

Outputs:
  data/patent_links.json
  ../open-alice/public/data/patent_links.json  (verbatim copy)
"""

from __future__ import annotations

import json
from pathlib import Path

root = Path(__file__).resolve().parent
universities = json.loads((root / "universities.json").read_text())

viz_path = root / "data" / "people_viz.json"
viz = json.loads(viz_path.read_text())
viz_people = {p["id"]: p["name"] for p in viz["people"]}

# epo_id -> {author_id: author_name}
by_epo_id: dict[str, dict[str, str]] = {}

n_people_with_patents = 0
for key in universities:
    path = root / "data" / key / "outputs" / "author_patents.json"
    if not path.exists():
        continue
    entries = json.loads(path.read_text())
    for aid, entry in entries.items():
        if aid not in viz_people:
            continue
        patents = entry.get("patents") or []
        if not patents:
            continue
        n_people_with_patents += 1
        author_name = viz_people.get(aid, entry.get("author_name", ""))
        for patent in patents:
            epo_id = patent.get("epo_id")
            if not epo_id:
                continue
            by_epo_id.setdefault(epo_id, {})[aid] = author_name

n_distinct_epo_ids = len(by_epo_id)

shared: dict[str, list[dict]] = {}
for epo_id, authors in by_epo_id.items():
    if len(authors) < 2:
        continue
    shared[epo_id] = [
        {"id": aid, "name": name} for aid, name in sorted(authors.items(), key=lambda kv: kv[1])
    ]

out_path = root / "data" / "patent_links.json"
out_path.write_text(json.dumps(shared, ensure_ascii=False, indent=1))


print(f"[patent_links] {n_people_with_patents} people had patents, "
      f"{n_distinct_epo_ids} distinct epo_ids, {len(shared)} shared (2+ people) "
      f"-> {out_path}")

examples = sorted(shared.items(), key=lambda kv: -len(kv[1]))[:3]
for epo_id, people in examples:
    names = ", ".join(p["name"] for p in people)
    print(f"  example: {epo_id} shared by {len(people)}: {names}")
