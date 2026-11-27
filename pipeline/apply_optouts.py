#!/usr/bin/env python3
"""Final export step: remove people who opted out (GDPR) from what is published, and nothing else.

WHAT AN OPT-OUT MEANS HERE
    The person stops being disseminated as a person: no point on any map, no profile record (so no ORCID, no
    metrics, no topic), no collaboration profile of their own, no per-person paper page. Their PRODUCTION stays
    in the data: their papers still count in every aggregate (paper_counts_geo.json, the taxonomy tree) and still
    appear in their co-authors' paper lists, and their name stays where it is a property of a paper or patent
    (co-author chips, co-inventors). Their ORCID is removed everywhere.

WHAT IT DOES
    Reads the finished data in pipeline/data and writes the files the app loads into the staging folder
    (`make stage` calls this). The sources are never modified, so aggregates are computed from the complete data
    and a later rebuild applies the same registry again: an opt-out never "comes back".

      people_viz.json, people_taxonomy.json   the person's entry is dropped from `people` (meta.n_people follows)
      collaborations.json                     the person's OWN entry (keyed by their ids) is dropped; mentions of them
                                              in other people's `top_coauthors` stay (a property of the papers)
      person_papers/*.json (if present)       the person's own shard entry is dropped; the papers stay in co-authors'
      patent_links.json, paper_counts_geo.json  copied unchanged

    A person matches the registry by ANY of: their OpenAlex author id, one of their merged ids (`merged_from`), or their
    ORCID. Never by name: names collide. A registry entry that matches nobody is kept as a standing suppression
    (the person may appear after a later rebuild).

THE REGISTRY (private: it is personal data and this repository is public, so it is git-ignored)
    A JSONL file, one opt-out per line, at $OPTOUT_FILE (default: private/optout.jsonl in the repo root):
      {"openalex_ids": ["A5000000000"], "orcids": ["0000-0000-0000-0000"], "requested_at": "2026-11-27", "reference": "internal ref"}
    Either list may be empty but not both. Ids may be bare (A123), numeric or full URLs. See docs/optout.example.jsonl.

SAFETY CHECKS (the export fails, and nothing is left staged, if one fails)
    * no removed ORCID appears anywhere in any staged file;
    * no removed id appears as a person (people[].id / merged_from, collaboration key, shard key).
    A counts-only report (no identifiers) is written to pipeline/data/optout_report.json.

    python apply_optouts.py --out build/data            # what `make stage` runs
    python apply_optouts.py --out /tmp/x --registry /tmp/registry.jsonl
"""

import argparse
import json
import os
import re
import shutil
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
DATA = HERE / "data"
DEFAULT_REGISTRY = Path(os.environ.get("OPTOUT_FILE") or ROOT / "private" / "optout.jsonl")
ORCID_RE = re.compile(r"(\d{4}-\d{4}-\d{4}-\d{3}[\dXx])")


def norm_oa(x) -> str | None:
    m = re.search(r"(?:^|/)([A-Za-z])?(\d{3,})$", str(x).strip())
    return f"https://openalex.org/A{int(m.group(2))}" if m else None


def norm_orcid(x) -> str | None:
    m = ORCID_RE.search(str(x or ""))
    return m.group(1).upper() if m else None


def load_registry(path: Path, required: bool) -> tuple[set, set, int]:
    ids, orcids, rows = set(), set(), 0
    if not path.exists():
        msg = f"no opt-out registry at {path}"
        if required:
            sys.exit(f"ERROR: {msg} and --require-registry is set")
        print(f"WARNING: {msg}: exporting everyone. Create it (see docs/optout.example.jsonl) before the first opt-out.")
        return ids, orcids, 0
    for n, line in enumerate(open(path, encoding="utf-8"), 1):
        if not line.strip():
            continue
        r = json.loads(line)
        got_id = [norm_oa(x) for x in r.get("openalex_ids") or r.get("ids") or []]
        got_orcid = [norm_orcid(x) for x in r.get("orcids") or []]
        if None in got_id or None in got_orcid or not (got_id or got_orcid):
            sys.exit(f"ERROR: registry line {n} has no valid OpenAlex id / ORCID (nothing printed: it is personal data)")
        ids.update(got_id)
        orcids.update(got_orcid)
        rows += 1
    return ids, orcids, rows


def matches(person: dict, ids: set, orcids: set) -> bool:
    aliases = {person.get("id"), *(person.get("merged_from") or [])}
    return bool(aliases & ids) or (norm_orcid(person.get("orcid")) in orcids)


def write_json(path: Path, obj) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, separators=(",", ":"))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", type=Path, required=True, help="staging folder")
    ap.add_argument("--data", type=Path, default=DATA)
    ap.add_argument("--registry", type=Path, default=DEFAULT_REGISTRY)
    ap.add_argument("--require-registry", action="store_true",
                    default=os.environ.get("OPTOUT_REQUIRED") == "1", help="fail instead of warning when the registry is missing")
    ap.add_argument("--person-papers", type=Path, default=None, help="folder of per-person paper shards (default: <data>/person_papers if it exists)")
    args = ap.parse_args()
    data, out = args.data, args.out
    out.mkdir(parents=True, exist_ok=True)

    reg_ids, reg_orcids, reg_rows = load_registry(args.registry, args.require_registry)
    removed_ids, removed_orcids = set(reg_ids), set(reg_orcids)
    stats = {"registry_entries": reg_rows}

    # --- decide who is removed, from people_viz (the reference list of people) and people_taxonomy
    pv = json.loads((data / "people_viz.json").read_text(encoding="utf-8"))
    pt = json.loads((data / "people_taxonomy.json").read_text(encoding="utf-8"))
    for p in pv["people"] + pt["people"]:
        if matches(p, reg_ids, reg_orcids):
            removed_ids.update({p.get("id"), *(p.get("merged_from") or [])})
            if norm_orcid(p.get("orcid")):
                removed_orcids.add(norm_orcid(p["orcid"]))
    removed_ids.discard(None)
    keep = lambda p: not matches(p, removed_ids, removed_orcids)  # noqa: E731  (re-test with every alias found)

    written = []
    n0 = len(pv["people"])
    pv["people"] = [p for p in pv["people"] if keep(p)]
    pv["meta"]["n_people"] = len(pv["people"])
    stats["people_viz_removed"] = n0 - len(pv["people"])
    write_json(out / "people_viz.json", pv)
    written.append(out / "people_viz.json")

    n0 = len(pt["people"])
    pt["people"] = [p for p in pt["people"] if keep(p)]
    pt["meta"]["n_people"] = len(pt["people"])  # the taxonomy tree and n_papers_* are aggregates: untouched
    stats["people_taxonomy_removed"] = n0 - len(pt["people"])
    write_json(out / "people_taxonomy.json", pt)
    written.append(out / "people_taxonomy.json")

    co = json.loads((data / "collaborations.json").read_text(encoding="utf-8"))
    n0 = len(co)
    co = {k: v for k, v in co.items() if k not in removed_ids}  # their own entry; mentions in others' lists stay
    stats["collaboration_entries_removed"] = n0 - len(co)
    write_json(out / "collaborations.json", co)
    written.append(out / "collaborations.json")

    for name in ("patent_links.json", "paper_counts_geo.json"):  # production and paper properties: unchanged
        shutil.copy2(data / name, out / name)
        written.append(out / name)

    shards_in = args.person_papers or (data / "person_papers")
    shard_keys_removed = 0
    if shards_in.is_dir():
        (out / "person_papers").mkdir(exist_ok=True)
        for f in sorted(shards_in.glob("*.json")):
            d = json.loads(f.read_text(encoding="utf-8"))
            for k in [k for k in d if k in removed_ids]:
                del d[k]
                shard_keys_removed += 1
            write_json(out / "person_papers" / f.name, d)
            written.append(out / "person_papers" / f.name)
    stats["paper_shard_entries_removed"] = shard_keys_removed

    # --- safety checks: fail the export, leaving nothing staged, if a removed person is still disseminated
    problems = []
    if removed_orcids:
        needle = re.compile("|".join(re.escape(o) for o in removed_orcids), re.IGNORECASE)
        for f in written:
            if needle.search(f.read_text(encoding="utf-8")):
                problems.append(f"a removed ORCID is still present in {f.name}")
    leftover = [p for p in pv["people"] + pt["people"] if {p.get("id"), *(p.get("merged_from") or [])} & removed_ids]
    if leftover:
        problems.append(f"{len(leftover)} removed people are still in a people array")
    if set(co) & removed_ids:
        problems.append("a removed person still has a collaboration entry")
    if problems:
        for f in written:
            f.unlink(missing_ok=True)
        sys.exit("OPT-OUT CHECK FAILED, nothing staged:\n  " + "\n  ".join(problems))

    stats.update({"people_in_export": len(pv["people"]), "ids_suppressed": len(removed_ids), "orcids_suppressed": len(removed_orcids),
                  "checks": "passed"})
    (data / "optout_report.json").write_text(json.dumps(stats, indent=2) + "\n", encoding="utf-8")
    print("opt-outs applied (counts only):", json.dumps(stats))


if __name__ == "__main__":
    main()
