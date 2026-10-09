"""Old (Sprint 13) corpus vs new corpus, and why old works are missing from the new one."""
import csv, json, collections

H = str(__import__("pathlib").Path(__file__).resolve().parents[3].parent) + "/"
OLD = H + "Qsip/SPRINT 13/data/all_universities_works.jsonl"
NEW = H + "open-alice-data/corpus/data/sweden_quantum_works.jsonl"
AUDIT = H + "open-alice-data/corpus/data/sweden_quantum_candidate_audit.csv"

old = {}
for l in open(OLD, encoding="utf-8"):
    w = json.loads(l)
    old[w["id"]] = (w.get("publication_year"), w.get("type"), w.get("language"))
new = {}
for l in open(NEW, encoding="utf-8"):
    w = json.loads(l)
    new[w["id"]] = (w.get("publication_year"), w.get("type"), w.get("language"))
print(f"old corpus works {len(old)} | new {len(new)} | both {len(old.keys() & new.keys())} | "
      f"old-only {len(old.keys() - new.keys())} | new-only {len(new.keys() - old.keys())}")

old_only = old.keys() - new.keys()
rd = csv.DictReader(open(AUDIT, encoding="utf-8"))
print("audit columns:", rd.fieldnames)
status = {}
for r in rd:
    wid = r.get("work_id") or r.get("id")
    if wid and not wid.startswith("http"):
        wid = "https://openalex.org/" + wid
    status[wid] = r
reasons = collections.Counter()
for w in old_only:
    r = status.get(w)
    if r is None:
        reasons["NOT a candidate in the new run (topic/keyword retrieval no longer finds it)"] += 1
    else:
        key = next((r[c] for c in ("exclusion_reason", "reason", "decision", "excluded_reason") if c in r and r[c]), "excluded (no reason column)")
        reasons[f"candidate, excluded: {key}"] += 1
for k, v in reasons.most_common(12):
    print(f"  {v:6d}  {k}")

def decade(y):
    return f"{(y or 0)//10*10}s"
print("old-only by decade:", sorted(collections.Counter(decade(old[w][0]) for w in old_only).items())[-6:])
print("old-only by type:", collections.Counter(old[w][1] for w in old_only).most_common(5))
print("old-only by language:", collections.Counter(old[w][2] for w in old_only).most_common(5))
print("new-only by decade:", sorted(collections.Counter(decade(new[w][0]) for w in new.keys() - old.keys()).items())[-6:])
