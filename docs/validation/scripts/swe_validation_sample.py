"""SweMetrics enrichment, validation: draw a blind sample for independent labelling.

Strata (shuffled together; the stratum is hidden from the labellers):
  admitted routes     40 each, the precision of every way a work enters the corpus
    clf_oa            admitted by the classifier alone, found by the OpenAlex text net
    eikos_oa          admitted after Eikos adjudicated the uncertain band, OpenAlex text net
    swepub_country    OpenAlex lacked the Nordic country; Swepub is the only evidence the work is Swedish
    swepub_only       no OpenAlex record at all (dissertations, reports, ...)
  controls            what the gate left out
    band_rejected     40 uncertain-band works Eikos did NOT admit (a CORE rate here = quality lost)
    low_score         30 works the classifier scored below the reject threshold (expected: almost none CORE)

Writes docs/validation/swe/batches/batch_NN.jsonl (50 items each) and sample_map.json. Run after `make swe-apply`.
"""
import json
import random
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
DATA = ROOT / "corpus" / "data"
OUT = ROOT / "docs" / "validation" / "swe"
rnd = random.Random(20261127)

pool = {}
for line in open(DATA / "swe_pool.jsonl", encoding="utf-8"):
    r = json.loads(line)
    pool[r["id"]] = r
dec = json.load(open(DATA / "swe_decisions.json"))

def route(wid):
    d, src = dec[wid], pool[wid]["sources"][0]
    if d["admitted"] and src == "swepub_country": return "swepub_country"
    if d["admitted"] and src == "swepub_only": return "swepub_only"
    if d["admitted"] and d["stage"] == "classifier": return "clf_oa"
    if d["admitted"] and d["stage"] == "eikos": return "eikos_oa"
    if d["stage"] == "eikos": return "band_rejected"
    if d["stage"] == "rejected": return "low_score"
    return None

by = {}
for wid in dec:
    by.setdefault(route(wid), []).append(wid)
N = {"clf_oa": 40, "eikos_oa": 40, "swepub_country": 40, "swepub_only": 40, "band_rejected": 40, "low_score": 30}
print({k: (len(by.get(k, [])), N[k]) for k in N})
rows = [(s, wid) for s, n in N.items() for wid in rnd.sample(by[s], min(n, len(by[s])))]
rnd.shuffle(rows)
mp = {}
for k in range(0, len(rows), 50):
    with open(OUT / "batches" / f"batch_{k // 50 + 1:02d}.jsonl", "w", encoding="utf-8") as f:
        for j, (stratum, wid) in enumerate(rows[k:k + 50], k + 1):
            item = f"sv_{j:04d}"
            w = pool[wid]
            mp[item] = {"stratum": stratum, "work_id": wid, "p_core": dec[wid]["p_core"]}
            f.write(json.dumps({"item": item, "year": w["publication_year"], "title": w["display_name"],
                                "abstract": w["abstract"][:1400]}, ensure_ascii=False) + "\n")
(OUT / "sample_map.json").write_text(json.dumps(mp, indent=1))
print(f"wrote {len(rows)} items in {(len(rows) + 49) // 50} batches")
