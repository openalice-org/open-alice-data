"""Rescue pass, validation: draw a blind sample for the Haiku swarm.

Three strata of 100 works each, shuffled together with the stratum hidden from the labelers:
  extension   band works Eikos calls CORE with P(CORE) in [0.5, 0.7): the works a lower floor would add
  admitted    band works Eikos calls CORE with P(CORE) >= 0.7:       the works the floor admits
  classifier  works the classifier admits on its own (P(core) >= accept threshold)
Their CORE rate per stratum is the precision of each admission route, and it decides the floor.

Writes docs/validation/rescue/batches/batch_NN.jsonl (50 items each) and sample_map.json.
"""
import json
import random
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
DATA = ROOT / "corpus" / "data"
OUT = ROOT / "docs" / "validation" / "rescue"
N = 100
rnd = random.Random(20261008)

band = {json.loads(l)["item"]: json.loads(l) for l in open(DATA / "rescue_band.jsonl", encoding="utf-8")}
lab = {x["item"]: x for x in json.load(open(DATA / "rescue_band_eikos.json"))}
core = [(lab[i]["probs"][0], band[i]) for i in lab if lab[i]["label"] == "CORE"]
ext = [b for p, b in core if 0.5 <= p < 0.7]
adm = [b for p, b in core if p >= 0.7]
train = json.load(open(DATA / "rescue_train_manifest.json"))
accept = train["thresholds"]["accept"]

import numpy as np
z = np.load(DATA / "rescue_scores.npz")
accepted_ids = {i for i, p in zip(z["ids"].tolist(), z["p_core"].tolist()) if p >= accept}
clf = [json.loads(l) for l in open(DATA / "rescue_pool.jsonl", encoding="utf-8")]
clf = [{"id": w["id"], "year": w["publication_year"], "title": w["display_name"], "abstract": w["abstract"]}
       for w in clf if w["id"] in accepted_ids]
print(f"available: extension {len(ext)}, admitted {len(adm)}, classifier {len(clf)}")

rows = [("extension", w) for w in rnd.sample(ext, min(N, len(ext)))] + \
       [("admitted", w) for w in rnd.sample(adm, min(N, len(adm)))] + \
       [("classifier", w) for w in rnd.sample(clf, min(N, len(clf)))]
rnd.shuffle(rows)
mp = {}
for k in range(0, len(rows), 50):
    with open(OUT / "batches" / f"batch_{k // 50 + 1:02d}.jsonl", "w", encoding="utf-8") as f:
        for j, (stratum, w) in enumerate(rows[k:k + 50], k + 1):
            item = f"rv_{j:04d}"
            mp[item] = {"stratum": stratum, "work_id": w["id"]}
            f.write(json.dumps({"item": item, "year": w["year"], "title": w["title"], "abstract": w["abstract"][:1400]},
                               ensure_ascii=False) + "\n")
(OUT / "sample_map.json").write_text(json.dumps(mp, indent=1))
print(f"wrote {len(rows)} items in {(len(rows) + 49) // 50} batches")
