"""SweMetrics enrichment, validation report: precision per admission route, and what the gate left out.

Reads docs/validation/swe/sample_map.json and the blind labels in swe/labels_haiku/, and prints, per stratum, the
share labelled CORE / PERIPHERAL / NOT with a 95 % Wilson interval. For the admitted routes the CORE share is the
route's precision. For the control strata it estimates how many CORE works the gate left out, scaled to the size of
the stratum in the full pool (corpus/data/swe_decisions.json, if present).
"""
import glob
import json
import math
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
V = ROOT / "docs" / "validation" / "swe"
mp = json.load(open(V / "sample_map.json"))
lab = {g["item"]: g["label"] for p in sorted(glob.glob(str(V / "labels_haiku" / "batch_*.json"))) for g in json.load(open(p))}

def wilson(k, n, z=1.96):
    if not n: return (0, 0)
    p = k / n; d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d; h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return max(0, c - h), min(1, c + h)

full = {}
dec = ROOT / "corpus" / "data" / "swe_decisions.json"
if dec.exists():
    pool = {json.loads(l)["id"]: json.loads(l) for l in open(ROOT / "corpus" / "data" / "swe_pool.jsonl", encoding="utf-8")}
    d = json.load(open(dec))
    def route(w):
        x, s = d[w], pool[w]["sources"][0]
        if x["admitted"]: return "swepub_country" if s == "swepub_country" else "swepub_only" if s == "swepub_only" else "clf_oa" if x["stage"] == "classifier" else "eikos_oa"
        return "band_rejected" if x["stage"] == "eikos" else "low_score" if x["stage"] == "rejected" else None
    for w in d:
        full[route(w)] = full.get(route(w), 0) + 1

order = ["clf_oa", "eikos_oa", "swepub_country", "swepub_only", "band_rejected", "low_score"]
print(f"labelled {len(lab)} of {len(mp)} items\n")
print(f"{'stratum':<16}{'n':>4}   {'CORE':>6}  {'95% interval':<13}{'PERIPH':>7}{'NOT':>7}   works in pool")
tot = {"admitted": [0, 0], "ctrl": 0.0}
for s in order:
    ks = [k for k, v in mp.items() if v["stratum"] == s and k in lab]
    if not ks: continue
    c = {l: sum(lab[k] == l for k in ks) for l in ("CORE", "PERIPHERAL", "NOT")}
    lo, hi = wilson(c["CORE"], len(ks))
    print(f"{s:<16}{len(ks):>4}   {c['CORE'] / len(ks):6.0%}  [{lo:4.0%}, {hi:4.0%}]  {c['PERIPHERAL'] / len(ks):7.0%}{c['NOT'] / len(ks):7.0%}   {full.get(s, '')}")
    if s in ("band_rejected", "low_score") and full.get(s):
        est = c["CORE"] / len(ks) * full[s]
        tot["ctrl"] += est
        print(f"{'':<16}      -> about {est:.0f} CORE works left out in this stratum (range {lo * full[s]:.0f}-{hi * full[s]:.0f})")
    elif s not in ("band_rejected", "low_score") and full.get(s):
        tot["admitted"][0] += c["CORE"] / len(ks) * full[s]; tot["admitted"][1] += full[s]
if tot["admitted"][1]:
    print(f"\nadmitted overall: about {tot['admitted'][0]:.0f} of {tot['admitted'][1]} admitted works are CORE ({tot['admitted'][0] / tot['admitted'][1]:.0%} precision, weighted by route size)")
    print(f"left out: about {tot['ctrl']:.0f} CORE works among the rejected (so recall of the gate on this pool is about {tot['admitted'][0] / (tot['admitted'][0] + tot['ctrl']):.0%})")
