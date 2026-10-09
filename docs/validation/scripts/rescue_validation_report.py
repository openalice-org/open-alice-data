"""Rescue pass, validation: precision of each admission route, from the blind Haiku labels.

Reads docs/validation/rescue/{sample_map.json, labels_haiku/*.json} and the Eikos band labels, and
reports for each route the share of works Haiku labels CORE (the route's precision), plus how many
works each Eikos probability floor would admit from the whole labelled band.
Precision here counts only CORE; PERIPHERAL is not rescued by design (see pipeline/rescue_rubric.py).
"""
import collections
import glob
import json
import math
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
V = ROOT / "docs" / "validation" / "rescue"
mp = json.load(open(V / "sample_map.json"))
hk = {g["item"]: g["label"] for p in sorted(glob.glob(str(V / "labels_haiku/batch_*.json"))) for g in json.load(open(p))}
print(f"labelled by Haiku: {len(hk)} of {len(mp)}")
by = collections.defaultdict(list)
for item, m in mp.items():
    if item in hk:
        by[m["stratum"]].append(hk[item])
print("\nroute        n   CORE   PERIPHERAL   NOT    precision (95% Wilson)")
for s in ("classifier", "admitted", "extension"):
    c = collections.Counter(by[s]); n = len(by[s])
    if not n:
        continue
    p = c["CORE"] / n
    z = 1.96; centre = (p + z * z / (2 * n)) / (1 + z * z / n)
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / (1 + z * z / n)
    print(f"{s:11s} {n:3d}  {c['CORE']:4d}   {c['PERIPHERAL']:9d}   {c['NOT']:4d}    {p:5.0%}  ({centre - half:.0%}-{centre + half:.0%})")
band = json.load(open(ROOT / "corpus" / "data" / "rescue_band_eikos.json"))
core = [x["probs"][0] for x in band if x["label"] == "CORE"]
print("\nEikos CORE calls in the 6,000-work band by P(CORE) floor:")
for fl in (0.5, 0.6, 0.7, 0.8):
    print(f"  floor {fl:.1f}: admits {sum(p >= fl for p in core)} works")
