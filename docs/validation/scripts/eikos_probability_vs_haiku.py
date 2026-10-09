"""Does Eikos's own probability tell good CORE calls from weak ones? (audit set, 500 papers)

Bins Eikos's P(CORE), for the papers Eikos called CORE, and shows how often Haiku and Sonnet
agree. This is the evidence behind EIKOS_MIN_P in pipeline/rescue_apply.py.
"""
import glob
import json
from pathlib import Path

DATA = Path(__file__).resolve().parents[1] / "data"
hk = {g["item"]: g["label"] for p in glob.glob(str(DATA / "labels_haiku/batch_*.json")) for g in json.load(open(p))}
so = {g["item"]: g["label"] for g in json.load(open(DATA / "labels_sonnet/batch_01.json"))}
ek = {x["item"]: x for x in json.load(open(DATA / "eikos_labels.json"))}
core = [(x["probs"][0], i) for i, x in ek.items() if x["label"] == "CORE"]
print("Eikos CORE calls on the audit set:", len(core))
print("P(CORE) bin     n   Haiku=CORE   Haiku=NOT   Sonnet=CORE (where labelled)")
for lo, hi in [(0.0, 0.5), (0.5, 0.6), (0.6, 0.7), (0.7, 0.8), (0.8, 1.01)]:
    its = [i for p, i in core if lo <= p < hi]
    if not its:
        continue
    h = sum(hk[i] == "CORE" for i in its) / len(its)
    n = sum(hk[i] == "NOT" for i in its) / len(its)
    s = [i for i in its if i in so]
    sc = (sum(so[i] == "CORE" for i in s) / len(s)) if s else float("nan")
    print(f"  [{lo:.1f},{min(hi, 1):.1f})  {len(its):4d}     {h:6.0%}      {n:6.0%}      {sc:6.0%} (n={len(s)})")
