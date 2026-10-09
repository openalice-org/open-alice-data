"""Step 1: load people_viz.json, normalize fields, check exact-ORCID
duplicates, and build blocked candidate pairs (same normalized surname) for
the splink model in 02_link.py.
"""
import json
import re
import unicodedata
from collections import defaultdict
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent
VIZ = ROOT.parent / "pipeline" / "data" / "people_viz.json"
OUT_DIR = ROOT / "out"
OUT_DIR.mkdir(exist_ok=True)


def strip_accents(s: str) -> str:
    return "".join(c for c in unicodedata.normalize("NFD", s) if unicodedata.category(c) != "Mn")


def norm_name(name: str) -> str:
    n = strip_accents(name or "").lower()
    n = re.sub(r"[^a-z\s\-']", " ", n)
    return re.sub(r"\s+", " ", n).strip()


def surname_key(name: str) -> str:
    parts = norm_name(name).split()
    if not parts:
        return ""
    # last token as surname; drop short particles like "de"/"van"/"la" from
    # being used ALONE as the key by preferring the last non-particle token.
    particles = {"de", "van", "der", "den", "la", "le", "di", "von", "af", "el"}
    for tok in reversed(parts):
        if tok not in particles and len(tok) > 1:
            return tok
    return parts[-1]


def main():
    data = json.loads(VIZ.read_text(encoding="utf-8"))
    people = data["people"]
    print(f"loaded {len(people)} people")

    rows = []
    for p in people:
        rows.append(
            {
                "id": p["id"],
                "name": p["name"],
                "norm_name": norm_name(p["name"]),
                "surname_key": surname_key(p["name"]),
                "orcid": (p.get("orcid") or "").strip() or None,
                "institution": p.get("institution") or "",
                "university": p.get("university") or "",
                "country": p.get("country") or "",
                "topic": p.get("topic") or "",
                "topics": p.get("topics") or [],
                "keywords": p.get("keywords") or [],
                "first_year": p.get("first_year"),
                "last_year": p.get("last_year"),
                "quantum_papers": p.get("quantum_papers") or 0,
                "total_papers": p.get("total_papers") or 0,
                "h_index": p.get("h_index"),
            }
        )
    df = pd.DataFrame(rows)
    df.to_pickle(OUT_DIR / "people.pkl")
    print(f"wrote {OUT_DIR / 'people.pkl'} ({len(df)} rows)")

    # ---- exact-ORCID duplicates: two different OpenAlex ids sharing one
    # non-null ORCID is about as close to certain as this dataset gets.
    by_orcid = defaultdict(list)
    for p in people:
        oc = (p.get("orcid") or "").strip()
        if oc:
            by_orcid[oc].append(p)
    orcid_dupes = {oc: ps for oc, ps in by_orcid.items() if len(ps) > 1}
    print(f"ORCID collisions (same ORCID, different id): {len(orcid_dupes)} groups")
    with open(OUT_DIR / "orcid_dupes.json", "w", encoding="utf-8") as f:
        json.dump(
            [
                {"orcid": oc, "people": [{"id": p["id"], "name": p["name"], "university": p.get("university")} for p in ps]}
                for oc, ps in orcid_dupes.items()
            ],
            f,
            indent=2,
            ensure_ascii=False,
        )

    # ---- blocked candidate pairs: same surname_key, different id.
    by_surname = defaultdict(list)
    for _, r in df.iterrows():
        if r["surname_key"]:
            by_surname[r["surname_key"]].append(r["id"])

    pairs = []
    for key, ids in by_surname.items():
        if len(ids) < 2:
            continue
        for i in range(len(ids)):
            for j in range(i + 1, len(ids)):
                pairs.append((ids[i], ids[j]))
    print(f"candidate pairs (same surname key): {len(pairs)}")
    pd.DataFrame(pairs, columns=["id_l", "id_r"]).to_pickle(OUT_DIR / "candidate_pairs.pkl")
    print(f"wrote {OUT_DIR / 'candidate_pairs.pkl'}")


if __name__ == "__main__":
    main()
