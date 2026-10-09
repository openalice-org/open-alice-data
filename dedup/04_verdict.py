"""Step 4: classify ALL 452 splink clusters (not just the top-40 enriched in
03_report.py) as worth-fusing or not, using real coauthor overlap as the
decisive signal — a name/metadata match with near-zero coauthor overlap is
almost always just a common name, not a split profile (see the "H. Smith"
case in the main report).

Verdict rule (deliberately simple/legible, not another trained model):
  - FUSE:    max pairwise coauthor jaccard >= 0.15, OR a solo/no-paper member
             (0 coauthors on both sides — can't be contradicted, name+metadata
             match stands) paired with best_match_probability >= 0.9
  - REVIEW:  0 < max jaccard < 0.15 (name matches, weak/ambiguous overlap)
  - REJECT:  max jaccard == 0 with BOTH members having a nonzero coauthor set
             (real, disjoint collaboration networks under the same name)
"""
import json
import re
import unicodedata
from collections import defaultdict
from pathlib import Path

import pandas as pd


def strip_accents(s: str) -> str:
    return "".join(c for c in unicodedata.normalize("NFD", s) if unicodedata.category(c) != "Mn")


def first_initial(name: str) -> str:
    n = strip_accents(name or "").lower()
    n = re.sub(r"[^a-z\s]", " ", n)
    parts = n.split()
    return parts[0][0] if parts else ""

ROOT = Path(__file__).resolve().parent
OUT_DIR = ROOT / "out"
SPRINT13 = ROOT.parent / "pipeline"
MIN_PROB_FOR_REPORT = 0.5
FUSE_JACCARD = 0.15

people = pd.read_pickle(OUT_DIR / "people.pkl").set_index("id")
preds = pd.read_pickle(OUT_DIR / "predictions.pkl")

parent: dict[str, str] = {}


def find(x: str) -> str:
    parent.setdefault(x, x)
    while parent[x] != x:
        parent[x] = parent[parent[x]]
        x = parent[x]
    return x


def union(a: str, b: str) -> None:
    ra, rb = find(a), find(b)
    if ra != rb:
        parent[ra] = rb


strong = preds[preds["match_probability"] >= MIN_PROB_FOR_REPORT]
for _, r in strong.iterrows():
    union(r["id_l"], r["id_r"])

clusters: dict[str, set[str]] = defaultdict(set)
for pid in set(strong["id_l"]) | set(strong["id_r"]):
    clusters[find(pid)].add(pid)

cluster_best_prob: dict[str, float] = defaultdict(float)
for _, r in strong.iterrows():
    root = find(r["id_l"])
    cluster_best_prob[root] = max(cluster_best_prob[root], r["match_probability"])

print(f"{len(clusters)} clusters total")

UNIVERSITIES_JSON = json.loads((SPRINT13 / "universities.json").read_text())
LABEL_TO_KEY = {u["label"]: key for key, u in UNIVERSITIES_JSON.items()}

ids_by_uni: dict[str, set[str]] = defaultdict(set)
for members in clusters.values():
    for pid in members:
        uni_label = people.loc[pid, "university"] if pid in people.index else None
        key = LABEL_TO_KEY.get(uni_label)
        if key:
            ids_by_uni[key].add(pid)

print(f"scanning {len(ids_by_uni)} university corpora for {sum(len(v) for v in ids_by_uni.values())} ids...")


def load_uni_coauthors(uni_key: str, needed_ids: set[str]) -> dict[str, set[str]]:
    path = SPRINT13 / "data" / uni_key / "quantum_classified_works_specter_reviewed.jsonl"
    result: dict[str, set[str]] = {pid: set() for pid in needed_ids}
    if not path.exists():
        return result
    with open(path, encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            if not any(pid.rsplit("/", 1)[-1] in line for pid in needed_ids):
                continue
            try:
                work = json.loads(line)
            except json.JSONDecodeError:
                continue
            author_ids = set()
            for auth in work.get("authorships") or []:
                a = (auth or {}).get("author") or {}
                aid = a.get("id")
                if aid:
                    author_ids.add(aid)
            hit = author_ids & needed_ids
            for pid in hit:
                result[pid] |= author_ids - {pid}
    return result


coauthors: dict[str, set[str]] = {}
for uni_key, ids in ids_by_uni.items():
    coauthors.update(load_uni_coauthors(uni_key, ids))

rows = []
for root, members in clusters.items():
    members = sorted(members)
    prob = cluster_best_prob[root]
    pair_jaccards = []
    both_nonzero_seen = False
    for i in range(len(members)):
        for j in range(i + 1, len(members)):
            a, b = members[i], members[j]
            ca, cb = coauthors.get(a, set()), coauthors.get(b, set())
            if ca and cb:
                both_nonzero_seen = True
            union_n = len(ca | cb)
            jac = (len(ca & cb) / union_n) if union_n else None
            if jac is not None:
                pair_jaccards.append(jac)

    max_jac = max(pair_jaccards) if pair_jaccards else None
    if max_jac is not None and max_jac >= FUSE_JACCARD:
        verdict = "FUSE"
        reason = f"coauthor jaccard {max_jac:.2f} >= {FUSE_JACCARD}"
    elif max_jac == 0 and both_nonzero_seen:
        verdict = "REJECT"
        reason = "disjoint coauthor networks (both nonzero, zero overlap) — likely just a common name"
    elif max_jac is None and prob >= 0.9:
        verdict = "FUSE"
        reason = f"no coauthor data (0 matched papers); model probability {prob:.2f} >= 0.9"
    else:
        reason_bits = []
        if max_jac is not None:
            reason_bits.append(f"weak coauthor jaccard {max_jac:.2f}")
        else:
            reason_bits.append("no coauthor data")
        reason_bits.append(f"model probability {prob:.2f}")
        verdict = "REVIEW"
        reason = "; ".join(reason_bits)

    # Suggested surviving record: most quantum_papers (the corpus-match
    # evidence this dataset trusts most), tying on total_papers then h_index.
    # A pure suggestion — the reviewer can override by editing suggested_primary_id.
    def sort_key(pid: str):
        r = people.loc[pid] if pid in people.index else None
        if r is None:
            return (0, 0, 0)
        return (int(r["quantum_papers"] or 0), int(r["total_papers"] or 0), r["h_index"] or 0)

    primary_id = max(members, key=sort_key)

    member_names = sorted({people.loc[m, "name"] for m in members if m in people.index})
    initials = {first_initial(n) for n in member_names}
    initials.discard("")
    first_name_flag = "DIFFERENT FIRST INITIALS" if len(initials) > 1 else ""

    rows.append(
        {
            "cluster_id": root,
            "verdict": verdict,
            "flag": first_name_flag,
            "cluster_size": len(members),
            "best_match_probability": round(prob, 4),
            "max_coauthor_jaccard": round(max_jac, 3) if max_jac is not None else None,
            "reason": reason,
            "names": " | ".join(member_names),
            "suggested_primary_id": primary_id,
            "ids": " | ".join(members),
        }
    )

df = pd.DataFrame(rows).sort_values(["verdict", "best_match_probability"], ascending=[True, False])
df.to_csv(OUT_DIR / "verdicts.csv", index=False)

counts = df["verdict"].value_counts()
total_people_fused = df[df["verdict"] == "FUSE"]["cluster_size"].sum()
print("\n--- verdict counts (clusters) ---")
print(counts.to_string())
print(f"\nFUSE clusters cover {total_people_fused} OpenAlex records "
      f"({total_people_fused - (df['verdict'] == 'FUSE').sum()} would be removed if merged 1-per-cluster)")
print(f"\nwrote {OUT_DIR / 'verdicts.csv'}")
