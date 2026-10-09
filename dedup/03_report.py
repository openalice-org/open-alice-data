"""Step 3: cluster splink's pairwise predictions into candidate duplicate
groups, enrich the top ones with real coauthor overlap (scanned from the
underlying per-university classified-works corpus), and write a markdown
report.
"""
import json
from collections import defaultdict
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent
OUT_DIR = ROOT / "out"
SPRINT13 = ROOT.parent / "pipeline"

TOP_N_ENRICHED = 40  # how many clusters get the expensive coauthor scan
MIN_PROB_FOR_REPORT = 0.5

people = pd.read_pickle(OUT_DIR / "people.pkl").set_index("id")
preds = pd.read_pickle(OUT_DIR / "predictions.pkl")


# ---- union-find over pairs at/above MIN_PROB_FOR_REPORT, so a name split
# across 3+ OpenAlex ids (e.g. I.C. Arsene above) reports as ONE cluster.
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

# best (max) pairwise probability observed within each cluster, for ranking
cluster_best_prob: dict[str, float] = defaultdict(float)
for _, r in strong.iterrows():
    root = find(r["id_l"])
    cluster_best_prob[root] = max(cluster_best_prob[root], r["match_probability"])

ranked_roots = sorted(clusters, key=lambda r: -cluster_best_prob[r])
print(f"{len(ranked_roots)} candidate clusters at match_probability >= {MIN_PROB_FOR_REPORT}")

# ---- ORCID collisions (near-certain, computed in step 1) merge in as their
# own top-priority clusters, deduped against anything splink already found.
orcid_dupes = json.loads((OUT_DIR / "orcid_dupes.json").read_text())


def load_uni_coauthors(uni_key: str, needed_ids: set[str]) -> dict[str, set[str]]:
    """Scan one university's classified-works jsonl once, returning
    {person_id: {coauthor_id, ...}} for just the ids we need."""
    path = SPRINT13 / "data" / uni_key / "quantum_classified_works_specter_reviewed.jsonl"
    result: dict[str, set[str]] = {pid: set() for pid in needed_ids}
    if not path.exists():
        return result
    with open(path, encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            # cheap prefilter before the (relatively) expensive json.loads
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


def uni_key_for(university_label: str) -> str | None:
    universities = json.loads((SPRINT13 / "universities.json").read_text())
    for key, u in universities.items():
        if u.get("label") == university_label:
            return key
    return None


UNIVERSITIES_JSON = json.loads((SPRINT13 / "universities.json").read_text())
LABEL_TO_KEY = {u["label"]: key for key, u in UNIVERSITIES_JSON.items()}

# ---- group top clusters' member ids by which university file needs scanning
top_roots = ranked_roots[:TOP_N_ENRICHED]
ids_by_uni: dict[str, set[str]] = defaultdict(set)
for root in top_roots:
    for pid in clusters[root]:
        uni_label = people.loc[pid, "university"] if pid in people.index else None
        key = LABEL_TO_KEY.get(uni_label)
        if key:
            ids_by_uni[key].add(pid)

coauthors: dict[str, set[str]] = {}
for uni_key, ids in ids_by_uni.items():
    print(f"scanning {uni_key} for {len(ids)} ids...")
    coauthors.update(load_uni_coauthors(uni_key, ids))


def fmt_person(pid: str) -> dict:
    if pid not in people.index:
        return {"id": pid, "name": "?"}
    r = people.loc[pid]
    return {
        "id": pid,
        "name": r["name"],
        "university": r["university"],
        "country": r["country"],
        "institution": r["institution"],
        "first_year": r["first_year"],
        "last_year": r["last_year"],
        "quantum_papers": int(r["quantum_papers"]),
        "topics": r["topics"][:3],
    }


lines = ["# Potential duplicate people — splink experiment\n"]
lines.append(
    f"Blocked on normalized surname (9,352 people -> {len(preds)} scored pairs), "
    f"probabilistic model trained unsupervised (EM) on name similarity, first/last "
    f"publication year, country/university/institution match, and topic/keyword "
    f"overlap. {len(ranked_roots)} clusters scored >= {MIN_PROB_FOR_REPORT}; "
    f"showing the top {min(TOP_N_ENRICHED, len(ranked_roots))}, enriched with real "
    "coauthor overlap scanned from the underlying paper corpus.\n"
)

lines.append("## Exact-ORCID collisions (near-certain — same ORCID, different OpenAlex id)\n")
if orcid_dupes:
    for group in orcid_dupes:
        names = {p["name"] for p in group["people"]}
        flag = "" if len(names) == 1 else " ⚠️ **different names on the same ORCID — likely an OpenAlex ORCID mis-link, not a duplicate person**"
        lines.append(f"- `{group['orcid']}`{flag}")
        for p in group["people"]:
            lines.append(f"  - {p['name']} — {p['university']} — `{p['id']}`")
else:
    lines.append("(none found)")
lines.append("")

lines.append(f"## Top {min(TOP_N_ENRICHED, len(ranked_roots))} splink clusters\n")
for i, root in enumerate(top_roots, 1):
    members = sorted(clusters[root])
    prob = cluster_best_prob[root]
    lines.append(f"### {i}. {people.loc[members[0], 'name'] if members[0] in people.index else '?'} — best match_probability {prob:.3f} ({len(members)} records)\n")
    for pid in members:
        info = fmt_person(pid)
        lines.append(
            f"- **{info['name']}** — {info.get('university')}, {info.get('country')} — "
            f"active {info.get('first_year')}–{info.get('last_year')} — "
            f"{info.get('quantum_papers')} quantum papers — `{pid}`"
        )
    # pairwise coauthor overlap across members that have coauthor data
    pairs_with_data = [pid for pid in members if pid in coauthors]
    if len(pairs_with_data) >= 2:
        lines.append("  \n  Coauthor overlap:")
        for a_i in range(len(pairs_with_data)):
            for b_i in range(a_i + 1, len(pairs_with_data)):
                a, b = pairs_with_data[a_i], pairs_with_data[b_i]
                ca, cb = coauthors.get(a, set()), coauthors.get(b, set())
                shared = ca & cb
                union_n = len(ca | cb) or 1
                jaccard = len(shared) / union_n
                lines.append(
                    f"  - `{a[-8:]}` vs `{b[-8:]}`: {len(shared)} shared coauthors "
                    f"(of {len(ca)}/{len(cb)}), jaccard={jaccard:.2f}"
                )
    lines.append("")

report_path = OUT_DIR / "REPORT.md"
report_path.write_text("\n".join(lines), encoding="utf-8")
print(f"wrote {report_path}")
