"""Step 5: apply the reviewed verdicts.csv to people_viz.json — fold every
non-primary record in a FUSE cluster into its suggested_primary_id and drop
the rest. REJECT/REVIEW rows are left untouched (this only acts on FUSE).

Hard veto: a FUSE cluster whose members carry 2+ DIFFERENT non-null ORCIDs
is skipped outright, regardless of verdict. ORCID is a verified personal
identifier, so that's proof of multiple distinct real people (typically the
same common-surname small research community sharing enough coauthors to
trip the jaccard threshold — Andersson, Nielsen, Hansen, Zhang, etc.), not
one person split across records. Found the hard way: 43 of the first 151
FUSE clusters merged this way turned out to have conflicting ORCIDs.

Field-merge rules (each chosen because the underlying per-record fields were
computed over a DISJOINT subset of the same real person's papers — OpenAlex
split their profile, it didn't duplicate their papers — so most stats are
correctly additive, not double-counted):

  total_papers, quantum_papers, matched_papers, total_patents,
  quantum_patents          -> summed across the cluster
  patents (array)          -> unioned, deduped by epo_id
  yearly (per-year papers/citations) -> summed per year, keyed union
  first_year / last_year   -> min / max across the cluster
  h_index, quantum_h_index,
  global_h_index           -> max across the cluster (each is itself computed
                               over a subset of the true combined corpus, so
                               the true merged h-index is >= every member's;
                               max is the conservative estimate available
                               without recomputing h-index from raw citations)
  quantum_share, x, x_raw,
  confidence               -> recomputed EXACTLY from the merged
                               quantum_papers/total_papers via the same
                               formula build_people_data.py uses (not an
                               approximation)
  y, y_raw, sensing_share,
  communication_share,
  computing_share          -> quantum_papers-weighted average across the
                               cluster (these come from NLP/embedding scoring
                               over each member's own paper subset, which
                               can't be cheaply recomputed here; a weighted
                               average over the disjoint subsets approximates
                               what re-scoring the union would give)
  name                     -> the longest name string among the cluster
                               (most complete display form, e.g. "Valéry
                               Zwiller" over "V. Zwiller")
  orcid                    -> primary's, falling back to the first non-null
                               ORCID found among the cluster
  topics, keywords,
  example_titles           -> union, primary's own entries first, deduped,
                               capped to the original list length
  institution, university,
  country, tier            -> primary's (already the majority/highest-
                               evidence value for the surviving id)
  id                       -> primary's id; a `merged_from` field records the
                               dropped ids for provenance
"""
import json
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent
VIZ = ROOT.parent / "pipeline" / "data" / "people_viz.json"
OUT_DIR = ROOT / "out"

verdicts = pd.read_csv(OUT_DIR / "verdicts.csv")
fuse_rows = verdicts[verdicts["verdict"] == "FUSE"]
print(f"{len(fuse_rows)} FUSE clusters to merge")

data = json.loads(VIZ.read_text(encoding="utf-8"))
people = data["people"]
by_id = {p["id"]: p for p in people}

drop_ids: set[str] = set()
n_merged_people = 0
skipped_orcid_conflict: list[tuple[str, list[str]]] = []

for _, row in fuse_rows.iterrows():
    ids = [i.strip() for i in row["ids"].split("|")]
    ids = [i for i in ids if i in by_id]
    if len(ids) < 2:
        continue
    primary_id = row["suggested_primary_id"].strip()
    if primary_id not in by_id:
        primary_id = ids[0]
    others = [i for i in ids if i != primary_id]

    primary = by_id[primary_id]
    members = [by_id[i] for i in ids]

    # Hard veto: two or more DIFFERENT non-null ORCIDs among the cluster's
    # members is proof of multiple distinct real people (ORCID is a verified
    # personal identifier) — overrides the coauthor-jaccard verdict no
    # matter how high, since jaccard alone can't distinguish "same person"
    # from "same lab / same common surname's small research community".
    distinct_orcids = {m["orcid"] for m in members if m.get("orcid")}
    if len(distinct_orcids) > 1:
        skipped_orcid_conflict.append((row["names"], sorted(distinct_orcids)))
        continue

    def sum_field(key: str) -> int:
        return sum(int(m.get(key) or 0) for m in members)

    total_papers = sum_field("total_papers")
    quantum_papers = sum_field("quantum_papers")
    matched_papers = sum_field("matched_papers")
    total_patents = sum_field("total_patents")
    quantum_patents = sum_field("quantum_patents")

    # quantum_share/x/x_raw/confidence: exact recompute, same formula as
    # build_people_data.py's quantum_share_score + confidence shrinkage.
    CONFIDENCE_K = 2  # verified against existing x/x_raw/confidence triples
    if total_papers > 0:
        share = max(0.0, min(1.0, quantum_papers / total_papers))
    else:
        share = 0.0
    x_raw = max(-1.0, min(1.0, 2.0 * share - 1.0))
    conf = quantum_papers / (quantum_papers + CONFIDENCE_K) if (quantum_papers + CONFIDENCE_K) else 0.0
    x = x_raw * conf

    # y/y_raw/domain shares: quantum_papers-weighted average (see docstring)
    def weighted_avg(key: str):
        total_w = 0
        acc = 0.0
        for m in members:
            v = m.get(key)
            if v is None:
                continue
            w = int(m.get("quantum_papers") or 0) or 1
            acc += v * w
            total_w += w
        return round(acc / total_w, 4) if total_w else None

    y_raw = weighted_avg("y_raw")
    y = y_raw * conf if y_raw is not None else None
    sensing_share = weighted_avg("sensing_share")
    communication_share = weighted_avg("communication_share")
    computing_share = weighted_avg("computing_share")

    first_year = min((m["first_year"] for m in members if m.get("first_year") is not None), default=None)
    last_year = max((m["last_year"] for m in members if m.get("last_year") is not None), default=None)

    h_index = max((m["h_index"] for m in members if m.get("h_index") is not None), default=None)
    quantum_h_index = max((m["quantum_h_index"] for m in members if m.get("quantum_h_index") is not None), default=None)
    global_h_index = max((m["global_h_index"] for m in members if m.get("global_h_index") is not None), default=None)

    # yearly: sum papers/citations per year across the cluster
    yearly: dict[str, dict[str, int]] = {}
    for m in members:
        for yr, v in (m.get("yearly") or {}).items():
            slot = yearly.setdefault(yr, {"papers": 0, "citations": 0})
            slot["papers"] += int(v.get("papers") or 0)
            slot["citations"] += int(v.get("citations") or 0)
    yearly = dict(sorted(yearly.items()))

    # patents: union, deduped by epo_id
    patents_by_epo = {}
    for m in members:
        for p in m.get("patents") or []:
            patents_by_epo.setdefault(p["epo_id"], p)
    patents = sorted(patents_by_epo.values(), key=lambda p: -(p.get("year") or 0))

    def union_capped(key: str) -> list:
        cap = len(primary.get(key) or []) or 5
        seen: list = []
        for m in [primary] + [mm for mm in members if mm is not primary]:
            for v in m.get(key) or []:
                if v not in seen:
                    seen.append(v)
        return seen[:cap]

    topics = union_capped("topics")
    keywords = union_capped("keywords")
    example_titles = union_capped("example_titles")

    name = max((m["name"] for m in members), key=len)
    orcid = primary.get("orcid") or next((m.get("orcid") for m in members if m.get("orcid")), None)

    primary.update(
        {
            "name": name,
            "orcid": orcid,
            "total_papers": total_papers,
            "quantum_papers": quantum_papers,
            "matched_papers": matched_papers,
            "total_patents": total_patents,
            "quantum_patents": quantum_patents,
            "patents": patents,
            "yearly": yearly,
            "first_year": first_year,
            "last_year": last_year,
            "h_index": h_index,
            "quantum_h_index": quantum_h_index,
            "global_h_index": global_h_index,
            "quantum_share": round(share, 4),
            "x": round(x, 4),
            "x_raw": round(x_raw, 4),
            "confidence": round(conf, 4),
            "y": y,
            "y_raw": y_raw,
            "sensing_share": sensing_share,
            "communication_share": communication_share,
            "computing_share": computing_share,
            "topics": topics,
            "keywords": keywords,
            "example_titles": example_titles,
            "topic": topics[0] if topics else primary.get("topic"),
            "merged_from": sorted(set(primary.get("merged_from") or []) | set(others)),
        }
    )
    drop_ids.update(others)
    n_merged_people += len(others)

new_people = [p for p in people if p["id"] not in drop_ids]
data["people"] = new_people
data.setdefault("meta", {})["n_people"] = len(new_people)

VIZ.write_text(json.dumps(data, ensure_ascii=False, indent=1), encoding="utf-8")
print(f"merged {n_merged_people} records into their primaries; {len(people)} -> {len(new_people)} people")
print(f"skipped {len(skipped_orcid_conflict)} FUSE clusters on the ORCID-conflict veto:")
for names, orcids in skipped_orcid_conflict:
    print(f"  - {names}  ({len(orcids)} distinct ORCIDs)")
print(f"wrote {VIZ}")
