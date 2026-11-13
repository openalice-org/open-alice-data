#!/usr/bin/env python3
"""Export per-person paper lists for the frontend's 3D paper explorer.

For every person in people_viz.json, writes the quantum-corpus papers they
authored with: publication year, citations, author count, DOI, title and the
paper's position in the Sensing/Communication/Computing triangle.

The triangle position is the paper's own domain distribution, computed exactly
as build_domain_specter.py does it (bge-large-en-v1.5 work embeddings from the
shared cache, anchor centroids, per-anchor centering, softmax / tau), then
renormalised over the three real domains (the "other" mass is dropped, as for
person shares). Centering means and tau are taken from people_taxonomy.json's
meta so the paper positions stay consistent with the published person shares.

Output: one JSON shard per (OpenAlex author number mod N_SHARDS), so the
frontend fetches ~0.5 MB when a person is opened rather than the whole set.

  shard file:  {"<person id>": [[wid, year, cites, n_authors, s, cm, cp, title, doi], ...]}
  wid   = OpenAlex work number (W123 -> 123)
  s/cm/cp = sensing / communication / computing share, integer per-mille (sum ~1000)

Run (from this directory, after build_domain_specter.py):
  ../.venv/bin/python3 build_person_papers.py --people PATH/people_viz.json --out DIR
"""

import argparse
import json
import os
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from build_domain_specter import (  # noqa: E402
    ALL_DOMAINS,
    IDS_CACHE,
    EMB_CACHE,
    load_anchors_config,
    make_embedder,
    softmax_rows,
)
from build_taxonomy_data import ACCEPTED_LABELS, SPRINT2_WORKS  # noqa: E402

HERE = Path(__file__).resolve().parent
TAXONOMY = HERE / "data" / "people_taxonomy.json"
WORK_LABELS = HERE / "data" / "work_domain_labels.json"
N_SHARDS = 64  # keep in sync with src/lib/personPapers.ts


def shard_of(person_id: str) -> int:
    return int(person_id.rsplit("/A", 1)[1]) % N_SHARDS


def num(openalex_id: str) -> int:
    return int(openalex_id.rsplit("/W", 1)[1])


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--people", required=True, help="people_viz.json")
    ap.add_argument("--out", required=True, help="output directory (person_papers/)")
    args = ap.parse_args()

    people = json.load(open(args.people, encoding="utf-8"))["people"]
    owner = {}  # author id (incl. merged duplicates) -> person id
    for p in people:
        owner[p["id"]] = p["id"]
        for m in p.get("merged_from") or []:
            owner[m] = p["id"]
    print(f"{len(people)} people, {len(owner)} author ids")

    # ---- pass 1: stream works, keep accepted ones with a tracked author ----
    meta = {}  # wid -> record
    work_people = defaultdict(set)
    with open(SPRINT2_WORKS, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                w = json.loads(line)
            except json.JSONDecodeError:
                continue
            if w.get("classification_label") not in ACCEPTED_LABELS:
                continue
            auths = w.get("authorships") or []
            owners = set()
            for a in auths:
                aid = (a.get("author") or {}).get("id") if isinstance(a, dict) else None
                if aid in owner:
                    owners.add(owner[aid])
            if not owners or not w.get("id"):
                continue
            meta[w["id"]] = (
                w.get("publication_year") or 0,
                w.get("cited_by_count") or 0,
                len(auths),
                (w.get("display_name") or "").strip(),
                (w.get("doi") or "").replace("https://doi.org/", ""),
            )
            work_people[w["id"]] |= owners
    print(f"{len(meta)} distinct works with a tracked author")

    # ---- domain distribution per work (same recipe as build_domain_specter) ----
    tax_meta = json.load(open(TAXONOMY, encoding="utf-8"))["meta"]["specter"]
    tau = float(tax_meta["tau"])
    col_means = np.array([tax_meta["sim_centering"]["anchor_mean_sims"][d] for d in ALL_DOMAINS])

    ids = json.load(open(IDS_CACHE, encoding="utf-8"))
    emb = np.load(EMB_CACHE, mmap_mode="r")
    idx = {w: i for i, w in enumerate(ids)}
    wids = sorted(w for w in meta if w in idx)
    missing = len(meta) - len(wids)
    if missing:
        print(f"WARNING: {missing} works have no cached embedding and are skipped")
    work_emb = np.asarray(emb[[idx[w] for w in wids]], dtype=np.float32)

    model_name, _anchor_tau, anchor_texts = load_anchors_config()
    embed, _tok, _dev = make_embedder(model_name)
    sentences, owners_ = [], []
    for d in ALL_DOMAINS:
        for t in anchor_texts[d]:
            sentences.append(t)
            owners_.append(d)
    a_emb = embed(sentences, desc="anchors")
    centroids = np.vstack([a_emb[[i for i, d in enumerate(owners_) if d == dom]].mean(axis=0) for dom in ALL_DOMAINS])

    wn = work_emb / np.linalg.norm(work_emb, axis=1, keepdims=True)
    cn = centroids / np.linalg.norm(centroids, axis=1, keepdims=True)
    probs = softmax_rows((wn @ cn.T - col_means[None, :]) / tau)

    # Sanity: hard labels must reproduce the published per-work labels.
    if WORK_LABELS.exists():
        pub = json.load(open(WORK_LABELS, encoding="utf-8"))
        real = [ALL_DOMAINS.index(d) for d in ("computing", "communication", "sensing")]
        names = ("computing", "communication", "sensing")
        same = tot = 0
        for i, w in enumerate(wids):
            if w in pub:
                tot += 1
                same += names[int(probs[i, real].argmax())] == pub[w]
        print(f"hard-label agreement with work_domain_labels.json: {same}/{tot} = {same / max(tot, 1):.4f}")

    # ---- triangle shares (sensing, communication, computing), per-mille ----
    s_i, cm_i, cp_i = (ALL_DOMAINS.index(d) for d in ("sensing", "communication", "computing"))
    tri = probs[:, [s_i, cm_i, cp_i]]
    tri = tri / tri.sum(axis=1, keepdims=True)
    permille = np.rint(tri * 1000).astype(int)

    shards = defaultdict(lambda: defaultdict(list))
    for i, w in enumerate(wids):
        year, cites, n_auth, title, doi = meta[w]
        row = [num(w), year, cites, n_auth, int(permille[i, 0]), int(permille[i, 1]), int(permille[i, 2]), title, doi]
        for pid in work_people[w]:
            shards[shard_of(pid)][pid].append(row)

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    total_rows = 0
    for s in range(N_SHARDS):
        data = {pid: sorted(rows, key=lambda r: (r[1], r[0])) for pid, rows in shards.get(s, {}).items()}
        total_rows += sum(len(r) for r in data.values())
        with open(out / f"{s:02d}.json", "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, separators=(",", ":"))
    size = sum(p.stat().st_size for p in out.glob("*.json"))
    print(f"wrote {N_SHARDS} shards, {total_rows} person-paper rows, {size / 1e6:.1f} MB -> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
