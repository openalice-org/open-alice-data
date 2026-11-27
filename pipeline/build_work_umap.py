#!/usr/bin/env python3
"""Fit one global 2-D UMAP over every accepted corpus work.

Input is the shared bge-large-en-v1.5 work-embedding cache (caches/work_embeddings_bge.npy),
the same vectors build_domain_specter.py uses for the Sensing/Communication/Computing
triangle, so nothing is embedded here. The map is fitted once over ALL accepted works, not
per person, so a paper has the same position whoever's profile it appears in and positions
are comparable across people.

Recipe: L2-normalise, PCA to 50 dimensions (cuts UMAP's nearest-neighbour cost and memory;
the usual preprocessing for high-dimensional text embeddings), then UMAP with cosine
distance. The seed is fixed, so the fit is reproducible on the same inputs and library
versions (a seeded UMAP runs single-threaded, hence a few minutes rather than seconds).

Output: caches/work_umap.npz {ids: OpenAlex work ids, xy: float32 (n, 2)} and
caches/work_umap_manifest.json (parameters, versions, counts).

Run under a memory cap (peak is a few GB):
  systemd-run --user --scope --quiet -p MemoryMax=8G -p MemorySwapMax=0 \
      ../.venv/bin/python -u build_work_umap.py
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from build_domain_specter import EMB_CACHE, IDS_CACHE  # noqa: E402
from build_taxonomy_data import ACCEPTED_LABELS, SPRINT2_WORKS  # noqa: E402

HERE = Path(__file__).resolve().parent
OUT_NPZ = HERE / "caches" / "work_umap.npz"
OUT_MANIFEST = HERE / "caches" / "work_umap_manifest.json"

PCA_DIMS = 50
N_NEIGHBORS = 15
MIN_DIST = 0.0  # tight clumps with clear gaps: the best spatial contrast of the settings compared
SPREAD = 1.0
SEED = 42


def accepted_work_ids() -> list[str]:
    ids = set()
    with open(SPRINT2_WORKS, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                w = json.loads(line)
            except json.JSONDecodeError:
                continue
            if w.get("classification_label") in ACCEPTED_LABELS and w.get("id"):
                ids.add(w["id"])
    return sorted(ids)  # sorted: row order, and so the fit, does not depend on file order


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=0, help="fit only the first N works (dry run)")
    ap.add_argument("--n-neighbors", type=int, default=N_NEIGHBORS)
    ap.add_argument("--min-dist", type=float, default=MIN_DIST)
    ap.add_argument("--spread", type=float, default=SPREAD)
    ap.add_argument("--tag", default="", help="write caches/work_umap_<tag>.npz instead of the default (parameter sweeps)")
    args = ap.parse_args()

    import umap
    from sklearn.decomposition import PCA

    wids = accepted_work_ids()
    cached = {w: i for i, w in enumerate(json.load(open(IDS_CACHE, encoding="utf-8")))}
    wids = [w for w in wids if w in cached]
    if args.limit:
        wids = wids[: args.limit]
    print(f"{len(wids)} accepted works with a cached embedding")

    emb = np.load(EMB_CACHE, mmap_mode="r")
    x = np.asarray(emb[[cached[w] for w in wids]], dtype=np.float32)
    x /= np.linalg.norm(x, axis=1, keepdims=True)

    t0 = time.time()
    pca = PCA(n_components=PCA_DIMS, random_state=SEED)
    x50 = pca.fit_transform(x).astype(np.float32)
    explained = float(pca.explained_variance_ratio_.sum())
    print(f"PCA {x.shape[1]} -> {PCA_DIMS}, explained variance {explained:.3f} ({time.time() - t0:.0f}s)")
    del x

    t0 = time.time()
    xy = umap.UMAP(
        n_components=2,
        n_neighbors=args.n_neighbors,
        min_dist=args.min_dist,
        spread=args.spread,
        metric="cosine",
        random_state=SEED,
    ).fit_transform(x50)
    print(f"UMAP done in {time.time() - t0:.0f}s")

    if args.limit:
        print("dry run: nothing written")
        return 0
    suffix = f"_{args.tag}" if args.tag else ""
    out_npz = OUT_NPZ.with_name(f"work_umap{suffix}.npz")
    out_manifest = OUT_MANIFEST.with_name(f"work_umap{suffix}_manifest.json")
    out_npz.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out_npz, ids=np.array(wids), xy=xy.astype(np.float32))
    json.dump(
        {
            "works": len(wids),
            "pca_dims": PCA_DIMS,
            "pca_explained_variance": round(explained, 4),
            "n_neighbors": args.n_neighbors,
            "min_dist": args.min_dist,
            "spread": args.spread,
            "metric": "cosine",
            "seed": SEED,
            "umap_learn": umap.__version__,
            "embedding_cache": os.path.basename(EMB_CACHE),
        },
        open(out_manifest, "w"),
        indent=2,
    )
    print(f"wrote {out_npz}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
