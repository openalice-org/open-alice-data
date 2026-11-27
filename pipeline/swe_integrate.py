#!/usr/bin/env python3
"""Merge the SweMetrics corpus extension into the corpus the downstream steps split.

Input : the base corpus (the rescued corpus when `make rescue` produced it, else the rule corpus) and
        corpus/data/swe_corpus_extension.jsonl (swepool_apply.py: works admitted under swe_admission.json).
Output: corpus/data/sweden_quantum_works_with_swe.jsonl = base + extension, one row per work id (a work
        already in the base keeps its base row), and the main embedding cache extended with the vectors the
        extension works brought from the SweMetrics pool (so the later steps never re-embed them).

    python swe_integrate.py [--dry-run]
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
DATA = HERE.parent / "corpus" / "data"


def read_jsonl(path):
    with open(path, encoding="utf-8") as f:
        for line in f:
            if line.strip():
                yield json.loads(line)


def extend_cache(work_ids):
    from build_domain_specter import EMB_CACHE, IDS_CACHE

    main_ids = json.load(open(IDS_CACHE, encoding="utf-8"))
    have = set(main_ids)
    pool_ids = json.load(open(DATA / "swe_pool_bge_ids.json", encoding="utf-8"))
    row = {w: i for i, w in enumerate(pool_ids)}
    new = [w for w in sorted(work_ids) if w not in have and w in row]
    if not new:
        return 0
    main_emb = np.load(EMB_CACHE)
    assert len(main_ids) == main_emb.shape[0], "main cache id/embedding length mismatch"
    add = np.asarray(np.load(DATA / "swe_pool_bge.npy", mmap_mode="r")[[row[w] for w in new]]).astype(main_emb.dtype)
    tmp_emb, tmp_ids = str(EMB_CACHE) + ".tmp", str(IDS_CACHE) + ".tmp"
    with open(tmp_emb, "wb") as f:
        np.save(f, np.vstack([main_emb, add]))
    with open(tmp_ids, "w", encoding="utf-8") as f:
        json.dump(main_ids + new, f)
    Path(tmp_emb).replace(EMB_CACHE)
    Path(tmp_ids).replace(IDS_CACHE)
    return len(new)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()
    rescued = DATA / "sweden_quantum_works_with_rescue.jsonl"
    base = rescued if rescued.exists() else DATA / "sweden_quantum_works.jsonl"
    out = DATA / "sweden_quantum_works_with_swe.jsonl"

    ext = {}
    for r in read_jsonl(DATA / "swe_corpus_extension.jsonl"):
        ext[r["id"]] = r
    seen, n_base = set(), 0
    tmp = out.with_suffix(".jsonl.tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        for r in read_jsonl(base):
            if r["id"] in seen:
                continue
            seen.add(r["id"])
            n_base += 1
            f.write(json.dumps(r, ensure_ascii=False, sort_keys=True) + "\n")
        added = [w for w in sorted(ext) if w not in seen]
        for w in added:
            f.write(json.dumps(ext[w], ensure_ascii=False, sort_keys=True) + "\n")
    print(f"base {base.name}: {n_base} works | extension {len(ext)} ({len(ext) - len(added)} already in base) | "
          f"final {n_base + len(added)}")
    if args.dry_run:
        tmp.unlink()
        return
    tmp.replace(out)
    print(f"wrote {out}; main embedding cache extended with {extend_cache(added)} works")


if __name__ == "__main__":
    main()
