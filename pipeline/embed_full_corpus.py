#!/usr/bin/env python3
"""Pre-embed the ENTIRE filtered Nordic corpus (all quantum_included works in
KTH.Tools.QuantumAuthors' data/sweden_quantum_works.jsonl, ~95k works) into
the same bge-large cache build_domain_specter.py reads (EMB_CACHE/IDS_CACHE) —
not just the subset matched to the 9 currently-tracked Swedish universities.

Once the Nordic university list is curated later, build_domain_specter.py's
embedding step is then a cache hit for every work instead of a fresh GPU run.

Chunked (not one giant call) so get_work_embeddings() checkpoints the cache
to disk after every chunk — an interruption mid-run only costs that chunk,
not the whole job.

Run standalone (not concurrently with build_domain_specter.py — both write
the same cache files):
  .venv/bin/python3 embed_full_corpus.py
"""

import json
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from build_domain_specter import get_work_embeddings, load_anchors_config, reconstruct_abstract  # noqa: E402

SPRINT13 = pathlib.Path(__file__).resolve().parent
CORPUS_PATH = (SPRINT13.parent / "KTH.Tools.QuantumAuthors" / "data"
               / "sweden_quantum_works.jsonl")
CHUNK_SIZE = 2000


def load_all_works():
    works = {}
    with open(CORPUS_PATH, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            work = json.loads(line)
            if not work.get("quantum_included"):
                continue
            wid = work.get("id")
            if not wid:
                continue
            title = work.get("display_name") or ""
            abstract = reconstruct_abstract(work.get("abstract_inverted_index"))
            works[wid] = (title, abstract)
    return works


def main():
    model_name, _tau, _anchors = load_anchors_config()
    print(f"Model: {model_name}")

    works = load_all_works()
    print(f"Loaded {len(works)} quantum_included works from {CORPUS_PATH}")

    all_ids = sorted(works)
    n_chunks = (len(all_ids) + CHUNK_SIZE - 1) // CHUNK_SIZE
    for i in range(0, len(all_ids), CHUNK_SIZE):
        chunk_ids = all_ids[i:i + CHUNK_SIZE]
        chunk_works = {wid: works[wid] for wid in chunk_ids}
        chunk_no = i // CHUNK_SIZE + 1
        print(f"\n=== chunk {chunk_no}/{n_chunks} ({len(chunk_ids)} works) ===")
        wids, _emb, _embed_fn, device = get_work_embeddings(chunk_works, model_name)
        print(f"chunk {chunk_no}/{n_chunks} done, cache checkpointed "
              f"(device={device}, cache now covers {len(wids)} ids from this chunk's view)")

    print(f"\nAll {len(all_ids)} works embedded/cached.")


if __name__ == "__main__":
    main()
