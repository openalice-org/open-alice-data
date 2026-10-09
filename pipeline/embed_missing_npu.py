#!/usr/bin/env python3
"""Embed corpus works missing from caches/work_embeddings_bge.npy on the Intel NPU.

build_domain_specter.py needs one bge-large-en-v1.5 CLS embedding per work. The cache
already holds most of them; this fills in the rest through OpenVINO. Anything it does not
embed is still picked up by build_domain_specter.py's own (slower) embedder, so every
stage here is optional and safe to skip.

Texts are built exactly like build_domain_specter.get_work_embeddings:
    f"{title}{tokenizer.sep_token}{abstract}", truncated to MAX_LENGTH, CLS token pooled.

Memory-conscious and resumable by design (NPU compiles are RAM-hungry):
  * each run compiles only the length buckets you pass (--buckets 128:32,256:16) and
    handles only the texts that fit them, so one process = one small set of compiles;
  * results go to caches/bge_partial/part_<tag>.npz as soon as a run finishes, so a
    killed run never loses finished work and later runs skip what is already done;
  * --merge folds all parts into the main cache (atomic replace).
Run it under a cgroup cap (Makefile target `embed-npu` does this).

By default it extends the main pipeline cache. For the rescue pass (see docs/METHODOLOGY.md)
the same machinery embeds a different set of works into a SEPARATE cache, so the main cache
is not bloated with candidates that never enter the corpus:
    --works corpus/data/rescue_pool.jsonl --target-emb corpus/data/rescue_pool_bge.npy \
    --target-ids corpus/data/rescue_pool_bge_ids.json --parts-dir caches/bge_partial_rescue
Validation always compares against the MAIN cache and the corpus file, whatever the target.

Before a run embeds anything, a sample of works that ARE already cached is re-embedded and
compared with the cached vectors (cosine); the run aborts on a mismatch.

    python embed_missing_npu.py --buckets 128:32,256:16 --limit 256   # dry run
    python embed_missing_npu.py --buckets 128:32,256:16               # stage 1
    python embed_missing_npu.py --buckets 512:8 --tag long            # stage 2 (heavy compile)
    python embed_missing_npu.py --merge
"""

import argparse
import ctypes
import gc
import json
import os
import random
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from build_domain_specter import (  # noqa: E402
    EMB_CACHE,
    IDS_CACHE,
    MAX_LENGTH,
    load_anchors_config,
    reconstruct_abstract,
)

DEFAULT_WORKS = HERE.parent / "corpus" / "data" / "sweden_quantum_works.jsonl"
PARTS_DIR = HERE / "caches" / "bge_partial"


def rss_mb():
    with open("/proc/self/status") as f:
        for line in f:
            if line.startswith("VmRSS"):
                return int(line.split()[1]) // 1024
    return -1


def hwm_mb():
    """Peak RSS so far (VmHWM): shows the spike inside a phase that rss_mb() after it would miss."""
    with open("/proc/self/status") as f:
        for line in f:
            if line.startswith("VmHWM"):
                return int(line.split()[1]) // 1024
    return -1


def release_memory():
    gc.collect()
    try:
        ctypes.CDLL("libc.so.6").malloc_trim(0)
    except OSError:
        pass


def parse_buckets(spec):
    return sorted((int(a), int(b)) for a, b in (p.split(":") for p in spec.split(",")))


def load_texts(path, sep, wanted=None):
    texts = {}
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            work = json.loads(line)
            wid = work.get("id")
            if not wid or (wanted is not None and wid not in wanted):
                continue
            title = work.get("display_name") or ""
            abstract = work.get("abstract") or reconstruct_abstract(work.get("abstract_inverted_index"))
            texts[wid] = f"{title}{sep}{abstract}"
    return texts


def done_ids(parts_dir=PARTS_DIR):
    done = set()
    if parts_dir.exists():
        for p in parts_dir.glob("part_*.npz"):
            with np.load(p, allow_pickle=False) as z:
                done.update(z["ids"].tolist())
    return done


def compile_buckets(model_name, device, buckets, cache_dir):
    """Compile one bucket at a time; free the torch model and graphs as early as possible."""
    import openvino as ov
    import torch
    from transformers import AutoModel

    class ClsOnly(torch.nn.Module):
        def __init__(self, inner):
            super().__init__()
            self.inner = inner

        def forward(self, input_ids, attention_mask, token_type_ids):
            out = self.inner(
                input_ids=input_ids,
                attention_mask=attention_mask,
                token_type_ids=token_type_ids,
            )
            return out.last_hidden_state[:, 0, :]  # BGE pooling: CLS token

    model = ClsOnly(AutoModel.from_pretrained(model_name).eval())
    for param in model.parameters():
        param.requires_grad_(False)  # no autograd bookkeeping while tracing (big memory saver)
    print(f"[load] torch model  rss={rss_mb()}MB  peak={hwm_mb()}MB", flush=True)
    core = ov.Core()
    core.set_property({"CACHE_DIR": str(cache_dir)})  # reuse compiled blobs across runs
    compiled = {}
    for length, batch in buckets:
        example = {
            "input_ids": torch.zeros(batch, length, dtype=torch.long),
            "attention_mask": torch.ones(batch, length, dtype=torch.long),
            "token_type_ids": torch.zeros(batch, length, dtype=torch.long),
        }
        t0 = time.time()
        ov_model = ov.convert_model(model, example_input=example)
        ov_model.reshape({inp.get_any_name(): [batch, length] for inp in ov_model.inputs})
        print(f"[convert] len={length}: {time.time() - t0:.0f}s  rss={rss_mb()}MB  peak={hwm_mb()}MB", flush=True)
        compiled[length] = (core.compile_model(ov_model, device), batch)
        del ov_model, example
        release_memory()
        print(f"[compile] {device} len={length} batch={batch}: {time.time() - t0:.0f}s  rss={rss_mb()}MB  peak={hwm_mb()}MB", flush=True)
    del model
    release_memory()
    print(f"[compile] torch model freed  rss={rss_mb()}MB", flush=True)
    return compiled


def embed(texts, enc, compiled, pad_id, label="embed"):
    lengths = sorted(compiled)
    groups = {length: [] for length in lengths}
    for i, ids in enumerate(enc):
        groups[next(length for length in lengths if len(ids) <= length)].append(i)
    out = np.zeros((len(texts), 1024), dtype=np.float32)
    done, t0 = 0, time.time()
    for length in lengths:
        cm, batch = compiled[length]
        idx = groups[length]
        for n, s in enumerate(range(0, len(idx), batch)):
            chunk = idx[s : s + batch]
            ids = np.full((batch, length), pad_id, dtype=np.int64)
            mask = np.zeros((batch, length), dtype=np.int64)
            mask[:, 0] = 1  # keep dummy rows from being fully masked
            for r, i in enumerate(chunk):
                ids[r, : len(enc[i])] = enc[i]
                mask[r, : len(enc[i])] = 1
            result = cm({"input_ids": ids, "attention_mask": mask, "token_type_ids": np.zeros_like(ids)})[0]
            out[chunk] = np.asarray(result)[: len(chunk)]
            done += len(chunk)
            if n % 100 == 0:
                print(f"[{label}] {done}/{len(texts)}  {done / (time.time() - t0):.1f} texts/s  len={length}  rss={rss_mb()}MB", flush=True)
    return out


def merge(emb_path=EMB_CACHE, ids_path=IDS_CACHE, parts_dir=PARTS_DIR):
    if Path(ids_path).exists():
        cached_ids = json.load(open(ids_path, encoding="utf-8"))
        cached_emb = np.load(emb_path)
    else:  # a fresh separate target (rescue pool): start empty
        cached_ids, cached_emb = [], np.zeros((0, 1024), dtype=np.float32)
    assert len(cached_ids) == cached_emb.shape[0], "cache id/embedding length mismatch"
    have = set(cached_ids)
    new_ids, new_parts = [], []
    for p in sorted(Path(parts_dir).glob("part_*.npz")):
        with np.load(p, allow_pickle=False) as z:
            ids, emb = z["ids"].tolist(), z["emb"]
        keep = [i for i, w in enumerate(ids) if w not in have]
        have.update(ids[i] for i in keep)
        new_ids += [ids[i] for i in keep]
        new_parts.append(emb[keep])
        print(f"[merge] {p.name}: {len(keep)}/{len(ids)} new")
    if not new_ids:
        print("[merge] nothing to add")
        return
    all_emb = np.vstack([cached_emb] + new_parts).astype(cached_emb.dtype)
    tmp_emb, tmp_ids = str(emb_path) + ".tmp", str(ids_path) + ".tmp"
    with open(tmp_emb, "wb") as f:
        np.save(f, all_emb)
    with open(tmp_ids, "w", encoding="utf-8") as f:
        json.dump(cached_ids + new_ids, f)
    os.replace(tmp_emb, emb_path)
    os.replace(tmp_ids, ids_path)
    print(f"[merge] cache now {len(cached_ids) + len(new_ids)} embeddings (+{len(new_ids)})")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--works", type=Path, default=DEFAULT_WORKS)
    ap.add_argument("--device", default="NPU")
    ap.add_argument("--buckets", default="128:32,256:16", help="length:batch,... compiled by THIS run")
    ap.add_argument("--tag", default=None, help="name for the partial file (default: the bucket lengths)")
    ap.add_argument("--limit", type=int, default=None, help="embed only N works and do not save (dry run)")
    ap.add_argument("--validate", type=int, default=48, help="cached works re-embedded for a cosine check")
    ap.add_argument("--min-cosine", type=float, default=0.98)
    ap.add_argument("--merge", action="store_true", help="fold all partial files into the target cache and exit")
    ap.add_argument("--target-emb", type=Path, default=EMB_CACHE, help="where embeddings accumulate (default: main cache)")
    ap.add_argument("--target-ids", type=Path, default=IDS_CACHE)
    ap.add_argument("--parts-dir", type=Path, default=PARTS_DIR, help="partial results of this target")
    ap.add_argument("--validate-works", type=Path, default=DEFAULT_WORKS,
                    help="works file whose cached vectors validate the NPU output (default: the corpus)")
    args = ap.parse_args()

    if args.merge:
        merge(args.target_emb, args.target_ids, args.parts_dir)
        return

    from transformers import AutoTokenizer

    buckets = parse_buckets(args.buckets)
    max_len = max(length for length, _ in buckets)
    tag = args.tag or "_".join(str(length) for length, _ in buckets)

    model_name, _tau, _anchors = load_anchors_config()
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    main_ids = json.load(open(IDS_CACHE, encoding="utf-8"))
    main_idx = {w: i for i, w in enumerate(main_ids)}  # the MAIN cache: reference for validation
    del main_ids
    target_ids = json.load(open(args.target_ids, encoding="utf-8")) if args.target_ids.exists() else []
    in_target = set(target_ids)  # works already embedded into the TARGET cache
    del target_ids

    all_texts = load_texts(args.works, tokenizer.sep_token)
    already = done_ids(args.parts_dir)
    todo = sorted(w for w in all_texts if w not in in_target and w not in already)
    print(f"[cache] {len(all_texts) - len([w for w in all_texts if w not in in_target])}/{len(all_texts)} in target; "
          f"{len(already)} in partial files; {len(todo)} still to embed  rss={rss_mb()}MB", flush=True)

    lens = {}
    for s in range(0, len(todo), 2000):
        part = todo[s : s + 2000]
        for w, ids in zip(part, tokenizer([all_texts[w] for w in part], truncation=True, max_length=MAX_LENGTH)["input_ids"]):
            lens[w] = ids
    todo = [w for w in todo if len(lens[w]) <= max_len]
    print(f"[plan] {len(todo)} works fit buckets {buckets}", flush=True)
    if args.limit:
        todo = todo[: args.limit]
    if not todo:
        print("nothing to do for these buckets")
        return

    ref_texts = all_texts if args.validate_works == args.works else load_texts(args.validate_works, tokenizer.sep_token)
    sample_pool = [w for w in ref_texts if w in main_idx]
    sample = random.Random(0).sample(sample_pool, min(len(sample_pool), args.validate * 4))
    sample_enc = tokenizer([ref_texts[w] for w in sample], truncation=True, max_length=MAX_LENGTH)["input_ids"]
    sample = [w for w, ids in zip(sample, sample_enc) if len(ids) <= max_len][: args.validate]
    sample_enc = [tokenizer(ref_texts[w], truncation=True, max_length=MAX_LENGTH)["input_ids"] for w in sample]
    todo_enc = [lens[w] for w in todo]
    todo_texts = [all_texts[w] for w in todo]
    ref = np.load(EMB_CACHE, mmap_mode="r")[[main_idx[w] for w in sample]].astype(np.float32)
    pad_id = tokenizer.pad_token_id
    del all_texts, ref_texts, lens, main_idx, in_target
    release_memory()

    compiled = compile_buckets(model_name, args.device, buckets, HERE / "caches" / f"ov_{args.device.lower()}_blobs")

    got = embed(sample, sample_enc, compiled, pad_id, label="validate")
    cos = (got * ref).sum(1) / (np.linalg.norm(got, axis=1) * np.linalg.norm(ref, axis=1))
    # A cached vector can legitimately differ if OpenAlex changed that work's title/abstract since
    # the cache was built (verified: for such works NPU == CPU fp32 at 1.00000 while cached != both).
    # So judge the sample robustly, not by its single worst item.
    bad = int((cos < args.min_cosine).sum())
    print(f"[validate] cosine vs cached: min={cos.min():.5f} median={np.median(cos):.5f} mean={cos.mean():.5f} "
          f"below_{args.min_cosine}={bad}/{len(sample)} (n={len(sample)})", flush=True)
    if np.median(cos) < 0.999 or bad > max(1, len(sample) // 10):
        sys.exit(f"ABORT: median {np.median(cos):.4f}, {bad} samples < {args.min_cosine}; nothing saved")

    t0 = time.time()
    new_emb = embed(todo_texts, todo_enc, compiled, pad_id)
    print(f"[done] {len(todo)} works in {time.time() - t0:.0f}s", flush=True)

    if args.limit:
        print("dry run (--limit): nothing saved")
        return
    args.parts_dir.mkdir(parents=True, exist_ok=True)
    out = args.parts_dir / f"part_{tag}.npz"
    with open(str(out) + ".tmp", "wb") as f:
        np.savez(f, ids=np.array(todo), emb=new_emb)
    os.replace(str(out) + ".tmp", out)
    print(f"[saved] {out}  ({len(todo)} works). Run --merge when all stages are done.")


if __name__ == "__main__":
    main()
