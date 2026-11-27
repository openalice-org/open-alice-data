#!/usr/bin/env python3
"""Bring the SweMetrics-enriched corpus up to date, doing ONLY the work that is not already done.

    python swe_update.py --plan     # what is pending at every stage, with time estimates; changes nothing
    python swe_update.py            # run the pending stages, in order
    python swe_update.py --only embed,label   # just these stages

Every stage decides from stored results what is left to do:

    snapshots  the SweMetrics snapshot dates are compared with the last processed ones; nothing changed = done
    harvest    Swepub (about a minute, whole) and the OpenAlex text net: FULL the first time, then only the work
               ids created since the last harvest (swepool_build.py textnet)
    assemble   join the raw files into the pool (seconds, local); skipped when the inputs are unchanged
    embed      only pool works missing from the embedding cache (embed_missing_npu.py skips cached ids)
    sample     the labelled sample is drawn ONCE and frozen (rescue_sample.py)
    label      only sample items missing from the label store (label_store.py): a paper is labelled once, ever
    train      logistic regression on the embeddings + labels, scores EVERY pool work (seconds to a minute)
    band       only uncertain-band works missing from the label store
    apply      decisions for every work; records are fetched only for newly admitted works (swepool_apply.py)

The embed / label / band stages run through `make swe-embed` / `swe-label` / `swe-band` and need the OpenVINO
environment (NPU_PY, NPU_LIB, EIKOS_DIR as for `make rescue`; they are memory-capped by MEM_CAP).
Run state (snapshots, harvest watermark, stage fingerprints, run history) lives in corpus/data/swe_state.json.
"""

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import swe_state  # noqa: E402
from label_store import LabelStore, labeller_key, text_hash  # noqa: E402

ROOT = HERE.parent
DATA = ROOT / "corpus" / "data"
PY = sys.executable
STAGES = ["snapshots", "harvest", "assemble", "embed", "sample", "label", "train", "band", "apply"]
# rough measured rates on this machine, for the plan only
SEC_PER_EMBED, SEC_PER_LABEL, SEC_PER_RECORD = 0.05, 0.6, 0.4


def jl(path):
    if Path(path).exists():
        with open(path, encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    yield json.loads(line)


def fmt(sec: float) -> str:
    return f"{sec / 3600:.1f} h" if sec >= 3600 else f"{sec / 60:.0f} min" if sec >= 90 else f"{sec:.0f} s"


def labeller():
    from rescue_rubric import QUESTION

    return labeller_key("eikos-4b-int4-ov", QUESTION)


def missing_labels(items_path: Path) -> tuple[int, int]:
    store, key = LabelStore(DATA / "label_store.jsonl"), labeller()
    items = list(jl(items_path))
    return sum(1 for it in items if not store.get(it["id"], text_hash(it["title"], it["abstract"]), key)), len(items)


def cached_ids(path: Path) -> set:
    return set(json.loads(path.read_text())) if path.exists() else set()


def plan(state: dict, snaps: dict) -> dict:
    """{stage: {"todo": int|str|None, "seconds": float, "note": str}} (todo None/0 = nothing to do)."""
    p = {}
    changed = {k: state["snapshots"].get(k) != v for k, v in snaps.items()}
    p["snapshots"] = {"todo": sum(changed.values()), "seconds": 2,
                      "note": ", ".join(f"{k} {v}{' (changed)' if changed[k] else ''}" for k, v in snaps.items())}
    tn = state.get("textnet", {})
    if not tn.get("watermark"):
        harvest = ("FULL harvest (first run)", 3600 + 60)
    elif tn.get("snapshot") != snaps["oa"] or changed["swepub"]:
        harvest = ("incremental text net" + (" + swepub" if changed["swepub"] else ""), 300 + 60 * changed["swepub"])
    else:
        harvest = (0, 0)
    p["harvest"] = {"todo": harvest[0], "seconds": harvest[1], "note": f"text net last full {tn.get('last_full', 'never')}"}
    raw = [DATA / "swe_textnet_raw.jsonl", DATA / "swe_swepub_raw.jsonl"]
    fp = swe_state.fingerprint(*[r for r in raw if r.exists()])
    pool = list(jl(DATA / "swe_pool.jsonl"))
    need = not pool or state["stamps"].get("assemble") != fp
    p["assemble"] = {"todo": int(need), "seconds": 15, "note": f"pool now {len(pool)} works"}
    emb = cached_ids(DATA / "swe_pool_bge_ids.json")
    to_embed = sum(1 for r in pool if r["id"] not in emb)
    p["embed"] = {"todo": to_embed, "seconds": to_embed * SEC_PER_EMBED + (60 if to_embed else 0), "note": f"{len(emb)} cached"}
    sample = DATA / "swe_label_sample.jsonl"
    if sample.exists():
        todo, n = missing_labels(sample)
        p["sample"] = {"todo": 0, "seconds": 0, "note": f"frozen sample of {n}"}
        p["label"] = {"todo": todo, "seconds": todo * SEC_PER_LABEL, "note": f"{n - todo} of {n} already in the label store"}
    else:
        p["sample"] = {"todo": 1, "seconds": 5, "note": "not drawn yet (2000 uniform + 2000 targeted, then frozen)"}
        p["label"] = {"todo": "up to 4000", "seconds": 4000 * SEC_PER_LABEL, "note": "first run: the sample is labelled once"}
    labels_fp = swe_state.fingerprint(DATA / "swe_pool_bge.npy", DATA / "label_store.jsonl", DATA / "swe_pool.jsonl")
    p["train"] = {"todo": int(state["stamps"].get("train") != labels_fp or not (DATA / "swe_scores.npz").exists()), "seconds": 120, "note": ""}
    band = DATA / "swe_band.jsonl"
    if band.exists():
        todo, n = missing_labels(band)
        p["band"] = {"todo": todo, "seconds": todo * SEC_PER_LABEL, "note": f"{n} band works, {n - todo} already labelled"}
    else:
        p["band"] = {"todo": "after train", "seconds": 0, "note": "the band is known once the classifier has scored the pool"}
    p["apply"] = {"todo": "see swepool_apply.py --dry-run", "seconds": 30, "note": f"records ~{SEC_PER_RECORD}s per newly admitted work"}
    return p


def run(cmd, label):
    t = time.time()
    print(f"\n== {label}: {' '.join(str(c) for c in cmd)}", flush=True)
    subprocess.run(cmd, check=True, cwd=ROOT)
    print(f"== {label} finished in {fmt(time.time() - t)}", flush=True)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--plan", action="store_true", help="show what is pending and the time it would take; run nothing")
    ap.add_argument("--only", help="comma-separated stages to run (default: every pending stage)")
    ap.add_argument("--full-harvest", action="store_true", help="redo the whole OpenAlex text net (do this about quarterly)")
    ap.add_argument("--force", action="store_true", help="run the harvest even when the snapshots are unchanged")
    args = ap.parse_args()

    from swepool_build import snapshots

    state, snaps = swe_state.load(), snapshots()
    pl = plan(state, snaps)
    total = 0
    print(f"{'stage':<10} {'pending':>16} {'estimate':>9}   note")
    for st in STAGES:
        e = pl[st]
        total += e["seconds"] if e["todo"] else 0
        print(f"{st:<10} {str(e['todo'] or '-'):>16} {fmt(e['seconds']) if e['todo'] else '-':>9}   {e['note']}")
    print(f"estimated total: {fmt(total)}")
    last = state.get("textnet", {}).get("last_full")
    if last and (time.time() - time.mktime(time.strptime(last, "%Y-%m-%d"))) > 90 * 86400:
        print(f"NOTE: the text net was last harvested in full on {last}; run with --full-harvest to pick up old works that changed")
    if args.plan:
        return

    only = set(args.only.split(",")) if args.only else None
    started = time.time()
    ran = {}

    def want(stage):
        return (only is None and bool(pl[stage]["todo"])) or (only is not None and stage in only)

    if want("harvest") or args.force or args.full_harvest:
        if pl["snapshots"]["todo"] or args.force or args.full_harvest or (only and "harvest" in only):
            run([PY, "-u", "pipeline/swepool_build.py", "swepub"], "harvest swepub")
            cmd = [PY, "-u", "pipeline/swepool_build.py", "textnet"] + (["--full"] if args.full_harvest else []) + (["--force"] if args.force or only else [])
            run(cmd, "harvest text net")
            state = swe_state.load()
            state["snapshots"] = dict(snaps)
            swe_state.save(state)
            ran["harvest"] = True
    state = swe_state.load()
    if want("assemble") or ran:
        run([PY, "pipeline/swepool_build.py", "assemble"], "assemble")
        raw = [DATA / "swe_textnet_raw.jsonl", DATA / "swe_swepub_raw.jsonl"]
        state["stamps"]["assemble"] = swe_state.fingerprint(*[r for r in raw if r.exists()])
        swe_state.save(state)
    for stage, target in (("embed", "swe-embed"), ("sample", "swe-sample"), ("label", "swe-label"), ("train", "swe-train"), ("band", "swe-band")):
        if (only is None and (ran or pl[stage]["todo"] or stage in ("sample",))) or (only is not None and stage in only):
            if stage == "label" and (DATA / "swe_label_sample.jsonl").exists() and only is None and not missing_labels(DATA / "swe_label_sample.jsonl")[0]:
                continue
            if stage == "band" and not (DATA / "swe_band.jsonl").exists():
                continue
            run(["make", target], stage)
            if stage == "train":
                state = swe_state.load()
                state["stamps"]["train"] = swe_state.fingerprint(DATA / "swe_pool_bge.npy", DATA / "label_store.jsonl", DATA / "swe_pool.jsonl")
                swe_state.save(state)
                if only is None:  # the band file now exists: label what is missing, then decide
                    todo, _ = missing_labels(DATA / "swe_band.jsonl")
                    if todo:
                        run(["make", "swe-band"], "band")
    if only is None or "apply" in only:
        # the admission policy can be stricter than the trained thresholds: label what it needs, then decide
        run([PY, "-u", "pipeline/swepool_apply.py", "--prepare-band"], "apply (band extra)")
        if (DATA / "swe_band_extra.jsonl").stat().st_size:
            (DATA / "swe_band_extra_eikos.json").unlink(missing_ok=True)
            run(["make", "swe-band"], "band (policy extra)")
        run([PY, "-u", "pipeline/swepool_apply.py"], "apply")
    state = swe_state.load()
    swe_state.record_run(state, started, {"snapshots": snaps})
    swe_state.save(state)
    print(f"\nupdate finished in {fmt(time.time() - started)}")


if __name__ == "__main__":
    main()
