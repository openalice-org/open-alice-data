#!/usr/bin/env python3
"""SweMetrics enrichment, stage S5: turn gate decisions into corpus works.

Reads the classifier scores from `rescue_train.py --prefix swe`, the Eikos labels from the persistent label store and
the admission policy in corpus/config/swe_admission.json, with one policy per source family (OpenAlex text net, Swepub):

    P(core) >= classifier_min_p            admitted by the classifier
    reject <= P(core) < classifier_min_p   admitted only if Eikos-4B labels it CORE with P(CORE) >= eikos_min_p
    P(core) < reject                       not admitted

When the policy is stricter than the thresholds the classifier was trained with, works between the two need an Eikos
opinion they never had: `--prepare-band` writes them to swe_band_extra.jsonl for `make swe-band`, then apply is rerun.

Incremental by construction:
  * decisions are cheap and recomputed every run (corpus/data/swe_decisions.json);
  * full OpenAlex-shaped records (swepool_records.py) are built ONLY for works admitted since the last run and
    appended, chunk by chunk, to corpus/data/swe_accepted_works.jsonl, so an interrupted run resumes and a
    weekly run fetches a handful of works, not the whole set;
  * works already in the store are never fetched again. The store is only a CACHE of fetched records: the corpus
    extension (corpus/data/swe_corpus_extension.jsonl) is rebuilt from the store on every run, keeping the works the
    CURRENT decisions admit, so tightening or loosening the policy needs no refetch (the report counts the cached
    records the current policy no longer admits).

Swepub-only works (no OpenAlex id) are admitted by the same rule but cannot join the corpus until their authors
are linked to people: they go to corpus/data/swe_swepub_only_accepted.jsonl with their curated Swepub authors.

    python swepool_apply.py --dry-run          # counts only
    python swepool_apply.py --prepare-band     # write the works that still need an Eikos opinion, then stop
    python swepool_apply.py
"""

import argparse
import collections
import json
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from label_store import LabelStore, labeller_key, text_hash  # noqa: E402

ROOT = HERE.parent
DATA = ROOT / "corpus" / "data"
EIKOS_MIN_P = 0.50
EIKOS_MODEL = "eikos-4b-int4-ov"


def read_jsonl(path):
    if not Path(path).exists():
        return
    with open(path, encoding="utf-8") as f:
        for line in f:
            if line.strip():
                yield json.loads(line)


def load_policy(data: Path, accept: float) -> dict:
    cfg = json.loads((ROOT / "corpus" / "config" / "swe_admission.json").read_text())
    out = {}
    for fam in ("openalex", "swepub"):
        c = cfg[fam]
        out[fam] = {"classifier_min_p": c["classifier_min_p"] if c["classifier_min_p"] is not None else accept,
                    "eikos_min_p": c["eikos_min_p"]}
    return out


def family(row: dict) -> str:
    return "swepub" if row["sources"][0].startswith("swepub") else "openalex"


def decide(data: Path):
    """({work id: decision}, pool, band counts, policy, reject, work ids still needing an Eikos label)."""
    from rescue_rubric import QUESTION

    train = json.loads((data / "swe_train_manifest.json").read_text())
    accept, reject = train["thresholds"]["accept"], train["thresholds"]["reject"]
    policy = load_policy(data, accept)
    with np.load(data / "swe_scores.npz") as z:
        ids, p = z["ids"].tolist(), z["p_core"].tolist()
    pool = {r["id"]: r for r in read_jsonl(data / "swe_pool.jsonl")}
    store = LabelStore(data / "label_store.jsonl")
    key = labeller_key(EIKOS_MODEL, QUESTION)
    decisions, band, need = {}, collections.Counter(), []
    for wid, pc in zip(ids, p):
        row = pool[wid]
        pol = policy[family(row)]
        if pc >= pol["classifier_min_p"]:
            decisions[wid] = {"admitted": True, "stage": "classifier", "p_core": round(pc, 4)}
        elif pc >= reject:
            hit = store.get(wid, text_hash(row["display_name"], row["abstract"][:1400]), key)
            if hit is None:
                band["band: not adjudicated yet"] += 1
                need.append(wid)
                decisions[wid] = {"admitted": False, "stage": "band-pending", "p_core": round(pc, 4)}
                continue
            p_eik = hit["probs"][0]
            ok = hit["label"] == "CORE" and p_eik >= pol["eikos_min_p"]
            band["band: Eikos CORE above the floor (admitted)" if ok else
                 "band: Eikos CORE below the floor (not admitted)" if hit["label"] == "CORE" else
                 "band: Eikos not CORE (not admitted)"] += 1
            decisions[wid] = {"admitted": ok, "stage": "eikos", "p_core": round(pc, 4),
                              "eikos_label": hit["label"], "eikos_p_core": round(p_eik, 4)}
        else:
            decisions[wid] = {"admitted": False, "stage": "rejected", "p_core": round(pc, 4)}
    return decisions, pool, band, policy, reject, need


def reason_for(d: dict) -> str:
    if d["stage"] == "classifier":
        return f"SweMetrics pass: text classifier P(core)={d['p_core']:.2f} (bge-large embeddings + logistic regression)"
    return (f"SweMetrics pass: classifier P(core)={d['p_core']:.2f} in the uncertain band, "
            f"adjudicated CORE by Eikos-4B (P={d['eikos_p_core']:.2f})")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", type=Path, default=DATA)
    ap.add_argument("--dry-run", action="store_true", help="print the counts only, fetch and write nothing")
    ap.add_argument("--prepare-band", action="store_true",
                    help="write the works that still need an Eikos opinion to swe_band_extra.jsonl and stop")
    args = ap.parse_args()
    data = args.data
    t0 = time.time()

    decisions, pool, band, policy, reject, need = decide(data)
    if args.prepare_band:
        with open(data / "swe_band_extra.jsonl", "w", encoding="utf-8") as f:
            for i, wid in enumerate(need):
                r = pool[wid]
                f.write(json.dumps({"item": f"rx_{i + 1:05d}", "id": wid, "year": r.get("publication_year"),
                                    "title": r["display_name"], "abstract": r["abstract"][:1400]}, ensure_ascii=False) + "\n")
        print(f"{len(need)} works need an Eikos opinion -> swe_band_extra.jsonl")
        return
    admitted = {w for w, d in decisions.items() if d["admitted"]}
    have = {r["id"] for r in read_jsonl(data / "swe_accepted_works.jsonl")}
    oa_new = sorted(w for w in admitted if w.startswith("https://openalex.org/") and w not in have)
    only = sorted(w for w in admitted if w.startswith("swepub:"))
    drift = sorted(have - admitted)
    by_src = collections.Counter(pool[w]["sources"][0] for w in admitted)
    print(f"policy {json.dumps(policy)} reject<{reject} | scored {len(decisions)} | admitted {len(admitted)} "
          f"(by source: {dict(by_src)})")
    for k, v in sorted(band.items()):
        print(f"  {v:7d}  {k}")
    print(f"records in the store: {len(have)} | to fetch now: {len(oa_new)} | swepub-only admitted: {len(only)} | "
          f"cached records the current policy no longer admits: {len(drift)}")
    if need:
        print(f"NOTE: {len(need)} band works have no Eikos opinion yet (run --prepare-band, make swe-band)")
    if args.dry_run:
        return

    (data / "swe_decisions.json").write_text(json.dumps(decisions, indent=1) + "\n", encoding="utf-8")

    # --- OpenAlex works: build the records, checkpointing every chunk
    ids = [int(w.rsplit("/W", 1)[1]) for w in oa_new]
    fetched = missing = 0
    if ids:
        from swepool_records import iter_records
    with open(data / "swe_accepted_works.jsonl", "a", encoding="utf-8") as out:
        for part in (iter_records(ids, progress=True) if ids else []):
            for wid, rec in sorted(part.items()):
                row = pool[rec["id"]]
                d = decisions[rec["id"]]
                reason = reason_for(d)
                if row.get("abstract_source", "").startswith("swepub") and not rec.get("abstract_inverted_index"):
                    from swepool_records import inverted_index
                    rec["abstract_inverted_index"] = inverted_index(row["abstract"])
                rec.update({
                    "quantum_included": True, "quantum_inclusion_reasons": [reason], "swe_pass": d,
                    "retrieval_sources": ["swemetrics:" + s for s in row["sources"]], "text_hits": row["text_hits"],
                    "keywords_swepub": row.get("keywords") or [], "abstract_source": row.get("abstract_source"),
                    "swepub_record_id": row.get("swepub_record_id"),
                })
                out.write(json.dumps(rec, ensure_ascii=False, sort_keys=True) + "\n")
                fetched += 1
            out.flush()
    missing = len(ids) - fetched
    if missing:
        print(f"WARNING: {missing} admitted works were not found in the oa mirror (left for the next run)")

    # --- Swepub-only works: kept with their curated authors until the people linking exists
    only_have = {r["id"] for r in read_jsonl(data / "swe_swepub_only_accepted.jsonl")}
    todo = [w for w in only if w not in only_have]
    if todo:
        authors = collections.defaultdict(list)
        want = {pool[w]["swepub_internal_work_id"] for w in todo}
        for a in read_jsonl(data / "swe_swepub_authors.jsonl"):
            if a["internal_work_id"] in want:
                authors[a["internal_work_id"]].append(a)
        with open(data / "swe_swepub_only_accepted.jsonl", "a", encoding="utf-8") as out:
            for w in todo:
                row = pool[w]
                out.write(json.dumps({**row, "swe_pass": decisions[w], "authors": sorted(authors[row["swepub_internal_work_id"]], key=lambda a: a["seq"] or 0)},
                                     ensure_ascii=False) + "\n")
    # --- corpus extension: the store filtered by the CURRENT decisions, flags refreshed
    n_ext = 0
    with open(data / "swe_corpus_extension.jsonl", "w", encoding="utf-8") as out:
        for rec in read_jsonl(data / "swe_accepted_works.jsonl"):
            d = decisions.get(rec["id"])
            if not d or not d["admitted"]:
                continue
            rec["swe_pass"] = d
            rec["quantum_inclusion_reasons"] = [reason_for(d)]
            out.write(json.dumps(rec, ensure_ascii=False, sort_keys=True) + "\n")
            n_ext += 1
    manifest = {
        "stage": "S5 apply", "scored": len(decisions), "admitted": len(admitted), "admitted_by_source": dict(by_src),
        "band": dict(band), "accepted_store_before": len(have), "records_fetched": fetched, "swepub_only_added": len(todo),
        "stored_but_no_longer_admitted": len(drift), "extension_works": n_ext, "policy": policy, "reject": reject, "band_pending": len(need),
        "seconds": round(time.time() - t0),
    }
    (data / "swe_apply_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
