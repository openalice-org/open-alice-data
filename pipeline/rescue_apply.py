#!/usr/bin/env python3
"""Rescue pass, stage R4: apply the decisions and write the final corpus.

Decision for every pool work (thresholds come from rescue_train_manifest.json):
    P(core) >= accept            admitted by the classifier         stage "classifier"
    reject <= P(core) < accept   admitted only if Eikos-4B labels it CORE with its own
                                 P(CORE) >= EIKOS_MIN_P (0.50)     stage "eikos"
                                 (a band work without an Eikos label stays excluded)

Why Eikos needs a probability floor and not just "label == CORE", and why it is 0.50.
Eikos's probabilities are flat but informative: on the 500-paper audit set Haiku agrees with its
CORE calls 55 % of the time at P in [0.5, 0.6), 67 % in [0.6, 0.7), 82 % in [0.7, 0.8) and 100 %
from 0.8 (docs/validation/scripts/eikos_probability_vs_haiku.py). That audit set is not the band
(the band holds works the classifier is unsure about), so the admission routes were measured on the
band itself with 100 blind Haiku-labelled works each (docs/validation/scripts/rescue_validation_report.py):

    route                         works   CORE   PERIPHERAL   NOT
    classifier alone (P >= 0.71)    100    87 %      9 %       4 %
    Eikos CORE, P >= 0.70           100    91 %      9 %       0 %
    Eikos CORE, 0.5 <= P < 0.70     100    73 %     25 %       2 %

Even the weakest tier is almost never non-quantum (2 % NOT) and its misses are mostly PERIPHERAL.
The rule-based corpus itself holds far noisier works: in the audit, the works only the new rules
admitted were 28 % CORE and 43 % NOT. So the floor is set at 0.50, the lowest tier that was measured;
Eikos CORE calls below 0.50 (438 of 2,221 in the band) have no evidence behind them and stay excluded.
    P(core) < reject             stays excluded

Eikos labels come from two places: the labelled training sample (rescue_label_sample_eikos.json)
and the band adjudication run (rescue_band_eikos.json).

The final corpus is the ORIGINAL corpus, copied through unchanged, followed by the rescued works.
A rescued row is the candidate's full OpenAlex record (the same fields every corpus row has) plus
    quantum_included        True
    quantum_inclusion_reasons   one sentence saying which stage admitted it and with what score
    rescue_pass             {stage, p_core, eikos_label, eikos_p_core}
The rule-evidence fields of corpus step 05 (paper_level_quantum_terms, ...) are not recomputed:
by construction the rules found no literal quantum evidence in these works, which is exactly why
they were excluded. Downstream steps only need quantum_included and the OpenAlex fields.

It also copies the rescued works' bge embeddings from the rescue-pool cache into the MAIN embedding
cache (same model, same vector space), so the domain classifier finds them cached instead of
re-embedding them on the CPU (--no-extend-cache to skip).

Outputs
    corpus/data/sweden_quantum_works_with_rescue.jsonl
    corpus/data/sweden_quantum_works_with_rescue_manifest.json
    pipeline/caches/work_embeddings_bge.npy + work_embedding_ids_bge.json   (extended)

Usage:  python rescue_apply.py [--dry-run]
"""

import argparse
import collections
import hashlib
import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
DATA = HERE.parent / "corpus" / "data"


EIKOS_MIN_P = 0.50  # see the module docstring


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def extend_main_embedding_cache(work_ids):
    """Append the rescued works' vectors (computed for the rescue pool) to the main embedding cache."""
    from build_domain_specter import EMB_CACHE, IDS_CACHE

    main_ids = json.load(open(IDS_CACHE, encoding="utf-8"))
    have = set(main_ids)
    pool_ids = json.load(open(DATA / "rescue_pool_bge_ids.json", encoding="utf-8"))
    row = {w: i for i, w in enumerate(pool_ids)}
    new = [w for w in sorted(work_ids) if w not in have]
    if not new:
        return 0
    main_emb = np.load(EMB_CACHE)
    assert len(main_ids) == main_emb.shape[0], "main cache id/embedding length mismatch"
    add = np.asarray(np.load(DATA / "rescue_pool_bge.npy", mmap_mode="r")[[row[w] for w in new]]).astype(main_emb.dtype)
    tmp_emb, tmp_ids = str(EMB_CACHE) + ".tmp", str(IDS_CACHE) + ".tmp"
    with open(tmp_emb, "wb") as f:
        np.save(f, np.vstack([main_emb, add]))
    with open(tmp_ids, "w", encoding="utf-8") as f:
        json.dump(main_ids + new, f)
    Path(tmp_emb).replace(EMB_CACHE)
    Path(tmp_ids).replace(IDS_CACHE)
    return len(new)


def eikos_by_work(sample_path, labels_path):
    """work id -> (label, P(CORE)) from one labelled file; items map back to work ids via the jsonl."""
    if not labels_path.exists():
        return {}
    item_to_id = {}
    for line in open(sample_path, encoding="utf-8"):
        s = json.loads(line)
        item_to_id[s["item"]] = s["id"]
    return {item_to_id[x["item"]]: (x["label"], x["probs"][0]) for x in json.load(open(labels_path)) if x["item"] in item_to_id}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--corpus", type=Path, default=DATA / "sweden_quantum_works.jsonl")
    ap.add_argument("--union", type=Path, default=DATA / "sweden_quantum_candidate_union.jsonl")
    ap.add_argument("--out", type=Path, default=DATA / "sweden_quantum_works_with_rescue.jsonl")
    ap.add_argument("--eikos-min-p", type=float, default=EIKOS_MIN_P,
                    help="minimum Eikos P(CORE) for a band work to be admitted")
    ap.add_argument("--no-extend-cache", action="store_true", help="do not copy rescued embeddings into the main cache")
    ap.add_argument("--dry-run", action="store_true", help="print the counts only, write nothing")
    args = ap.parse_args()

    train = json.loads((DATA / "rescue_train_manifest.json").read_text())
    accept, reject = train["thresholds"]["accept"], train["thresholds"]["reject"]
    with np.load(DATA / "rescue_scores.npz") as z:
        ids, p = z["ids"].tolist(), z["p_core"].tolist()
    eikos = eikos_by_work(DATA / "rescue_label_sample.jsonl", DATA / "rescue_label_sample_eikos.json")
    eikos.update(eikos_by_work(DATA / "rescue_band.jsonl", DATA / "rescue_band_eikos.json"))

    rescued = {}
    stages = collections.Counter()
    for wid, pc in zip(ids, p):
        if pc >= accept:
            rescued[wid] = {"stage": "classifier", "p_core": round(pc, 4), "eikos_label": None, "eikos_p_core": None}
        elif pc >= reject:
            label, p_eikos = eikos.get(wid, (None, None))
            admitted = label == "CORE" and p_eikos >= args.eikos_min_p
            stages["band: no Eikos label (stays excluded)" if label is None else
                   "band: Eikos CORE above the floor (admitted)" if admitted else
                   "band: Eikos CORE below the floor (excluded)" if label == "CORE" else
                   "band: Eikos not CORE (excluded)"] += 1
            if admitted:
                rescued[wid] = {"stage": "eikos", "p_core": round(pc, 4), "eikos_label": label, "eikos_p_core": round(p_eikos, 4)}
    stages["admitted by classifier"] = sum(1 for r in rescued.values() if r["stage"] == "classifier")
    stages["admitted by Eikos adjudication"] = sum(1 for r in rescued.values() if r["stage"] == "eikos")
    print(f"thresholds: accept>={accept} reject<{reject} | rescued {len(rescued)} of {len(ids)} pool works")
    for k, v in sorted(stages.items()):
        print(f"  {v:7d}  {k}")
    if args.dry_run:
        return

    n_corpus = 0
    corpus_ids = set()
    tmp = args.out.with_suffix(".jsonl.tmp")
    with open(tmp, "w", encoding="utf-8") as out:
        for line in open(args.corpus, encoding="utf-8"):
            out.write(line)
            corpus_ids.add(json.loads(line)["id"])
            n_corpus += 1
        assert not (corpus_ids & rescued.keys()), "a rescued work is already in the corpus"
        found = {}
        for line in open(args.union, encoding="utf-8"):
            work = json.loads(line)
            if work["id"] in rescued:
                found[work["id"]] = work
        assert len(found) == len(rescued), "some rescued works are missing from the candidate union"
        for wid in sorted(found):
            r = rescued[wid]
            reason = (f"rescue pass: text classifier P(core)={r['p_core']:.2f} (bge-large embeddings + logistic regression)"
                      if r["stage"] == "classifier" else
                      f"rescue pass: classifier P(core)={r['p_core']:.2f} in the uncertain band, "
                      f"adjudicated CORE by Eikos-4B (P={r['eikos_p_core']:.2f})")
            row = {**found[wid], "quantum_included": True, "quantum_inclusion_reasons": [reason], "rescue_pass": r}
            out.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
    tmp.replace(args.out)

    manifest = {
        "schema_version": 1, "stage": "R4 apply",
        "corpus_works": n_corpus, "rescued_works": len(rescued), "final_works": n_corpus + len(rescued),
        "rescued_by_stage": dict(collections.Counter(r["stage"] for r in rescued.values())),
        "band_outcomes": {k: v for k, v in stages.items() if k.startswith("band")},
        "thresholds": {"accept": accept, "reject": reject, "eikos_min_p": args.eikos_min_p},
        "inputs_sha256": {n: sha256(DATA / n) for n in ("rescue_scores.npz", "rescue_train_manifest.json", "rescue_pool_manifest.json")},
        "output": str(args.out), "output_sha256": sha256(args.out),
    }
    args.out.with_name(args.out.stem + "_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(f"wrote {args.out}: {n_corpus} corpus + {len(rescued)} rescued = {n_corpus + len(rescued)} works")
    if not args.no_extend_cache:
        print(f"main embedding cache extended with {extend_main_embedding_cache(rescued.keys())} rescued works")


if __name__ == "__main__":
    main()
