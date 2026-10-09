#!/usr/bin/env python3
"""Rescue pass, stage R3b: train the CORE classifier and score the whole pool.

Inputs
    corpus/data/rescue_pool.jsonl                 the pool (stage R1)
    corpus/data/rescue_pool_bge.npy + _ids.json   bge-large CLS embeddings of the pool (stage R2)
    corpus/data/rescue_label_sample.jsonl         the sample that was labelled (stage R3a)
    corpus/data/rescue_label_sample_eikos.json    its CORE / PERIPHERAL / NOT labels (Eikos-4B)

Model
    Logistic regression on L2-normalised bge-large embeddings, target = (label == CORE).

    The labelled sample is NOT a fair sample of the pool (the targeted stratum is enriched in
    likely positives), so every labelled work is weighted by 1 / its inclusion probability
    (Horvitz-Thompson). This restores the pool's real class balance, so P(core) means what it
    says *on the pool* and the thresholds below are meaningful.

Evaluation and thresholds
    5-fold cross-validation, out-of-fold. Metrics, precision, recall and the choice of
    thresholds use ONLY the uniform stratum, which is a fair sample of the pool.
      accept  the lowest threshold whose out-of-fold precision on the uniform stratum is at
              least TARGET_PRECISION (works at or above it are admitted by the classifier)
      reject  the highest threshold that still keeps TARGET_RECALL of the uniform positives
              (works below it are dropped without further review)
      band    everything in between is adjudicated by Eikos-4B (stage R4); the band is capped
              at --band-cap works, closest to 0.5 first, the rest stay excluded.

Outputs
    corpus/data/rescue_scores.npz          ids, p_core for every pool work
    corpus/data/rescue_band.jsonl          the band works that need an Eikos adjudication
    corpus/data/rescue_train_manifest.json hyper-parameters, metrics, thresholds, counts

Usage:  python rescue_train.py
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from rescue_sample import TARGET_LEXICON  # noqa: E402  (the same lexicon that steered sampling)

DATA = HERE.parent / "corpus" / "data"
TARGET_PRECISION = 0.90
TARGET_RECALL = 0.95
C_GRID = (0.1, 1.0, 10.0, 100.0, 1000.0)
SEED = 7


def load_pool(path):
    pool = {}
    for line in open(path, encoding="utf-8"):
        w = json.loads(line)
        pool[w["id"]] = w
    return pool


def inclusion_weights(sample, pool):
    """1 / P(work was drawn into the labelled sample), for every labelled work."""
    n = len(pool)
    uniform_ids = {s["id"] for s in sample if s["stratum"] == "uniform"}
    n_targeted = sum(1 for s in sample if s["stratum"] == "targeted")
    in_lexicon = {i: bool(TARGET_LEXICON.search(f"{w['display_name']} {w['abstract']}")) for i, w in pool.items()}
    n_candidates = sum(1 for i, hit in in_lexicon.items() if hit and i not in uniform_ids)
    p_uniform = len(uniform_ids) / n
    p_targeted = n_targeted / max(1, n_candidates)
    pi = {i: p_uniform + (1 - p_uniform) * (p_targeted if hit else 0.0) for i, hit in in_lexicon.items()}
    return {s["id"]: 1.0 / pi[s["id"]] for s in sample}


def out_of_fold(X, y, w, c, seed=SEED):
    oof = np.zeros(len(y))
    for train, test in StratifiedKFold(5, shuffle=True, random_state=seed).split(X, y):
        model = LogisticRegression(C=c, max_iter=3000)
        model.fit(X[train], y[train], sample_weight=w[train] / w[train].mean())
        oof[test] = model.predict_proba(X[test])[:, 1]
    return oof


def choose_thresholds(p_uniform, y_uniform):
    """Accept / reject thresholds from the uniform stratum (see module docstring)."""
    grid = np.round(np.arange(0.02, 0.995, 0.01), 2)
    accept = None
    for t in grid:
        sel = p_uniform >= t
        if sel.sum() >= 10 and y_uniform[sel].mean() >= TARGET_PRECISION:
            accept = float(t)
            break
    note = ""
    if accept is None:
        accept, note = 0.90, "no threshold reached the target precision on the uniform stratum; fell back to 0.90"
    reject = float(grid[0])
    positives = max(1, y_uniform.sum())
    for t in grid[grid <= accept][::-1]:
        if y_uniform[p_uniform >= t].sum() / positives >= TARGET_RECALL:
            reject = float(t)
            break
    return accept, reject, note


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--band-cap", type=int, default=6000,
                    help="max band works sent to Eikos, closest to 0.5 first. By the model's calibrated "
                         "estimates the first 4,000, 6,000 and 8,000 hold 63, 78 and 87 percent of the band's "
                         "core works; 6,000 (about 60 min of iGPU time) is the default trade-off")
    args = ap.parse_args()

    pool = load_pool(DATA / "rescue_pool.jsonl")
    emb = np.load(DATA / "rescue_pool_bge.npy").astype(np.float32)
    ids = json.load(open(DATA / "rescue_pool_bge_ids.json"))
    emb /= np.linalg.norm(emb, axis=1, keepdims=True)
    row = {w: i for i, w in enumerate(ids)}
    missing = [w for w in pool if w not in row]
    assert not missing, f"{len(missing)} pool works have no embedding; finish stage R2 first"

    sample = [json.loads(l) for l in open(DATA / "rescue_label_sample.jsonl", encoding="utf-8")]
    labels = {x["item"]: x for x in json.load(open(DATA / "rescue_label_sample_eikos.json"))}
    sample = [s for s in sample if s["item"] in labels]
    X = emb[[row[s["id"]] for s in sample]]
    y = np.array([labels[s["item"]]["label"] == "CORE" for s in sample])
    weights = inclusion_weights(sample, pool)
    w = np.array([weights[s["id"]] for s in sample])
    is_uniform = np.array([s["stratum"] == "uniform" for s in sample])
    print(f"labelled: {len(sample)} (uniform {is_uniform.sum()}, targeted {(~is_uniform).sum()}) | "
          f"CORE: uniform {y[is_uniform].sum()} ({y[is_uniform].mean():.1%}), targeted {y[~is_uniform].sum()} ({y[~is_uniform].mean():.1%})")

    best = None
    for c in C_GRID:
        oof = out_of_fold(X, y, w, c)
        ap_u = average_precision_score(y[is_uniform], oof[is_uniform])
        print(f"  C={c:<5} out-of-fold on uniform stratum: AUC={roc_auc_score(y[is_uniform], oof[is_uniform]):.3f} AP={ap_u:.3f}")
        if best is None or ap_u > best[0] + 1e-9:
            best = (ap_u, c, oof)
    _, c, oof = best
    accept, reject, note = choose_thresholds(oof[is_uniform], y[is_uniform])

    model = LogisticRegression(C=c, max_iter=3000).fit(X, y, sample_weight=w / w.mean())
    p_pool = model.predict_proba(emb[[row[i] for i in pool]])[:, 1]
    pool_ids = list(pool)
    n_accept = int((p_pool >= accept).sum())
    in_band = [(abs(p - 0.5), i) for i, p in zip(pool_ids, p_pool) if reject <= p < accept]
    in_band.sort()
    band = [i for _, i in in_band[: args.band_cap]]
    pu, yu = oof[is_uniform], y[is_uniform]
    sel = pu >= accept
    print(f"C={c} | accept>={accept:.2f} (uniform out-of-fold precision {yu[sel].mean():.2f} on {int(sel.sum())} works) | "
          f"reject<{reject:.2f} keeps {yu[pu >= reject].sum() / max(1, yu.sum()):.0%} of positives")
    print(f"pool of {len(pool_ids)}: accepted by classifier {n_accept} | band {len(in_band)} (sent to Eikos: {len(band)}) | "
          f"dropped {len(pool_ids) - n_accept - len(in_band)}")
    if note:
        print("NOTE:", note)

    np.savez(DATA / "rescue_scores.npz", ids=np.array(pool_ids), p_core=p_pool.astype(np.float32))
    with open(DATA / "rescue_band.jsonl", "w", encoding="utf-8") as f:
        for k, i in enumerate(band, 1):
            wk = pool[i]
            f.write(json.dumps({"item": f"rb_{k:05d}", "id": i, "year": wk["publication_year"],
                                "title": wk["display_name"], "abstract": wk["abstract"][:1400]}, ensure_ascii=False) + "\n")
    manifest = {
        "schema_version": 1, "stage": "R3b train + score",
        "labelled": {"total": len(sample), "uniform": int(is_uniform.sum()), "targeted": int((~is_uniform).sum()),
                     "core_uniform": int(y[is_uniform].sum()), "core_targeted": int(y[~is_uniform].sum()),
                     "labeller": "Eikos-4B (pipeline/label_with_eikos.py), rubric pipeline/rescue_rubric.py"},
        "model": {"type": "logistic regression on L2-normalised bge-large-en-v1.5 CLS embeddings",
                  "C": c, "c_grid": list(C_GRID), "weights": "inverse inclusion probability (Horvitz-Thompson)"},
        "cross_validation": {"folds": 5, "scope": "uniform stratum only",
                             "auc": float(roc_auc_score(yu, pu)), "average_precision": float(average_precision_score(yu, pu))},
        "thresholds": {"accept": accept, "reject": reject, "target_precision": TARGET_PRECISION,
                       "target_recall": TARGET_RECALL, "note": note,
                       "uniform_precision_at_accept": float(yu[sel].mean()) if sel.any() else None},
        "pool": {"size": len(pool_ids), "accepted_by_classifier": n_accept, "band": len(in_band),
                 "band_sent_to_eikos": len(band), "dropped": len(pool_ids) - n_accept - len(in_band)},
    }
    (DATA / "rescue_train_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
