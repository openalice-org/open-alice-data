#!/usr/bin/env python3
"""SPRINT 13 — BGE domain classification (Nordic anchors).

Embeds four application-domain anchors and every matched quantum work with
BAAI/bge-large-en-v1.5 (CLS token), computes per-work domain distributions
via softmax(cos_sim / tau), aggregates per person, and MERGES the result into
app/data/people_taxonomy.json (adds fields, never removes).

Anchor text for computing/communication/sensing comes from
config/quantum_area_anchors.json (vendored from
KTH.Tools.QuantumAuthors/config/sweden_quantum_area_anchors.json — each
domain has multiple anchor sentences, embedded separately and averaged into
one centroid per domain). "other" has no equivalent in that config and stays
a hardcoded catch-all definition.

Anchor-centrality debias: raw cosine similarities are biased — some anchors
(sensing, other) sit closer to ALL works (column means 0.714 vs 0.692-0.697),
which made sensing absorb 61% of works and flipped Val Zwiller away from
communication. Each similarity column is therefore centered by its mean over
the matched-work corpus before the softmax. This raised work-level agreement
with the lexicon weak truth from 0.62 to 0.73 and restored the expected
labels for the sanity people (Zwiller -> communication, Bjork -> computing).

tau is calibrated against the lexicon classification of build_taxonomy_data.py:
works of high-lexicon-confidence people (domain_confidence >= 0.3) are weak
ground truth. Note: work-level argmax over the 3 real domains is invariant to
tau (softmax is monotone), so tau is chosen to maximize PERSON-level agreement
(mean-of-distributions aggregation does depend on tau), with a sharpness
guardrail so distributions are neither uniform mush nor winner-take-all. The
anchor config's own softmax_temperature is added to the grid so it competes
on equal footing with the calibrated candidates instead of being trusted
blindly for a different corpus/use case.

Run (taxonomy first):
  "../.venv/bin/python3" "build_taxonomy_data.py"
  "../.venv/bin/python3" "build_domain_specter.py"
"""

import hashlib
import json
import os
import pathlib
import sys
from collections import Counter, defaultdict
from datetime import datetime, timezone

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from build_taxonomy_data import (  # noqa: E402
    ACCEPTED_LABELS,
    OUT_PATH as TAXONOMY_PATH,
    SPRINT2_WORKS,
    reconstruct_abstract,
)

SPRINT11 = pathlib.Path(__file__).resolve().parent
CACHE_DIR = str(SPRINT11 / "caches")
EMB_CACHE = os.path.join(CACHE_DIR, "work_embeddings_bge.npy")
IDS_CACHE = os.path.join(CACHE_DIR, "work_embedding_ids_bge.json")
ANCHORS_CONFIG_PATH = SPRINT11 / "config" / "quantum_area_anchors.json"
WORK_DOMAIN_LABELS_PATH = SPRINT11 / "data" / "work_domain_labels.json"

MAX_LENGTH = 512

REAL_DOMAINS = ("computing", "communication", "sensing")
ALL_DOMAINS = REAL_DOMAINS + ("other",)

# Lexicon weak-ground-truth threshold and tau grid for calibration.
CONF_THRESHOLD = 0.3
TAU_GRID = (0.01, 0.02, 0.03, 0.04, 0.05, 0.07, 0.1)

# "other" has no equivalent area in the anchors config — kept as a hardcoded
# catch-all so foundational/materials work still has somewhere to land.
OTHER_DEFINITION = (
    "Foundations and materials of quantum science: theoretical quantum "
    "mechanics and quantum field theory, condensed matter and "
    "solid-state physics, quantum optics fundamentals, many-body "
    "theory, semiconductor physics, material growth and "
    "characterization — fundamental quantum research without a specific "
    "application to computing, communication, or sensing."
)


def load_anchors_config():
    """Load the vendored Nordic anchor config.

    Returns (model_name, temperature, domain_anchor_texts) where
    domain_anchor_texts maps computing/communication/sensing -> list of
    anchor sentences, plus "other" -> [OTHER_DEFINITION].
    """
    with open(ANCHORS_CONFIG_PATH, encoding="utf-8") as f:
        cfg = json.load(f)
    domain_anchor_texts = {area["domain_key"]: list(area["anchors"])
                            for area in cfg["areas"]}
    missing = set(REAL_DOMAINS) - set(domain_anchor_texts)
    if missing:
        raise ValueError(f"Anchor config missing domains: {sorted(missing)}")
    domain_anchor_texts["other"] = [OTHER_DEFINITION]
    return cfg["encoder"]["model"], float(cfg["softmax_temperature"]), domain_anchor_texts


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------
def load_matched_works(person_ids):
    """Return (works: {wid: text}, work_people: {wid: set(pid)}).

    Same filtering/matching rules as build_taxonomy_data.main().
    """
    works = {}
    work_people = defaultdict(set)
    with open(SPRINT2_WORKS, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                work = json.loads(line)
            except json.JSONDecodeError:
                continue
            if work.get("classification_label") not in ACCEPTED_LABELS:
                continue
            matched = set()
            for auth in work.get("authorships") or []:
                if not isinstance(auth, dict):
                    continue
                a = auth.get("author") or {}
                aid = a.get("id") if isinstance(a, dict) else None
                if aid in person_ids:
                    matched.add(aid)
            if not matched:
                continue
            wid = work.get("id")
            if not wid:
                continue
            title = work.get("display_name") or ""
            abstract = reconstruct_abstract(work.get("abstract_inverted_index"))
            works[wid] = (title, abstract)
            work_people[wid] |= matched
    return works, work_people


# ---------------------------------------------------------------------------
# Embedding (model loaded lazily — exactly as get_paper_distance_v3.py)
# ---------------------------------------------------------------------------
def make_embedder(model_name):
    import torch
    from transformers import AutoTokenizer, AutoModel

    cuda = torch.cuda.is_available()
    device = torch.device("cuda" if cuda else "cpu")
    if cuda:
        print(f"[bge] device=cuda ({torch.cuda.get_device_name(0)})")
    else:
        print("[bge] device=cpu (CUDA not available — slower)")

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name)
    if cuda:
        model = model.half()  # fp16 — 6GB VRAM budget, bge-large is 1024-dim
    model.to(device)
    model.eval()

    batch_size = 48 if cuda else 8

    def embed(texts, desc="embedding"):
        try:
            from tqdm import tqdm
            it = tqdm(range(0, len(texts), batch_size), desc=desc)
        except ImportError:
            it = range(0, len(texts), batch_size)
        out = []
        for i in it:
            batch = texts[i:i + batch_size]
            inputs = tokenizer(
                batch, padding=True, truncation=True, return_tensors="pt",
                max_length=MAX_LENGTH,
            )
            inputs = {k: v.to(device) for k, v in inputs.items()}
            with torch.no_grad():
                output = model(**inputs)
                # BGE's documented pooling: CLS token (index 0), L2-normalized
                # downstream alongside every other embedding in main().
                emb = output.last_hidden_state[:, 0, :]
            out.append(emb.float().cpu().numpy())  # back to fp32 for cosine/softmax math
        return np.vstack(out)

    return embed, tokenizer, ("cuda" if cuda else "cpu")


def get_work_embeddings(works, model_name):
    """Embed all works, using/extending the on-disk cache. Returns
    (ids list, embeddings ndarray [n,1024], sep_token, device str)."""
    wids = sorted(works)

    cached_ids, cached_emb = [], None
    if os.path.exists(EMB_CACHE) and os.path.exists(IDS_CACHE):
        with open(IDS_CACHE, encoding="utf-8") as f:
            cached_ids = json.load(f)
        cached_emb = np.load(EMB_CACHE)
        if len(cached_ids) != cached_emb.shape[0]:
            print("[cache] id/embedding length mismatch — rebuilding")
            cached_ids, cached_emb = [], None

    cache_idx = {wid: i for i, wid in enumerate(cached_ids)}
    missing = [w for w in wids if w not in cache_idx]
    print(f"[cache] {len(wids) - len(missing)}/{len(wids)} work embeddings "
          f"cached; {len(missing)} to embed")

    embed = tokenizer = device = None
    if missing:
        embed, tokenizer, device = make_embedder(model_name)
        sep = tokenizer.sep_token
        texts = [f"{works[w][0]}{sep}{works[w][1]}" for w in missing]
        new_emb = embed(texts, desc="works")
        if cached_emb is not None and len(cached_ids):
            all_ids = cached_ids + missing
            all_emb = np.vstack([cached_emb, new_emb])
        else:
            all_ids, all_emb = missing, new_emb
        os.makedirs(CACHE_DIR, exist_ok=True)
        np.save(EMB_CACHE, all_emb)
        with open(IDS_CACHE, "w", encoding="utf-8") as f:
            json.dump(all_ids, f)
        cached_ids, cached_emb = all_ids, all_emb
        cache_idx = {wid: i for i, wid in enumerate(cached_ids)}

    emb = cached_emb[[cache_idx[w] for w in wids]]
    return wids, emb, embed, device


# ---------------------------------------------------------------------------
# Distributions / calibration
# ---------------------------------------------------------------------------
def softmax_rows(x):
    x = x - x.max(axis=1, keepdims=True)
    e = np.exp(x)
    return e / e.sum(axis=1, keepdims=True)


def work_distributions(sims, tau):
    """sims [n,4] cosine similarities -> [n,4] probability rows."""
    return softmax_rows(sims / tau)


def person_aggregate(probs, person_work_idx):
    """Mean of a person's work distributions, normalized. {pid: [4]}"""
    agg = {}
    for pid, idxs in person_work_idx.items():
        if not idxs:
            continue
        m = probs[idxs].mean(axis=0)
        agg[pid] = m / m.sum()
    return agg


def calibrate_tau(sims, person_work_idx, lexicon_truth, tau_grid=TAU_GRID):
    """Pick tau from tau_grid maximizing person-level agreement with the
    lexicon (high-confidence people only). Returns (tau, table rows)."""
    dom_idx = {d: i for i, d in enumerate(REAL_DOMAINS)}

    # Work-level argmax over the 3 real domains is tau-invariant.
    work_argmax = sims[:, :3].argmax(axis=1)
    n_w, n_w_ok = 0, 0
    for pid, true_dom in lexicon_truth.items():
        for i in person_work_idx.get(pid, ()):
            n_w += 1
            n_w_ok += int(work_argmax[i] == dom_idx[true_dom])
    work_acc = n_w_ok / n_w if n_w else 0.0

    rows = []
    for tau in tau_grid:
        probs = work_distributions(sims, tau)
        agg = person_aggregate(probs, person_work_idx)
        n_p, n_p_ok = 0, 0
        for pid, true_dom in lexicon_truth.items():
            if pid not in agg:
                continue
            n_p += 1
            real = agg[pid][:3]
            if int(real.argmax()) == dom_idx[true_dom]:
                n_p_ok += 1
        person_acc = n_p_ok / n_p if n_p else 0.0
        mean_top = float(probs.max(axis=1).mean())
        rows.append({"tau": tau, "person_agreement": round(person_acc, 4),
                     "work_agreement": round(work_acc, 4),
                     "mean_work_top_prob": round(mean_top, 4),
                     "n_people": n_p, "n_works": n_w})

    # Best person-level agreement; among near-ties (within 0.005) prefer the
    # tau whose mean work top-prob is closest to 0.7 (sharp but not
    # winner-take-all).
    best = max(r["person_agreement"] for r in rows)
    near = [r for r in rows if best - r["person_agreement"] <= 0.005]
    chosen = min(near, key=lambda r: abs(r["mean_work_top_prob"] - 0.7))
    return chosen["tau"], rows


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    model_name, anchor_tau, domain_anchor_texts = load_anchors_config()
    print(f"Anchors: {ANCHORS_CONFIG_PATH} (model={model_name}, "
          f"config_tau={anchor_tau})")

    with open(TAXONOMY_PATH, encoding="utf-8") as f:
        data = json.load(f)
    people = data["people"]
    person_ids = {p["id"] for p in people}
    print(f"Loaded {len(people)} people from {TAXONOMY_PATH}")

    works, work_people = load_matched_works(person_ids)
    print(f"Matched quantum works to embed: {len(works)}")

    wids, work_emb, embed_fn, device = get_work_embeddings(works, model_name)
    if embed_fn is None:  # everything was cached; still need anchors embedded
        embed_fn, _tok, device = make_embedder(model_name)

    # Each domain anchor is a list of sentences — embed them individually and
    # average into one centroid vector per domain (same idea as the
    # centroid-of-positives trick in specter_review_corpus.py).
    anchor_sentences, anchor_owner = [], []
    for d in ALL_DOMAINS:
        for text in domain_anchor_texts[d]:
            anchor_sentences.append(text)
            anchor_owner.append(d)
    anchor_emb = embed_fn(anchor_sentences, desc="anchors")
    def_emb = np.vstack([
        anchor_emb[[i for i, d in enumerate(anchor_owner) if d == domain]].mean(axis=0)
        for domain in ALL_DOMAINS
    ])

    # Cosine similarity works x domains
    wn = work_emb / np.linalg.norm(work_emb, axis=1, keepdims=True)
    dn = def_emb / np.linalg.norm(def_emb, axis=1, keepdims=True)
    sims_raw = wn @ dn.T  # [n_works, 4]
    print(f"Raw similarity: min={sims_raw.min():.4f} max={sims_raw.max():.4f} "
          f"mean_row_spread={(sims_raw.max(1) - sims_raw.min(1)).mean():.4f}")
    col_means = sims_raw.mean(axis=0)
    print("Per-anchor mean similarity (centrality bias): "
          + ", ".join(f"{d}={col_means[i]:.4f}"
                      for i, d in enumerate(ALL_DOMAINS)))
    # Debias anchor centrality: center each column by its corpus mean.
    sims = sims_raw - col_means[None, :]

    # person -> indices into wids
    wid_idx = {w: i for i, w in enumerate(wids)}
    person_work_idx = defaultdict(list)
    for wid, pids in work_people.items():
        for pid in pids:
            person_work_idx[pid].append(wid_idx[wid])

    # Lexicon weak ground truth: high-confidence people
    lexicon_truth = {
        p["id"]: p["domain_top"] for p in people
        if p.get("domain_top") in REAL_DOMAINS
        and p.get("domain_confidence", 0) >= CONF_THRESHOLD
    }
    print(f"Calibration set: {len(lexicon_truth)} people with lexicon "
          f"domain_confidence >= {CONF_THRESHOLD}")

    tau_grid = tuple(sorted(set(TAU_GRID) | {anchor_tau}))
    tau, table = calibrate_tau(sims, person_work_idx, lexicon_truth, tau_grid)
    print("\n=== tau calibration (agreement vs lexicon weak truth) ===")
    print(f"  {'tau':>6} {'person_agr':>11} {'work_agr':>9} {'mean_top_p':>11}")
    for r in table:
        mark = " <-- chosen" if r["tau"] == tau else ""
        print(f"  {r['tau']:>6} {r['person_agreement']:>11} "
              f"{r['work_agreement']:>9} {r['mean_work_top_prob']:>11}{mark}")
    print(f"Chosen tau = {tau}")

    # Final distributions and per-person fields
    probs = work_distributions(sims, tau)
    agg = person_aggregate(probs, person_work_idx)

    # Per-WORK hard domain label (argmax over the 3 real domains — "other" never
    # wins here since sims/probs already come from a corpus pre-filtered to
    # core/likely_quantum works, so it's not a competing option the way it is
    # for a person-level unknown/low-confidence check). Lets build_paper_counts.py
    # count each distinct paper into exactly one institution/country/domain
    # bucket, instead of index.html re-deriving a "mix" from each co-author's own
    # (person-aggregated-mean, double-counted) shares.
    real_idx = [ALL_DOMAINS.index(d) for d in REAL_DOMAINS]
    work_domain = {wid: REAL_DOMAINS[int(probs[i, real_idx].argmax())] for wid, i in wid_idx.items()}
    with open(WORK_DOMAIN_LABELS_PATH, "w", encoding="utf-8") as f:
        json.dump(work_domain, f)
    print(f"Wrote per-work hard domain labels for {len(work_domain)} works to {WORK_DOMAIN_LABELS_PATH}")

    for p in people:
        pid = p["id"]
        n = len(person_work_idx.get(pid, ()))
        if pid in agg:
            vec = agg[pid]
            dist = {d: float(vec[i]) for i, d in enumerate(ALL_DOMAINS)}
        else:
            dist = {"computing": 0.0, "communication": 0.0, "sensing": 0.0,
                    "other": 1.0}
        p_other = dist["other"]
        real_mass = sum(dist[d] for d in REAL_DOMAINS)

        if n == 0 or p_other >= 0.85 or real_mass <= 0:
            top, conf = "unknown", 0.0
        else:
            ranked = sorted(REAL_DOMAINS, key=lambda d: (-dist[d], d))
            top = ranked[0]
            q = [dist[d] / real_mass for d in ranked]
            conf = (q[0] - q[1]) * (n / (n + 3.0)) * (1.0 - p_other)
            conf = max(0.0, min(1.0, conf))

        p["domain_probs_specter"] = {d: round(dist[d], 4) for d in ALL_DOMAINS}
        p["domain_top_specter"] = top
        p["domain_confidence_specter"] = round(conf, 4)
        p["domain_evidence_specter"] = n

    # ----------------------------------------------------------- merge meta
    def_hash = hashlib.sha256(
        "\n".join(t for d in ALL_DOMAINS for t in domain_anchor_texts[d]).encode("utf-8")
    ).hexdigest()
    data["meta"]["generated_utc"] = datetime.now(timezone.utc).isoformat()
    data["meta"]["specter"] = {
        "model": model_name,
        "anchors_config": ANCHORS_CONFIG_PATH.relative_to(SPRINT11.parent).as_posix(),
        "anchors_config_tau": anchor_tau,
        "device": device,
        "tau": tau,
        "sim_centering": {
            "method": "per-anchor column mean subtracted before softmax "
                      "(anchor centrality debias)",
            "anchor_mean_sims": {d: round(float(col_means[i]), 4)
                                 for i, d in enumerate(ALL_DOMAINS)},
        },
        "definitions": domain_anchor_texts,
        "definitions_hash": def_hash,
        "calibration": {
            "weak_truth": f"lexicon domain_confidence >= {CONF_THRESHOLD}",
            "n_people": len(lexicon_truth),
            "agreement_table": table,
        },
    }

    with open(TAXONOMY_PATH, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=1)
    size = os.path.getsize(TAXONOMY_PATH)
    print(f"\nMerged into {TAXONOMY_PATH} ({size/1024:.1f} KiB)")

    # ----------------------------------------------------------- reporting
    print("\n=== Domain distribution: specter vs lexicon ===")
    spec_c = Counter(p["domain_top_specter"] for p in people)
    lex_c = Counter(p["domain_top"] for p in people)
    print(f"  {'domain':<14} {'specter':>8} {'lexicon':>8}")
    for d in ("computing", "communication", "sensing", "unknown"):
        print(f"  {d:<14} {spec_c.get(d, 0):>8} {lex_c.get(d, 0):>8}")
    sc = [p["domain_confidence_specter"] for p in people
          if p["domain_top_specter"] != "unknown"]
    lc = [p["domain_confidence"] for p in people
          if p["domain_top"] != "unknown"]
    if sc:
        print(f"  mean conf (non-unknown): specter {sum(sc)/len(sc):.4f} "
              f"vs lexicon {sum(lc)/len(lc):.4f}")

    print("\n=== Sanity checks ===")
    # NB people_viz contains duplicate names with distinct OpenAlex ids
    # (e.g. two "Val Zwiller"); keep the entry with the most evidence.
    by_name = {}
    for p in people:
        cur = by_name.get(p["name"])
        if cur is None or p["tax_evidence"] > cur["tax_evidence"]:
            by_name[p["name"]] = p
    sensing_people = [p for p in people if p["domain_top_specter"] == "sensing"]
    top_sensing = (max(sensing_people,
                       key=lambda p: p["domain_confidence_specter"])
                   if sensing_people else None)
    checks = [by_name.get("Gunnar Björk"), by_name.get("Val Zwiller"),
              top_sensing]
    for p in checks:
        if not p:
            continue
        print(f"  {p['name']}:")
        print(f"    specter: {p['domain_probs_specter']} -> "
              f"{p['domain_top_specter']} "
              f"(conf {p['domain_confidence_specter']}, "
              f"evid {p['domain_evidence_specter']})")
        print(f"    lexicon: {p['domain_probs']} -> {p['domain_top']} "
              f"(conf {p['domain_confidence']})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
