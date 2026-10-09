#!/usr/bin/env python3
"""
build_people_data.py — Sprint 5 data pipeline for the "people map" app.

Turns the Qsip paper + author corpus into ONE JSON the monolithic D3 app
fetches: one record per PERSON, positioned on two ABSTRACT axes.

    X axis : Tourist (left, -1)  <->  Resident (right, +1)
             "works on quantum in SOME of their papers" vs "works on it in
             MOST/ALL of their papers".
             -> quantum SHARE of the person's papers:
                  share = quantum_papers / total_papers
                  x_raw = 2 * share - 1   (share 0 -> -1, share 1 -> +1)

             total_papers = ALL works of the person (author_summary.csv
             paper_total_count). quantum_papers = papers the SPRINT 2
             classifier called quantum AND the OpenRouter LLM verifier
             agreed were quantum (see the verification layer below).

    Y axis : Thinker (bottom, -1)  <->  Builder (top, +1)
             "theoretical science" vs "applied science", from TITLE + ABSTRACT.
             -> three methods (selectable via --method):
                  --method llm       : average the per-paper applied_theoretical
                                       verdicts (from the OpenRouter verifier)
                                       over the person's quantum papers. SPECTER2
                                       is used as a SECONDARY/FALLBACK score when a
                                       person has no usable LLM verdicts. (default)
                  --method specter2  : SPECTER2 embeddings + cosine similarity to
                                       axis-label embeddings, on GPU (cuda) when
                                       available. No LLM Y signal.
                  --method heuristic : applied-vs-theory lexicon over the person's
                                       abstracts / topics / keywords. Pure stdlib.

OpenRouter LLM verification layer (NEW):
    Every paper's (title + abstract) is sent to OpenRouter for a JSON verdict:
        {"is_quantum": bool,
         "applied_theoretical": float in [-1,+1]   # -1 theoretical, +1 applied}
    Results are cached append-only in caches/llm_verify.jsonl keyed by paper id,
    so reruns skip already-verified papers (resumable, cheap). Requests run
    concurrently with retry/backoff on 429/5xx. The model comes from the example
    file (operouter_example.md): "openrouter/free" (auto-routes to a free model,
    e.g. openai/gpt-oss-20b:free) -> $0 cost.

Point SIZE in the app toggles (the "glasses") between:
    - h_index      : REAL h-index from each person's papers' cited_by_count.
    - total_patents: KTH/SE-validated patent count (patent_kth_filtered_count).

Usage:
    python build_people_data.py                       # llm (default)
    python build_people_data.py --method specter2     # embeddings only
    python build_people_data.py --no-llm              # skip the OpenRouter pass
    python build_people_data.py --min-works 2 --out app/data/people_viz.json
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import re
import sys
import threading
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path

# ----------------------------------------------------------------------------
# Paths (relative to the Qsip client root, which is this file's parent.parent)
# ----------------------------------------------------------------------------
ROOT = Path(__file__).resolve().parent.parent
SPRINT11 = Path(__file__).resolve().parent
DEFAULT_OUT = SPRINT11 / "data" / "people_viz.json"

# LLM verifier configuration
OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"
OPENROUTER_KEY_FILE = SPRINT11 / "openrouter_key.txt"
LLM_CACHE = SPRINT11 / "caches" / "llm_verify.jsonl"
# Model from operouter_example.md: "openrouter/free" auto-routes to a free model.
LLM_MODEL = "openrouter/free"
LLM_WORKERS = 12          # concurrent requests
LLM_MAX_RETRIES = 5
LLM_TIMEOUT = 90          # seconds per request

CURRENT_YEAR = 2026  # project "now"; keep in sync with the corpus tail.

# Confidence shrinkage: a person's axis position is pulled toward the neutral
# centre by  n / (n + K), where n = their quantum_papers count. People with 1-2
# quantum papers carry little evidence, so they should NOT sit at the extremes —
# a single abstract is noise, not a strong "builder" or "thinker" signal.
DEFAULT_CONFIDENCE_K = 2.0

# Paper classification labels that count as "quantum" for the quantum_papers
# tally. The corpus uses {core_quantum, likely_quantum, reject}. We treat both
# core_quantum and likely_quantum as candidate quantum work; reject is excluded.
# (There is no "review" label in this corpus — the spec's "review" maps to
# nothing here; had one existed it would have been resolved by the LLM verdict.)
QUANTUM_LABELS = {"core_quantum", "likely_quantum"}

# ----------------------------------------------------------------------------
# Axis configuration — the axes are deliberately "abstract" / swappable.
# ----------------------------------------------------------------------------
AXES = {
    "x": {"neg": "Tourist", "pos": "Resident", "field": "quantum_share",
          "desc": "Quantum in some papers vs. quantum in most/all papers"},
    "y": {"neg": "Thinker", "pos": "Builder", "field": "builder_thinker",
          "desc": "Theoretical science vs. applied science"},
}

# Axis-label prompts for the SPECTER2 path (mirrors get_paper_distance_v3.py).
SPECTER_AXIS_LABELS = {
    "builder": "Applied experimental physics: device fabrication, hardware, "
               "measurement, engineering and demonstration of quantum systems.",
    "thinker": "Theoretical and foundational physics: formalism, mathematical "
               "proofs, models, and abstract principles of quantum mechanics.",
}

# Lexicon for the heuristic path. Weighted term -> applied(+) / theory(-).
APPLIED_TERMS = {
    "experiment", "experimental", "experimentally", "device", "devices",
    "fabricate", "fabricated", "fabrication", "chip", "on-chip", "photonic",
    "photonics", "sensor", "sensing", "detector", "detection", "implementation",
    "implement", "demonstrate", "demonstration", "prototype", "integrated",
    "circuit", "circuits", "hardware", "measurement", "measured", "nanowire",
    "waveguide", "superconducting", "fabricating", "manufacturing", "platform",
    "scalable", "scalability", "engineering", "engineered", "application",
    "applications", "readout", "fidelity", "qubit", "qubits", "cryostat",
    "wafer", "lithography", "deposition", "characterization", "prototype",
    "calibration", "interferometer", "cavity", "laser", "diode", "transmon",
}
THEORY_TERMS = {
    "theory", "theoretical", "theoretically", "theorem", "proof", "prove",
    "formalism", "framework", "analytic", "analytical", "mathematical",
    "foundational", "foundations", "principle", "principles", "conjecture",
    "derivation", "derive", "derived", "model", "models", "equation",
    "equations", "hamiltonian", "lagrangian", "bound", "bounds", "complexity",
    "abstract", "axiom", "lemma", "inequality", "asymptotic", "perturbative",
    "renormalization", "symmetry", "topological", "ansatz", "variational",
    "entanglement", "nonlocality", "decoherence", "formal", "rigorous",
    "conjectured", "paradigm", "interpretation",
}

# ----------------------------------------------------------------------------
# Small helpers
# ----------------------------------------------------------------------------
_word_re = re.compile(r"[a-z][a-z\-]+")
_json_re = re.compile(r"\{.*\}", re.S)


def norm_name(name: str) -> str:
    """Normalize an author display name for fuzzy matching."""
    if not name:
        return ""
    n = name.strip().lower().replace(".", " ")
    n = re.sub(r"\s+", " ", n)
    return n.strip()


def orcid_key(orcid: str) -> str:
    """Reduce an ORCID (URL or bare) to its 16-digit id."""
    if not orcid:
        return ""
    return orcid.rstrip("/").split("/")[-1].strip().lower()


def reconstruct_abstract(inv_index) -> str:
    """Rebuild abstract text from an OpenAlex abstract_inverted_index."""
    if not inv_index:
        return ""
    positions = []
    for word, idxs in inv_index.items():
        for i in idxs:
            positions.append((i, word))
    positions.sort()
    return " ".join(w for _, w in positions)


def h_index(citations: list[int]) -> int:
    """Standard h-index from a list of per-paper citation counts."""
    s = sorted((c for c in citations if c is not None), reverse=True)
    h = 0
    for i, c in enumerate(s, start=1):
        if c >= i:
            h = i
        else:
            break
    return h


def clamp(v: float, lo: float = -1.0, hi: float = 1.0) -> float:
    return max(lo, min(hi, v))


def quantum_share_score(quantum_papers: int, total_papers: int) -> tuple[float, float]:
    """
    Tourist(-1) <-> Resident(+1), from the quantum SHARE of a person's papers.

        share  = quantum_papers / total_papers   (guarded against div-by-zero
                 and clamped to [0,1] in case quantum_papers > total_papers)
        x_raw  = 2 * share - 1

    Returns (x_raw, share).
    """
    if not total_papers or total_papers <= 0:
        return 0.0, 0.0
    share = quantum_papers / total_papers
    share = max(0.0, min(1.0, share))
    return clamp(2.0 * share - 1.0), share


# ----------------------------------------------------------------------------
# OpenRouter LLM verification layer
# ----------------------------------------------------------------------------
def _load_openrouter_key() -> str | None:
    """Read the OpenRouter key from env or the key file (never printed)."""
    env = os.environ.get("OPENROUTER_API_KEY") or os.environ.get("OPENROUTER_KEY")
    if env:
        return env.strip()
    if not OPENROUTER_KEY_FILE.exists():
        return None
    for line in OPENROUTER_KEY_FILE.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        if "=" in line and line.upper().startswith("OPENROUTER"):
            return line.split("=", 1)[1].strip()
        if line.startswith("sk-"):
            return line
    return None


def _load_llm_cache() -> dict:
    """Load the append-only verdict cache keyed by paper id."""
    cache = {}
    if not LLM_CACHE.exists():
        return cache
    with open(LLM_CACHE, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            pid = rec.get("id")
            if pid:
                cache[pid] = rec
    return cache


_VERIFY_PROMPT = (
    "You are a physics paper classifier. Given a paper TITLE and ABSTRACT, "
    "decide (1) whether it is about quantum science/technology, and (2) where "
    "it sits on a theoretical<->applied scale.\n"
    "Respond with ONE compact JSON object and nothing else:\n"
    '{"is_quantum": <true|false>, "applied_theoretical": <float in [-1,1]>}\n'
    "applied_theoretical: -1 = purely theoretical (formalism, proofs, abstract "
    "models), +1 = purely applied (device fabrication, hardware, measurement, "
    "engineering, demonstration). 0 = balanced.\n\n"
    "Title: __TITLE__\nAbstract: __ABSTRACT__"
)


def _verify_one(session, key, pid, title, abstract) -> dict:
    """Call OpenRouter for a single paper with retry/backoff. Returns a verdict
    record {id, is_quantum, applied_theoretical, ok, error?}."""
    abstract = (abstract or "")[:4000]
    prompt = (_VERIFY_PROMPT
              .replace("__TITLE__", title or "(no title)")
              .replace("__ABSTRACT__", abstract or "(no abstract)"))
    payload = {
        "model": LLM_MODEL,
        "stream": False,
        "messages": [{"role": "user", "content": prompt}],
    }
    headers = {"Authorization": f"Bearer {key}", "Content-Type": "application/json"}
    last_err = ""
    for attempt in range(LLM_MAX_RETRIES):
        try:
            r = session.post(OPENROUTER_URL, headers=headers, json=payload,
                             timeout=LLM_TIMEOUT)
        except Exception as e:  # noqa: BLE001 - network errors, retry
            last_err = f"req:{type(e).__name__}"
            time.sleep(min(2 ** attempt, 30))
            continue
        if r.status_code == 429 or r.status_code >= 500:
            last_err = f"http:{r.status_code}"
            # honour Retry-After when present
            ra = r.headers.get("Retry-After")
            delay = float(ra) if (ra and ra.isdigit()) else min(2 ** attempt, 30)
            time.sleep(delay)
            continue
        if r.status_code != 200:
            return {"id": pid, "ok": False, "error": f"http:{r.status_code}"}
        try:
            j = r.json()
            content = j["choices"][0]["message"]["content"]
        except Exception:  # noqa: BLE001
            last_err = "badbody"
            time.sleep(min(2 ** attempt, 15))
            continue
        m = _json_re.search(content or "")
        if not m:
            last_err = "nojson"
            time.sleep(min(2 ** attempt, 15))
            continue
        try:
            v = json.loads(m.group(0))
        except json.JSONDecodeError:
            last_err = "parsefail"
            time.sleep(min(2 ** attempt, 15))
            continue
        try:
            at = float(v.get("applied_theoretical"))
        except (TypeError, ValueError):
            at = 0.0
        return {
            "id": pid,
            "is_quantum": bool(v.get("is_quantum")),
            "applied_theoretical": clamp(at),
            "model": j.get("model", LLM_MODEL),
            "ok": True,
        }
    return {"id": pid, "ok": False, "error": last_err or "retries_exhausted"}


def verify_papers_llm(papers: list[dict]) -> dict:
    """
    Run the OpenRouter verification pass over `papers` (each: {id, title,
    abstract}). Append-only cache => only un-cached ids are queried. Concurrent
    with retry. Returns {paper_id: verdict_record}.
    """
    import requests

    key = _load_openrouter_key()
    if not key:
        print("[llm] no OpenRouter key found (env or openrouter_key.txt); "
              "skipping verification.", file=sys.stderr)
        return {}

    cache = _load_llm_cache()
    todo = [p for p in papers if p["id"] not in cache or not cache[p["id"]].get("ok")]
    print(f"[llm] {len(cache)} cached, {len(todo)} to verify "
          f"({len(papers)} total) with model={LLM_MODEL}", file=sys.stderr)

    if not todo:
        return cache

    LLM_CACHE.parent.mkdir(parents=True, exist_ok=True)
    write_lock = threading.Lock()
    cache_f = open(LLM_CACHE, "a", encoding="utf-8")
    session = requests.Session()

    done = 0
    n_ok = 0
    t0 = time.time()
    try:
        with ThreadPoolExecutor(max_workers=LLM_WORKERS) as ex:
            futs = {ex.submit(_verify_one, session, key, p["id"], p["title"],
                              p["abstract"]): p["id"] for p in todo}
            for fut in as_completed(futs):
                rec = fut.result()
                cache[rec["id"]] = rec
                with write_lock:
                    cache_f.write(json.dumps(rec, ensure_ascii=False) + "\n")
                    cache_f.flush()
                done += 1
                if rec.get("ok"):
                    n_ok += 1
                if done % 50 == 0 or done == len(todo):
                    rate = done / max(1e-6, time.time() - t0)
                    print(f"[llm] {done}/{len(todo)} ({n_ok} ok) "
                          f"{rate:.1f}/s", file=sys.stderr)
    finally:
        cache_f.close()
        session.close()
    return cache


# ----------------------------------------------------------------------------
# Load authors (CSV v2)
# ----------------------------------------------------------------------------
def load_authors(authors_csv):
    by_orcid, authors = {}, {}
    name_ids = {}
    with open(authors_csv, encoding="utf-8") as f:
        for row in csv.DictReader(f):
            aid = row["author_openalex_id"].strip()
            if not aid:
                continue

            def to_int(v):
                try:
                    return int(float(v))
                except (TypeError, ValueError):
                    return None

            rec = {
                "id": aid,
                "name": row["author_name"].strip(),
                "orcid": row.get("orcid", "").strip(),
                "tier": row.get("kth_author_tier", "").strip(),
                "quantum_work_count": to_int(row.get("quantum_work_count")) or 0,
                # CSV quantum-only span: kept as last-resort fallback for first/last_year.
                "csv_first_quantum_year": to_int(row.get("first_quantum_year")),
                "csv_last_quantum_year": to_int(row.get("last_quantum_year")),
                "top_topics": [t.strip() for t in
                               (row.get("top_primary_topics") or "").split(";;") if t.strip()],
                "top_keywords": [k.strip() for k in
                                 (row.get("top_keywords") or "").split(";;") if k.strip()],
                "example_titles": [t.strip() for t in
                                   (row.get("example_titles") or "").split(";;") if t.strip()][:5],
                # accumulators filled in from papers
                "_citations": [],
                "_abstracts": [],          # title + abstract text, for Y axis
                "_quantum_paper_ids": [],  # paper ids classified quantum (SPRINT 2)
                # min/max publication_year over ALL attributed papers (any label /
                # verdict) -> person's overall paper span (see build()).
                "_min_pub_year": None,
                "_max_pub_year": None,
                # per-year {papers, citations} over QUANTUM-labeled papers only
                # (matches quantum_papers/example_titles scope) -> emitted as the
                # new, additive "yearly" field (see build()). year (int) -> dict.
                "_yearly": {},
                "_institutions": Counter(),
                "_countries": Counter(),
                "_matched": 0,
            }
            authors[aid] = rec
            ok = orcid_key(rec["orcid"])
            if ok:
                by_orcid[ok] = aid
            nm = norm_name(rec["name"])
            if nm:
                name_ids.setdefault(nm, set()).add(aid)
    # A display name is a safe fallback only when it identifies exactly one
    # author in this dataset. Split OpenAlex profiles frequently share names.
    by_name = {nm: next(iter(ids)) for nm, ids in name_ids.items() if len(ids) == 1}
    return authors, by_orcid, by_name


# ----------------------------------------------------------------------------
# Load patent counts (author_summary.csv)
# ----------------------------------------------------------------------------
def load_summary(patents_csv):
    """
    Per-person paper + patent totals from author_summary.csv.

    Fields:
      total_papers   = paper_total_count  (ALL works of the person; this is the
                       only reliable total-papers source — openalex_cache.jsonl
                       carries no works_count).
      total_patents  = patent_kth_filtered_count  (SE-inventor / KTH-applicant
                       validated set; raw patent_total_count is unreliable due to
                       EPO name collisions, values up to 65k for common names).
      quantum_patents= patent_quantum_aligned_count.
    """
    by_id, by_name = {}, {}
    if not patents_csv.exists():
        return by_id, by_name
    with open(patents_csv, encoding="utf-8") as f:
        for row in csv.DictReader(f):
            def gi(k):
                try:
                    return int(float(row.get(k) or 0))
                except ValueError:
                    return 0
            rec = {
                "total_papers": gi("paper_total_count"),
                "summary_quantum_papers": gi("paper_quantum_count"),
                "total_patents": gi("patent_kth_filtered_count"),
                "quantum_patents": gi("patent_quantum_aligned_count"),
                "patent_total_raw": gi("patent_total_count"),
            }
            aid = (row.get("openalex_id") or "").strip()
            if aid:
                by_id[aid] = rec
            nm = norm_name(row.get("author_name", ""))
            if nm:
                by_name.setdefault(nm, rec)
    return by_id, by_name


# ----------------------------------------------------------------------------
# Scan papers: build the verification queue AND attribute papers to authors.
# ----------------------------------------------------------------------------
def scan_papers(authors, by_orcid, by_name, papers_jsonl):
    """
    Single pass over the corpus. Returns:
      papers   : [{id, title, abstract, label, is_quantum_label}]  (for the LLM)
      n_papers : total papers scanned

    Side effect: fills each author's accumulators (citations, abstracts,
    institutions, countries, _quantum_paper_ids, _matched). Quantum-paper
    attribution is by paper id so the LLM verdict can be applied AFTER the pass.
    """
    papers = []
    n_papers = 0
    with open(papers_jsonl, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            n_papers += 1
            pid = rec.get("id")
            cites = rec.get("cited_by_count")
            pub_year = rec.get("publication_year")
            try:
                pub_year = int(pub_year) if pub_year is not None else None
            except (TypeError, ValueError):
                pub_year = None
            title = (rec.get("display_name") or "").strip()
            abstract = reconstruct_abstract(rec.get("abstract_inverted_index"))
            label = rec.get("classification_label")
            is_quantum_label = label in QUANTUM_LABELS
            # Y-axis text: title + abstract (both). Title alone still carries
            # signal when the abstract is missing.
            doc = (title + ". " + abstract).strip(". ").strip() if (title or abstract) else ""

            papers.append({
                "id": pid,
                "title": title,
                "abstract": abstract,
                "label": label,
                "is_quantum_label": is_quantum_label,
            })

            for ship in rec.get("authorships", []):
                au = ship.get("author", {}) or {}
                raw_aid = (au.get("id") or "").strip()
                aid = raw_aid if raw_aid in authors else None
                if aid is None:
                    ok = orcid_key(au.get("orcid", ""))
                    if ok and ok in by_orcid:
                        aid = by_orcid[ok]
                if aid is None:
                    nm = norm_name(au.get("display_name", ""))
                    if nm in by_name:
                        aid = by_name[nm]
                if aid is None:
                    continue
                a = authors[aid]
                a["_matched"] += 1
                # overall paper span: track min/max year over ALL attributed
                # papers, regardless of classification_label or LLM verdict.
                if pub_year is not None and 1500 <= pub_year <= CURRENT_YEAR + 1:
                    if a["_min_pub_year"] is None or pub_year < a["_min_pub_year"]:
                        a["_min_pub_year"] = pub_year
                    if a["_max_pub_year"] is None or pub_year > a["_max_pub_year"]:
                        a["_max_pub_year"] = pub_year
                if is_quantum_label and pid:
                    a["_quantum_paper_ids"].append(pid)
                    if pub_year is not None and 1500 <= pub_year <= CURRENT_YEAR + 1:
                        yr = a["_yearly"].setdefault(pub_year, {"papers": 0, "citations": 0})
                        yr["papers"] += 1
                        if cites is not None:
                            yr["citations"] += cites
                if cites is not None:
                    a["_citations"].append(cites)
                if doc:
                    a["_abstracts"].append(doc)
                for inst in ship.get("institutions", []) or []:
                    dn = inst.get("display_name")
                    if dn:
                        a["_institutions"][dn] += 1
                    cc = inst.get("country_code")
                    if cc:
                        a["_countries"][cc] += 1
    return papers, n_papers


# ----------------------------------------------------------------------------
# Builder/Thinker — heuristic
# ----------------------------------------------------------------------------
def builder_thinker_heuristic(author) -> float:
    """Applied(+1) vs theoretical(-1) from text the person is associated with."""
    text_parts = list(author["_abstracts"])
    text_parts.extend(author["top_topics"])
    text_parts.extend(author["top_keywords"])
    text_parts.extend(author["example_titles"])
    blob = " ".join(text_parts).lower()
    if not blob:
        return 0.0
    tokens = _word_re.findall(blob)
    applied = sum(1 for t in tokens if t in APPLIED_TERMS)
    theory = sum(1 for t in tokens if t in THEORY_TERMS)
    if applied + theory == 0:
        return 0.0
    return clamp((applied - theory) / (applied + theory))


# ----------------------------------------------------------------------------
# Builder/Thinker — SPECTER2 (ported from get_paper_distance_v3.py), GPU first
# ----------------------------------------------------------------------------
def builder_thinker_specter(authors_with_papers):
    """
    Returns {author_id: score in [-1,1]}.
    Embeds every distinct abstract once on GPU (cuda) when available, embeds the
    two axis prompts, scores each abstract by cos(builder) - cos(thinker), then
    averages per author.
    """
    import numpy as np
    import torch
    from transformers import AutoTokenizer
    from adapters import AutoAdapterModel

    cuda = torch.cuda.is_available()
    device = torch.device("cuda" if cuda else "cpu")
    if cuda:
        print(f"[specter2] device=cuda ({torch.cuda.get_device_name(0)})",
              file=sys.stderr)
    else:
        print("[specter2] device=cpu (CUDA not available — slower)",
              file=sys.stderr)
    tok = AutoTokenizer.from_pretrained("allenai/specter2_base")
    model = AutoAdapterModel.from_pretrained("allenai/specter2_base")
    model.load_adapter("allenai/specter2", source="hf",
                       load_as="proximity", set_active=True)
    model.to(device).eval()

    def embed(batch):
        inp = tok(batch, padding=True, truncation=True, return_tensors="pt",
                  return_token_type_ids=False, max_length=512)
        inp = {k: v.to(device) for k, v in inp.items()}
        with torch.no_grad():
            out = model(**inp)
        return out.last_hidden_state[:, 0, :].cpu().numpy()

    # dedup abstracts
    uniq, index = [], {}
    for a in authors_with_papers:
        for ab in a["_abstracts"]:
            if ab not in index:
                index[ab] = len(uniq)
                uniq.append(ab)
    print(f"[specter2] embedding {len(uniq)} unique abstracts", file=sys.stderr)

    batch_sz = 32 if cuda else 16
    embs = []
    for i in range(0, len(uniq), batch_sz):
        embs.append(embed(uniq[i:i + batch_sz]))
        print(f"[specter2] {min(i + batch_sz, len(uniq))}/{len(uniq)}", file=sys.stderr)
    embs = np.vstack(embs) if embs else np.zeros((0, 768))

    axis = embed([SPECTER_AXIS_LABELS["builder"], SPECTER_AXIS_LABELS["thinker"]])
    embs_n = embs / (np.linalg.norm(embs, axis=1, keepdims=True) + 1e-9)
    axis_n = axis / (np.linalg.norm(axis, axis=1, keepdims=True) + 1e-9)
    sims = embs_n @ axis_n.T               # (n, 2): [:,0]=builder [:,1]=thinker
    per_abstract = sims[:, 0] - sims[:, 1] if len(embs) else []

    scores = {}
    for a in authors_with_papers:
        vals = [per_abstract[index[ab]] for ab in a["_abstracts"]]
        if len(vals):
            # scale raw cosine-diff into a usable [-1,1] spread
            scores[a["id"]] = clamp(float(np.mean(vals)) * 6.0)
    return scores


# ----------------------------------------------------------------------------
# Assemble final records
# ----------------------------------------------------------------------------
def build(method: str, min_works: int, confidence_k: float, use_llm: bool,
          authors_csv: Path, papers_jsonl: Path, patents_csv: Path):
    authors, by_orcid, by_name = load_authors(authors_csv)
    papers, n_papers = scan_papers(authors, by_orcid, by_name, papers_jsonl)
    sum_by_id, sum_by_name = load_summary(patents_csv)
    print(f"[build] {len(authors)} authors, {n_papers} papers scanned, "
          f"{len(sum_by_id)} with summary records", file=sys.stderr)

    # ---- OpenRouter LLM verification pass (resumable, cached) ----
    verdicts = {}
    llm_model_used = None
    n_disagree = 0
    n_label_quantum = sum(1 for p in papers if p["is_quantum_label"])
    n_llm_agree = 0
    if use_llm:
        verdicts = verify_papers_llm(papers)
        for p in papers:
            if not p["is_quantum_label"]:
                continue
            v = verdicts.get(p["id"])
            if v and v.get("ok"):
                if v.get("is_quantum"):
                    n_llm_agree += 1
                else:
                    n_disagree += 1
        for v in verdicts.values():
            if v.get("ok") and v.get("model"):
                llm_model_used = v["model"]
                break

    def llm_says_quantum(pid: str) -> bool | None:
        """True/False per LLM, or None if no usable verdict."""
        v = verdicts.get(pid)
        if v and v.get("ok"):
            return bool(v.get("is_quantum"))
        return None

    # ---- SPECTER2 Y scores (only when needed) ----
    specter_scores = {}
    if method in ("specter2", "llm"):
        # SPECTER2 is the PRIMARY score for --method specter2 and the FALLBACK
        # for --method llm (people with no usable LLM applied/theoretical
        # verdicts). For pure --method heuristic we skip the embeddings.
        with_papers = [a for a in authors.values() if a["_abstracts"]]
        specter_scores = builder_thinker_specter(with_papers)

    n_missing_summary = 0
    n_quantum_gt_total = 0
    n_y_from_llm = 0
    n_y_from_fallback = 0
    people = []
    for a in authors.values():
        if a["quantum_work_count"] < min_works:
            continue
        hidx = h_index(a["_citations"])

        # ---- quantum_papers: SPRINT2 label AND (LLM agrees OR no LLM verdict) ----
        # Combination rule: a paper counts as quantum iff the SPRINT 2 classifier
        # labelled it quantum (core_quantum|likely_quantum) AND the LLM did NOT
        # explicitly reject it. When the LLM has no usable verdict we KEEP the
        # SPRINT 2 label (do not penalise corpus coverage gaps). When use_llm is
        # off, this reduces to the original SPRINT 2 count.
        q_ids = a["_quantum_paper_ids"]
        if use_llm:
            confirmed = []
            for pid in q_ids:
                verdict = llm_says_quantum(pid)
                if verdict is False:
                    continue          # LLM rejected -> drop
                confirmed.append(pid)  # agreed or no-verdict -> keep
            quantum_papers = len(confirmed)
            quantum_paper_ids = confirmed
        else:
            quantum_papers = len(q_ids)
            quantum_paper_ids = q_ids
        # fall back to the author CSV count if nothing was attributed at all
        if quantum_papers == 0:
            quantum_papers = a["quantum_work_count"]

        # ---- Y axis (builder/thinker) ----
        y_llm = None
        if use_llm and quantum_paper_ids:
            ats = []
            for pid in quantum_paper_ids:
                v = verdicts.get(pid)
                if v and v.get("ok") and v.get("applied_theoretical") is not None:
                    ats.append(float(v["applied_theoretical"]))
            if ats:
                y_llm = sum(ats) / len(ats)

        if method == "heuristic":
            y_raw = builder_thinker_heuristic(a)
        elif method == "specter2":
            y_raw = specter_scores.get(a["id"], builder_thinker_heuristic(a))
        else:  # method == "llm"
            if y_llm is not None:
                y_raw = clamp(y_llm)
                n_y_from_llm += 1
            elif a["id"] in specter_scores:
                y_raw = specter_scores[a["id"]]   # SPECTER2 fallback
                n_y_from_fallback += 1
            else:
                y_raw = builder_thinker_heuristic(a)
                n_y_from_fallback += 1

        summ = sum_by_id.get(a["id"]) or sum_by_name.get(norm_name(a["name"])) or {}
        if not summ:
            n_missing_summary += 1

        # total_papers: ALL works of the person from author_summary.csv. When the
        # author is missing from that file, fall back to their quantum count
        # (yields share=1; reported as an anomaly).
        total_papers = summ.get("total_papers", 0) or quantum_papers
        # OpenAlex splits some authors across several ids, so author_summary's
        # paper_total_count (per-id) can be SMALLER than the quantum papers we
        # attributed to the person by name in SPRINT 2 (e.g. total=1, quantum=90).
        # Floor total_papers at quantum_papers: the person provably has at least
        # that many works, share becomes 1.0 (Resident), and the tooltip stops
        # showing the nonsensical "1 total / 90 quantum".
        if quantum_papers > total_papers:
            n_quantum_gt_total += 1
            total_papers = quantum_papers

        x_raw, share = quantum_share_score(quantum_papers, total_papers)

        # confidence shrinkage toward centre — few quantum papers => weak evidence
        n = quantum_papers
        conf = n / (n + confidence_k)
        x = x_raw * conf
        y = y_raw * conf

        total_patents = summ.get("total_patents", 0)
        quantum_patents = summ.get("quantum_patents", 0)
        inst = a["_institutions"].most_common(1)
        country = a["_countries"].most_common(1)
        people.append({
            "id": a["id"],
            "name": a["name"],
            "orcid": a["orcid"],
            "tier": a["tier"],
            "institution": inst[0][0] if inst else "Unknown",
            "country": country[0][0] if country else "??",
            "topic": a["top_topics"][0] if a["top_topics"] else "Unknown",
            "topics": a["top_topics"],
            "keywords": a["top_keywords"],
            "x": round(x, 4),
            "y": round(y, 4),
            "x_raw": round(x_raw, 4),
            "y_raw": round(y_raw, 4),
            "quantum_share": round(share, 4),
            "confidence": round(conf, 3),
            "h_index": hidx,
            "total_papers": total_papers,
            "quantum_papers": quantum_papers,
            "total_patents": total_patents,
            "quantum_patents": quantum_patents,
            "matched_papers": a["_matched"],
            # first/last_year = span of the person's papers IN GENERAL (min/max
            # publication_year over ALL attributed papers in the SPRINT 2 corpus,
            # any label or LLM verdict). NOTE: the corpus is quantum-candidate-only,
            # so this underestimates a full career span. Falls back to the CSV
            # quantum-only span (first/last_quantum_year) when no paper carried a
            # usable year.
            "first_year": a["_min_pub_year"] if a["_min_pub_year"] is not None
                          else a["csv_first_quantum_year"],
            "last_year": a["_max_pub_year"] if a["_max_pub_year"] is not None
                         else a["csv_last_quantum_year"],
            "example_titles": a["example_titles"],
            # NEW, additive field: per-year {papers, citations} over quantum
            # papers only. Old consumers ignore unknown keys, so this is
            # backward compatible; new consumers must treat it as optional
            # since data files built before this change won't have it.
            "yearly": {str(y): v for y, v in sorted(a["_yearly"].items())},
        })

    disagree_rate = (n_disagree / n_label_quantum) if n_label_quantum else 0.0
    print(f"[build] {n_missing_summary} people missing from author_summary.csv "
          f"(total_papers fell back to quantum count); "
          f"{n_quantum_gt_total} with quantum_papers > total_papers",
          file=sys.stderr)
    if use_llm:
        print(f"[build] LLM: {n_llm_agree} agree / {n_disagree} disagree "
              f"of {n_label_quantum} SPRINT2-quantum papers "
              f"(disagreement {disagree_rate:.1%}); "
              f"Y from LLM={n_y_from_llm}, fallback={n_y_from_fallback}",
              file=sys.stderr)

    people.sort(key=lambda p: p["quantum_papers"], reverse=True)

    topics = sorted({p["topic"] for p in people if p["topic"] != "Unknown"})
    institutions = sorted({p["institution"] for p in people if p["institution"] != "Unknown"})

    return {
        "meta": {
            "generated_utc": datetime.now(timezone.utc).isoformat(),
            "method": method,
            "llm_verification": use_llm,
            "llm_model": llm_model_used or (LLM_MODEL if use_llm else None),
            "llm_disagreement_rate": round(disagree_rate, 4) if use_llm else None,
            "current_year": CURRENT_YEAR,
            "min_works": min_works,
            "confidence_k": confidence_k,
            "n_people": len(people),
            "n_papers_scanned": n_papers,
            "axes": AXES,
            "size_modes": {
                "h_index": "h-index from cited_by_count of matched papers",
                "total_patents": "patents (KTH/SE-validated, patent_kth_filtered_count)",
            },
            "note": ("X = quantum share of papers (LLM-verified quantum count / "
                     "total works). Y (builder/thinker) is %s-derived"
                     "%s." % (method,
                              " with LLM applied/theoretical verdicts" if (method == "llm" and use_llm) else "")),
        },
        # placeholders for the top TOPIC@INSTITUTION and bottom aggregation UI
        "filters": {
            "topics": topics,
            "institutions": institutions,
            "aggregations": ["person", "institution", "country"],
        },
        "people": people,
    }


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--authors-csv", type=Path, required=True,
                    help="Path to authors_v2.csv for this university")
    ap.add_argument("--papers-jsonl", type=Path, required=True,
                    help="Path to quantum_classified_works_specter_reviewed.jsonl for this university")
    ap.add_argument("--patents-csv", type=Path, required=True,
                    help="Path to author_summary.csv for this university")
    ap.add_argument("--method", choices=["llm", "specter2", "heuristic"],
                    default="llm", help="builder/thinker scoring method")
    ap.add_argument("--no-llm", dest="use_llm", action="store_false",
                    help="skip the OpenRouter verification pass entirely")
    ap.add_argument("--min-works", type=int, default=4,
                    help="drop authors with fewer than N quantum works "
                         "(below ~4 there isn't enough signal to place them)")
    ap.add_argument("--confidence-k", type=float, default=DEFAULT_CONFIDENCE_K,
                    help="shrinkage constant; axis pos *= n/(n+k). Higher = "
                         "pull low-output authors harder toward the centre.")
    ap.add_argument("--out", type=Path, default=DEFAULT_OUT,
                    help="output JSON path")
    ap.set_defaults(use_llm=True)
    args = ap.parse_args()

    data = build(args.method, args.min_works, args.confidence_k, args.use_llm,
                 args.authors_csv, args.papers_jsonl, args.patents_csv)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    print(f"[done] {data['meta']['n_people']} people -> {args.out}",
          file=sys.stderr)


if __name__ == "__main__":
    main()
