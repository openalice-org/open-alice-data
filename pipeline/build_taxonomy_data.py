#!/usr/bin/env python3
"""SPRINT 8 "Taxi Driver" — build people_taxonomy.json.

Path-parameterized port for Sprint 8's multi-university corpus. Joins
Sprint 8's merged people_viz.json with the concatenation of
all three universities' classified works (see concat_works.py) to produce,
per person:
  - an OpenAlex taxonomy path (macro=field, meso=subfield, micro=topic)
  - probabilistic application-domain mapping (computing/communication/sensing/other)
plus a pruned sunburst tree over all people's paths.

Stdlib only. Run:
  "../.venv/bin/python3" "build_taxonomy_data.py"
"""

import json
import os
import pathlib
import re
import sys
from collections import Counter, defaultdict
from datetime import datetime, timezone

SPRINT11 = pathlib.Path(__file__).resolve().parent
SPRINT5_PEOPLE = str(SPRINT11 / "data" / "people_viz.json")
SPRINT2_WORKS = str(SPRINT11 / "data" / "all_universities_works.jsonl")
OUT_PATH = str(SPRINT11 / "data" / "people_taxonomy.json")

ACCEPTED_LABELS = {"core_quantum", "likely_quantum"}

# Pruning thresholds for sunburst legibility.
MIN_PEOPLE_MICRO = 2  # topic nodes with fewer people are dropped -> 2-level path
MIN_PEOPLE_MESO = 2   # subfield nodes with fewer people collapse -> 1-level path

# ---------------------------------------------------------------------------
# Application-domain lexicon (transparent, extend here).
# Stems are intentional (e.g. "quantum comput" matches computing/computation/computer).
# Matching: case-insensitive, leading word boundary, spaces/hyphens interchangeable.
# ---------------------------------------------------------------------------
DOMAIN_LEXICON = {
    "computing": [
        "qubit", "quantum comput", "quantum processor", "quantum algorithm",
        "quantum gate", "quantum circuit", "error correction", "quantum error",
        "quantum simulat", "quantum annealing", "quantum supremacy",
        "quantum advantage", "transmon", "superconducting qubit", "ion trap",
        "trapped ion", "quantum logic", "NISQ", "variational quantum",
        "quantum machine learning",
        # extensions
        "surface code", "topological quantum comput", "quantum volume",
        "QAOA", "quantum software", "quantum hardware",
    ],
    "communication": [
        "quantum key distribution", "QKD", "quantum communication",
        "quantum network", "quantum internet", "quantum repeater",
        "quantum teleport", "entanglement distribution", "quantum cryptograph",
        "secure communication", "quantum channel", "single-photon source",
        "photon pair", "quantum memory", "satellite quantum",
        # extensions
        "BB84", "entanglement swapping", "quantum random number",
        "device-independent", "heralded single photon",
    ],
    "sensing": [
        "quantum sens", "quantum metrolog", "magnetometer", "quantum imaging",
        "atomic clock", "quantum radar", "gravimeter", "NV center",
        "nitrogen-vacancy", "quantum-enhanced measurement", "interferometr",
        "squeezed light", "quantum illumination", "optomechanic",
        "precision measurement",
        # extensions
        "optical clock", "atom interferometer", "squeezed state",
        "quantum-limited", "magnetometry",
    ],
}

REAL_DOMAINS = ("computing", "communication", "sensing")
ALL_DOMAINS = REAL_DOMAINS + ("other",)

# Weights per text source when scoring a work.
W_TITLE = 3.0
W_ABSTRACT = 1.0
W_KW_TOPICS = 2.0


def compile_lexicon(lexicon):
    """One compiled alternation regex per category.

    Each term: leading word boundary; spaces/hyphens match any whitespace/hyphen
    run; no trailing boundary so stems ("quantum comput") match inflections.
    """
    compiled = {}
    for cat, terms in lexicon.items():
        parts = []
        for term in terms:
            words = [re.escape(w) for w in re.split(r"[\s\-]+", term.lower()) if w]
            parts.append(r"\b" + r"[\s\-]+".join(words))
        compiled[cat] = re.compile("|".join(parts), re.IGNORECASE)
    return compiled


LEXICON_RE = compile_lexicon(DOMAIN_LEXICON)


def reconstruct_abstract(aii):
    """Rebuild plain text from an OpenAlex abstract_inverted_index."""
    if not isinstance(aii, dict) or not aii:
        return ""
    pos_word = []
    for word, positions in aii.items():
        if not isinstance(positions, list):
            continue
        for p in positions:
            if isinstance(p, int):
                pos_word.append((p, word))
    pos_word.sort()
    return " ".join(w for _, w in pos_word)


def count_hits(pattern, text):
    if not text:
        return 0
    return len(pattern.findall(text))


def classify_work_domains(work):
    """Return (distribution dict over ALL_DOMAINS, has_signal bool)."""
    title = work.get("display_name") or ""
    abstract = reconstruct_abstract(work.get("abstract_inverted_index"))

    kw_topic_bits = []
    for kw in work.get("keywords") or []:
        if isinstance(kw, dict) and kw.get("display_name"):
            kw_topic_bits.append(kw["display_name"])
    for t in work.get("topics") or []:
        if isinstance(t, dict) and t.get("display_name"):
            kw_topic_bits.append(t["display_name"])
    pt = work.get("primary_topic")
    if isinstance(pt, dict) and pt.get("display_name"):
        kw_topic_bits.append(pt["display_name"])
    kw_topic_text = " ; ".join(kw_topic_bits)

    scores = {}
    for cat in REAL_DOMAINS:
        pat = LEXICON_RE[cat]
        scores[cat] = (
            W_TITLE * count_hits(pat, title)
            + W_ABSTRACT * count_hits(pat, abstract)
            + W_KW_TOPICS * count_hits(pat, kw_topic_text)
        )

    total = sum(scores.values())
    if total <= 0:
        return ({"computing": 0.0, "communication": 0.0, "sensing": 0.0,
                 "other": 1.0}, False)
    dist = {cat: scores[cat] / total for cat in REAL_DOMAINS}
    dist["other"] = 0.0
    return dist, True


def extract_topic_entries(work):
    """Yield (field, subfield, topic, weight) tuples, defensively."""
    topics = work.get("topics") or []
    if not topics:
        pt = work.get("primary_topic")
        topics = [pt] if isinstance(pt, dict) else []
        default_weight = 1.0
    else:
        default_weight = None

    for t in topics:
        if not isinstance(t, dict):
            continue
        micro = t.get("display_name")
        sub = t.get("subfield") or {}
        fld = t.get("field") or {}
        meso = sub.get("display_name") if isinstance(sub, dict) else None
        macro = fld.get("display_name") if isinstance(fld, dict) else None
        score = t.get("score")
        if default_weight is not None:
            weight = default_weight
        else:
            weight = float(score) if isinstance(score, (int, float)) and score > 0 else 1.0
        if macro:  # need at least a macro to be useful
            yield macro, meso, micro, weight


def greedy_path(macro_w, meso_w, micro_w):
    """Greedy argmax: macro -> meso within macro -> micro within meso.

    Deterministic tie-break by name. Returns list of 1-3 levels.
    """
    if not macro_w:
        return []
    macro = min(macro_w, key=lambda k: (-macro_w[k], k))
    mesos = {m: w for (ma, m), w in meso_w.items() if ma == macro}
    if not mesos:
        return [macro]
    meso = min(mesos, key=lambda k: (-mesos[k], k))
    micros = {mi: w for (ma, me, mi), w in micro_w.items()
              if ma == macro and me == meso}
    if not micros:
        return [macro, meso]
    micro = min(micros, key=lambda k: (-micros[k], k))
    return [macro, meso, micro]


def build_tax_levels(path, macro_w, meso_w, micro_w):
    """Per-level dominance shares for a (pruned) tax_path.

    share = node weight / parent weight:
      macro: macro weight / person's total weight (sum over all macros)
      meso : meso weight  / chosen macro weight
      micro: micro weight / chosen meso weight
    Entries correspond 1:1 with `path`.
    """
    levels = []
    if not path:
        return levels
    total = sum(macro_w.values())
    macro = path[0]
    mw = macro_w.get(macro, 0.0)
    levels.append({
        "name": macro, "level": "macro",
        "share": round(mw / total, 4) if total > 0 else 0.0,
    })
    if len(path) >= 2:
        meso = path[1]
        sw = meso_w.get((macro, meso), 0.0)
        levels.append({
            "name": meso, "level": "meso",
            "share": round(sw / mw, 4) if mw > 0 else 0.0,
        })
        if len(path) == 3:
            micro = path[2]
            tw = micro_w.get((macro, meso, micro), 0.0)
            levels.append({
                "name": micro, "level": "micro",
                "share": round(tw / sw, 4) if sw > 0 else 0.0,
            })
    return levels


def main():
    # ------------------------------------------------------------------ load
    with open(SPRINT5_PEOPLE, encoding="utf-8") as f:
        sprint5 = json.load(f)
    people = sprint5["people"]
    person_ids = {p["id"] for p in people}
    print(f"Loaded {len(people)} people from Sprint 5")

    # Per-person accumulators
    tax_macro = defaultdict(lambda: defaultdict(float))   # pid -> macro -> w
    tax_meso = defaultdict(lambda: defaultdict(float))    # pid -> (macro,meso) -> w
    tax_micro = defaultdict(lambda: defaultdict(float))   # pid -> (macro,meso,micro) -> w
    work_count = Counter()                                # pid -> matched works
    domain_dists = defaultdict(list)                      # pid -> [work dist]
    domain_signal_n = Counter()                           # pid -> works w/ signal
    matched_work_ids = set()
    n_scanned = 0
    n_quantum_label = 0

    # ------------------------------------------------------------ scan works
    with open(SPRINT2_WORKS, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            n_scanned += 1
            try:
                work = json.loads(line)
            except json.JSONDecodeError:
                continue
            if work.get("classification_label") not in ACCEPTED_LABELS:
                continue
            n_quantum_label += 1

            # match authors
            matched_pids = set()
            for auth in work.get("authorships") or []:
                if not isinstance(auth, dict):
                    continue
                a = auth.get("author") or {}
                aid = a.get("id") if isinstance(a, dict) else None
                if aid in person_ids:
                    matched_pids.add(aid)
            if not matched_pids:
                continue
            matched_work_ids.add(work.get("id"))

            entries = list(extract_topic_entries(work))
            dist, has_signal = classify_work_domains(work)

            for pid in matched_pids:
                work_count[pid] += 1
                domain_dists[pid].append(dist)
                if has_signal:
                    domain_signal_n[pid] += 1
                for macro, meso, micro, w in entries:
                    tax_macro[pid][macro] += w
                    if meso:
                        tax_meso[pid][(macro, meso)] += w
                        if micro:
                            tax_micro[pid][(macro, meso, micro)] += w

    n_matched_people = sum(1 for p in people if work_count[p["id"]] > 0)
    print(f"Scanned {n_scanned} works; {n_quantum_label} core/likely quantum; "
          f"{len(matched_work_ids)} distinct works matched to people")
    print(f"People with >=1 matched work: {n_matched_people}/{len(people)}")

    # ------------------------------------------------------- raw paths
    raw_paths = {}
    for p in people:
        pid = p["id"]
        if work_count[pid] == 0:
            raw_paths[pid] = []
        else:
            raw_paths[pid] = greedy_path(tax_macro[pid], tax_meso[pid],
                                         tax_micro[pid])

    # ----------------------------------------------------------- pruning
    # meso occupancy = people whose path passes through (macro, meso)
    meso_people = Counter(tuple(pa[:2]) for pa in raw_paths.values() if len(pa) >= 2)
    # micro occupancy = people with that exact 3-level path
    micro_people = Counter(tuple(pa) for pa in raw_paths.values() if len(pa) == 3)

    final_paths = {}
    for pid, pa in raw_paths.items():
        if len(pa) >= 2 and meso_people[tuple(pa[:2])] < MIN_PEOPLE_MESO:
            pa = pa[:1]  # collapse meso into macro
        elif len(pa) == 3 and micro_people[tuple(pa)] < MIN_PEOPLE_MICRO:
            pa = pa[:2]  # drop micro
        final_paths[pid] = pa

    # ----------------------------------------------------------- sunburst
    def make_node(name, level):
        return {"name": name, "level": level, "count": 0, "children": []}

    root = {"name": "Quantum", "children": []}
    macro_nodes = {}
    for pid in sorted(final_paths, key=lambda k: final_paths[k]):
        pa = final_paths[pid]
        if not pa:
            continue
        macro = pa[0]
        mnode = macro_nodes.get(macro)
        if mnode is None:
            mnode = make_node(macro, "macro")
            macro_nodes[macro] = mnode
            root["children"].append(mnode)
        mnode["count"] += 1
        if len(pa) >= 2:
            snode = next((c for c in mnode["children"] if c["name"] == pa[1]), None)
            if snode is None:
                snode = make_node(pa[1], "meso")
                mnode["children"].append(snode)
            snode["count"] += 1
            if len(pa) == 3:
                tnode = next((c for c in snode["children"] if c["name"] == pa[2]),
                             None)
                if tnode is None:
                    tnode = make_node(pa[2], "micro")
                    snode["children"].append(tnode)
                tnode["count"] += 1

    # sort children by count desc, name asc; drop empty children lists
    def tidy(node):
        kids = node.get("children", [])
        kids.sort(key=lambda c: (-c["count"], c["name"]))
        for k in kids:
            tidy(k)
        if not kids:
            node.pop("children", None)

    root["children"].sort(key=lambda c: (-c["count"], c["name"]))
    for c in root["children"]:
        tidy(c)

    # ----------------------------------------------------------- per-person
    out_people = []
    for p in people:
        pid = p["id"]
        rec = dict(p)  # pass through all Sprint 5 fields unchanged
        rec["tax_path"] = final_paths[pid]
        rec["tax_levels"] = build_tax_levels(
            final_paths[pid], tax_macro[pid], tax_meso[pid], tax_micro[pid])
        rec["tax_evidence"] = work_count[pid]

        dists = domain_dists[pid]
        if dists:
            avg = {d: sum(x[d] for x in dists) / len(dists) for d in ALL_DOMAINS}
            tot = sum(avg.values()) or 1.0
            probs = {d: avg[d] / tot for d in ALL_DOMAINS}
        else:
            probs = {"computing": 0.0, "communication": 0.0, "sensing": 0.0,
                     "other": 1.0}

        n_sig = domain_signal_n[pid]
        p_other = probs["other"]
        real_mass = sum(probs[d] for d in REAL_DOMAINS)

        if work_count[pid] == 0 or p_other >= 0.85 or real_mass <= 0:
            top = "unknown"
            conf = 0.0
        else:
            ranked = sorted(REAL_DOMAINS, key=lambda d: (-probs[d], d))
            top = ranked[0]
            q = [probs[d] / real_mass for d in ranked]
            conf = (q[0] - q[1]) * (n_sig / (n_sig + 3.0)) * (1.0 - p_other)
            conf = max(0.0, min(1.0, conf))

        rec["domain_probs"] = {d: round(probs[d], 4) for d in ALL_DOMAINS}
        rec["domain_top"] = top
        rec["domain_confidence"] = round(conf, 4)
        rec["domain_evidence"] = n_sig
        out_people.append(rec)

    # ----------------------------------------------------------- output
    out = {
        "meta": {
            "generated_utc": datetime.now(timezone.utc).isoformat(),
            "source_sprint5": "people_viz.json",
            "n_people": len(out_people),
            "n_papers_quantum": len(matched_work_ids),
            "n_papers_scanned": n_scanned,
            "rings": ["field", "subfield", "topic"],
            "domains": {
                "computing": {"label": "Computing", "color": "#c0392b"},
                "communication": {"label": "Communication", "color": "#1e8a4c"},
                "sensing": {"label": "Sensing", "color": "#2563c4"},
                "unknown": {"label": "Undecided", "color": "#9a9a92"},
            },
            "notes": ("macro/meso/micro = OpenAlex field/subfield/topic; "
                      "domain mapping lexicon-based, probabilistic"),
        },
        "taxonomy": root,
        "people": out_people,
    }

    os.makedirs(os.path.dirname(OUT_PATH), exist_ok=True)
    with open(OUT_PATH, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=1)
    size = os.path.getsize(OUT_PATH)
    print(f"\nWrote {OUT_PATH} ({size/1024:.1f} KiB)")

    # ----------------------------------------------------------- reporting
    print("\n=== People per macro sector ===")
    macro_counter = Counter(pa[0] for pa in final_paths.values() if pa)
    unmapped = sum(1 for pa in final_paths.values() if not pa)
    for name, cnt in macro_counter.most_common():
        print(f"  {name:<45s} {cnt:4d}")
    print(f"  {'(Unmapped)':<45s} {unmapped:4d}")

    print("\n=== Domain distribution ===")
    top_counter = Counter(r["domain_top"] for r in out_people)
    for d in ("computing", "communication", "sensing", "unknown"):
        print(f"  {d:<14s} {top_counter.get(d, 0):4d}")
    confs = [r["domain_confidence"] for r in out_people
             if r["domain_top"] != "unknown"]
    if confs:
        print(f"  mean confidence (non-unknown): {sum(confs)/len(confs):.4f}")

    n_macro = len(root["children"])
    n_meso = sum(len(c.get("children", [])) for c in root["children"])
    n_micro = sum(len(s.get("children", []))
                  for c in root["children"] for s in c.get("children", []))
    print(f"\n=== Pruned tree: {n_macro} macro / {n_meso} meso / "
          f"{n_micro} micro nodes ===")

    print("\n=== Sanity check: top-3 by quantum_papers ===")
    for r in sorted(out_people, key=lambda r: -(r.get("quantum_papers") or 0))[:3]:
        print(f"  {r['name']} (quantum_papers={r.get('quantum_papers')}, "
              f"tax_evidence={r['tax_evidence']})")
        print(f"    tax_path: {r['tax_path']}")
        print(f"    tax_levels: {r['tax_levels']}")
        print(f"    domain_probs: {r['domain_probs']} -> {r['domain_top']} "
              f"(conf {r['domain_confidence']}, evid {r['domain_evidence']})")

    return 0


if __name__ == "__main__":
    sys.exit(main())
