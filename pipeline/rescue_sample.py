#!/usr/bin/env python3
"""Rescue pass, stage R3a: draw the sample of pool works that gets labelled.

The classifier of the rescue pass needs labelled examples drawn from the pool itself, and the
pool is overwhelmingly NOT quantum research, so a purely random sample would hold very few
positives. Two strata are therefore drawn, and kept apart in the output:

  uniform   N_UNIFORM works drawn uniformly at random from the pool. Because it is a fair
            sample, it is the ONLY stratum used to estimate base rates, precision and recall.
  targeted  N_TARGETED works drawn (excluding the uniform ones) from those whose title or
            abstract contains a quantum-adjacent physics term (Josephson, condensate,
            superfluid, Majorana, ...). This lexicon only steers SAMPLING toward likely
            positives so the classifier sees enough of them; it is never part of the
            accept/reject decision. These works are used for training only.

Output: corpus/data/rescue_label_sample.jsonl  ({item, id, year, title, abstract, stratum}).

Usage:  python rescue_sample.py
"""

import argparse
import json
import random
import re
from pathlib import Path

HERE = Path(__file__).resolve().parent
DATA = HERE.parent / "corpus" / "data"
N_UNIFORM = 2000
N_TARGETED = 2000
SEED = 20261008

# Quantum-adjacent vocabulary that does not rely on the literal word "quantum" (the rule-based
# filter already looked for that). Sampling aid only; see the module docstring.
TARGET_LEXICON = re.compile(
    r"josephson|superconduct|superfluid|bose[- ]einstein|condensat|squeez|single[- ]photon|"
    r"photon(?:ic)? (?:pair|entangl|correl)|cooper pair|majorana|anyon|rydberg|cold atom|ultracold|"
    r"laser cooling|optical lattice|spin (?:chain|liquid|ice|wave)|tunnel(?:l)?ing|hamiltonian|"
    r"topological (?:insulator|phase|order)|two-level|coherent control|rabi|ion trap|trapped ion|"
    r"nitrogen[- ]vacancy|fermi gas|bosonic|fermionic|dirac|weyl|casimir|zeeman|lamb shift|"
    r"parametric down|fock|bell inequalit|schr.dinger|many-body|mesoscopic|nanowire|"
    r"vortex|vortices|magnon|polariton|exciton",
    re.IGNORECASE,
)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--pool", type=Path, default=DATA / "rescue_pool.jsonl")
    ap.add_argument("--out", type=Path, default=DATA / "rescue_label_sample.jsonl")
    ap.add_argument("--uniform", type=int, default=N_UNIFORM)
    ap.add_argument("--targeted", type=int, default=N_TARGETED)
    args = ap.parse_args()

    pool = [json.loads(line) for line in open(args.pool, encoding="utf-8")]
    rnd = random.Random(SEED)
    uniform_idx = set(rnd.sample(range(len(pool)), min(args.uniform, len(pool))))
    candidates = [
        i for i, w in enumerate(pool)
        if i not in uniform_idx and TARGET_LEXICON.search(f"{w['display_name']} {w['abstract']}")
    ]
    targeted_idx = set(rnd.sample(candidates, min(args.targeted, len(candidates))))
    print(f"pool {len(pool)} | lexicon matches outside the uniform draw: {len(candidates)} "
          f"({len(candidates) / (len(pool) - len(uniform_idx)):.1%} of the rest)")

    chosen = [(i, "uniform") for i in sorted(uniform_idx)] + [(i, "targeted") for i in sorted(targeted_idx)]
    rnd.shuffle(chosen)  # labellers must not see the stratum order
    with open(args.out, "w", encoding="utf-8") as f:
        for k, (i, stratum) in enumerate(chosen, 1):
            w = pool[i]
            f.write(json.dumps({"item": f"rs_{k:05d}", "id": w["id"], "year": w["publication_year"],
                                "title": w["display_name"], "abstract": w["abstract"][:1400],
                                "stratum": stratum}, ensure_ascii=False) + "\n")
    print(f"wrote {len(chosen)} items ({len(uniform_idx)} uniform, {len(targeted_idx)} targeted) -> {args.out}")


if __name__ == "__main__":
    main()
