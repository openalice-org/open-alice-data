# Splink dedup experiment

Probabilistic record-linkage pass over `public/data/people_viz.json` (9,352
people), looking for the same physical person split across multiple OpenAlex
author-id records — a real gap: `merge_universities.py` only dedupes on
*exact* OpenAlex id, so two ids for one person (common — OpenAlex splits an
author's works into several profiles when its own name-matching gets
confused, especially on short/abbreviated names and large multi-author
physics collaborations) sail through untouched today.

## Method

1. **`01_prepare.py`** — loads people_viz.json, normalizes names, flags exact
   ORCID collisions (two different ids sharing one ORCID — about as close to
   certain as this dataset gets), and blocks candidate pairs by normalized
   surname (9,352 people -> 18,988 candidate pairs; a full pairwise
   comparison isn't warranted for this question).
2. **`02_link.py`** — a [splink](https://moj-analytical-services.github.io/splink/)
   dedupe model over those candidate pairs: name (Jaro-Winkler), first/last
   publication year (absolute-difference bands), country/university (exact),
   institution string (Jaro-Winkler), topic/keyword overlap (array
   intersection). Trained unsupervised via EM (no labeled truth set) —
   4,289 pairs scored >= 0.1 match probability.
3. **`03_report.py`** — clusters pairs at >= 0.5 match probability
   (union-find, so a name split across 3+ ids reports as one cluster), then
   enriches the top 40 clusters with **real coauthor overlap**, scanned
   directly from each university's `quantum_classified_works_specter_reviewed.jsonl`
   (not a splink feature — genuine ground truth from the corpus). High
   coauthor Jaccard overlap on a name match is strong independent
   confirmation; near-zero overlap on a name match (see the "H. Smith"
   cluster) is a flag that it's probably just a common name, not a split
   profile.

## Output

- `out/REPORT.md` — the readable report (ORCID collisions + top 40 clusters
  with coauthor evidence).
- `out/all_clusters.csv` — all 452 clusters at match_probability >= 0.5
  (names/ids only, no coauthor enrichment).
- `out/orcid_dupes.json`, `out/predictions.pkl`, `out/people.pkl` —
  intermediate data (`.pkl` files gitignored, regenerate via the steps
  below).

## Reproduce

```bash
uv venv .venv --python 3.12
uv pip install --python .venv/bin/python "splink>=4,<5" pandas
.venv/bin/python 01_prepare.py
.venv/bin/python 02_link.py
.venv/bin/python 03_report.py
```

## Caveats

- Purely a **candidate-generation** pass — nothing here merges records or
  touches `people_viz.json`. Every cluster needs a human look before any
  merge; several examples (e.g. "H. Smith") are generic names where the
  coauthor evidence argues AGAINST merging despite the name match.
- Coauthor overlap is scanned only for the top 40 enriched clusters (not all
  452) to keep the pass fast; `all_clusters.csv` has the full candidate list
  without that enrichment.
- Blocking on surname means a duplicate caused by a genuinely different
  surname spelling (e.g. a transliteration that changes the first letter)
  would be missed — this pass only catches same-surname splits.
