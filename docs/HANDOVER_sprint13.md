> Historical document from an earlier build; superseded by README.md and docs/METHODOLOGY.md. Paths and defaults described here are out of date.

# Sprint 13 handover — Sweden → full Nordics

## What changed vs Sprint 12

| Metric | Sprint 12 | Sprint 13 |
| --- | ---: | ---: |
| Institutions | 9 (Sweden only) | **45** (SE 9, NO 11, IS 5, DK 8, FI 12) |
| People | 2,499 | **9,352** |
| Classified works | 22,297 | **87,307** |
| Taxonomy tree | 10 macro / 29 meso / 103 micro | **16 macro / 44 meso / 204 micro** |
| Domain classifier | SPECTER2, hardcoded anchors | **BAAI/bge-large-en-v1.5, anchors from `config/quantum_area_anchors.json`** |
| Gender | Ollama-inferred (male/female/unknown) | **null for everyone — deliberately skipped, see below** |
| Patents | fetched per university | **skipped for everyone (`--skip-patents`) — EPO deferred, see below** |
| Global h-index | per Sprint 12 run | refreshed, 9,352/9,352 enriched |

## Why: new entry point

KTH.Tools.QuantumAuthors replaced our own fetch+classify stage. It outputs one file,
`KTH.Tools.QuantumAuthors/data/sweden_quantum_works.jsonl` (~95k rows, ~1.4GB) — every
row is already an accepted quantum work (`quantum_included: true`), country-scoped to
SE|NO|DK|IS|FI, not tied to a specific institution list. `fetch_openalex_corpus.py` and
`specter_review_corpus.py` are retired (files still on disk, no longer run).

`split_nordic_corpus.py` is the new bridge: it streams that file once and, for every
institution in `universities.json`, writes matching works to
`data/<uni>/quantum_classified_works_specter_reviewed.jsonl` — same filename the old
stage used, so `extract_authors.py` and everything downstream needed zero changes.
It also stamps `classification_label: "core_quantum"` on every row (the new corpus has
no likely/review/reject tiers — `quantum_included` already is the accept decision) so
`ACCEPTED_LABELS` checks across the pipeline keep working.

## Institution list — curated by hand, not verified against EPO

`universities.json` went from 9 to 45 entries. Each new one has a real OpenAlex
institution ID (given by the user, not fabricated) but the `applicant_regex` field
(used for EPO patent-applicant matching) is my best-effort guess from standard
English/Norwegian/Danish/Finnish/Icelandic naming conventions — **not spot-checked
against real EPO applicant strings**. Worth a sanity check whenever patents actually
get fetched for these 36. One institution (University of Vaasa) has two OpenAlex IDs
(a merge/rename in OpenAlex); handled via a new optional
`openalex_institution_id_aliases` array field, additive, doesn't affect the other 44.

## Domain classifier rebuilt on bge-large, not SPECTER2

`build_domain_specter.py` now reads its anchor sentences from
`config/quantum_area_anchors.json` (vendored from KTH.Tools.QuantumAuthors' own
`sweden_quantum_area_anchors.json`) and embeds with `BAAI/bge-large-en-v1.5` instead of
SPECTER2. Same domain keys (`computing`/`communication`/`sensing`/`other`) so
`build_domain_shares.py` and `index.html` needed no changes. tau calibration still
grid-searches against the lexicon weak-truth (now picked 0.03, was 0.01 pre-expansion)
rather than trusting the config's own `softmax_temperature: 0.0365` blindly — it's in
the grid and can win, but didn't here.

The full 95,174-work corpus is pre-embedded into
`caches/work_embeddings_bge.npy` (391MB) — every rebuild since has been a 100% cache
hit, seconds instead of ~45 min on a 6GB-VRAM laptop GPU (fp16, batch 48).

## Deliberately skipped this sprint

- **EPO patents** — too slow to run today by the user's own call. Every university ran
  `fetch_epo_patents.py --skip-patents` (OpenAlex paper totals only, no patent search),
  so `patent_quantum_aligned_count` etc. are 0 for all 9,352 people, not missing/broken.
  Re-run without `--skip-patents` per university whenever there's time for it — this
  will also be the first real test of the 36 new `applicant_regex` patterns above.
- **Gender** — `build_gender_ollama.py` didn't run. `null_gender.py` sets
  `gender: null` for everyone instead of inferring it, kept schema-compatible
  (`index.html`'s `genderColor[d.gender] || genderColor.unknown` already treats any
  non-key value, including null, as neutral — verified, no frontend change needed).
  Swap `null_gender.py` for `build_gender_ollama.py` in the rebuild order whenever
  gender inference is wanted again.

## New files this sprint

- `split_nordic_corpus.py` — corpus → per-university split (see above)
- `embed_full_corpus.py` + `wait_and_embed_full.sh` — pre-embed the full 95k-work
  corpus overnight, chunked (2000 works/chunk) so an interruption only costs the
  in-flight chunk, not the whole run
- `null_gender.py` — gender-skip stand-in (see above)
- `config/quantum_area_anchors.json` — vendored anchor config for the domain classifier
- `run_overnight.sh`, `run_new_nordic_authors.sh` — one-off orchestration scripts used
  to run the per-university chain across the 36 new institutions; not part of the
  documented pipeline, safe to delete or keep as a template for the next batch
- `build_paper_counts.py` — distinct (deduped) quantum-paper count AND domain mix
  (computing/communication/sensing) per institution and per country, written to
  `data/paper_counts_geo.json`. Fixes a caveat found after the Nordics expansion:
  `index.html`'s geo aggregates (institution/country bubbles, and the bottom-left
  colophon "N papers" stat) summed each person's own `quantum_papers` (and their
  own domain shares weighted by it), so a paper co-authored by 2+ people at the
  same institution — or across institutions/countries — was counted once per
  author, not once per paper. This script dedupes by OpenAlex work id across
  `data/all_universities_works.jsonl` (which `split_nordic_corpus.py` duplicates
  once per matching institution) and counts each distinct paper once per tracked
  institution/country it touches, plus a Nordics-wide `total` (NOT the same as
  summing the per-country or per-institution totals — those double-count a paper
  spanning 2+ tracked countries/institutions). `index.html` loads this file
  alongside `people_viz.json` and prefers it for the "Quantum papers" figure and
  domain-mix percentages in geo tooltips AND the colophon counter; if the file
  (or just its domain-mix half) is missing, it falls back to the old
  person-summed numbers, with tooltip copy that reflects which one is showing.

  Domain mix needs `data/work_domain_labels.json` — a new output of
  `build_domain_specter.py`: one hard domain label per paper (argmax of its
  existing per-work BGE-embedding distribution, before it gets averaged into
  each person's `domain_probs_specter`). A cheaper stdlib-only alternative was
  tried first — `build_taxonomy_data.py`'s lexicon classifier
  (`classify_work_domains`) — and rejected: measured at only 9.2% signal
  coverage over the 76,109 distinct papers, which would've left ~91% of papers
  domain-unclassified. The BGE model covers ~83-87% instead (gap is papers
  matched to a tracked *institution* here but not to a specific tracked
  *person* in `build_domain_specter.py`'s own matching).

  Turns out this did NOT need a GPU machine: the full 95k-work corpus was
  already 100% cached in `caches/work_embeddings_bge.npy` from a prior run, so
  re-running `build_domain_specter.py` only needed to embed the handful of
  anchor sentences (BAAI/bge-large-en-v1.5) — trivial on CPU. Installed a
  throwaway `.venv/` here (`torch` CPU wheel + `transformers` + `adapters`) and
  ran the whole thing in ~23s. If the embedding cache is ever NOT already
  warm (e.g. a fresh checkout with no `caches/`), that part genuinely does
  need the GPU per the original Sprint 13 notes above (~45 min on a 6GB-VRAM
  laptop GPU) — the CPU-only shortcut only works because encoding was already
  done.

## Geo map outline: bumped to ~3x point density

`NORDICS_GEO` in `index.html` (inline Natural Earth 1:50m outline) was
regenerated at a finer `topojson-simplify` threshold — weight 0.01 instead of
whatever aggressive default Sprint 13 originally used, chosen by sweeping
weight values and picking the point count where fjords/archipelagos read as
real coastline rather than blocky polygons. ~1000 points across the 5
countries now (was ~334), ~14KB inline (was ~5KB) — still light. Regenerated
from `world-atlas`'s `countries-50m.json` via `topojson-client`/`-server`/
`-simplify` (Node, not part of the Python pipeline — no rebuild step, this is
static outline data, not derived from the corpus). Svalbard + Jan Mayen still
excluded (same reasoning as before); a stray Jan Mayen fragment survived the
first pass (its raw-resolution ring had a vertex just outside the original
exclusion bbox) — widened the bbox and confirmed it's gone. Zero-area rings
left behind by simplification (tiny islands collapsed to one repeated point)
are pruned before embedding.

## Rebuild order (current, patents/gender deferred)

```
python split_nordic_corpus.py
# per university: extract_authors.py, fetch_epo_patents.py --skip-patents, build_people_data.py
python merge_universities.py --auto-from-config --out data/people_viz.json
python concat_works.py --out data/all_universities_works.jsonl
python build_taxonomy_data.py
python build_domain_specter.py    # writes data/work_domain_labels.json (per-work hard domain)
python build_paper_counts.py --works data/all_universities_works.jsonl --out data/paper_counts_geo.json
python build_domain_shares.py --in data/people_viz.json --out data/people_viz.json
python null_gender.py --in data/people_viz.json --out data/people_viz.json
python fetch_global_h_index.py
```

## Known gaps / next steps

1. Spot-check the 36 new `applicant_regex` patterns against real EPO data once patents
   are back in scope.
2. Decide whether to keep gender skipped long-term or re-enable `build_gender_ollama.py`.
3. `universities.json`'s Nordic list is what the user supplied — not claimed to be
   exhaustive for any of the 4 new countries, just what was handed over.
4. ~~The colophon "N papers" counter still double-counts~~ — fixed: `updateGeoScope()`
   now reads `PAPER_COUNTS.total` / `PAPER_COUNTS.by_country[cc]` for the Nordics/
   country-level colophon, same distinct-count fix as the geo bubbles. The OTHER
   two `rollCounter("paper-count", ...)` call sites (scatter/domains/taxonomy views,
   `index.html` — search for `paper-count`) are correctly left on the person-summed
   total: those show arbitrary search/render-filtered subsets that have no
   precomputed equivalent, which is a legitimately different use case, not the
   same bug.
5. ~~Domain-mix dedup needs a GPU rebuild~~ — done, see above: ran
   `build_domain_specter.py` + `build_paper_counts.py` right in this checkout using
   a throwaway CPU-only `.venv/` (embeddings were already cached), so
   `data/paper_counts_geo.json` now HAS `domain_by_institution`/`domain_by_country`
   with real coverage (~83-87%). The `.venv/` itself is disposable/gitignore-worthy —
   not a project dependency, just what this one-off rerun needed.
