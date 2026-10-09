# Methodology: how the open-alice dataset is generated

This document describes how the five JSON files behind the open-alice map are produced, why they
change from one build to the next, and how the pipeline compensates for the part of that change
that costs recall. It is meant to be read as a methods note: every number quoted here was
measured, and the script that measured it is in the repository (`docs/validation/scripts/`
for the audit, `pipeline/` for the pipeline itself).

**Changes between builds are normal.** The corpus is built from a live, evolving source
(OpenAlex). Between two builds a few weeks apart, OpenAlex re-tagged topics for 68 % of the works
that dropped out and replaced its keyword field outright. Two builds are therefore two
different views of the same literature, not a deterministic function of the code. Section 4
quantifies this, and section 3.2 describes the rescue pass added because of it.

## 1. What is produced

| File | Content |
| --- | --- |
| `people_viz.json` | One record per researcher: affiliation, paper and patent counts, h-indices, the two map axes (quantum share, builder/thinker), domain shares |
| `people_taxonomy.json` | The topic taxonomy and each researcher's place in it, plus a per-researcher domain distribution |
| `paper_counts_geo.json` | Distinct quantum papers per institution and per country, with domain mix (a paper is counted once per institution or country it touches) |
| `patent_links.json` | Patents shared by two or more tracked inventors |
| `collaborations.json` | Per researcher, top co-authors and co-institutions inside the quantum corpus |

## 2. Pipeline at a glance

```
OpenAlex ──► corpus 01-05 ──► rescue pass ──► split ──► per-institution people ──► merge ──►
             (rules)          (classifier)    (53)      (authors, patents, people)  (+works)
        ──► taxonomy ──► domain classifier ──► paper counts ──► shares / h-index ──► patents
        ──► splink dedup ──► (taxonomy, domains, shares) again ──► patent links ──► collaborations ──► stage (build/data)
```

`make help` lists the targets; `make all` runs everything after the corpus, `make rescue` runs
the rescue pass. The README has the order and the run instructions.

## 3. Stages

### 3.1 Corpus (`corpus/`, steps 01-05)

A topic-guided, high-recall corpus of Nordic and Baltic quantum research, built from OpenAlex
metadata only (no full text). Scope: works with at least one authorship in SE, NO, DK, IS, FI,
EE, LV or LT, published 1950-2026. The tool is a copy of `KTH.Tools.QuantumAuthors`; the four local
changes are in `corpus/LOCAL_CHANGES.md`.

1. **Topic snapshot and candidate topics (01-02).** The OpenAlex topic taxonomy is downloaded and
   topics whose name, keywords or description match quantum discovery terms become candidates; a
   taxonomy gate marks each *eligible*, *conditional* or *blocked*.
2. **Candidate download (03).** Works are retrieved through 60 topic routes (18 explicit-quantum
   and 42 broad-related) and through 676 paper keywords derived from the explicit-quantum
   topics (in a high-confidence and a title-confirmed tier). This gives 205,464 unique
   candidates.
3. **Reconciliation (04).** Recovers works lost to source regressions by comparing with an
   earlier corpus, when there is one. In this build there was none, so nothing was reconciled.
4. **Inclusion rule (05).** A candidate is included when any of these holds:
   it matched an explicit-quantum or intrinsically-quantum topic; it matched a high-confidence
   derived keyword; it matched a lower-confidence derived keyword *and* its title contains a
   quantum term; or it matched only a **broad-related** topic *and* its title, abstract or
   OpenAlex keywords contain a literal quantum term (`quantum`, `qubit`, `entangl…`,
   `decoher…`, and the other patterns in `corpus/config/sweden_quantum_paper_filter.json`).
   Two intentional scope gates apply on top: language (English or missing) and work type
   (datasets, errata, editorials and similar are excluded), plus a safeguard that
   keyword-only matches in the Social Sciences need extra evidence.

   Result: **83,732 included**, 121,732 excluded. Of all 205,464 candidates, 15,669 fail a
   record gate, 448 fail the domain safeguard and **105,615 are excluded by the content rule
   alone** (the broad-topic rule found no quantum term).

### 3.2 The rescue pass (`pipeline/rescue_*.py`, `make rescue`)

**Why it exists.** The content rule is only as good as its evidence. A blind audit of 200 works
that the earlier build had included and this build excluded (section 5) found that about a
quarter are core quantum research. The cause is visible in the data: OpenAlex replaced the
`keywords` field between the two builds. The old keywords were generic concepts ("Quantum
mechanics", "Quantum tunnelling"); the new ones are paper-specific phrases ("exciton-polariton
condensates", "effective Hamiltonian"). The old build often passed the literal-term check through
one of those generic keywords. A clearly quantum paper whose title and abstract never use the
word "quantum" (for example *Josephson scanning tunneling microscopy*) now has nothing for the
rule to match. A text-semantic classifier does not depend on the field that changed.

**What it does.**

| Stage | Script | What |
| --- | --- | --- |
| R1 pool | `rescue_pool.py` | Selects candidates excluded by the content rule only, with an abstract of at least 200 characters. Record-gate and domain-safeguard exclusions are intentional scope decisions and are never revisited. Works without a usable abstract (35,424) stay excluded: there is not enough text to judge. **Pool: 70,191 works.** |
| R2 embed | `embed_missing_npu.py` | bge-large-en-v1.5 CLS embeddings of the pool on the Intel NPU (OpenVINO), into a separate cache. |
| R3a label | `rescue_sample.py`, `label_with_eikos.py` | Eikos-4B labels a sample of the pool CORE / PERIPHERAL / NOT with the rubric in `rescue_rubric.py`: 2,000 works drawn uniformly (the only stratum used for estimates) and 2,000 drawn from works containing quantum-adjacent vocabulary (training only, so the classifier sees enough positives). |
| R3b train | `rescue_train.py` | Logistic regression on the frozen, L2-normalised embeddings, target CORE. |
| R4 decide | `label_with_eikos.py`, `rescue_apply.py` | The classifier admits confident works; Eikos adjudicates the uncertain band; the final corpus is the original corpus plus the rescued works. |

**What "CORE" means.** The rescue admits only works where quantum physics or quantum technology
is the *central* subject, the same definition the corpus filter states
(`corpus/config/sweden_quantum_paper_filter.json`). Papers where quantum mechanics is only a
tool (quantum chemistry, quantum dots as nanomaterials, collider analyses) are labelled
PERIPHERAL and are deliberately **not** rescued. The audit found the excluded pool full of them.

**Design decisions, with the evidence for each.**

- *Embed the whole pool; no lexical prefilter.* In the uniform sample only 52 % of the core works
  match the quantum-adjacent vocabulary used for sampling, so a lexicon prefilter would have lost
  nearly half of what the rescue is for. The lexicon only steers *which works get labelled*.
- *Weights restore the pool's class balance.* The labelled sample is enriched in positives, so
  each labelled work is weighted by the inverse of its probability of being sampled
  (Horvitz-Thompson). Predicted probabilities then mean what they say on the pool.
- *Everything is estimated on the uniform stratum.* Out-of-fold (5-fold) results on the 2,000
  uniform works: base rate of CORE **6.2 %** (about 4,350 core works in the pool, ±740), AUC 0.961
  and average precision 0.729 at the best regularisation (C=10, an interior optimum of the grid
  0.1 ... 1000).
- *Thresholds come from the same out-of-fold results.* Accept at P(core) ≥ 0.71, the lowest
  threshold with at least 90 % out-of-fold precision (0.91 on 46 works); reject below 0.04,
  the highest threshold that keeps 95 % of the positives. In between is the **uncertain band**.
- *Eikos needs a probability floor, and the floor is measured, not assumed.* Eikos's probabilities
  are flat (median P(CORE) 0.78 among its CORE calls) but informative: on the 500-paper audit set Haiku
  agrees with its CORE calls 55 % of the time at P in [0.5, 0.6), 67 % in [0.6, 0.7), 82 % in [0.7, 0.8)
  and 100 % from 0.8. The audit set is not the band, though, so the three admission routes were then
  measured on the band itself with 100 blind Haiku-labelled works each:

  | Route | Works | CORE | PERIPHERAL | NOT |
  | --- | ---: | ---: | ---: | ---: |
  | Classifier alone (P ≥ 0.71) | 2,071 | 87 % (79-92 %) | 9 % | 4 % |
  | Eikos CORE, P ≥ 0.70 | 584 | 91 % (84-95 %) | 9 % | 0 % |
  | Eikos CORE, 0.5 ≤ P < 0.70 | 1,199 | 73 % (64-81 %) | 25 % | 2 % |

  Even the weakest tier is almost never non-quantum, and its misses are mostly PERIPHERAL. The
  rule-based corpus holds far noisier works: in the audit, the works only the new rules admitted were
  28 % CORE and 43 % NOT. The floor is therefore **0.50**, the lowest tier that was measured; Eikos CORE
  calls below 0.50 (438 of the 2,221 in the band) have no evidence behind them and stay excluded.
- *The band is capped by effort, not by chance.* The band holds 13,698 works. Sorted by closeness to
  0.5, the model's calibrated estimates put 63 %, 78 % and 87 % of the band's core works in the
  first 4,000, 6,000 and 8,000 works. The first 6,000 (about 60 minutes of iGPU time) are sent to
  Eikos; the rest stay excluded. The cap is a parameter (`rescue_train.py --band-cap`).
- *A frozen-embedding linear head, not a fine-tuned encoder.* About 4,000 labels from a noisy
  teacher (Eikos agrees with Haiku on 81 % of papers) would let a flexible model fit the labelers'
  noise. The linear head has few parameters, is regularised and is checked out-of-fold; the
  cross-validation numbers above show there is no obvious headroom a heavier model would use.

**Provenance.** Every rescued row carries `quantum_included: true`, a one-sentence
`quantum_inclusion_reasons` entry saying which stage admitted it and with what score, and a
`rescue_pass` object (stage, classifier probability, Eikos label and probability). The original
corpus is copied through unchanged, so the rule-only corpus is always recoverable
(`make split RESCUE=0`).

**Results (this build).**

| | Works |
| --- | ---: |
| Pool (content-rule exclusions with an abstract) | 70,191 |
| Dropped by the classifier (P(core) < 0.04; about 325 core works lost by its own estimate) | 54,422 |
| Uncertain band | 13,698 |
| Band works adjudicated by Eikos (the 6,000 closest to 0.5, plus 36 that were already labelled) | 6,036 |
| Admitted by the classifier | 2,071 |
| Admitted by Eikos (CORE with P ≥ 0.50) | 1,819 |
| **Rescued** | **3,890** |
| Band works left excluded: Eikos not CORE / CORE below the floor / beyond the cap | 4,331 / 447 / 7,101 |

The final corpus has 87,622 works (83,732 + 3,890, +4.6 %). From the measured precisions above, about
83 % of the rescued works are CORE and about 3 % are not quantum research at all. Against an estimated
4,350 core works in the pool (±740) that is a recall of roughly three quarters, with wide uncertainty:
the pool estimate comes from 124 core works in the uniform sample, and each precision from 100 labelled works.

Effect on the dataset: people 8,753 to 9,116 (370 new ids, 7 lost), distinct papers 77,107 to 80,757.
The Nordic country totals moved back toward the previously deployed figures: Sweden 33,513 to 35,419
(35,955 deployed), Denmark 21,146 to 21,948 (20,784), Finland 12,818 to 13,287 (13,563), Norway 8,910 to
9,324 (9,499).

### 3.3 Split and authors

`split_nordic_corpus.py` streams the final corpus once and writes, for each of the 53
institutions in `pipeline/universities.json`, the works whose authorships list that institution
(a work co-authored across institutions is written once per institution). `extract_authors.py`
then builds each institution's author table. Four institutions end up with no works and are skipped.

### 3.4 People

`build_people_data.py` (heuristic method, no LLM) turns each institution's authors and works into
person records: paper counts, h-indices from the matched works, and the two axes of the map,
*quantum share* (Tourist to Resident: how much of a person's output is quantum) and
*builder/thinker* (applied versus theoretical, from title and abstract). Patents come from the EPO
caches (below). `merge_universities.py` merges the institutions; `fetch_global_h_index.py` adds
OpenAlex's global h-index (cached).

### 3.5 Taxonomy and domain classifier

`build_taxonomy_data.py` builds the topic taxonomy and places each person in it.
`build_domain_specter.py` embeds every work with bge-large-en-v1.5 (CLS token) and scores it against
sentence anchors for *computing*, *communication*, *sensing* and *other*
(`pipeline/config/quantum_area_anchors.json`), with the softmax temperature calibrated by a
grid search against a lexicon weak-truth. Per-work domain labels feed `build_paper_counts.py`
(distinct-paper counts and domain mix per institution and country) and per-person shares
(`build_domain_shares.py`). Embeddings are cached; the NPU embedder fills the cache for new works.

### 3.6 Duplicate-author fusion (`dedup`, `make dedup`)

OpenAlex sometimes splits one researcher over several author ids. A splink record-linkage model
(name, publication years, institution, topics; blocked on surname) proposes clusters, and a
transparent rule decides: **FUSE** when real co-author overlap is high (Jaccard ≥ 0.15) or a
no-paper record matches at ≥ 0.9; **REJECT** when both records have co-authors and none are shared;
otherwise **REVIEW**. Merging fuses only FUSE clusters and **never** a cluster whose members carry two
different ORCIDs. That is proof of distinct people (the common Nordic surnames Andersson, Nielsen,
Hansen, Persson, ...). In this build 385 clusters were found, 133 FUSE, and 126 records were
merged away. The dedup had been applied once before (app commit `60cf6d3`) and was then
overwritten by later data rebuilds, so it is now a pipeline step that runs last.

### 3.7 Patents and collaborations

Patents come from EPO OPS through per-author searches whose results are cached
(`data/<institution>/caches/`), filtered by an institution-specific applicant pattern. The
default `PATENTS=cache` rebuilds counts from the caches without calling EPO; `PATENTS=full`
fetches live. `build_patent_links.py` and `build_collaborations.py` derive the two network files.
Both key on person ids, which is why they run after the dedup.

## 4. Why two builds differ

The previous deployed corpus and the present one share only **46,515 works** (78,769 and 83,732
distinct works respectively): 32,254 works are only in the old one and 37,217 only in the new one.

Cause of the 18,545 old-only works that the new run does not retrieve at all, checked against live
OpenAlex for every one of them (`docs/validation/scripts/live_all.py`):

| Cause | Works | Share |
| --- | ---: | ---: |
| Re-tagged: the topic that retrieved the work is no longer on it | 12,699 | 68.5 % |
| The old topic is still on the work but is not one of this run's routes (different broad-topic set) | 2,930 | 15.8 % |
| Deleted or merged in OpenAlex | 2,337 | 12.6 % |
| No in-scope country on the work any more | 577 | 3.1 % |
| Anomalies | 2 | 0.0 % |

The other 13,709 old-only works were retrieved but excluded; 98.2 % of them by the content rule,
for the keyword reason described in section 3.2.

**Effect on the figures the map shows.** Before the rescue pass, distinct papers moved from 78,526 to 77,107 (−1.8 %); with it they are 80,757 (+2.8 % against the deployed build).
The four large countries moved by −7 % to +2 %; institution paper counts correlate at 0.979
(Spearman) and the ten largest institutions are the same ten. The Baltics grew (+76 % to +129 %)
because they are now classified by the same rules as the Nordics, from small bases. People
moved most (9,766 deployed, 8,753 after the dedup without the rescue, 9,116 with it), concentrated in researchers
with few papers: only 6,703 person ids appear in both the deployed and the pre-rescue builds.

## 5. Audit method

To judge whether the dropped and the added works are good ones, 500 works were sampled blind from three
groups (200 only in the old corpus, 200 only in the new one, 100 in both) with the group hidden from
the labelers, and labelled CORE / PERIPHERAL / NOT from title and abstract with the rubric in
`pipeline/rescue_rubric.py`. Labelers: ten Haiku sub-agents (50 papers each), one Sonnet
sub-agent on the first 50 as a quality reference, and Eikos-4B (on the Intel iGPU, 0.6 s per
paper). The inputs, labels and scripts are in `docs/validation/`.

| Group | CORE | NOT (Haiku / Eikos) |
| --- | ---: | ---: |
| In both corpora | 51 % / 57 % | 11 % / 9 % |
| Only in the new corpus | 28 % / 30 % | 43 % / 36 % |
| Only in the old corpus | 19 % / 20 % | 56 % / 50 % |

Agreement: Haiku and Eikos 80.8 % (kappa 0.71, 500 papers), Haiku and Sonnet 78 % (kappa 0.67, 50),
Sonnet and Eikos 88 % (kappa 0.81, 50). Almost all disagreement is at the PERIPHERAL / NOT boundary,
where Haiku is the strictest.

Reading: the shared core is stable and on-topic, and **the churn is at the edge of the corpus**.
The works re-tagged away were almost never core (about 1 % CORE); the works dropped by the content
rule were about a quarter core, which is the loss the rescue pass addresses.

Caveats. The rubric is ours, not validated by a domain expert. The labels are model judgements from
title and abstract and are not ground truth. About 30 % of works in every group have no usable abstract and
could not be labelled (the rate is equal across groups, so it does not bias the comparison).
Samples are 200 per group.

## 6. Reproducibility and limits

- **Not in git.** `corpus/data/` (the corpus and its intermediates), `pipeline/caches/` (embeddings,
  compiled NPU blobs) and `pipeline/data/<institution>/caches/` (EPO responses). Regenerating the
  corpus from OpenAlex will not give the same corpus (section 4).
- **Cost.** Corpus 01-05: about 1.5 hours of API calls. Rescue pass: about 2 hours (NPU embedding
  about 55 minutes, Eikos labelling about 45 minutes plus about 60 for the band). Pipeline after the
  corpus: about 25 minutes. Peak memory is bounded by cgroup caps in the Makefile (`systemd-run`).
- **Environments.** Python environments are created with `uv` (`make setup`, `make setup-dedup`,
  `make setup-openvino`). The rescue pass needs the OpenVINO environment; see
  [NPU_SETUP.md](NPU_SETUP.md).
- **Large inputs.** The corpus, embedding caches and EPO caches are not in git. In the future they will be
  served as downloadable artifacts from an S3 bucket; this is not available yet (see the README).
- **Hardware used.** Intel Core Ultra 7 355 (NPU for single-pass encoders, iGPU for the LLM), 22 GB RAM,
  no NVIDIA GPU. The NPU needs a version-matched compiler (see [NPU_SETUP.md](NPU_SETUP.md)).
- **Patents** are rebuilt from earlier EPO caches, so they under-report for institutions or authors
  that were never searched; a live `PATENTS=full` run fills the gap.
- **Abstract-less works** (35,424 content-rule exclusions) cannot be rescued and remain excluded.
- **Uncertain band tail.** Band works beyond the effort cap stay excluded; by the model's own estimate
  this is about 22 % of the band's core works (the cap keeps 78 %).
