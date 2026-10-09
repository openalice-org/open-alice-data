# Swedish Quantum Corpus

Builds a corpus of Swedish-affiliated quantum papers from OpenAlex metadata, ranks it against an
explicitly-quantum core, and renders `quantum_core_review_report.html`.

## Setup

```bash
uv venv --python=3.13
uv sync
```

Steps 01–05 call the OpenAlex API and need either `OPENALEX_API_KEY` or `OPENALEX_MAILTO` (the
polite pool). If both are set, `OPENALEX_API_KEY` takes priority. Copy `.env.example` to `.env`
and fill it in; settings are loaded from `.env` automatically (see `utils/settings.py`).

Steps 06–10 are local and make no network requests.

## Run

```bash
make corpus     # 01-05, needs the API key
make review     # 06-10, local only
```

`make all` runs both. `make help` lists every target. Once a corpus exists, `make review` alone
rebuilds the report.

Run the steps in numeric order. Each one verifies its inputs against the SHA-256 hashes recorded in
the previous step's manifest and refuses to run on a mismatch, so a skipped or stale step is an error
rather than a silently wrong report.

## Scripts

Run in this order.

| # | Script | `make` | Reads | Writes |
| --- | --- | --- | --- | --- |
| 01 | `01_snapshot_openalex_topics.py` | `snapshot` | OpenAlex API | `openalex_topics_snapshot.jsonl` + manifest |
| 02 | `02_inventory_quantum_topic_candidates.py` | `topics` | the snapshot, `discovery_terms` | `sweden_quantum_topic_candidates.csv` + manifest |
| 03 | `03_download_sweden_quantum_candidates.py` | `download` | topic candidates, `scope`, `keyword_derivation`, `paper_filter` | `candidate_union.jsonl`, `derived_keywords.csv`, per-route cache + manifests |
| 04 | `04_reconcile_sweden_quantum_source_regressions.py` | `reconcile` | the union, `scope`, `paper_filter` | `source_reconciliations.jsonl`, reconciliation audit + manifest |
| 05 | `05_filter_sweden_quantum_candidates.py` | `filter` | the union, reconciliations, `scope`, `keyword_derivation`, `paper_filter`, `record_gates` | `sweden_quantum_works.jsonl` (the corpus), `candidate_audit.csv` + manifest |
| 06 | `06_build_quantum_core_set.py` | `core` | the union, topic candidates, `core_and_outliers`, `scope` | `core_works.jsonl`, `candidate_features.jsonl`, `core_summary.csv` + manifest |
| 07 | `07_embed_candidate_texts.py` | `embed` | `candidate_features.jsonl`, `core_and_outliers` | `embeddings_masked.npy`, `embeddings_unmasked.npy`, `embedding_work_ids.txt` + manifest |
| 08 | `08_detect_quantum_outliers.py` | `outliers` | features, embeddings, candidate audit, `core_and_outliers` | `outlier_scores.csv`, `outlier_clusters.csv`, `core_centers.csv` + manifest |
| 09 | `09_score_quantum_area_affinities.py` | `affinities` | embeddings, outlier scores, `area_anchors` | `area_affinities.csv`, `area_prototypes.npy` + manifest |
| 10 | `10_report_quantum_core_review.py` | `report` | everything above | **`quantum_core_review_report.html`** |

Steps 01–05 build the corpus. Steps 06–10 are a review instrument: they change no inclusion
decision, and step 10 re-imports the core rule from step 06 so the report cannot drift from the
pipeline that produced the corpus.

Data files live in `data/` (all gitignored) and are named `sweden_quantum_*` unless shown otherwise.

### Helpers

| File | Purpose |
| --- | --- |
| `utils/openalex_utils.py` | OpenAlex paging, URL building, `file_sha256` |
| `utils/quantum_topic_groups.py` | The `explicit_quantum` / `intrinsically_quantum` / `broad_related` grouping |
| `utils/abstract_utils.py` | Rebuilds abstracts from OpenAlex inverted indexes |
| `utils/config_digest.py` | Canonical JSON hash, used to pin one config section instead of a whole file |
| `utils/settings.py` | `pydantic-settings`-backed `Settings`, loads `OPENALEX_API_KEY` / `OPENALEX_MAILTO` from the env and `.env` |

## Configuration

| File | Read by | Controls |
| --- | --- | --- |
| `sweden_quantum_discovery_terms.json` | 02 | Terms that discover quantum topics; the taxonomy domain gate |
| `sweden_quantum_scope.json` | 03, 04, 05, 06 | Swedish-affiliation rule and the year windows |
| `sweden_quantum_keyword_derivation.json` | 03, 05 | Thresholds for deriving paper-keyword routes |
| `sweden_quantum_paper_filter.json` | 03, 04, 05 | Topic acceptance, `broad_related` confirmation, paper-level term patterns |
| `sweden_quantum_record_gates.json` | 05, 10 | Accepted languages and excluded work types |
| `sweden_quantum_core_and_outliers.json` | 06, 07, 08, 10 | The core rule, the two feature views, the score bands |
| `sweden_quantum_area_anchors.json` | 09 | Prose anchors for the three quantum research areas |

Changing a config invalidates the steps downstream of it, because manifests pin config hashes.
`scope`, `keyword_derivation` and `paper_filter` are pinned by the candidate-union manifest, so
editing one of those requires re-running the download in step 03.

### Changing the target country

The corpus is scoped to Sweden via the `"filter_value": "SE"` entries under `country_evidence` in
`config/sweden_quantum_scope.json`. To target a different country (or set of countries), change
both the `primary` and `audit` `filter_value` fields to that country's OpenAlex country code, then
re-run from step 03. Both fields accept a `|`-separated list, so all five Nordic countries would be:

```json
"filter_value": "SE|NO|DK|FI|IS"
```

## Notes

- KTH OpenAlex institution lineage ID: `I86987016`. The corpus is national; KTH is only where the
  project started and why the repository is named this way.
- No PDFs or full text are downloaded, and no trained classifier or LLM makes any inclusion decision.
