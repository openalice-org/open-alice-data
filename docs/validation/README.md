# Validation evidence

This folder holds the audit behind [METHODOLOGY.md](../METHODOLOGY.md): blind label samples, the labels
produced by each labeler, and the scripts that analyse them. Paths below are relative to this folder.

## Contents

| Path | What |
| --- | --- |
| `data/batches/` | The 500-paper blind sample as 10 batches (`batch_NN.jsonl`: title, abstract, year; group hidden). |
| `data/sample_map.json` | Maps each sample item to its work id and group (`old_only`, `new_only`, `shared`). |
| `data/labels_haiku/` | Haiku labels (CORE / PERIPHERAL / NOT) for all 500 papers. |
| `data/labels_sonnet/` | Sonnet labels for a 50-paper subset (cross-check of Haiku). |
| `data/eikos_labels.json` | Eikos-4B labels and class probabilities for the 500 papers. |
| `data/live_all_result.json` | Per-work attribution of old-only works to a cause, from a live OpenAlex query. |
| `rescue/batches/`, `rescue/sample_map.json`, `rescue/labels_haiku/` | The 300-paper validation of the rescue pass admission routes (100 per stratum: classifier, admitted, extension). |
| `scripts/` | Analysis and sampling scripts. |

## Scripts

### Self-contained

Run only from files in this folder (the third needs one rescue output, see below).

| Script | Reads | Reports |
| --- | --- | --- |
| `compare_labels.py` | `data/` | Label mix by group; Haiku vs Sonnet and Haiku vs Eikos agreement and kappa. |
| `eikos_probability_vs_haiku.py` | `data/` | Haiku and Sonnet agreement by Eikos P(CORE) bin (evidence for `EIKOS_MIN_P`). |
| `rescue_validation_report.py` | `rescue/` and `corpus/data/rescue_band_eikos.json` | Precision per rescue admission route. |

`rescue_validation_report.py` prints its precision table from `rescue/` alone, but its last block (admitted
works per probability floor) reads `corpus/data/rescue_band_eikos.json`, an output of `make rescue` that is
not in git. Without it the script stops with a file-not-found error after the table.

Commands (from the repository root):

```bash
python3 -I docs/validation/scripts/compare_labels.py
python3 -I docs/validation/scripts/eikos_probability_vs_haiku.py
python3 -I docs/validation/scripts/rescue_validation_report.py
```

Expected key numbers:

| Script | Output |
| --- | --- |
| `compare_labels.py` | Haiku vs Eikos on 500 papers: agreement 80.8 %, kappa 0.71. Also Haiku vs Sonnet on 50: 78.0 %, kappa 0.67; Sonnet vs Eikos on 50: 88.0 %, kappa 0.81. |
| `eikos_probability_vs_haiku.py` | 156 Eikos CORE calls; Haiku=CORE rate rises with P(CORE): 55 % in [0.5,0.6), 67 % in [0.6,0.7), 82 % in [0.7,0.8), 100 % at 0.8 and above. |
| `rescue_validation_report.py` | Precision (Haiku=CORE): classifier 87 %, Eikos P >= 0.70 (admitted) 91 %, Eikos P in [0.5, 0.7) (extension) 73 %. Band works admitted by floor: 1783 / 1194 / 584 / 92 at 0.5 / 0.6 / 0.7 / 0.8. |

### Need legacy data (kept for provenance)

These were used to produce the audit but cannot run from this repository alone. They expect the previously
deployed build and the earlier corpus, which are not part of this repo, plus local corpus intermediates. They
resolve paths relative to a parent workspace folder that contained `Qsip/SPRINT 13/` and the deployed app.

| Script | Expects |
| --- | --- |
| `make_sample.py` | Old corpus `Qsip/SPRINT 13/data/all_universities_works.jsonl`; new `corpus/data/sweden_quantum_works.jsonl`. Draws the 500-paper sample (seed 42). |
| `abstract_availability.py` | Old corpus (above); `corpus/data/sweden_quantum_candidate_audit.csv` and `sweden_quantum_candidate_union.jsonl`. |
| `churn_old_vs_new_corpus.py` | Old corpus; new corpus; `sweden_quantum_candidate_audit.csv`. |
| `why_excluded.py` | Old corpus; new corpus; candidate audit CSV; `corpus/config/sweden_quantum_record_gates.json`. |
| `keyword_evidence.py` | Old corpus; new corpus; candidate audit CSV; `corpus/config/sweden_quantum_paper_filter.json`. Shows that 73 % of the works the new rules drop had a quantum term only in the old OpenAlex keywords. |
| `excluded_core_examples.py` | Old corpus; candidate audit CSV and union; `data/` labels. |
| `live_all.py` | Old corpus; candidate audit CSV; union manifest; a valid OpenAlex key in `.env`; live OpenAlex access. Writes `data/live_all_result.json`. |
| `headline_figure_impact.py` | A previously deployed `paper_counts_geo.json` (read from the app repo's git history) and a new `pipeline/data/paper_counts_geo.json`; `data/live_all_result.json`. |
| `rescue_validation_sample.py` | `corpus/data/rescue_band.jsonl` and `rescue_band_eikos.json` and the classifier outputs from `make rescue`. Draws the 300-paper rescue sample (seed fixed in the script). |

Re-running them is not expected; the data they produced is stored in `data/` and `rescue/`.
