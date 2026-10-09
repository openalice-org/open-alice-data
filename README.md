# open-alice-data

Data pipeline that regenerates the five JSON files the open-alice interactive map loads:
`people_viz`, `people_taxonomy`, `paper_counts_geo`, `patent_links`, `collaborations`.
It starts from a quantum-research corpus built from OpenAlex (Nordic and Baltic institutions), classifies and
deduplicates authors, and writes the files to `build/data/`.

Methods, evidence and caveats are in [docs/METHODOLOGY.md](docs/METHODOLOGY.md). Read it first.

## Prerequisites

| Requirement | Needed for | Notes |
| --- | --- | --- |
| Linux, `git`, `make` | everything | |
| [`uv`](https://docs.astral.sh/uv/) | everything | Installs Python 3.13 itself; no system Python needed. |
| `systemd` (`systemd-run`) | memory caps | Optional. Without it `MEM_CAP` is not enforced. |
| Intel NPU + iGPU, OpenVINO | rescue pass, `make embed-npu` | Optional. See [docs/NPU_SETUP.md](docs/NPU_SETUP.md). |
| OpenAlex key or contact e-mail | corpus steps 01-05, global h-index | `OPENALEX_API_KEY` or `OPENALEX_MAILTO`. |
| EPO OPS credentials | `PATENTS=full` only | `EPO_CONSUMER_KEY`, `EPO_CONSUMER_SECRET_KEY`. |

## Setup

```bash
cp .env.example .env     # then edit .env
make setup               # .venv (Python 3.13), CPU-only torch, pinned requirements.txt
make setup-dedup         # dedup/.venv with splink 4.x
make setup-openvino      # optional: .venv-openvino, only for the rescue pass
```

| Variable | Purpose | Default |
| --- | --- | --- |
| `OPENALEX_API_KEY` / `OPENALEX_MAILTO` | corpus steps 01-05, `make hindex` | none |
| `EPO_CONSUMER_KEY`, `EPO_CONSUMER_SECRET_KEY` | live EPO patents (`PATENTS=full`) | none |
| `PATENTS` | `cache` (rebuild from caches), `full` (fetch live), `skip` (all patent counts 0) | `cache` |
| `RESCUE` | use the rescue-augmented corpus in `make split` | `1` |
| `EIKOS_DIR` | OpenVINO folder of Eikos-4B | `models/eikos` |
| `NPU_PY`, `NPU_LIB` | python of the OpenVINO env; optional dir with a matching NPU compiler library | see [NPU setup](docs/NPU_SETUP.md) |
| `EMBED_DEVICE`, `LABEL_DEVICE` | device for embeddings / for Eikos labelling | `NPU` / `GPU` |
| `MEM_CAP` | memory cap per stage (via `systemd-run`) | `7G` |
| `STAGE_DIR` | where `make stage` writes the five JSONs | `build/data` |
| `APP_PUBLIC` | the app's `public/data` folder, target of `make publish` | see `.env.example` |

## Quickstart (no NPU, no rescue)

Assumes you already have the corpus file (see [Inputs that are not in git](#inputs-that-are-not-in-git)).

```bash
mkdir -p corpus/data
# place the corpus at corpus/data/sweden_quantum_works.jsonl
make split RESCUE=0      # corpus -> one file per institution
make all                 # people, merge, taxonomy, domains, counts, shares, hindex, patents, dedup, stage
```

Outputs are in `build/data/`. `make all` never touches the app. To update the app, run `make publish`
explicitly; it copies `build/data/` to `APP_PUBLIC`.

## Full run

| Phase | Command | Rough time | Needs |
| --- | --- | --- | --- |
| Corpus (OpenAlex steps 01-05) | `make corpus` | about 1.5 h of API calls | OpenAlex key or e-mail |
| Rescue pass | `make rescue` | about 2 h | OpenVINO env, Eikos, NPU and iGPU |
| Split by institution | `make split` | minutes | corpus file |
| Data steps | `make all` | about 10-25 min | caches (see below) |

Disk use: corpus data about 16 GB (candidate union about 4 GB plus route caches); pipeline data and
caches about 7.5 GB. Peak memory per stage is capped by `MEM_CAP`. Timings and the breakdown of the rescue
pass are in [METHODOLOGY section 6](docs/METHODOLOGY.md).

```bash
make corpus      # OpenAlex steps 01-05
make rescue      # optional but recommended
make split       # uses the rescued corpus when present (RESCUE=1)
make all
```

## Make targets

`make help` prints the same list.

| Target | What it does |
| --- | --- |
| `help` | List targets. |
| `setup` | Create `.venv` with uv (Python 3.13), CPU-only torch, pinned `requirements.txt`. |
| `setup-dedup` | Create `dedup/.venv` (splink 4.x). |
| `setup-openvino` | Create `.venv-openvino` (OpenVINO, transformers 4.57.6). |
| `corpus` | Build and classify the corpus from OpenAlex (steps 01-05). |
| `rescue` | Whole rescue pass: `rescue-pool`, `rescue-embed`, `rescue-sample`, `rescue-label`, `rescue-train`, `rescue-band`, `rescue-apply`. |
| `split` | Write one corpus file per institution (`RESCUE=0` forces the rule-only corpus). |
| `authors`, `patents`, `people` | Per-institution author extraction, patents, people data. |
| `merge`, `concat` | Merge institutions into `people_viz.json`; concatenate works. |
| `taxonomy` | `people_taxonomy.json`. |
| `domains` | Domain labels per work (bge-large embeddings). |
| `paper-counts` | `paper_counts_geo.json`. |
| `shares`, `hindex` | Domain shares and global h-index, written into `people_viz.json`. |
| `patch-patents` | Apply patent counts to people and institutions. |
| `links` | `patent_links.json`. |
| `collab` | `collaborations.json`. |
| `dedup` | Duplicate-author fusion (splink), then re-run the id-keyed steps. |
| `stage` | Copy the five JSONs to `STAGE_DIR` (default `build/data`). |
| `publish` | Copy from `STAGE_DIR` to `APP_PUBLIC`. Never part of `all`. |
| `all` | `people merge taxonomy domains paper-counts shares hindex patch-patents dedup stage`. |
| `embed` | Embed all works on CUDA (optional). |
| `embed-npu` | Embed missing works on the Intel NPU. |

## Repository layout

```
corpus/            Step 0: builds and classifies the quantum corpus from OpenAlex (own uv project;
                   see corpus/README.md and corpus/LOCAL_CHANGES.md)
pipeline/          Rescue pass (rescue_*.py) and data steps; data/ and caches/ are created inside it
dedup/             Splink duplicate-author fusion (run by make dedup; set up with make setup-dedup)
scripts/           Helper scripts, e.g. export_eikos_openvino.py (Eikos-4B -> OpenVINO)
docs/              METHODOLOGY.md (methods note), NPU_SETUP.md, HANDOVER_sprint13.md (historical)
docs/validation/   Audit evidence: blind samples, labels, analysis scripts (see its README)
build/             Generated output; build/data/ holds the five JSONs (git-ignored)
models/            Local model files, e.g. models/eikos (git-ignored)
```

## Inputs that are not in git

> **Planned, not available yet.** These inputs are intended to be served as downloadable artifacts from an
> S3 bucket in the future. No such bucket or download command exists today; until then, regenerate or supply
> them yourself as described below.

| Path | Size | Regenerated by | Without it |
| --- | --- | --- | --- |
| `corpus/data/sweden_quantum_works.jsonl` (and `sweden_quantum_works_with_rescue.jsonl`) | about 1.4 GB each | `make corpus` / `make rescue` | The pipeline cannot start. A regenerated corpus will not be byte-identical, because OpenAlex changes. |
| `pipeline/caches/work_embeddings_bge.npy` + `work_embedding_ids_bge.json` | about 400 MB | `make embed` or `make embed-npu` | `make domains` re-embeds every work: hours on CPU. |
| `pipeline/data/<institution>/caches/` (EPO search, biblio, text) | about 330 MB | `PATENTS=full` (slow, needs EPO credentials) | Use `PATENTS=skip`, or a slow `PATENTS=full`. |
| `pipeline/caches/openalex_global_h_index.jsonl` | small | `make hindex` | Refetched from OpenAlex when missing. |

## Known gaps

- The rescue embedding and Eikos labelling are OpenVINO-only (Intel NPU / iGPU). There is no CUDA or ROCm path yet.
- Patents are rebuilt from caches only unless `PATENTS=full`.
- 15 institutions have no EPO cache and get 0 patents under `PATENTS=cache`.
- Builds are not reproducible byte for byte (the source changes; see METHODOLOGY section 4).
- `applicant_regex` for the 36 institutions added later was never validated against real EPO data.

## Troubleshooting

| Symptom | Cause and fix |
| --- | --- |
| A stage is killed (OOM) | The stage exceeded `MEM_CAP` (default 7G). Raise it, e.g. `make rescue MEM_CAP=9G`. The 512-token NPU compile peaks near 6.5 GB. |
| NPU compile fails with `vclAllocatedExecutableCreate2` / `ZE_RESULT_ERROR_UNKNOWN` | Packaged compiler out of sync with the driver. See [docs/NPU_SETUP.md](docs/NPU_SETUP.md) and set `NPU_LIB`. |
| splink import or API errors in `make dedup` | splink 5 is incompatible. `make setup-dedup` pins 4.x; re-run it rather than installing splink into `.venv`. |
| `make domains` takes hours | Embedding cache missing; see [Inputs that are not in git](#inputs-that-are-not-in-git). |

## License

MIT, see [LICENSE](LICENSE).
