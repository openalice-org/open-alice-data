# Open Alice data pipeline: corpus -> per-university people -> merged datasets -> 5 JSONs the app loads.
# Run `make help`. Order below is the order the targets must run in (`make all` does it).
-include .env
export
PY          ?= $(CURDIR)/.venv/bin/python
# publish target: copies the staged JSONs into the app checkout. Default is a sibling checkout (relative path); override in .env.
APP_PUBLIC  ?= $(CURDIR)/../open-alice/public/data
STAGE_DIR   ?= build/data
RESCUE      ?= 1
PATENTS     ?= cache    # cache = rebuild from pipeline/data/<uni>/caches only, no EPO calls (fast, no creds)
#                         full  = live EPO fetch, resumes from caches (slow, needs EPO_* in .env)
#                         skip  = no patent data at all (all patent counts 0)
UNIS        := $(shell python3 -c "import json;print(' '.join(json.load(open('pipeline/universities.json'))))")
P            = cd pipeline &&
EPO_FLAGS    = $(if $(filter full,$(PATENTS)),,$(if $(filter cache,$(PATENTS)),--skip-fetch,--skip-patents))

.PHONY: help setup setup-openvino setup-dedup dedup rescue rescue-pool rescue-embed rescue-sample rescue-label rescue-train rescue-band rescue-apply \
        corpus split authors patents people merge concat taxonomy embed embed-npu domains \
        paper-counts shares hindex patch-patents links collab stage publish all
help:
	@echo "Setup (uv)"
	@echo "  setup           create .venv (Python 3.13, CPU torch) and install requirements.txt"
	@echo "  setup-openvino  create .venv-openvino (OpenVINO, for NPU embedding / Eikos labelling)"
	@echo "  setup-dedup     create dedup/.venv (splink 4.x)"
	@echo "Corpus"
	@echo "  corpus          OpenAlex corpus steps 01-05 (API, slow; skip if you already have the corpus)"
	@echo "Rescue pass (OpenVINO only: Intel NPU/iGPU; KNOWN GAP, no CUDA/ROCm path yet)"
	@echo "  rescue          all rescue-* steps in order (~2 h)"
	@echo "  rescue-pool     R1  select rule-excluded candidates to re-screen"
	@echo "  rescue-embed    R2  embed the pool (EMBED_DEVICE, default NPU)"
	@echo "  rescue-sample   R3a sample of the pool to label"
	@echo "  rescue-label    R3a label the sample with Eikos-4B (LABEL_DEVICE, default GPU)"
	@echo "  rescue-train    R3b logistic regression, thresholds, uncertain band"
	@echo "  rescue-band     R4a Eikos adjudicates the uncertain band"
	@echo "  rescue-apply    R4b write the corpus with rescued works"
	@echo "Data steps"
	@echo "  split           corpus -> pipeline/data/<uni>/ reviewed works"
	@echo "  authors         extract authors per university"
	@echo "  patents         EPO patents per university (PATENTS=cache|full|skip)"
	@echo "  people          authors + patents + build_people_data per university"
	@echo "  merge           merge universities -> people_viz.json"
	@echo "  concat          concatenate all works (also run by merge)"
	@echo "  taxonomy        people_taxonomy.json"
	@echo "  domains         domain classifier -> work_domain_labels.json (needs caches/)"
	@echo "  paper-counts    paper_counts_geo.json"
	@echo "  shares          domain shares into people_viz.json"
	@echo "  hindex          global h-index into people_viz.json (OpenAlex API)"
	@echo "  patch-patents   inject per-patent detail and counts"
	@echo "  links           patent_links.json"
	@echo "  collab          collaborations.json"
	@echo "Duplicate authors"
	@echo "  dedup           splink duplicate-author fusion (mandatory), then re-run id-keyed steps"
	@echo "Output"
	@echo "  stage           copy the 5 JSONs to STAGE_DIR=$(STAGE_DIR)"
	@echo "  publish         copy STAGE_DIR to APP_PUBLIC=$(APP_PUBLIC) (explicit, never part of all)"
	@echo "  all             people merge taxonomy domains paper-counts shares hindex patch-patents dedup stage"
	@echo "Embedding (optional)"
	@echo "  embed           pre-warm bge embedding cache on CUDA (~45 min)"
	@echo "  embed-npu       embed works missing from the cache on the NPU (EMBED_DEVICE)"

setup:   # CPU-only torch first: the default index pulls the multi-GB CUDA stack
	uv venv --python 3.13 .venv
	uv pip install --python .venv/bin/python torch --index-url https://download.pytorch.org/whl/cpu
	uv pip install --python .venv/bin/python -r requirements.txt

setup-openvino:   # for NPU embedding and Eikos labelling (see requirements-openvino.txt)
	uv venv --python 3.13 .venv-openvino
	uv pip install --python .venv-openvino/bin/python torch==2.14.0 --index-url https://download.pytorch.org/whl/cpu
	uv pip install --python .venv-openvino/bin/python -r requirements-openvino.txt

corpus:
	$(MAKE) -C corpus corpus   # steps 01-05 only. First: cd corpus && uv sync; needs OPENALEX_API_KEY in corpus/.env

# RESCUE=1 (default): split the corpus WITH the rescue pass when `make rescue` has produced it; RESCUE=0 forces the rule-only corpus.
CORPUS_RESCUED = corpus/data/sweden_quantum_works_with_rescue.jsonl
CORPUS_FILE    = $(if $(and $(filter 1,$(RESCUE)),$(wildcard $(CORPUS_RESCUED))),../$(CORPUS_RESCUED),../corpus/data/sweden_quantum_works.jsonl)
split:
	@echo "splitting $(CORPUS_FILE)"
	$(P) $(PY) split_nordic_corpus.py $(CORPUS_FILE)

authors:
	$(P) for u in $(UNIS); do echo "== $$u"; $(PY) extract_authors.py --university $$u || echo "FAILED $$u"; done

patents:
	@$(P) for u in $(UNIS); do echo "== $$u"; \
	  $(PY) fetch_epo_patents.py --university $$u $(EPO_FLAGS) || echo "FAILED $$u"; \
	  $(if $(filter skip,$(PATENTS)),,$(PY) build_author_patents.py --university $$u || echo "FAILED author_patents $$u";) \
	done

people: authors patents
	$(P) for u in $(UNIS); do echo "== $$u"; $(PY) build_people_data.py --method heuristic --no-llm \
	  --authors-csv data/$$u/authors_v2.csv \
	  --papers-jsonl data/$$u/quantum_classified_works_specter_reviewed.jsonl \
	  --patents-csv data/$$u/outputs/author_summary.csv \
	  --out data/$$u/people_viz_$$u.json || echo "FAILED $$u"; done

merge:
	$(P) $(PY) merge_universities.py --auto-from-config --out data/people_viz.json
	$(MAKE) concat

concat:
	$(P) $(PY) concat_works.py --out data/all_universities_works.jsonl

taxonomy:
	$(P) $(PY) build_taxonomy_data.py

embed:   # optional: pre-warm caches/work_embeddings_bge.npy on CUDA (~45 min). Skip if caches/ is copied in.
	$(P) $(PY) -u embed_full_corpus.py

# Embed works missing from the cache on the Intel NPU (see README, 'NPU embedding'). Staged + memory-capped:
# conversion peaks ~3 GB, the 512-token compile ~6.5 GB; the cgroup cap kills only this job if it overshoots.
NPU_RUN      = $(NPU_CAPPED) embed_missing_npu.py --device $(EMBED_DEVICE)

setup-dedup:   # splink 4.x (the scripts use the 4.x Linker API; 5.x breaks them)
	cd dedup && uv venv .venv --python 3.13 && uv pip install --python .venv/bin/python "splink>=4,<5" pandas

dedup:   # splink duplicate-author fusion. Needs `make setup-dedup` once. Was applied once (commit 60cf6d3) then lost to a later data rebuild.
	cd dedup && for s in 01_prepare 02_link 03_report 04_verdict 05_merge; do $(CURDIR)/dedup/.venv/bin/python $$s.py || exit 1; done
	$(MAKE) taxonomy domains shares links collab

# OpenVINO env (NPU/iGPU steps). Override in .env.
NPU_PY      ?= $(CURDIR)/.venv-openvino/bin/python
NPU_LIB     ?=#   only for distros whose NPU compiler package is version-skewed; adds LD_LIBRARY_PATH when set
EIKOS_DIR   ?= $(CURDIR)/models/eikos
EMBED_DEVICE ?= NPU
LABEL_DEVICE ?= GPU
MEM_CAP     ?= 7G
# cgroup memory cap, only where systemd-run exists (kills just this job if it overshoots)
CAPPED       = $(if $(shell command -v systemd-run),systemd-run --user --scope --quiet -p MemoryMax=$(MEM_CAP) -p MemorySwapMax=0,)
NPU_CAPPED   = $(CAPPED) env $(if $(NPU_LIB),LD_LIBRARY_PATH=$(NPU_LIB):$$LD_LIBRARY_PATH,) $(NPU_PY) -u
RESCUE_EMB   = --works ../corpus/data/rescue_pool.jsonl --target-emb ../corpus/data/rescue_pool_bge.npy --target-ids ../corpus/data/rescue_pool_bge_ids.json --parts-dir caches/bge_partial_rescue

# ############################################################################################
# # KNOWN GAP: the rescue embedding (bge on the Intel NPU) and the Eikos-4B labelling (Intel   #
# # iGPU) are OpenVINO-ONLY. There is no CUDA/ROCm path yet, and `--device CPU` through        #
# # OpenVINO is untested and slow. Without Intel hardware, skip `make rescue` (RESCUE=0).      #
# ############################################################################################
# ---- Rescue pass (docs/METHODOLOGY.md, 'The rescue pass'): re-screen rule-excluded candidates with a text classifier.
# Runs after corpus steps 01-05 and before `split`. About 2 hours end to end (NPU embedding ~55 min, Eikos labelling ~45 min + band).
rescue: rescue-pool rescue-embed rescue-sample rescue-label rescue-train rescue-band rescue-apply

rescue-pool:    # R1: which excluded candidates deserve a second look
	$(P) $(PY) rescue_pool.py

rescue-embed:   # R2: bge-large embeddings of the pool on the NPU (3 capped stages + merge)
	$(P) $(NPU_CAPPED) embed_missing_npu.py --device $(EMBED_DEVICE) $(RESCUE_EMB) --buckets 128:32,256:16
	$(P) $(NPU_CAPPED) embed_missing_npu.py --device $(EMBED_DEVICE) $(RESCUE_EMB) --buckets 512:8 --tag long
	$(P) $(NPU_CAPPED) embed_missing_npu.py --device $(EMBED_DEVICE) $(RESCUE_EMB) --merge

rescue-sample:  # R3a: uniform + targeted sample of the pool to label
	$(P) $(PY) rescue_sample.py

rescue-label:   # R3a: label that sample with Eikos-4B (Intel iGPU)
	$(P) $(NPU_CAPPED) label_with_eikos.py --device $(LABEL_DEVICE) --in ../corpus/data/rescue_label_sample.jsonl --out ../corpus/data/rescue_label_sample_eikos.json

rescue-train:   # R3b: weighted logistic regression, thresholds from the uniform stratum, writes the uncertain band
	$(P) $(PY) rescue_train.py

rescue-band:    # R4a: Eikos adjudicates the uncertain band
	$(P) $(NPU_CAPPED) label_with_eikos.py --device $(LABEL_DEVICE) --in ../corpus/data/rescue_band.jsonl --out ../corpus/data/rescue_band_eikos.json

rescue-apply:   # R4b: write corpus/data/sweden_quantum_works_with_rescue.jsonl (+ manifest)
	$(P) $(PY) rescue_apply.py

embed-npu:
	$(P) $(NPU_RUN) --buckets 128:32,256:16
	$(P) $(NPU_RUN) --buckets 512:8 --tag long
	$(P) $(NPU_RUN) --merge

domains:
	$(P) $(PY) build_domain_specter.py

paper-counts:
	$(P) $(PY) build_paper_counts.py --works data/all_universities_works.jsonl --out data/paper_counts_geo.json

shares:
	$(P) $(PY) build_domain_shares.py --in data/people_viz.json --out data/people_viz.json

hindex:
	$(P) $(PY) fetch_global_h_index.py

patch-patents:
ifeq ($(PATENTS),skip)
	@echo "PATENTS=skip: skipping patch-patents"
else
	$(P) $(PY) patch_people_patents.py && $(PY) patch_patent_counts.py
endif

links:
	$(P) $(PY) build_patent_links.py

collab:
	$(P) $(PY) build_collaborations.py

stage:   # collect the 5 JSONs the app loads into STAGE_DIR (does not touch the app)
	mkdir -p $(STAGE_DIR)
	cd pipeline/data && cp people_viz.json people_taxonomy.json paper_counts_geo.json patent_links.json collaborations.json $(abspath $(STAGE_DIR))/
	@echo "Staged 5 JSON files in $(abspath $(STAGE_DIR)). Copy them to the app's public/data, or run: make publish"

publish:   # explicit: STAGE_DIR -> APP_PUBLIC (default is a sibling checkout ../open-alice/public/data)
	@test -d $(STAGE_DIR) || { echo "STAGE_DIR=$(STAGE_DIR) does not exist: run 'make stage' (or 'make all') first"; exit 1; }
	mkdir -p $(APP_PUBLIC)
	cd $(STAGE_DIR) && cp people_viz.json people_taxonomy.json paper_counts_geo.json patent_links.json collaborations.json $(APP_PUBLIC)/

# dedup rewrites people_viz.json (fuses split author records), then re-runs every id-keyed step, so links/collab come after it.
# `all` never touches the app: it ends with `stage`; run `make publish` explicitly.
all: people merge taxonomy domains paper-counts shares hindex patch-patents dedup stage
