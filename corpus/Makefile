PYTHON = .venv/bin/python
SRC    = src

.PHONY: help
help:
	@echo "Run in this order. 01-05 need OPENALEX_API_KEY; 06-10 are local."
	@echo ""
	@echo "  make snapshot      01  Download the OpenAlex topic taxonomy"
	@echo "  make topics        02  Pick the quantum topic candidates"
	@echo "  make download      03  Download candidate works, derive keywords"
	@echo "  make reconcile     04  Recover works lost to source regressions"
	@echo "  make filter        05  Apply the inclusion rule -> the corpus"
	@echo "  make core          06  Build the explicitly-quantum core set"
	@echo "  make embed         07  Embed candidate titles and abstracts"
	@echo "  make outliers      08  Score, cluster, and rank against the core"
	@echo "  make affinities    09  Score works against the quantum area anchors"
	@echo "  make report        10  Render quantum_core_review_report.html"
	@echo ""
	@echo "  make corpus        01-05"
	@echo "  make review        06-10"
	@echo "  make all          01-10"

.PHONY: snapshot topics download reconcile filter

snapshot:
	$(PYTHON) $(SRC)/01_snapshot_openalex_topics.py

topics: data/openalex_topics_snapshot.jsonl config/sweden_quantum_discovery_terms.json
	$(PYTHON) $(SRC)/02_inventory_quantum_topic_candidates.py

download: data/sweden_quantum_topic_candidates.csv config/sweden_quantum_scope.json config/sweden_quantum_keyword_derivation.json config/sweden_quantum_paper_filter.json
	$(PYTHON) $(SRC)/03_download_sweden_quantum_candidates.py

reconcile: data/sweden_quantum_candidate_union.jsonl data/sweden_quantum_derived_keywords.csv
	$(PYTHON) $(SRC)/04_reconcile_sweden_quantum_source_regressions.py

filter: data/sweden_quantum_candidate_union.jsonl data/sweden_quantum_source_reconciliations.jsonl config/sweden_quantum_record_gates.json
	$(PYTHON) $(SRC)/05_filter_sweden_quantum_candidates.py

.PHONY: core embed outliers affinities report

core: data/sweden_quantum_candidate_union.jsonl data/sweden_quantum_topic_candidates.csv config/sweden_quantum_core_and_outliers.json
	$(PYTHON) $(SRC)/06_build_quantum_core_set.py

embed: data/sweden_quantum_candidate_features.jsonl data/sweden_quantum_core_manifest.json
	$(PYTHON) $(SRC)/07_embed_candidate_texts.py

outliers: data/sweden_quantum_candidate_features.jsonl data/sweden_quantum_embeddings_manifest.json data/sweden_quantum_candidate_audit.csv
	$(PYTHON) $(SRC)/08_detect_quantum_outliers.py

affinities: data/sweden_quantum_embeddings_manifest.json data/sweden_quantum_outlier_scores.csv config/sweden_quantum_area_anchors.json
	$(PYTHON) $(SRC)/09_score_quantum_area_affinities.py

report: data/sweden_quantum_outlier_scores.csv data/sweden_quantum_outlier_clusters.csv data/sweden_quantum_area_affinities.csv
	$(PYTHON) $(SRC)/10_report_quantum_core_review.py

.PHONY: corpus review all
corpus: snapshot topics download reconcile filter
review: core embed outliers affinities report
all: corpus review
