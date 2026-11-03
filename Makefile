# BMW Text & ML Pipeline — orchestration.
#
# Each target maps to one pipeline step (`python -m pipeline.<step>`).
# All steps are idempotent: safe to re-run, no duplicates are created.

PYTHON ?= python3

.PHONY: help install verify-env collect prepare train evaluate generate \
        run-all smoke monitoring-up monitoring-down

help: ## Show available targets
	@grep -E '^[a-zA-Z_-]+:.*?## ' $(MAKEFILE_LIST) | \
		awk 'BEGIN {FS = ":.*?## "}; {printf "  %-18s %s\n", $$1, $$2}'

install: ## Install pinned Python dependencies
	$(PYTHON) -m pip install -r requirements.txt

verify-env: ## Preflight: libraries, GPU, and pinned model access
	$(PYTHON) scripts/verify_env.py

collect: ## Step 1: fetch BMW press articles into data/raw/
	$(PYTHON) -m pipeline.collect

prepare: ## Step 2: clean texts and split into train/eval sets
	$(PYTHON) -m pipeline.prepare

train: ## Step 3: fine-tune (auto-resumes from latest checkpoint)
	$(PYTHON) -m pipeline.train

evaluate: ## Step 4: held-out perplexity, before vs. after fine-tuning
	$(PYTHON) -m pipeline.evaluate

generate: ## Step 5: generate BMW-style text for fixed prompts
	$(PYTHON) -m pipeline.generate

run-all: collect prepare train evaluate generate ## Full pipeline, in order

smoke: ## Fast end-to-end test using tiny settings (same code paths)
	rm -rf smoke-data smoke-artifacts
	BMW_DATA_DIR=smoke-data BMW_ARTIFACTS_DIR=smoke-artifacts \
	BMW_MAX_DOCS=3 BMW_MAX_STEPS=5 BMW_CHECKPOINT_EVERY=2 \
	BMW_LOG_EVERY=1 BMW_BATCH_SIZE=2 BMW_EVAL_BATCH_SIZE=2 \
	BMW_METRICS_DISABLE=1 \
	$(MAKE) run-all

monitoring-up: ## Start Prometheus, Pushgateway, Alertmanager, Grafana
	./monitoring/up.sh

monitoring-down: ## Stop the monitoring stack
	./monitoring/down.sh
