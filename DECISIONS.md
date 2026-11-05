# Decision Log

Completed, partially completed and deliberately omitted items, with the
reasoning behind each. Interviewers care more about *why* than *what*.

## Completed

| Decision | Reasoning |
|---|---|
| SmolLM2-360M, pinned to a revision hash | Small, open (Apache-2.0), fast to fine-tune|
| 1,000 English-language BMW Group articles, cutoff 2023-01-01 | The English `/global/` newsroom covers BMW, MINI, Rolls-Royce and Motorrad; 1,000 documents are enough for a 360M model without making collection or training a bottleneck |
| Full fine-tune, not LoRA | At 360M full fine-tuning is cheap on one GPU, and checkpoints stay plain model/optimizer state dicts — simpler to save and resume |
| Custom PyTorch loop | Checkpointing and resume are the core of the assignment; We have full control  |
| `MAX_DOCS` = total corpus target | can be easily increased |
| `sha256(url)`-based train/eval split | A document's split is a pure function of its URL: never changes as the corpus grows, no leakage possible |
| Atomic checkpoint save (`.tmp` + rename) | A rename within one filesystem is atomic, so even `kill -9` mid-save cannot corrupt a checkpoint |
| Keep the last 2 checkpoints | Bounds disk use while keeping one older checkpoint available for a manual rollback; the newest checkpoint is still enough for automatic resume |
| Pushgateway for metrics | Batch jobs push and exit; the gateway is the canonical bridge to pull-based Prometheus |
| Two alert rules with runbook annotations | `TrainingStalled` + `CollectionErrorRateHigh` cover the two failure modes that matter; each annotation says what to do |
| Grafana datasource + dashboard provisioned from the repo | Dashboard is reproducible from a fresh clone; nothing clicked in the UI |
| Makefile as orchestrator | Five idempotent steps; |
| Binary launcher for the monitoring stack | Five small components run as pinned local binaries; no container runtime is required, and one script starts/stops them all |
| Defaults match the reported run (`MAX_DOCS=1000`, `MAX_STEPS=800`) | `make run-all` reproduces the documented configuration; `make smoke` still provides a fast small pass through the same code paths |


## Partially completed

| Item | State |
|---|---|
| Generation quality | Style is learned well (datelines, corporate diction); facts are invented — expected for a 360M base model with a style-adaptation objective |
| Evaluation depth | Perplexity is the only automatic metric. It measures fit to the corpus distribution, not factual correctness or helpfulness; the side-by-side generations are qualitative evidence|
| Overfitting mitigation | Train loss (~1.04) is much lower than held-out loss (~2.48) after ~6 epochs. Early stopping on eval loss and a held-out-based checkpoint policy are the obvious next steps, but were left out of the timebox |
| Dependency pinning | Python dependencies and monitoring binaries are pinned; `torch` is intentionally not pinned because the development host provides a CUDA-matched build. Improvement could be containarisation|
| Exact corpus reproducibility | The collector is deterministic for a fixed sitemap and corpus, but the live BMW newsroom changes. A fresh clone can reproduce the code path and configuration, not the exact 1,000 documents or their hash split |

## Deliberately omitted

| Item | Why it was left out |
|---|---|
| Model serving / inference API | The assignment asks to adapt and evaluate a model, not deploy it. A serving stack would add operational surface without strengthening the pipeline story |
| CI (e.g. GitHub Actions) | Timeboxed. `make smoke` is the local equivalent and exercises all five steps in throwaway directories |
| LoRA / PEFT | Full fine-tuning is cheap at 360M and keeps checkpoints as plain state dicts, which makes the resume mechanism easier to demonstrate |
| Containerized deployment | The monitoring stack runs as pinned local binaries and the pipeline runs in the host Python environment; Docker would add a build layer without improving the demo |
| Online / scheduled ingestion | The pipeline is deliberately batch and manually triggered; continuous ingestion needs scheduling, freshness semantics and a different storage design |
 
