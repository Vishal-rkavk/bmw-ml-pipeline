# Decision Log

Completed, partially completed, and deliberately omitted items, with the
reasoning behind each. Interviewers care more about *why* than *what*.

## Completed

| Decision | Reasoning |
|---|---|
| SmolLM2-360M, pinned to a revision hash | Small, open (Apache-2.0), fast to fine-tune; a revision pin keeps the base weights reproducible (`main` moves, commits do not) |
| Full fine-tune, not LoRA | At 360M full fine-tuning is cheap on one GPU, and checkpoints stay plain model/optimizer state dicts — simpler to save and resume |
| Custom PyTorch loop, not HF Trainer | Checkpointing and resume are the core of the assignment; ~150 lines we fully control are easier to trust and explain than a framework flag |
| Sitemap for discovery | The only reliable machine-readable article index (RSS is a soft-404, listing pages are JS-rendered) |
| JSON-LD metadata + `<p>` heuristic for parsing | Structured metadata beats scraping visual markup; the paragraph rule was validated against multiple page templates |
| Per-article files as the corpus, `_rejected.json` as a cache | No separate manifest that could drift; rejected URLs are not refetched |
| `MAX_DOCS` = total corpus target | A per-run cap made re-runs grow the corpus by surprise; top-up semantics make re-runs exact no-ops |
| `sha256(url)`-based train/eval split | A document's split is a pure function of its URL: never changes as the corpus grows, no leakage possible |
| Atomic checkpoint save (`.tmp` + rename) | A rename within one filesystem is atomic, so even `kill -9` mid-save cannot corrupt a checkpoint |
| RNG state inside checkpoints | Restores model, optimizer, schedule **and** sampling exactly; no signal handler needed (`kill -9` is uncatchable anyway) |
| Pushgateway for metrics | Batch jobs push and exit; the gateway is the canonical bridge to pull-based Prometheus |
| One Pushgateway push per step | A push replaces the group, so scalars and labeled series go out in a single request; separate pushes would silently delete each other |
| Two alert rules with runbook annotations | `TrainingStalled` + `CollectionErrorRateHigh` cover the two failure modes that matter; each annotation says what to do |
| Grafana datasource + dashboard provisioned from the repo | Dashboard is reproducible from a fresh clone; nothing clicked in the UI |
| Makefile as orchestrator | Five idempotent steps; a workflow engine is unjustified at this scope |
| Binary launcher for the monitoring stack | Five small components run as pinned local binaries; no container runtime is required, and one script starts/stops them all |
| `config.py` + `BMW_*` env overrides | One source of configuration; overrides let smoke tests and demos run the same code on throwaway directories |

## Partially completed

| Item | State |
|---|---|
| Generation quality | Style is learned well (datelines, corporate diction); facts are invented — expected for a 360M base model with a style-adaptation objective |
| Dashboard screenshots | Taken manually via browser; instructions in `docs/results.md` |

