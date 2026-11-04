# BMW Text & ML Pipeline

Collects BMW press releases from the BMW Group newsroom, fine-tunes a
small open model (SmolLM2-360M) on the corpus, evaluates it on held-out
data, and reports every step's metrics to a Grafana dashboard that is
provisioned from this repository.

```
  make collect -> make prepare -> make train -> make evaluate -> make generate
   data/raw/      data/processed/  artifacts/     artifacts/eval/  artifacts/generations/
                                    checkpoints,
                                    model/
        |
        +-- every step pushes metrics --> Pushgateway --> Prometheus --> Grafana
                                                             |
                                                             +--> Alertmanager
```

| Step | Command | What it does | Output |
|---|---|---|---|
| 1 | `make collect` | sitemap -> articles, dedupe by URL + content hash | `data/raw/*.json` |
| 2 | `make prepare` | clean, deterministic hash split 80/20 | `data/processed/{train,eval}.jsonl` |
| 3 | `make train` | fine-tune, atomic checkpoints, auto-resume | `artifacts/checkpoints/`, `artifacts/model/` |
| 4 | `make evaluate` | held-out perplexity, before vs. after | `artifacts/eval/results.json` |
| 5 | `make generate` | BMW prompts, fine-tuned vs. base model | `artifacts/generations/generations.md` |

## Requirements

- Python 3.10+; `make install` (torch is provided by the host environment)
- A CUDA GPU is recommended (an A100 was used here); CPU works but is slow
- ~10 GB free disk (each checkpoint is 4.3 GB, the last 2 are kept)
- Internet for the first run (BMW newsroom + Hugging Face model download)

## Quickstart

```bash
make install
make verify-env          # libraries, GPU, pinned model revision
make collect             # first run ~10 min (400 articles, polite delays); re-runs ~1 s
make prepare
make monitoring-up       # Prometheus, Alertmanager, Grafana, Pushgateway
make train               # ~90 s on an A100; safe to interrupt and re-run
make evaluate
make generate
```

Then open Grafana at <http://localhost:3000> (`admin/admin`) -> folder
**BMW Pipeline**.

## Interruption and resume

Training checkpoints every `CHECKPOINT_EVERY` steps. To resume after a
crash, kill, or reboot: just run `make train` again. It loads the newest
`artifacts/checkpoints/step-*.pt` and continues.

Prove it end-to-end with the automated demo:

```bash
./scripts/demo_resume.sh
```

It trains, waits for two checkpoints, sends `kill -9`, restarts, and
prints the resume evidence.

## Monitoring

| Component | URL | Purpose |
|---|---|---|
| Grafana | <http://localhost:3000> | dashboard (admin/admin) |
| Prometheus | <http://localhost:9090> | metrics storage, `/alerts` |
| Alertmanager | <http://localhost:9093> | firing alerts |
| Pushgateway | <http://localhost:9091> | batch-job metrics inbox |

Check that it works:

```bash
curl -s localhost:9090/-/healthy      # Prometheus Server is Healthy.
curl -s localhost:9091/-/ready        # OK
curl -s localhost:9093/-/healthy      # OK
curl -s localhost:3000/api/health     # {"database":"ok",...}
```

Prometheus targets (<http://localhost:9090/targets>) must both be UP.
Useful queries in the Prometheus UI: `bmw_corpus_docs`,
`bmw_train_loss`, `bmw_train_tokens_per_second`, `bmw_eval_perplexity`.

### Alerts

| Alert | Fires when | First response |
|---|---|---|
| `BmwTrainingStalled` | no training metric push for 5+ min | check the training log, run `make train` (it resumes) |
| `BmwCollectionErrorRateHigh` | >10% of the last collection's fetches failed | check connectivity / the site; re-run `make collect` |

Both rules live in `monitoring/alerts.yml` and are versioned.

### Metrics pushed by each step

| Step | Metrics |
|---|---|
| collect | `bmw_collect_docs{status=ok\|duplicate\|error\|skipped}`, `bmw_collect_new_docs`, `bmw_corpus_docs`, `bmw_collect_duration_seconds` |
| prepare | `bmw_docs_split{split=train\|eval}`, `bmw_prepare_duration_seconds` |
| train | `bmw_train_loss`, `bmw_train_global_step`, `bmw_train_step_duration_seconds`, `bmw_train_checkpoint_duration_seconds`, `bmw_train_tokens_per_second` |
| evaluate | `bmw_eval_perplexity{phase=before\|after}`, `bmw_eval_loss`, `bmw_eval_duration_seconds` |
| generate | `bmw_generation_prompts`, `bmw_generation_duration_seconds` |

Metrics pushes are best-effort: with the stack down the pipeline still
runs, it just warns once.

## Configuration

All knobs are constants in `pipeline/config.py`; selected ones can be
overridden with `BMW_*` environment variables (used by `make smoke` and
the demo scripts to run throwaway copies of the same code paths).

```bash
make smoke    # full pipeline on 3 articles / 5 steps in throwaway dirs
BMW_MAX_STEPS=600 make train      # train longer
BMW_RUN_ID=exp2 make train        # separate loss curve for a new run
```

Monitoring-specific overrides: `BMW_RUN_ID` labels a training run's
series (default `main`; keep it stable across a resume, change it for a
new experiment), and `BMW_METRICS_DISABLE=1` turns all metric pushes off
(`make smoke` sets it so throwaway runs never overwrite the dashboard).

## Results

See [docs/results.md](docs/results.md). Summary: 400 articles, 306/94
train/eval split, perplexity on held-out data **16.7 -> 13.2 (-21%)**,
training interrupted with `kill -9` and resumed from checkpoint.

## Repository layout

```
pipeline/       config, metrics, collect, prepare, data, train, evaluate, generate
monitoring/     Prometheus/Alertmanager/Grafana configs + launcher scripts
scripts/        verify_env.py (preflight), demo_resume.sh (resume proof)
docs/           results.md, interview_prep.md, presentation/
Makefile        one target per step; `make help` lists them
```

## Limitations and deliberate omissions

See [DECISIONS.md](DECISIONS.md). In short: English BMW articles only, a
360M base model (style adaptation, not a knowledge base), no early
stopping, no CI, no model serving. The monitoring stack runs as pinned
local binaries started by `make monitoring-up`.
