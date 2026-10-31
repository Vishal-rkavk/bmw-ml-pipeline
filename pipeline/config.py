"""Central configuration for the pipeline.

Every tunable knob lives in this one file so the pipeline can be
reconfigured without touching code. Selected values can be overridden
with BMW_* environment variables; that is how `make smoke` and the
resume demo run fast, small passes through the exact same code paths.

Run `python -m pipeline.config` to print the effective configuration.
"""

import os
from pathlib import Path


def _int(name: str, default: int) -> int:
    """Read an integer override from the environment, else the default."""
    return int(os.environ.get(name, default))


def _str(name: str, default: str) -> str:
    """Read a string override from the environment, else the default."""
    return os.environ.get(name, default)


# --------------------------------------------------------------------------
# Paths (all pipeline outputs live under data/ and artifacts/, which are
# gitignored: they are regenerable and too large for git).
# --------------------------------------------------------------------------
REPO_ROOT = Path(__file__).resolve().parent.parent

RAW_DIR = REPO_ROOT / "data" / "raw"                # one JSON per article
PROCESSED_DIR = REPO_ROOT / "data" / "processed"    # train.jsonl / eval.jsonl
CHECKPOINT_DIR = REPO_ROOT / "artifacts" / "checkpoints"
EVAL_DIR = REPO_ROOT / "artifacts" / "eval"
GENERATIONS_DIR = REPO_ROOT / "artifacts" / "generations"
LOGS_DIR = REPO_ROOT / "artifacts" / "logs"

# --------------------------------------------------------------------------
# Step 1: data collection (pipeline/collect.py)
# --------------------------------------------------------------------------
SITEMAP_URL = "https://newsroom.bmwgroup.com/global/sitemap.xml"
BASE_URL = "https://newsroom.bmwgroup.com"

MAX_DOCS = _int("BMW_MAX_DOCS", 400)     # TOTAL corpus target: collection
                                         # tops up to this; re-runs at the
                                         # target are no-ops
MIN_DATE = _str("BMW_MIN_DATE", "2023-01-01")  # only keep articles >= this
REQUEST_DELAY_S = 1.0                    # politeness delay between requests
REQUEST_TIMEOUT_S = 15.0
MAX_RETRIES = 3                          # per-article retries with backoff
MIN_TEXT_CHARS = 500                     # articles below this are discarded

# Sitemap entries that look like articles but are static pages.
# "de"/"en" are the locale landing pages, which match the article URL
# pattern but are not articles.
EXCLUDE_PATHS = (
    "media-library",
    "legal-notice",
    "privacy-policy",
    "cookies",
    "e-mail-subscription",
    "de",
    "en",
)

# --------------------------------------------------------------------------
# Step 2: preparation and split (pipeline/prepare.py)
# --------------------------------------------------------------------------
EVAL_PERCENT = 20  # a document goes to eval if hash(doc_id) % 100 < 20.
                   # Hash-based => a document's split never changes when
                   # the corpus grows, and re-runs are deterministic.

# --------------------------------------------------------------------------
# Model (used by training, evaluation and generation)
# --------------------------------------------------------------------------
# Pinned to an exact revision: `main` moves, revisions don't.
MODEL_ID = "HuggingFaceTB/SmolLM2-360M"
MODEL_REVISION = "f8027fd0eaeea54caa13c31d31b9fdc459c38b49"
BLOCK_SIZE = 512  # training sequence length in tokens

# --------------------------------------------------------------------------
# Step 3: training (pipeline/train.py)
# --------------------------------------------------------------------------
SEED = 42
MAX_STEPS = _int("BMW_MAX_STEPS", 300)
BATCH_SIZE = _int("BMW_BATCH_SIZE", 16)
LEARNING_RATE = 1e-4
WEIGHT_DECAY = 0.01
WARMUP_STEPS = 30
LOG_EVERY = _int("BMW_LOG_EVERY", 10)          # steps between log lines
CHECKPOINT_EVERY = _int("BMW_CHECKPOINT_EVERY", 50)  # steps between ckpts
KEEP_CHECKPOINTS = 2                           # older checkpoints are deleted
USE_BF16 = True                                # A100 supports bfloat16

# --------------------------------------------------------------------------
# Step 4: evaluation (pipeline/evaluate.py)
# --------------------------------------------------------------------------
EVAL_BATCH_SIZE = _int("BMW_EVAL_BATCH_SIZE", 8)

# --------------------------------------------------------------------------
# Step 5: generation (pipeline/generate.py)
# --------------------------------------------------------------------------
GEN_TEMPERATURE = 0.8
GEN_TOP_P = 0.95
GEN_MAX_NEW_TOKENS = 120

# --------------------------------------------------------------------------
# Monitoring (used by pipeline/metrics.py; stack starts with
# `make monitoring-up`). Pushes are best-effort: a step never fails
# just because monitoring is down.
# --------------------------------------------------------------------------
PUSHGATEWAY_URL = _str("BMW_PUSHGATEWAY_URL", "http://localhost:9091")


def ensure_dirs() -> None:
    """Create every directory the pipeline writes to (idempotent)."""
    for d in (RAW_DIR, PROCESSED_DIR, CHECKPOINT_DIR, EVAL_DIR,
              GENERATIONS_DIR, LOGS_DIR):
        d.mkdir(parents=True, exist_ok=True)


if __name__ == "__main__":
    # `python -m pipeline.config` prints the effective configuration.
    for key, value in sorted(vars().items()):
        if key.isupper() and not key.startswith("_"):
            print(f"{key} = {value!r}")
