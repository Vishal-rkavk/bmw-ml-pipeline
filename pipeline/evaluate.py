"""Step 4: held-out perplexity, before vs. after fine-tuning.

Perplexity = exp(mean cross-entropy loss) on the eval split — the
standard automatic metric for language models; lower is better. Both
the base model (pinned revision) and the fine-tuned export are scored
on the same held-out blocks, written to artifacts/eval/results.json.
"""

import json
import math
import sys
import time
from datetime import datetime, timezone

import torch
from transformers import AutoModelForCausalLM

from pipeline import config, metrics
from pipeline.data import build_blocks, load_texts

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def eval_loss(model, blocks) -> float:
    """Mean cross-entropy loss over the eval blocks (all equal size)."""
    model.to(DEVICE)
    model.eval()
    losses = []
    with torch.no_grad():
        for i in range(0, len(blocks), config.EVAL_BATCH_SIZE):
            batch = blocks[i:i + config.EVAL_BATCH_SIZE].to(DEVICE)
            losses.append(model(input_ids=batch, labels=batch).loss.item())
    return sum(losses) / len(losses)


def main() -> int:
    config.ensure_dirs()
    started = time.time()

    blocks = build_blocks(load_texts("eval"))
    if len(blocks) == 0:
        print("[evaluate][warn] eval split is empty (very small corpus) — "
              "nothing to evaluate")
        return 0
    print(f"[evaluate] {len(blocks)} eval blocks ({blocks.numel():,} tokens)")

    models = {
        "before": AutoModelForCausalLM.from_pretrained(
            config.MODEL_ID, revision=config.MODEL_REVISION),
        "after": AutoModelForCausalLM.from_pretrained(config.MODEL_DIR),
    }
    results = {"eval_tokens": blocks.numel(), "phases": {}}
    for phase, model in models.items():
        loss = eval_loss(model, blocks)
        perplexity = math.exp(loss)
        results["phases"][phase] = {
            "loss": round(loss, 4),
            "perplexity": round(perplexity, 2),
        }
        print(f"[evaluate] {phase:>6}: loss {loss:.4f} | "
              f"perplexity {perplexity:.1f}")

    before = results["phases"]["before"]["perplexity"]
    after = results["phases"]["after"]["perplexity"]
    print(f"[evaluate] perplexity dropped {100 * (before - after) / before:.1f}% "
          f"({before:.1f} -> {after:.1f}) — the model adapted to BMW language")

    duration = time.time() - started
    results["evaluated_at"] = datetime.now(timezone.utc).isoformat(
        timespec="seconds")
    results["duration_s"] = round(duration, 1)
    out = config.EVAL_DIR / "results.json"
    out.write_text(json.dumps(results, indent=2) + "\n", encoding="utf-8")
    print(f"[evaluate] results written to {out}")

    metrics.push(
        "bmw-evaluate",
        values={"bmw_eval_duration_seconds": round(duration, 1)},
        series={
            "bmw_eval_perplexity": (
                "phase", {p: r["perplexity"]
                          for p, r in results["phases"].items()}),
            "bmw_eval_loss": (
                "phase", {p: r["loss"]
                          for p, r in results["phases"].items()}),
        },
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
