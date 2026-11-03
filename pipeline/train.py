"""Step 3: fine-tune the model with a plain PyTorch training loop.

Checkpoints (artifacts/checkpoints/step-N.pt) contain the model,
optimizer, scheduler, global step and torch RNG states. They are
written to a .tmp file and renamed into place — a rename is atomic, so
a crash (even kill -9) can never leave a half-written checkpoint. On
start, the newest checkpoint is loaded and training continues from it;
model state, optimizer, schedule and RNG are restored exactly.
"""

import sys
import time

import torch
from torch.utils.data import DataLoader, TensorDataset
from transformers import (AutoModelForCausalLM, AutoTokenizer,
                          get_linear_schedule_with_warmup)

from pipeline import config, metrics
from pipeline.data import build_blocks, load_texts

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
TOKENS_PER_STEP = config.BLOCK_SIZE * config.BATCH_SIZE


def checkpoint_path(step: int):
    """Path of the checkpoint file for a training step."""
    return config.CHECKPOINT_DIR / f"step-{step:06d}.pt"


def latest_checkpoint():
    """Newest checkpoint file, or None if there is none."""
    paths = sorted(config.CHECKPOINT_DIR.glob("step-*.pt"))
    return paths[-1] if paths else None


def save_checkpoint(step, model, optimizer, scheduler) -> float:
    """Write one checkpoint atomically; return seconds taken."""
    started = time.time()
    payload = {
        "step": step,
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict(),
        "rng_torch": torch.get_rng_state(),
        "rng_cuda": (torch.cuda.get_rng_state_all()
                     if DEVICE == "cuda" else None),
    }
    tmp_path = config.CHECKPOINT_DIR / f"step-{step:06d}.pt.tmp"
    torch.save(payload, tmp_path)
    tmp_path.rename(checkpoint_path(step))
    old_files = sorted(config.CHECKPOINT_DIR.glob("step-*.pt"))
    for old in old_files[:-config.KEEP_CHECKPOINTS]:
        old.unlink()
    return time.time() - started


def main() -> int:
    config.ensure_dirs()
    torch.manual_seed(config.SEED)
    if DEVICE == "cuda":
        torch.cuda.manual_seed_all(config.SEED)

    tokenizer = AutoTokenizer.from_pretrained(
        config.MODEL_ID, revision=config.MODEL_REVISION)
    model = AutoModelForCausalLM.from_pretrained(
        config.MODEL_ID, revision=config.MODEL_REVISION).to(DEVICE)

    blocks = build_blocks(load_texts("train"))
    steps_per_epoch = len(blocks) // config.BATCH_SIZE
    if steps_per_epoch == 0:
        print(f"[train][FAIL] corpus too small: {len(blocks)} blocks "
              f"for batch size {config.BATCH_SIZE}")
        return 1
    loader = DataLoader(TensorDataset(blocks), batch_size=config.BATCH_SIZE,
                        shuffle=True, drop_last=True)
    print(f"[train] {len(blocks)} blocks x {config.BLOCK_SIZE} tokens | "
          f"{steps_per_epoch} steps/epoch | {DEVICE} | "
          f"{sum(p.numel() for p in model.parameters()) / 1e6:.0f}M params")

    optimizer = torch.optim.AdamW(model.parameters(),
                                  lr=config.LEARNING_RATE,
                                  weight_decay=config.WEIGHT_DECAY)
    scheduler = get_linear_schedule_with_warmup(
        optimizer, config.WARMUP_STEPS, config.MAX_STEPS)

    # --- resume: restore everything from the newest checkpoint -------
    step = 0
    ckpt = latest_checkpoint()
    if ckpt is not None:
        payload = torch.load(ckpt, map_location="cpu")
        model.load_state_dict(payload["model"])
        optimizer.load_state_dict(payload["optimizer"])
        scheduler.load_state_dict(payload["scheduler"])
        torch.set_rng_state(payload["rng_torch"])
        if DEVICE == "cuda" and payload["rng_cuda"] is not None:
            torch.cuda.set_rng_state_all(payload["rng_cuda"])
        step = payload["step"]
        print(f"[train] resumed from {ckpt.name} (step {step})")

    model.train()

    window_loss = 0.0
    window_seconds = 0.0
    while step < config.MAX_STEPS:
        for batch in loader:
            tokens = batch[0].to(DEVICE)

            started = time.time()
            with torch.autocast(device_type=DEVICE, dtype=torch.bfloat16,
                                enabled=config.USE_BF16 and DEVICE == "cuda"):
                loss = model(input_ids=tokens, labels=tokens).loss
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(),
                                           config.GRADIENT_CLIP)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()
            step += 1
            window_seconds += time.time() - started
            window_loss += loss.item()

            if step % config.LOG_EVERY == 0:
                per_step = window_seconds / config.LOG_EVERY
                avg_loss = window_loss / config.LOG_EVERY
                print(f"[train] step {step}/{config.MAX_STEPS} | "
                      f"loss {avg_loss:.3f} | {per_step:.2f}s/step | "
                      f"{TOKENS_PER_STEP / per_step / 1000:.0f}k tok/s")
                metrics.push("bmw-train", values={
                    "bmw_train_loss": round(avg_loss, 4),
                    "bmw_train_step_duration_seconds": round(per_step, 3),
                    "bmw_train_tokens_per_second": round(TOKENS_PER_STEP / per_step),
                    "bmw_train_global_step": step,
                }, grouping={"run": config.RUN_ID, "kind": "step"})
                window_loss, window_seconds = 0.0, 0.0

            if step % config.CHECKPOINT_EVERY == 0:
                seconds = save_checkpoint(step, model, optimizer, scheduler)
                print(f"[train] checkpoint step-{step:06d} saved "
                      f"in {seconds:.1f}s")
                metrics.push("bmw-train", values={
                    "bmw_train_checkpoint_duration_seconds": round(seconds, 2)},
                    grouping={"run": config.RUN_ID, "kind": "checkpoint"})

            if step >= config.MAX_STEPS:
                break

    if not checkpoint_path(step).exists():
        seconds = save_checkpoint(step, model, optimizer, scheduler)
        print(f"[train] final checkpoint step-{step:06d} saved "
              f"in {seconds:.1f}s")
        metrics.push("bmw-train", values={
            "bmw_train_checkpoint_duration_seconds": round(seconds, 2)},
            grouping={"run": config.RUN_ID, "kind": "checkpoint"})

    model.save_pretrained(config.MODEL_DIR)
    tokenizer.save_pretrained(config.MODEL_DIR)
    print(f"[train] done: {step} steps | model exported to {config.MODEL_DIR}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
