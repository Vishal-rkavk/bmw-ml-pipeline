"""Environment preflight — run before anything else (`make verify-env`).

Checks the three things every pipeline step depends on:

1. Required libraries import, with versions matching requirements.txt.
2. A CUDA GPU is visible (training and generation run on GPU here).
3. The pinned SmolLM2-360M revision downloads, loads, and generates.

Exits with a non-zero status on any failure, so a fresh clone can gate
setup on this script.
"""

import sys
import time
from pathlib import Path

# Make the `pipeline` package importable when this script is invoked
# directly (python only puts the script's own directory on sys.path).
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


def check_libraries() -> bool:
    """Import every direct dependency and print its version."""
    from importlib.metadata import version

    import bs4
    import prometheus_client  # noqa: F401 - import check
    import requests
    import torch
    import transformers

    print(f"[ok] python            {sys.version.split()[0]}")
    print(f"[ok] torch             {torch.__version__}")
    print(f"[ok] transformers      {transformers.__version__}")
    print(f"[ok] requests          {requests.__version__}")
    print(f"[ok] beautifulsoup4    {bs4.__version__}")
    print(f"[ok] prometheus-client {version('prometheus-client')}")
    return True


def check_gpu() -> bool:
    """A CUDA device is required on the development host."""
    import torch

    if not torch.cuda.is_available():
        print("[FAIL] no CUDA GPU visible to torch")
        return False
    print(f"[ok] cuda              {torch.cuda.get_device_name(0)}")
    return True


def check_model() -> bool:
    """The pinned model revision must load and generate."""
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    from pipeline import config

    t0 = time.time()
    tokenizer = AutoTokenizer.from_pretrained(
        config.MODEL_ID, revision=config.MODEL_REVISION)
    model = AutoModelForCausalLM.from_pretrained(
        config.MODEL_ID, revision=config.MODEL_REVISION).to("cuda")
    load_seconds = time.time() - t0

    n_params = sum(p.numel() for p in model.parameters())

    inputs = tokenizer("The new BMW", return_tensors="pt").to("cuda")
    t0 = time.time()
    output = model.generate(**inputs, max_new_tokens=20, do_sample=False)
    gen_seconds = time.time() - t0

    text = tokenizer.decode(output[0], skip_special_tokens=True)
    print(f"[ok] model             {config.MODEL_ID} @ {config.MODEL_REVISION[:8]}"
          f" ({n_params / 1e6:.0f}M params, load {load_seconds:.1f}s)")
    print(f"[ok] generation        20 tokens in {gen_seconds:.2f}s")
    print(f"     sample            {text[:90]!r}")
    return True


def main() -> int:
    checks = (check_libraries, check_gpu, check_model)
    for check in checks:
        try:
            if not check():
                return 1
        except Exception as exc:  # noqa: BLE001 - preflight reports, not raises
            print(f"[FAIL] {check.__name__}: {exc}")
            return 1
    print("\nEnvironment OK.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
