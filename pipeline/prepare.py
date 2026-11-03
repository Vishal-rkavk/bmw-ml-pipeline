"""Step 2: clean the corpus and split it into train/eval JSONL files.

Documents are "<title>\\n\\n<body>": the title gives the model a natural
conditioning prefix. A document goes to eval iff hash(url) % 100 <
EVAL_PERCENT — a pure function of its own URL, so splits never change
on re-runs and documents can never leak between them. Output is
written in a stable order with fixed formatting: re-runs are
byte-identical.
"""

import hashlib
import json
import re
import sys
import time

from pipeline import config, metrics


def clean_text(text: str) -> str:
    """Normalize whitespace; the collector already delivers clean prose."""
    text = text.strip()
    text = re.sub(r" {2,}", " ", text)      # leftover double spaces
    text = re.sub(r"\n{3,}", "\n\n", text)  # blank-line runs
    return text


def split_for(url: str) -> str:
    """Deterministic train/eval assignment for one document URL."""
    digest = hashlib.sha256(url.encode("utf-8")).hexdigest()
    return "eval" if int(digest, 16) % 100 < config.EVAL_PERCENT else "train"


def main() -> int:
    config.ensure_dirs()
    started = time.time()

    raw_files = sorted(p for p in config.RAW_DIR.glob("*.json")
                       if not p.name.startswith("_"))
    if not raw_files:
        print("[prepare][FAIL] no collected documents in data/raw/ — "
              "run `make collect` first")
        return 1
    print(f"[prepare] {len(raw_files)} documents in corpus")

    splits: dict[str, list[dict]] = {"train": [], "eval": []}
    for path in raw_files:
        doc = json.loads(path.read_text(encoding="utf-8"))
        text = clean_text(f"{doc['title']}\n\n{doc['text']}")
        if len(text) < config.MIN_TEXT_CHARS:
            continue
        splits[split_for(doc["url"])].append(
            {"id": doc["url"], "date": doc["date"], "text": text})

    for split, docs in splits.items():
        docs.sort(key=lambda d: (d["date"], d["id"]))  # stable order
        out_path = config.PROCESSED_DIR / f"{split}.jsonl"
        with out_path.open("w", encoding="utf-8") as fh:
            for doc in docs:
                fh.write(json.dumps(doc, ensure_ascii=False) + "\n")
        chars = sum(len(doc["text"]) for doc in docs)
        print(f"[prepare] {split}: {len(docs)} docs | {chars:,} chars "
              f"| ~{chars // 4:,} tokens (rule of thumb: 4 chars/token)")

    duration = time.time() - started
    print(f"[prepare] done in {duration:.1f}s "
          f"({len(splits['train'])} train / {len(splits['eval'])} eval)")

    metrics.push(
        "bmw-prepare",
        values={"bmw_prepare_duration_seconds": round(duration, 1)},
        series={"bmw_docs_split": ("split",
                                   {name: len(docs) for name, docs in splits.items()})},
    )
    return 0 if splits["train"] else 1


if __name__ == "__main__":
    sys.exit(main())
