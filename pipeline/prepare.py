"""Step 2: prepare training/evaluation data from the collected corpus.

For every article in data/raw/:
- assemble the document text as "<title>\n\n<body>" — the title gives
  the model a natural conditioning prefix (the generation prompts in
  step 5 are press-release openings);
- apply light cleaning only (the collector already extracts clean
  prose): normalize whitespace and drop documents too short to be a
  useful training signal;
- split deterministically: a document goes to the EVAL set iff
  int(sha256(url), 16) % 100 < EVAL_PERCENT. Hashing the stable URL —
  not the mutable text, not the file listing order — means:
  * a document always lands in the same split, no matter how often
    prepare re-runs or how much the corpus grows afterwards;
  * no leakage is possible: a document's split is a pure function of
    its own URL, so later-collected articles can never migrate into
    an evaluation set that was already used.

Outputs: data/processed/train.jsonl and eval.jsonl, one JSON object
per line: {"id", "date", "text"}. Documents are written in a stable
order (date, then id) with fixed JSON formatting, so identical inputs
produce byte-identical outputs — re-running prepare is a no-op by diff.

Note: with very small corpora (e.g. `make smoke`) the eval split can
end up empty by chance; evaluate.py handles that by warning and
skipping the metric.
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

    metrics.push_series("bmw-prepare", "bmw_docs_split", "split",
                        {name: len(docs) for name, docs in splits.items()})
    metrics.push_metrics("bmw-prepare", {
        "bmw_prepare_duration_seconds": round(duration, 1)})
    return 0 if splits["train"] else 1


if __name__ == "__main__":
    sys.exit(main())
