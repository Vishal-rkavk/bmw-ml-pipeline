"""Training data: tokenize a split and cut it into fixed-length blocks.

The whole split is tokenized once, joined into one stream and sliced
into BLOCK_SIZE blocks — every token is training signal, no padding.
Document boundaries are not marked: for style adaptation on one
homogeneous corpus a plain stream is sufficient.
"""

import json

import torch
from transformers import AutoTokenizer

from pipeline import config


def load_texts(split: str) -> list[str]:
    """All document texts of one split ("train" or "eval")."""
    path = config.PROCESSED_DIR / f"{split}.jsonl"
    texts = []
    for line in path.read_text(encoding="utf-8").splitlines():
        texts.append(json.loads(line)["text"])
    return texts


def build_blocks(texts: list[str]) -> torch.Tensor:
    """Tokenize all texts and slice into an (n_blocks, BLOCK_SIZE) tensor."""
    if not texts:  # empty split (possible with tiny corpora): no blocks
        return torch.empty(0, config.BLOCK_SIZE, dtype=torch.long)
    tokenizer = AutoTokenizer.from_pretrained(
        config.MODEL_ID, revision=config.MODEL_REVISION)
    encoded = tokenizer(texts)["input_ids"]
    ids = [token_id for doc in encoded for token_id in doc]
    n_blocks = len(ids) // config.BLOCK_SIZE
    if n_blocks == 0:
        return torch.empty(0, config.BLOCK_SIZE, dtype=torch.long)
    stream = torch.tensor(ids[: n_blocks * config.BLOCK_SIZE],
                          dtype=torch.long)
    return stream.view(n_blocks, config.BLOCK_SIZE)
