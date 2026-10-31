"""BMW text & ML pipeline package.

Five idempotent steps, each runnable on its own via `python -m pipeline.<step>`:

    collect    -> fetch BMW press articles into data/raw/
    prepare    -> clean texts and split into train/eval sets
    train      -> fine-tune the model (resumes from the latest checkpoint)
    evaluate   -> perplexity on held-out data, before vs. after fine-tuning
    generate   -> produce BMW-style text for a fixed set of prompts

All configuration lives in pipeline/config.py.
"""
