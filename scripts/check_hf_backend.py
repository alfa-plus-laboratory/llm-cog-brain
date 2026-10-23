#!/usr/bin/env python3
"""Exercise the hf (transformers) activation backend without downloading anything.

Builds a random-initialised 3-layer GPT-2 and a word-level tokenizer for the
smoke stimuli under tmp/hf_check/, then checks: all three poolings run, sum ==
mean * n_tokens (i.e. pooling covers exactly the current sentence's tokens),
the full pipeline runs on it, and no CUDA context was created.

    data/venv/bin/python scripts/check_hf_backend.py
"""
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
os.environ["HF_HUB_OFFLINE"] = "1"

import numpy as np  # noqa: E402
from tokenizers import Tokenizer, models, pre_tokenizers  # noqa: E402
from transformers import GPT2Config, GPT2Model, PreTrainedTokenizerFast  # noqa: E402

from d1 import activations, config, data, pipeline  # noqa: E402


def main() -> int:
    out = ROOT / "tmp" / "hf_check"
    base = config.load(ROOT / "configs" / "smoke.yaml")
    ds = data.load(base)
    vocab = {"<unk>": 0, **{f"w{i}": i for i in range(1, base["dataset"]["synthetic"]["vocab"])}}
    tok = Tokenizer(models.WordLevel(vocab=vocab, unk_token="<unk>"))
    tok.pre_tokenizer = pre_tokenizers.WhitespaceSplit()
    PreTrainedTokenizerFast(tokenizer_object=tok, unk_token="<unk>").save_pretrained(out / "model")
    GPT2Model(GPT2Config(vocab_size=len(vocab), n_layer=3, n_embd=32, n_head=2, n_positions=256,
                         bos_token_id=0, eos_token_id=0)).save_pretrained(out / "model")

    cfg = config.load(ROOT / "configs" / "smoke.yaml", {
        "run_name": "hf-check", "output_dir": str(out / "out"),
        "model": {"backend": "hf", "name": str(out / "model"), "layers": [1, 2, 3],
                  "cache_dir": str(out / "cache")},
        "null_test": {"n_perm": 200}, "bootstrap": {"n_boot": 200}})
    ok = True
    feats = {}
    for pool in ("last", "mean", "sum"):
        feats[pool], _ = activations.get(ds, cfg, "contiguous", pooling=pool)
        finite = all(np.isfinite(f).all() for f in feats[pool].values())
        print(f"[hf-check] pooling={pool}: shapes {[f.shape for f in feats[pool].values()]} finite={finite}")
        ok &= finite
    same = np.allclose(feats["sum"][2], feats["mean"][2] * ds.n_words[:, None], atol=1e-4)
    print(f"[hf-check] sum == mean * n_tokens: {same}")
    ok &= same
    _, line, _ = pipeline.run(cfg)
    print(f"[hf-check] pipeline: {line[:120]}...")
    import torch
    cuda = torch.cuda.is_initialized()
    print(f"[hf-check] CUDA initialised: {cuda}  (CUDA_VISIBLE_DEVICES={os.environ.get('CUDA_VISIBLE_DEVICES')!r})")
    ok &= not cuda
    print(f"[hf-check] {'PASS' if ok else 'FAIL'}")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
