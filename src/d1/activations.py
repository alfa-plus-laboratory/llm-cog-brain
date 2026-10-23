"""LLM activations per sample, pooled over the sample's tokens.

Pooling (over the tokens of the *current* sample only; context tokens are fed
to the model but never pooled):

    last  hidden state of the sample's final token
    mean  average over the sample's tokens
    sum   sum over the sample's tokens -- scales with token count, i.e. it
          carries sentence length by construction; that is exactly why the
          word-rate baseline has to be in the joint model.

Cache key is (model, layer, pooling, split) as the task book requires, plus
the context mode and a hash of the stimuli. Activations do not depend on the
split -- inference sees no labels -- so the split is in the key for provenance
only; a second split reuses the first one's array via a hard link.

Backends: ``synthetic`` (a fixed random toy LM, numpy only -- used by --smoke)
and ``hf`` (transformers, forced onto CPU, inference only).
"""
from __future__ import annotations

import json
import os
import re
import time
from pathlib import Path

import numpy as np

from .config import POOLINGS
from .data import Dataset

_TOY_DIM, _TOY_LAYERS, _TOY_SEED = 48, 6, 20261016


def toy_embeddings(vocab: int, dim: int = _TOY_DIM) -> np.ndarray:
    E = np.random.default_rng(_TOY_SEED).standard_normal((vocab, dim)) / np.sqrt(dim)
    E[0] = 0.0                                    # id 0 = filler token
    return E


def _toy_lm(vocab: int):
    """A fixed, untrained, causal toy LM: embeddings + 6 layers of mixing.

    Each layer adds to a residual stream a nonlinear transform of the token's
    own state plus a linear read of the causal mean of all earlier states (a
    crude stand-in for attention), so the last token carries its context and
    deeper layers drift further from the raw embeddings. Weights are fixed by a seed: this is a
    stand-in for "a frozen pretrained model", nothing is fit.
    """
    E = toy_embeddings(vocab)
    rng = np.random.default_rng(_TOY_SEED + 1)
    Ws = [rng.standard_normal((_TOY_DIM, _TOY_DIM)) * 1.2 / np.sqrt(_TOY_DIM) for _ in range(_TOY_LAYERS)]
    Us = [rng.standard_normal((_TOY_DIM, _TOY_DIM)) / np.sqrt(_TOY_DIM) for _ in range(_TOY_LAYERS)]

    def hidden_states(ids: np.ndarray) -> np.ndarray:
        h = E[ids] * np.sqrt(_TOY_DIM)
        out = [h]
        steps = np.arange(1, len(ids) + 1)[:, None]
        for W, U in zip(Ws, Us):
            # causal pooling of earlier states (a crude stand-in for attention), rescaled by
            # sqrt(t) so context does not fade relative to the current token as t grows
            ctx = np.cumsum(h, axis=0) / np.sqrt(steps)
            h = h + 0.5 * np.tanh(h @ W) + 1.5 * ctx @ U          # residual stream
            h = h / np.sqrt((h ** 2).mean(1, keepdims=True))      # RMS norm
            out.append(h)
        return np.stack(out)                      # (n_layers + 1, T, dim)

    return hidden_states


def pool(h: np.ndarray, how: str) -> np.ndarray:
    """h: (T, dim) states of the current sample's tokens."""
    if how == "last":
        return h[-1]
    if how == "mean":
        return h.mean(0)
    if how == "sum":
        return h.sum(0)
    raise ValueError(f"pooling must be one of {POOLINGS}, got {how!r}")


def _context_ranges(ds: Dataset, context: str):
    """Yield (first sample index fed as context, sample index) per sample."""
    if context not in ("passage", "sentence"):
        raise ValueError("model.context must be 'passage' or 'sentence'")
    start = 0
    for i in range(ds.n):
        if i > 0 and ds.block_ids[i] != ds.block_ids[i - 1]:
            start = i
        yield (start if context == "passage" else i), i


def _synthetic_states(ds: Dataset, mcfg: dict, layers: list[int], poolings) -> dict:
    vocab = int(ds.meta.get("vocab", max(int(t.max()) for t in ds.tokens) + 1))
    lm = _toy_lm(vocab)
    out = {(L, p): [] for L in layers for p in poolings}
    for c0, i in _context_ranges(ds, mcfg["context"]):
        ids = np.concatenate(ds.tokens[c0:i + 1])[-mcfg["max_tokens"]:]
        cur = min(len(ds.tokens[i]), len(ids))
        hs = lm(ids)
        for L in layers:
            for p in poolings:
                out[(L, p)].append(pool(hs[L, -cur:], p))
    return {k: np.stack(v) for k, v in out.items()}


def _hf_states(ds: Dataset, mcfg: dict, layers: list[int], poolings) -> dict:
    if ds.texts is None:
        raise ValueError("hf backend needs dataset texts")
    # Hide every GPU from torch before it is imported: CPU inference only, by construction.
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    import torch
    from transformers import AutoModel, AutoTokenizer

    torch.set_grad_enabled(False)
    tok = AutoTokenizer.from_pretrained(mcfg["name"])
    model = AutoModel.from_pretrained(mcfg["name"]).eval().to("cpu")
    n_hidden = model.config.num_hidden_layers
    if max(layers) > n_hidden:
        raise ValueError(f"{mcfg['name']} has {n_hidden} layers; asked for {max(layers)}")
    out = {(L, p): [] for L in layers for p in poolings}
    for c0, i in _context_ranges(ds, mcfg["context"]):
        ctx = " ".join(ds.texts[c0:i])
        ctx_ids = tok(ctx, add_special_tokens=False)["input_ids"] if ctx else []
        cur_ids = tok((" " if ctx else "") + ds.texts[i], add_special_tokens=False)["input_ids"]
        ids = (ctx_ids + cur_ids)[-mcfg["max_tokens"]:]
        cur = min(len(cur_ids), len(ids))
        hs = model(torch.tensor([ids]), output_hidden_states=True).hidden_states
        for L in layers:
            h = hs[L][0, -cur:].float().numpy()
            for p in poolings:
                out[(L, p)].append(pool(h, p))
    return {k: np.stack(v) for k, v in out.items()}


def _slug(s: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", s)


def cache_path(cache_dir: Path, model: str, context: str, layer: int, pooling: str,
               split: str, stim_hash: str) -> Path:
    return (Path(cache_dir) / f"{_slug(model)}__ctx-{context}" / f"L{layer:02d}" / pooling / split
            / f"{stim_hash}.npy")


def get(ds: Dataset, cfg: dict, split: str, pooling: str | None = None,
        layers: list[int] | None = None) -> tuple[dict[int, np.ndarray], dict]:
    """Return {layer: (n_samples, dim)} for one pooling, plus cache provenance."""
    m = cfg["model"]
    pooling = pooling or cfg["pooling"]
    layers = list(layers or m["layers"])
    sh = ds.stimulus_hash()
    paths = {L: cache_path(m["cache_dir"], m["name"], m["context"], L, pooling, split, sh) for L in layers}
    feats, missing, linked = {}, [], []
    for L, p in paths.items():
        if p.exists():
            feats[L] = np.load(p)
            continue
        # Same activations already cached under another split? Link instead of recomputing.
        for other in p.parent.parent.iterdir() if p.parent.parent.exists() else []:
            q = other / p.name
            if q != p and q.exists():
                p.parent.mkdir(parents=True, exist_ok=True)
                try:
                    os.link(q, p)
                except OSError:
                    p.write_bytes(q.read_bytes())
                meta = json.loads(Path(str(q) + ".json").read_text())
                meta.update(split=split, linked_from=str(q))
                Path(str(p) + ".json").write_text(json.dumps(meta, indent=1))
                feats[L] = np.load(p)
                linked.append(L)
                break
        else:
            missing.append(L)

    t0 = time.time()
    if missing:
        fn = {"synthetic": _synthetic_states, "hf": _hf_states}[m["backend"]]
        new = fn(ds, m, missing, [pooling])
        for L in missing:
            p = paths[L]
            p.parent.mkdir(parents=True, exist_ok=True)
            np.save(p, new[(L, pooling)])
            Path(str(p) + ".json").write_text(json.dumps({
                "model": m["name"], "backend": m["backend"], "layer": L, "pooling": pooling,
                "split": split, "context": m["context"], "stimulus_hash": sh,
                "shape": list(new[(L, pooling)].shape)}, indent=1))
            feats[L] = new[(L, pooling)]
    prov = {"cache_dir": str(m["cache_dir"]), "stimulus_hash": sh, "computed_layers": missing,
            "linked_layers": linked, "seconds": round(time.time() - t0, 3)}
    return feats, prov
