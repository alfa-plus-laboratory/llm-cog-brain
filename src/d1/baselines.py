"""Confound baselines and ablations.

Confounds (enter the joint model, decide the verdict):
    position   where the sample sits inside its block. Pereira-style short
               blocks: one-hot of the position (Hadidi, Feghhi et al. use a
               one-hot 4-D vector for Pereira2018). Long blocks (stories, TRs):
               one-hot of 10 equal bins of relative position.
    word_rate  words per second of the sample (语速). With a fixed presentation
               time per sentence this is sentence length.

Diagnostic (reported, never decides anything):
    oasm       Orthogonal Autocorrelated Sequences Model: identity over samples,
               Gaussian-smoothed within each block. Knows nothing about language,
               only "nearby samples look alike". Under a contiguous split it
               cannot predict a held-out block at all; under a shuffled split it
               can, and how well is a direct read-out of the leak.

Ablation:
    zero_input_ablation   rerun prediction with the inputs zeroed; if the score
               survives, the model is not using its input.
"""
from __future__ import annotations

import numpy as np

from .data import Dataset

LONG_BLOCK = 12          # blocks longer than this get binned relative position


def position(ds: Dataset) -> np.ndarray:
    blk_len = np.bincount(np.r_[0, np.cumsum(np.diff(ds.block_ids) != 0)])
    if blk_len.max() <= LONG_BLOCK:
        return np.eye(int(ds.position.max()) + 1)[ds.position]
    rank = np.r_[0, np.cumsum(np.diff(ds.block_ids) != 0)]
    rel = ds.position / np.maximum(blk_len[rank] - 1, 1)
    return np.eye(10)[np.minimum((rel * 10).astype(int), 9)]


def word_rate(ds: Dataset) -> np.ndarray:
    return (ds.n_words / ds.duration)[:, None].astype(float)


def oasm(ds: Dataset, sigma: float) -> np.ndarray:
    n = ds.n
    X = np.zeros((n, n))
    idx = np.arange(n)
    for b in np.unique(ds.block_ids):
        m = idx[ds.block_ids == b]
        d = m[:, None] - m[None, :]
        X[np.ix_(m, m)] = np.exp(-0.5 * (d / sigma) ** 2)
    return X


CONFOUND_FNS = {"position": position, "word_rate": word_rate}


def confounds(ds: Dataset, names: list[str]) -> tuple[np.ndarray, list[tuple[str, int, int]]]:
    """Stack the requested confounds; returns X and (name, lo, hi) column ranges."""
    cols, spans, lo = [], [], 0
    for name in names:
        x = CONFOUND_FNS[name](ds)
        cols.append(x)
        spans.append((name, lo, lo + x.shape[1]))
        lo += x.shape[1]
    return np.hstack(cols), spans


def zero_input_ablation(predict_fn, X: np.ndarray, Y: np.ndarray, cols=None, score_fn=None) -> dict:
    """The critical control, generalised to zero a subset of columns.

    cols=None zeroes every input. With a linear readout standardised on train,
    an all-zero input gives a constant prediction and r = 0 by construction: the
    check then only proves predictions depend on X at all (it catches leaks such
    as predictions built from Y). Zeroing just the LLM columns of a joint model
    is the informative variant -- what survives is what the confounds carry.
    """
    from .readout import voxelwise_r
    score_fn = score_fn or (lambda p, a: float(np.mean(voxelwise_r(p, a))))
    Xz = np.array(X, dtype=float, copy=True)
    if cols is None:
        Xz[:] = 0.0
    else:
        Xz[:, cols] = 0.0
    real = score_fn(predict_fn(X), Y)
    zeroed = score_fn(predict_fn(Xz), Y)
    retained = zeroed / real if real != 0 else float("nan")
    return {"real_score": real, "zeroed_score": zeroed, "retained_fraction": retained}
