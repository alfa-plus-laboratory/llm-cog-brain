"""Train/test splits. ``contiguous`` is the primary; ``shuffled`` exists only as a control.

Definitions:

contiguous  every block (passage/story) lies wholly in one fold, and each test
            fold is one unbroken stretch of the presentation timeline.
shuffled    individual samples are dealt to folds at random, ignoring blocks --
            what Hadidi, Feghhi et al. (Nat Commun 2026) call a shuffled split and
            what Brain-Score's legacy Pereira2018 benchmark does (ShuffleSplit
            over stimulus_id). The (stimulus, response) pairing is NOT broken;
            the leak is that a test sentence's passage-mates sit in train.

Breaking the pairing is a null, not a split -- that lives in null.py, and it
shifts circularly rather than permuting.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .config import SPLIT_KINDS


@dataclass
class Split:
    kind: str
    n_folds: int
    seed: int
    fold: np.ndarray            # (n,) outer test fold of each sample

    def outer(self):
        for k in range(self.n_folds):
            yield np.flatnonzero(self.fold != k), np.flatnonzero(self.fold == k)

    def describe(self, block_ids: np.ndarray) -> dict:
        leaking = sum(len(np.unique(self.fold[block_ids == b])) > 1 for b in np.unique(block_ids))
        return {
            "kind": self.kind,
            "role": "primary" if self.kind == "contiguous" else "control",
            "unit": "block" if self.kind == "contiguous" else "sample",
            "n_folds": self.n_folds,
            "seed": self.seed,
            "n_samples": int(len(self.fold)),
            "test_fold_sizes": np.bincount(self.fold, minlength=self.n_folds).tolist(),
            "blocks_spanning_train_and_test": int(leaking),
            "fold_of_sample": self.fold.tolist(),
        }


def _contiguous_folds(block_ids: np.ndarray, n_folds: int) -> np.ndarray:
    """Cut the timeline at block boundaries into n_folds runs of ~equal sample count."""
    rank = np.r_[0, np.cumsum(np.diff(block_ids) != 0)]   # i-th block in presentation order
    sizes = np.bincount(rank)
    if len(sizes) < n_folds:
        raise ValueError(f"{len(sizes)} blocks cannot fill {n_folds} contiguous folds")
    cum = np.cumsum(sizes)
    # Each block goes to the fold its midpoint falls in.
    mid = (cum - sizes / 2) / cum[-1]
    block_fold = np.minimum((mid * n_folds).astype(int), n_folds - 1)
    if len(np.unique(block_fold)) != n_folds:
        raise ValueError("a contiguous fold came out empty; use fewer folds")
    return block_fold[rank]


def _shuffled_folds(n: int, n_folds: int, rng: np.random.Generator) -> np.ndarray:
    fold = np.arange(n) % n_folds
    rng.shuffle(fold)
    return fold


def make(kind: str, block_ids: np.ndarray, n_folds: int, seed: int) -> Split:
    """Outer split over all samples (samples must be in presentation order)."""
    if kind not in SPLIT_KINDS:
        raise ValueError(f"split kind must be one of {SPLIT_KINDS}")
    block_ids = np.asarray(block_ids)
    if kind == "contiguous":
        fold = _contiguous_folds(block_ids, n_folds)
    else:
        fold = _shuffled_folds(len(block_ids), n_folds, np.random.default_rng(seed))
    s = Split(kind, n_folds, seed, fold)
    if kind == "contiguous":
        assert_no_block_leak(s, block_ids)
    return s


def inner(kind: str, block_ids_train: np.ndarray, n_inner: int, seed: int):
    """Inner (validation) folds inside an outer-train set, same kind as the outer split.

    Returns a list of (train_idx, val_idx) indexing into the outer-train subset.
    """
    s = make(kind, block_ids_train, n_inner, seed)
    return list(s.outer())


def assert_no_block_leak(split: Split, block_ids: np.ndarray) -> None:
    for b in np.unique(block_ids):
        if len(np.unique(split.fold[block_ids == b])) > 1:
            raise AssertionError(f"contiguous split leaks block {b} across folds")
    # And every test fold is one unbroken run of samples.
    for k in range(split.n_folds):
        idx = np.flatnonzero(split.fold == k)
        if len(idx) and idx[-1] - idx[0] + 1 != len(idx):
            raise AssertionError(f"contiguous fold {k} is not one unbroken stretch")
