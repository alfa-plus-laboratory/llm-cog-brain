"""Significance null that keeps temporal autocorrelation: circular shift only.

Discipline: shift the measured response circularly
against the (fixed, out-of-fold) predictions and rescore. Both series keep their
autocorrelation; only the stimulus alignment is broken. An i.i.d. permutation
destroys the autocorrelation, yields a null that is far too narrow and makes
almost anything look significant -- there is deliberately no such function here,
and config.validate rejects asking for one.

Differences from the implementation this was ported from:
* every shift is scored at once with an FFT cross-correlation, so the null is
  exact: a series of n samples has only n - 2*min_shift + 1 distinct shifts.
  If that is <= n_perm all of them are used (no duplicates masquerading as a
  bigger null); otherwise n_perm distinct shifts are drawn. The smallest
  attainable p is therefore 1 / (n_shifts + 1) -- reported, not hidden.
* the statistic is the subject-averaged mean-over-voxels r, with one shift
  applied to all subjects at once (they saw the same stimulus sequence).

Known property: with fixed-length blocks, shifts that are multiples of the
block length keep within-block position aligned. That widens the null for
anything carrying position information, i.e. it errs conservative.
"""
from __future__ import annotations

import numpy as np

_EPS = 1e-8


def _unit(a: np.ndarray) -> np.ndarray:
    a = a - a.mean(0, keepdims=True)
    nrm = np.sqrt((a ** 2).sum(0, keepdims=True))
    return np.where(nrm > _EPS, a / np.where(nrm > _EPS, nrm, 1.0), 0.0)


def all_shift_r(pred: np.ndarray, actual: np.ndarray) -> np.ndarray:
    """r[s, v] = pearson(pred[:, v], np.roll(actual, s, axis=0)[:, v]) for every s in 0..n-1."""
    n = pred.shape[0]
    P = np.fft.rfft(_unit(pred), axis=0)
    A = np.fft.rfft(_unit(actual), axis=0)
    return np.fft.irfft(P * np.conj(A), n=n, axis=0)


def circular_shift_null(preds: dict[str, list[np.ndarray]], Y: list[np.ndarray], *,
                        n_perm: int = 1000, min_shift: int = 10, seed: int = 0,
                        contrasts: dict[str, tuple[str, str]] | None = None) -> dict:
    n = Y[0].shape[0]
    if n <= 2 * min_shift:
        raise ValueError(f"series too short ({n}) for min_shift={min_shift}")
    valid = np.arange(min_shift, n - min_shift + 1)
    rng = np.random.default_rng(seed)
    shifts = valid if len(valid) <= n_perm else np.sort(rng.choice(valid, n_perm, replace=False))

    curves = {}
    for name, per_subj in preds.items():
        # (n_shifts_total,) subject-averaged, voxel-averaged r for every shift
        curves[name] = np.mean([all_shift_r(p, y).mean(1) for p, y in zip(per_subj, Y)], axis=0)
    for cname, (a, b) in (contrasts or {}).items():
        curves[cname] = curves[a] - curves[b]

    out = {}
    for name, c in curves.items():
        obs, null = float(c[0]), c[shifts]
        sd = float(null.std(ddof=1))
        out[name] = {
            "obs": obs,
            "null_mean": float(null.mean()),
            "null_sd": sd,
            "z": (obs - float(null.mean())) / sd if sd > 0 else float("nan"),
            "p": float((1 + np.sum(null >= obs)) / (1 + len(null))),
        }
    out["_meta"] = {"method": "circular_shift", "n_shifts": int(len(shifts)),
                    "n_distinct_possible": int(len(valid)), "min_shift": int(min_shift),
                    "min_attainable_p": 1.0 / (1 + len(shifts)), "n_samples": int(n)}
    return out
