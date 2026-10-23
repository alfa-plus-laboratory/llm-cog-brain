"""Noise ceiling: how well can the other subjects' brains predict this subject's?

Schrimpf et al. 2021 (PNAS 118(45), Methods 7; code: mschrimpf/neural-nlp
benchmarks/neural.py::_PereiraBenchmark and benchmarks/ceiling.py) divide
model predictivity by this ceiling and report 0.32 for Pereira2018. It is not
a split-half Spearman-Brown reliability. The procedure:

  1. for k = 2..N, draw subsets of k subjects;
  2. inside a subset, hold one subject out as the target and predict each of
     its voxels from all voxels of the other k-1 subjects, with the regression
     used for models (OLS, 5 cross-validation folds, Pearson r per test fold);
  3. per voxel, fit r(k) = v0 * (1 - exp(-k / tau)) to bootstrap means over
     subsets x folds (100 bootstraps); the voxel's ceiling is the median v0,
     i.e. r extrapolated to infinitely many subjects;
  4. ceiling = median of the per-voxel ceilings over all voxels.

Why a regression and not split-half: each subject has its own voxels (native
space, ~1300 language-network voxels per subject in Pereira2018), so there is
no voxel-to-voxel correspondence to average across subjects. `split_half`
needs a shared response space (e.g. one mean per fROI) and measures something
else -- the reliability of the group-mean response. It is a diagnostic and
cannot be compared with 0.32.

Where this differs from neural-nlp (the structure is the same, some details are not):
  * every subject in a subset is held out once. neural-nlp draws 5 targets
    with replacement. Subsets are distinct, min(n_subsets, C(N, k)) per k;
    neural-nlp draws 10 ordered tuples, so at k = N it repeats one subset 10 times;
  * the pool is the language-network voxels only. neural-nlp's pool held every
    atlas of its assembly (language, MD, DMN, auditory, visual), restricted to
    the pool subjects who did both experiments;
  * one experiment at a time. neural-nlp averages each voxel over both;
  * the saturating fit is a grid search over tau > 0 with v0 solved in closed
    form and clipped to [0, 1]. neural-nlp uses scipy curve_fit from (1, 1);
  * folds come from splits.make, so the ceiling can also be computed under the
    contiguous split. 'shuffled' with 5 folds is the Schrimpf setting.

Ridge ceiling (`extrapolation_ceiling(..., ridge=Ridge(...))`). The model score
r_LLM comes from readout.nested_predict, which is ridge with each voxel's alpha
chosen on inner folds, not OLS. Dividing a ridge r by an OLS ceiling mixes two
estimators, and on Pereira2018 the ridge r even exceeds the OLS ceiling. With
`ridge` set, step 2 uses the readout's estimator instead: the pool's voxels are
z-scored on the training rows, alpha is chosen per voxel on the readout's inner
folds (same kind and seeds), and the model is refitted on the outer-train rows.
Only n x n kernels are needed (see `_ridge_fit_folds`). Steps 1, 3 and 4 do not change.
`matched=True` also scores each draw the way r_LLM is scored (out-of-fold
predictions pooled over folds and fold-centred) and aggregates it the same way
(mean over voxels, then over subjects). For a mean, the subject's mean-r curve is
extrapolated rather than averaging per-voxel v0 (see the comment in the code).
"""
from __future__ import annotations

import itertools
import math
from dataclasses import dataclass

import numpy as np

from . import readout
from . import splits as splits_mod
from .readout import voxelwise_r

SCHRIMPF_N_FOLDS = 5
_TAUS = np.logspace(-3, 4, 281)


def ols_predict_from_gram(G: np.ndarray, Y: np.ndarray, tr: np.ndarray, te: np.ndarray,
                          rcond: float = 1e-10) -> np.ndarray:
    """Minimum-norm OLS with intercept (what sklearn's LinearRegression returns), via the Gram matrix.

    G = X X^T of the *uncentred* features over all samples. The features are centred
    on the training rows inside the kernel, so a pool's Gram is just the sum of its
    subjects' Grams and never has to be rebuilt from voxels.
    Returns predictions for the rows `te`, shape (len(te), Y.shape[1]).
    """
    Gtt, Gvt = G[np.ix_(tr, tr)], G[np.ix_(te, tr)]
    m = Gtt.mean(1)
    c = m.mean()
    K = Gtt - m[:, None] - m[None, :] + c                       # Xc Xc^T
    Kx = Gvt - Gvt.mean(1)[:, None] - m[None, :] + c            # (X_te - mu) Xc^T
    lam, Q = np.linalg.eigh(K)
    keep = lam > rcond * lam.max()                              # drops the centring null space
    Q, lam = Q[:, keep], lam[keep]
    mu = Y[tr].mean(0)
    return (Kx @ Q) @ ((Q.T @ (Y[tr] - mu)) / lam[:, None]) + mu


def _subsets(n: int, k: int, n_subsets: int, rng: np.random.Generator) -> list[tuple[int, ...]]:
    if math.comb(n, k) <= n_subsets:
        return list(itertools.combinations(range(n), k))
    out: set[tuple[int, ...]] = set()
    while len(out) < n_subsets:
        out.add(tuple(sorted(rng.choice(n, k, replace=False).tolist())))
    return sorted(out)


def fit_saturating(k: np.ndarray, y: np.ndarray, taus: np.ndarray = _TAUS, return_tau: bool = False):
    """Least-squares v0 of y = v0 * (1 - exp(-k / tau)), v0 in [0, 1], tau > 0.

    y: (..., len(k)). For a fixed tau the best v0 is closed form, so tau is found
    by grid search (no local optimum to get stuck in). Returns v0, shape y.shape[:-1]
    (and the fitted tau, same shape, if return_tau).
    """
    k = np.asarray(k, dtype=float)
    yy = (y ** 2).sum(-1)
    best_sse = np.full(y.shape[:-1], np.inf)
    best_v0 = np.zeros(y.shape[:-1])
    best_tau = np.zeros(y.shape[:-1])
    for tau in taus:
        f = 1.0 - np.exp(-k / tau)
        ff = float(f @ f)
        yf = y @ f
        v0 = np.clip(yf / ff, 0.0, 1.0)
        sse = yy - 2 * v0 * yf + v0 ** 2 * ff
        better = sse < best_sse
        best_sse[better], best_v0[better], best_tau[better] = sse[better], v0[better], tau
    return (best_v0, best_tau) if return_tau else best_v0


@dataclass
class Ridge:
    """The readout's ridge (readout.nested_predict with one candidate), for the ceiling.

    block_ids: (n,) block of each sample; the inner folds are built from it exactly
    as the readout builds them. alphas, n_inner: the run's config.alphas(cfg) and
    cfg['split']['inner_folds'].
    """
    block_ids: np.ndarray
    alphas: np.ndarray
    n_inner: int
    voxel_chunk: int = 20000


def _ols_fit_folds(Y: list[np.ndarray], folds: list, rcond: float):
    grams = [y @ y.T for y in Y]

    def fit(pool: list[int], t: int) -> list[np.ndarray]:
        G = sum(grams[s] for s in pool)
        return [ols_predict_from_gram(G, Y[t], tr, te, rcond) for tr, te in folds]
    return fit


def _scaled_kernel(y: np.ndarray, tr: np.ndarray) -> np.ndarray:
    """One subject's voxels z-scored on the rows `tr` (readout._scaler), as a kernel: all rows x tr."""
    mu, scale = readout._scaler(y[tr])
    Z = (y - mu) * scale
    return Z @ Z[tr].T


def kernel_ridge_predict(lam, U, VE, Ytr: np.ndarray, alpha: np.ndarray) -> np.ndarray:
    """readout._fit(...).predict in kernel form: (lam, U, VE) from readout.kernel_basis, alpha per voxel."""
    y_mu = Ytr.mean(0)
    UtY = U.T @ (Ytr - y_mu)
    out = np.tile(y_mu, (VE.shape[0], 1))
    for a in np.unique(alpha):                     # one solve per distinct alpha, as in _fit
        m = alpha == a
        out[:, m] += VE @ (UtY[:, m] / (lam + a)[:, None])
    return out


def _ridge_fit_folds(Y: list[np.ndarray], split: splits_mod.Split, ridge: Ridge, chosen: dict | None = None):
    """Pool -> target predictions with the readout's ridge, from n x n kernels only.

    The readout z-scores the features on each training set, so a pool's kernel is
    not a sum of raw Grams. z-scoring is per column, though, so it IS the sum of
    each subject's own z-scored kernel for that training set. Those are computed
    once per subject and training set (outer train + its inner train sets), and
    every fit after that costs an eigendecomposition of at most n_train x n_train,
    whatever the pool size. Inner folds and their seeds are those of
    readout.nested_predict, so the alphas are chosen exactly as for the model.
    `chosen`, if given, collects {pool size: list of per-voxel alpha arrays}.
    """
    sets = []
    for k, (tr, te) in enumerate(split.outer()):
        inner = splits_mod.inner(split.kind, ridge.block_ids[tr], ridge.n_inner, split.seed + 1000 + k)
        sets.append((tr, te, [(tr[a], tr[b]) for a, b in inner]))
    kern = [[(_scaled_kernel(y, tr), [_scaled_kernel(y, itr) for itr, _ in inner]) for tr, _, inner in sets]
            for y in Y]

    def fit(pool: list[int], t: int) -> list[np.ndarray]:
        y, preds = Y[t], []
        for k, (tr, te, inner) in enumerate(sets):
            sc = np.empty((len(inner), len(ridge.alphas), y.shape[1]))
            for j, (itr, iva) in enumerate(inner):
                K = sum(kern[s][k][1][j] for s in pool)
                lam, U, VE = readout.kernel_basis(K[itr], K[iva])
                sc[j] = readout.sweep_basis(lam, U, VE, y[itr], y[iva], ridge.alphas, ridge.voxel_chunk)
            a = ridge.alphas[sc.mean(0).argmax(0)]             # as readout.nested_predict
            K = sum(kern[s][k][0] for s in pool)
            preds.append(kernel_ridge_predict(*readout.kernel_basis(K[tr], K[te]), y[tr], a))
            if chosen is not None:
                chosen.setdefault(len(pool), []).append(a)
        return preds
    return fit


def _holdout(Y: list[np.ndarray], folds: list, fit, *, n_subsets: int, seed: int, pooled: bool = False,
             log=None):
    """Score fit(pool, target) -> test-fold predictions for every k, subset and held-out target.

    Returns (per-fold r, pooled r): {subject: {k: (rows, n_vox)}}. Per-fold r has one
    row per draw x fold (Schrimpf). Pooled r (only if `pooled`) has one row per draw:
    the test folds' predictions put together, fold-centred, one r over all samples --
    how the pipeline scores r_LLM (readout.center_within_folds).
    """
    N = len(Y)
    if N < 2:
        raise ValueError("a ceiling needs at least 2 subjects")
    fold_of = np.empty(len(Y[0]), dtype=int)
    for i, (_, te) in enumerate(folds):
        fold_of[te] = i
    Yc = [readout.center_within_folds(y, fold_of) for y in Y] if pooled else None
    rng = np.random.default_rng(seed)
    per_fold: dict = {s: {} for s in range(N)}
    whole: dict = {s: {} for s in range(N)}
    for k in range(2, N + 1):
        subsets = _subsets(N, k, n_subsets, rng)
        for subset in subsets:
            for t in subset:
                preds = fit([s for s in subset if s != t], t)
                per_fold[t].setdefault(k, []).append(
                    np.stack([voxelwise_r(p, Y[t][te]) for p, (_, te) in zip(preds, folds)]))
                if pooled:
                    full = np.empty_like(Y[t])
                    for p, (_, te) in zip(preds, folds):
                        full[te] = p
                    whole[t].setdefault(k, []).append(
                        voxelwise_r(readout.center_within_folds(full, fold_of), Yc[t])[None])
        if log is not None:
            log(f"k={k}: {len(subsets)} subsets x {k} targets done")
    cat = lambda d: {s: {k: np.concatenate(v) for k, v in dd.items()} for s, dd in d.items()}  # noqa: E731
    return cat(per_fold), (cat(whole) if pooled else None)


def holdout_scores(Y: list[np.ndarray], folds: list, *, n_subsets: int = 10, seed: int = 0,
                   rcond: float = 1e-10) -> dict:
    """Pool -> held-out-subject Pearson r (OLS) for every subset size k.

    Returns {subject index: {k: (n_draws * n_folds, n_vox) per-fold r}}; a subject
    that was never drawn at some k has no entry for that k.
    """
    return _holdout(Y, folds, _ols_fit_folds(Y, folds, rcond), n_subsets=n_subsets, seed=seed)[0]


def _extrapolate(scores: dict, n_boot: int, seed: int) -> list[np.ndarray]:
    """Per subject, each voxel's ceiling: median over bootstraps of the extrapolated v0."""
    rng = np.random.default_rng(seed + 1)
    out = []
    for s, by_k in scores.items():
        ks = sorted(by_k)
        boots = []
        for k in ks:
            v = by_k[k]                                         # (rows, n_vox)
            idx = rng.integers(0, len(v), (n_boot, len(v)))
            boots.append(v[idx].mean(1))                        # (n_boot, n_vox)
        v0 = fit_saturating(np.array(ks), np.stack(boots, -1))  # (n_boot, n_vox)
        out.append(np.median(v0, 0))
    return out


def extrapolation_ceiling(Y: list[np.ndarray], split: splits_mod.Split, *, n_subsets: int = 10,
                          n_boot: int = 100, seed: int = 0, subjects: list[str] | None = None,
                          ridge: Ridge | None = None, matched: bool = False, log=None) -> dict:
    """Schrimpf-style ceiling (module docstring), from per-subject responses (n, n_vox_s).

    Returns the ceiling (median over all voxels of each voxel's extrapolated v0)
    plus what it was built from; `per_voxel` is an array, keep it out of JSON.
    ridge: None = OLS (Schrimpf); a Ridge = the readout's ridge with per-voxel alpha.
    matched: also return `matched`, the ceiling scored and aggregated like r_LLM,
        and `by_scoring_and_aggregation`: both scorings x three aggregations.
    """
    folds = list(split.outer())
    chosen: dict = {}
    fit = _ols_fit_folds(Y, folds, 1e-10) if ridge is None else _ridge_fit_folds(Y, split, ridge, chosen)
    scores, whole = _holdout(Y, folds, fit, n_subsets=n_subsets, seed=seed, pooled=matched, log=log)
    names = subjects or [str(i) for i in range(len(Y))]
    by_subject = _extrapolate(scores, n_boot, seed)
    per_voxel = np.concatenate(by_subject)
    N = len(Y)
    at_n = [float(np.median(scores[s][N].mean(0))) for s in scores if N in scores[s]]
    out = {
        "ceiling": float(np.median(per_voxel)),
        "estimator": "ols" if ridge is None else "ridge",
        "method": f"holdout-subject {'OLS' if ridge is None else 'ridge (readout: z-scored, per-voxel alpha on inner folds)'}"
                  " from the other subjects' voxels, extrapolated in pool size "
                  "(Schrimpf et al. 2021 Methods 7); median over voxels",
        "split_kind": split.kind, "n_folds": split.n_folds,
        "n_subjects": N, "n_voxels": int(len(per_voxel)), "n_subsets": n_subsets, "n_boot": n_boot,
        "per_subject_median": {names[s]: float(np.median(c)) for s, c in enumerate(by_subject)},
        "median_r_by_k": {names[s]: {int(k): float(np.median(v.mean(0))) for k, v in sorted(by_k.items())}
                          for s, by_k in scores.items()},     # observed (not extrapolated) consistency
        "median_r_at_all_subjects": float(np.median(at_n)) if at_n else None,
        "per_voxel": per_voxel,
    }
    if ridge is not None:
        a_all = np.concatenate([a for v in chosen.values() for a in v])
        out["ridge"] = {
            "alphas": [float(a) for a in ridge.alphas], "n_inner": ridge.n_inner, "inner_kind": split.kind,
            "inner_seed": "split.seed + 1000 + outer fold (= readout.nested_predict)",
            "alpha_chosen_share": {f"{a:g}": float(np.mean(a_all == a)) for a in ridge.alphas},
            "alpha_median_by_pool_size": {int(m): float(np.median(np.concatenate(v))) for m, v in sorted(chosen.items())},
        }
    if matched:
        # Mean over voxels: average r first, then extrapolate the subject's mean curve. Single-voxel
        # curves are noisy and their extrapolations have a long right tail (some up to the clip at 1);
        # the median above is immune to that, a mean is not (Pereira2018 contiguous: mean of per-voxel
        # v0 0.21-0.24 vs median 0.10-0.15, while the mean-curve v0 stays near the observed r at k = N).
        def curve_ceiling(sc):                                  # per subject: v0 of the mean-r curve
            agg = {s: {k: v.mean(1, keepdims=True) for k, v in by_k.items()} for s, by_k in sc.items()}
            return [float(c[0]) for c in _extrapolate(agg, n_boot, seed)]
        w_voxel = _extrapolate(whole, n_boot, seed)
        w_subject = curve_ceiling(whole)
        at_n_w = [float(whole[s][N].mean()) for s in whole if N in whole[s]]
        out["matched"] = {
            "ceiling": float(np.mean(w_subject)),
            "scoring": "test folds pooled, fold-centred (readout.center_within_folds), one r per voxel and draw",
            "aggregation": "mean over voxels per draw, curve extrapolated per subject, mean over subjects (= r_LLM)",
            "per_subject": {names[s]: c for s, c in enumerate(w_subject)},
            "mean_r_by_k": {names[s]: {int(k): float(v.mean()) for k, v in sorted(by_k.items())}
                            for s, by_k in whole.items()},  # observed (not extrapolated)
            "mean_r_at_all_subjects": float(np.mean(at_n_w)) if at_n_w else None,
            "per_voxel": np.concatenate(w_voxel),
        }
        mean_v0 = lambda vs: float(np.mean([v.mean() for v in vs]))  # noqa: E731
        out["by_scoring_and_aggregation"] = {
            "per_fold_r": {"median_of_voxel_v0": out["ceiling"], "mean_of_voxel_v0": mean_v0(by_subject),
                           "v0_of_mean_r": float(np.mean(curve_ceiling(scores)))},
            "pooled_fold_centred_r": {"median_of_voxel_v0": float(np.median(out["matched"]["per_voxel"])),
                                      "mean_of_voxel_v0": mean_v0(w_voxel),
                                      "v0_of_mean_r": out["matched"]["ceiling"]},
            "share_of_voxel_v0_at_clip": {"per_fold_r": float(np.mean(per_voxel >= 0.999)),
                                          "pooled_fold_centred_r": float(np.mean(out["matched"]["per_voxel"] >= 0.999))},
        }
    return out


# ---------------------------------------------------------------- split-half (diagnostic)

def spearman_brown(r, factor: float = 2.0):
    r = np.asarray(r, dtype=float)
    return factor * r / (1 + (factor - 1) * r)


def roi_means(Y: list[np.ndarray], roi_index: list, n_rois: int) -> list[np.ndarray]:
    """Per subject, the mean response of each ROI: a response space every subject shares."""
    out = []
    for y, idx in zip(Y, roi_index):
        idx = np.asarray(idx)
        out.append(np.stack([y[:, idx == j].mean(1) if (idx == j).any() else np.full(len(y), np.nan)
                             for j in range(n_rois)], 1))
    return out


def split_half(R: list[np.ndarray], n_splits: int = 200, seed: int = 0) -> dict:
    """Split-half reliability of the group-mean response, Spearman-Brown corrected.

    R: per subject (n, m) responses in a *shared* space (same m columns for every
    subject, e.g. `roi_means`). Each split averages two disjoint random halves of
    floor(N/2) subjects (one subject sits out when N is odd), correlates the halves
    column by column, and the mean r over splits is stepped up by Spearman-Brown
    (factor 2): the reliability of a 2*floor(N/2)-subject average.
    """
    N = len(R)
    h = N // 2
    if h < 1:
        raise ValueError("split-half needs at least 2 subjects")
    A = np.stack(R)                                             # (N, n, m)
    rng = np.random.default_rng(seed)
    rs = []
    for _ in range(n_splits):
        p = rng.permutation(N)
        rs.append(voxelwise_r(A[p[:h]].mean(0), A[p[h:2 * h]].mean(0)))
    r_half = np.mean(rs, 0)
    sb = spearman_brown(r_half)
    return {"r_half": r_half.tolist(), "spearman_brown": sb.tolist(),
            "mean_spearman_brown": float(np.mean(sb)), "subjects_per_half": h, "n_splits": n_splits}
