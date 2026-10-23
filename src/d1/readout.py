"""Ridge readout (features -> every voxel), nested CV, block-bootstrap CIs.

Ridge is solved once per training set and every alpha is swept from that one
decomposition (numpy, CPU):

    X = U S V^T   =>   W_a = V diag(s / (s^2 + a)) U^T Y

so the cost of an alpha is one (n_val x k) @ (k x n_vox) matmul. When features
outnumber samples (LLM layers vs a few hundred sentences) the same quantities
come from the n x n kernel X X^T instead, which is cheaper (see `_basis`). Alpha is chosen
per voxel on inner folds of the same split kind as the outer split, never on
the test fold. Layer choice (when the model offers several) is made the same way,
on a cross-fitted inner score. The joint LLM + confound model is built so it
contains the confound-only model exactly (see `nested_predict_joint`).

Effect sizes are reported with block-bootstrap CIs; see `bootstrap`.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from . import splits as splits_mod

_EPS = 1e-8


def voxelwise_r(pred: np.ndarray, actual: np.ndarray) -> np.ndarray:
    """Pearson r per voxel; zero-variance voxels give 0.0."""
    if pred.shape != actual.shape:
        raise ValueError(f"shape mismatch: {pred.shape} vs {actual.shape}")
    p = pred - pred.mean(0, keepdims=True)
    a = actual - actual.mean(0, keepdims=True)
    den = np.sqrt((p ** 2).sum(0) * (a ** 2).sum(0))
    out = np.zeros(pred.shape[1])
    nz = den > _EPS
    out[nz] = (p * a).sum(0)[nz] / den[nz]
    return out


def _scaler(Xtr: np.ndarray, llm_cols: slice | None = None, gain: float = 1.0):
    """z-score with train stats; columns constant in train get weight 0.

    In a joint model the LLM columns are further multiplied by gain/sqrt(width):
    ridge with penalty `a` on those columns then acts like penalty a/gain^2 on a
    unit-norm LLM band, whatever its width. The confound columns are left exactly
    as in the confound-only model, so their per-voxel alphas carry over.
    """
    mu = Xtr.mean(0)
    sd = Xtr.std(0)
    scale = np.where(sd > _EPS, 1.0 / np.where(sd > _EPS, sd, 1.0), 0.0)
    if llm_cols is not None:
        width = len(range(*llm_cols.indices(Xtr.shape[1])))
        scale[llm_cols] *= gain / np.sqrt(width)
    return mu, scale


def _basis(Xtr: np.ndarray, Xva: np.ndarray | None = None):
    """Everything ridge needs for any alpha, from one decomposition.

    Returns (lam, U, V_eval, W_back) such that, for centred targets Yc,
        predictions on Xva = V_eval @ diag(1 / (lam + a)) @ U.T @ Yc
        weights            = W_back @ diag(1 / (lam + a)) @ U.T @ Yc
    p >= n (e.g. 1600 LLM dims, a few hundred sentences): eigendecomposition of
    the n x n kernel X X^T -- several times cheaper than an n x p SVD.
    p < n: thin SVD, X = U S V^T, lam = s^2.
    """
    n, p = Xtr.shape
    if p >= n:
        lam, U, V_eval = kernel_basis(Xtr @ Xtr.T, None if Xva is None else Xva @ Xtr.T)
        return lam, U, V_eval, Xtr.T @ U
    U, s, Vt = np.linalg.svd(Xtr, full_matrices=False)
    # s / (s^2 + a) = s * 1/(lam + a): fold the s into the eval/back matrices
    V_eval = None if Xva is None else (Xva @ Vt.T) * s
    return s ** 2, U, V_eval, Vt.T * s


def kernel_basis(K: np.ndarray, Kva: np.ndarray | None = None):
    """`_basis` (lam, U, V_eval) from the kernel alone: K = Xtr Xtr^T, Kva = Xva Xtr^T.

    Callers that can build the kernel without the features (ceiling.py sums
    per-subject kernels into a pool's) get the same ridge as the readout.
    """
    lam, U = np.linalg.eigh(K)
    return np.clip(lam, 0, None), U, None if Kva is None else Kva @ U


def sweep(Xtr, Ytr, Xva, Yva, alphas, voxel_chunk: int = 20000) -> np.ndarray:
    """Validation r for every (alpha, voxel) from one decomposition. Inputs already scaled."""
    lam, U, VE, _ = _basis(Xtr, Xva)
    return sweep_basis(lam, U, VE, Ytr, Yva, alphas, voxel_chunk)


def sweep_basis(lam, U, VE, Ytr, Yva, alphas, voxel_chunk: int = 20000) -> np.ndarray:
    """`sweep` from a precomputed (lam, U, V_eval) -- see `_basis` / `kernel_basis`."""
    Yc = Ytr - Ytr.mean(0, keepdims=True)
    out = np.empty((len(alphas), Ytr.shape[1]))
    for lo in range(0, Ytr.shape[1], voxel_chunk):
        hi = min(lo + voxel_chunk, Ytr.shape[1])
        UtY = U.T @ Yc[:, lo:hi]
        yv = Yva[:, lo:hi] - Yva[:, lo:hi].mean(0, keepdims=True)
        yv /= np.maximum(np.sqrt((yv ** 2).sum(0, keepdims=True)), _EPS)
        for ai, a in enumerate(alphas):
            pv = VE @ (UtY / (lam + a)[:, None])
            pv -= pv.mean(0, keepdims=True)
            nrm = np.sqrt((pv ** 2).sum(0))
            out[ai, lo:hi] = np.where(nrm > _EPS, (pv * yv).sum(0) / np.maximum(nrm, _EPS), 0.0)
    return out


@dataclass
class Fitted:
    mu: np.ndarray
    parts: list              # [(voxel_idx, scale (p,), W (p, len(voxel_idx)))], one per distinct LLM gain
    y_mu: np.ndarray
    alpha: np.ndarray        # (n_voxels,)
    choice: object           # which candidate (e.g. layer) was selected
    llm_penalty: float | None = None   # joint model only: shared LLM-band penalty (inf = band off)

    @property
    def n_vox(self) -> int:
        return len(self.y_mu)

    def predict(self, X: np.ndarray) -> np.ndarray:
        out = np.tile(self.y_mu, (X.shape[0], 1))
        for idx, scale, W in self.parts:
            out[:, idx] += ((X - self.mu) * scale) @ W
        return out


def _fit(Xtr_raw, Ytr, alpha_per_voxel, choice, voxel_chunk, llm_cols=None, gain_per_voxel=None,
         llm_penalty=None) -> Fitted:
    y_mu = Ytr.mean(0)
    gain_per_voxel = np.ones(Ytr.shape[1]) if gain_per_voxel is None else gain_per_voxel
    parts, mu = [], None
    for g in np.unique(gain_per_voxel):
        idx = np.flatnonzero(gain_per_voxel == g)
        mu, scale = _scaler(Xtr_raw, llm_cols, g)
        lam, U, _, back = _basis((Xtr_raw - mu) * scale)
        W = np.zeros((Xtr_raw.shape[1], len(idx)))
        for lo in range(0, len(idx), voxel_chunk):
            sub = idx[lo:lo + voxel_chunk]
            UtY = U.T @ (Ytr[:, sub] - y_mu[sub])
            a_sub = alpha_per_voxel[sub]
            for a in np.unique(a_sub):         # one solve per distinct alpha, not per voxel
                m = a_sub == a
                W[:, lo:lo + len(sub)][:, m] = back @ (UtY[:, m] / (lam + a)[:, None])
        parts.append((idx, scale, W))
    return Fitted(mu, parts, y_mu, alpha_per_voxel, choice, llm_penalty)


def _inner_scores(X, Y, inner, alphas, voxel_chunk) -> np.ndarray:
    """Inner-validation r per inner fold, shape (n_inner, n_alphas, n_voxels)."""
    out = np.zeros((len(inner), len(alphas), Y.shape[1]))
    for j, (itr, iva) in enumerate(inner):
        mu, scale = _scaler(X[itr])
        out[j] = sweep((X[itr] - mu) * scale, Y[itr], (X[iva] - mu) * scale, Y[iva], alphas, voxel_chunk)
    return out


def _crossfit_score(sc: np.ndarray) -> float:
    """Honest mean-over-voxels inner score of one candidate, sc: (n_inner, n_alphas, v).

    "max over alpha of the inner score" is optimistic, and more so for more
    flexible candidates, so comparing candidates by it favours flexibility that
    does not generalise. Instead, for each inner fold, alpha is picked per voxel
    on the OTHER inner folds and scored on this one.
    """
    J = sc.shape[0]
    total = sc.sum(0)
    out = 0.0
    for j in range(J):
        a = ((total - sc[j]) / (J - 1)).argmax(0)               # (v,)
        out += float(np.take_along_axis(sc[j], a[None, :], 0).mean())
    return out / J


def _voxel_subset(n_vox: int, k: int, seed: int) -> np.ndarray:
    if n_vox <= k:
        return np.arange(n_vox)
    return np.sort(np.random.default_rng(seed).choice(n_vox, k, replace=False))


def nested_predict(candidates: dict, Y: np.ndarray, split: splits_mod.Split, block_ids: np.ndarray,
                   n_inner: int, alphas: np.ndarray, *, per_candidate: bool = False,
                   select_voxels: int = 2000, voxel_chunk: int = 20000):
    """Out-of-fold predictions for every sample, everything chosen on inner folds.

    Args:
        candidates: {choice: X (n, p)}; e.g. {layer: activations}. With one entry
            there is nothing to select. The candidate is chosen on the cross-fitted
            inner score; alpha is then chosen per voxel on the inner folds.
        per_candidate: also return every candidate's own out-of-fold prediction
            on the selection voxels (descriptive per-layer curve; reuses the inner scores).
        select_voxels: choosing a candidate is one global decision made on the mean
            inner r, which a fixed random voxel subset estimates well; per-voxel
            alphas for the chosen candidate then use every voxel.
    Returns:
        pred (n, n_vox), list of (test_idx, Fitted) per outer fold
        [, ({choice: pred on subset}, subset voxel idx) if per_candidate].
    """
    multi = len(candidates) > 1
    sub = _voxel_subset(Y.shape[1], select_voxels, split.seed + 3000) if multi else np.arange(Y.shape[1])
    pred = np.full(Y.shape, np.nan)
    each = {c: np.full((Y.shape[0], len(sub)), np.nan) for c in candidates} if per_candidate else None
    fitted = []
    for k, (tr, te) in enumerate(split.outer()):
        inner = splits_mod.inner(split.kind, block_ids[tr], n_inner, split.seed + 1000 + k)
        best = None
        for c, X in candidates.items():
            sc = _inner_scores(X[tr], Y[tr][:, sub], inner, alphas, voxel_chunk)
            if per_candidate:
                each[c][te] = _fit(X[tr], Y[tr][:, sub], alphas[sc.mean(0).argmax(0)], c,
                                   voxel_chunk).predict(X[te])
            score = _crossfit_score(sc) if multi else 0.0
            if best is None or score > best[0]:
                best = (score, c, sc)
        _, c, sc = best
        if len(sub) < Y.shape[1]:                        # per-voxel alpha needs every voxel
            sc = _inner_scores(candidates[c][tr], Y[tr], inner, alphas, voxel_chunk)
        f = _fit(candidates[c][tr], Y[tr], alphas[sc.mean(0).argmax(0)], c, voxel_chunk)
        pred[te] = f.predict(candidates[c][te])
        fitted.append((te, f))
    return (pred, fitted, (each, sub)) if per_candidate else (pred, fitted)


def _joint_inner_score(X, Y, inner, a_vox, penalties, llm_cols) -> np.ndarray:
    """Mean inner-validation r for each shared LLM-band penalty, shape (n_penalties,).

    Kernel form: with the LLM band at gain g the training kernel is
    g^2 K_llm + K_conf, so each distinct g costs one n x n eigendecomposition
    instead of an n x p SVD (n = a few hundred sentences, p = 1600+ features).
    """
    groups = [(a, np.flatnonzero(a_vox == a)) for a in np.unique(a_vox)]
    conf_cols = np.setdiff1d(np.arange(X.shape[1]), np.arange(X.shape[1])[llm_cols])
    score = np.zeros(len(penalties))
    for itr, iva in inner:
        mu, scale = _scaler(X[itr], llm_cols, 1.0)
        Xt, Xv = (X[itr] - mu) * scale, (X[iva] - mu) * scale
        Kl, Kc = Xt[:, llm_cols] @ Xt[:, llm_cols].T, Xt[:, conf_cols] @ Xt[:, conf_cols].T
        Vl, Vc = Xv[:, llm_cols] @ Xt[:, llm_cols].T, Xv[:, conf_cols] @ Xt[:, conf_cols].T
        Yt = Y[itr] - Y[itr].mean(0)
        cache = {}
        for li, L in enumerate(penalties):
            for a, vox in groups:
                g2 = 0.0 if np.isinf(L) else float(a / L)
                key = round(g2, 12)
                if key not in cache:
                    lam, Q = np.linalg.eigh(g2 * Kl + Kc)
                    cache[key] = (np.clip(lam, 0, None), Q, (g2 * Vl + Vc) @ Q)
                lam, Q, VQ = cache[key]
                p = VQ @ ((Q.T @ Yt[:, vox]) / (lam + a)[:, None])
                score[li] += voxelwise_r(p, Y[iva][:, vox]).sum()
    return score / (len(inner) * Y.shape[1])


def nested_predict_joint(llm: dict, Xc: np.ndarray, Y: np.ndarray, split: splits_mod.Split,
                         block_ids: np.ndarray, n_inner: int, alphas: np.ndarray, *,
                         conf_fitted: list, forced: dict, select_voxels: int = 2000,
                         voxel_chunk: int = 20000):
    """LLM + confounds, built so that Δ = r(joint) - r(confounds) is fair.

    The confound columns keep, voxel by voxel, the alpha the confound-only model
    chose in the same outer fold. The LLM band is added with ONE penalty shared by
    all voxels, picked on inner folds from {inf} + alphas; inf switches the band
    off and reproduces the confound-only model exactly, and wins ties. Letting
    the LLM band's strength vary per voxel instead lets inner-fold noise switch a
    useless band on for some voxels; on the smoke data that pushed a pure-noise
    LLM band 0.02 r *below* the confound-only model, i.e. biased Δ downwards.

    Args:
        llm: {layer: X_llm}; forced: {fold: layer} (the LLM-only model's choice).
        conf_fitted: the confound-only model's [(test_idx, Fitted)] for the same split.
        select_voxels: the shared penalty is one scalar chosen on the mean inner r,
            which a fixed random subset of this many voxels estimates well; the
            final fit uses every voxel.
    """
    penalties = np.r_[np.inf, alphas]
    pred = np.full(Y.shape, np.nan)
    fitted = []
    for k, (tr, te) in enumerate(split.outer()):
        X = np.hstack([llm[forced[k]], Xc])
        llm_cols = slice(0, llm[forced[k]].shape[1])
        a_vox = conf_fitted[k][1].alpha
        inner = splits_mod.inner(split.kind, block_ids[tr], n_inner, split.seed + 1000 + k)
        sub = _voxel_subset(Y.shape[1], select_voxels, split.seed + 2000 + k)
        sc = _joint_inner_score(X[tr], Y[tr][:, sub], inner, a_vox[sub], penalties, llm_cols)
        L = penalties[int(sc.argmax())]
        gains = np.zeros(Y.shape[1]) if np.isinf(L) else np.sqrt(a_vox / L)
        f = _fit(X[tr], Y[tr], a_vox, forced[k], voxel_chunk, llm_cols, gains, float(L))
        pred[te] = f.predict(X[te])
        fitted.append((te, f))
    return pred, fitted


def center_within_folds(a: np.ndarray, fold: np.ndarray) -> np.ndarray:
    """Remove each outer test fold's own mean before pooling folds into one series.

    Why: a fold's intercept is the mean of its *training* data, which moves
    opposite to the held-out fold's mean (the overall mean is fixed). Pooled
    across contiguous folds, that alone makes predictions anti-correlated with
    the response -- r goes negative even for a constant prediction, and the
    zero-input ablation "fails" for no reason. Demeaning prediction and response
    within each test fold scores within-fold covariation only. It also discards
    any genuine between-fold offset the model might have predicted, so it errs
    conservative.
    """
    out = np.array(a, dtype=float, copy=True)
    for k in np.unique(fold):
        m = fold == k
        out[m] -= out[m].mean(0, keepdims=True)
    return out


# ---------------------------------------------------------------- bootstrap

def _block_sums(pred: np.ndarray, y: np.ndarray, rank: np.ndarray, n_blocks: int) -> np.ndarray:
    """Per-block sufficient statistics for Pearson r: (5, n_blocks, n_vox)."""
    out = np.zeros((5, n_blocks, y.shape[1]))
    for i, v in enumerate((pred, y, pred * pred, y * y, pred * y)):
        np.add.at(out[i], rank, v)
    return out


def _r_from_sums(S: np.ndarray, n: np.ndarray) -> np.ndarray:
    """S: (5, B, v) summed with weights already; n: (B,) sample counts -> r (B, v)."""
    sp, sy, spp, syy, spy = S
    n = n[:, None]
    cov = spy - sp * sy / n
    vp = spp - sp ** 2 / n
    vy = syy - sy ** 2 / n
    den = np.sqrt(np.clip(vp * vy, 0, None))
    return np.where(den > _EPS, cov / np.where(den > _EPS, den, 1.0), 0.0)


def bootstrap(preds: dict[str, list[np.ndarray]], Y: list[np.ndarray], block_ids: np.ndarray, *,
              contrasts: dict[str, tuple[str, str]] | None = None, n_boot: int = 1000,
              ci: float = 0.95, hierarchical: bool = True, seed: int = 0) -> dict:
    """Paired block bootstrap of mean-over-voxels Pearson r.

    Resampling unit is the block (passage/story), never the voxel and never the
    single sample: samples inside a block are autocorrelated, voxels are not
    the population the claim generalises over. With several subjects, each
    replicate first resamples subjects, then draws ONE set of blocks applied to
    every drawn subject (stimuli are crossed with subjects, so drawing blocks
    independently per subject would average stimulus variance away).
    All models share the same draws, so contrasts (e.g. joint - confounds) are paired.

    Statistic: mean over subjects of (mean over voxels of r).
    """
    rank = np.r_[0, np.cumsum(np.diff(block_ids) != 0)]
    nb, ns = int(rank.max()) + 1, len(Y)
    sizes = np.bincount(rank, minlength=nb).astype(float)
    rng = np.random.default_rng(seed)
    counts = np.stack([np.bincount(rng.integers(0, nb, nb), minlength=nb) for _ in range(n_boot)]).astype(float)
    subj = (np.stack([rng.integers(0, ns, ns) for _ in range(n_boot)]) if hierarchical and ns > 1
            else np.tile(np.arange(ns), (n_boot, 1)))
    n_eff = counts @ sizes                                        # samples per replicate

    point, boot = {}, {}
    for name, per_subj in preds.items():
        per_s_point, per_s_boot = [], []
        for p, y in zip(per_subj, Y):
            per_s_point.append(voxelwise_r(p, y).mean())
            S = _block_sums(p, y, rank, nb)                       # (5, nb, v)
            step = max(1, int(2 ** 25 // (5 * y.shape[1])))       # keep (5, step, v) under ~256 MB
            per_s_boot.append(np.concatenate([
                _r_from_sums(np.einsum("rb,kbv->krv", counts[lo:lo + step], S), n_eff[lo:lo + step]).mean(1)
                for lo in range(0, n_boot, step)]))               # (n_boot,)
        per_s_boot = np.stack(per_s_boot, 1)                      # (n_boot, ns)
        point[name] = float(np.mean(per_s_point))
        boot[name] = np.take_along_axis(per_s_boot, subj, 1).mean(1)
    for cname, (a, b) in (contrasts or {}).items():
        point[cname] = point[a] - point[b]
        boot[cname] = boot[a] - boot[b]

    q = (1 - ci) / 2
    return {name: {"est": point[name],
                   "ci": [float(np.quantile(boot[name], q)), float(np.quantile(boot[name], 1 - q))],
                   "se": float(boot[name].std(ddof=1))}
            for name in point} | {"_meta": {"n_boot": n_boot, "ci": ci, "unit": "block",
                                            "hierarchical": bool(hierarchical and ns > 1),
                                            "n_blocks": nb, "n_subjects": ns}}
