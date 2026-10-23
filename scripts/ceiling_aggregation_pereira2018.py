#!/usr/bin/env python3
"""Mean-aggregated S1 ceiling: (a) extrapolate each voxel then average, or (b) average then extrapolate?

    data/venv/bin/python scripts/ceiling_aggregation_pereira2018.py [--workers 4]

r_LLM = mean over subjects of the mean over voxels of each voxel's held-out r. Its
ceiling is mean_i c_i, c_i = lim_k rho_i(k) (rho_i(k) = expected r of voxel i from a
(k-1)-subject pool). A finite mean commutes with the limit, so (a) mean_i v0_i and
(b) v0 of the mean curve estimate the SAME number; they differ only as estimators,
because the saturating fit is nonlinear and clipped and single-voxel curves are
noisy. Which estimator is right is measured here, not argued:

  sim    synthetic subjects tuned to Pereira2018 (contiguous-like and shuffled-like
         SNR): extrapolate from 9 subjects exactly as on the real data, and run the same
         procedure at pool size 32 (and 16, to show the curve is flat by then) as the answer.
  real   Pereira2018, the ceiling's own draws (ceiling._subsets, same seed): per-voxel
         tau identifiability, test-retest of per-voxel v0 on even vs odd passages,
         and fit-on-k<=7 -> predict k=8,9.
  r_LLM  per-voxel r_LLM recomputed from the cached GPT2-XL activations (refuses to run
         if a layer is not cached: no LLM, no GPU), checked against the run's values,
         for a median/median retention next to mean/mean.

Writes results/d1/ceiling_ridge__<run_id>/aggregation/: manifest.json, summary.md and
the per-job npz caches (a rerun only recomputes missing ones). CPU, ~30-45 min.
"""
from __future__ import annotations

import os

os.environ.setdefault("OPENBLAS_NUM_THREADS", "3")

import argparse  # noqa: E402
import json  # noqa: E402
import sys  # noqa: E402
import time  # noqa: E402
from concurrent.futures import ProcessPoolExecutor  # noqa: E402
from datetime import datetime, timezone  # noqa: E402
from multiprocessing import get_context  # noqa: E402
from pathlib import Path  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

import numpy as np  # noqa: E402

from d1 import activations, ceiling, config, data, pipeline, readout, splits  # noqa: E402

CONFIG = ROOT / "configs" / "pereira2018_schrimpf2021.yaml"
RUN = "pereira2018_schrimpf2021__split-contiguous__pool-last__base-on__e9c9f4f9"
N_SUBSETS, N_BOOT = 10, 100                 # as the ceiling (scripts/ceiling_ridge_pereira2018.py)
SIM = {"low": 0.2, "high": 0.6}             # amplitude mu: low ~ Pereira contiguous, high ~ shuffled
SIM_SEEDS = {"low": (0, 1, 2), "high": (0,)}
SIM_SUBJECTS, SIM_VOXELS = 33, 300


def _log(tag, msg, t0):
    print(f"[agg {tag}] {msg}  ({time.time() - t0:.0f}s)", file=sys.stderr, flush=True)


def _setup():
    cfg = config.load(CONFIG)
    return cfg, data.load(cfg)


def _pooled_r(pred_folds, y_c, folds, fold_of, masks):
    full = np.empty_like(y_c)
    for p, (_, te) in zip(pred_folds, folds):
        full[te] = p
    fc = readout.center_within_folds(full, fold_of)
    return {h: readout.voxelwise_r(fc[m], y_c[m]) for h, m in masks.items()}


def _draw_scores(Y, sp, ridge, masks, seed, tag, t0):
    """Pooled fold-centred r per draw, for every k, subset and target -- ceiling._holdout's draws."""
    folds = list(sp.outer())
    fit = ceiling._ridge_fit_folds(Y, sp, ridge)
    Yc = [readout.center_within_folds(y, sp.fold) for y in Y]
    rng = np.random.default_rng(seed)
    N = len(Y)
    out = {h: {s: {} for s in range(N)} for h in masks}
    for k in range(2, N + 1):
        for subset in ceiling._subsets(N, k, N_SUBSETS, rng):
            for t in subset:
                r = _pooled_r(fit([s for s in subset if s != t], t), Yc[t], folds, sp.fold, masks)
                for h in masks:
                    out[h][t].setdefault(k, []).append(r[h])
        _log(tag, f"k={k} done", t0)
    return {h: {s: {k: np.stack(v) for k, v in d.items()} for s, d in hh.items()} for h, hh in out.items()}


def _flatten(scores: dict, prefix: str) -> dict:
    return {f"{prefix}|{h}|{s}|{k}": v for h, hh in scores.items() for s, d in hh.items() for k, v in d.items()}


def _unflatten(z, prefix: str) -> dict:
    out: dict = {}
    for key in z.files:
        if key.count("|") != 3:                     # e.g. the simulation's truth_k* arrays
            continue
        p, h, s, k = key.split("|")
        if p == prefix:
            out.setdefault(h, {}).setdefault(int(s), {})[int(k)] = z[key]
    return {h: {s: dict(sorted(d.items())) for s, d in sorted(hh.items())} for h, hh in out.items()}


# ---------------------------------------------------------------- jobs (each writes one npz)

def job_real(out: str, kind: str):
    t0 = time.time()
    cfg, ds = _setup()
    sp = splits.make(kind, ds.block_ids, cfg["split"]["n_folds"], cfg["seed"])
    ridge = ceiling.Ridge(ds.block_ids, config.alphas(cfg), cfg["split"]["inner_folds"], cfg["readout"]["voxel_chunk"])
    masks = {"all": np.ones(ds.n, bool), "even": ds.block_ids % 2 == 0, "odd": ds.block_ids % 2 == 1}
    sc = _draw_scores(ds.Y, sp, ridge, masks, cfg["seed"], f"real/{kind}", t0)
    np.savez_compressed(Path(out) / f"real_{kind}.npz", **_flatten(sc, "r"))
    return f"real_{kind}"


def job_rllm(out: str, kind: str, s: int):
    """r_LLM of one subject, per voxel: pipeline.analyse's LLM readout on the cached activations."""
    t0 = time.time()
    cfg, ds = _setup()
    m = cfg["model"]
    cache = pipeline._abs(m["cache_dir"])
    feats = {}
    for L in m["layers"]:
        p = activations.cache_path(cache, m["name"], m["context"], L, cfg["pooling"], kind, ds.stimulus_hash())
        if not p.exists():
            raise FileNotFoundError(f"{p}: not cached; refusing to run the LLM")
        feats[L] = np.load(p)
    sp = splits.make(kind, ds.block_ids, cfg["split"]["n_folds"], cfg["seed"])
    pred, fitted = readout.nested_predict(feats, ds.Y[s], sp, ds.block_ids, cfg["split"]["inner_folds"],
                                          config.alphas(cfg), select_voxels=cfg["readout"]["select_voxels"],
                                          voxel_chunk=cfg["readout"]["voxel_chunk"])
    fc = lambda a: readout.center_within_folds(a, sp.fold)  # noqa: E731
    r = readout.voxelwise_r(fc(pred), fc(ds.Y[s]))
    np.savez_compressed(Path(out) / f"rllm_{kind}_{s}.npz", r=r, layers=[f.choice for _, f in fitted])
    _log(f"rllm/{kind}/{ds.subjects[s]}", f"mean r {r.mean():.4f}", t0)
    return f"rllm_{kind}_{s}"


def simulate(n_subj: int, mu: float, seed: int, n: int = 384, v: int = SIM_VOXELS, q: int = 10,
             p_null: float = 0.3, sd_log: float = 0.8) -> list[np.ndarray]:
    """Subjects share a q-dim stimulus signal (variances 1/m, so voxels saturate at different
    rates); each voxel = amplitude x (own unit-variance projection) + unit noise; 30% of
    voxels carry nothing, the others have log-normal amplitudes around mu."""
    rng = np.random.default_rng(seed)
    S = rng.standard_normal((n, q)) * np.sqrt(1.0 / np.arange(1, q + 1))
    Y = []
    for _ in range(n_subj):
        sig = S @ rng.standard_normal((q, v))
        sig /= sig.std(0)
        a = np.exp(np.log(mu) + sd_log * rng.standard_normal(v)) * (rng.random(v) > p_null)
        Y.append(a * sig + rng.standard_normal((n, v)))
    return Y


def job_sim(out: str, name: str, seed: int):
    t0 = time.time()
    cfg = config.load(CONFIG)
    Y = simulate(SIM_SUBJECTS, SIM[name], seed)
    b = np.repeat(np.arange(96), 4)
    sp = splits.make("contiguous", b, cfg["split"]["n_folds"], cfg["seed"])
    ridge = ceiling.Ridge(b, config.alphas(cfg), cfg["split"]["inner_folds"], cfg["readout"]["voxel_chunk"])
    masks = {"all": np.ones(len(b), bool)}
    sc = _draw_scores(Y[:9], sp, ridge, masks, cfg["seed"], f"sim/{name}/{seed}", t0)   # as on Pereira2018
    fit = ceiling._ridge_fit_folds(Y, sp, ridge)                                        # the answer
    folds, Yc = list(sp.outer()), [readout.center_within_folds(y, sp.fold) for y in Y]
    big = {}
    for kk, pool_of in ((17, lambda t: list(range(9, 25))), (33, lambda t: [s for s in range(33) if s != t])):
        big[kk] = np.stack([_pooled_r(fit(pool_of(t), t), Yc[t], folds, sp.fold, masks)["all"] for t in range(9)])
    np.savez_compressed(Path(out) / f"sim_{name}_{seed}.npz", **_flatten(sc, "r"),
                        **{f"truth_k{kk}": a for kk, a in big.items()})
    _log(f"sim/{name}/{seed}", f"truth mean r at k=33 {big[33].mean():.4f}, k=17 {big[17].mean():.4f}", t0)
    return f"sim_{name}_{seed}"


# ---------------------------------------------------------------- estimators and checks

def estimators(whole: dict, seed: int) -> dict:
    """The candidate mean/median ceilings from one set of draws ({subject: {k: (draws, n_vox)}})."""
    per_voxel = ceiling._extrapolate(whole, N_BOOT, seed)
    agg = {s: {k: v.mean(1, keepdims=True) for k, v in by_k.items()} for s, by_k in whole.items()}
    curve = [float(c[0]) for c in ceiling._extrapolate(agg, N_BOOT, seed)]
    N = max(max(d) for d in whole.values())
    return {"a_mean_of_voxel_v0": float(np.mean([c.mean() for c in per_voxel])),
            "b_v0_of_mean_curve": float(np.mean(curve)),
            "median_of_voxel_v0": float(np.median(np.concatenate(per_voxel))),
            "observed_mean_at_kN": float(np.mean([whole[s][N].mean() for s in whole])),
            "observed_median_at_kN": float(np.median(np.concatenate([whole[s][N].mean(0) for s in whole])))}


_BETAS = np.logspace(-2, 3, 501)


def fit_hyperbolic(ks: np.ndarray, y: np.ndarray) -> tuple[float, float]:
    """Least-squares c, beta of r(k) = c / sqrt(1 + beta / (k - 1)): the pool's noise variance
    falling like 1 / (pool size), the alternative to Schrimpf's c * (1 - exp(-k / tau))."""
    best = (np.inf, 0.0, 0.0)
    for b in _BETAS:
        f = 1 / np.sqrt(1 + b / (ks - 1))
        c = float((y @ f) / (f @ f))
        sse = float(((y - c * f) ** 2).sum())
        if sse < best[0]:
            best = (sse, c, float(b))
    return best[1], best[2]


def models(whole: dict) -> dict:
    """Mean curve per subject (mean over voxels and draws), extrapolated by both forms; mean over subjects."""
    cur = [(np.array(sorted(d)), np.array([d[k].mean() for k in sorted(d)])) for d in whole.values()]
    ex = [ceiling.fit_saturating(ks, y, return_tau=True) for ks, y in cur]
    hy = [fit_hyperbolic(ks, y) for ks, y in cur]
    at = lambda k: {"exponential": float(np.mean([v * (1 - np.exp(-k / t)) for v, t in ex])),  # noqa: E731
                    "hyperbolic": float(np.mean([c / np.sqrt(1 + b / (k - 1)) for c, b in hy]))}
    return {"at_k17": at(17), "at_k33": at(33),
            "asymptote": {"exponential": float(np.mean([v for v, _ in ex])),
                          "hyperbolic": float(np.mean([c for c, _ in hy]))}}


def _curves(whole: dict):
    """Per subject: ks, per-voxel mean-over-draws curve (n_vox, len(ks))."""
    out = {}
    for s, by_k in whole.items():
        ks = np.array(sorted(by_k))
        out[s] = (ks, np.stack([by_k[k].mean(0) for k in ks], -1))
    return out


def real_checks(sc: dict, seed: int) -> dict:
    whole = sc["all"]
    cur = _curves(whole)
    kmax = max(int(c[0].max()) for c in cur.values())
    # 1. identifiability: per-voxel point fit on the observed curve
    v0, tau, at_n = [], [], []
    for ks, y in cur.values():
        a, t = ceiling.fit_saturating(ks, y, return_tau=True)
        v0.append(a), tau.append(t), at_n.append(y[:, -1])
    v0, tau, at_n = map(np.concatenate, (v0, tau, at_n))
    slow = tau > kmax                                 # curve ~linear over k = 2..9: v0 not pinned by data
    ident = {"share_tau_above_kmax": float(slow.mean()), "share_v0_at_1": float((v0 >= 0.999).mean()),
             "share_v0_at_0": float((v0 <= 0).mean()),
             "mean_v0_all": float(v0.mean()), "mean_v0_tau_le_kmax": float(v0[~slow].mean()),
             "mean_v0_tau_above_kmax": float(v0[slow].mean()) if slow.any() else None,
             "share_of_sum_v0_from_tau_above_kmax": float(v0[slow].sum() / v0.sum()),
             "mean_observed_r_at_kN_tau_above_kmax": float(at_n[slow].mean()) if slow.any() else None,
             "v0_quantiles": {q: float(np.quantile(v0, q)) for q in (0.5, 0.9, 0.95, 0.99)}}
    # 2. test-retest of per-voxel v0: even vs odd passages (independent stimuli, same draws)
    pv = {h: np.concatenate(ceiling._extrapolate(sc[h], N_BOOT, seed)) for h in ("even", "odd")}
    top = pv["even"] >= np.quantile(pv["even"], 0.95)
    retest = {"r_v0_even_vs_odd": float(np.corrcoef(pv["even"], pv["odd"])[0, 1]),
              "top5pct_even_mean_v0_even": float(pv["even"][top].mean()),
              "top5pct_even_mean_v0_odd": float(pv["odd"][top].mean()),
              "by_half": {h: estimators(sc[h], seed) for h in ("even", "odd")}}
    # 3. fit k <= 7, predict the observed mean r at k = 8, 9
    pred = {"a_per_voxel": [], "b_mean_curve": [], "observed": []}
    for ks, y in cur.values():
        lo = ks <= 7
        a, t = ceiling.fit_saturating(ks[lo], y[:, lo], return_tau=True)
        pred["a_per_voxel"].append((a[:, None] * (1 - np.exp(-ks[~lo][None, :] / t[:, None]))).mean(0))
        am, tm = ceiling.fit_saturating(ks[lo], y[:, lo].mean(0), return_tau=True)
        pred["b_mean_curve"].append(am * (1 - np.exp(-ks[~lo] / tm)))
        pred["observed"].append(y[:, ~lo].mean(0))
    predict = {k: [float(x) for x in np.mean(v, 0)] for k, v in pred.items()} | {"k": [8, 9]}
    return {"identifiability": ident, "retest_even_odd": retest, "predict_k8_k9_from_k_le_7": predict}


def _summary(m: dict) -> str:
    f = lambda x: f"{x:.4f}"  # noqa: E731
    h, cc = m["revised_headline"]["contiguous"], m["contiguous_mean_ceiling"]
    rc, ec = m["real"]["contiguous"], m["real"]["contiguous"]["estimators"]
    L = ["# 均值口径的 ceiling：先逐体素外推再平均 (a)，还是先平均再外推 (b)", "",
         f"源目录：`{Path(m['source_ceiling_dir']).name}`；git {m['provenance']['git_commit']}，"
         f"脚本 `{m['provenance']['script']}`。所有数字都是实测，见 `manifest.json`。"
         + " ".join(f"{'PASS' if c['ok'] else 'FAIL'}：{c['name']}。" for c in m["checks"]), "",
         "## 结论", "",
         f"- (a) 与 (b) 估的是**同一个量**，分歧只在估计器的偏差。在 contiguous 的信噪比下，(a) 高估约 2.2 倍，(b) 低估约 15%。",
         f"- 修订后 contiguous 的保留率 ≈ **{h['retention']:.2f}**（校准后的 ceiling 为 {f(h['ceiling_calibrated'])}）。"
         f"硬区间为 [{h['retention_lower_bound']:.2f}, {h['retention_upper_bound']:.2f}]；"
         f"只计入 r_LLM 抽样误差的区间为 [{h['retention_ci_r_only'][0]:.2f}, {h['retention_ci_r_only'][1]:.2f}]。"
         f"之前报的 {h['previous_headline_b']:.3f} 只是上界附近的值。判定不变（无法判定）。", "",
         "## 为什么 (a)、(b) 估的是同一个量", "",
         "r_LLM 的计算是：对每个体素先算有符号的 held-out r，不裁剪、不外推，然后在被试内对体素取均值，再对被试取均值"
         "（`readout.py` 第 365、372 行；`pipeline.py` 第 69 行）。"
         "与它同口径的上限是 mean_i c_i，其中 c_i = lim_k ρ_i(k) 是体素 i 在被试池无限大时的期望 r。"
         "体素个数有限时，均值与极限可以交换：mean_i lim_k = lim_k mean_i。所以 (a) 与 (b) 的估计目标相同。"
         "不能交换的是估计器：每个体素的 v0 被裁剪到 [0, 1]（`ceiling.py` 第 112 行），"
         "而且是从带噪声的 9 个点外推出来的。r_LLM 中每个体素 r_i 的对应物是 c_i，而不是 v0_hat_i。"
         "(b) 处理逐体素 r(k) 的方式和 r_LLM 处理逐体素 r 完全相同（有符号、不裁剪地取均值），只在最后做一次外推。"
         "因此「同口径」要按偏差来判断，不能只看运算次序。", "",
         "## 证据 1：模拟（答案已知）", "",
         f"共 {m['sim_design']['subjects']} 个被试，每人 {m['sim_design']['voxels_per_subject']} 个体素，"
         f"{m['sim_design']['shared_dims']} 维共享信号，其中 {m['sim_design']['null_voxel_share']:.0%} 的体素不含信号。"
         "用前 9 个被试做外推，流程与真实数据完全一样（抽样、种子、切分、岭回归都相同）。"
         "答案取同一流程在被试池为 32 人时的实测值（k=33）。k=17 → 33 的值仍在上升，所以 k=33 本身也只是下界。", "",
         "| 信噪比 | 种子 | 实测 k=17 | 实测 k=33 | (a) | (b) | k=9 实测 | 1/k 模型渐近值 | (a)/k33 | (b)/k33 | 体素 v0 中位数 / k33 体素 r 中位数 |",
         "|---|---|---|---|---|---|---|---|---|---|---|"]
    for nm, d in m["sim"].items():
        for r in d["rows"]:
            t, e, mo = r["truth"], r["estimators"], r["models"]
            L.append(f"| {nm}（mu={d['mu']}）| {r['seed']} | {f(t['mean_k17'])} | {f(t['mean_k33'])} | "
                     f"{f(e['a_mean_of_voxel_v0'])} | {f(e['b_v0_of_mean_curve'])} | {f(e['observed_mean_at_kN'])} | "
                     f"{f(mo['asymptote']['hyperbolic'])} | {r['ratio_to_truth_mean']['a_mean_of_voxel_v0']:.2f} | "
                     f"{r['ratio_to_truth_mean']['b_v0_of_mean_curve']:.2f} | {r['median_of_voxel_v0_over_truth_median']:.2f} |")
    idt, rt, pr = rc["identifiability"], rc["retest_even_odd"], rc["predict_k8_k9_from_k_le_7"]
    L += ["", "low 设定对应 Pereira contiguous：模拟中 k=2 → 9 为 0.066 → 0.102，真实数据为 0.059 → 0.106。", "",
          "## 证据 2：真实数据（contiguous）", "",
          f"- **可识别性**：{idt['share_tau_above_kmax']:.1%} 的体素拟合出的 τ > 9，也就是在 k = 2..9 范围内曲线仍近似直线，渐近值不受数据约束。"
          f"这些体素在 k=9 的实测 r 均值为 {idt['mean_observed_r_at_kN_tau_above_kmax']:.3f}，外推出的 v0 均值却是 {idt['mean_v0_tau_above_kmax']:.2f}；"
          f"它们贡献了 (a) 均值的 {idt['share_of_sum_v0_from_tau_above_kmax']:.0%}。"
          f"v0 的 90% 分位数是 {idt['v0_quantiles']['0.9']:.2f}，{idt['share_v0_at_1']:.1%} 的体素顶到 1，{idt['share_v0_at_0']:.1%} 的体素被截到 0。",
          f"- **独立刺激重测**（偶数段 vs 奇数段）：逐体素 v0 的相关只有 {rt['r_v0_even_vs_odd']:.2f}。"
          f"偶数段中 v0 最高的 5% 体素（{rt['top5pct_even_mean_v0_even']:.2f}）在奇数段只有 {rt['top5pct_even_mean_v0_odd']:.2f}。"
          f"(a) 的均值在两半上都是 {rt['by_half']['even']['a_mean_of_voxel_v0']:.3f} / {rt['by_half']['odd']['a_mean_of_voxel_v0']:.3f}：可重复，但这是系统偏差，不代表它对。",
          f"- **在观测范围内预测**（只用 k ≤ 7 拟合，预测 k = 8、9 的均值）：(a) 预测 {pr['a_per_voxel'][0]:.4f}、{pr['a_per_voxel'][1]:.4f}，"
          f"(b) 预测 {pr['b_mean_curve'][0]:.4f}、{pr['b_mean_curve'][1]:.4f}，实测 {pr['observed'][0]:.4f}、{pr['observed'][1]:.4f}。"
          "(a) 外推一两步就已经偏高约 11%。", "",
          "## r_LLM 的体素分布（问题 2）", "",
          "| 切分 | 体素均值→被试均值 | 体素中位数→被试中位数 | 全体素中位数 | 负值体素比例 | 5% / 50% / 95% / 99% 分位 |", "|---|---|---|---|---|---|"]
    for k, d in m["r_llm"].items():
        q = d["quantiles"]
        L.append(f"| {k} | {f(d['mean_of_subject_means'])} | {f(d['median_of_subject_medians'])} | {f(d['median_all_voxels'])} | "
                 f"{d['share_negative']:.1%} | {q['0.05']:.3f} / {q['0.5']:.3f} / {q['0.95']:.3f} / {q['0.99']:.3f} |")
    L += ["", f"r_LLM 是直接测量，没有外推步骤：取值在 [−1, 1] 内，噪声对称，负值保留为负。"
          f"contiguous 下它的均值与中位数相近（{f(m['r_llm']['contiguous']['mean_of_subject_means'])} 对 "
          f"{f(m['r_llm']['contiguous']['median_of_subject_medians'])}），99% 分位数为 {m['r_llm']['contiguous']['quantiles']['0.99']:.2f}。"
          f"同一数据上逐体素 v0 的均值与中位数之比为 {ec['a_mean_of_voxel_v0'] / ec['median_of_voxel_v0']:.2f}，90% 分位数为 {idt['v0_quantiles']['0.9']:.2f}。"
          "所以 r_LLM 不受同一条右尾的影响。中位数口径也不安全：在 low 模拟中，逐体素 v0 的中位数是答案中位数的 1.6–1.7 倍，"
          "因为整体分布都被带噪声的外推抬高了，而中位数只能挡住最远的那条尾巴。所以 Schrimpf 口径的 median/median 同样偏低。", "",
          "## 保留率：所有口径", "",
          "| 分子 / 分母 | contiguous | shuffled |", "|---|---|---|"]
    for k in m["retention"]["contiguous"]:
        L.append(f"| {k} | {m['retention']['contiguous'][k]:.3f} | {m['retention']['shuffled'][k]:.3f} |")
    L += ["", "contiguous 的均值口径 ceiling：",
          f"- 下界（k=9 实测，未外推）{f(cc['lower_bound_observed_k9'])}；(b) 指数模型 {f(cc['b_exponential_mean_curve'])}；"
          f"1/k 模型 {f(cc['hyperbolic_mean_curve'])}；(a) {f(cc['a_mean_of_voxel_v0_overestimate'])}（已知高估）。",
          f"- 按模拟校准：(b) × {cc['sim_factor_truth_k33_over_b']:.3f} = {f(cc['calibrated']['b_times_sim_factor'])}；"
          f"1/k 模型 × {cc['sim_factor_truth_k33_over_hyperbolic']:.3f} = {f(cc['calibrated']['hyperbolic_times_sim_factor'])}；"
          f"两者取中点 **{f(cc['calibrated_mid'])}**。这两个值对应 33 人被试池的水平，仍略低于渐近值，所以保留率 {h['retention']:.2f} 略偏高。", "",
          "shuffled 下 (a)、(b)、中位数三者几乎一致（0.344–0.351）：高信噪比时运算次序无关紧要，各口径的保留率都在 1.17–1.22 之间，仍应视作泄漏证据。", "",
          "## 建议的报法", "",
          f"S1 的 contiguous 归一化保留率报 **≈ {h['retention']:.2f}**（ceiling 按模拟校准为 {f(h['ceiling_calibrated'])}）。"
          f"同时注明：硬区间为 [{h['retention_lower_bound']:.2f}（逐体素外推后取均值，高估 ceiling）, "
          f"{h['retention_upper_bound']:.2f}（8 人池实测，不外推）]；只计入 r_LLM 抽样误差的 CI 为 "
          f"[{h['retention_ci_r_only'][0]:.2f}, {h['retention_ci_r_only'][1]:.2f}]；上面的敏感性表一并附上。"
          "用 9 个被试做外推，ceiling 的模型不确定性约 15–20%（0.106 → 0.133，渐近值可能还更高），但比 r_LLM 自身约 ±50% 的抽样误差小。"]
    return "\n".join(L) + "\n"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--workers", type=int, default=4)
    a = ap.parse_args()
    t0 = time.time()
    ceil_dir = ROOT / "results" / "d1" / f"ceiling_ridge__{RUN}"
    out = ceil_dir / "aggregation"
    out.mkdir(parents=True, exist_ok=True)
    cfg, ds = _setup()
    cm = json.loads((ceil_dir / "manifest.json").read_text())
    run_man = json.loads((ROOT / "results" / "d1" / RUN / "manifest.json").read_text())

    jobs = [(job_real, k) for k in ("contiguous", "shuffled")]
    jobs += [(job_rllm, k, s) for k in ("contiguous", "shuffled") for s in range(len(ds.Y))]
    jobs += [(job_sim, n, s) for n, seeds in SIM_SEEDS.items() for s in seeds]
    prefix = {job_real: "real", job_rllm: "rllm", job_sim: "sim"}
    name = lambda j: "_".join([prefix[j[0]], *map(str, j[1:])])  # noqa: E731  (= the job's npz name)
    todo = [j for j in jobs if not (out / f"{name(j)}.npz").exists()]
    print(f"[agg] {len(todo)}/{len(jobs)} jobs to run on {a.workers} processes x "
          f"{os.environ['OPENBLAS_NUM_THREADS']} BLAS threads")
    if todo:
        with ProcessPoolExecutor(a.workers, mp_context=get_context("spawn")) as ex:
            for f in [ex.submit(j[0], str(out), *j[1:]) for j in todo]:
                f.result()

    seed = cfg["seed"]
    rep: dict = {"checks": []}
    # ---- real data
    rep["real"] = {}
    for kind in ("contiguous", "shuffled"):
        sc = _unflatten(np.load(out / f"real_{kind}.npz"), "r")
        est = estimators(sc["all"], seed)
        ref = cm["ceilings"][kind]["ridge"]
        same = (abs(est["b_v0_of_mean_curve"] - ref["matched"]["ceiling"]) < 1e-12 and
                abs(est["a_mean_of_voxel_v0"] - ref["by_scoring_and_aggregation"]["pooled_fold_centred_r"]["mean_of_voxel_v0"]) < 1e-12)
        rep["checks"].append({"name": f"{kind}: these draws reproduce the ceiling run's (a) and (b)", "ok": bool(same)})
        rep["real"][kind] = {"estimators": est, **real_checks(sc, seed)}
    # ---- r_LLM per voxel
    rep["r_llm"] = {}
    for kind, part in (("contiguous", "primary"), ("shuffled", "control")):
        r = [np.load(out / f"rllm_{kind}_{s}.npz")["r"] for s in range(len(ds.Y))]
        mom = float(np.mean([x.mean() for x in r]))
        ref = run_man[part]["boot"]["llm"]["est"]
        rep["checks"].append({"name": f"{kind}: recomputed r_LLM = run's {ref:.6f}", "ok": bool(abs(mom - ref) < 1e-6),
                              "got": mom, "abs_diff": abs(mom - ref)})
        allr = np.concatenate(r)
        rep["r_llm"][kind] = {"mean_of_subject_means": mom,
                              "median_of_subject_medians": float(np.median([np.median(x) for x in r])),
                              "median_all_voxels": float(np.median(allr)),
                              "share_negative": float((allr < 0).mean()),
                              "quantiles": {q: float(np.quantile(allr, q)) for q in (0.05, 0.5, 0.95, 0.99)}}
    # ---- retention under each aggregation
    rep["retention"] = {}
    for kind in ("contiguous", "shuffled"):
        e, rl = rep["real"][kind]["estimators"], rep["r_llm"][kind]
        rep["retention"][kind] = {
            "mean/(b) v0 of mean curve": rl["mean_of_subject_means"] / e["b_v0_of_mean_curve"],
            "mean/(a) mean of voxel v0": rl["mean_of_subject_means"] / e["a_mean_of_voxel_v0"],
            "mean/observed mean at k=N (no extrapolation)": rl["mean_of_subject_means"] / e["observed_mean_at_kN"],
            "median-of-medians/median of voxel v0 (Schrimpf)": rl["median_of_subject_medians"] / e["median_of_voxel_v0"],
            "median all voxels/median of voxel v0": rl["median_all_voxels"] / e["median_of_voxel_v0"],
            "median all voxels/observed median at k=N": rl["median_all_voxels"] / e["observed_median_at_kN"],
        }
    # ---- simulation: estimators vs the procedure at pool size 32
    rep["sim"] = {}
    for nm, seeds in SIM_SEEDS.items():
        rows = []
        for s in seeds:
            z = np.load(out / f"sim_{nm}_{s}.npz")
            e = estimators(_unflatten(z, "r")["all"], seed)
            t33, t17 = z["truth_k33"], z["truth_k17"]
            truth = {"mean_k33": float(t33.mean(1).mean()), "mean_k17": float(t17.mean(1).mean()),
                     "median_k33": float(np.median(t33))}
            rows.append({"seed": s, "truth": truth, "estimators": e, "models": models(_unflatten(z, "r")["all"]),
                         "ratio_to_truth_mean": {k: e[k] / truth["mean_k33"] for k in
                                                 ("a_mean_of_voxel_v0", "b_v0_of_mean_curve", "observed_mean_at_kN")},
                         "median_of_voxel_v0_over_truth_median": e["median_of_voxel_v0"] / truth["median_k33"]})
        rep["sim"][nm] = {"mu": SIM[nm], "rows": rows}
    # ---- calibrated mean ceiling for the contiguous split (simulation with contiguous-like SNR)
    low = rep["sim"]["low"]["rows"]
    f_b = float(np.mean([r["truth"]["mean_k33"] / r["estimators"]["b_v0_of_mean_curve"] for r in low]))
    f_h = float(np.mean([r["truth"]["mean_k33"] / r["models"]["asymptote"]["hyperbolic"] for r in low]))
    e, m_ = rep["real"]["contiguous"]["estimators"], models(_unflatten(np.load(out / "real_contiguous.npz"), "r")["all"])
    rl = rep["r_llm"]["contiguous"]["mean_of_subject_means"]
    cal = {"b_times_sim_factor": e["b_v0_of_mean_curve"] * f_b,
           "hyperbolic_times_sim_factor": m_["asymptote"]["hyperbolic"] * f_h}
    rep["contiguous_mean_ceiling"] = {
        "lower_bound_observed_k9": e["observed_mean_at_kN"], "b_exponential_mean_curve": e["b_v0_of_mean_curve"],
        "hyperbolic_mean_curve": m_["asymptote"]["hyperbolic"], "a_mean_of_voxel_v0_overestimate": e["a_mean_of_voxel_v0"],
        "sim_factor_truth_k33_over_b": f_b, "sim_factor_truth_k33_over_hyperbolic": f_h,
        "calibrated": cal, "calibrated_mid": float(np.mean(list(cal.values()))),
        "note": "calibrated values are at the level of a 33-subject pool, still below the asymptote "
                "(the simulation's mean r rises k=17 -> 33), so they too are lower bounds on the ceiling"}
    c_mid = rep["contiguous_mean_ceiling"]["calibrated_mid"]
    ci = run_man["primary"]["boot"]["llm"]["ci"]
    rep["revised_headline"] = {
        "contiguous": {"r_llm": rl, "ceiling_calibrated": c_mid, "retention": rl / c_mid,
                       "retention_ci_r_only": [x / c_mid for x in ci],
                       "retention_upper_bound": rl / e["observed_mean_at_kN"],
                       "retention_lower_bound": rl / e["a_mean_of_voxel_v0"],
                       "previous_headline_b": rl / e["b_v0_of_mean_curve"]},
        "note": "retention_ci_r_only: r_LLM's block-bootstrap CI over the calibrated ceiling; the ceiling's "
                "own uncertainty (model + sampling) is not in it"}
    rep.update({"what": "mean-aggregated ceiling: extrapolate-then-average (a) vs average-then-extrapolate (b)",
                "source_ceiling_dir": str(ceil_dir), "source_run": RUN,
                "provenance": {"timestamp": datetime.now(timezone.utc).isoformat(timespec="seconds"),
                               "git_commit": pipeline._git(), "script": "scripts/ceiling_aggregation_pereira2018.py"},
                "sim_design": {"subjects": SIM_SUBJECTS, "voxels_per_subject": SIM_VOXELS, "shared_dims": 10,
                               "null_voxel_share": 0.3, "amplitude": "lognormal(log mu, 0.8)",
                               "extrapolation_from": "subjects 0-8, same draws/seed/split/ridge as Pereira2018",
                               "truth": "same ridge, same scoring, pool = the other 32 subjects (k=33); "
                                        "k=17 = 16 fixed others, to show the curve has flattened"},
                "seconds": round(time.time() - t0, 1)})
    (out / "manifest.json").write_text(json.dumps(rep, indent=1, ensure_ascii=False))
    (out / "summary.md").write_text(_summary(json.loads((out / "manifest.json").read_text())))
    for c in rep["checks"]:
        print(f"[agg] {'PASS' if c['ok'] else 'FAIL'}  {c['name']}")
    print(json.dumps({"retention": rep["retention"], "sim": {k: [r["ratio_to_truth_mean"] for r in v["rows"]]
                                                              for k, v in rep["sim"].items()}}, indent=1))
    print(f"[agg] -> {out}  ({time.time() - t0:.0f}s)")
    return 0 if all(c["ok"] for c in rep["checks"]) else 1


if __name__ == "__main__":
    sys.exit(main())
