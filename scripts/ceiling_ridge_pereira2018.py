#!/usr/bin/env python3
"""Ridge noise ceiling for S1 (Pereira2018 384sentences) and the retention r_LLM / ceiling.

    data/venv/bin/python scripts/ceiling_ridge_pereira2018.py \
        [--run results/d1/<run_id>] [--no-schrimpf] [--workers 3] [--summary-only]

r_LLM comes from the readout (src/d1/readout.py: ridge, per-voxel alpha chosen on
inner folds, out-of-fold predictions fold-centred, mean over voxels and subjects).
The Schrimpf ceiling (ceiling.extrapolation_ceiling) is OLS, scored per fold,
median over voxels. On Pereira2018 the ridge r beats the OLS ceiling, so r / OLS
ceiling is not a retention. This script recomputes the ceiling with the readout's
own estimator on the source run's own folds (checked fold by fold against its
split_<kind>.json), inner folds, alphas and seed:

  settings  contiguous 8-fold (primary), shuffled 8-fold (the run's control)
            [+ shuffled 5-fold = Schrimpf protocol, reference only]
  x         OLS (as before) and ridge (ceiling.Ridge)
  x         scoring/aggregation: per fold + median over voxels (Schrimpf) and
            pooled fold-centred + mean over voxels and subjects (= r_LLM, "matched")

and divides the run's r_LLM by each. Writes results/d1/ceiling_ridge__<run_id>/:
manifest.json, retention.tsv, per_voxel.npz, summary.md. CPU only, no LLM,
nothing trained; ~15-25 min with 3 processes x 3 BLAS threads.
"""
from __future__ import annotations

import os

os.environ.setdefault("OPENBLAS_NUM_THREADS", "3")   # small matrices: more threads per process only hurts

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

from d1 import ceiling, config, data, pipeline, splits  # noqa: E402

CONFIG = ROOT / "configs" / "pereira2018_schrimpf2021.yaml"
RUN = ROOT / "results" / "d1" / "pereira2018_schrimpf2021__split-contiguous__pool-last__base-on__e9c9f4f9"
# OLS ceilings measured on 2026-10-17 (scripts/check_pereira2018.py); recomputed here
PREVIOUS_OLS = {"contiguous": 0.0951, "schrimpf": 0.3225}


def _settings(cfg: dict, schrimpf: bool) -> dict:
    n = cfg["split"]["n_folds"]
    out = {"contiguous": ("contiguous", n, "primary split of the run"),
           "shuffled": ("shuffled", n, "control split of the run")}
    if schrimpf:
        out["schrimpf"] = ("shuffled", ceiling.SCHRIMPF_N_FOLDS, "Schrimpf protocol, reference only (no r_LLM)")
    return out


def _job(cfg_path: str, setting: str, kind: str, n_folds: int, estimator: str, n_subsets: int, n_boot: int):
    t0 = time.time()
    cfg = config.load(cfg_path)
    ds = data.load(cfg)
    sp = splits.make(kind, ds.block_ids, n_folds, cfg["seed"])
    ridge = (ceiling.Ridge(ds.block_ids, config.alphas(cfg), cfg["split"]["inner_folds"],
                           cfg["readout"]["voxel_chunk"]) if estimator == "ridge" else None)

    def log(msg):
        print(f"[ceiling {setting}/{estimator}] {msg}  ({time.time() - t0:.0f}s)", file=sys.stderr, flush=True)
    c = ceiling.extrapolation_ceiling(ds.Y, sp, n_subsets=n_subsets, n_boot=n_boot, seed=cfg["seed"],
                                      subjects=ds.subjects, ridge=ridge, matched=True, log=log)
    arrays = {"per_fold_median": c.pop("per_voxel"), "matched": c["matched"].pop("per_voxel")}
    c["seconds"] = round(time.time() - t0, 1)
    log(f"ceiling {c['ceiling']:.4f}, matched {c['matched']['ceiling']:.4f}")
    return setting, estimator, c, arrays


def _check_run(run: Path, cfg: dict, ds: data.Dataset) -> tuple[list, dict, dict]:
    """The ceiling must be computed on the run's own folds and settings; returns (checks, row, manifest)."""
    man = json.loads((run / "manifest.json").read_text())
    row = json.loads((run / "rows.jsonl").read_text().splitlines()[0])
    rc = man["config"]
    checks = [
        ("same split settings (n_folds, inner_folds, seed)",
         (rc["split"]["n_folds"], rc["split"]["inner_folds"], rc["seed"])
         == (cfg["split"]["n_folds"], cfg["split"]["inner_folds"], cfg["seed"])),
        ("same alpha grid", np.array_equal(config.alphas(rc), config.alphas(cfg))),
        ("same dataset release and stimuli",
         (man["dataset_meta"]["sha1"], row["stimulus_hash"]) == (ds.meta["sha1"], ds.stimulus_hash())),
        ("same experiment and passage order",
         (rc["dataset"]["experiment"], rc["dataset"]["passage_order"])
         == (cfg["dataset"]["experiment"], cfg["dataset"]["passage_order"])),
    ]
    for kind in ("contiguous", "shuffled"):
        sp = splits.make(kind, ds.block_ids, cfg["split"]["n_folds"], cfg["seed"])
        saved = json.loads((run / f"split_{kind}.json").read_text())["fold_of_sample"]
        checks.append((f"{kind} folds identical to the run's split_{kind}.json", saved == sp.fold.tolist()))
    checks.append(("verdict row r_LLM = bootstrap point estimate",
                   (row["r_llm"], row["control_r_llm"]) == (man["primary"]["boot"]["llm"]["est"],
                                                            man["control"]["boot"]["llm"]["est"])))
    return checks, row, man


MODELS = {"llm": "r_LLM", "conf": "r_conf (position + word rate)", "joint": "r_joint (LLM + confounds)",
          "delta": "Δ = r_joint − r_conf", "oasm": "r_OASM (passage identity + position, no language)"}


def _numerators(run: Path, man: dict) -> dict:
    """Every model score of the run (its block-bootstrap point estimate and CI) + the best layer."""
    curve = json.loads((run / "layer_curve.json").read_text())
    out = {}
    for kind, part in (("contiguous", "primary"), ("shuffled", "control")):
        boot = man[part]["boot"]
        assert man[part]["split_kind"] == kind
        out[kind] = [{"name": MODELS[k], "key": k, "r": boot[k]["est"], "ci": boot[k]["ci"],
                      "note": f"{part} split of the run, block bootstrap; mean over voxels then subjects"}
                     for k in MODELS if k in boot]
        best = max(curve[kind], key=lambda L: curve[kind][L])
        out[kind].append({"name": f"r_LLM best layer (L{best})", "key": "llm_best_layer", "r": curve[kind][best],
                          "ci": None, "note": "layer-curve argmax: layer picked after seeing the test r, on the "
                                              "2000-voxel selection subset (descriptive, optimistic)"})
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default=str(CONFIG))
    ap.add_argument("--run", default=str(RUN), help="D1 run directory whose r_LLM is divided")
    ap.add_argument("--no-schrimpf", action="store_true", help="skip the shuffled 5-fold reference setting")
    ap.add_argument("--workers", type=int, default=3)
    ap.add_argument("--n-subsets", type=int, default=10)
    ap.add_argument("--n-boot", type=int, default=100)
    ap.add_argument("--out", help="output directory (default results/d1/ceiling_ridge__<run>)")
    ap.add_argument("--summary-only", action="store_true",
                    help="only re-render summary.md from the output directory's manifest.json")
    a = ap.parse_args()
    t0 = time.time()
    if a.summary_only:
        out = Path(a.out) if a.out else ROOT / "results" / "d1" / f"ceiling_ridge__{Path(a.run).resolve().name}"
        (out / "summary.md").write_text(_summary(json.loads((out / "manifest.json").read_text()), _agg(out)))
        print(f"[ceiling] {out / 'summary.md'} re-rendered from manifest.json")
        return 0
    cfg = config.load(a.config)
    ds = data.load(cfg)
    run = Path(a.run).resolve()
    checks, row, run_man = _check_run(run, cfg, ds)
    for name, ok in checks:
        print(f"[ceiling] {'PASS' if ok else 'FAIL'}  {name}")
    if not all(ok for _, ok in checks):
        print("[ceiling] the run does not match this config/data: refusing to divide", file=sys.stderr)
        return 1

    settings = _settings(cfg, not a.no_schrimpf)
    jobs = [(s, e) for e in ("ridge", "ols") for s in settings]     # the slow ridge jobs first
    print(f"[ceiling] {len(jobs)} jobs on {a.workers} processes x {os.environ['OPENBLAS_NUM_THREADS']} BLAS threads")
    ceilings: dict = {s: {} for s in settings}
    arrays = {}
    with ProcessPoolExecutor(a.workers, mp_context=get_context("spawn")) as ex:
        futs = [ex.submit(_job, a.config, s, *settings[s][:2], e, a.n_subsets, a.n_boot) for s, e in jobs]
        for f in futs:
            s, e, c, arr = f.result()
            ceilings[s][e] = c
            arrays.update({f"{s}__{e}__{k}": v for k, v in arr.items()})

    # ---- retention = r / ceiling (ceiling treated as fixed: its sampling error is not propagated)
    nums = _numerators(run, run_man)
    den = [("ridge", "matched", True), ("ridge", "matched_at_k=N", False), ("ridge", "per_fold_median", False),
           ("ols", "matched", False), ("ols", "per_fold_median", False)]
    pick = {"matched": lambda c: c["matched"]["ceiling"],            # same scoring + aggregation as r_LLM
            "matched_at_k=N": lambda c: c["matched"]["mean_r_at_all_subjects"],   # 8-subject pool, observed
            "per_fold_median": lambda c: c["ceiling"]}                # Schrimpf scoring + aggregation
    table = []
    for kind in ("contiguous", "shuffled"):
        for n in nums[kind]:
            for est, scoring, headline in den:
                cv = pick[scoring](ceilings[kind][est])
                table.append({"split": kind, "numerator": n["name"], "r": n["r"],
                              "r_ci": n["ci"], "ceiling_estimator": est, "ceiling_scoring": scoring,
                              "ceiling": cv, "retention": n["r"] / cv,
                              "retention_ci": [x / cv for x in n["ci"]] if n["ci"] else None,
                              "r_exceeds_ceiling": bool(n["r"] > cv),
                              "headline": headline and n["key"] == "llm"})

    out = Path(a.out) if a.out else ROOT / "results" / "d1" / f"ceiling_ridge__{run.name}"
    out.mkdir(parents=True, exist_ok=True)
    nv = [y.shape[1] for y in ds.Y]
    np.savez_compressed(out / "per_voxel.npz", subject_of_voxel=np.repeat(ds.subjects, nv), **arrays)
    head = {r["split"]: r for r in table if r["headline"]}
    manifest = {
        "what": "S1 noise ceiling with the readout's ridge, and r_LLM / ceiling on the same estimator, "
                "folds, scoring and aggregation",
        "source_run": run.name, "source_run_dir": str(run),
        "provenance": {"timestamp": datetime.now(timezone.utc).isoformat(timespec="seconds"),
                       "git_commit": pipeline._git(), "config": str(Path(a.config).resolve()),
                       "config_sha256": config.digest(cfg), "script": "scripts/ceiling_ridge_pereira2018.py",
                       "blas_threads_per_process": int(os.environ["OPENBLAS_NUM_THREADS"]),
                       "workers": a.workers},
        "dataset": {"name": ds.name, "experiment": cfg["dataset"]["experiment"], "sha1": ds.meta["sha1"],
                    "stimulus_hash": ds.stimulus_hash(), "n_samples": ds.n,
                    "n_blocks": int(len(np.unique(ds.block_ids))), "subjects": ds.subjects,
                    "n_voxels": int(sum(nv))},
        "settings": {s: {"split_kind": k, "n_folds": n, "role": r} for s, (k, n, r) in settings.items()},
        "ridge": {"alphas": [float(x) for x in config.alphas(cfg)], "inner_folds": cfg["split"]["inner_folds"],
                  "alpha_choice": "per voxel, argmax of mean inner-fold r (readout.nested_predict)"},
        "extrapolation": {"n_subsets": a.n_subsets, "n_boot": a.n_boot, "seed": cfg["seed"]},
        "checks": [{"name": n, "ok": bool(o)} for n, o in checks],
        "numerators": nums,
        "ceilings": ceilings,
        "previous_ols_ceilings": PREVIOUS_OLS,
        "retention": table,
        "headline": {k: {"r": v["r"], "r_ci": v["r_ci"], "ceiling": v["ceiling"], "retention": v["retention"],
                         "retention_ci": v["retention_ci"]} for k, v in head.items()},
        "retention_ci_note": "r's block-bootstrap CI divided by the ceiling; the ceiling's own sampling error "
                             "is not propagated, so the interval is too narrow",
        "seconds": round(time.time() - t0, 1),
    }
    (out / "manifest.json").write_text(json.dumps(manifest, indent=1, ensure_ascii=False))
    cols = ["split", "numerator", "r", "ceiling_estimator", "ceiling_scoring", "ceiling", "retention",
            "retention_ci_lo", "retention_ci_hi", "r_exceeds_ceiling", "headline"]
    lines = ["\t".join(cols)]
    for r in table:
        ci = r["retention_ci"] or ["", ""]
        lines.append("\t".join(str(x) for x in [r["split"], r["numerator"], f"{r['r']:.4f}", r["ceiling_estimator"],
                                                r["ceiling_scoring"], f"{r['ceiling']:.4f}", f"{r['retention']:.3f}",
                                                *(f"{x:.3f}" if x != "" else "" for x in ci),
                                                r["r_exceeds_ceiling"], r["headline"]]))
    (out / "retention.tsv").write_text("\n".join(lines) + "\n")
    (out / "summary.md").write_text(_summary(json.loads((out / "manifest.json").read_text()), _agg(out)))

    for k, v in head.items():
        print(f"[ceiling] {k}: r_LLM {v['r']:.4f} / ridge ceiling (matched) {v['ceiling']:.4f} = "
              f"{v['retention']:.3f}  [{v['retention_ci'][0]:.3f}, {v['retention_ci'][1]:.3f}]")
    print(f"[ceiling] -> {out}  ({time.time() - t0:.0f}s)")
    return 0


def _agg(out: Path) -> dict | None:
    """The aggregation check (scripts/ceiling_aggregation_pereira2018.py), if it has been run."""
    p = out / "aggregation" / "manifest.json"
    return json.loads(p.read_text()) if p.exists() else None


def _summary(m: dict, agg: dict | None = None) -> str:
    c, h = m["ceilings"], m["headline"]
    f = lambda x: f"{x:.4f}"  # noqa: E731
    ci = lambda r: f"[{r['retention_ci'][0]:.3f}, {r['retention_ci'][1]:.3f}]" if r["retention_ci"] else "—"  # noqa: E731
    L = ["# S1 噪声上限（岭回归口径）与归一化保留率", "",
         f"源运行：`{m['source_run']}`。生成于 {m['provenance']['timestamp']}，git {m['provenance']['git_commit']}，"
         f"脚本 `{m['provenance']['script']}`。下面的数字都是本目录的实测值，原始数据见 `manifest.json`。", "",
         "## 以前为什么不能给 r / ceiling", "",
         "r_LLM 来自 readout：岭回归，逐体素 alpha 在内折上选；先在折内去均值，再合并全部测试折算 r，"
         "对体素取均值后再对被试取均值。旧 ceiling 用的是最小范数 OLS，每折单独算 r，再对全部体素取中位数。"
         f"在这种口径下 r_LLM 比上限还高：contiguous 0.105 > {m['previous_ols_ceilings']['contiguous']}，"
         "shuffled 0.409 > 0.3225（而且后者是 5 折，r_LLM 是 8 折）。"
         "估计器、计分、聚合三者都不一致，所以这个比值不是保留率。", "",
         "## 现在怎么算", "",
         "ceiling 的流程不变：取 k = 2..9 个被试的子集，留一个被试作目标，用其余被试的全部体素去预测，"
         "再拟合 v0·(1−e^(−k/τ)) 外推到无穷多被试。"
         "第 2 步的估计器换成 readout 的岭回归：在训练行上 z-score，逐体素 alpha 在同一内折上选"
         f"（{len(m['ridge']['alphas'])} 个 alpha，{m['ridge']['inner_folds']} 个内折，种子与 readout 相同），"
         "用 n×n 核计算。外折与源运行逐样本一致（见 `checks`）。"
         "「同口径」（matched）的计分与 r_LLM 相同，聚合同样取均值：每次抽样先对体素取均值，"
         "对每个被试的均值曲线做外推，再对被试取均值。"
         "这个外推在 contiguous 的信噪比下偏低，复核见 `aggregation/summary.md`。", "",
         "## ceiling", "",
         "| 切分 | 估计器 | Schrimpf 口径：每折 r，逐体素外推，体素中位数 | **同口径**：折内去均值合并 r，均值曲线外推，被试均值 | 同口径 k=N 实测（8 人池，未外推）|",
         "|---|---|---|---|---|"]
    for s, d in c.items():
        for e in ("ols", "ridge"):
            x = d[e]
            L.append(f"| {s} {m['settings'][s]['n_folds']} 折 | {e.upper()} | {f(x['ceiling'])} | "
                     f"{f(x['matched']['ceiling'])} | {f(x['matched']['mean_r_at_all_subjects'])} |")
    L += ["", f"旧值（2026-10-17，同一 OLS 方法）：contiguous {m['previous_ols_ceilings']['contiguous']}，"
          f"Schrimpf 口径（shuffled 5 折）{m['previous_ols_ceilings']['schrimpf']}。本次 OLS 重算结果应与之相同。", "",
          "同口径实测曲线：被试池为 k−1 人时的 r，先对体素取均值，再对被试取均值，未外推。"
          "曲线在 k≈5 之后趋平，外推值离 k=N 的实测值不远。", "",
          "| 切分 | 估计器 | " + " | ".join(f"k={k}" for k in range(2, 10)) + " |",
          "|---|---|" + "---|" * 8]
    for s, d in c.items():
        for e in ("ols", "ridge"):
            by = d[e]["matched"]["mean_r_by_k"].values()
            row = [np.mean([v[k] for v in by if k in v]) for k in [str(k) for k in range(2, 10)]]
            L.append(f"| {s} | {e.upper()} | " + " | ".join(f"{x:.3f}" for x in row) + " |")
    L += ["", "计分 × 聚合的全部组合见 `by_scoring_and_aggregation`。「体素 v0 均值」一列只作诊断："
          "单个体素的曲线噪声大，外推结果有一条长右尾（个别体素顶到上限 1），会把均值拉高，中位数不受影响。"
          "所以对均值口径，要先平均、后外推。", "",
          "| 切分 | 估计器 | 计分 | 体素 v0 中位数 | 体素 v0 均值 | 均值曲线外推 | v0 顶到 1 的体素比例 |",
          "|---|---|---|---|---|---|---|"]
    for s, d in c.items():
        for e in ("ols", "ridge"):
            t = d[e]["by_scoring_and_aggregation"]
            for sc in ("per_fold_r", "pooled_fold_centred_r"):
                L.append(f"| {s} | {e.upper()} | {sc} | {f(t[sc]['median_of_voxel_v0'])} | "
                         f"{f(t[sc]['mean_of_voxel_v0'])} | {f(t[sc]['v0_of_mean_r'])} | "
                         f"{t['share_of_voxel_v0_at_clip'][sc]:.3f} |")
    L += ["", "## 归一化保留率 r_LLM / ceiling", "",
          "| 切分 | 分子 | r | ceiling（估计器，计分）| 保留率 | 95% CI* | r > ceiling? |", "|---|---|---|---|---|---|---|"]
    for r in m["retention"]:
        if not r["numerator"].startswith("r_LLM"):
            continue
        mark = "**" if r["headline"] else ""
        L.append(f"| {r['split']} | {r['numerator']} | {f(r['r'])} | {f(r['ceiling'])}（{r['ceiling_estimator']}，"
                 f"{r['ceiling_scoring']}）| {mark}{r['retention']:.3f}{mark} | {ci(r)} | {'是' if r['r_exceeds_ceiling'] else '否'} |")
    L += ["", "\\* 区间是 r 的块 bootstrap CI 除以 ceiling 的点估计。ceiling 自身的抽样误差没有算进去，所以区间偏窄。"
          "best layer 是看过测试 r 之后挑出的层，只作描述。matched_at_k=N 的分母是 8 人被试池的实测一致性，"
          "没有外推；如果曲线还在上升，这一行就是保留率的上界。"
          "分母与分子的口径不一致的行（per_fold_median，即体素中位数；OLS）只用来对照，不作结论。", "",
          "## 其他模型，同一个分母（岭回归，同口径）", "",
          "| 切分 | 分子 | r | 95% CI | ÷ ceiling | 95% CI* |", "|---|---|---|---|---|---|"]
    for r in m["retention"]:
        if (r["ceiling_estimator"], r["ceiling_scoring"]) != ("ridge", "matched") or r["numerator"].startswith("r_LLM"):
            continue
        rci = f"[{r['r_ci'][0]:.4f}, {r['r_ci'][1]:.4f}]" if r["r_ci"] else "—"
        L.append(f"| {r['split']} | {r['numerator']} | {f(r['r'])} | {rci} | {r['retention']:.3f} | {ci(r)} |")
    L += ["", "## 同口径结论（加粗行）", ""]
    for k, v in h.items():
        L.append(f"- {k}：r_LLM {f(v['r'])} ÷ 岭 ceiling {f(v['ceiling'])} = **{v['retention']:.3f}**"
                 f"（CI* [{v['retention_ci'][0]:.3f}, {v['retention_ci'][1]:.3f}]）")
    if agg:
        a = agg["revised_headline"]["contiguous"]
        L += ["", f"**复核（`aggregation/summary.md`）**：上面的分母（均值曲线做指数外推）在 contiguous 的信噪比下偏低约 15%，"
              f"所以 {a['previous_headline_b']:.3f} 应视作上界附近的值。按模拟校准后，ceiling 为 {a['ceiling_calibrated']:.4f}，"
              f"contiguous 保留率 ≈ **{a['retention']:.2f}**，硬区间 [{a['retention_lower_bound']:.2f}, {a['retention_upper_bound']:.2f}]。"
              "逐体素外推后再取均值（0.2386）会把 ceiling 高估约 2 倍。"]
    L += ["", "判定规则（judge）不使用保留率（`min_retention: null`），所以本目录不改变源运行的判定。"]
    return "\n".join(L) + "\n"


if __name__ == "__main__":
    sys.exit(main())
