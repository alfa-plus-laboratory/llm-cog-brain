"""One experiment end to end: data -> activations -> readout -> baselines -> null -> verdict.

Every run writes to <output_dir>/<run_id>/ where run_id names the split kind and
pooling, and every file inside repeats them:

    manifest.json      full resolved config, split + pooling, provenance, timings
    split_<kind>.json  fold of every sample, for the primary and the control split
    rows.jsonl         the verdict row (schema: judge.ROW_FIELDS)
    rows.tsv           same row, flat
    layer_curve.json   per-layer r under each split (descriptive, never used to decide)
    verdict.txt        the one line
"""
from __future__ import annotations

import json
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from . import activations, baselines, config, data, judge, null, readout, splits

ROOT = Path(__file__).resolve().parents[2]


def _abs(p: str | Path) -> Path:
    p = Path(p)
    return p if p.is_absolute() else ROOT / p


def _git() -> str:
    try:
        sha = subprocess.run(["git", "-C", str(ROOT), "rev-parse", "--short", "HEAD"],
                             capture_output=True, text=True, timeout=10).stdout.strip()
        dirty = subprocess.run(["git", "-C", str(ROOT), "status", "--porcelain", "--", "src", "configs"],
                               capture_output=True, text=True, timeout=10).stdout.strip()
        return (sha or "no-commit") + ("+dirty" if dirty else "")
    except (OSError, subprocess.SubprocessError):
        return "unknown"


def analyse(cfg: dict, ds: data.Dataset, split_kind: str, with_null: bool) -> dict:
    t0 = time.time()
    sp = splits.make(split_kind, ds.block_ids, cfg["split"]["n_folds"], cfg["seed"])
    feats, prov = activations.get(ds, cfg, split_kind)
    m, al, chunk = cfg["model"], config.alphas(cfg), cfg["readout"]["voxel_chunk"]
    n_inner = cfg["split"]["inner_folds"]
    cands = feats if m["layer_selection"] == "nested" else {m["fixed_layer"]: feats[m["fixed_layer"]]}
    use_conf = cfg["baselines"]["enabled"] and cfg["baselines"]["confounds"]
    Xc = baselines.confounds(ds, cfg["baselines"]["confounds"])[0] if use_conf else None
    Xo = (baselines.oasm(ds, cfg["baselines"]["oasm_sigma"])
          if cfg["baselines"]["enabled"] and "oasm" in cfg["baselines"]["diagnostics"] else None)

    def oof(fitted, X, zero_cols=None):
        out = np.full((ds.n, fitted[0][1].n_vox), np.nan)
        for te, f in fitted:
            x = np.array(X[te], dtype=float, copy=True)
            if zero_cols == "all":
                x[:] = 0.0
            elif zero_cols is not None:
                x[:, zero_cols] = 0.0
            out[te] = f.predict(x)
        return out

    # Score on fold-demeaned series (readout.center_within_folds explains why).
    fc = lambda a: readout.center_within_folds(a, sp.fold)  # noqa: E731
    r_ = lambda p, yc: float(readout.voxelwise_r(fc(p), yc).mean())  # noqa: E731
    Yc = [fc(y) for y in ds.Y]

    preds = {k: [] for k in ("llm", "conf", "joint", "oasm")}
    zero_all, zero_llm, chosen_all, llm_pen, curve = [], [], [], [], {L: [] for L in feats}
    for y, yc in zip(ds.Y, Yc):
        p_llm, fit_llm, (per_layer, vsub) = readout.nested_predict(
            cands, y, sp, ds.block_ids, n_inner, al, per_candidate=True,
            select_voxels=cfg["readout"]["select_voxels"], voxel_chunk=chunk)
        chosen = {k: f.choice for k, (_, f) in enumerate(fit_llm)}
        chosen_all.append([int(c) for c in chosen.values()])
        preds["llm"].append(fc(p_llm))
        X_ch = np.vstack([feats[chosen[k]][[i]] for i, k in enumerate(sp.fold)])  # chosen layer per sample
        zero_all.append((r_(p_llm, yc), r_(oof(fit_llm, X_ch, "all"), yc)))
        if use_conf:
            p_conf, fit_conf = readout.nested_predict({"conf": Xc}, y, sp, ds.block_ids, n_inner, al,
                                                      voxel_chunk=chunk)
            preds["conf"].append(fc(p_conf))
            p_dim = next(iter(feats.values())).shape[1]
            p_joint, fit_joint = readout.nested_predict_joint(
                feats, Xc, y, sp, ds.block_ids, n_inner, al, conf_fitted=fit_conf, forced=chosen,
                select_voxels=cfg["readout"]["select_voxels"], voxel_chunk=chunk)
            preds["joint"].append(fc(p_joint))
            llm_pen.append(["inf" if np.isinf(f.llm_penalty) else f.llm_penalty for _, f in fit_joint])
            Xj = np.hstack([X_ch, Xc])
            zero_llm.append((r_(p_joint, yc), r_(oof(fit_joint, Xj, slice(0, p_dim)), yc)))
        if Xo is not None:
            preds["oasm"].append(fc(readout.nested_predict({"oasm": Xo}, y, sp, ds.block_ids, n_inner, al,
                                                           voxel_chunk=chunk)[0]))
        if cfg["report"]["layer_curve"]:
            for L, pl in per_layer.items():
                curve[L].append(r_(pl, yc[:, vsub]))
    preds = {k: v for k, v in preds.items() if v}
    assert all(np.isfinite(p).all() for v in preds.values() for p in v), "some sample never got a prediction"

    contrasts = {"delta": ("joint", "conf")} if use_conf else None
    b = cfg["bootstrap"]
    boot = readout.bootstrap(preds, Yc, ds.block_ids, contrasts=contrasts, n_boot=b["n_boot"], ci=b["ci"],
                             hierarchical=b["hierarchical"], seed=cfg["seed"])
    nul = None
    if with_null:
        nc = cfg["null_test"]
        main = {k: v for k, v in preds.items() if k in ("llm", "conf", "joint")}
        nul = null.circular_shift_null(main, Yc, n_perm=nc["n_perm"], min_shift=nc["min_shift"],
                                       seed=cfg["seed"], contrasts=contrasts)

    def ratio(pairs):
        real, zero = np.mean([a for a, _ in pairs]), np.mean([z for _, z in pairs])
        return float(zero / real) if real != 0 else float("nan")

    return {
        "split_kind": split_kind, "split": sp.describe(ds.block_ids), "pooling": cfg["pooling"],
        "baselines_enabled": bool(use_conf), "stimulus_hash": ds.stimulus_hash(),
        "n_subjects": len(ds.Y), "n_samples": ds.n, "n_blocks": int(len(np.unique(ds.block_ids))),
        "layers_chosen": sorted({c for cs in chosen_all for c in cs}),
        "layers_chosen_per_subject_fold": chosen_all,
        "joint_llm_penalty_per_subject_fold": llm_pen,   # inf = LLM band switched off in that fold
        "boot": boot, "null": nul,
        "zero_input": {"all": ratio(zero_all), "llm_in_joint": ratio(zero_llm) if zero_llm else None},
        "layer_curve": {int(L): float(np.mean(v)) for L, v in curve.items() if v},
        "activations": prov, "seconds": round(time.time() - t0, 2),
    }


def run(cfg: dict, overrides_note: dict | None = None) -> tuple[dict, str, Path]:
    t0 = time.time()
    cfg = dict(cfg)
    cfg["model"] = dict(cfg["model"], cache_dir=str(_abs(cfg["model"]["cache_dir"])))
    ds = data.load(cfg)
    kind, ctrl_kind = cfg["split"]["kind"], cfg["split"]["control"]
    res = analyse(cfg, ds, kind, with_null=True)
    ctrl = analyse(cfg, ds, ctrl_kind, with_null=False) if ctrl_kind else None

    sha = config.digest(cfg)
    run_id = f"{cfg['run_name']}__split-{kind}__pool-{cfg['pooling']}__base-{'on' if res['baselines_enabled'] else 'off'}__{sha[:8]}"
    prov = {"run_id": run_id, "timestamp": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "git_commit": _git(), "config_sha256": sha}
    dec = judge.decide(res, ctrl, cfg)
    r = judge.row(res, ctrl, dec, cfg, prov)
    one = judge.line(r)

    out = _abs(cfg["output_dir"]) / run_id
    out.mkdir(parents=True, exist_ok=True)
    strip = lambda d: {k: v for k, v in d.items() if k not in ("split",)}  # noqa: E731
    (out / "manifest.json").write_text(json.dumps({
        "run_id": run_id, "split_kind": kind, "split_role": "primary", "control_split_kind": ctrl_kind,
        "pooling": cfg["pooling"], "baselines_enabled": res["baselines_enabled"],
        "overrides": overrides_note or {}, "provenance": prov, "dataset_meta": ds.meta,
        "config": cfg, "primary": strip(res), "control": strip(ctrl) if ctrl else None,
        "decision": dec, "seconds_total": round(time.time() - t0, 2),
    }, indent=1, ensure_ascii=False, default=str))
    for rr in filter(None, (res, ctrl)):
        (out / f"split_{rr['split_kind']}.json").write_text(json.dumps(rr["split"], ensure_ascii=False))
    (out / "rows.jsonl").write_text(json.dumps(r, ensure_ascii=False, default=str) + "\n")
    (out / "rows.tsv").write_text(judge.to_tsv([r]))
    (out / "layer_curve.json").write_text(json.dumps({
        "note": "descriptive only; layer choice for the verdict is made on inner folds",
        "pooling": cfg["pooling"], kind: res["layer_curve"],
        **({ctrl_kind: ctrl["layer_curve"]} if ctrl else {})}, indent=1))
    (out / "verdict.txt").write_text(one + "\n")
    return r, one, out
