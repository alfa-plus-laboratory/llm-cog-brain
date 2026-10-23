"""Numbers -> one verdict: 仍成立 / 不成立 / 无法判定.

Pre-registered rule (thresholds come from the config's `judge` and `claim`):

Hard preconditions -- any failure gives 无法判定, and the row says which:
  P1  primary split is contiguous (shuffled is a control; it never decides)
  P2  significance comes from the circular-shift null
  P3  zero-input ablation passes (an all-zero input must not keep the score)
  P4  >= judge.min_test_blocks blocks
  P5  the null can resolve judge.alpha (min attainable p <= alpha)

Then, with Δ = r(LLM + confounds) - r(confounds), paired block-bootstrap CI:
  仍成立  confounds were controlled, Δ CI lower bound > 0 and Δ beats its
          circular-shift null (p < alpha); and, if the claim sets
          reference_value + min_retention, the retention CI lower bound clears it.
  不成立  Δ CI upper bound < judge.sesoi_delta (the LLM reliably adds less than
          the smallest effect of interest), or the retention CI upper bound is
          below min_retention, or -- without confound control -- even the
          LLM alone stays below sesoi.
  无法判定 everything else; the reason names what is missing (usually: CI too
          wide, need more subjects/blocks; or confounds not controlled).
"""
from __future__ import annotations

import json

SCHEMA_VERSION = "d1-row/0.1"
HOLDS, FAILS, UNDECIDED = "仍成立", "不成立", "无法判定"

# One row of the verdict table (docs/judgement_table.md consumes this). Order is the TSV column order.
ROW_FIELDS = [
    "schema_version", "run_id", "timestamp", "git_commit", "config_sha256",
    "paper", "claim_id", "claim_text", "claim_scope",
    "dataset", "stimulus_hash", "n_subjects", "n_samples", "n_blocks",
    "model", "backend", "context", "layer_grid", "layer_selection", "layers_chosen",
    "pooling",
    "split_kind", "split_role", "split_unit", "n_folds", "inner_folds",
    "baselines_enabled", "confounds", "metric",
    "r_llm", "r_llm_ci_lo", "r_llm_ci_hi",
    "r_conf", "r_conf_ci_lo", "r_conf_ci_hi",
    "r_joint", "r_joint_ci_lo", "r_joint_ci_hi",
    "delta", "delta_ci_lo", "delta_ci_hi",
    "z_llm", "p_llm", "z_delta", "p_delta",
    "null_method", "null_n_shifts", "null_min_shift", "null_min_p",
    "boot_n", "boot_unit", "boot_hierarchical", "ci_level",
    "zero_input_retained", "llm_zeroed_in_joint_retained",
    "reference_value", "reference_metric", "reference_source", "min_retention",
    "retention", "retention_ci_lo", "retention_ci_hi",
    "control_split_kind", "control_r_llm", "control_r_llm_ci_lo", "control_r_llm_ci_hi",
    "control_inflation", "oasm_r_primary", "oasm_r_control",
    "alpha", "sesoi_delta",
    "verdict", "verdict_reason", "missing",
]


def _f(x, nd=3):
    return "NA" if x is None else f"{x:.{nd}f}"


def decide(res: dict, ctrl: dict | None, cfg: dict) -> dict:
    j, claim = cfg["judge"], cfg["claim"]
    missing, reasons = [], []
    boot, nul = res["boot"], res.get("null")

    # ---- hard preconditions
    if res["split_kind"] != "contiguous":
        missing.append("主切分是 shuffled：shuffled 只作对照，需要 contiguous 结果才能判定")
    if not nul or nul.get("_meta", {}).get("method") != "circular_shift":
        missing.append("缺循环移位 null（保留时间自相关）的显著性")
    zi = res["zero_input"].get("all")
    if zi is None:
        missing.append("缺零输入消融")
    elif not (abs(zi) < j["max_zero_input_retained"]):
        missing.append(f"零输入消融未通过（保留 {zi:.2f} ≥ {j['max_zero_input_retained']}）：预测不依赖输入，管线可疑")
    if res["n_blocks"] < j["min_test_blocks"]:
        missing.append(f"测试块只有 {res['n_blocks']} 个（< {j['min_test_blocks']}）")
    if nul and nul["_meta"]["min_attainable_p"] > j["alpha"]:
        missing.append(f"循环移位只有 {nul['_meta']['n_shifts']} 个，最小可达 p > α={j['alpha']}")

    # ---- quantitative part (only if the claim supplies a reference value)
    ref, min_ret = claim["reference_value"], claim["min_retention"]
    ret = ret_lo = ret_hi = None
    if ref:
        ret = boot["llm"]["est"] / ref
        ret_lo, ret_hi = boot["llm"]["ci"][0] / ref, boot["llm"]["ci"][1] / ref
    quant_fails = ref is not None and min_ret is not None and ret_hi < min_ret
    quant_holds = ref is None or min_ret is None or ret_lo >= min_ret
    if ref is None:
        scope = "定性：LLM 在控制混淆后是否仍有独立预测力（参考值未提供，定量部分不判）"
    elif min_ret is None:
        scope = "定性 + 报告保留率（未设 min_retention，保留率不进判定）"
    else:
        scope = f"定性 + 定量（保留率 ≥ {min_ret}）"

    sesoi, alpha = j["sesoi_delta"], j["alpha"]
    if res["baselines_enabled"]:
        d_lo, d_hi = boot["delta"]["ci"]
        p_d = nul["delta"]["p"] if nul else None
        q_fails = d_hi < sesoi
        q_holds = d_lo > 0 and p_d is not None and p_d < alpha
    else:
        missing.append("未控制混淆基线（位置 / 语速）：最多只能判「不成立」")
        q_fails = boot["llm"]["ci"][1] < sesoi
        q_holds = False

    hard = [m for m in missing if not m.startswith("未控制混淆")]
    if hard:
        verdict = UNDECIDED
        reasons.append("前提不满足")
    elif q_fails or quant_fails:
        verdict = FAILS
        if q_fails and res["baselines_enabled"]:
            reasons.append(f"Δ 的 CI 上界 {d_hi:.3f} < SESOI {sesoi}：控制混淆后 LLM 没有可信的独立贡献")
        elif q_fails:
            reasons.append(f"LLM 单独的 r CI 上界 < SESOI {sesoi}")
        if quant_fails:
            reasons.append(f"保留率 CI 上界 {ret_hi:.2f} < min_retention {min_ret}")
    elif q_holds and quant_holds:
        verdict = HOLDS
        reasons.append(f"Δ CI 下界 {d_lo:.3f} > 0 且 p_null(Δ)={p_d:.4f} < {alpha}")
    else:
        verdict = UNDECIDED
        if res["baselines_enabled"]:
            if not q_holds:
                missing.append(f"Δ 的 CI [{d_lo:.3f}, {d_hi:.3f}] 跨 0 / SESOI，或 p_null(Δ) ≥ α：效应不确定，"
                               "需要更多被试或测试块")
            if not quant_holds:
                missing.append(f"保留率 CI [{ret_lo:.2f}, {ret_hi:.2f}] 跨 min_retention {min_ret}")
        reasons.append("证据不足以定论")

    return {"verdict": verdict, "reasons": reasons, "missing": missing, "scope": scope,
            "retention": ret, "retention_ci": [ret_lo, ret_hi]}


def row(res: dict, ctrl: dict | None, dec: dict, cfg: dict, prov: dict) -> dict:
    b, n = res["boot"], res.get("null") or {}
    g = lambda k, i=None: (b[k]["est"] if i is None else b[k]["ci"][i]) if k in b else None  # noqa: E731
    r = {
        "schema_version": SCHEMA_VERSION,
        "run_id": prov["run_id"], "timestamp": prov["timestamp"], "git_commit": prov["git_commit"],
        "config_sha256": prov["config_sha256"],
        "paper": cfg["claim"]["paper"], "claim_id": cfg["claim"]["claim_id"],
        "claim_text": cfg["claim"]["text"], "claim_scope": dec["scope"],
        "dataset": cfg["dataset"]["name"], "stimulus_hash": res["stimulus_hash"],
        "n_subjects": res["n_subjects"], "n_samples": res["n_samples"], "n_blocks": res["n_blocks"],
        "model": cfg["model"]["name"], "backend": cfg["model"]["backend"], "context": cfg["model"]["context"],
        "layer_grid": cfg["model"]["layers"], "layer_selection": cfg["model"]["layer_selection"],
        "layers_chosen": res["layers_chosen"],
        "pooling": res["pooling"],
        "split_kind": res["split_kind"], "split_role": res["split"]["role"], "split_unit": res["split"]["unit"],
        "n_folds": res["split"]["n_folds"], "inner_folds": cfg["split"]["inner_folds"],
        "baselines_enabled": res["baselines_enabled"],
        "confounds": cfg["baselines"]["confounds"] if res["baselines_enabled"] else [],
        "metric": "pearson_r: voxelwise on pooled out-of-fold predictions, mean over voxels, mean over subjects",
        "r_llm": g("llm"), "r_llm_ci_lo": g("llm", 0), "r_llm_ci_hi": g("llm", 1),
        "r_conf": g("conf"), "r_conf_ci_lo": g("conf", 0), "r_conf_ci_hi": g("conf", 1),
        "r_joint": g("joint"), "r_joint_ci_lo": g("joint", 0), "r_joint_ci_hi": g("joint", 1),
        "delta": g("delta"), "delta_ci_lo": g("delta", 0), "delta_ci_hi": g("delta", 1),
        "z_llm": n.get("llm", {}).get("z"), "p_llm": n.get("llm", {}).get("p"),
        "z_delta": n.get("delta", {}).get("z"), "p_delta": n.get("delta", {}).get("p"),
        "null_method": n.get("_meta", {}).get("method"), "null_n_shifts": n.get("_meta", {}).get("n_shifts"),
        "null_min_shift": n.get("_meta", {}).get("min_shift"),
        "null_min_p": n.get("_meta", {}).get("min_attainable_p"),
        "boot_n": b["_meta"]["n_boot"], "boot_unit": b["_meta"]["unit"],
        "boot_hierarchical": b["_meta"]["hierarchical"], "ci_level": b["_meta"]["ci"],
        "zero_input_retained": res["zero_input"].get("all"),
        "llm_zeroed_in_joint_retained": res["zero_input"].get("llm_in_joint"),
        "reference_value": cfg["claim"]["reference_value"], "reference_metric": cfg["claim"]["reference_metric"],
        "reference_source": cfg["claim"]["reference_source"], "min_retention": cfg["claim"]["min_retention"],
        "retention": dec["retention"], "retention_ci_lo": dec["retention_ci"][0],
        "retention_ci_hi": dec["retention_ci"][1],
        "control_split_kind": ctrl["split_kind"] if ctrl else None,
        "control_r_llm": ctrl["boot"]["llm"]["est"] if ctrl else None,
        "control_r_llm_ci_lo": ctrl["boot"]["llm"]["ci"][0] if ctrl else None,
        "control_r_llm_ci_hi": ctrl["boot"]["llm"]["ci"][1] if ctrl else None,
        "control_inflation": (ctrl["boot"]["llm"]["est"] / b["llm"]["est"]
                              if ctrl and b["llm"]["est"] > 0 else None),
        "oasm_r_primary": g("oasm"),
        "oasm_r_control": ctrl["boot"]["oasm"]["est"] if ctrl and "oasm" in ctrl["boot"] else None,
        "alpha": cfg["judge"]["alpha"], "sesoi_delta": cfg["judge"]["sesoi_delta"],
        "verdict": dec["verdict"], "verdict_reason": "；".join(dec["reasons"]),
        "missing": dec["missing"],
    }
    assert list(r) == ROW_FIELDS, "row keys drifted from ROW_FIELDS"
    return r


def line(r: dict) -> str:
    """The one line: verdict + the numbers it rests on."""
    parts = [f"判定={r['verdict']}",
             f"切分={r['split_kind']}({r['split_role']})"
             + (f" 对照={r['control_split_kind']}" if r["control_split_kind"] else ""),
             f"pooling={r['pooling']}",
             f"层={r['layers_chosen']}",
             f"基线={'+'.join(r['confounds']) if r['baselines_enabled'] else '无'}"]
    if r["baselines_enabled"]:
        parts.append(f"Δr(LLM+基线−基线)={_f(r['delta'])} [{int(r['ci_level'] * 100)}%CI {_f(r['delta_ci_lo'])},"
                     f"{_f(r['delta_ci_hi'])}] p_null={_f(r['p_delta'], 4)}")
    parts.append(f"r_LLM={_f(r['r_llm'])} [{_f(r['r_llm_ci_lo'])},{_f(r['r_llm_ci_hi'])}] "
                 f"z_null={_f(r['z_llm'], 1)} p_null={_f(r['p_llm'], 4)}")
    if r["baselines_enabled"]:
        parts.append(f"r_基线={_f(r['r_conf'])}")
    parts.append(f"null=循环移位×{r['null_n_shifts']}(min_shift={r['null_min_shift']})")
    parts.append(f"零输入保留={_f(r['zero_input_retained'], 2)}")
    if r["control_split_kind"]:
        parts.append(f"shuffled对照 r_LLM={_f(r['control_r_llm'])}(×{_f(r['control_inflation'], 2)}) "
                     f"OASM {_f(r['oasm_r_primary'])}→{_f(r['oasm_r_control'])}")
    parts.append("参考值=" + ("未提供" if r["reference_value"] is None
                             else f"{r['reference_value']} 保留率={_f(r['retention'], 2)}"))
    if r["verdict"] != HOLDS and r["missing"]:
        parts.append("缺：" + "；".join(r["missing"]))
    return " | ".join(parts)


def to_tsv(rows: list[dict]) -> str:
    def cell(v):
        if v is None:
            return ""
        if isinstance(v, (list, dict)):
            return json.dumps(v, ensure_ascii=False)
        return str(v).replace("\t", " ").replace("\n", " ")
    return "\n".join(["\t".join(ROW_FIELDS)] + ["\t".join(cell(r[k]) for k in ROW_FIELDS) for r in rows]) + "\n"
