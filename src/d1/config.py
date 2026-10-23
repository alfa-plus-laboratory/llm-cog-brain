"""One YAML file = every switch of one experiment.

The three axes the task book asks to flip with one line each:

    split.kind        contiguous | shuffled     (shuffled is a control, never the basis of a verdict)
    pooling           last | mean | sum
    baselines.enabled true | false

Unknown keys are rejected, so a typo cannot silently fall back to a default.
"""
from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path

import yaml

SPLIT_KINDS = ("contiguous", "shuffled")
POOLINGS = ("last", "mean", "sum")
PEREIRA_EXPERIMENTS = ("384sentences", "243sentences")
PASSAGE_ORDERS = ("topic_interleaved", "stimulus_list")   # see data.pereira2018
CONFOUNDS = ("position", "word_rate")
DIAGNOSTICS = ("oasm",)
NULL_METHODS = ("circular_shift",)
# Named here only so the rejection message can be specific.
FORBIDDEN_NULLS = ("iid", "permutation", "shuffle", "iid_permutation")

DEFAULTS: dict = {
    "run_name": "unnamed",
    "seed": 0,
    "output_dir": "results/d1",
    "dataset": {
        "name": "synthetic",          # synthetic | pereira2018
        "path": None,
        "experiment": "384sentences", # pereira2018 only: 384sentences | 243sentences
        "passage_order": "topic_interleaved",  # pereira2018 only; why there is a choice: data.pereira2018
        "synthetic": {
            "n_subjects": 3,
            "n_blocks": 60,           # passages
            "block_len": 4,           # sentences per passage
            "n_voxels": 400,
            "vocab": 400,
            "min_words": 4,
            "max_words": 14,
            "duration_s": 4.0,        # presentation time per sentence (word rate = words / duration)
            "signal": True,           # False -> responses carry no stimulus semantics (negative control)
            "gains": {"semantic": 1.0, "position": 0.6, "word_rate": 0.6,
                      "drift": 1.0, "noise": 1.6},
            "ar_rho": 0.35,           # temporal autocorrelation carried across consecutive samples
        },
    },
    "model": {
        "backend": "synthetic",       # synthetic | hf
        "name": "toy-lm",
        "layers": [1, 2, 3, 4, 5, 6],
        "layer_selection": "nested",  # nested (chosen on inner folds) | fixed
        "fixed_layer": None,
        "context": "passage",         # passage: preceding sentences of the same passage are fed as context
        "device": "cpu",
        "max_tokens": 1024,
        "cache_dir": "data/cache/activations",
    },
    "pooling": "last",
    "split": {
        "kind": "contiguous",
        "control": "shuffled",        # also run this kind, report it as a control only; null to skip
        "n_folds": 5,
        "inner_folds": 3,
    },
    "readout": {
        "alphas": {"logspace": [-1, 5, 13]},
        "voxel_chunk": 20000,
        "select_voxels": 2000,        # global choices (layer, joint LLM penalty) are made on this many voxels
    },
    "baselines": {
        "enabled": True,
        "confounds": ["position", "word_rate"],
        "diagnostics": ["oasm"],
        "oasm_sigma": 1.5,
        "zero_input_ablation": True,
    },
    "null_test": {
        "method": "circular_shift",
        "n_perm": 1000,
        "min_shift": 8,               # samples; must exceed one block, otherwise the shift stays partly aligned
    },
    "bootstrap": {
        "n_boot": 1000,
        "unit": "block",              # resample whole passages/stories; never voxels
        "hierarchical": True,         # resample subjects first, then blocks within subject
        "ci": 0.95,
    },
    "judge": {
        "alpha": 0.05,
        "sesoi_delta": 0.02,          # smallest unique Δr that would still count as "LLM adds something"
        "min_test_blocks": 10,
        "max_zero_input_retained": 0.10,
    },
    "claim": {
        "paper": None,
        "claim_id": None,
        "text": None,
        "reference_value": None,      # number the paper reported; filled from the verdict table, never hard-coded
        "reference_metric": None,
        "reference_source": None,
        "reference_ceiling": None,    # the paper's noise ceiling, if it reports ceiling-normalized scores
        "min_retention": None,        # fraction of reference_value that must survive; null = report only
    },
    "report": {"layer_curve": True},
}


def _merge(base: dict, over: dict, path: str = "") -> dict:
    out = copy.deepcopy(base)
    for k, v in (over or {}).items():
        where = f"{path}{k}"
        if k not in base:
            raise KeyError(f"unknown config key: {where}")
        # `gains` and `alphas` are leaf dicts: merge into them, but their keys are still checked.
        if isinstance(base[k], dict) and isinstance(v, dict):
            out[k] = _merge(base[k], v, where + ".")
        else:
            out[k] = v
    return out


def validate(cfg: dict) -> dict:
    if cfg["split"]["kind"] not in SPLIT_KINDS:
        raise ValueError(f"split.kind must be one of {SPLIT_KINDS}, got {cfg['split']['kind']!r}")
    ctrl = cfg["split"]["control"]
    if ctrl is not None and (ctrl not in SPLIT_KINDS or ctrl == cfg["split"]["kind"]):
        raise ValueError(f"split.control must be null or the other split kind, got {ctrl!r}")
    if cfg["pooling"] not in POOLINGS:
        raise ValueError(f"pooling must be one of {POOLINGS}, got {cfg['pooling']!r}")
    method = cfg["null_test"]["method"]
    if method in FORBIDDEN_NULLS:
        raise ValueError(
            f"null_test.method={method!r} is forbidden: an i.i.d. null destroys BOLD autocorrelation, "
            "comes out far too narrow and overstates significance. Use 'circular_shift'.")
    if method not in NULL_METHODS:
        raise ValueError(f"null_test.method must be one of {NULL_METHODS}, got {method!r}")
    bad = set(cfg["baselines"]["confounds"]) - set(CONFOUNDS)
    if bad:
        raise ValueError(f"unknown confound baselines {sorted(bad)}; known: {CONFOUNDS}")
    bad = set(cfg["baselines"]["diagnostics"]) - set(DIAGNOSTICS)
    if bad:
        raise ValueError(f"unknown diagnostic baselines {sorted(bad)}; known: {DIAGNOSTICS}")
    if cfg["bootstrap"]["unit"] != "block":
        raise ValueError("bootstrap.unit must be 'block' (see docs/checklist.md, 3.3)")
    if cfg["model"]["device"] != "cpu":
        raise ValueError("model.device must be 'cpu': this pipeline never uses a GPU")
    if cfg["model"]["layer_selection"] not in ("nested", "fixed"):
        raise ValueError("model.layer_selection must be 'nested' or 'fixed'")
    if cfg["model"]["layer_selection"] == "fixed" and cfg["model"]["fixed_layer"] not in cfg["model"]["layers"]:
        raise ValueError("layer_selection=fixed needs model.fixed_layer, and it must be in model.layers")
    if cfg["split"]["n_folds"] < 2 or cfg["split"]["inner_folds"] < 2:
        raise ValueError("need >= 2 outer and >= 2 inner folds")
    if cfg["dataset"]["name"] == "pereira2018":
        if cfg["dataset"]["experiment"] not in PEREIRA_EXPERIMENTS:
            raise ValueError(f"dataset.experiment must be one of {PEREIRA_EXPERIMENTS}")
        if cfg["dataset"]["passage_order"] not in PASSAGE_ORDERS:
            raise ValueError(f"dataset.passage_order must be one of {PASSAGE_ORDERS}")
    return cfg


def load(path: str | Path | None = None, overrides: dict | None = None) -> dict:
    user = yaml.safe_load(Path(path).read_text(encoding="utf-8")) if path else {}
    cfg = _merge(DEFAULTS, user or {})
    if overrides:
        cfg = _merge(cfg, overrides)
    return validate(cfg)


def alphas(cfg: dict):
    import numpy as np
    spec = cfg["readout"]["alphas"]
    if isinstance(spec, dict) and "logspace" in spec:
        lo, hi, n = spec["logspace"]
        return np.logspace(lo, hi, int(n))
    return np.asarray(spec, dtype=float)


def digest(cfg: dict) -> str:
    return hashlib.sha256(json.dumps(cfg, sort_keys=True, default=str).encode()).hexdigest()
