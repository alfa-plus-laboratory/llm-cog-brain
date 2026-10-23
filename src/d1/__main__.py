"""CLI.

    python scripts/run_d1.py --config configs/pereira2018_schrimpf2021.yaml
    python scripts/run_d1.py --smoke                     # synthetic, CPU, < 1 min, writes tmp/smoke/
    python scripts/run_d1.py --smoke --pooling sum --split shuffled --no-baselines
    python scripts/run_d1.py --smoke --matrix            # all 2 x 3 x 2 switch combinations

The three switches can also be flipped in the YAML; the flags exist so the
smoke test can exercise them without editing files, and are recorded in the
manifest when used. Exit code: 0 = ran (whatever the verdict), 1 = error or a
smoke self-check failed.
"""
from __future__ import annotations

import argparse
import copy
import itertools
import os
import sys
import time

if "--smoke" in sys.argv:
    # The smoke matrices are tiny; multithreaded BLAS spends ~4x more wall time on
    # thread overhead than on arithmetic. Must be set before numpy is imported.
    for _v in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
        os.environ.setdefault(_v, "1")

from . import config, judge, pipeline  # noqa: E402

SMOKE_CONFIG = pipeline.ROOT / "configs" / "smoke.yaml"


def _overrides(a) -> dict:
    o = {}
    if a.pooling:
        o["pooling"] = a.pooling
    if a.split:
        o["split"] = {"kind": a.split, "control": None if a.split == "shuffled" else "shuffled"}
    if a.no_baselines:
        o["baselines"] = {"enabled": False}
    return o


def _self_checks(cfg: dict, pos: dict, neg: dict) -> list[tuple[str, bool, str]]:
    checks = []
    try:
        config.validate(copy.deepcopy(cfg) | {"null_test": dict(cfg["null_test"], method="iid")})
        checks.append(("i.i.d. null is rejected", False, "validate() accepted method=iid"))
    except ValueError:
        checks.append(("i.i.d. null is rejected", True, "validate() raises on method=iid"))
    if pos["split_kind"] == "contiguous":
        man = pipeline._abs(cfg["output_dir"]) / pos["run_id"] / "split_contiguous.json"
        import json
        leak = json.loads(man.read_text())["blocks_spanning_train_and_test"]
        checks.append(("contiguous split keeps blocks whole", leak == 0, f"{leak} blocks span folds"))
    checks.append(("negative control (no stimulus signal) is not judged 仍成立",
                   neg["verdict"] != judge.HOLDS, f"negative verdict = {neg['verdict']}"))
    if pos.get("oasm_r_control") is not None:
        ok = pos["oasm_r_control"] > pos["oasm_r_primary"] + 0.05
        checks.append(("shuffled split leaks (OASM r rises vs contiguous)", ok,
                       f"OASM r {pos['oasm_r_primary']:.3f} -> {pos['oasm_r_control']:.3f}"))
    return checks


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="d1", description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", help="experiment YAML")
    ap.add_argument("--smoke", action="store_true", help="synthetic end-to-end run (configs/smoke.yaml)")
    ap.add_argument("--matrix", action="store_true", help="run every split x pooling x baselines combination")
    ap.add_argument("--pooling", choices=config.POOLINGS)
    ap.add_argument("--split", choices=config.SPLIT_KINDS)
    ap.add_argument("--no-baselines", action="store_true")
    a = ap.parse_args(argv)
    if bool(a.smoke) == bool(a.config):
        ap.error("give exactly one of --smoke / --config")
    t0 = time.time()
    path = SMOKE_CONFIG if a.smoke else a.config

    if a.matrix:
        rows = []
        for kind, pool, base in itertools.product(config.SPLIT_KINDS, config.POOLINGS, (True, False)):
            o = {"pooling": pool, "split": {"kind": kind, "control": None if kind == "shuffled" else "shuffled"},
                 "baselines": {"enabled": base}}
            r, one, _ = pipeline.run(config.load(path, o), o)
            rows.append(r)
            print(one, flush=True)
        out = pipeline._abs(config.load(path)["output_dir"]) / "matrix.tsv"
        out.write_text(judge.to_tsv(rows))
        print(f"[d1] matrix -> {out}  ({time.time() - t0:.1f}s)")
        return 0

    o = _overrides(a)
    cfg = config.load(path, o)
    r, one, out = pipeline.run(cfg, o)
    print(f"[d1] outputs -> {out}")
    if a.smoke:
        neg_o = copy.deepcopy(o)
        neg_o["run_name"] = "smoke-negative"
        neg_o["dataset"] = {"synthetic": {"signal": False}}
        neg_o["report"] = {"layer_curve": False}
        neg, neg_line, _ = pipeline.run(config.load(path, neg_o), neg_o)
        print(f"[d1] negative control: {neg_line}")
        fails = 0
        for name, ok, info in _self_checks(cfg, r, neg):
            print(f"[d1] self-check {'PASS' if ok else 'FAIL'}  {name}  ({info})")
            fails += not ok
        dt = time.time() - t0
        print(f"[d1] smoke wall time {dt:.1f}s ({'within' if dt < 60 else 'OVER'} the 60 s budget)")
        if fails:
            print(one)
            return 1
    print(one)
    return 0


if __name__ == "__main__":
    sys.exit(main())
