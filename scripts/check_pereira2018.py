#!/usr/bin/env python3
"""Self-check of the real Pereira2018 path: download -> loader -> Dataset.check() -> noise ceiling.

    data/venv/bin/python scripts/check_pereira2018.py --download   # once, ~36 MB
    data/venv/bin/python scripts/check_pereira2018.py [--experiment 243sentences] [--quick | --no-ceiling]

Prints n_samples / n_blocks / n_subjects / n_voxels, the first passages in the
order the loader returns them, and the noise ceiling: Schrimpf's protocol
(shuffled 5-fold OLS from the other subjects, extrapolated in pool size; compare
with 0.32), the same ceiling under the pipeline's contiguous split, and a
split-half fROI reliability (diagnostic only). Writes a JSON report to
results/pereira2018/ (data/ may be read-only). Exit 0 = every check passed.
Regression and reliability estimates only: no GPU, no LLM, nothing trained.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

import numpy as np  # noqa: E402

from d1 import ceiling, config, data, splits  # noqa: E402

CONFIG = ROOT / "configs" / "pereira2018_schrimpf2021.yaml"
# Schrimpf et al.'s own precomputed ceiling (neural-nlp, benchmarks/neural.py: load_s3 key
# 'Pereira2018-encoding-ceiling'); single-part upload, so its S3 ETag is the file's MD5.
SCHRIMPF_CEILING_URL = "https://brainio-language.s3.amazonaws.com/Pereira2018-encoding-ceiling.nc"
SCHRIMPF_CEILING_MD5 = "60e14266b0397e427d8a04fb1663be65"
SCHRIMPF_CEILING_FILE = ROOT / "data" / "raw" / "pereira2018" / "neural-nlp" / "Pereira2018-encoding-ceiling.nc"
EXPECTED = {"384sentences": (384, 96, 9), "243sentences": (243, 72, 6)}   # Pereira 2018; Schrimpf 2021 Methods 1
# Pereira et al. 2018 Methods: sentence lengths in words
PAPER_WORDS = {"384sentences": "7-18, mean 11.8, sd 2.1", "243sentences": "5-20, mean 13.15, sd 2.92"}
SAME_MAGNITUDE = 0.05


def _download(url: str, dest: Path, check) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.is_file() and check(dest):
        print(f"[check] have {dest} (verified)")
        return
    part = dest.with_name(dest.name + ".part")
    print(f"[check] downloading {url} -> {dest.resolve()}")
    urllib.request.urlretrieve(url, part)
    if not check(part):
        part.unlink()
        sys.exit(f"[check] {url}: checksum mismatch, download discarded")
    part.rename(dest)


def _digest(path: Path, algo: str) -> str:
    h = hashlib.new(algo)
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _schrimpf_ceiling() -> float | None:
    if not SCHRIMPF_CEILING_FILE.is_file():
        return None
    import h5py
    with h5py.File(SCHRIMPF_CEILING_FILE, "r") as f:
        names = [x.decode() for x in f["aggregation_"][:]]
        return float(f["__xarray_dataarray_variable__"][:][names.index("center")])


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default=str(CONFIG))
    ap.add_argument("--experiment", choices=config.PEREIRA_EXPERIMENTS)
    ap.add_argument("--download", action="store_true", help="fetch the assembly (+ Schrimpf's ceiling file) first")
    ap.add_argument("--quick", action="store_true", help="skip the contiguous-split ceiling")
    ap.add_argument("--no-ceiling", action="store_true", help="skip both noise ceilings (and the ceiling check)")
    a = ap.parse_args()
    t0 = time.time()
    over = {"dataset": {"experiment": a.experiment}} if a.experiment else None
    cfg = config.load(a.config, over)
    d = cfg["dataset"]

    if a.download:
        nc = Path(d["path"]) if Path(d["path"]).is_absolute() else ROOT / d["path"]
        _download(data.PEREIRA_URL, nc, lambda p: _digest(p, "sha1") == data.PEREIRA_SHA1)
        _download(SCHRIMPF_CEILING_URL, SCHRIMPF_CEILING_FILE, lambda p: _digest(p, "md5") == SCHRIMPF_CEILING_MD5)

    checks: list[tuple[str, bool, str]] = []
    ds = data.load(cfg).check()                       # sha1 verified inside the loader
    checks.append(("release sha1 = Brain-Score registry", ds.meta["sha1"] == data.PEREIRA_SHA1, ds.meta["sha1"]))
    checks.append(("Dataset.check() (blocks are contiguous runs)", True, "passed"))
    n_blocks = len(np.unique(ds.block_ids))
    got = (ds.n, n_blocks, len(ds.subjects))
    checks.append(("n_samples / n_blocks / n_subjects as published", got == EXPECTED[d["experiment"]],
                   f"{got} vs {EXPECTED[d['experiment']]}"))
    ok = all(np.array_equal(ds.position[ds.block_ids == b], np.arange((ds.block_ids == b).sum()))
             for b in np.unique(ds.block_ids))
    checks.append(("within-passage positions are 0..k-1 in reading order", ok, ""))
    sp = splits.make("contiguous", ds.block_ids, cfg["split"]["n_folds"], cfg["seed"])
    leak = sp.describe(ds.block_ids)["blocks_spanning_train_and_test"]
    checks.append((f"contiguous {sp.n_folds}-fold split keeps passages whole", leak == 0, f"{leak} passages span folds"))
    topic = np.array(ds.meta["topic"])
    topics_per_fold = [len(set(topic[sp.fold == k].tolist())) for k in range(sp.n_folds)]
    whole_topics = sum(len(set(sp.fold[topic == t].tolist())) == 1 for t in set(topic.tolist()))

    nv = [y.shape[1] for y in ds.Y]
    print(f"[check] {ds.name} {d['experiment']}  release={ds.meta['release']}")
    print(f"[check] n_samples={ds.n}  n_blocks={n_blocks}  n_subjects={len(ds.subjects)}  "
          f"n_voxels={sum(nv)} total ({min(nv)}-{max(nv)} per subject)")
    print(f"[check] subjects={ds.subjects}")
    print(f"[check] passage_order={d['passage_order']}  (between-passage presentation order: "
          f"{ds.meta['order']['between_passage']})")
    starts = np.flatnonzero(np.r_[True, np.diff(ds.block_ids) != 0])
    print("[check] first passages in the order returned (block id, passage, topic):",
          [(int(ds.block_ids[s]), ds.meta["passage_label"][s], ds.meta["topic"][s]) for s in starts[:6]])
    for s in starts[:2]:
        for i in range(s, s + int((ds.block_ids == ds.block_ids[s]).sum())):
            print(f"         block {ds.block_ids[i]:>3} pos {ds.position[i]}  {ds.n_words[i]:>2} words  {ds.texts[i]}")
    print(f"[check] contiguous {sp.n_folds}-fold: topics per test fold {topics_per_fold}; "
          f"topics held out whole: {whole_topics}/{len(set(topic.tolist()))}")
    w = ds.n_words
    print(f"[check] words per sentence: {w.min()}-{w.max()}, mean {w.mean():.2f}, sd {w.std(ddof=1):.2f} "
          f"(Pereira 2018: {PAPER_WORDS[d['experiment']]}; whitespace tokens here)")

    report = {"experiment": d["experiment"], "passage_order": d["passage_order"], "n_samples": ds.n,
              "n_blocks": n_blocks, "n_subjects": len(ds.subjects), "subjects": ds.subjects,
              "n_voxels_per_subject": dict(zip(ds.subjects, nv)), "stimulus_hash": ds.stimulus_hash(),
              "sha1": ds.meta["sha1"], "contiguous_topics_per_fold": topics_per_fold,
              "contiguous_topics_held_out_whole": whole_topics,
              "words": {"min": int(w.min()), "max": int(w.max()), "mean": float(w.mean()), "sd": float(w.std(ddof=1))}}

    # ---- noise ceiling
    ref_paper, ref_file = cfg["claim"]["reference_ceiling"], _schrimpf_ceiling()
    strip = lambda c: {k: v for k, v in c.items() if k != "per_voxel"}  # noqa: E731
    if a.no_ceiling:
        print("[check] --no-ceiling: noise ceilings skipped")
    else:
        t = time.time()
        c_s = ceiling.extrapolation_ceiling(ds.Y, splits.make("shuffled", ds.block_ids, ceiling.SCHRIMPF_N_FOLDS,
                                                              cfg["seed"]), subjects=ds.subjects, seed=cfg["seed"])
        print(f"[check] ceiling, Schrimpf protocol (shuffled {ceiling.SCHRIMPF_N_FOLDS}-fold OLS, extrapolated): "
              f"{c_s['ceiling']:.4f}  [{time.time() - t:.0f}s]  per-subject medians "
              f"{ {k: round(v, 3) for k, v in c_s['per_subject_median'].items()} }")
        print(f"         reference: Schrimpf 2021 text {ref_paper}; Schrimpf's own S3 ceiling file "
              f"{'not downloaded' if ref_file is None else f'{ref_file:.4f}'} "
              "(both experiments pooled, all-atlas pool; see ceiling.py for the differences)")
        ref = ref_file if ref_file is not None else ref_paper
        if ref is not None:
            checks.append((f"ceiling within ±{SAME_MAGNITUDE} of the reference {ref:.4f}",
                           abs(c_s["ceiling"] - ref) < SAME_MAGNITUDE, f"{c_s['ceiling']:.4f}"))
        report["ceiling_schrimpf_protocol"] = strip(c_s)
    report["reference_ceiling"] = {"schrimpf2021_text": ref_paper, "schrimpf_s3_file": ref_file}

    if not (a.quick or a.no_ceiling):
        t = time.time()
        c_c = ceiling.extrapolation_ceiling(ds.Y, sp, subjects=ds.subjects, seed=cfg["seed"])
        print(f"[check] ceiling, contiguous {sp.n_folds}-fold (this pipeline's primary split): "
              f"{c_c['ceiling']:.4f}  [{time.time() - t:.0f}s]")
        report["ceiling_contiguous"] = strip(c_c)

    v = ds.meta["voxels"]
    R = ceiling.roi_means(ds.Y, [v["roi_index"][s] for s in ds.subjects], len(v["roi_names"]))
    sh = ceiling.split_half(R, seed=cfg["seed"])
    print(f"[check] diagnostic only, not comparable to 0.32: split-half Spearman-Brown reliability of the "
          f"group-mean fROI response ({len(v['roi_names'])} fROIs, {sh['subjects_per_half']} subjects per half): "
          f"mean {sh['mean_spearman_brown']:.3f}")
    report["split_half_froi"] = dict(sh, roi_names=v["roi_names"])

    out = ROOT / "results" / "pereira2018" / f"check_{d['experiment']}.json"   # not data/: it may be read-only
    out.parent.mkdir(parents=True, exist_ok=True)
    report["checks"] = [{"name": n, "ok": bool(o), "info": i} for n, o, i in checks]
    report["seconds"] = round(time.time() - t0, 1)
    out.write_text(json.dumps(report, indent=1, ensure_ascii=False))
    fails = 0
    for name, ok, info in checks:
        print(f"[check] {'PASS' if ok else 'FAIL'}  {name}" + (f"  ({info})" if info else ""))
        fails += not ok
    print(f"[check] report -> {out.resolve()}  ({time.time() - t0:.0f}s)")
    return 1 if fails else 0


if __name__ == "__main__":
    sys.exit(main())
