"""Datasets, in one shape: samples in presentation order, grouped into blocks.

A *sample* is one row of the response matrix (a sentence for Pereira2018, a TR
for story-listening data). A *block* is the unit that must never straddle the
train/test boundary (a passage, a story). Everything downstream relies on the
samples being in the order they were presented -- the circular-shift null and
the contiguous split are both defined on that order. Where the data has no
single presentation order between blocks (Pereira2018), the loader builds one
explicitly and says so in meta["order"].
"""
from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from .config import PASSAGE_ORDERS, PEREIRA_EXPERIMENTS

_ROOT = Path(__file__).resolve().parents[2]


@dataclass
class Dataset:
    name: str
    tokens: list[np.ndarray] | None      # per-sample token ids (synthetic backend)
    texts: list[str] | None              # per-sample text (hf backend)
    block_ids: np.ndarray                # (n,) block (passage/story) of each sample, presentation order
    position: np.ndarray                 # (n,) 0-based position of the sample within its block
    n_words: np.ndarray                  # (n,)
    duration: np.ndarray                 # (n,) seconds covered by the sample
    Y: list[np.ndarray]                  # per subject (n, n_voxels)
    subjects: list[str]
    meta: dict = field(default_factory=dict)

    @property
    def n(self) -> int:
        return len(self.block_ids)

    def stimulus_hash(self) -> str:
        h = hashlib.sha256()
        if self.texts is not None:
            for t in self.texts:
                h.update(t.encode())
                h.update(b"\0")
        else:
            for t in self.tokens:
                h.update(np.asarray(t, dtype=np.int64).tobytes())
                h.update(b"\0")
        h.update(np.asarray(self.block_ids, dtype=np.int64).tobytes())
        return h.hexdigest()[:16]

    def check(self) -> "Dataset":
        n = self.n
        for name in ("position", "n_words", "duration"):
            if len(getattr(self, name)) != n:
                raise ValueError(f"{name} has {len(getattr(self, name))} rows, expected {n}")
        for s, y in zip(self.subjects, self.Y):
            if y.shape[0] != n:
                raise ValueError(f"subject {s}: Y has {y.shape[0]} rows, expected {n}")
        # Blocks must be contiguous runs in presentation order.
        change = np.flatnonzero(np.diff(self.block_ids)) + 1
        starts = np.r_[0, change]
        if len(np.unique(self.block_ids)) != len(starts):
            raise ValueError("block_ids are not contiguous runs in presentation order")
        return self


def _ar1(x: np.ndarray, rho: float) -> np.ndarray:
    out = np.empty_like(x)
    out[0] = x[0]
    for t in range(1, len(x)):
        out[t] = rho * out[t - 1] + x[t]
    return out * np.sqrt(1 - rho ** 2)


def _z(a: np.ndarray) -> np.ndarray:
    sd = a.std(0, keepdims=True)
    return (a - a.mean(0, keepdims=True)) / np.where(sd > 0, sd, 1.0)


def synthetic(cfg: dict, seed: int) -> Dataset:
    """Pereira2018-shaped toy data whose ground truth is known.

    Responses = stimulus semantics + sentence position + word rate (the two
    confounds) + a passage-level drift that has nothing to do with the stimulus
    (the thing a shuffled split leaks) + noise, smoothed in time.

    The semantic drive is built from the same token embeddings the toy LM sees
    (`activations._toy_lm`), so the LM can predict it; with ``signal: false`` it
    is absent and an honest pipeline must not report the claim as holding.
    """
    from .activations import toy_embeddings

    s = cfg
    rng = np.random.default_rng(seed)
    nb, bl, nv = s["n_blocks"], s["block_len"], s["n_voxels"]
    n = nb * bl
    block_ids = np.repeat(np.arange(nb), bl)
    position = np.tile(np.arange(bl), nb)
    n_words = rng.integers(s["min_words"], s["max_words"] + 1, size=n)
    # Content tokens 1..vocab-1; id 0 is reserved as the filler/pad token.
    tokens = [rng.integers(1, s["vocab"], size=k) for k in n_words]
    duration = np.full(n, float(s["duration_s"]))

    emb = toy_embeddings(s["vocab"])                        # shared with the toy LM
    k_sem = 8
    A = np.random.default_rng(seed + 1).standard_normal((emb.shape[1], k_sem))
    sem = _z(np.stack([emb[t].mean(0) for t in tokens]) @ A)  # (n, k_sem)
    pos1h = np.eye(bl)[position]
    wr = _z((n_words / duration)[:, None])
    g = s["gains"]

    Y, subjects = [], []
    for si in range(s["n_subjects"]):
        r = np.random.default_rng(seed + 100 + si)
        y = np.zeros((n, nv))
        if s["signal"]:
            y += g["semantic"] * sem @ r.standard_normal((k_sem, nv)) / np.sqrt(k_sem)
        y += g["position"] * pos1h @ r.standard_normal((bl, nv))
        y += g["word_rate"] * wr @ r.standard_normal((1, nv))
        y += g["drift"] * r.standard_normal((nb, nv))[block_ids]
        y += g["noise"] * r.standard_normal((n, nv))
        Y.append(_ar1(y, s["ar_rho"]))
        subjects.append(f"synth{si + 1:02d}")

    # Pseudo-words, so the hf backend can be exercised on the same stimuli.
    texts = [" ".join(f"w{t}" for t in toks) for toks in tokens]
    return Dataset(
        name="synthetic", tokens=tokens, texts=texts, block_ids=block_ids, position=position,
        n_words=n_words, duration=duration, Y=Y, subjects=subjects,
        meta={"signal": bool(s["signal"]), "seed": seed, "n_blocks": nb, "block_len": bl,
              "n_voxels": nv, "vocab": s["vocab"], "generator": "d1.data.synthetic"},
    ).check()


# ---------------------------------------------------------------- Pereira2018

PEREIRA_FILE = "assy_Pereira2018_language.nc"
PEREIRA_URL = "https://brainscore-storage.s3.amazonaws.com/brainscore-language/" + PEREIRA_FILE
# brainscore_language/data/pereira2018/__init__.py: data_registry['Pereira2018.language']
PEREIRA_SHA1 = "f8434b4022f5b2c862f0ff2854d5b3f5f2a7fb96"
PEREIRA_SENTENCE_SECONDS = 4.0          # Pereira et al. 2018 Methods: "Each sentence was shown for 4 s"


def _sha1(path: Path) -> str:
    h = hashlib.sha1()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _strings(a) -> np.ndarray:
    return np.array([x.decode() if isinstance(x, bytes) else str(x) for x in a])


def _pereira_passage_order(topic: np.ndarray, how: str) -> np.ndarray:
    """Order of the passages (given in sentence-list order, with their topics).

    stimulus_list      as listed in the stimulus file (passage_index order).
    topic_interleaved  first passage of every topic, then every topic's second
                       passage, ...; topics in order of first appearance.
    """
    if how == "stimulus_list":
        return np.arange(len(topic))
    first, seen, rank = {}, {}, np.zeros(len(topic), dtype=int)
    for i, t in enumerate(topic):
        first.setdefault(t, i)
        rank[i] = seen.get(t, 0)        # this passage is the rank-th of its topic
        seen[t] = rank[i] + 1
    return np.array(sorted(range(len(topic)), key=lambda i: (rank[i], first[topic[i]])))


def pereira2018(path: str | Path | None, experiment: str = "384sentences",
                passage_order: str = "topic_interleaved", verify: bool = True) -> Dataset:
    """Pereira et al. 2018 (Nat Commun 9:963) experiment 2 or 3, as Schrimpf et al. 2021 used it.

    Release: the Brain-Score language assembly ``Pereira2018.language`` (one
    netCDF file, read with h5py). Its values are identical to the language-network
    voxels of the assembly Schrimpf et al.'s neural-nlp code loaded
    (s3://brainio-language/Pereira2018.nc); provenance is checked on load
    (sha1). One response per sentence per voxel.

    Sample order -- read this before using the circular-shift null or the contiguous split:

    * Within a passage the order is the presentation order: sentences of a
      passage were always shown consecutively in reading order, and the
      stimulus list keeps that order (stimulus_num).
    * Between passages there is NO presentation order to recover. Pereira et al.
      randomised the division of passages into runs, and their order, per
      participant and per scanning session; each participant saw every passage
      in 3 sessions, and the released response to a sentence is one image built
      from those 3 presentations. No single passage order applies to the data.
      The between-passage order returned here is therefore a construction,
      chosen by ``passage_order`` and recorded in ``meta["order"]``:

        topic_interleaved (default)  one passage of every topic, then the next
            passage of every topic, ... A contiguous fold then holds out whole
            passages drawn from many topics, which is how Hadidi, Feghhi et al.
            build their Pereira folds (beyond-brainscore, PereiraSplitter: one
            passage per topic, 12 topics per fold, 8 folds for 384sentences).
        stimulus_list  the stimulus-file order. For 243sentences it groups the
            3 passages of a topic together, so contiguous folds hold out whole
            topics (a stricter test than the critique's).

      Neither is the presentation order, so the circular-shift null and the
      contiguous folds are defined on a constructed passage sequence; with
      passages as the blocks, nothing straddles train/test within a passage.

    Args:
        path: the .nc file or the directory holding it; relative to the repo root.
        experiment: "384sentences" (Pereira exp. 2, 9 subjects) or "243sentences" (exp. 3, 6 subjects).
        verify: check the file's SHA-1 against the Brain-Score data registry.
    """
    import h5py

    if experiment not in PEREIRA_EXPERIMENTS:
        raise ValueError(f"experiment must be one of {PEREIRA_EXPERIMENTS}, got {experiment!r}")
    if passage_order not in PASSAGE_ORDERS:
        raise ValueError(f"passage_order must be one of {PASSAGE_ORDERS}, got {passage_order!r}")
    if path is None:
        raise ValueError("dataset.path is not set")
    p = Path(path)
    p = p if p.is_absolute() else _ROOT / p
    if p.is_dir():
        p = p / PEREIRA_FILE
    if not p.is_file():
        raise FileNotFoundError(f"{p} not found; fetch it with scripts/check_pereira2018.py --download")
    sha1 = _sha1(p) if verify else None
    if verify and sha1 != PEREIRA_SHA1:
        raise ValueError(f"{p}: sha1 {sha1} != Brain-Score registry {PEREIRA_SHA1}; not the audited release")

    with h5py.File(p, "r") as f:
        rows = np.flatnonzero(_strings(f["experiment"][:]) == experiment)
        if not len(rows):
            raise ValueError(f"{p}: no rows for experiment {experiment!r}")
        stim_num = f["stimulus_num"][:][rows]
        passage = f["passage_index"][:][rows]
        topic = _strings(f["passage_category"][:])[rows]
        label = _strings(f["passage_label"][:])[rows]
        texts = _strings(f["sentence"][:])[rows]
        subj_of_vox = _strings(f["subject"][:])
        roi_of_vox = _strings(f["roi"][:])
        X = f["data"][:][rows]

    # 1. sentence-list order; within a passage this is the reading (= presentation) order
    o = np.argsort(stim_num, kind="stable")
    rows, stim_num, passage, topic, label, texts, X = (a[o] for a in (rows, stim_num, passage, topic, label, texts, X))
    if not np.array_equal(stim_num, np.arange(len(stim_num))):
        raise ValueError("stimulus_num is not 0..n-1 within the experiment")
    starts = np.flatnonzero(np.r_[True, passage[1:] != passage[:-1]])
    if len(starts) != len(np.unique(passage)):
        raise ValueError("a passage's sentences are not consecutive in the stimulus list")
    ends = np.r_[starts[1:], len(passage)]
    for s, e in zip(starts, ends):
        if len(set(topic[s:e])) != 1:
            raise ValueError(f"passage {passage[s]} spans several topics")

    # 2. between-passage order: a construction (see docstring)
    porder = _pereira_passage_order(topic[starts], passage_order)
    order = np.concatenate([np.arange(starts[i], ends[i]) for i in porder])
    position = np.concatenate([np.arange(ends[i] - starts[i]) for i in porder])
    rows, passage, topic, label, texts, X = (a[order] for a in (rows, passage, topic, label, texts, X))

    # 3. voxels: each subject's own language-network voxels; NaN = subject did not do this experiment
    Y, subjects, roi_index = [], [], {}
    roi_names = sorted(set(roi_of_vox.tolist()))
    for s in sorted(set(subj_of_vox.tolist())):
        cols = np.flatnonzero(subj_of_vox == s)
        nan = np.isnan(X[:, cols])
        if nan.all():
            continue
        if (nan.any(0) & ~nan.all(0)).any():
            raise ValueError(f"subject {s}: some voxels are missing only some sentences")
        keep = cols[~nan.any(0)]
        Y.append(np.ascontiguousarray(X[:, keep]))
        subjects.append(s)
        roi_index[s] = [roi_names.index(r) for r in roi_of_vox[keep]]

    n_words = np.array([len(t.split()) for t in texts])
    meta = {
        "release": "Brain-Score language assembly Pereira2018.language",
        "source_url": PEREIRA_URL,
        "file": str(p), "sha1": sha1, "sha1_expected": PEREIRA_SHA1, "sha1_verified": bool(verify),
        "lineage": ("Pereira et al. 2018 (Nat Commun 9:963, OSF crwz7) exps 2-3; per-sentence responses "
                    "of the language-network voxels as packaged for Schrimpf et al. 2021 (neural-nlp "
                    "load_Pereira2018_Blank, *_persent.mat); identical to the language atlas of "
                    "s3://brainio-language/Pereira2018.nc"),
        "experiment": experiment,
        "n_subjects": len(subjects), "n_sentences": int(len(texts)),
        "n_passages": int(len(starts)), "n_topics": len(set(topic.tolist())),
        "voxels": {
            "selection": ("atlas=language, atlas_selection=from90to100prcnt (Brain-Score data_packaging: "
                          "atlas_selection_lower=90); voxels complete for this experiment"),
            "per_subject": {s: int(y.shape[1]) for s, y in zip(subjects, Y)},
            "roi_names": roi_names,
            "roi_index": roi_index,
        },
        "order": {
            "within_passage": "presentation order (reading order; stimulus_num)",
            "between_passage": ("not recoverable: passage order was randomised per participant and session, "
                                "and each response combines 3 sessions (Pereira et al. 2018, Methods)"),
            "passage_order": passage_order,
            "passages": passage[np.r_[True, passage[1:] != passage[:-1]]].tolist(),
        },
        "duration": f"{PEREIRA_SENTENCE_SECONDS} s per sentence on screen (then 4 s fixation)",
        "source_row": rows.tolist(),
        "passage_label": label.tolist(),
        "topic": topic.tolist(),
    }
    return Dataset(
        name="pereira2018", tokens=None, texts=texts.tolist(), block_ids=passage.astype(int),
        position=position, n_words=n_words, duration=np.full(len(texts), PEREIRA_SENTENCE_SECONDS),
        Y=Y, subjects=subjects, meta=meta,
    ).check()


def load(cfg: dict) -> Dataset:
    d = cfg["dataset"]
    if d["name"] == "synthetic":
        return synthetic(d["synthetic"], cfg["seed"])
    if d["name"] == "pereira2018":
        return pereira2018(d["path"], d["experiment"], d["passage_order"])
    raise ValueError(f"unknown dataset {d['name']!r}")
