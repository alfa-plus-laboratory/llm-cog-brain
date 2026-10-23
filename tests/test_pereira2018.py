"""Tests for d1.data.pereira2018: sample order, blocks, subjects, provenance.

Most tests run on a small fabricated file with the Brain-Score assembly's layout
(rows of both experiments interleaved, as in the real file). One test runs on
the real file if it has been downloaded (scripts/check_pereira2018.py --download).
"""
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from d1 import data, splits  # noqa: E402

REAL = ROOT / "data" / "raw" / "pereira2018" / "brainscore" / data.PEREIRA_FILE

# experiment -> passages as (topic, n sentences); passage_index is 1-based in listed order
FAKE = {
    "384sentences": [("music", 4), ("dwelling", 4), ("music", 4), ("fruit", 4), ("dwelling", 4), ("fruit", 4)],
    "243sentences": [("bees", 3), ("bees", 4), ("dreams", 3), ("dreams", 3)],
}
# subject -> (n voxels, experiments done)
FAKE_SUBJECTS = {"018": (3, ("384sentences", "243sentences")), "199": (2, ("384sentences",)),
                 "215": (4, ("243sentences",))}


def _write_fake(path, partial_nan=False):
    import h5py
    rows = []
    for exp, passages in FAKE.items():
        num = 0
        for pi, (topic, k) in enumerate(passages, start=1):
            for j in range(k):
                rows.append((exp, num, pi, topic, f"{topic}{pi}", f"{exp} p{pi} s{j} w " + "x " * j))
                num += 1
    order = np.random.default_rng(0).permutation(len(rows))    # file rows are not in stimulus order
    rows = [rows[i] for i in order]
    subj, vox = [], []
    for s, (nv, _) in FAKE_SUBJECTS.items():
        subj += [s] * nv
        vox += list(range(nv))
    X = np.full((len(rows), len(subj)), np.nan)
    for i, (exp, num, *_rest) in enumerate(rows):
        for c, s in enumerate(subj):
            if exp in FAKE_SUBJECTS[s][1]:
                X[i, c] = 1000 * (exp == "384sentences") + num + c / 100  # encodes (experiment, sentence, voxel)
    if partial_nan:            # one 384sentences sentence missing for one voxel of subject 018
        X[next(i for i, r in enumerate(rows) if r[0] == "384sentences"), 0] = np.nan

    def strings(f, name, values):
        f.create_dataset(name, data=list(values), dtype=h5py.string_dtype())
    with h5py.File(path, "w") as f:
        strings(f, "experiment", (r[0] for r in rows))
        f["stimulus_num"] = np.array([r[1] for r in rows])
        f["passage_index"] = np.array([r[2] for r in rows])
        strings(f, "passage_category", (r[3] for r in rows))
        strings(f, "passage_label", (r[4] for r in rows))
        strings(f, "sentence", (r[5] for r in rows))
        strings(f, "subject", subj)
        strings(f, "roi", ("LH_IFG" if v % 2 else "LH_AntTemp" for v in vox))
        f["data"] = X


class TestFakeFile(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.TemporaryDirectory()
        cls.path = Path(cls.tmp.name) / data.PEREIRA_FILE
        _write_fake(cls.path)

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()

    def load(self, exp="384sentences", order="topic_interleaved"):
        return data.pereira2018(self.path, exp, order, verify=False)

    def test_within_passage_reading_order_and_contiguous_blocks(self):
        for exp in FAKE:
            for order in ("topic_interleaved", "stimulus_list"):
                ds = self.load(exp, order)                       # .check() ran inside
                for b in np.unique(ds.block_ids):
                    idx = np.flatnonzero(ds.block_ids == b)
                    self.assertTrue(np.array_equal(idx, np.arange(idx[0], idx[-1] + 1)))
                    self.assertEqual(ds.position[idx].tolist(), list(range(len(idx))))
                    self.assertEqual([t.split()[2] for t in np.array(ds.texts)[idx]],
                                     [f"s{j}" for j in range(len(idx))])

    def test_passage_orders(self):
        def passages(ds):
            return ds.block_ids[np.r_[True, np.diff(ds.block_ids) != 0]].tolist()
        self.assertEqual(passages(self.load(order="stimulus_list")), [1, 2, 3, 4, 5, 6])
        # topics by first appearance: music(1, 3), dwelling(2, 5), fruit(4, 6)
        self.assertEqual(passages(self.load(order="topic_interleaved")), [1, 2, 4, 3, 5, 6])
        self.assertEqual(passages(self.load("243sentences", "topic_interleaved")), [1, 3, 2, 4])

    def test_subjects_and_responses_follow_the_rows(self):
        ds = self.load()
        self.assertEqual(ds.subjects, ["018", "199"])            # 215 did not do 384sentences
        self.assertEqual([y.shape[1] for y in ds.Y], [3, 2])
        num = np.round(ds.Y[0][:, 0] - 1000).astype(int)        # fake value encodes stimulus_num
        texts_num = [FAKE_index(t) for t in ds.texts]
        self.assertEqual(num.tolist(), texts_num)
        self.assertEqual(ds.meta["voxels"]["per_subject"], {"018": 3, "199": 2})

    def test_word_counts_duration_meta(self):
        ds = self.load("243sentences")
        self.assertEqual(ds.n_words.tolist(), [len(t.split()) for t in ds.texts])
        self.assertTrue((ds.duration == data.PEREIRA_SENTENCE_SECONDS).all())
        self.assertEqual(ds.meta["order"]["passage_order"], "topic_interleaved")
        self.assertIn("not recoverable", ds.meta["order"]["between_passage"])
        self.assertEqual(len(ds.meta["source_row"]), ds.n)

    def test_contiguous_split_keeps_passages_whole(self):
        ds = self.load()
        sp = splits.make("contiguous", ds.block_ids, 3, 0)
        self.assertEqual(sp.describe(ds.block_ids)["blocks_spanning_train_and_test"], 0)

    def test_sha1_mismatch_is_refused(self):
        with self.assertRaises(ValueError):
            data.pereira2018(self.path, verify=True)

    def test_bad_arguments(self):
        with self.assertRaises(ValueError):
            data.pereira2018(self.path, "180concepts", verify=False)
        with self.assertRaises(ValueError):
            data.pereira2018(self.path, passage_order="by_passage_id", verify=False)

    def test_partial_missing_voxels_are_refused(self):
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / data.PEREIRA_FILE
            _write_fake(p, partial_nan=True)
            with self.assertRaises(ValueError):
                data.pereira2018(p, "384sentences", verify=False)
            data.pereira2018(p, "243sentences", verify=False)   # the other experiment is intact


def FAKE_index(text):
    """stimulus_num of a fake sentence, from its text: '<exp> p<passage> s<j> ...'."""
    exp, p, s = text.split()[:3]
    sizes = [k for _, k in FAKE[exp]]
    return sum(sizes[:int(p[1:]) - 1]) + int(s[1:])


@unittest.skipUnless(REAL.is_file(), "real Pereira2018 assembly not downloaded")
class TestRealFile(unittest.TestCase):
    def test_shape_and_provenance(self):
        ds = data.pereira2018(REAL, "384sentences")
        self.assertEqual((ds.n, len(np.unique(ds.block_ids)), len(ds.subjects)), (384, 96, 9))
        self.assertEqual(ds.meta["sha1"], data.PEREIRA_SHA1)
        self.assertEqual(ds.texts[0], "An accordion is a portable musical instrument with two keyboards.")
        ds3 = data.pereira2018(REAL, "243sentences")
        self.assertEqual((ds3.n, len(np.unique(ds3.block_ids)), len(ds3.subjects)), (243, 72, 6))


if __name__ == "__main__":
    unittest.main()
