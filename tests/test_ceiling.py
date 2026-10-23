"""Unit tests for d1.ceiling.  Run:  .../venv/bin/python -m unittest discover -s tests -v"""
import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from d1 import ceiling, splits  # noqa: E402


def _shared_signal(N, n, v, q, gain, seed):
    """N subjects, each with its own v voxels = gain * (shared signal, own mixing) + unit noise."""
    rng = np.random.default_rng(seed)
    S = rng.standard_normal((n, q))
    return [gain * S @ (rng.standard_normal((q, v)) / np.sqrt(q)) + rng.standard_normal((n, v))
            for _ in range(N)]


class TestOLSFromGram(unittest.TestCase):
    def _check(self, n, p):
        from sklearn.linear_model import LinearRegression
        rng = np.random.default_rng(0)
        X, Y = rng.standard_normal((n, p)) + 0.5, rng.standard_normal((n, 7))
        te = np.arange(0, n, 5)
        tr = np.setdiff1d(np.arange(n), te)
        ours = ceiling.ols_predict_from_gram(X @ X.T, Y, tr, te)
        ref = LinearRegression().fit(X[tr], Y[tr]).predict(X[te])
        np.testing.assert_allclose(ours, ref, atol=1e-8)

    def test_more_features_than_samples(self):      # the ceiling's case: thousands of pool voxels
        self._check(60, 300)

    def test_fewer_features_than_samples(self):
        self._check(80, 10)

    def test_pool_gram_is_sum_of_subject_grams(self):
        rng = np.random.default_rng(1)
        A, B, Y = rng.standard_normal((40, 30)), rng.standard_normal((40, 50)), rng.standard_normal((40, 3))
        tr, te = np.arange(30), np.arange(30, 40)
        np.testing.assert_allclose(ceiling.ols_predict_from_gram(A @ A.T + B @ B.T, Y, tr, te),
                                   ceiling.ols_predict_from_gram(np.hstack([A, B]) @ np.hstack([A, B]).T,
                                                                 Y, tr, te), atol=1e-10)


class TestSaturatingFit(unittest.TestCase):
    def test_recovers_noiseless_curve(self):
        k = np.arange(2, 10)
        for v0, tau in [(0.3, 1.5), (0.8, 4.0), (0.05, 0.7)]:
            got = ceiling.fit_saturating(k, v0 * (1 - np.exp(-k / tau)))
            self.assertAlmostEqual(float(got), v0, delta=0.01)

    def test_v0_is_clipped_to_unit_interval(self):
        k = np.arange(2, 10)
        self.assertEqual(float(ceiling.fit_saturating(k, -0.1 * np.ones(8))), 0.0)
        self.assertEqual(float(ceiling.fit_saturating(k, 0.5 * k)), 1.0)


class TestExtrapolationCeiling(unittest.TestCase):
    def _ceil(self, gain, seed=0, kind="shuffled"):
        Y = _shared_signal(6, 120, 60, 3, gain, seed)
        sp = splits.make(kind, np.arange(120) // 4, 5, 0)
        return ceiling.extrapolation_ceiling(Y, sp, n_subsets=5, n_boot=50)

    def test_pure_noise_has_no_ceiling(self):
        self.assertLess(self._ceil(0.0)["ceiling"], 0.05)

    def test_monotone_in_signal(self):
        c = [self._ceil(g)["ceiling"] for g in (0.0, 0.5, 1.0, 3.0)]
        self.assertEqual(c, sorted(c), c)

    def test_close_to_the_true_limit(self):
        # best possible predictor of a voxel is its shared part: r = g / sqrt(g^2 + 1) = 0.707 at g = 1
        self.assertAlmostEqual(self._ceil(1.0)["ceiling"], 1 / np.sqrt(2), delta=0.1)

    def test_deterministic(self):
        a, b = self._ceil(1.0, seed=3), self._ceil(1.0, seed=3)
        self.assertEqual(a["ceiling"], b["ceiling"])
        np.testing.assert_array_equal(a["per_voxel"], b["per_voxel"])

    def test_contiguous_split_runs(self):
        c = self._ceil(1.0, kind="contiguous")
        self.assertEqual(c["split_kind"], "contiguous")
        self.assertTrue(0 < c["ceiling"] <= 1)

    def test_every_subject_and_size_is_scored(self):
        Y = _shared_signal(4, 40, 10, 2, 1.0, 0)
        folds = list(splits.make("shuffled", np.arange(40) // 4, 5, 0).outer())
        sc = ceiling.holdout_scores(Y, folds, n_subsets=10)
        for s in range(4):          # C(4, k) <= 10 for every k: all subsets, every subject in every k
            self.assertEqual(sorted(sc[s]), [2, 3, 4])
            self.assertEqual(sc[s][4].shape, (5, 10))           # one subset x 5 folds, 10 voxels


class TestRidgeCeiling(unittest.TestCase):
    """The ridge ceiling must be the readout's estimator, and reduce to the OLS ceiling as alpha -> 0."""

    def _same_as_readout(self, v, kind):
        from d1 import readout
        Y = _shared_signal(3, 64, v, 2, 1.0, 0)
        b = np.arange(64) // 4
        sp = splits.make(kind, b, 4, 0)
        ridge = ceiling.Ridge(b, np.logspace(-1, 4, 6), n_inner=3)
        ours = ceiling._ridge_fit_folds(Y, sp, ridge)([0, 1], 2)
        ref, _ = readout.nested_predict({"pool": np.hstack([Y[0], Y[1]])}, Y[2], sp, b, 3, ridge.alphas)
        for p, (_, te) in zip(ours, sp.outer()):
            np.testing.assert_allclose(p, ref[te], atol=1e-8)

    def test_equals_readout_more_features_than_samples(self):    # readout takes its kernel path
        self._same_as_readout(40, "contiguous")

    def test_equals_readout_fewer_features_than_samples(self):   # readout takes its SVD path
        self._same_as_readout(6, "shuffled")

    def test_alpha_to_zero_is_the_ols_ceiling(self):
        # pool width <= 3 x 6 = 18 < 96 training rows: OLS is unique and z-scoring cannot change it,
        # so ridge with a vanishing alpha must give the OLS ceiling voxel for voxel
        Y = _shared_signal(4, 120, 6, 2, 1.0, 1)
        b = np.arange(120) // 4
        sp = splits.make("shuffled", b, 5, 0)
        ols = ceiling.extrapolation_ceiling(Y, sp, n_subsets=4, n_boot=30)
        gap = {}
        for a in (1e-6, 1e1, 1e3):
            r = ceiling.extrapolation_ceiling(Y, sp, n_subsets=4, n_boot=30,
                                              ridge=ceiling.Ridge(b, np.array([a]), n_inner=3))
            self.assertEqual(r["estimator"], "ridge")
            gap[a] = float(np.abs(r["per_voxel"] - ols["per_voxel"]).max())
        self.assertLess(gap[1e-6], 1e-6, gap)
        self.assertLess(gap[1e-6], gap[1e1], gap)
        self.assertLess(gap[1e1], gap[1e3], gap)              # shrinking more moves further from OLS

    def test_matched_scoring_and_alpha_report(self):
        Y = _shared_signal(4, 80, 30, 2, 1.0, 2)
        b = np.arange(80) // 4
        sp = splits.make("contiguous", b, 5, 0)
        r = ceiling.extrapolation_ceiling(Y, sp, n_subsets=3, n_boot=20, matched=True,
                                          ridge=ceiling.Ridge(b, np.logspace(-1, 4, 6), n_inner=4))
        m, t = r["matched"], r["by_scoring_and_aggregation"]
        self.assertEqual(m["per_voxel"].shape, (120,))
        self.assertAlmostEqual(m["ceiling"], np.mean(list(m["per_subject"].values())))
        self.assertEqual(t["per_fold_r"]["median_of_voxel_v0"], r["ceiling"])
        self.assertEqual(t["pooled_fold_centred_r"]["v0_of_mean_r"], m["ceiling"])
        self.assertAlmostEqual(t["pooled_fold_centred_r"]["mean_of_voxel_v0"], m["per_voxel"].mean())
        # extrapolating to infinitely many subjects cannot fall far below what 3 pool subjects already reach
        self.assertGreater(m["ceiling"], m["mean_r_at_all_subjects"] - 0.05)
        self.assertAlmostEqual(sum(r["ridge"]["alpha_chosen_share"].values()), 1.0)
        self.assertTrue(0 < m["ceiling"] <= 1)


class TestSplitHalf(unittest.TestCase):
    def test_spearman_brown(self):
        self.assertAlmostEqual(float(ceiling.spearman_brown(0.5)), 2 / 3)
        self.assertAlmostEqual(float(ceiling.spearman_brown(1.0)), 1.0)

    def test_matches_theory(self):
        # R_i = s + e_i, var e = sigma^2: reliability of a 2h-subject mean = 1 / (1 + sigma^2 / (2h))
        rng = np.random.default_rng(0)
        N, n, m, sigma = 8, 4000, 5, 2.0
        s = rng.standard_normal((n, m))
        R = [s + sigma * rng.standard_normal((n, m)) for _ in range(N)]
        out = ceiling.split_half(R, n_splits=50)
        self.assertAlmostEqual(out["mean_spearman_brown"], 1 / (1 + sigma ** 2 / N), delta=0.02)

    def test_noise_only_is_zero(self):
        rng = np.random.default_rng(1)
        R = [rng.standard_normal((4000, 5)) for _ in range(6)]
        self.assertLess(abs(ceiling.split_half(R, n_splits=50)["mean_spearman_brown"]), 0.03)

    def test_roi_means_shared_space(self):
        Y = [np.arange(12, dtype=float).reshape(3, 4), np.ones((3, 2))]
        out = ceiling.roi_means(Y, [[0, 0, 1, 1], [1, 0]], 2)
        np.testing.assert_allclose(out[0], [[0.5, 2.5], [4.5, 6.5], [8.5, 10.5]])
        self.assertEqual(out[1].shape, (3, 2))


if __name__ == "__main__":
    unittest.main()
