"""Metrics on synthetic data with known answers (PLAN.md §9 tests). CPU only."""
import unittest

import numpy as np

from analysis.metrics import (ProblemBootstrap, auroc, cohen_kappa, ece, pair_thresholds, ranks,
                              threshold_at_tar)


class MetricTests(unittest.TestCase):
    def test_auroc_known_values(self):
        y = np.array([1, 1, 0, 0])
        self.assertEqual(auroc(np.array([4.0, 3, 2, 1]), y), 1.0)
        self.assertEqual(auroc(np.array([1.0, 2, 3, 4]), y), 0.0)
        self.assertEqual(auroc(np.array([1.0, 1, 1, 1]), y), 0.5)  # ties count half
        self.assertEqual(auroc(np.array([3.0, 1, 2, 0]), y), 0.75)

    def test_ranks_average_ties(self):
        np.testing.assert_array_equal(ranks(np.array([10.0, 20, 10, 30])), [1.5, 3, 1.5, 4])

    def test_threshold_at_tar(self):
        c = np.arange(10, dtype=float)
        t = threshold_at_tar(c, 0.9)
        self.assertGreaterEqual(np.mean(c >= t), 0.9)
        self.assertLess(np.mean(c >= t + 1e-9), 0.9)

    def test_pair_reaches_target_tar(self):
        rng = np.random.default_rng(0)
        ca, cb = rng.normal(1, 1, 500), rng.normal(1, 1, 500)
        ta, tb, r = pair_thresholds(ca, cb, 0.9)
        self.assertGreaterEqual(np.mean((ca >= ta) & (cb >= tb)), 0.9)
        self.assertGreater(r, 0.9)  # independent verifiers need each above 90% (about 95%)

    def test_identical_verifiers_need_no_extra_tar(self):
        c = np.random.default_rng(1).normal(size=400)
        _, _, r = pair_thresholds(c, c, 0.9)
        self.assertAlmostEqual(r, 0.9, places=2)

    def test_kappa(self):
        x = np.array([1, 1, 0, 0, 1, 0])
        self.assertEqual(cohen_kappa(x, x), 1.0)
        self.assertAlmostEqual(cohen_kappa(x, 1 - x), -1.0)

    def test_ece_perfect_calibration_is_small(self):
        rng = np.random.default_rng(2)
        s = rng.normal(size=20000)
        y = (rng.random(20000) < 1 / (1 + np.exp(-s))).astype(float)
        self.assertLess(ece(s, y), 0.02)

    def test_bootstrap_keeps_problems_together(self):
        pids = ["a", "a", "b", "b", "c"]
        boot = ProblemBootstrap(pids, n=50)
        for ix in boot.indices():
            for g in ("a", "b"):
                members = [i for i in ix if pids[i] == g]
                self.assertEqual(len(members) % 2, 0)


if __name__ == "__main__":
    unittest.main()
