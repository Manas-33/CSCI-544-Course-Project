"""Bootstrap, thresholds, matching check and decision table on data with known answers. CPU only."""
import unittest
from dataclasses import replace
from statistics import NormalDist

import numpy as np

from analysis.final_data import Frame
from analysis.final_evaluate import delta_far, rates
from analysis.final_settings import RULES
from analysis.final_stats import (Boot, accepts, auroc_boot, decide, detectable, fast_pair_thresholds, interval, kappa_boot,
                                  kappa_from_counts, matching_check, spearman_boot)
from analysis.metrics import ProblemBootstrap, auroc, cohen_kappa, pair_thresholds, spearman

PHI = NormalDist().cdf


def clustered(rng, problems, per_problem, effect_points, base=0.6, sd=1.0):
    """Wrong answers of `problems` problems: accepts of two pairs with a shared problem effect. Returns
    (problem ids, accepts of x, accepts of y). True FAR(x) - FAR(y) = effect_points / 100."""
    ids = np.repeat([f"p{i:04d}" for i in range(problems)], per_problem)
    u = np.repeat(rng.normal(0, sd, problems), per_problem)
    z = NormalDist().inv_cdf
    total = np.sqrt(1 + sd ** 2)
    ty, tx = z(base) * total, z(base + effect_points / 100) * total
    noise = rng.normal(size=len(ids))
    return ids, u + 0.6 * noise + 0.8 * rng.normal(size=len(ids)) < tx, u + 0.6 * noise + 0.8 * rng.normal(size=len(ids)) < ty


def frame_of(ids, labels):
    return Frame("math", [{} for _ in ids], list(ids), {}, {}, {}, np.asarray(labels))


class BootTests(unittest.TestCase):
    def setUp(self):
        rng = np.random.default_rng(0)
        self.ids = sorted(f"p{i:03d}" for i in rng.integers(0, 50, 160))
        self.y = np.tile([0, 1], 80)
        self.x = np.round(rng.normal(size=160) + 0.5 * self.y, 1)
        self.z = np.round(rng.normal(size=160) + 0.5 * self.x, 1)
        self.boot, self.slow = Boot(self.ids, n=200), ProblemBootstrap(list(self.ids), n=200)

    def replicates(self, stat):
        return np.array([stat(ix) for ix in self.slow.indices()])

    def test_same_draws_as_the_plain_problem_bootstrap(self):
        w = self.y == 0
        np.testing.assert_allclose(auroc_boot(self.boot, self.x, self.y), self.replicates(lambda ix: auroc(self.x[ix], self.y[ix])))
        np.testing.assert_allclose(self.boot.ratio((self.x > 0) & w, w), self.replicates(lambda ix: np.mean(self.x[ix][w[ix]] > 0)))
        np.testing.assert_allclose(spearman_boot(self.boot, self.x, self.z, np.flatnonzero(w)),
                                   self.replicates(lambda ix: spearman(self.x[ix][w[ix]], self.z[ix][w[ix]])), atol=1e-12)
        np.testing.assert_allclose(kappa_boot(self.boot, self.x > 0, self.z > 0, w),
                                   self.replicates(lambda ix: cohen_kappa(self.x[ix][w[ix]] > 0, self.z[ix][w[ix]] > 0)))
        self.assertEqual(interval(self.boot.ratio((self.x > 0) & w, w)),
                         self.slow.interval(lambda ix: np.mean(self.x[ix][w[ix]] > 0)))

    def test_row_order_does_not_matter(self):
        order = np.random.default_rng(1).permutation(160)
        other = Boot(np.array(self.ids)[order], n=200)
        np.testing.assert_allclose(auroc_boot(self.boot, self.x, self.y), auroc_boot(other, self.x[order], self.y[order]))

    def test_whole_problems_are_drawn(self):
        ids = np.repeat(["a", "b", "c"], [4, 2, 1])
        boot = Boot(ids, n=50)
        items = boot.item_weights()
        for group in ("a", "b"):
            block = items[:, ids == group]
            self.assertTrue((block == block[:, :1]).all())
        self.assertTrue((boot.weights.sum(axis=1) == 3).all())

    def test_strata_are_resampled_separately(self):
        boot = Boot(["p1", "p2", "p3", "p1", "p2"], n=40, strata=["m", "m", "m", "c", "c"])
        self.assertEqual(boot.n_problems, 5)            # p1 of one domain is not p1 of the other
        self.assertTrue((boot.weights[:, :2].sum(axis=1) == 2).all() and (boot.weights[:, 2:].sum(axis=1) == 3).all())

    def test_problem_bootstrap_is_wider_than_one_over_answers(self):
        ids, x, y = clustered(np.random.default_rng(3), 60, 6, 0, sd=2.0)
        widths = []
        for groups in (ids, [f"row{i}" for i in range(len(ids))]):
            lo, hi = interval(Boot(groups, n=1000).ratio(x, np.ones(len(ids))))
            widths.append(hi - lo)
        self.assertGreater(widths[0], 1.5 * widths[1])


class ThresholdTests(unittest.TestCase):
    def test_fast_thresholds_equal_the_plans_procedure(self):
        rng = np.random.default_rng(5)
        for _ in range(150):
            n, steps = int(rng.integers(5, 300)), int(rng.integers(0, 3))
            a = np.round(rng.normal(size=n), steps)
            b = np.round(rng.normal(size=n) + 0.5 * a, steps)
            for tar in (0.8, 0.9, 0.95):
                self.assertEqual(fast_pair_thresholds(a, b, tar), pair_thresholds(a, b, tar))

    def test_all_scores_tied(self):
        t = fast_pair_thresholds(np.zeros(20), np.zeros(20), 0.9)
        self.assertTrue(accepts(np.zeros(5), np.zeros(5), t[:2]).all())

    def test_and_rule_accepts_at_the_threshold(self):
        np.testing.assert_array_equal(accepts([1, 1, 0.9], [2, 1.9, 2], (1, 2)), [True, False, False])


class DeltaTests(unittest.TestCase):
    """delta_far: the estimate, its interval and its coverage."""

    def test_counts_and_points(self):
        labels = np.array([1, 0] * 4)
        f = frame_of([f"p{i // 2}" for i in range(8)], labels)
        x, y = np.array([1, 1, 1, 0, 1, 0, 0, 0], dtype=bool), np.array([1, 1, 1, 1, 1, 1, 1, 0], dtype=bool)
        boot = Boot(f.problems, n=100)
        d = delta_far(f, x, y, boot, RULES)
        self.assertEqual((d["n_wrong"], d["points"]), (4, -50.0))
        r = rates(f, x, boot)
        self.assertEqual((r["accepted_correct"], r["n_correct"], r["accepted_wrong"], r["n_wrong"]), (3, 4, 1, 4))

    def test_a_planted_effect_is_recovered_and_covered(self):
        for effect in (-10.0, 0.0):
            rng = np.random.default_rng(int(100 - effect))
            covered, estimates, runs = 0, [], 200
            for _ in range(runs):
                ids, x, y = clustered(rng, 150, 3, effect)
                f = frame_of(ids, np.zeros(len(ids), dtype=int))
                d = delta_far(f, x, y, Boot(ids, n=400), RULES)
                estimates.append(d["points"])
                covered += d["ci95"][0] <= effect <= d["ci95"][1]
            self.assertLess(abs(np.mean(estimates) - effect), 0.6, effect)      # standard error of the mean is about 0.2
            self.assertTrue(0.90 <= covered / runs <= 0.99, f"coverage {covered / runs} for {effect}")

    def test_null_effect_verdicts(self):
        """With no effect, "better" is called about as often as a 95% interval allows on one side."""
        rng, better = np.random.default_rng(9), 0
        for _ in range(200):
            ids, x, y = clustered(rng, 150, 3, 0.0)
            d = delta_far(frame_of(ids, np.zeros(len(ids), dtype=int)), x, y, Boot(ids, n=400), RULES)
            better += decide(True, d["points"], d["ci95"], d["ci90"]).startswith("better")
        self.assertLessEqual(better / 200, 0.06)

    def test_detectable_effect(self):
        self.assertAlmostEqual(detectable(1.0), 2.8016, places=3)

    def test_a_group_without_wrong_answers(self):
        f = frame_of(["a", "a"], [1, 0])
        d = delta_far(f, np.array([True, True]), np.array([True, True]), Boot(["a", "a"], n=10), RULES, mask=np.array([True, False]))
        self.assertEqual(d["n_wrong"], 0)
        self.assertTrue(np.isnan(d["points"]))


class MatchingTests(unittest.TestCase):
    def test_limits_are_inclusive(self):
        for a, b, ok in ((0.85, 0.85, True), (0.95, 0.95, True), (0.849, 0.85, False), (0.951, 0.95, False), (0.90, 0.93, True),
                         (0.90, 0.9301, False), (0.87, 0.90, True), (0.85, 0.88, True), (0.88, 0.92, False)):
            self.assertEqual(matching_check(a, b)["passes"], ok, (a, b))

    def test_limits_hold_for_counts(self):
        for n in (200, 350, 700, 713):      # exactly 5 points off and exactly 3 points apart, whatever the rounding
            for k in range(n + 1):
                off = abs(1000 * k - 900 * n)       # exact integers: thousandths of n
                self.assertEqual(matching_check(k / n, k / n)["passes"], off <= 50 * n, (k, n))

    def test_old_limit_is_a_setting_not_a_constant(self):
        old = replace(RULES, match_tar_points=3.0)
        self.assertFalse(matching_check(0.86, 0.86, old)["passes"])
        self.assertTrue(matching_check(0.86, 0.86)["passes"])


class DecisionTests(unittest.TestCase):
    def check(self, expected, delta, ci95, ci90, matched=True):
        self.assertEqual(decide(matched, delta, ci95, ci90), expected)

    def test_each_row(self):
        self.check("better_meaningful", -6, (-9, -3), (-8.5, -3.5))
        self.check("better_small", -2.5, (-4.9, -0.1), (-4.5, -0.5))
        self.check("no_meaningful_benefit", 0.2, (-3.1, 3.2), (-2.6, 2.8))
        self.check("worse", 5, (1, 9), (2, 8))
        self.check("inconclusive", -2, (-6, 2), (-5, 1))

    def test_failed_matching_comes_first(self):
        for delta, ci95, ci90 in ((-6, (-9, -3), (-8.5, -3.5)), (5, (1, 9), (2, 8)), (0, (-1, 1), (-0.8, 0.8))):
            self.check("not_matched", delta, ci95, ci90, matched=False)

    def test_row_order(self):
        # entirely below 0 beats equivalence, even when the 90% interval is also inside the margin
        self.check("better_small", -1.0, (-1.9, -0.1), (-1.7, -0.3))
        # equivalence beats worse: entirely above 0 but inside the margin
        self.check("no_meaningful_benefit", 1.0, (0.1, 2.5), (0.3, 2.2))
        # 95% sticks out of the margin, 90% does not
        self.check("no_meaningful_benefit", 0.0, (-3.4, 3.4), (-2.9, 2.9))

    def test_boundaries(self):
        self.check("better_meaningful", -3.0, (-5.9, -0.1), (-5.5, -0.5))       # exactly -3 is meaningful
        self.check("better_meaningful", 100 * (-21 / 700), (-5.9, -0.1), (-5.5, -0.5))
        self.check("better_small", -2.9999, (-5.9, -0.1), (-5.5, -0.5))
        self.check("inconclusive", -3.5, (-7, 0.0), (-6, -0.2))                 # an upper bound of exactly 0 is not below 0
        self.check("inconclusive", 3.5, (0.0, 7), (0.2, 6))                     # a lower bound of exactly 0 is not above 0
        self.check("inconclusive", 0.5, (-3.5, 3.5), (-3.0, 2.0))               # the margin is open: exactly -3 is not inside
        self.check("inconclusive", 0.5, (-3.5, 3.5), (-2.0, 3.0))

    def test_not_a_number_is_an_error(self):
        with self.assertRaises(ValueError):
            decide(True, float("nan"), (-1, 1), (-1, 1))


class KappaTests(unittest.TestCase):
    def test_matches_the_existing_kappa(self):
        rng = np.random.default_rng(0)
        for _ in range(20):
            x, y = rng.random(40) < 0.6, rng.random(40) < 0.4
            k = kappa_from_counts((x & y).sum(), (x & ~y).sum(), (~x & y).sum(), (~x & ~y).sum())
            self.assertAlmostEqual(float(k), cohen_kappa(x, y))

    def test_undefined(self):
        self.assertTrue(np.isnan(kappa_from_counts(10, 0, 0, 0)))
        self.assertTrue(np.isnan(kappa_from_counts(0, 0, 0, 0)))


if __name__ == "__main__":
    unittest.main()
