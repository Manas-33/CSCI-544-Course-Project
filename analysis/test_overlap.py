"""Kappa, CAPA, the competence trend and E1 on cases worked by hand. CPU only."""
import unittest
from dataclasses import replace

import numpy as np

from analysis.final_data import Frame
from analysis.final_settings import RULES, Member, make_pair
from analysis.final_stats import Boot
from analysis.metrics import cohen_kappa
from analysis.overlap import (ABOVE, BELOW, INCONCLUSIVE, ON, capa, capa_boot, competence_trend, e1_rows, fit_lines, kappa_row,
                              overlap_row, status_of, trend_status, verdict_probabilities)


class CapaTests(unittest.TestCase):
    def test_two_items_by_hand(self):
        # correct option is 1 for both. p1(correct) = 0.9, 0.2; p2(correct) = 0.8, 0.4
        # observed = mean(0.9*0.8 + 0.1*0.2, 0.2*0.4 + 0.8*0.6) = mean(0.74, 0.56) = 0.65
        # expected = 0.55*0.6 + 0.45*0.4 = 0.51; CAPA = 0.14 / 0.49
        p1, p2 = [[0.1, 0.9], [0.8, 0.2]], [[0.2, 0.8], [0.6, 0.4]]
        self.assertAlmostEqual(capa(p1, p2, [1, 1]), 0.14 / 0.49)

    def test_four_options_by_hand(self):
        # one-hot answers, correct option 0. Model 1: right, wrong(1), wrong(1), right. Model 2: right, wrong(1), wrong(2), wrong(1)
        # observed = 2/4; q1 = 2/4, q2 = 1/4; expected = 1/8 + (1/2)(3/4)(1/3) = 1/4; CAPA = (1/2 - 1/4) / (3/4) = 1/3
        eye = np.eye(4)
        self.assertAlmostEqual(capa(eye[[0, 1, 1, 0]], eye[[0, 1, 2, 1]], [0, 0, 0, 0]), 1 / 3)

    def test_zero_one_verdicts_on_wrong_answers_equal_kappa_on_false_accepts(self):
        rng = np.random.default_rng(0)
        a, b = rng.random(60) < 0.7, rng.random(60) < 0.6
        self.assertAlmostEqual(capa(verdict_probabilities(a.astype(float)), verdict_probabilities(b.astype(float)), np.zeros(60, int)),
                               cohen_kappa(a, b))

    def test_identical_confident_models(self):
        p = verdict_probabilities(np.array([1.0, 0.0, 1.0, 0.0]))
        self.assertAlmostEqual(capa(p, p, [0, 0, 0, 0]), 1.0)

    def test_independent_models_are_near_zero(self):
        rng = np.random.default_rng(1)
        a, b = (rng.random(40000) < 0.5).astype(float), (rng.random(40000) < 0.5).astype(float)
        self.assertLess(abs(capa(verdict_probabilities(a), verdict_probabilities(b), np.zeros(40000, int))), 0.02)

    def test_undefined_and_bad_input(self):
        one = verdict_probabilities(np.ones(3))
        self.assertTrue(np.isnan(capa(one, one, [1, 1, 1])))        # both always right: chance agreement is 1
        with self.assertRaises(ValueError):
            capa([[1.0]], [[1.0]], [0])

    def test_bootstrap_replicates(self):
        rng = np.random.default_rng(2)
        ids = np.array(sorted(f"p{i:02d}" for i in rng.integers(0, 20, 60)))
        pa, pb, y = rng.random(60), rng.random(60), np.tile([0, 1], 30)
        boot = Boot(ids, n=30)
        point, reps = capa_boot(boot, pa, pb, y, y == 0)
        self.assertAlmostEqual(point, capa(verdict_probabilities(pa)[y == 0], verdict_probabilities(pb)[y == 0], y[y == 0]))
        for b in range(5):
            ix = np.repeat(np.arange(60), boot.weights[b][boot.group].astype(int))
            ix = ix[y[ix] == 0]
            self.assertAlmostEqual(reps[b], capa(verdict_probabilities(pa)[ix], verdict_probabilities(pb)[ix], y[ix]))


AR = ("Qwen2.5-7B", "Llama-3.1-8B", "Gemma-2-9B", "Mistral-7B", "Llama-3.2-3B", "Qwen2.5-1.5B")
SEP = dict(zip(AR, (1.2, 1.0, 0.9, 0.5, 0.3, 0.6)))


def study(diffusion_common, n=1600, seed=0, diffusion_separation=0.8):
    """AR verifiers share a common error; the diffusion verifier shares `diffusion_common` of it."""
    rng = np.random.default_rng(seed)
    ids = [f"p{i // 4:04d}" for i in range(n)]
    y = np.tile([0, 1], n // 2)
    common = rng.normal(size=n)
    scores = {(v, "T1"): SEP[v] * y + 0.7 * common + np.sqrt(1 - 0.49) * rng.normal(size=n) for v in AR}
    w = diffusion_common
    scores[("LLaDA-8B", "T1")] = diffusion_separation * y + w * common + np.sqrt(1 - w ** 2) * rng.normal(size=n)
    scores[("Dream-7B", "T1")] = 0.8 * y + 0.7 * common + np.sqrt(1 - 0.49) * rng.normal(size=n)
    p_yes = {k: 1 / (1 + np.exp(-v)) for k, v in scores.items()}
    return Frame("math", [{"source": "s"} for _ in ids], ids, scores, p_yes, p_yes, y)


AR_PAIRS = [make_pair(Member(a), Member(b), ("h2_trend",)) for i, a in enumerate(AR) for b in AR[i + 1:]]
MIXED = make_pair(Member("LLaDA-8B"), Member("Qwen2.5-7B"), ("h1_mixed", "e1"))
DREAM = make_pair(Member("Dream-7B"), Member("Qwen2.5-7B"), ("e1",))


def thresholds(pairs):
    return {p.name: (0.0, 0.0) for p in pairs}


class TrendTests(unittest.TestCase):
    def status(self, frame, n=400, margin=0.15):
        pairs = AR_PAIRS + [MIXED]
        return competence_trend(frame, AR_PAIRS, [MIXED], thresholds(pairs), Boot(frame.problems, n=n),
                                replace(RULES, h2_on_trend_kappa=margin))

    def test_a_decorrelated_verifier_is_below_the_trend(self):
        self.assertEqual(status_of(self.status(study(0.0)), MIXED.name), BELOW)

    def test_an_equally_correlated_verifier_is_on_the_trend(self):
        self.assertEqual(status_of(self.status(study(0.7)), MIXED.name), ON)

    def test_a_more_correlated_verifier_is_above_the_trend(self):
        frame = study(0.7)
        frame.scores[("LLaDA-8B", "T1")] = frame.scores[("Qwen2.5-7B", "T1")] + 0.05 * np.random.default_rng(1).normal(size=1600)
        self.assertEqual(status_of(self.status(frame), MIXED.name), ABOVE)

    def test_low_overlap_with_a_weak_verifier_alone_is_not_evidence(self):
        """A verifier with little skill and the AR models' shared error is read against the line at its own competence."""
        trend = self.status(study(0.7, diffusion_separation=0.1))
        self.assertEqual(status_of(trend, MIXED.name), ON)

    def test_an_interval_that_only_contains_zero_is_inconclusive(self):
        """ "On the trend" is a statement that the distance is small, so it needs an interval inside the margin."""
        self.assertEqual(status_of(self.status(study(0.7), margin=0.001), MIXED.name), INCONCLUSIVE)
        small = study(0.7, n=120)
        self.assertEqual(status_of(self.status(small), MIXED.name), INCONCLUSIVE)

    def test_status_boundaries(self):
        x = np.linspace(-1, 1, 2001)
        for shift, scale, expected in ((-0.3, 0.2, BELOW), (0.3, 0.2, ABOVE), (0.0, 0.1, ON), (0.0, 0.2, INCONCLUSIVE),
                                       (-0.02, 0.03, ON)):
            self.assertEqual(trend_status(shift, shift + scale * x, 0.10), expected, (shift, scale))
        self.assertEqual(trend_status(0.0, np.r_[0.01 * x, np.full(400, np.nan)], 0.10), INCONCLUSIVE)
        self.assertEqual(trend_status(float("nan"), x, 0.10), INCONCLUSIVE)

    def test_within_an_error_type(self):
        frame = study(0.0)
        mask = np.arange(1600) % 4 == 0      # kappa on some of the wrong answers; AUROC still on all answers
        pairs = AR_PAIRS + [MIXED]
        boot = Boot(frame.problems, n=50)
        part = competence_trend(frame, AR_PAIRS, [MIXED], thresholds(pairs), boot, mask=mask)
        whole = competence_trend(frame, AR_PAIRS, [MIXED], thresholds(pairs), boot)
        self.assertEqual(part["n_wrong"], int((frame.wrong & mask).sum()))
        self.assertEqual(part["auroc"], whole["auroc"])
        w = frame.wrong & mask
        row = next(r for r in part["pairs"] if r["pair"] == MIXED.name)
        self.assertAlmostEqual(row["kappa"], cohen_kappa(frame.member(MIXED.a)[w] >= 0, frame.member(MIXED.b)[w] >= 0))

    def test_undefined_kappa_is_inconclusive(self):
        frame = study(0.0)
        frame.scores[("LLaDA-8B", "T1")] = np.full(1600, 9.0)        # accepts everything: kappa is undefined
        trend = competence_trend(frame, AR_PAIRS, [MIXED], {**thresholds(AR_PAIRS), MIXED.name: (0.0, -9.0)},
                                 Boot(frame.problems, n=50))
        self.assertEqual(status_of(trend, MIXED.name), INCONCLUSIVE)

    def test_too_few_ar_pairs(self):
        frame = study(0.0)
        trend = competence_trend(frame, AR_PAIRS[:2], [MIXED], thresholds(AR_PAIRS + [MIXED]), Boot(frame.problems, n=50))
        self.assertEqual(status_of(trend, MIXED.name), INCONCLUSIVE)
        self.assertEqual(status_of(trend, "no such pair"), INCONCLUSIVE)

    def test_fit_lines(self):
        x = np.array([[0.5, 0.6, 0.7, 0.8], [0.5, 0.6, 0.7, 0.8], [0.5, 0.5, 0.5, 0.5]])
        y = np.array([[1.0, 1.2, 1.4, 1.6], [1.0, np.nan, 1.4, 1.6], [1.0, 2.0, 3.0, 4.0]])
        slope, intercept = fit_lines(x, y)
        np.testing.assert_allclose(slope[:2], 2.0)
        np.testing.assert_allclose(intercept[:2], 0.0, atol=1e-12)
        self.assertTrue(np.isnan(slope[2]))     # all x equal


class RowTests(unittest.TestCase):
    def test_overlap_row_counts(self):
        frame = study(0.3)
        row = overlap_row(frame, MIXED, (0.0, 0.0), Boot(frame.problems, n=50))
        w = frame.wrong
        fa, fb = frame.member(MIXED.a)[w] >= 0, frame.member(MIXED.b)[w] >= 0
        self.assertEqual((row["both_accept"], row["only_a"], row["only_b"], row["neither"]),
                         ((fa & fb).sum(), (fa & ~fb).sum(), (~fa & fb).sum(), (~fa & ~fb).sum()))
        self.assertAlmostEqual(row["kappa"], cohen_kappa(fa, fb))
        self.assertLess(row["kappa_ci"][0], row["kappa"])
        self.assertGreater(row["kappa_ci"][1], row["kappa"])

    def test_a_reconstruction_score_has_no_capa(self):
        frame = study(0.3)
        frame.scores[("LLaDA-8B", "recon")] = frame.scores[("LLaDA-8B", "T1")] - 3
        pair = make_pair(Member("LLaDA-8B", "recon"), Member("Qwen2.5-7B"), ("h1_mixed",))
        row = overlap_row(frame, pair, (-3.0, 0.0), Boot(frame.problems, n=20))
        self.assertIsNone(row["capa_wrong"])
        self.assertTrue(np.isfinite(row["kappa"]))

    def test_kappa_by_group(self):
        frame = study(0.3)
        mask = np.arange(1600) < 800
        k, (lo, hi) = kappa_row(frame, MIXED, (0.0, 0.0), Boot(frame.problems, n=100), mask)
        w = frame.wrong & mask
        self.assertAlmostEqual(k, cohen_kappa(frame.member(MIXED.a)[w] >= 0, frame.member(MIXED.b)[w] >= 0))
        self.assertTrue(lo < k < hi)

    def test_e1_differences_are_against_qwen_plus_dream(self):
        frame = study(0.0)
        rows = {r["pair"]: r for r in e1_rows(frame, [MIXED, DREAM], DREAM, thresholds([MIXED, DREAM]), Boot(frame.problems, n=200))}
        self.assertEqual(rows[DREAM.name]["far_minus_dream"], 0.0)
        self.assertEqual(rows[DREAM.name]["kappa_minus_dream_ci"], (0.0, 0.0))
        self.assertLess(rows[MIXED.name]["kappa_minus_dream_ci"][1], 0)     # planted: LLaDA is decorrelated, Dream is not
        self.assertAlmostEqual(rows[MIXED.name]["far_minus_dream"], rows[MIXED.name]["far"] - rows[DREAM.name]["far"])


if __name__ == "__main__":
    unittest.main()
