"""The precision check and the matching simulation. CPU only; reads the committed pilot files."""
import unittest

import numpy as np

from analysis.final_pilot import pilot_inputs
from analysis.final_settings import RULES
from analysis.metrics import pair_thresholds
from analysis.precision import PairDraws, choose, design_effect, precision_rows, simulate_matching


class DesignEffectTests(unittest.TestCase):
    def test_one_answer_per_problem(self):
        v = np.array([0.0, 1, 0, 1, 1])
        self.assertAlmostEqual(design_effect(v, list("abcde")), 1.0)

    def test_fully_clustered_answers(self):
        # every problem has two identical answers: the variance of the mean doubles
        v = np.repeat([0.0, 1, 1, 0, 1], 2)
        self.assertAlmostEqual(design_effect(v, list("aabbccddee")), 2.0)

    def test_opposite_answers_cancel(self):
        self.assertAlmostEqual(design_effect(np.array([0.0, 1, 0, 1]), list("aabb")), 0.0)


class PilotTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.settings, _, _, cls.frames = pilot_inputs()

    def test_plan_numbers(self):
        """PLAN.md §6: disagreement 8-23%, design effect about 1.0-1.2, detectable about 3-5.5 points."""
        rows = precision_rows(self.frames["math"], self.settings)
        self.assertEqual(len(rows), 9)
        self.assertEqual({r["n_wrong_pilot"] for r in rows}, {125})
        disagree = [r["disagree"] for r in rows]
        self.assertTrue(0.075 <= min(disagree) and max(disagree) <= 0.235)
        self.assertTrue(all(0.9 <= r["design_effect"] <= 1.25 for r in rows))
        detectable = [r["detectable_points"] for r in rows]
        self.assertTrue(2.9 <= min(detectable) <= 3.2 and 5.2 <= max(detectable) <= 5.7)

    def test_draws_are_whole_problems(self):
        frame = self.frames["math"]
        ix = PairDraws(frame).draw(np.random.default_rng(0), 60)
        self.assertGreaterEqual(len(ix), 120)
        drawn = {}
        for i in ix:
            drawn.setdefault(frame.problems[i], []).append(i)
        full = {p: sum(q == p for q in frame.problems) for p in drawn}
        self.assertTrue(all(len(rows) % full[p] == 0 for p, rows in drawn.items()))

    def test_choose_is_the_selection_rule(self):
        frame = self.frames["math"]
        from analysis.final_validation import h1_groups
        groups = h1_groups(self.settings)
        scores = {m: frame.member(m) for items in groups.values() for p in items for m in p.members}
        pair, t = choose(scores, frame.labels, groups["ar_ar"], RULES)
        self.assertEqual(pair.name, "Gemma-2-9B + Qwen2.5-7B")      # the pilot's lowest in-sample FAR, 0.696
        self.assertEqual(t, pair_thresholds(scores[pair.a][frame.correct], scores[pair.b][frame.correct])[:2])

    def test_simulation_is_seeded_and_ordered(self):
        a = simulate_matching(self.frames["math"], self.settings, runs=60)
        b = simulate_matching(self.frames["math"], self.settings, runs=60)
        self.assertEqual(a, b)
        rate = {(r["off_target_points"], r["gap_points"]): r["passes"] for r in a["pass_rate"]}
        self.assertLessEqual(rate[(3, 3)], rate[(5, 3)])
        self.assertLessEqual(rate[(5, 3)], rate[(5, 5)])
        self.assertEqual(rate[(5, 3)], a["adopted_rule_passes"])


if __name__ == "__main__":
    unittest.main()
