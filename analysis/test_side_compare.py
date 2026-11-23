"""Side-experiment comparison on synthetic answers with known scores. CPU only."""
import unittest

import numpy as np

from analysis import side_compare as S


def make_rows(n=40):
    return [{"candidate_id": f"c{i}", "problem_id": f"p{i // 2}"} for i in range(n)]


def make_run(rows, scores):
    return {"settings": {}, "rows": {r["candidate_id"]: {"status": "ok", "score": float(s)} for r, s in zip(rows, scores)}}


class LikelihoodByModel(unittest.TestCase):
    def test_keeps_only_the_side_runs(self):
        rows = make_rows()
        runs = {("LLaDA-8B", "diffusion_mc", 128, 544): make_run(rows, range(40)),
                ("LLaDA-8B", "diffusion_mc", 128, 545): make_run(rows, [0] * 40),
                ("LLaDA-8B", "diffusion_mc", 32, 544): make_run(rows, [0] * 40),
                ("Qwen2.5-7B", "ar_exact", None, None): make_run(rows, range(40, 0, -1))}
        out = S.likelihood_by_model(runs, rows)
        self.assertEqual(sorted(out), ["LLaDA-8B", "Qwen2.5-7B"])
        np.testing.assert_array_equal(out["LLaDA-8B"], np.arange(40))
        np.testing.assert_array_equal(out["Qwen2.5-7B"], np.arange(40, 0, -1))

    def test_refuses_two_runs_of_one_model(self):
        rows = make_rows()
        runs = {("Qwen2.5-7B", "ar_exact", None, None): make_run(rows, range(40)),
                ("Qwen2.5-7B", "ar_exact", None, 1): make_run(rows, range(40))}
        with self.assertRaises(SystemExit):
            S.likelihood_by_model(runs, rows)


class Compare(unittest.TestCase):
    def test_known_aurocs(self):
        rows = make_rows()
        y = np.array([i % 2 for i in range(40)], dtype=float)
        perfect, reversed_ = y.copy(), 1 - y
        verdict = {name: perfect for name in S.MODELS}
        likelihood = {name: reversed_ for name in S.MODELS}
        out = S.compare(rows, y, verdict, likelihood, n_boot=200)
        for name in S.MODELS:
            m = out[name]
            self.assertAlmostEqual(m["verdict_auroc"], 1.0)
            self.assertAlmostEqual(m["likelihood_auroc"], 0.0)
            self.assertAlmostEqual(m["likelihood_minus_verdict"], -1.0)
            self.assertAlmostEqual(m["rank_correlation"], -1.0)
            lo, hi = m["likelihood_minus_verdict_ci"]
            self.assertAlmostEqual(lo, -1.0)
            self.assertAlmostEqual(hi, -1.0)

    def test_refuses_a_missing_score(self):
        rows = make_rows()
        y = np.array([i % 2 for i in range(40)], dtype=float)
        gap = y.copy()
        gap[3] = np.nan
        with self.assertRaises(SystemExit):
            S.compare(rows, y, {n: y for n in S.MODELS}, {n: gap for n in S.MODELS}, n_boot=10)


if __name__ == "__main__":
    unittest.main()
