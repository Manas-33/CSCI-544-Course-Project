"""Every row of PLAN.md §7's go/no-go table as settings, and the pairs each one gives. CPU only."""
import json
import unittest
from pathlib import Path

from analysis.final_settings import (RULES, Member, SettingsError, final_test_members, make_pair, needed_scores,
                                     parse_settings, plan_pairs, primary_members, same_verifiers, with_role)

GO = json.loads(Path("configs/final_models.example.json").read_text(encoding="utf-8"))


def settings(**changes):
    return parse_settings({**GO, **changes})


def names(pairs):
    return sorted(p.name for p in pairs)


class RuleTests(unittest.TestCase):
    def test_fixed_numbers(self):
        self.assertEqual((RULES.pair_tar, RULES.tars), (0.90, (0.80, 0.90, 0.95)))
        self.assertEqual((RULES.margin_points, RULES.match_tar_points, RULES.match_gap_points), (3.0, 5.0, 3.0))
        self.assertEqual((RULES.bootstrap, RULES.seed, RULES.cost_ratio), (10_000, 544, 1.5))

    def test_settings_cannot_change_a_fixed_number(self):
        for key in ("margin_points", "bootstrap", "seed", "pair_tar"):
            with self.assertRaises(SettingsError):
                settings(**{key: 1})


class GoTests(unittest.TestCase):
    def test_example_is_the_plans_table(self):
        pairs = plan_pairs(settings(), "math")
        self.assertEqual(names(with_role(pairs, "h1_ar")), ["Gemma-2-9B + Llama-3.1-8B", "Gemma-2-9B + Qwen2.5-7B",
                                                           "Llama-3.1-8B + Qwen2.5-7B"])
        self.assertEqual(names(with_role(pairs, "h1_mixed")), ["Gemma-2-9B + LLaDA-8B", "LLaDA-8B + Llama-3.1-8B",
                                                              "LLaDA-8B + Qwen2.5-7B"])

    def test_dream_and_mistral_are_never_in_the_final_test(self):
        for p in plan_pairs(settings(), "math"):
            if {"h1_ar", "h1_mixed"} & set(p.roles):
                self.assertFalse({"Dream-7B", "Mistral-7B", "Llama-3.2-3B", "Qwen2.5-1.5B", "Qwen3-8B"}
                                 & {m.verifier for m in p.members})

    def test_trend_has_every_cross_family_ar_pair(self):
        trend = with_role(plan_pairs(settings(), "math"), "h2_trend")
        self.assertEqual(len(trend), 13)        # 15 pairs of 6 AR models, minus the two same-family pairs
        self.assertNotIn("Qwen2.5-1.5B + Qwen2.5-7B", names(trend))
        self.assertNotIn("Llama-3.1-8B + Llama-3.2-3B", names(trend))

    def test_e1_pairs(self):
        self.assertEqual(names(with_role(plan_pairs(settings(), "math"), "e1")),
                         ["Dream-7B + Qwen2.5-7B", "Gemma-2-9B + Qwen2.5-7B", "LLaDA-8B + Qwen2.5-7B",
                          "Llama-3.1-8B + Qwen2.5-7B", "Mistral-7B + Qwen2.5-7B"])

    def test_pairs_are_listed_once(self):
        all_names = names(plan_pairs(settings(), "math"))
        self.assertEqual(len(all_names), len(set(all_names)))

    def test_two_diffusion_models_are_never_paired(self):
        with self.assertRaises(ValueError):
            make_pair(Member("LLaDA-8B"), Member("Dream-7B"), ())


class OtherRowTests(unittest.TestCase):
    def test_repair_uses_the_reconstruction_score_on_math_only(self):
        s = settings(mode="repair", llada_score="reconstruction", reason="LLaDA fails test 4")
        self.assertEqual(s.test_name, "AR verdict versus diffusion reconstruction score")
        math = names(with_role(plan_pairs(s, "math"), "h1_mixed"))
        self.assertEqual(math, ["Gemma-2-9B + LLaDA-8B (reconstruction)", "LLaDA-8B (reconstruction) + Llama-3.1-8B",
                                "LLaDA-8B (reconstruction) + Qwen2.5-7B"])
        self.assertEqual(with_role(plan_pairs(s, "code"), "h1_mixed"), [])     # code keeps the verdict score, in no final test
        self.assertIn("Gemma-2-9B + LLaDA-8B", names(with_role(plan_pairs(s, "code"), "background")))
        self.assertIn(("LLaDA-8B", "recon"), needed_scores(s, "math"))
        self.assertNotIn(("LLaDA-8B", "recon"), needed_scores(s, "code"))
        self.assertIn("LLaDA-8B + Qwen2.5-7B", names(with_role(plan_pairs(s, "math"), "e1")))     # verdict score, like for like

    def test_pivot_is_the_six_ar_pairs_and_no_h1(self):
        s = settings(mode="pivot", llada_score=None, reason="reconstruction score failed", dropped={"LLaDA-8B": "fails test 4"})
        self.assertFalse(s.runs_h1)
        for domain in ("math", "code"):
            pairs = plan_pairs(s, domain)
            self.assertEqual(len(with_role(pairs, "ar_family")), 6)
            self.assertFalse(with_role(pairs, "h1_ar") or with_role(pairs, "h1_mixed"))

    def test_a_pivot_needs_a_cause(self):
        with self.assertRaises(SettingsError):      # nothing failed
            settings(mode="pivot", llada_score=None, reason="we prefer it")
        s = settings(mode="pivot", llada_score=None, reason="no cross-family pair left", h1_ar=["Qwen2.5-7B"],
                     dropped={"Gemma-2-9B": "x", "Llama-3.1-8B": "y"})
        self.assertFalse(s.runs_h1)

    def test_best_single_verifier_is_not_limited_to_the_final_test(self):
        names = {m.verifier for m in primary_members(settings())}
        self.assertTrue({"Mistral-7B", "Qwen2.5-1.5B", "Llama-3.2-3B", "Dream-7B", "LLaDA-8B"} <= names)
        repair = settings(mode="repair", llada_score="reconstruction", reason="LLaDA fails test 4")
        self.assertIn(Member("LLaDA-8B", "recon"), primary_members(repair))

    def test_the_same_verifiers_in_the_other_domain(self):
        repair = settings(mode="repair", llada_score="reconstruction", reason="LLaDA fails test 4")
        pair = with_role(plan_pairs(repair, "math"), "h1_mixed")[0]
        self.assertEqual(same_verifiers(repair, pair, "math").name, pair.name)
        self.assertEqual(same_verifiers(repair, pair, "code").name, pair.name.replace(" (reconstruction)", ""))

    def test_two_ar_verifiers_bring_in_qwen3_for_the_ar_pairs_only(self):
        s = settings(h1_ar=["Llama-3.1-8B", "Qwen2.5-7B"], dropped={"Gemma-2-9B": "fails test 4"}, qwen3_scored=True)
        pairs = plan_pairs(s, "math")
        self.assertEqual(names(with_role(pairs, "h1_ar")), ["Llama-3.1-8B + Qwen2.5-7B", "Llama-3.1-8B + Qwen3-8B"])
        self.assertEqual(names(with_role(pairs, "h1_mixed")), ["LLaDA-8B + Llama-3.1-8B", "LLaDA-8B + Qwen2.5-7B"])
        self.assertIn("Qwen3-8B", [m.verifier for m in final_test_members(s)])
        with self.assertRaises(SettingsError):
            settings(h1_ar=["Llama-3.1-8B", "Qwen2.5-7B"], dropped={"Gemma-2-9B": "fails test 4"})      # Qwen3 not scored

    def test_qwen3_that_failed_its_own_checks_does_not_join(self):
        failed = "fails the repeat-run check"
        s = settings(h1_ar=["Llama-3.1-8B", "Qwen2.5-7B"], dropped={"Gemma-2-9B": "x", "Qwen3-8B": failed})
        self.assertEqual(names(with_role(plan_pairs(s, "math"), "h1_ar")), ["Llama-3.1-8B + Qwen2.5-7B"])
        self.assertNotIn(("Qwen3-8B", "T1"), needed_scores(s, "math"))
        gone = {"Llama-3.1-8B": "x", "Qwen2.5-7B": "y", "Qwen3-8B": failed}
        with self.assertRaises(SettingsError):
            settings(h1_ar=["Gemma-2-9B"], dropped=gone)
        self.assertFalse(settings(mode="pivot", llada_score=None, reason="no pair", h1_ar=["Gemma-2-9B"], dropped=gone).runs_h1)
        with self.assertRaises(SettingsError):      # with all three left it plays no part
            settings(dropped={"Qwen3-8B": failed})
        with self.assertRaises(SettingsError):
            settings(h1_ar=["Llama-3.1-8B", "Qwen2.5-7B"], dropped={"Gemma-2-9B": "x", "Qwen3-8B": failed}, qwen3_scored=True)

    def test_one_ar_verifier(self):
        s = settings(h1_ar=["Llama-3.1-8B"], dropped={"Gemma-2-9B": "x", "Qwen2.5-7B": "y"}, qwen3_scored=True)
        self.assertEqual(names(with_role(plan_pairs(s, "math"), "h1_ar")), ["Llama-3.1-8B + Qwen3-8B"])

    def test_no_cross_family_pair_means_no_h1(self):
        with self.assertRaises(SettingsError):      # Qwen2.5 and Qwen3 are one family
            settings(h1_ar=["Qwen2.5-7B"], dropped={"Gemma-2-9B": "x", "Llama-3.1-8B": "y"}, qwen3_scored=True)

    def test_rows_combine(self):
        s = settings(mode="repair", llada_score="reconstruction", reason="r", h1_ar=["Gemma-2-9B", "Llama-3.1-8B"],
                     dropped={"Qwen2.5-7B": "fails gate A on corrected labels"}, qwen3_scored=True)
        self.assertEqual(len(with_role(plan_pairs(s, "math"), "h1_ar")), 3)
        self.assertEqual(len(with_role(plan_pairs(s, "math"), "h1_mixed")), 2)

    def test_the_pool_can_only_shrink(self):
        for bad in (["Mistral-7B"], ["Qwen3-8B", "Qwen2.5-7B"], ["Llama-3.1-8B", "Llama-3.1-8B"]):
            with self.assertRaises(SettingsError):
                settings(h1_ar=bad)

    def test_inconsistent_settings_are_refused(self):
        for changes in ({"mode": "go", "llada_score": "reconstruction"}, {"mode": "repair"}, {"mode": "maybe"}, {"record": ""},
                        {"dropped": {"Gemma-2-9B": "but it is still in h1_ar"}}, {"schema": 2},
                        {"mode": "pivot", "llada_score": None, "reason": ""}):
            with self.assertRaises(SettingsError):
                settings(**changes)


if __name__ == "__main__":
    unittest.main()
