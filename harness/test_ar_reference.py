"""Offline tests: per-model attention settings and the AR reference comparison (no models)."""
import copy
import json
import unittest

from .ar_reference import balanced, compare_step
from .run import attention_summary, validate_config


def config(**attention):
    base = json.load(open("configs/ar_pair_sanity.json"))
    for name, value in attention.items():
        base["models"][name]["attn_implementation"] = value
    return base


class AttentionConfigTests(unittest.TestCase):
    def test_default_is_sdpa_and_eager_is_allowed_per_model(self):
        validate_config(config())
        self.assertEqual(attention_summary(config()), "sdpa")
        mixed = config(B="eager")
        validate_config(mixed)
        self.assertEqual(attention_summary(mixed), "per_model")

    def test_unknown_attention_is_refused(self):
        with self.assertRaisesRegex(ValueError, "attn_implementation"):
            validate_config(config(A="flash_attention_2"))

    def test_new_configs_are_valid(self):
        for path in ("configs/ar_sanity_llama_gemma.json", "configs/ar_sanity_small.json"):
            c = json.load(open(path))
            validate_config(c)
            self.assertEqual(c["input_format"], "sanity")
        gemma = json.load(open("configs/ar_sanity_llama_gemma.json"))["models"]["B"]
        self.assertEqual(gemma["attn_implementation"], "eager")


class ReferenceTests(unittest.TestCase):
    def test_same_logits_pass_and_different_top_token_fails(self):
        import torch
        sets = {"yes": {"ids": [1]}, "no": {"ids": [2]}}
        logits = torch.tensor([0.0, 3.0, 1.0, -1.0])
        self.assertTrue(compare_step(logits, logits.clone(), sets)["passed"])
        moved = logits.clone()
        moved[3] = 5.0  # a different top token, same Yes/No logits
        result = compare_step(logits, moved, sets)
        self.assertFalse(result["passed"])
        self.assertLess(result["score_delta"], 1e-6)

    def test_balanced_takes_both_domains(self):
        records = [{"id": f"c{i}", "domain": "code"} for i in range(24)] + [{"id": f"m{i}", "domain": "math"} for i in range(26)]
        chosen = balanced(records, 20)
        self.assertEqual(sum(r["domain"] == "math" for r in chosen), 10)
        self.assertEqual(len(chosen), 20)


if __name__ == "__main__":
    unittest.main()


class SmallHelpersTests(unittest.TestCase):
    def test_single_bos_and_spearman(self):
        from types import SimpleNamespace
        from .ar_reference import single_bos, spearman
        tok = SimpleNamespace(bos_token_id=1)
        self.assertTrue(single_bos([1, 5, 6], tok))
        self.assertFalse(single_bos([1, 1, 5], tok))
        self.assertTrue(single_bos([1, 1, 5], SimpleNamespace(bos_token_id=None)))
        self.assertAlmostEqual(spearman([1, 2, 3], [10, 20, 30]), 1.0)
        self.assertAlmostEqual(spearman([1, 2, 3], [3, 2, 1]), -1.0)
