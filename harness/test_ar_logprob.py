"""Offline tests for the AR exact log-probability arm, on tiny random-weight causal models."""
import json
import math
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import torch

from harness import ar_logprob
from harness.ar_logprob import answer_logprobs, score_items
from harness.diffusion.reconstruction import encode_pair
from harness.diffusion.run_reconstruction import ROW_FIELDS, previous_rows
from harness.diffusion.tests.mocks import VOCAB, MockTokenizer
from harness.run import attention_of, validate_config

PROMPT, ANSWER = (3, 41, 42, 43), (50, 51, 50, 52)


class TinyCausal(torch.nn.Module):
    """Random-weight model whose logits at position i depend on tokens 0..i only."""

    def __init__(self):
        super().__init__()
        torch.manual_seed(0)
        self.emb = torch.nn.Embedding(VOCAB, 32)
        self.hidden = torch.nn.Linear(32, 32)
        self.head = torch.nn.Linear(32, VOCAB)
        self.calls = []

    def forward(self, input_ids, attention_mask=None, use_cache=None):
        self.calls.append((input_ids.clone(), attention_mask, use_cache))
        x = self.emb(input_ids)
        steps = torch.arange(1, input_ids.shape[1] + 1)[None, :, None]
        x = torch.tanh(self.hidden(x.cumsum(dim=1) / steps) + x)
        return SimpleNamespace(logits=self.head(x))


def manual_sum(model, prompt, answer):
    """Token by token: one forward pass per prefix, softmax written out."""
    out = []
    for i, token in enumerate(answer):
        with torch.no_grad():
            row = model(torch.tensor([list(prompt) + list(answer[:i])])).logits[0, -1].tolist()
        top = max(row)
        out.append(row[token] - top - math.log(sum(math.exp(x - top) for x in row)))
    return out


def settings(**changes):
    return {"bank_hash": "h", "model": "Qwen/Qwen2.5-7B-Instruct", "revision": "r", "method": "ar_exact",
            "n_samples": None, "seed": None, "mc_group": None, "micro_batch": 1, "attention": "sdpa", **changes}


class LogProbTests(unittest.TestCase):
    def test_matches_a_manual_sum_over_prefixes(self):
        model = TinyCausal()
        picked = answer_logprobs(model, PROMPT, ANSWER, "cpu")
        for a, b in zip(picked, manual_sum(model, PROMPT, ANSWER)):
            self.assertAlmostEqual(a, b, places=5)
        self.assertEqual(len(picked), len(ANSWER))

    def test_matches_a_manual_sum_on_a_real_causal_architecture(self):
        try:
            from transformers import LlamaConfig, LlamaForCausalLM
        except ImportError:
            self.skipTest("transformers not installed")
        torch.manual_seed(0)
        model = LlamaForCausalLM(LlamaConfig(
            vocab_size=VOCAB, hidden_size=32, intermediate_size=64, num_hidden_layers=2, num_attention_heads=4,
            num_key_value_heads=2, max_position_embeddings=64)).float().eval()
        for a, b in zip(answer_logprobs(model, PROMPT, ANSWER, "cpu"), manual_sum(model, PROMPT, ANSWER)):
            self.assertAlmostEqual(a, b, places=4)

    def test_long_answers_are_read_in_chunks_without_changing_a_value(self):
        model, answer = TinyCausal(), tuple(40 + i % 20 for i in range(2 * ar_logprob.CHUNK + 5))
        picked = answer_logprobs(model, PROMPT, answer, "cpu")
        with torch.no_grad():
            logits = model(torch.tensor([list(PROMPT + answer)])).logits[0].float()
        n = len(PROMPT)
        expected = torch.log_softmax(logits, -1)[range(n - 1, n - 1 + len(answer)), list(answer)]
        self.assertEqual(len(picked), len(answer))
        for a, b in zip(picked, expected.tolist()):
            self.assertAlmostEqual(a, b, places=5)

    def test_one_unpadded_pass_without_cache_and_nothing_after_the_answer(self):
        model = TinyCausal()
        answer_logprobs(model, PROMPT, ANSWER, "cpu")
        (ids, mask, use_cache), = model.calls
        self.assertEqual(ids.tolist(), [list(PROMPT + ANSWER)])  # no end-of-turn token is shown or scored
        self.assertTrue(bool((mask == 1).all()))
        self.assertIs(use_cache, False)

    def test_bf16_logits_are_read_in_float32(self):
        class Half(TinyCausal):
            def forward(self, *args, **kwargs):
                out = super().forward(*args, **kwargs)
                return SimpleNamespace(logits=out.logits.to(torch.bfloat16))

        model = Half()
        with torch.no_grad():
            logits = model(torch.tensor([list(PROMPT + ANSWER)])).logits[0].float()
        expected = torch.log_softmax(logits, -1)[range(len(PROMPT) - 1, len(PROMPT + ANSWER) - 1), list(ANSWER)]
        for a, b in zip(answer_logprobs(model, PROMPT, ANSWER, "cpu"), expected.tolist()):
            self.assertAlmostEqual(a, b, places=6)

    def test_nonfinite_values_are_refused(self):
        class Broken(TinyCausal):
            def forward(self, *args, **kwargs):
                return SimpleNamespace(logits=super().forward(*args, **kwargs).logits * float("nan"))

        with self.assertRaises(ValueError):
            answer_logprobs(Broken(), PROMPT, ANSWER, "cpu")


class RowTests(unittest.TestCase):
    ITEMS = [{"candidate_id": "a", "problem": "add two numbers", "candidate": "it is four\n", "label": 1},
             {"candidate_id": "empty", "problem": "p", "candidate": "  \n"},
             {"candidate_id": "long", "problem": "p", "candidate": "four " * ar_logprob.MAX_TOKENS}]

    def test_rows_use_the_reconstruction_format(self):
        model, tokenizer = TinyCausal(), MockTokenizer()
        rows = list(score_items(model, tokenizer, self.ITEMS, settings(), "cpu"))
        self.assertEqual([r["status"] for r in rows], ["ok", "empty", "too_long"])
        for row in rows:
            self.assertEqual(tuple(row), ROW_FIELDS)
            self.assertEqual((row["method"], row["n_samples"], row["seed"]), ("ar_exact", None, None))
            json.dumps(row)
        ok = rows[0]
        pair = encode_pair(tokenizer, self.ITEMS[0]["problem"], self.ITEMS[0]["candidate"])
        expected = manual_sum(model, pair.prompt, pair.answer)
        self.assertAlmostEqual(ok["loglik"], sum(expected), places=4)
        self.assertAlmostEqual(ok["score"], sum(expected) / 3, places=4)
        self.assertEqual(ok["n_tokens"], 3)
        self.assertEqual(list(pair.prompt), tokenizer.encode(
            tokenizer.apply_chat_template([{"role": "user", "content": self.ITEMS[0]["problem"]}],
                                          add_generation_prompt=True, tokenize=False)))

    def test_a_failing_item_becomes_an_error_row(self):
        class Fails(TinyCausal):
            def forward(self, *args, **kwargs):
                raise RuntimeError("boom")

        rows = list(score_items(Fails(), MockTokenizer(), self.ITEMS[:1], settings(), "cpu"))
        self.assertEqual((rows[0]["status"], rows[0]["score"]), ("error", None))

    def test_resume_refuses_another_attention_or_revision(self):
        with tempfile.TemporaryDirectory() as folder:
            out = Path(folder) / "rows.jsonl"
            rows = list(score_items(TinyCausal(), MockTokenizer(), self.ITEMS, settings(), "cpu"))
            out.write_text("".join(json.dumps(r) + "\n" for r in rows))
            self.assertEqual(len(previous_rows(out, settings())), 3)
            for change in ({"attention": "eager"}, {"revision": "x"}, {"method": "diffusion_mc"}):
                with self.assertRaises(SystemExit):
                    previous_rows(out, settings(**change))


class ConfigTests(unittest.TestCase):
    def test_the_four_ar_models_come_from_the_verdict_configs(self):
        found = {}
        for name in ("pilot_qwen_mistral", "pilot_llama_gemma"):
            config = json.loads(Path(f"configs/{name}.json").read_text(encoding="utf-8"))
            validate_config(config)
            self.assertEqual(config["dtype"], "bf16")
            found.update({spec["model_id"]: attention_of(spec) for spec in config["models"].values()})
        self.assertEqual(found, {"Qwen/Qwen2.5-7B-Instruct": "sdpa", "mistralai/Mistral-7B-Instruct-v0.3": "sdpa",
                                 "meta-llama/Llama-3.1-8B-Instruct": "sdpa", "google/gemma-2-9b-it": "eager"})


if __name__ == "__main__":
    unittest.main()
