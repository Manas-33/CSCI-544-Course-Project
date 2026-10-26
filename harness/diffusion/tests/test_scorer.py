"""CPU unit tests for the diffusion scorers: logit shift, mask-slot indexing,
Yes/No token sets and padding, with tiny mocks instead of real models.

    python -m pytest harness/diffusion/tests -q
"""
import math
import sys
from types import SimpleNamespace

import pytest
import torch

from harness.diffusion.scorer import (
    SPECS, DreamScorer, ILLaDAScorer, LLaDAScorer, PromptTooLong, answer_slot_index,
    build_batch, check_environment, verdict_from_logits, verdict_tokens,
)

from .mocks import BOS, MASK, NO, PAD, VOCAB, YES, MockTokenizer, SlotMarkerModel, TinyBidirectional

SCORER_FOR = {"llada": LLaDAScorer, "illada": ILLaDAScorer, "dream": DreamScorer}


def make(key, model, n_slots=8, **spec_changes):
    spec = SPECS[key]
    spec = type(spec)(**{**spec.__dict__, "mask_id": MASK, **spec_changes})
    return SCORER_FOR[key](model, MockTokenizer(), spec, n_slots, device="cpu")


PROMPTS = [[BOS, 41, 42, 43], [BOS, 44], [BOS, 45, 46, 47, 48, 49, 50], [BOS, 51, 52]]


# --- the specs match the official repos -------------------------------------

def test_specs_match_official_values():
    assert (SPECS["llada"].mask_id, SPECS["llada"].logit_shift) == (126336, 0)
    assert (SPECS["illada"].mask_id, SPECS["illada"].logit_shift) == (5, 0)
    assert (SPECS["dream"].mask_id, SPECS["dream"].logit_shift) == (151666, 1)
    assert [SPECS[k].mask_token for k in ("llada", "illada", "dream")] == [
        "<|mdm_mask|>", "<[MASK]>", "<|mask|>"]
    assert SPECS["illada"].batched is False
    assert SPECS["llada"].transformers == "4.38.2"
    assert SPECS["illada"].transformers == "4.57.1"
    assert (SPECS["dream"].transformers, SPECS["dream"].torch) == ("4.46.2", "2.5.1")
    assert all(len(s.revision) == 40 for s in SPECS.values())


# --- Yes/No token sets ------------------------------------------------------

def test_verdict_tokens_keep_single_token_spellings_only():
    t = verdict_tokens(MockTokenizer())
    assert t.yes_ids == (YES, YES + 1, YES + 2, YES + 3)  # "yes"/" yes" share one id
    assert t.no_ids == (NO, NO + 1, NO + 2, NO + 3)
    assert set(t.dropped) == {"YES", "NO"}  # two tokens each
    assert "YES" not in t.kept and " YES" in t.kept


def test_verdict_tokens_reject_overlap_and_empty_sets():
    tok = MockTokenizer()
    tok.pieces["no"] = [YES]
    with pytest.raises(ValueError, match="share"):
        verdict_tokens(tok)
    with pytest.raises(ValueError, match="empty"):
        verdict_tokens(MockTokenizer(), yes=("YES",))


def test_verdict_tokens_encode_without_special_tokens():
    tok = MockTokenizer()
    verdict_tokens(tok)
    assert tok.encode_calls and not any(tok.encode_calls)


def test_alternative_answer_words():
    tok = MockTokenizer({"Correct": [11], "Incorrect": [21], " Correct": [12], " Incorrect": [22, 23]})
    t = verdict_tokens(tok, yes=("Correct", " Correct"), no=("Incorrect", " Incorrect"))
    assert (t.yes_ids, t.no_ids, t.dropped) == ((11, 12), (21,), (" Incorrect",))


# --- reading the verdict ----------------------------------------------------

def test_verdict_from_logits_sums_spellings():
    probs = torch.full((VOCAB,), 0.0)
    probs[YES], probs[YES + 1] = 0.4, 0.2      # P(Yes) = 0.6
    probs[NO], probs[NO + 2] = 0.1, 0.1         # P(No) = 0.2
    probs[40] = 0.2                             # other
    tokens = verdict_tokens(MockTokenizer())
    v = verdict_from_logits(torch.log(probs + 1e-12) + 7.0, tokens)  # softmax is shift-invariant
    assert v.p_yes == pytest.approx(0.6, abs=1e-6)
    assert v.p_no == pytest.approx(0.2, abs=1e-6)
    assert v.mass == pytest.approx(0.8, abs=1e-6)
    assert v.score == pytest.approx(math.log(3), abs=1e-5)
    assert v.top_id == YES


def test_verdict_from_bf16_logits_is_computed_in_float32():
    logits = torch.randn(VOCAB).to(torch.bfloat16)
    tokens = verdict_tokens(MockTokenizer())
    v = verdict_from_logits(logits, tokens)
    ref = torch.log_softmax(logits.float(), -1)
    expected = (torch.logsumexp(ref[list(tokens.yes_ids)], 0)
                - torch.logsumexp(ref[list(tokens.no_ids)], 0)).item()
    assert v.score == pytest.approx(expected, abs=1e-6)


# --- mask-slot indexing and Dream's shift -----------------------------------

def test_answer_slot_index():
    assert answer_slot_index(17, 0) == 17  # LLaDA / iLLaDA: the slot itself
    assert answer_slot_index(17, 1) == 16  # Dream: one position earlier


def test_dream_read_position_equals_official_shifted_logits():
    raw = torch.randn(3, 12, VOCAB)
    official = torch.cat([raw[:, :1], raw[:, :-1]], dim=1)  # generation_utils.py _sample
    prompt_len = 7
    assert torch.equal(official[:, prompt_len], raw[:, answer_slot_index(prompt_len, 1)])


def test_build_batch_left_pads_and_appends_slots():
    ids, att, prompt_len = build_batch([[BOS, 41, 42], [BOS]], n_slots=4, mask_id=MASK, pad_id=PAD)
    assert prompt_len == 3
    assert ids.tolist() == [[BOS, 41, 42] + [MASK] * 4, [PAD, PAD, BOS] + [MASK] * 4]
    assert att.tolist() == [[1] * 7, [0, 0] + [1] * 5]
    assert (ids[:, prompt_len] == MASK).all() and (ids[:, :prompt_len] != MASK).all()


@pytest.mark.parametrize("prompts, n_slots, pad, message", [
    ([[BOS, MASK]], 1, PAD, "already contains"),
    ([[BOS, 41], [BOS]], 1, MASK, "pad id"),
    ([[BOS]], 0, PAD, "at least one"),
    ([[]], 1, PAD, "empty"),
])
def test_build_batch_rejects_bad_input(prompts, n_slots, pad, message):
    with pytest.raises(ValueError, match=message):
        build_batch(prompts, n_slots, MASK, pad)


@pytest.mark.parametrize("key", ["llada", "illada", "dream"])
@pytest.mark.parametrize("n_slots", [1, 8, 32])
def test_scorer_reads_the_first_masked_slot(key, n_slots):
    model = SlotMarkerModel(shift=SPECS[key].logit_shift)
    verdicts = make(key, model, n_slots).score_ids(PROMPTS)
    assert len(verdicts) == len(PROMPTS)
    assert all(v.score > 3 and v.top_id == YES for v in verdicts)


def test_a_missing_shift_is_caught():
    # Dream-style raw logits read without the shift land on a NO position.
    model = SlotMarkerModel(shift=1)
    verdicts = make("dream", model, logit_shift=0).score_ids(PROMPTS)
    assert all(v.score < -3 and v.top_id == NO for v in verdicts)


def test_illada_is_scored_unbatched_without_attention_mask():
    model = SlotMarkerModel()
    make("illada", model).score_ids(PROMPTS)
    assert len(model.calls) == len(PROMPTS)
    for call, prompt in zip(model.calls, PROMPTS):
        assert call.input_ids.shape == (1, len(prompt) + 8)
        assert call.attention_mask is None


def test_llada_one_forward_pass_per_batch():
    model = SlotMarkerModel()
    make("llada", model).score_ids(PROMPTS)
    assert len(model.calls) == 1
    assert model.calls[0].attention_mask.shape == (len(PROMPTS), 7 + 8)


def test_dream_forward_matches_official_mask_and_positions():
    model = SlotMarkerModel(shift=1)
    scorer = make("dream", model, n_slots=2)
    scorer.score_ids([[BOS, 41, 42]])  # no padding: "full" attention, no position ids
    assert model.calls[-1].attention_mask == "full" and model.calls[-1].position_ids is None
    scorer.score_ids([[BOS, 41, 42], [BOS]])  # padding: 4D bool mask and tok_idx
    call = model.calls[-1]
    assert call.attention_mask.shape == (2, 1, 5, 5) and call.attention_mask.dtype == torch.bool
    assert call.position_ids.tolist() == [[0, 1, 2, 3, 4], [1, 1, 0, 1, 2]]
    assert not call.attention_mask[1, 0, :, :2].any()  # nobody attends to padding


def test_prompt_too_long():
    scorer = make("dream", SlotMarkerModel(shift=1), n_slots=8, max_len=10)
    with pytest.raises(PromptTooLong):
        scorer.score_ids([[BOS] + [41] * 2, [BOS] + [41] * 5])


# --- padding does not leak (tiny real attention models) ---------------------

def _batched_vs_single(scorer):
    single = [scorer.score_ids([p])[0].score for p in PROMPTS]
    batched = [v.score for v in scorer.score_ids(PROMPTS)]
    return max(abs(a - b) for a, b in zip(single, batched))


def test_llada_left_padding_matches_unbatched():
    assert _batched_vs_single(make("llada", TinyBidirectional("rope"))) < 1e-4


def test_dream_left_padding_matches_unbatched():
    assert _batched_vs_single(make("dream", TinyBidirectional("absolute"))) < 1e-4


def test_padding_leak_is_caught():
    class NoMaskLLaDA(LLaDAScorer):  # forgets the attention mask
        def forward(self, input_ids, attention_mask):
            return self.model(input_ids).logits

    spec = type(SPECS["llada"])(**{**SPECS["llada"].__dict__, "mask_id": MASK})
    leaky = NoMaskLLaDA(TinyBidirectional("rope"), MockTokenizer(), spec, 8, device="cpu")
    assert _batched_vs_single(leaky) > 1e-3


# --- prompts and environments ------------------------------------------------

def test_prompt_uses_chat_template_without_extra_special_tokens():
    scorer = make("llada", SlotMarkerModel())
    scorer.tokenizer.encode_calls.clear()
    ids = scorer.prompt_ids("Is it right")
    assert scorer.render("Is it right").endswith("<bos>assistant\n")
    assert ids[0] == BOS and ids.count(BOS) == 1
    assert scorer.tokenizer.encode_calls == [False]


def test_check_environment(monkeypatch):
    monkeypatch.setitem(sys.modules, "transformers", SimpleNamespace(__version__="4.38.2"))
    check_environment(SPECS["llada"])
    with pytest.raises(RuntimeError, match="own environment"):
        check_environment(SPECS["illada"])
    monkeypatch.setattr(torch, "__version__", "2.6.0+cu124")
    monkeypatch.setitem(sys.modules, "transformers", SimpleNamespace(__version__="4.46.2"))
    with pytest.raises(RuntimeError, match="torch 2.6.0"):
        check_environment(SPECS["dream"])
    monkeypatch.setattr(torch, "__version__", "2.5.1+cu124")
    check_environment(SPECS["dream"])
