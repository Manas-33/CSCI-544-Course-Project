"""CPU unit tests for the reconstruction score, on tiny models instead of real weights.

    python -m pytest harness/diffusion/tests/test_reconstruction.py -q
"""
import inspect
import itertools
import json
import math
from types import SimpleNamespace

import pytest
import torch
import torch.nn.functional as F

from harness.diffusion.checks import check_reconstruction
from harness.diffusion.checks.check_reconstruction import check_dream, check_item, check_llada
from harness.diffusion.interface import write_jsonl
from harness.diffusion.reconstruction import (
    MC_GROUP, N_SAMPLES, REPEAT_SEED, SAMPLE_COUNTS, SEED, EmptyAnswer, chat_texts, draw_masks, encode_pair,
    item_generator, mask_counts, partial_samples, reconstruct, sample_logliks,
)
from harness.diffusion.run_reconstruction import (
    ROW_FIELDS, SETTINGS, backup_row, gpu_type, load_items, main, previous_rows, score_items,
)
from harness.diffusion.scorer import SPECS, DreamScorer, LLaDAScorer

from .mocks import BOS, MASK, VOCAB, MockTokenizer, TinyBidirectional

SCORER_FOR = {"llada": LLaDAScorer, "dream": DreamScorer}
POSITIONS = {"llada": "rope", "dream": "absolute"}
PROMPT, ANSWER = (BOS, 41, 42, 43, 44), (50, 51, 50)


def make(key="llada", model=None, tokenizer=None, **spec_changes):
    spec = type(SPECS[key])(**{**SPECS[key].__dict__, "mask_id": MASK, **spec_changes})
    model = model if model is not None else TinyBidirectional(POSITIONS[key])
    return SCORER_FOR[key](model, tokenizer or MockTokenizer(), spec, 1, device="cpu")


def all_masks(n):
    """Every non-empty subset of n positions, as boolean rows."""
    return torch.tensor([m for m in itertools.product([False, True], repeat=n) if any(m)])


# --- a toy case, enumerated by hand -----------------------------------------

def by_hand(model, prompt, answer, mask, shift, attention_mask=None):
    """(L / k) * the masked tokens' log-probabilities, with plain loops."""
    ids = list(prompt) + [MASK if m else a for a, m in zip(answer, mask)]
    with torch.no_grad():
        logits = model(torch.tensor([ids]), attention_mask, None).logits[0]
    total = 0.0
    for i, m in enumerate(mask):
        if m:
            row = logits[len(prompt) + i - shift].tolist()
            top = max(row)
            log_z = top + math.log(sum(math.exp(x - top) for x in row))
            total += row[answer[i]] - log_z
    return total * len(answer) / sum(bool(m) for m in mask)


@pytest.mark.parametrize("key", ["llada", "dream"])
def test_every_mask_of_a_toy_answer_matches_the_hand_calculation(key):
    scorer = make(key)
    masks = all_masks(len(ANSWER))
    values = sample_logliks(scorer, PROMPT, ANSWER, masks)
    attention = "full" if key == "dream" else None
    expected = [by_hand(scorer.model, PROMPT, ANSWER, m.tolist(), scorer.spec.logit_shift, attention) for m in masks]
    assert values == pytest.approx(expected, abs=1e-4)


def test_the_estimate_converges_to_the_enumerated_expectation():
    # The target is the mean over k = 1..L of the mean over the subsets of size k.
    scorer, n = make(), len(ANSWER)
    masks = all_masks(n)
    value_of = {tuple(m.tolist()): v for m, v in zip(masks, sample_logliks(scorer, PROMPT, ANSWER, masks))}
    exact = sum(sum(v for m, v in value_of.items() if sum(m) == k) / math.comb(n, k) for k in range(1, n + 1)) / n
    drawn = draw_masks(n, 16 * 2000, torch.Generator().manual_seed(0))
    estimate = sum(value_of[tuple(m)] for m in drawn.tolist()) / len(drawn)
    assert estimate == pytest.approx(exact, rel=0.01)


def test_the_score_is_per_token():
    scorer = make()
    pair = SimpleNamespace(prompt=PROMPT, answer=ANSWER)
    r = reconstruct(scorer, pair, "c1", seed=3, n_samples=32)
    values = sample_logliks(scorer, PROMPT, ANSWER, draw_masks(3, 32, item_generator(3, "c1")))
    assert r.loglik == pytest.approx(sum(values) / 32) and r.n_tokens == 3
    assert r.score == pytest.approx(r.loglik / 3) and r.score < 0


def test_higher_means_more_plausible():
    class Knows(torch.nn.Module):  # predicts token 50 wherever it looks
        def forward(self, input_ids, attention_mask=None, position_ids=None):
            logits = torch.zeros(*input_ids.shape, VOCAB)
            logits[..., 50] = 5.0
            return SimpleNamespace(logits=logits)

    scorer = make(model=Knows())
    good = reconstruct(scorer, SimpleNamespace(prompt=PROMPT, answer=(50, 50)), "c", n_samples=16)
    bad = reconstruct(scorer, SimpleNamespace(prompt=PROMPT, answer=(51, 52)), "c", n_samples=16)
    assert good.score > bad.score


# --- parity with the official get_log_likelihood ----------------------------
# Re-implemented from github.com/ML-GSAI/LLaDA get_log_likelihood.py (commit 9182493), which is not
# available to the tests. checks/check_reconstruction.py compares with the real file on a GPU.

def official_forward_process(batch, prompt_index, mask_id):
    b, l = batch.shape
    target_len = (l - prompt_index.sum()).item()
    k = torch.randint(1, target_len + 1, ())
    x = torch.round(torch.linspace(float(k), float(k + (b - 1) * (target_len / b)), steps=b)).long()
    x = ((x - 1) % target_len) + 1
    assert x.min() >= 1 and x.max() <= target_len
    indices = torch.arange(target_len).repeat(b, 1)
    is_mask = indices < x.unsqueeze(1)
    for i in range(b):
        is_mask[i] = is_mask[i][torch.randperm(target_len)]
    is_mask = torch.cat((torch.zeros(b, int(prompt_index.sum()), dtype=torch.bool), is_mask), dim=1)
    noisy_batch = torch.where(is_mask, mask_id, batch)
    return noisy_batch, (x / target_len).unsqueeze(1).repeat(1, l)


@torch.no_grad()
def official_get_log_likelihood(model, prompt, answer, mc_num=128, batch_size=16, cfg_scale=0.0, mask_id=126336):
    seq = torch.concatenate([prompt, answer])[None, :].repeat((batch_size, 1))
    prompt_index = torch.arange(seq.shape[1]) < len(prompt)
    losses = []
    for _ in range(mc_num // batch_size):
        perturbed_seq, p_mask = official_forward_process(seq, prompt_index, mask_id)
        mask_index = perturbed_seq == mask_id
        logits = model(perturbed_seq).logits
        loss = F.cross_entropy(logits[mask_index], seq[mask_index], reduction="none") / p_mask[mask_index]
        losses.append((loss.sum() / batch_size).item())
    return -sum(losses) / len(losses)


@pytest.mark.parametrize("n_answer", [1, 2, 5, 16, 37])
@pytest.mark.parametrize("n_samples", [16, 128])
def test_same_random_numbers_give_the_official_estimate(n_answer, n_samples):
    scorer = make()
    answer = tuple(40 + i % 20 for i in range(n_answer))
    torch.manual_seed(7)
    official = official_get_log_likelihood(scorer.model, torch.tensor(PROMPT), torch.tensor(answer),
                                           mc_num=n_samples, mask_id=MASK)
    masks = draw_masks(n_answer, n_samples, torch.Generator().manual_seed(7))
    values = sample_logliks(scorer, PROMPT, answer, masks)
    assert sum(values) / len(values) == pytest.approx(official, rel=1e-5)


def test_eight_of_sixteen_samples_have_the_official_counts_for_a_group_of_eight():
    for n_answer in (3, 8, 37, 200):
        for k in (1, 2, n_answer):
            sixteen = mask_counts(k, n_answer, 16)
            assert sixteen[partial_samples(8, 128)].tolist() == mask_counts(k, n_answer, 8).tolist()


# --- Dream's shift ----------------------------------------------------------

class Remembers(torch.nn.Module):
    """Raw logits at position i strongly predict token i + shift of a fixed sequence."""

    def __init__(self, sequence, shift):
        super().__init__()
        self.sequence, self.shift = list(sequence), shift

    def forward(self, input_ids, attention_mask=None, position_ids=None):
        b, n = input_ids.shape
        logits = torch.zeros(b, n, VOCAB)
        for i in range(n):
            if 0 <= i + self.shift < n:
                logits[:, i, self.sequence[i + self.shift]] = 30.0
        return SimpleNamespace(logits=logits)


def test_dream_reads_one_position_earlier_and_dropping_the_shift_is_caught():
    answer = (50, 51, 52, 53)
    model = Remembers(PROMPT + answer, shift=1)
    masks = all_masks(4)
    shifted = sample_logliks(make("dream", model), PROMPT, answer, masks)
    assert max(abs(v) for v in shifted) < 1e-6  # every masked token predicted exactly
    dropped = sample_logliks(make("dream", model, logit_shift=0), PROMPT, answer, masks)
    assert max(dropped) < -10  # at best the last token, which the unshifted read predicts at chance
    assert SPECS["dream"].logit_shift == 1 and SPECS["llada"].logit_shift == 0


def test_dream_estimate_equals_the_official_shifted_logits():
    scorer = make("dream")
    mask = torch.tensor([True, False, True])
    noisy = torch.tensor([list(PROMPT) + [MASK, ANSWER[1], MASK]])
    with torch.no_grad():
        raw = scorer.model(noisy, "full", None).logits
    official = torch.cat([raw[:, :1], raw[:, :-1]], dim=1)  # generation_utils.py _sample
    log_probs = torch.log_softmax(official[0], -1)
    expected = sum(log_probs[len(PROMPT) + i, ANSWER[i]].item() for i in (0, 2)) * 3 / 2
    assert sample_logliks(scorer, PROMPT, ANSWER, mask[None])[0] == pytest.approx(expected, abs=1e-4)


# --- what is masked ---------------------------------------------------------

class Recorder(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.calls = []

    def forward(self, input_ids, attention_mask=None, position_ids=None):
        self.calls.append((input_ids.clone(), attention_mask, position_ids))
        return SimpleNamespace(logits=torch.zeros(*input_ids.shape, VOCAB))


@pytest.mark.parametrize("key", ["llada", "dream"])
@pytest.mark.parametrize("micro_batch", [1, 5])
def test_the_prompt_is_never_masked_and_answer_tokens_at_most_once(key, micro_batch):
    model = Recorder()
    scorer = make(key, model)
    answer = tuple(range(45, 56))
    masks = draw_masks(len(answer), 32, item_generator(544, "c"))
    sample_logliks(scorer, PROMPT, answer, masks, micro_batch)
    seen = torch.cat([ids for ids, _, _ in model.calls])
    assert len(seen) == 32 and all(len(ids) <= micro_batch for ids, _, _ in model.calls)
    assert (seen[:, :len(PROMPT)] == torch.tensor(PROMPT)).all()
    masked = seen[:, len(PROMPT):] == MASK
    assert torch.equal(masked, masks)
    assert (seen[:, len(PROMPT):][~masked] == torch.tensor(answer).repeat(32, 1)[~masked]).all()
    assert masked.any(dim=1).all() and seen.shape[1] == len(PROMPT) + len(answer)
    for _, attention_mask, position_ids in model.calls:
        assert position_ids is None  # same length, no padding: positions 0..L-1
        assert attention_mask == "full" if key == "dream" else bool((attention_mask == 1).all())


def test_mask_counts_follow_the_official_spread():
    for n_answer in (1, 4, 16, 100):
        masks = draw_masks(n_answer, 128, torch.Generator().manual_seed(1))
        counts = masks.sum(dim=1)
        assert counts.min() >= 1 and counts.max() <= n_answer
        for g in range(8):
            group = counts[g * 16:(g + 1) * 16]
            assert group.tolist() == mask_counts(int(group[0]), n_answer, 16).tolist()
    assert len(set(mask_counts(3, 100, 16).tolist())) == 16  # evenly spread, not repeated


def test_bad_masks_and_inputs_are_refused():
    scorer = make()
    with pytest.raises(ValueError, match="at least one"):
        sample_logliks(scorer, PROMPT, ANSWER, torch.zeros(1, 3, dtype=torch.bool))
    with pytest.raises(ValueError, match="mask token"):
        sample_logliks(scorer, PROMPT, (50, MASK), all_masks(2))
    with pytest.raises(ValueError, match="mask token"):
        sample_logliks(scorer, PROMPT + (MASK,), ANSWER, all_masks(3))
    with pytest.raises(ValueError, match="micro_batch"):
        sample_logliks(scorer, PROMPT, ANSWER, all_masks(3), micro_batch=0)
    with pytest.raises(ValueError, match="groups"):
        draw_masks(3, 24, torch.Generator())


# --- prompt and answer tokens -----------------------------------------------

class Template:
    """Greedy longest-match tokenizer over a few multi-character pieces, with a chat template."""

    pad_token_id = eos_token_id = 0
    all_special_ids = [0]

    def __init__(self, style="plain", pieces=()):
        self.style = style
        self.vocab = {p: 1000 + i for i, p in enumerate(
            sorted((*pieces, "<s>", "<eot>", "<user>", "<assistant>", "\n\n"), key=len, reverse=True))}

    def encode(self, text, add_special_tokens=True):
        assert add_special_tokens is False
        ids, i = [], 0
        while i < len(text):
            piece = next((p for p in self.vocab if text.startswith(p, i)), text[i])
            ids.append(self.vocab[piece] if piece in self.vocab else ord(piece))
            i += len(piece)
        return ids

    def decode(self, ids):
        names = {v: k for k, v in self.vocab.items()}
        return "".join(names.get(i) or chr(i) for i in ids)

    def apply_chat_template(self, messages, add_generation_prompt=False, tokenize=True):
        assert tokenize is False
        turn = {"plain": "\n\n", "colon": ":", "mistral": ""}.get(self.style, "\n\n")
        text = "<s>"
        for m in messages:
            content = m["content"].strip() if self.style == "llada" else m["content"]
            if self.style == "mistral":
                text += f"<user>{content}" if m["role"] == "user" else f" {content}<eot>"
                text += "<assistant>" if m["role"] == "user" else ""
                continue
            if self.style == "thinks" and m["role"] == "assistant":
                content = "<think>\n\n</think>" + content
            if self.style == "drops" and m["role"] == "assistant":
                content = content.replace("x", "")
            text += f"<{m['role']}>{turn}{content}<eot>"
        if self.style == "llada" or (add_generation_prompt and self.style != "mistral"):
            text += f"<assistant>{turn}"  # LLaDA's template appends this every time
        return text


@pytest.mark.parametrize("style", ["plain", "llada", "mistral"])
def test_the_answer_is_exactly_the_candidate(style):
    tok = Template(style)
    pair = encode_pair(tok, "Add 1 and 2", "  It is 3.\nSo 3\n")
    prompt_text = tok.apply_chat_template([{"role": "user", "content": "Add 1 and 2"}],
                                          add_generation_prompt=True, tokenize=False)
    assert pair.boundary == "clean"
    assert list(pair.prompt) == tok.encode(prompt_text, add_special_tokens=False)
    # no end-of-turn token and, for LLaDA's template, no next generation prompt
    lead = " " if style == "mistral" else ""
    assert tok.decode(pair.answer) == lead + "It is 3.\nSo 3"


def test_no_token_spans_the_boundary():
    tok = Template("colon", pieces=(":)", ") and"))
    prompt_text, answer_text = chat_texts(tok, "p", ") and more")
    assert tok.encode(prompt_text + answer_text, add_special_tokens=False)[:len(
        tok.encode(prompt_text, add_special_tokens=False))] != tok.encode(prompt_text, add_special_tokens=False)
    pair = encode_pair(tok, "p", ") and more")
    assert pair.boundary == "separate"
    assert list(pair.prompt) == tok.encode(prompt_text, add_special_tokens=False)
    assert list(pair.answer) == tok.encode(answer_text, add_special_tokens=False)
    assert tok.decode(pair.prompt + pair.answer) == prompt_text + answer_text
    assert encode_pair(tok, "p", "fine").boundary == "clean"


def test_templates_that_add_or_change_answer_text_are_refused():
    with pytest.raises(ValueError, match="before the answer"):
        encode_pair(Template("thinks"), "p", "answer")
    with pytest.raises(ValueError, match="changes the answer"):
        encode_pair(Template("drops"), "p", "xyz")


def test_empty_answers():
    for text in ("", "  \n "):
        with pytest.raises(EmptyAnswer):
            encode_pair(MockTokenizer(), "p", text)


def test_the_prompt_is_the_scorers_own_prompt():
    scorer = make()
    pair = encode_pair(scorer.tokenizer, "add two numbers", "it is four")
    assert list(pair.prompt) == scorer.prompt_ids("add two numbers")


# --- seeds, resume and rows -------------------------------------------------

ITEMS = [{"candidate_id": f"c{i}", "domain": "math", "problem": "add two numbers " * (i + 1),
          "candidate": "it is clearly four " * (i + 1), "label": 1} for i in range(4)]


def settings(**changes):
    return {"bank_hash": "h", "model": "llada", "revision": "r", "method": "diffusion_mc", "n_samples": 16,
            "seed": 544, "mc_group": 16, "micro_batch": 1, "attention": None, **changes}


def test_seeds_give_the_same_estimate_in_any_order():
    a = {r["candidate_id"]: r for r in score_items(make(), ITEMS, settings())}
    b = {r["candidate_id"]: r for r in score_items(make(), ITEMS[::-1], settings())}
    assert {c: r["score"] for c, r in a.items()} == {c: r["score"] for c, r in b.items()}
    other = {r["candidate_id"]: r for r in score_items(make(), ITEMS, settings(seed=545))}
    assert all(a[c]["score"] != other[c]["score"] for c in a)


def test_item_generators_differ_by_seed_and_item():
    drawn = {(s, c): tuple(draw_masks(20, 16, item_generator(s, c)).flatten().tolist())
             for s in (544, 545) for c in ("a", "b")}
    assert len(set(drawn.values())) == 4
    assert torch.equal(draw_masks(20, 16, item_generator(544, "a")), draw_masks(20, 16, item_generator(544, "a")))


def test_fewer_samples_are_a_prefix_of_more():
    many = draw_masks(20, 128, item_generator(544, "a"))
    assert torch.equal(draw_masks(20, 32, item_generator(544, "a")), many[:32])
    scorer, pair = make(), SimpleNamespace(prompt=PROMPT, answer=ANSWER)
    r = reconstruct(scorer, pair, "a", n_samples=128)
    assert r.loglik_at["32"] == pytest.approx(reconstruct(scorer, pair, "a", n_samples=32).loglik)
    assert r.loglik_at["128"] == r.loglik and set(r.loglik_at) == {str(n) for n in SAMPLE_COUNTS}
    assert partial_samples(8, 128) == list(range(0, 16, 2)) and partial_samples(64, 128) == list(range(64))
    with pytest.raises(ValueError):
        partial_samples(24, 128)


def test_rows_have_the_contract_fields():
    rows = list(score_items(make(), ITEMS, settings()))
    contract = {"bank_hash", "candidate_id", "model", "revision", "method", "n_samples", "seed", "n_tokens",
                "loglik", "score", "status", "runtime_s", "peak_mem_gb"}
    assert contract <= set(ROW_FIELDS)
    for row, item in zip(rows, ITEMS):
        assert tuple(row) == ROW_FIELDS and json.loads(json.dumps(row)) == row
        assert row["candidate_id"] == item["candidate_id"] and row["status"] == "ok"
        assert row["score"] == pytest.approx(row["loglik"] / row["n_tokens"])
        assert row["n_tokens"] == len(item["candidate"].split()) and row["boundary"] == "clean"
        assert row["runtime_s"] > 0 and row["peak_mem_gb"] is None and row["gpu"] is None  # CPU
    assert not any("label" in row for row in rows)


def test_too_long_empty_and_failing_items_get_their_own_rows():
    class Fails(TinyBidirectional):
        def forward(self, input_ids, attention_mask=None, position_ids=None):
            if 63 in input_ids:
                raise RuntimeError("boom")
            return super().forward(input_ids, attention_mask, position_ids)

    tokenizer = MockTokenizer()
    tokenizer.encode = (lambda plain: lambda text, add_special_tokens=True: [
        MASK if w == "MASK" else 63 if w == "BOOM" else i
        for i, w in zip(plain(text, add_special_tokens), text.replace("<bos>", "<bos> ", 1).split())])(tokenizer.encode)
    items = [{"candidate_id": "ok", "problem": "p", "candidate": "four"},
             {"candidate_id": "empty", "problem": "p", "candidate": " \n"},
             {"candidate_id": "long", "problem": "p", "candidate": "four " * 40},
             {"candidate_id": "mask", "problem": "p", "candidate": "has MASK inside"},
             {"candidate_id": "boom", "problem": "p", "candidate": "has BOOM inside"},
             {"candidate_id": "ok2", "problem": "p", "candidate": "four"}]
    rows = list(score_items(make(model=Fails(), tokenizer=tokenizer, max_len=30), items, settings()))
    assert [r["status"] for r in rows] == ["ok", "empty", "too_long", "error", "error", "ok"]
    assert all((r["score"] is None) == (r["status"] != "ok") for r in rows)
    assert rows[2]["n_tokens"] == 40


def test_row_status_rules():
    with pytest.raises(ValueError):
        backup_row(settings(), "c", "ok")
    with pytest.raises(ValueError):
        backup_row(settings(), "c", "error", loglik=-1.0, score=-1.0)
    with pytest.raises(ValueError):
        backup_row(settings(), "c", "weird")
    with pytest.raises(ValueError):
        backup_row({k: None for k in SETTINGS[1:]}, "c", "empty")


def test_resume_keeps_good_rows_and_retries_errors(tmp_path):
    out = tmp_path / "rows.jsonl"
    rows = list(score_items(make(), ITEMS, settings()))
    write_jsonl(out, rows[:2] + [backup_row(settings(), "c2", "error")])
    with open(out, "a") as f:
        f.write(json.dumps(rows[3])[:40])  # the runtime died while writing
    kept = previous_rows(out, settings())
    assert [r["candidate_id"] for r in kept] == ["c0", "c1"]
    assert out.read_text().count("\n") == 2 and out.read_text().endswith("\n")
    done = {r["candidate_id"] for r in kept}
    write_jsonl(out, score_items(make(), [it for it in ITEMS if it["candidate_id"] not in done], settings()), append=True)
    resumed = {r["candidate_id"]: r["score"] for r in previous_rows(out, settings())}
    assert resumed == {r["candidate_id"]: r["score"] for r in rows}


@pytest.mark.parametrize("change", [{"n_samples": 128}, {"seed": 545}, {"micro_batch": 8}, {"revision": "x"},
                                    {"bank_hash": "x"}, {"model": "dream"}, {"mc_group": 8}])
def test_a_file_with_other_settings_is_refused(tmp_path, change):
    out = tmp_path / "rows.jsonl"
    write_jsonl(out, [backup_row(settings(), "c0", "empty")])
    with pytest.raises(SystemExit, match="other settings"):
        previous_rows(out, settings(**change))


def test_gpu_types_are_never_mixed_in_one_file(tmp_path):
    out = tmp_path / "rows.jsonl"
    values = dict(loglik=-1.0, score=-1.0, n_tokens=1, gpu="NVIDIA A100-SXM4-40GB")
    write_jsonl(out, [backup_row(settings(), "c0", **values)])
    assert len(previous_rows(out, settings(), "NVIDIA A100-SXM4-80GB")) == 1
    for other in ("NVIDIA L4", None):
        with pytest.raises(SystemExit, match="scored on"):
            previous_rows(out, settings(), other)
    assert [gpu_type(n) for n in ("NVIDIA A100-SXM4-40GB", "NVIDIA L4", "Tesla T4", None)] == ["A100", "L4", "T4", None]


def test_load_items_filters_and_checks_the_bank_hash(tmp_path):
    bank = tmp_path / "bank.jsonl"
    write_jsonl(bank, ITEMS + [{"candidate_id": "k", "domain": "code", "problem": "p", "candidate": "c d"}])
    items, digest = load_items(bank)
    assert len(items) == 5
    assert [it["candidate_id"] for it in load_items(bank, domain="code")[0]] == ["k"]
    with pytest.raises(SystemExit, match="sha256"):
        load_items(bank, bank_hash="0" * 64)
    subset = tmp_path / "subset.json"
    subset.write_text(json.dumps({"bank_hash": digest, "splits": {"validation": ["c1"], "test": ["c3"]}}))
    assert [it["candidate_id"] for it in load_items(bank, subset=subset)[0]] == ["c1", "c3"]
    subset.write_text(json.dumps({"bank_hash": digest, "splits": {"validation": ["c1"], "test": ["gone"]}}))
    with pytest.raises(SystemExit, match="not selected"):
        load_items(bank, subset=subset)
    subset.write_text(json.dumps({"bank_hash": "other", "splits": {}}))
    with pytest.raises(SystemExit, match="drawn from"):
        load_items(bank, subset=subset)


def test_command_line_runs_resumes_and_refuses(tmp_path, monkeypatch):
    import harness.diffusion.run_reconstruction as runner
    loads = []

    def fake_load(key, n_slots, device, revision, strict_env):
        loads.append(key)
        return make(key)

    monkeypatch.setattr(runner, "load_scorer", fake_load)
    monkeypatch.setattr(runner, "environment_info", lambda device: {})
    bank, out = tmp_path / "bank.jsonl", tmp_path / "out" / "rows.jsonl"
    write_jsonl(bank, ITEMS)
    args = ["--model", "llada", "--items", str(bank), "--out", str(out), "--device", "cpu", "--n-samples", "16"]
    assert main([*args, "--limit", "1"]) == 0 and main(args) == 0 and len(loads) == 2
    assert main(args) == 0 and len(loads) == 2  # complete: the model is not loaded again
    rows = [json.loads(line) for line in out.read_text().splitlines()]
    assert [r["candidate_id"] for r in rows] == ["c0", "c1", "c2", "c3"]
    assert {(r["seed"], r["n_samples"], r["mc_group"], r["micro_batch"]) for r in rows} == {(544, 16, 16, 1)}
    manifests = [json.loads(line) for line in open(str(out) + ".manifest.jsonl")]
    assert [m["completed"] for m in manifests] == [{"ok": 1}, {"ok": 3}]
    assert all(m["end_of_turn_scored"] is False for m in manifests)
    with pytest.raises(SystemExit, match="other settings"):
        main([*args, "--seed", "545"])


def test_the_repair_defaults_are_the_plans():
    assert (N_SAMPLES, SEED, REPEAT_SEED, MC_GROUP) == (128, 544, 545, 16)
    defaults = {k: p.default for k, p in inspect.signature(reconstruct).parameters.items()}
    assert (defaults["seed"], defaults["n_samples"], defaults["micro_batch"], defaults["group"]) == (544, 128, 1, 16)


# --- micro-batching ---------------------------------------------------------

@pytest.mark.parametrize("key", ["llada", "dream"])
def test_micro_batches_agree_in_fp32(key):
    # Equal-length samples of one item need no padding, so in FP32 only float rounding is left. On the
    # real BF16 weights a different matrix shape rounds differently (decisions/2026-10-26-diffusion-
    # scored-unbatched.md), so the default stays 1 and check_reconstruction reports the difference.
    scorer, pair = make(key), SimpleNamespace(prompt=PROMPT, answer=tuple(range(44, 62)))
    single = reconstruct(scorer, pair, "c", n_samples=32, micro_batch=1)
    for micro_batch in (2, 7, 16, 32, 64):
        batched = reconstruct(scorer, pair, "c", n_samples=32, micro_batch=micro_batch)
        assert batched.score == pytest.approx(single.score, abs=1e-5)


# --- the GPU check's logic, on tiny models ----------------------------------

class TinyDream(TinyBidirectional):
    """The parts of the official sampler that check_dream uses: append masks, one pass, shift, hook."""

    def diffusion_generate(self, prompt, attention_mask, max_new_tokens, steps, generation_logits_hook_func, **_):
        x = F.pad(prompt, (0, max_new_tokens), value=MASK)
        logits = self(x, "full", None).logits
        logits = torch.cat([logits[:, :1], logits[:, :-1]], dim=1)
        generation_logits_hook_func(0, x, logits)


class CheckTokenizer(MockTokenizer):
    def apply_chat_template(self, messages, add_generation_prompt=False, tokenize=True, return_tensors=None,
                            return_dict=False):
        text = super().apply_chat_template(messages, add_generation_prompt, tokenize=False)
        return {"input_ids": torch.tensor([self.encode(text)])} if return_dict else text

    def __call__(self, text, add_special_tokens=True):
        return {"input_ids": self.encode(text, add_special_tokens)}


def test_gpu_check_passes_on_a_matching_llada_and_catches_a_wrong_read():
    scorer = make("llada", tokenizer=CheckTokenizer())
    row = check_item(scorer, ITEMS[2], official_get_log_likelihood)
    assert row["ok"] and row["diff_per_token"] < 1e-5 and row["prompt_never_masked"] and row["same_mask_counts"]
    pair = encode_pair(scorer.tokenizer, ITEMS[2]["problem"], ITEMS[2]["candidate"])
    wrong = make("llada", logit_shift=1)  # reads one position too early
    assert check_llada(wrong, pair, official_get_log_likelihood)["diff_per_token"] > check_reconstruction.TOL


def test_gpu_check_passes_on_a_matching_dream_and_catches_a_missing_shift():
    model = TinyDream("absolute")
    row = check_item(make("dream", model, CheckTokenizer()), ITEMS[2])
    assert row["ok"] and row["same_noisy_input"] and row["diff_per_token"] < 1e-5
    pair = encode_pair(CheckTokenizer(), ITEMS[2]["problem"], ITEMS[2]["candidate"])
    wrong = check_dream(make("dream", model, logit_shift=0), pair, "c2")
    assert wrong["diff_per_token"] > check_reconstruction.TOL
    assert wrong["ours_per_token"] == pytest.approx(row["without_shift_per_token"], abs=1e-4)
