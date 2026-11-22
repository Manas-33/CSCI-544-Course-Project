"""Masked-reconstruction log-likelihood of a candidate given the problem (PLAN.md §4, §7).

The problem is a user turn in the model's own chat template and the candidate is the
assistant's answer. The estimate is LLaDA's official ``get_log_likelihood`` (ML-GSAI/LLaDA,
commit 9182493): for each Monte Carlo sample, k of the L answer tokens are replaced by the
mask token (the prompt never is), one forward pass predicts them, and

    sample = (L / k) * sum over the masked tokens of log p(token | everything unmasked)

The log-likelihood is the mean over samples and the score is that divided by L (per token),
so higher means more plausible. As in the official code, samples come in groups of 16: k is
uniform on 1..L for the first of a group and the others are spread evenly from it.

Dream uses the same estimate with its one-position logit shift: the raw logits at position
i - 1 predict token i. Its attention ("full") and position ids are those of scorer.py.

What is scored: exactly the answer's tokens. The end-of-turn token is neither shown nor
scored, because the official code scores the answer ids it is given and appends nothing.
The candidate's leading and trailing whitespace is dropped, which LLaDA's own template (and
Llama's and Gemma's) does anyway, so every model is scored on the same text.

Only torch is imported here; harness/ar_logprob.py reuses ``encode_pair``.
"""
from __future__ import annotations

import hashlib
import math
from dataclasses import dataclass
from typing import Sequence

import torch

METHOD = "diffusion_mc"
MODELS = ("llada", "dream")
N_SAMPLES = 128      # official default, and what the repair always uses (PLAN.md §7)
MC_GROUP = 16        # official default batch_size: mask counts are spread evenly within a group
SEED, REPEAT_SEED = 544, 545
SAMPLE_COUNTS = (8, 16, 32, 64, 128)  # the development-bank choice (PLAN.md §4)

_PROBE = "PROBEANSWER"


# ---------------------------------------------------------------- prompt and answer tokens

@dataclass(frozen=True)
class PromptAnswer:
    prompt: tuple   # token ids of the user turn and the generation prompt
    answer: tuple   # token ids of the answer, without any end-of-turn token
    boundary: str   # "clean": the prompt's ids are a prefix of the whole text's ids;
                    # "separate": they are not, so the answer was tokenized on its own


class EmptyAnswer(ValueError):
    pass


def chat_texts(tokenizer, problem: str, candidate: str):
    """(prompt text, answer text) in the model's own chat template.

    Whatever the template adds after an assistant answer is cut off: the end-of-turn token
    and, for LLaDA, the next generation prompt, which its template appends every time.
    Whitespace added before the answer (Mistral's " ") belongs to the answer's first token.
    """
    answer = candidate.strip()
    if not answer:
        raise EmptyAnswer("the candidate is empty")
    user = {"role": "user", "content": problem}
    prompt = tokenizer.apply_chat_template([user], add_generation_prompt=True, tokenize=False)

    def after_prompt(text):
        full = tokenizer.apply_chat_template(
            [user, {"role": "assistant", "content": text}], add_generation_prompt=False, tokenize=False)
        if not full.startswith(prompt):
            raise ValueError("the chat template renders the prompt differently when an answer follows")
        return full[len(prompt):]

    lead, probe, tail = after_prompt(_PROBE).partition(_PROBE)
    if not probe or lead.strip():
        raise ValueError(f"the chat template puts text before the answer: {lead!r}")
    if after_prompt(answer) != lead + answer + tail:
        raise ValueError("the chat template changes the answer text")
    return prompt, lead + answer


def encode_pair(tokenizer, problem: str, candidate: str) -> PromptAnswer:
    """Prompt and answer token ids with no token spanning the boundary between them.

    The whole text is tokenized and cut after the prompt's tokens, as the official evaluation
    does. If a token would span the boundary, the answer is tokenized on its own instead.
    """
    prompt_text, answer_text = chat_texts(tokenizer, problem, candidate)
    # The templates already insert BOS, so no special tokens are added here.
    prompt = list(tokenizer.encode(prompt_text, add_special_tokens=False))
    whole = list(tokenizer.encode(prompt_text + answer_text, add_special_tokens=False))
    boundary = "clean"
    answer = whole[len(prompt):]
    if whole[:len(prompt)] != prompt or not answer:
        boundary = "separate"
        answer = list(tokenizer.encode(answer_text, add_special_tokens=False))
    if not prompt or not answer:
        raise EmptyAnswer("the prompt or the answer has no tokens")
    return PromptAnswer(tuple(prompt), tuple(answer), boundary)


# ---------------------------------------------------------------- masks

def item_generator(seed: int, candidate_id: str) -> torch.Generator:
    """A CPU generator that depends on the seed and the item only, so an estimate is the same
    whatever the order of the items and after a resume."""
    digest = hashlib.sha256(f"{seed}:{candidate_id}".encode("utf-8")).digest()
    return torch.Generator().manual_seed(int.from_bytes(digest[:8], "big") >> 1)


def mask_counts(k: int, n_answer: int, group: int) -> torch.Tensor:
    """How many answer tokens each sample of a group masks (official ``forward_process``)."""
    x = torch.round(torch.linspace(float(k), k + (group - 1) * (n_answer / group), steps=group)).long()
    return ((x - 1) % n_answer) + 1


def draw_masks(n_answer: int, n_samples: int, generator: torch.Generator,
               group: int = MC_GROUP) -> torch.Tensor:
    """Boolean masks (n_samples, n_answer) over the answer tokens, at least one True per row.

    Same calls in the same order as the official ``forward_process``, once per group.
    """
    group = min(group, n_samples)
    if n_answer < 1 or n_samples < 1 or n_samples % group:
        raise ValueError(f"cannot draw {n_samples} samples in groups of {group} for {n_answer} tokens")
    rounds = []
    for _ in range(n_samples // group):
        k = int(torch.randint(1, n_answer + 1, (), generator=generator))
        is_mask = torch.arange(n_answer).repeat(group, 1) < mask_counts(k, n_answer, group).unsqueeze(1)
        for i in range(group):
            is_mask[i] = is_mask[i][torch.randperm(n_answer, generator=generator)]
        rounds.append(is_mask)
    return torch.cat(rounds)


def partial_samples(n: int, n_samples: int, group: int = MC_GROUP) -> list:
    """Which of ``n_samples`` samples make up an estimate from n of them: whole groups, or an
    evenly spaced part of the first group (the official mask counts for a group of n)."""
    group = min(group, n_samples)
    if not 1 <= n <= n_samples or (n % group if n >= group else group % n):
        raise ValueError(f"no {n}-sample estimate inside {n_samples} samples in groups of {group}")
    return list(range(n)) if n >= group else list(range(0, group, group // n))


# ---------------------------------------------------------------- estimate

@torch.no_grad()
def sample_logliks(scorer, prompt: Sequence[int], answer: Sequence[int], masks: torch.Tensor,
                   micro_batch: int = 1) -> list:
    """One value per mask: (L / k) times the summed log-probability of the k masked tokens.

    ``micro_batch`` is how many samples of this one item share a forward pass. They all have
    the same length, so nothing is padded; in BF16 the result can still move by rounding.
    """
    n_prompt, n_answer = len(prompt), len(answer)
    if masks.shape[1] != n_answer or not bool(masks.any(dim=1).all()):
        raise ValueError("every sample must mask at least one of the answer tokens")
    if scorer.spec.mask_id in prompt or scorer.spec.mask_id in answer:
        raise ValueError("the prompt or the answer already contains the mask token")
    if micro_batch < 1:
        raise ValueError("micro_batch must be at least 1")
    sequence = torch.tensor(list(prompt) + list(answer), dtype=torch.long)
    first = n_prompt - scorer.spec.logit_shift  # raw-logit position that predicts answer token 0
    values = []
    for start in range(0, len(masks), micro_batch):
        batch = masks[start:start + micro_batch]
        noisy = sequence.repeat(len(batch), 1)
        noisy[:, n_prompt:][batch] = scorer.spec.mask_id
        noisy = noisy.to(scorer.device)
        logits = scorer.forward(noisy, torch.ones_like(noisy))
        for row, masked in zip(logits, batch):
            where = masked.nonzero(as_tuple=True)[0]
            log_probs = torch.log_softmax(row[(first + where).to(row.device)].float(), dim=-1)
            targets = sequence[n_prompt + where].to(row.device)
            total = float(log_probs.gather(1, targets[:, None]).double().sum())
            if not math.isfinite(total):
                raise ValueError("nonfinite log-probability")
            values.append(total * n_answer / len(where))
    return values


@dataclass(frozen=True)
class Reconstruction:
    loglik: float     # estimated log-likelihood of the whole answer
    n_tokens: int     # answer tokens scored
    score: float      # loglik / n_tokens
    loglik_at: dict   # {"8": ..., "16": ...}: the same estimate from fewer samples


def reconstruct(scorer, pair: PromptAnswer, candidate_id: str, seed: int = SEED,
                n_samples: int = N_SAMPLES, micro_batch: int = 1, group: int = MC_GROUP) -> Reconstruction:
    masks = draw_masks(len(pair.answer), n_samples, item_generator(seed, candidate_id), group)
    values = sample_logliks(scorer, pair.prompt, pair.answer, masks, micro_batch)

    def mean(n):
        chosen = [values[i] for i in partial_samples(n, n_samples, group)]
        return math.fsum(chosen) / len(chosen)

    loglik, n_tokens = mean(n_samples), len(pair.answer)
    return Reconstruction(loglik, n_tokens, loglik / n_tokens,
                          {str(n): mean(n) for n in SAMPLE_COUNTS if n <= n_samples})


def fits(scorer, pair: PromptAnswer) -> bool:
    return len(pair.prompt) + len(pair.answer) <= scorer.spec.max_len
