"""One-pass Yes/No verdict scorer for LLaDA, iLLaDA and Dream (PLAN.md §4).

The prompt, in the model's own chat template, is followed by N masked answer
slots (1, 8 or 32). One forward pass with full bidirectional attention gives a
distribution at every slot; the score is read at the first masked slot:

    s = log P(Yes) - log P(No)

where P(Yes) and P(No) sum over every spelling that is a single token in that
model's tokenizer. Dream's logits are shifted right by one (its official
sampler does ``logits = cat([logits[:, :1], logits[:, :-1]], 1)``), so its
first slot is read from the raw logits one position earlier.

Only torch is imported here, so the logic is unit-tested on CPU with mock
models. ``load_scorer`` imports transformers and downloads the model.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Sequence

import torch


@dataclass(frozen=True)
class ModelSpec:
    key: str           # short name used in file names and score rows
    hf_id: str
    revision: str      # Hugging Face commit: pins the weights and the remote code
    mask_id: int
    mask_token: str    # the tokenizer's string for mask_id
    logit_shift: int   # 1 if raw logits at position i predict token i + 1 (Dream)
    batched: bool      # False: one unpadded prompt per forward pass
    max_len: int       # prompt + answer slots must fit
    transformers: str  # version the official code pins
    torch: Optional[str] = None  # torch version the official code pins, if any


# Mask ids, pins and revisions checked on 2026-10-19 against the model cards,
# config.json / tokenizer_config.json, github.com/ML-GSAI/LLaDA (generate.py,
# eval_illada.sh) and github.com/HKUNLP/Dream (README, generation_utils.py).
SPECS = {
    "llada": ModelSpec(
        key="llada",
        hf_id="GSAI-ML/LLaDA-8B-Instruct",
        revision="08b83a6feb34df1a6011b80c3c00c7563e963b07",  # 2025-10-21, adds attention_mask
        mask_id=126336,
        mask_token="<|mdm_mask|>",
        logit_shift=0,
        batched=True,
        max_len=4096,
        transformers="4.38.2",
    ),
    "illada": ModelSpec(
        key="illada",
        hf_id="GSAI-ML/iLLaDA-8B-Instruct",
        revision="5769f049224b2cff61106ebce0429ede5deeb434",  # 2026-06-28
        mask_id=5,
        mask_token="<[MASK]>",
        logit_shift=0,
        batched=False,  # official generate() asserts batch size 1 for iLLaDA
        max_len=8192,
        transformers="4.57.1",
    ),
    "dream": ModelSpec(
        key="dream",
        hf_id="Dream-org/Dream-v0-Instruct-7B",
        revision="05334cb9faaf763692dcf9d8737c642be2b2a6ae",  # 2025-07-15
        mask_id=151666,
        mask_token="<|mask|>",
        logit_shift=1,
        batched=True,
        max_len=2048,  # documented input + output context
        transformers="4.46.2",
        torch="2.5.1",
    ),
}

# Text appended after the chat template's generation prompt, before the answer slots. The
# user message (T1-T3) is the same for every model; a prefix only changes how the model's
# own turn begins. iLLaDA opens every answer with a <think> block (its official evaluation
# closes it with "</think>"), so its first slot holds "<", not Yes/No; an empty think block
# turns thinking off, as PLAN.md does for Qwen3-8B. The variant is chosen on the
# development bank by median Yes+No mass (choose_slots.py; decisions/2026-10-25-illada-thinking-off.md).
# Each variant has a newline before "</think>", so the closing tag tokenizes as the official
# end-of-thinking text encode("</think>") does ("</", "think", ">"); "<think></think>" would not.
ANSWER_PREFIXES = {
    "none": "",
    "think_nl": "<think>\n</think>\n",
    "think_nlnl": "<think>\n\n</think>\n\n",
}

YES_SPELLINGS = ("Yes", " Yes", "yes", " yes", "YES", " YES")
NO_SPELLINGS = ("No", " No", "no", " no", "NO", " NO")


@dataclass(frozen=True)
class VerdictTokens:
    yes_ids: tuple
    no_ids: tuple
    kept: tuple     # spellings that are one token
    dropped: tuple  # spellings that split into several tokens


def verdict_tokens(tokenizer, yes=YES_SPELLINGS, no=NO_SPELLINGS) -> VerdictTokens:
    """Token ids for the Yes and No sets; multi-token spellings are dropped."""
    kept, dropped = [], []

    def ids_for(spellings):
        ids = []
        for s in spellings:
            enc = tokenizer.encode(s, add_special_tokens=False)
            if len(enc) != 1:
                dropped.append(s)
                continue
            kept.append(s)
            if enc[0] not in ids:  # two spellings can share one token
                ids.append(enc[0])
        return tuple(ids)

    yes_ids, no_ids = ids_for(yes), ids_for(no)
    if not yes_ids or not no_ids:
        raise ValueError(f"empty Yes or No set (dropped: {dropped})")
    if set(yes_ids) & set(no_ids):
        raise ValueError(f"Yes and No sets share tokens: {set(yes_ids) & set(no_ids)}")
    return VerdictTokens(yes_ids, no_ids, tuple(kept), tuple(dropped))


@dataclass(frozen=True)
class Verdict:
    score: float  # log P(Yes) - log P(No)
    p_yes: float
    p_no: float
    mass: float   # P(Yes) + P(No), the format check
    top_id: int   # most likely token at the slot (diagnostic)


def verdict_from_logits(logits: torch.Tensor, tokens: VerdictTokens) -> Verdict:
    """Verdict from the logits (vocab,) that predict the first answer slot."""
    logp = torch.log_softmax(logits.float(), dim=-1)
    lp_yes = torch.logsumexp(logp[list(tokens.yes_ids)], dim=0)
    lp_no = torch.logsumexp(logp[list(tokens.no_ids)], dim=0)
    p_yes, p_no = lp_yes.exp().item(), lp_no.exp().item()
    return Verdict(
        score=(lp_yes - lp_no).item(),
        p_yes=p_yes,
        p_no=p_no,
        mass=p_yes + p_no,
        top_id=int(logp.argmax().item()),
    )


def answer_slot_index(prompt_len: int, logit_shift: int) -> int:
    """Raw-logit position that predicts the first answer slot.

    The first masked slot sits at index ``prompt_len`` (prompts are left-padded
    to a common length). With Dream's shift, token i is predicted at i - 1.
    """
    return prompt_len - logit_shift


def build_batch(prompts: Sequence[Sequence[int]], n_slots: int, mask_id: int, pad_id: int):
    """Left-pad prompts to a common length and append ``n_slots`` mask tokens.

    Returns ``(input_ids, attention_mask, prompt_len)``; the attention mask is 0
    on padding only, so prompt and answer slots attend to each other (as in the
    official LLaDA and Dream samplers).
    """
    if n_slots < 1:
        raise ValueError("need at least one answer slot")
    if not prompts or any(len(p) == 0 for p in prompts):
        raise ValueError("empty prompt")
    prompt_len = max(len(p) for p in prompts)
    if pad_id == mask_id and any(len(p) < prompt_len for p in prompts):
        raise ValueError("pad id equals mask id")
    rows, att = [], []
    for p in prompts:
        if mask_id in p:
            raise ValueError("prompt already contains the mask token")
        pad = prompt_len - len(p)
        rows.append([pad_id] * pad + list(p) + [mask_id] * n_slots)
        att.append([0] * pad + [1] * (len(p) + n_slots))
    return torch.tensor(rows, dtype=torch.long), torch.tensor(att, dtype=torch.long), prompt_len


class PromptTooLong(ValueError):
    pass


class DiffusionScorer:
    """Scores user messages with one forward pass each (or one per batch)."""

    def __init__(self, model, tokenizer, spec: ModelSpec, n_slots: int,
                 device=None, tokens: Optional[VerdictTokens] = None,
                 revision: Optional[str] = None, answer_prefix: str = "none"):
        if answer_prefix not in ANSWER_PREFIXES:
            raise ValueError(f"unknown answer prefix {answer_prefix!r}; choose from {sorted(ANSWER_PREFIXES)}")
        self.answer_prefix = answer_prefix
        self.model = model
        self.tokenizer = tokenizer
        self.spec = spec
        self.n_slots = n_slots
        self.revision = revision or spec.revision
        self.device = device if device is not None else next(model.parameters()).device
        self.tokens = tokens if tokens is not None else verdict_tokens(tokenizer)
        pad = getattr(tokenizer, "pad_token_id", None)
        self.pad_id = pad if pad is not None else tokenizer.eos_token_id

    def render(self, user_message: str) -> str:
        """The prompt text in the model's own chat template."""
        return self.tokenizer.apply_chat_template(
            [{"role": "user", "content": user_message}],
            add_generation_prompt=True, tokenize=False)

    def prompt_ids(self, user_message: str) -> list:
        # The templates already insert BOS, so no special tokens are added here.
        text = self.render(user_message) + ANSWER_PREFIXES[self.answer_prefix]
        return self.tokenizer.encode(text, add_special_tokens=False)

    def check_length(self, prompt: Sequence[int]) -> None:
        if len(prompt) + self.n_slots > self.spec.max_len:
            raise PromptTooLong(
                f"{len(prompt)} prompt tokens + {self.n_slots} slots > {self.spec.max_len}")

    def score(self, user_messages: Sequence[str]) -> list:
        return self.score_ids([self.prompt_ids(m) for m in user_messages])

    @torch.no_grad()
    def score_ids(self, prompts: Sequence[Sequence[int]]) -> list:
        """One Verdict per prompt; one forward pass for the batch."""
        if not self.spec.batched and len(prompts) > 1:
            return [v for p in prompts for v in self.score_ids([p])]
        for p in prompts:
            self.check_length(p)
        input_ids, attention_mask, prompt_len = build_batch(
            prompts, self.n_slots, self.spec.mask_id, self.pad_id)
        logits = self.forward(input_ids.to(self.device), attention_mask.to(self.device))
        pos = answer_slot_index(prompt_len, self.spec.logit_shift)
        return [verdict_from_logits(logits[b, pos], self.tokens) for b in range(len(prompts))]

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError


class LLaDAScorer(DiffusionScorer):
    def forward(self, input_ids, attention_mask):
        # As in generate.py: model(x, attention_mask=attention_mask).logits. The
        # model ignores a mask without zeros.
        return self.model(input_ids, attention_mask=attention_mask).logits


class ILLaDAScorer(DiffusionScorer):
    def forward(self, input_ids, attention_mask):
        # The official iLLaDA evaluation runs one unpadded prompt with no mask.
        if input_ids.shape[0] != 1:
            raise ValueError("iLLaDA is scored unbatched")
        return self.model(input_ids).logits


class DreamScorer(DiffusionScorer):
    def forward(self, input_ids, attention_mask):
        # Same mask and position ids as generation_utils.py _sample(). Returns raw
        # (unshifted) logits; answer_slot_index applies the shift.
        if torch.any(attention_mask == 0):
            tok_idx = attention_mask.long().cumsum(-1) - 1
            tok_idx.masked_fill_(attention_mask == 0, 1)
            attention_mask = torch.logical_and(
                attention_mask.unsqueeze(1).unsqueeze(-2),
                attention_mask.unsqueeze(1).unsqueeze(-1),
            )
        else:
            tok_idx, attention_mask = None, "full"
        return self.model(input_ids, attention_mask, tok_idx).logits


SCORERS = {"llada": LLaDAScorer, "illada": ILLaDAScorer, "dream": DreamScorer}


def check_environment(spec: ModelSpec) -> None:
    """Raise if the installed library versions differ from the official pins."""
    import transformers

    problems = []
    if transformers.__version__ != spec.transformers:
        problems.append(f"transformers {transformers.__version__} (need {spec.transformers})")
    if spec.torch and torch.__version__.split("+")[0] != spec.torch:
        problems.append(f"torch {torch.__version__} (need {spec.torch})")
    if problems:
        raise RuntimeError(
            f"{spec.key} needs its own environment: " + ", ".join(problems)
            + f". Install harness/diffusion/requirements/{spec.key}.txt.")


def load_scorer(key: str, n_slots: int, device: str = "cuda",
                revision: Optional[str] = None, strict_env: bool = True,
                answer_prefix: str = "none") -> DiffusionScorer:
    """Download and load one model in bf16 and wrap it in its scorer."""
    from transformers import AutoModel, AutoTokenizer

    spec = SPECS[key]
    if strict_env:
        check_environment(spec)
    revision = revision or spec.revision
    tokenizer = AutoTokenizer.from_pretrained(
        spec.hf_id, revision=revision, trust_remote_code=True)
    # device_map loads the weights straight onto the device (needs accelerate),
    # so a 16 GB model does not pass through Colab's ~12 GB of system RAM.
    model = AutoModel.from_pretrained(
        spec.hf_id, revision=revision, trust_remote_code=True,
        torch_dtype=torch.bfloat16, device_map={"": device}).eval()
    return SCORERS[key](model, tokenizer, spec, n_slots, device=device, revision=revision,
                        answer_prefix=answer_prefix)
