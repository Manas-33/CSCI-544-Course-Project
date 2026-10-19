"""Tiny tokenizer and model mocks for the CPU unit tests (no downloads)."""
from __future__ import annotations

import math
from types import SimpleNamespace

import torch

VOCAB = 64
MASK, PAD, BOS, EOS = 1, 2, 3, 4
YES, NO = 10, 20  # the single-token "Yes" and "No"


class MockTokenizer:
    """Word-level tokenizer; strings in ``pieces`` map to fixed id lists."""

    pad_token_id = PAD
    eos_token_id = EOS

    def __init__(self, pieces=None):
        self.pieces = pieces if pieces is not None else {
            "Yes": [YES], " Yes": [YES + 1], "yes": [YES + 2], " yes": [YES + 2],
            "YES": [30, 31], " YES": [YES + 3],
            "No": [NO], " No": [NO + 1], "no": [NO + 2], " no": [NO + 2],
            "NO": [32, 33], " NO": [NO + 3],
        }
        self.encode_calls = []

    def encode(self, text, add_special_tokens=True):
        self.encode_calls.append(add_special_tokens)
        if text in self.pieces:
            return list(self.pieces[text])
        ids = [BOS] if text.startswith("<bos>") else []
        body = text[len("<bos>"):] if ids else text
        # Content words map into 40..63, clear of the special and verdict ids.
        return ids + [40 + sum(map(ord, w)) % 24 for w in body.split()]

    def apply_chat_template(self, messages, add_generation_prompt=False, tokenize=True):
        assert not tokenize
        text = "".join(f"<bos>{m['role']}\n{m['content']}<eos>" for m in messages)
        return text + ("<bos>assistant\n" if add_generation_prompt else "")

    def decode(self, ids):
        return " ".join(str(i) for i in ids)


def _out(logits):
    return SimpleNamespace(logits=logits)


class SlotMarkerModel:
    """Logits favour YES only where the first masked slot is predicted.

    With ``shift=0`` position i predicts token i (LLaDA, iLLaDA); with
    ``shift=1`` position i predicts token i + 1 (Dream's raw logits). Every
    other position favours NO, so reading the wrong position gives s < 0.
    """

    def __init__(self, shift=0):
        self.shift = shift
        self.calls = []

    def __call__(self, input_ids, attention_mask=None, position_ids=None):
        self.calls.append(SimpleNamespace(
            input_ids=input_ids, attention_mask=attention_mask, position_ids=position_ids))
        b, n = input_ids.shape
        logits = torch.zeros(b, n, VOCAB)
        logits[:, :, NO] = 5.0
        for row in range(b):
            first = int((input_ids[row] == MASK).nonzero()[0])
            pos = first - self.shift
            logits[row, pos, NO] = 0.0
            logits[row, pos, YES] = 5.0
        return _out(logits)


def _rope(x, positions):
    """Rotary embedding on the last dim (pairs of features), as in LLaDA/Dream."""
    half = x.shape[-1] // 2
    freqs = 1.0 / (100.0 ** (torch.arange(half, dtype=torch.float32) / half))
    angles = positions[..., None].float() * freqs
    cos, sin = angles.cos(), angles.sin()
    x1, x2 = x[..., :half], x[..., half:]
    return torch.cat([x1 * cos - x2 * sin, x1 * sin + x2 * cos], dim=-1)


class TinyBidirectional(torch.nn.Module):
    """Two-layer bidirectional attention model with random weights.

    ``positions="rope"`` uses rotary positions 0..L-1 like LLaDA (left padding
    shifts all positions equally, which rotary attention ignores);
    ``positions="absolute"`` adds learned absolute embeddings at
    ``position_ids``, so padding is invisible only if the caller passes the
    unpadded positions, as Dream's sampler does with ``tok_idx``.

    Accepts a 2D key mask (LLaDA), a 4D bool mask or "full" (Dream).
    """

    def __init__(self, positions="rope", dim=32, layers=2, seed=0):
        super().__init__()
        torch.manual_seed(seed)
        self.positions = positions
        self.emb = torch.nn.Embedding(VOCAB, dim)
        self.pos = torch.nn.Embedding(256, dim)
        self.qkv = torch.nn.ModuleList(torch.nn.Linear(dim, 3 * dim) for _ in range(layers))
        self.head = torch.nn.Linear(dim, VOCAB)
        self.dim = dim

    def forward(self, input_ids, attention_mask=None, position_ids=None):
        b, n = input_ids.shape
        x = self.emb(input_ids)
        if position_ids is None:
            position_ids = torch.arange(n).expand(b, n)
        if self.positions == "absolute":
            x = x + self.pos(position_ids)
        allowed = None
        if isinstance(attention_mask, torch.Tensor):
            if attention_mask.dim() == 2:
                allowed = attention_mask.bool()[:, None, :].expand(b, n, n)
            else:
                allowed = attention_mask[:, 0].bool()
        for layer in self.qkv:
            q, k, v = layer(x).chunk(3, dim=-1)
            if self.positions == "rope":
                q, k = _rope(q, position_ids), _rope(k, position_ids)
            scores = q @ k.transpose(1, 2) / math.sqrt(self.dim)
            if allowed is not None:
                scores = scores.masked_fill(~allowed, float("-inf"))
            # Fully masked rows (padding queries under Dream's mask) give zeros.
            weights = torch.nan_to_num(torch.softmax(scores, dim=-1), nan=0.0)
            x = x + weights @ v
        return _out(self.head(x))
