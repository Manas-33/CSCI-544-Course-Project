"""Metrics for verifier scores: AUROC, calibration, matched-TAR pair rule, overlap, bootstrap.

Scores are oriented so that higher means "more likely correct". Labels are 1 (correct)
and 0 (wrong). Everything is plain numpy so it runs anywhere.
"""

from __future__ import annotations

from collections import defaultdict

import numpy as np


def ranks(x: np.ndarray) -> np.ndarray:
    """Average ranks (1-based), ties sharing the mean rank."""
    order = np.argsort(x, kind="mergesort")
    r = np.empty(len(x), dtype=float)
    sx = x[order]
    i = 0
    while i < len(x):
        j = i
        while j + 1 < len(x) and sx[j + 1] == sx[i]:
            j += 1
        r[order[i:j + 1]] = (i + j) / 2 + 1
        i = j + 1
    return r


def auroc(scores: np.ndarray, labels: np.ndarray) -> float:
    """Probability a random correct answer outscores a random wrong one (ties count half)."""
    pos, neg = labels == 1, labels == 0
    n_pos, n_neg = pos.sum(), neg.sum()
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    r = ranks(scores)
    return float((r[pos].sum() - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg))


def spearman(a: np.ndarray, b: np.ndarray) -> float:
    if len(a) < 3:
        return float("nan")
    ra, rb = ranks(a), ranks(b)
    if ra.std() == 0 or rb.std() == 0:
        return float("nan")
    return float(np.corrcoef(ra, rb)[0, 1])


def sigmoid(x: np.ndarray) -> np.ndarray:
    return 1 / (1 + np.exp(-np.clip(x, -50, 50)))


def ece(scores: np.ndarray, labels: np.ndarray, bins: int = 10) -> float:
    """Expected calibration error of P(correct) = P(Yes) / (P(Yes) + P(No)) = sigmoid(score)."""
    p = sigmoid(scores)
    idx = np.minimum((p * bins).astype(int), bins - 1)
    total = 0.0
    for b in range(bins):
        m = idx == b
        if m.any():
            total += m.mean() * abs(p[m].mean() - labels[m].mean())
    return float(total)


def threshold_at_tar(correct_scores: np.ndarray, tar: float) -> float:
    """Highest threshold t with P(score >= t | correct) >= tar."""
    s = np.sort(correct_scores)[::-1]
    k = int(np.ceil(tar * len(s))) - 1
    return float(s[min(max(k, 0), len(s) - 1)])


def pair_thresholds(ca: np.ndarray, cb: np.ndarray, pair_tar: float = 0.90) -> tuple[float, float, float]:
    """Per-verifier thresholds at a shared individual TAR r, the smallest r at which the AND pair
    accepts at least `pair_tar` of correct answers (PLAN.md §5 step 2). Returns (t_a, t_b, r)."""
    for r in np.linspace(pair_tar, 1.0, 2001):
        ta, tb = threshold_at_tar(ca, r), threshold_at_tar(cb, r)
        if np.mean((ca >= ta) & (cb >= tb)) >= pair_tar:
            return ta, tb, float(r)
    return float(ca.min()), float(cb.min()), 1.0


def cohen_kappa(x: np.ndarray, y: np.ndarray) -> float:
    """Chance-adjusted agreement of two binary vectors."""
    if len(x) == 0:
        return float("nan")
    po = np.mean(x == y)
    px, py = x.mean(), y.mean()
    pe = px * py + (1 - px) * (1 - py)
    return float("nan") if pe == 1 else float((po - pe) / (1 - pe))


class ProblemBootstrap:
    """Resample whole problems (all their candidates together), with a fixed seed (PLAN.md §5 step 8)."""

    def __init__(self, problem_ids: list[str], n: int = 2000, seed: int = 544):
        groups = defaultdict(list)
        for i, pid in enumerate(problem_ids):
            groups[pid].append(i)
        self.groups = [np.array(g) for g in groups.values()]
        rng = np.random.default_rng(seed)
        self.draws = [rng.integers(0, len(self.groups), len(self.groups)) for _ in range(n)]

    def indices(self):
        for d in self.draws:
            yield np.concatenate([self.groups[g] for g in d])

    def interval(self, stat, level: float = 0.95) -> tuple[float, float]:
        vals = np.array([stat(ix) for ix in self.indices()], dtype=float)
        vals = vals[np.isfinite(vals)]
        if len(vals) == 0:
            return float("nan"), float("nan")
        lo, hi = np.quantile(vals, [(1 - level) / 2, 1 - (1 - level) / 2])
        return float(lo), float(hi)
