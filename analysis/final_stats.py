"""Statistics of the final test: problem bootstrap, fast pair thresholds, matching check, decision table.

The bootstrap resamples whole problems (PLAN.md §5 step 8) and is written with weights: replicate b draws each
problem some number of times, so any ratio or rank statistic can be computed for all replicates at once. It
uses the same draws as analysis.metrics.ProblemBootstrap (a test checks that). Problems are sorted by name, so
the result does not depend on the order of the rows.
"""

from __future__ import annotations

from statistics import NormalDist

import numpy as np

from analysis.final_settings import RULES

TOL = 1e-9      # points; comparisons at a boundary (exactly -3, exactly 0) are not left to floating-point noise


class Boot:
    """Problem weights for n bootstrap replicates of one set of problems, or of several (strata, e.g. domains)
    resampled separately."""

    def __init__(self, problem_ids, n: int = RULES.bootstrap, seed=RULES.seed, strata=None):
        ids = np.asarray(problem_ids, dtype=str)
        strata = np.zeros(len(ids), dtype=int) if strata is None else np.unique(np.asarray(strata, dtype=str),
                                                                                return_inverse=True)[1]
        rng = np.random.default_rng(seed)
        blocks, self.group = [], np.empty(len(ids), dtype=int)
        offset = 0
        for s in range(strata.max() + 1 if len(ids) else 0):
            rows = np.flatnonzero(strata == s)
            names, inverse = np.unique(ids[rows], return_inverse=True)
            self.group[rows] = inverse + offset
            g = len(names)
            block = np.empty((n, g), dtype=np.float32)
            for b in range(n):
                block[b] = np.bincount(rng.integers(0, g, g), minlength=g)
            blocks.append(block)
            offset += g
        self.n, self.n_problems = n, offset
        self.weights = np.hstack(blocks) if blocks else np.empty((n, 0), dtype=np.float32)

    def per_problem(self, values) -> np.ndarray:
        return np.bincount(self.group, weights=np.asarray(values, dtype=float), minlength=self.n_problems)

    def sums(self, values) -> np.ndarray:
        """Sum of a per-row value in every replicate. Integer inputs stay exact (float64 product)."""
        return _product(self.weights, self.per_problem(values))

    def item_weights(self, rows=slice(None)) -> np.ndarray:
        """(replicates, rows) matrix: how often each row is drawn."""
        return self.weights[:, self.group[rows]]

    def ratio(self, numerator, denominator) -> np.ndarray:
        num, den = self.sums(numerator), self.sums(denominator)
        return np.divide(num, den, out=np.full(self.n, np.nan), where=den > 0)


def _product(weights: np.ndarray, values: np.ndarray, chunk: int = 2000) -> np.ndarray:
    return np.concatenate([weights[i:i + chunk].astype(float) @ values for i in range(0, len(weights), chunk)])


def interval(values, level: float = 0.95) -> tuple[float, float]:
    """Percentile interval over the finite replicates."""
    v = np.asarray(values, dtype=float)
    v = v[np.isfinite(v)]
    if len(v) == 0:
        return float("nan"), float("nan")
    lo, hi = np.quantile(v, [(1 - level) / 2, 1 - (1 - level) / 2])
    return float(lo), float(hi)


# ---------------------------------------------------------------- rank statistics, all replicates at once

def auroc_boot(boot: Boot, scores, labels) -> np.ndarray:
    """AUROC in every replicate. Wins (ties count half) are summed per pair of problems once, then weighted."""
    scores, labels = np.asarray(scores, dtype=float), np.asarray(labels)
    pos, neg = np.flatnonzero(labels == 1), np.flatnonzero(labels == 0)
    sp, sn = scores[pos], scores[neg]
    wins = (sp[:, None] > sn[None, :]).astype(np.float64) + 0.5 * (sp[:, None] == sn[None, :])
    by_pos = np.zeros((boot.n_problems, len(neg)))
    np.add.at(by_pos, boot.group[pos], wins)
    matrix = np.zeros((boot.n_problems, boot.n_problems))
    np.add.at(matrix.T, boot.group[neg], by_pos.T)
    n_pos, n_neg = boot.sums(labels == 1), boot.sums(labels == 0)
    out = np.empty(boot.n)
    for i in range(0, boot.n, 1000):
        w = boot.weights[i:i + 1000].astype(float)
        out[i:i + 1000] = np.einsum("ij,ij->i", w @ matrix, w)
    total = n_pos * n_neg
    return np.divide(out, total, out=np.full(boot.n, np.nan), where=total > 0)


def weighted_ranks(values, weights) -> np.ndarray:
    """Average ranks of each row when row i appears weights[b, i] times, for every replicate b."""
    unique, inverse = np.unique(values, return_inverse=True)
    order = np.argsort(inverse, kind="stable")
    starts = np.searchsorted(inverse[order], np.arange(len(unique)))
    level = np.add.reduceat(weights[:, order], starts, axis=1)
    below = np.cumsum(level, axis=1) - level
    return (below + (level + 1) / 2)[:, inverse]


def weighted_corr(x, y, w) -> np.ndarray:
    total = w.sum(axis=1)
    mx, my = (w * x).sum(axis=1) / total, (w * y).sum(axis=1) / total
    cov = (w * x * y).sum(axis=1) / total - mx * my
    vx, vy = (w * x * x).sum(axis=1) / total - mx ** 2, (w * y * y).sum(axis=1) / total - my ** 2
    ok = (vx > 1e-12) & (vy > 1e-12)
    return np.divide(cov, np.sqrt(np.where(ok, vx * vy, 1)), out=np.full(len(w), np.nan), where=ok)


def spearman_boot(boot: Boot, x, y, rows) -> np.ndarray:
    """Spearman correlation over some rows in every replicate."""
    x, y = np.asarray(x, dtype=float)[rows], np.asarray(y, dtype=float)[rows]
    out = np.empty(boot.n)
    for i in range(0, boot.n, 500):
        w = boot.weights[i:i + 500][:, boot.group[rows]].astype(float)
        out[i:i + 500] = weighted_corr(weighted_ranks(x, w), weighted_ranks(y, w), w)
    return out


def kappa_from_counts(n11, n10, n01, n00):
    """Cohen's kappa of two binary vectors from their 2x2 counts (arrays allowed); nan when it is undefined."""
    n11, n10, n01, n00 = (np.asarray(v, dtype=float) for v in (n11, n10, n01, n00))
    n = n11 + n10 + n01 + n00
    n = np.where(n > 0, n, np.nan)
    po = (n11 + n00) / n
    px, py = (n11 + n10) / n, (n11 + n01) / n
    pe = px * py + (1 - px) * (1 - py)
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(pe >= 1 - 1e-12, np.nan, (po - pe) / (1 - pe))


def kappa_boot(boot: Boot, x, y, rows_mask) -> np.ndarray:
    x, y, m = np.asarray(x, dtype=bool), np.asarray(y, dtype=bool), np.asarray(rows_mask, dtype=bool)
    return kappa_from_counts(boot.sums(m & x & y), boot.sums(m & x & ~y), boot.sums(m & ~x & y), boot.sums(m & ~x & ~y))


# ---------------------------------------------------------------- thresholds

def fast_pair_thresholds(ca, cb, pair_tar: float = RULES.pair_tar) -> tuple[float, float, float]:
    """Exactly analysis.metrics.pair_thresholds, found by bisection: lowering both thresholds can only accept
    more, so the pair's TAR does not fall along the grid. Used inside resampling loops."""
    sa, sb = np.sort(ca)[::-1], np.sort(cb)[::-1]
    n = len(sa)
    grid = np.linspace(pair_tar, 1.0, 2001)
    ks = np.clip(np.ceil(grid * n).astype(int) - 1, 0, n - 1)
    lo, hi = 0, len(grid) - 1
    while lo < hi:
        mid = (lo + hi) // 2
        if np.mean((ca >= sa[ks[mid]]) & (cb >= sb[ks[mid]])) >= pair_tar:
            hi = mid
        else:
            lo = mid + 1
    return float(sa[ks[lo]]), float(sb[ks[lo]]), float(grid[lo])


def accepts(x, y, thresholds) -> np.ndarray:
    """AND rule: the pair accepts a candidate only if both verifiers accept it."""
    return (np.asarray(x) >= thresholds[0]) & (np.asarray(y) >= thresholds[1])


# ---------------------------------------------------------------- matching check and decision table (PLAN §5)

def matching_check(tar_a: float, tar_b: float, rules=RULES) -> dict:
    """Step 7: both pairs' test TAR within the limit of 90% and within the limit of each other, point estimates."""
    off = [abs(100 * t - 100 * rules.pair_tar) for t in (tar_a, tar_b)]
    gap = abs(100 * tar_a - 100 * tar_b)
    return {"off_target_points": off, "gap_points": gap,
            "limits_points": {"off_target": rules.match_tar_points, "gap": rules.match_gap_points},
            "passes": bool(max(off) <= rules.match_tar_points + TOL and gap <= rules.match_gap_points + TOL)}


VERDICTS = {
    "not_matched": "Inconclusive (not matched); trade-off curves shown",
    "better_meaningful": "AR+diffusion pair better, by a meaningful margin",
    "better_small": "Better, but by less than the meaningful margin",
    "no_meaningful_benefit": "No meaningful benefit (equivalence)",
    "worse": "AR+diffusion pair worse",
    "inconclusive": "Inconclusive; report the interval and the smallest effect the data could have detected",
}


def decide(matched: bool, delta: float, ci95, ci90, rules=RULES) -> str:
    """The decision rule of PLAN.md §5, rows in order, the first that applies. Everything in points of FAR;
    delta = FAR(AR+diffusion) - FAR(AR+AR), negative favors the mixed pair. Returns a key of VERDICTS."""
    values = [delta, *ci95, *ci90]
    if not all(np.isfinite(values)):
        raise ValueError(f"the decision needs finite numbers, got {values}")
    below, above = ci95[1] < -TOL, ci95[0] > TOL
    if not matched:
        return "not_matched"
    if below and delta <= -rules.margin_points + TOL:
        return "better_meaningful"
    if below:
        return "better_small"
    if ci90[0] > -rules.margin_points + TOL and ci90[1] < rules.margin_points - TOL:
        return "no_meaningful_benefit"
    if above:
        return "worse"
    return "inconclusive"


def detectable(se: float, rules=RULES) -> float:
    """Smallest true effect found with the planned power by a 5% two-sided test, given its standard error."""
    z = NormalDist().inv_cdf
    return float((z(0.975) + z(rules.power)) * se)
