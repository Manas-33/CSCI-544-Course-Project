"""H2 and E1: how much two verifiers' mistakes overlap, and whether a pair is below the competence trend.

On wrong answers, at the pair's frozen 90%-pair-TAR thresholds (PLAN.md §5, H2):
- Cohen's kappa on false accepts (which wrong answers each verifier accepts), with problem-bootstrap intervals;
- Spearman correlation of the two scores (no threshold);
- CAPA from the verdict probabilities, secondary, because it depends on calibration.

Competence check: across cross-family AR+AR pairs, kappa is fitted on the pair's mean AUROC. An AR+diffusion pair
counts as more diverse only below that line, with a 95% interval that excludes zero. Low overlap with a weak
verifier alone is not evidence. "On the trend" is only said when the distance to the line is shown to be small
(trend_status); an interval that merely contains zero is inconclusive. The same table is made within each error type.

Called by analysis.final_evaluate; nothing here reads files.
"""

from __future__ import annotations

import numpy as np

from analysis.final_settings import RULES
from analysis.final_stats import Boot, auroc_boot, interval, kappa_boot, kappa_from_counts, spearman_boot
from analysis.metrics import auroc, cohen_kappa, spearman

# H2 status of one AR+diffusion pair
BELOW, ON, ABOVE, INCONCLUSIVE = "below the AR trend", "on the AR trend", "above the AR trend", "inconclusive"
MIN_TREND_PAIRS = 3             # a line needs at least three AR pairs to say anything
MIN_FINITE_REPLICATES = 0.95    # kappa or the fit can be undefined in a replicate; more than 5% of them is too many


# ---------------------------------------------------------------- CAPA

def capa_terms(p1, p2, correct) -> tuple:
    """Per-item terms of CAPA: agreement sum_o p1(o) p2(o), p1(correct), p2(correct), 1 / (options - 1)."""
    p1, p2, correct = np.asarray(p1, dtype=float), np.asarray(p2, dtype=float), np.asarray(correct, dtype=int)
    if p1.shape != p2.shape or p1.ndim != 2 or p1.shape[1] < 2:
        raise ValueError("CAPA needs two (items, options) arrays with at least two options")
    rows = np.arange(len(p1))
    return (p1 * p2).sum(axis=1), p1[rows, correct], p2[rows, correct], np.full(len(p1), 1 / (p1.shape[1] - 1))


def capa_from_means(observed, q1, q2, inverse):
    """Combine the means of the CAPA terms (arrays allowed); nan where chance agreement is 1."""
    expected = q1 * q2 + (1 - q1) * (1 - q2) * inverse
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(np.abs(1 - expected) < 1e-12, np.nan, (observed - expected) / (1 - expected))


def capa(p1, p2, correct) -> float:
    """CAPA (chance adjusted probabilistic agreement), Goel et al., 2025, arXiv:2502.04313.

        kappa_p = (c_obs - c_exp) / (1 - c_exp)
        c_obs = mean over items of sum over options o of p1(o) p2(o)
        c_exp = q1 q2 + (1 - q1)(1 - q2) * mean over items of 1 / (number of options - 1)

    where p1 and p2 are the two models' probabilities over the options (rows sum to 1) and q_j is model j's mean
    probability on the correct option. Chance agreement comes from accuracy alone: the rest of each model's
    probability is spread evenly over the wrong options.

    Checked against the paper on 2026-11-21: arXiv:2502.04313v2, Section 2.2, Eqs. 2-4 were read and match this
    formula (kappa_p, c_obs, c_exp with 1 / (options - 1) averaged over items, and the two-option reduction).
    Appendix A.1 was not read word for word. With two options and 0/1 probabilities it is Cohen's kappa on which items each
    model gets right (error consistency), so on wrong answers it equals kappa on false accepts; the tests check
    that and hand-worked cases.
    """
    agree, q1, q2, inverse = capa_terms(p1, p2, correct)
    if len(agree) == 0:
        return float("nan")
    return float(capa_from_means(agree.mean(), q1.mean(), q2.mean(), inverse.mean()))


def verdict_probabilities(p_yes) -> np.ndarray:
    """(items, 2) array of P(No), P(Yes), renormalized to the two verdicts."""
    p = np.asarray(p_yes, dtype=float)
    return np.stack([1 - p, p], axis=1)


def capa_boot(boot: Boot, p_yes_1, p_yes_2, labels, rows_mask) -> tuple[float, np.ndarray]:
    """(CAPA, CAPA in every replicate) over some rows. The correct verdict is Yes (1) for a correct answer and
    No (0) for a wrong one."""
    m = np.asarray(rows_mask, dtype=bool)
    terms = capa_terms(verdict_probabilities(p_yes_1), verdict_probabilities(p_yes_2), labels)
    means = [boot.ratio(np.where(m, t, 0), m) for t in terms]
    point = capa_from_means(*(t[m].mean() for t in terms)) if m.any() else np.nan
    return float(point), capa_from_means(*means)


# ---------------------------------------------------------------- one pair

def kappa_row(frame, pair, thresholds, boot: Boot, mask) -> tuple[float, tuple]:
    """Kappa on false accepts among the wrong answers in mask (a source, an error type), with its interval."""
    w = frame.wrong & mask
    fa, fb = frame.member(pair.a) >= thresholds[0], frame.member(pair.b) >= thresholds[1]
    return cohen_kappa(fa[w], fb[w]), interval(kappa_boot(boot, fa, fb, w))


def overlap_row(frame, pair, thresholds, boot: Boot) -> dict:
    """Overlap of one pair on all wrong answers."""
    wrong = frame.wrong
    x, y = frame.member(pair.a), frame.member(pair.b)
    fa, fb = x >= thresholds[0], y >= thresholds[1]
    rows = np.flatnonzero(wrong)
    out = {"pair": pair.name, "kind": pair.kind, "roles": list(pair.roles), "n_wrong": int(len(rows)),
           "both_accept": int((fa & fb)[rows].sum()), "only_a": int((fa & ~fb)[rows].sum()),
           "only_b": int((~fa & fb)[rows].sum()), "neither": int((~fa & ~fb)[rows].sum())}
    out["kappa"] = cohen_kappa(fa[rows], fb[rows])
    out["kappa_ci"] = interval(kappa_boot(boot, fa, fb, wrong))
    out["spearman"] = spearman(x[rows], y[rows])
    out["spearman_ci"] = interval(spearman_boot(boot, x, y, rows))
    keys = [(m.verifier, m.column) for m in pair.members]
    if all(k in frame.p_yes for k in keys):
        pa, pb = (frame.p_yes[k] for k in keys)
        out["capa_wrong"], boots = capa_boot(boot, pa, pb, frame.labels, wrong)
        out["capa_wrong_ci"] = interval(boots)
        out["capa_all"], boots = capa_boot(boot, pa, pb, frame.labels, np.ones(len(frame.labels), dtype=bool))
        out["capa_all_ci"] = interval(boots)
    else:   # a reconstruction score is not a probability of a verdict
        out["capa_wrong"] = out["capa_all"] = None
    return out


# ---------------------------------------------------------------- competence trend

def fit_lines(x: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Least-squares (slope, intercept) of y on x for every row of two (replicates, pairs) arrays, ignoring pairs
    whose value is not finite. nan where fewer than MIN_TREND_PAIRS are left or all x are equal."""
    ok = np.isfinite(x) & np.isfinite(y)
    n = ok.sum(axis=1)
    x0, y0 = np.where(ok, x, 0), np.where(ok, y, 0)
    with np.errstate(invalid="ignore", divide="ignore"):
        mx, my = x0.sum(axis=1) / n, y0.sum(axis=1) / n
        sxx = np.where(ok, (x0 - mx[:, None]) ** 2, 0).sum(axis=1)
        sxy = np.where(ok, (x0 - mx[:, None]) * (y0 - my[:, None]), 0).sum(axis=1)
        usable = (n >= MIN_TREND_PAIRS) & (sxx > 1e-12)
        slope = np.where(usable, sxy / np.where(usable, sxx, 1), np.nan)
    return slope, my - slope * mx


def auroc_tables(frame, pairs, boot: Boot) -> tuple[dict, dict]:
    """(AUROC, AUROC in every replicate) of every member of some pairs."""
    members = sorted({m for p in pairs for m in p.members}, key=lambda m: m.label)
    return ({m: auroc(frame.member(m), frame.labels) for m in members},
            {m: auroc_boot(boot, frame.member(m), frame.labels) for m in members})


def trend_status(residual: float, replicates, margin: float) -> str:
    """Where a pair's kappa lies against the AR trend. "On the trend" is a statement that the distance is small, so it
    needs the 90% interval inside +-margin, as the equivalence row of the decision rule does for FAR; an interval that
    only fails to exclude zero is inconclusive (PLAN.md §8; decisions/2026-11-21-mechanism-wording.md)."""
    if not np.isfinite(residual) or float(np.isfinite(replicates).mean()) < MIN_FINITE_REPLICATES:
        return INCONCLUSIVE
    (lo95, hi95), (lo90, hi90) = interval(replicates), interval(replicates, 0.90)
    if hi95 < 0:
        return BELOW
    if lo95 > 0:
        return ABOVE
    return ON if lo90 > -margin and hi90 < margin else INCONCLUSIVE


def competence_trend(frame, ar_pairs, target_pairs, thresholds, boot: Boot, rules=RULES, aucs=None, mask=None) -> dict:
    """Fit kappa on mean AUROC over cross-family AR pairs and place each target (AR+diffusion) pair against it.

    thresholds: pair name -> (t_a, t_b), the pair's frozen 90% thresholds. Everything that varies with the sample
    (kappa, AUROC, the fitted line) is recomputed in every replicate. mask: kappa on only some wrong answers (an
    error type); AUROC always uses all answers. aucs: auroc_tables() of the pairs, when already computed."""
    pairs = list(ar_pairs) + list(target_pairs)
    auc, auc_b = aucs or auroc_tables(frame, pairs, boot)
    w = frame.wrong if mask is None else frame.wrong & mask
    kappa, kappa_b = {}, {}
    for p in pairs:
        fa, fb = frame.member(p.a) >= thresholds[p.name][0], frame.member(p.b) >= thresholds[p.name][1]
        kappa[p.name] = float(kappa_from_counts((w & fa & fb).sum(), (w & fa & ~fb).sum(), (w & ~fa & fb).sum(),
                                                (w & ~fa & ~fb).sum()))
        kappa_b[p.name] = kappa_boot(boot, fa, fb, w)

    def mean_auc(p, table):
        return (table[p.a] + table[p.b]) / 2

    x = np.array([mean_auc(p, auc) for p in ar_pairs], dtype=float)
    y = np.array([kappa[p.name] for p in ar_pairs], dtype=float)
    slope, intercept = (float(v[0]) for v in fit_lines(x[None, :], y[None, :])) if len(ar_pairs) else (np.nan, np.nan)
    if len(ar_pairs):
        slope_b, intercept_b = fit_lines(np.stack([mean_auc(p, auc_b) for p in ar_pairs], axis=1),
                                         np.stack([kappa_b[p.name] for p in ar_pairs], axis=1))
    else:
        slope_b = intercept_b = np.full(boot.n, np.nan)
    rows = []
    for p in pairs:
        residual = kappa[p.name] - (intercept + slope * mean_auc(p, auc))
        residual_b = kappa_b[p.name] - (intercept_b + slope_b * mean_auc(p, auc_b))
        rows.append({"pair": p.name, "kind": p.kind, "roles": list(p.roles), "in_fit": p in ar_pairs,
                     "mean_auroc": float(mean_auc(p, auc)), "kappa": kappa[p.name],
                     "kappa_ci": interval(kappa_b[p.name]), "residual": float(residual),
                     "residual_ci": interval(residual_b), "residual_ci90": interval(residual_b, 0.90),
                     "finite_replicates": float(np.isfinite(residual_b).mean()),
                     "status": None if p in ar_pairs else trend_status(residual, residual_b, rules.h2_on_trend_kappa)})
    return {"slope": slope, "slope_ci": interval(slope_b), "intercept": intercept, "n_ar_pairs": len(ar_pairs),
            "n_wrong": int(w.sum()), "on_trend_margin": rules.h2_on_trend_kappa,
            "auroc": {m.label: float(a) for m, a in auc.items()}, "pairs": rows}


def status_of(trend: dict, pair_name: str) -> str:
    for row in trend["pairs"]:
        if row["pair"] == pair_name:
            return row["status"] or INCONCLUSIVE
    return INCONCLUSIVE


# ---------------------------------------------------------------- E1

def e1_rows(frame, pairs, anchor_pair, thresholds, boot: Boot) -> list[dict]:
    """Qwen2.5-7B + Dream against Qwen2.5-7B with each other partner: FAR and kappa, with paired differences
    (the other pair minus Qwen2.5-7B + Dream)."""
    def parts(p):
        ta, tb = thresholds[p.name]
        fa, fb = frame.member(p.a) >= ta, frame.member(p.b) >= tb
        w = frame.wrong
        return (fa & fb & w), cohen_kappa(fa[w], fb[w]), kappa_boot(boot, fa, fb, w)

    base_acc, base_kappa, base_kappa_b = parts(anchor_pair)
    n_wrong, n_wrong_b = int(frame.wrong.sum()), boot.sums(frame.wrong)
    out = []
    for p in pairs:
        acc, k, k_b = parts(p)
        far_b = boot.sums(acc) / n_wrong_b
        out.append({"pair": p.name, "kind": p.kind, "far": float(acc.sum() / n_wrong), "far_ci": interval(far_b),
                    "kappa": k, "kappa_ci": interval(k_b),
                    "far_minus_dream": float((int(acc.sum()) - int(base_acc.sum())) / n_wrong),
                    "far_minus_dream_ci": interval((boot.sums(acc) - boot.sums(base_acc)) / n_wrong_b),
                    "kappa_minus_dream": float(k - base_kappa), "kappa_minus_dream_ci": interval(k_b - base_kappa_b)})
    return out

