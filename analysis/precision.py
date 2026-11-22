"""Pilot precision check (PLAN.md §6) and the matching-check simulation, recomputed with committed code.

    python -m analysis.precision --out results/final/precision

Part 1, precision: on the pilot's wrong answers, how often each AR+diffusion pair and each AR+AR pair of the final
test disagree (one accepts, the other does not, at in-sample 90% pair TAR), how much answers to one problem cluster
(design effect), and the smallest ΔFAR the planned test set can detect with 80% power at 5% two-sided.

Part 2, matching: how often both selected pairs would pass the matching check on test, under several limits. Each run
draws whole pilot problems with replacement until it has the validation pairs, and again for the test pairs, selects
the two pairs and freezes the thresholds on the validation draw, and looks at test TAR. A second version adds noise
to every score, because resampling repeats the same answers.

Pilot numbers are exploratory: these are planning estimates, not results.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from analysis.final_pilot import pilot_inputs
from analysis.final_settings import RULES
from analysis.final_stats import TOL, accepts, detectable, fast_pair_thresholds, matching_check
from analysis.final_validation import h1_groups
from analysis.metrics import auroc, pair_thresholds

PLANNED_CANDIDATES = {"math": 2100, "code": 850}    # PLAN §2 and §6 (est.); two thirds are test, half of them wrong
VALIDATION_PAIRS, TEST_PAIRS, RUNS = 350, 700, 3000
NOISE_SD = 0.15                                     # in standard deviations of each verifier's scores
# (limit on |TAR - 90|, limit on the gap between the two pairs), in points; inf = no limit
MATCHING_LIMITS = ((3, 3), (4, 3), (5, 3), (6, 3), (float("inf"), 3), (4, 4), (5, 5))
GROUPS = ("ar_ar", "ar_diffusion")


# ---------------------------------------------------------------- part 1: precision

def design_effect(values, problems) -> float:
    """Variance of the mean when whole problems are sampled, over its variance with independent answers
    (linearized, problems of any size)."""
    v = np.asarray(values, dtype=float)
    residual = v - v.mean()
    _, inverse = np.unique(np.asarray(problems, dtype=str), return_inverse=True)
    iid = float((residual ** 2).sum())
    return float((np.bincount(inverse, weights=residual) ** 2).sum() / iid) if iid > 0 else float("nan")


def precision_rows(frame, settings, rules=RULES) -> list[dict]:
    """Every (AR+diffusion pair, AR+AR pair) combination of the final test on one domain of the pilot."""
    planned = round(PLANNED_CANDIDATES[frame.domain] * 2 / 3 / 2)
    wrong = np.flatnonzero(frame.wrong)
    problems = [frame.problems[i] for i in wrong]
    acc = {}
    for items in h1_groups(settings).values():
        for p in items:
            x, y = frame.member(p.a), frame.member(p.b)
            t = pair_thresholds(x[frame.correct], y[frame.correct], rules.pair_tar)[:2]
            acc[p.name] = accepts(x, y, t)[wrong]
    out = []
    for mixed in h1_groups(settings)["ar_diffusion"]:
        for ar in h1_groups(settings)["ar_ar"]:
            d = acc[mixed.name].astype(int) - acc[ar.name].astype(int)
            disagree, deff = float(np.mean(d != 0)), design_effect(d, problems)
            # Under no difference the variance of one answer's difference is the disagreement rate. A design effect
            # below 1 in 125 answers is not taken as a gain.
            se = float(np.sqrt(disagree * max(deff, 1.0) / planned))
            out.append({"domain": frame.domain, "ar_diffusion": mixed.name, "ar_ar": ar.name, "n_wrong_pilot": int(len(wrong)),
                        "disagree": disagree, "design_effect": deff, "pilot_delta_far_points": float(100 * d.mean()),
                        "planned_wrong_test": planned, "detectable_points": 100 * detectable(se, rules)})
    return out


# ---------------------------------------------------------------- part 2: matching check

class PairDraws:
    """Draws of whole problems until a number of matched pairs is reached."""

    def __init__(self, frame):
        _, inverse = np.unique(np.asarray(frame.problems, dtype=str), return_inverse=True)
        self.rows = [np.flatnonzero(inverse == g) for g in range(inverse.max() + 1)]
        self.pairs = np.array([len(r) // 2 for r in self.rows])

    def draw(self, rng, pairs: int) -> np.ndarray:
        out, have = [], 0
        while have < pairs:
            g = int(rng.integers(len(self.rows)))
            out.append(self.rows[g])
            have += self.pairs[g]
        return np.concatenate(out)


def choose(scores: dict, labels, items, rules):
    """The group's pair with the lowest FAR at 90% pair TAR and its thresholds (ties as in the final test)."""
    c, w = labels == 1, labels == 0
    found = []
    for p in items:
        x, y = scores[p.a], scores[p.b]
        t = fast_pair_thresholds(x[c], y[c], rules.pair_tar)[:2]
        found.append((int(accepts(x[w], y[w], t).sum()), p, t))
    best = min(f[0] for f in found)
    tied = [f for f in found if f[0] == best]
    if len(tied) > 1:
        mean = {p.name: round((auroc(scores[p.a], labels) + auroc(scores[p.b], labels)) / 2, 12) for _, p, _ in tied}
        tied.sort(key=lambda f: (-mean[f[1].name], f[1].name))
    return tied[0][1], tied[0][2]


def simulate_matching(frame, settings, runs=RUNS, noise=0.0, seed=RULES.seed, rules=RULES,
                      validation_pairs=VALIDATION_PAIRS, test_pairs=TEST_PAIRS) -> dict:
    rng, draws, groups = np.random.default_rng(seed), PairDraws(frame), h1_groups(settings)
    members = sorted({m for items in groups.values() for p in items for m in p.members}, key=lambda m: m.label)
    base = {m: frame.member(m) for m in members}

    def sample(pairs):
        ix = draws.draw(rng, pairs)
        noisy = {m: x[ix] + (rng.normal(0, noise * x.std(), len(ix)) if noise else 0.0) for m, x in base.items()}
        return noisy, frame.labels[ix]

    tars = np.empty((runs, 2))
    for r in range(runs):
        val, val_labels = sample(validation_pairs)
        test, test_labels = sample(test_pairs)
        for k, g in enumerate(GROUPS):
            pair, t = choose(val, val_labels, groups[g], rules)
            tars[r, k] = accepts(test[pair.a], test[pair.b], t)[test_labels == 1].mean()
    off = np.abs(100 * tars - 100 * rules.pair_tar).max(axis=1)
    gap = np.abs(100 * tars[:, 0] - 100 * tars[:, 1])
    adopted = np.array([matching_check(a, b, rules)["passes"] for a, b in tars])
    fails = ~((off <= 3 + TOL) & (gap <= 3 + TOL))
    return {"noise_sd": noise, "runs": runs, "seed": seed, "validation_pairs": validation_pairs, "test_pairs": test_pairs,
            "pass_rate": [{"off_target_points": a if np.isfinite(a) else None, "gap_points": b, "passes": float(np.mean((off <= a + TOL) & (gap <= b + TOL)))}
                          for a, b in MATCHING_LIMITS],
            "adopted_rule_passes": float(adopted.mean()),
            "monte_carlo_se": float(np.sqrt(adopted.mean() * (1 - adopted.mean()) / runs)),
            "mean_tar": dict(zip(GROUPS, (100 * tars.mean(axis=0)).tolist())),
            "sd_tar_points": dict(zip(GROUPS, (100 * tars.std(axis=0)).tolist())),
            "sd_gap_points": float((100 * (tars[:, 0] - tars[:, 1])).std()),
            "share_of_3_3_failures_with_gap_above_3": float(np.mean(gap[fails] > 3 + TOL)) if fails.any() else float("nan")}


def far_slope(frame, settings, tars=(0.85, 0.87, 0.90, 0.93, 0.95)) -> dict:
    """In-sample FAR of each final-test pair at several pair TARs: how much a TAR mismatch moves FAR."""
    out = {}
    for items in h1_groups(settings).values():
        for p in items:
            x, y = frame.member(p.a), frame.member(p.b)
            out[p.name] = {f"{t:.2f}": float(accepts(x, y, pair_thresholds(x[frame.correct], y[frame.correct], t)[:2])[frame.wrong].mean())
                           for t in tars}
    return out


# ---------------------------------------------------------------- report

def markdown(result: dict) -> str:
    L = ["# Precision and matching check from the pilot", "",
         "Pilot numbers: exploratory planning estimates, made by `analysis/precision.py` (PLAN.md §6). Not H1 results.", ""]
    for domain, rows in result["precision"].items():
        lo, hi = (f([r["disagree"] for r in rows]) for f in (min, max))
        dlo, dhi = (f([r["design_effect"] for r in rows]) for f in (min, max))
        mlo, mhi = (f([r["detectable_points"] for r in rows]) for f in (min, max))
        L += [f"## {domain.capitalize()}: {rows[0]['planned_wrong_test']} wrong test answers planned "
              f"({PLANNED_CANDIDATES[domain]} candidates)", "",
              f"The two pairs disagree on {lo:.0%}–{hi:.0%} of the pilot's {rows[0]['n_wrong_pilot']} wrong answers; design effect "
              f"{dlo:.2f}–{dhi:.2f}; detectable ΔFAR **{mlo:.1f}–{mhi:.1f} points** (80% power, 5% two-sided).", "",
              "| AR+diffusion pair | AR+AR pair | Disagree | Design effect | Detectable ΔFAR (points) |", "|---|---|---|---|---|"]
        L += [f"| {r['ar_diffusion']} | {r['ar_ar']} | {r['disagree']:.1%} | {r['design_effect']:.2f} | {r['detectable_points']:.1f} |"
              for r in sorted(rows, key=lambda r: r["detectable_points"])]
        L.append("")
    sims = result["matching"]
    s = sims[0]
    L += [f"## Matching check: {s['runs']:,} runs, {s['validation_pairs']} validation and {s['test_pairs']} test pairs, whole problems, "
          f"seed {s['seed']}", "",
          "| Limit on TAR from 90% | Limit on the gap | Passes | Passes with score noise |",
          "|---|---|---|---|"]
    for plain, noisy in zip(sims[0]["pass_rate"], sims[1]["pass_rate"]):
        limit = "none" if plain["off_target_points"] is None else f"{plain['off_target_points']:g}"
        mark = " (adopted)" if (plain["off_target_points"], plain["gap_points"]) == (RULES.match_tar_points, RULES.match_gap_points) else ""
        L.append(f"| {limit} | {plain['gap_points']:g}{mark} | {plain['passes']:.1%} | {noisy['passes']:.1%} |")
    for s in sims:
        L += ["", f"Noise {s['noise_sd']:g} SD: test TAR SD {s['sd_tar_points']['ar_ar']:.1f} (AR+AR) and {s['sd_tar_points']['ar_diffusion']:.1f} "
              f"(AR+diffusion) points, gap SD {s['sd_gap_points']:.1f}; of the 3/3 failures, {s['share_of_3_3_failures_with_gap_above_3']:.0%} "
              f"break the gap limit. Monte Carlo error about {100 * s['monte_carlo_se']:.1f} points."]
    L += ["", "## FAR by pair TAR on the pilot (in-sample)", "", "| Pair | " + " | ".join(f"{float(t):.0%}" for t in next(iter(result["far_slope"].values()))) + " |",
          "|---|" + "---|" * len(next(iter(result["far_slope"].values())))]
    L += [f"| {name} | " + " | ".join(f"{v:.3f}" for v in row.values()) + " |" for name, row in result["far_slope"].items()]
    return "\n".join(L) + "\n"


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--out", required=True)
    ap.add_argument("--runs", type=int, default=RUNS)
    args = ap.parse_args(argv)
    settings, _, _, frames = pilot_inputs()
    result = {"status": "pilot, exploratory planning estimates", "settings": settings.to_dict(),
              "precision": {d: precision_rows(f, settings) for d, f in frames.items()},
              "matching": [simulate_matching(frames["math"], settings, args.runs, noise) for noise in (0.0, NOISE_SD)],
              "far_slope": far_slope(frames["math"], settings)}
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "precision.json").write_text(json.dumps(result, indent=1, default=float) + "\n", encoding="utf-8", newline="\n")
    (out / "precision.md").write_text(markdown(result), encoding="utf-8", newline="\n")
    print(markdown(result))


if __name__ == "__main__":
    main()
