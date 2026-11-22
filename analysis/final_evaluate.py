"""Test step (PLAN.md §5 "On test", run once): apply the frozen thresholds and pairs to the test split.

    python -m analysis.final_evaluate --manifest results/final/analysis_manifest.json \\
        --test-labels math=bank/full/math/labels_test.jsonl code=bank/full/code/labels_test.jsonl \\
        --out results/final/test

Before any label is parsed, in this order: the manifest, the settings, the final model record and the analysis code
must be committed and unchanged; every hash in the manifest must match and the validation step must recompute to the
same manifest; the banks, scores and validation labels must load; and the join log must not hold an earlier join with
another manifest or other labels. The join is then written to the log (results/final/join_log.jsonl, the only one),
and only after that are the labels read. Commit the log right after. The same command may be run again to reproduce the
numbers (logged as a rerun, in another --out folder); anything that differs is refused.

Nothing is tuned here. Thresholds, pairs and weights come from the manifest. The only test-tuned numbers are in the
row "ideal thresholds (test-tuned)", which no verdict uses.
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
from dataclasses import replace
from pathlib import Path

import numpy as np

from analysis.final_data import (VALIDATION_LABELS, DataError, Frame, Scores, align_labels, load_banks, load_scores,
                                 load_validation_labels, make_frame, read_jsonl, sha256_file)
from analysis.final_settings import (E1_ANCHOR, E1_DIFFUSION, H1_DOMAIN, RULES, VERIFIERS, Member, Settings,
                                     final_test_members, make_pair, needed_scores, parse_settings, plan_pairs,
                                     pseudo_pair_name, same_verifiers, with_role)
from analysis.final_stats import VERDICTS, Boot, accepts, auroc_boot, decide, detectable, interval, matching_check
from analysis.final_validation import (CODE_FILES, JOIN_LOG, JOINS, NotFrozen, all_members, check_manifest, git, gpu_types,
                                       h1_groups, pair_seconds, read_log, resample_selection, tar_key, validation_boot)
from analysis.metrics import auroc, ece, pair_thresholds
from analysis.overlap import (ABOVE, BELOW, INCONCLUSIVE, ON, auroc_tables, competence_trend, e1_rows, kappa_row,
                              overlap_row, status_of)

TEST_LABELS = "labels_test.jsonl"


# ---------------------------------------------------------------- the gate in front of the test labels

def require_committed(root, paths) -> str:
    """Every path must be tracked and identical to HEAD. Returns the HEAD commit."""
    errors = []
    for p in paths:
        if git(root, "ls-files", "--error-unmatch", "--", p).returncode:
            errors.append(f"{p}: not committed")
        elif git(root, "diff", "--quiet", "HEAD", "--", p).returncode:
            errors.append(f"{p}: differs from the committed version")
    head = git(root, "rev-parse", "HEAD")
    if head.returncode:
        errors.append("no commit found")
    if errors:
        raise NotFrozen("refusing to join test labels (PLAN.md §5 step 5):\n  " + "\n  ".join(errors))
    return head.stdout.strip()


def write_log(root, entry: dict) -> None:
    path = Path(root) / JOIN_LOG
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a", encoding="utf-8", newline="\n") as f:
        f.write(json.dumps(entry, sort_keys=True) + "\n")


def now() -> str:
    return dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds")


def join_event(log_entries: list[dict], manifest_sha: str, label_shas: dict) -> str:
    """"join" the first time, "rerun" when the same manifest and labels were already joined; refuses anything else."""
    event = "join"
    for entry in (e for e in log_entries if e.get("event") in JOINS):
        for domain, label in entry["labels"].items():
            if domain not in label_shas:
                continue
            if entry["manifest_sha256"] != manifest_sha or label["sha256"] != label_shas[domain]:
                raise NotFrozen(f"{domain} test labels were already joined on {entry['utc']} with another manifest or "
                                "other labels; they are joined once (PLAN.md §2, label custody)")
            event = "rerun"
    return event


def load_test_labels(path, bank) -> tuple[np.ndarray, np.ndarray]:
    """(labels, error types) of the test split in bank order. Only run() calls this, after the gate above. Error types
    (code: from the execution logs, math: hand tags) are in the label rows."""
    rows = read_jsonl(path)
    labels = align_labels(rows, bank, "test", str(path))
    types = {r["candidate_id"]: str(r.get("error_type") or "") for r in rows}
    return labels, np.array([types[c] for c in bank.ids("test")], dtype=object)


# ---------------------------------------------------------------- rows of the main table

def pair_accepts(frame: Frame, pair, thresholds) -> np.ndarray:
    return accepts(frame.member(pair.a), frame.member(pair.b), thresholds)


def rates(frame: Frame, acc: np.ndarray, boot: Boot) -> dict:
    """TAR and FAR with counts and 95% problem-bootstrap intervals."""
    c, w = frame.correct, frame.wrong
    return {"n_correct": int(c.sum()), "accepted_correct": int((acc & c).sum()),
            "tar": float((acc & c).sum() / c.sum()), "tar_ci": interval(boot.ratio(acc & c, c)),
            "n_wrong": int(w.sum()), "accepted_wrong": int((acc & w).sum()),
            "far": float((acc & w).sum() / w.sum()), "far_ci": interval(boot.ratio(acc & w, w))}


def delta_far(frame: Frame, acc_x, acc_y, boot: Boot, rules, mask=None) -> dict:
    """FAR(x) - FAR(y) in points, on the same wrong answers, with paired problem-bootstrap intervals."""
    w = frame.wrong if mask is None else frame.wrong & mask
    n = int(w.sum())
    if n == 0:
        return {"n_wrong": 0, "points": float("nan"), "ci95": (float("nan"),) * 2, "ci90": (float("nan"),) * 2}
    difference = int((acc_x & w).sum()) - int((acc_y & w).sum())
    n_b = boot.sums(w)
    reps = np.divide(100 * (boot.sums(acc_x & w) - boot.sums(acc_y & w)), n_b, out=np.full(boot.n, np.nan), where=n_b > 0)
    se = float(np.nanstd(reps, ddof=1)) if boot.n > 1 else float("nan")
    return {"n_wrong": n, "points": 100 * difference / n, "ci95": interval(reps, 0.95), "ci90": interval(reps, 0.90),
            "se": se, "detectable_points": detectable(se, rules)}


def find_pair(settings: Settings, domain: str, name: str):
    for p in plan_pairs(settings, domain):
        if p.name == name:
            return p
    raise KeyError(f"{name} is not a pair of this plan on {domain}")


def frozen(validation: dict, domain: str, name: str, tar: float) -> tuple[float, float]:
    e = validation["thresholds"][domain][name][tar_key(tar)]
    return e["a"], e["b"]


# ---------------------------------------------------------------- the final test

def h1_test(frame: Frame, val_frame: Frame, validation: dict, settings: Settings, boot: Boot, rules) -> dict:
    domain, chosen = frame.domain, validation["h1"]["selected"]
    pairs = {g: find_pair(settings, domain, name) for g, name in chosen.items()}
    out = {"test_name": settings.test_name, "domain": domain, "pairs": chosen, "by_tar": {}}
    for tar in rules.tars:
        acc = {g: pair_accepts(frame, p, frozen(validation, domain, p.name, tar)) for g, p in pairs.items()}
        rows = {g: {**rates(frame, a, boot), "thresholds": list(frozen(validation, domain, pairs[g].name, tar)),
                    "seconds": validation["h1"]["cost"]["seconds_per_candidate"][g],
                    "gpu_types": validation["h1"]["cost"]["gpu_types"]} for g, a in acc.items()}
        c = frame.correct
        gap_b = boot.ratio(acc["ar_diffusion"] & c, c) - boot.ratio(acc["ar_ar"] & c, c)
        out["by_tar"][tar_key(tar)] = {"rows": rows, "delta_far": delta_far(frame, acc["ar_diffusion"], acc["ar_ar"], boot, rules),
                                       "tar_gap_points": 100 * (rows["ar_diffusion"]["tar"] - rows["ar_ar"]["tar"]),
                                       "tar_gap_ci": [100 * v for v in interval(gap_b)]}
    main = out["by_tar"][tar_key(rules.pair_tar)]
    match = matching_check(main["rows"]["ar_diffusion"]["tar"], main["rows"]["ar_ar"]["tar"], rules)
    d = main["delta_far"]
    key = decide(match["passes"], d["points"], d["ci95"], d["ci90"], rules)
    old = matching_check(main["rows"]["ar_diffusion"]["tar"], main["rows"]["ar_ar"]["tar"],
                         replace(rules, match_tar_points=rules.old_match_tar_points))
    old["row"] = decide(old["passes"], d["points"], d["ci95"], d["ci90"], rules)
    out.update(matching=match, matching_old_rule=old, verdict=key, verdict_text=VERDICTS[key],
               detectable_points=d["detectable_points"],
               comparable_cost_claim=validation["h1"]["cost"]["comparable_cost_claim"],
               llada_below_bar_on_validation=validation["h1"]["llada_recheck"]["below_bar"])
    out["selection_bootstrap"] = selection_bootstrap(frame, val_frame, validation, settings, boot, rules)
    out["pseudo_check"] = {g: pseudo_check(frame, p, validation, settings, boot, rules) for g, p in pairs.items()}
    out["ideal_thresholds_test_tuned"] = ideal_thresholds(frame, pairs, boot, rules)
    out["budget_matched"] = budget_matched(frame, pairs, validation, settings, boot, rules)
    out["tradeoff"] = tradeoff(frame, validation, settings)
    return out


def selection_bootstrap(frame: Frame, val_frame: Frame, validation: dict, settings: Settings, boot: Boot, rules) -> dict:
    """Secondary (step 8): repeat the validation-stage selection inside the bootstrap. Each replicate resamples
    validation problems and test problems. "choice_only" keeps the frozen thresholds of whichever pair wins;
    "choice_and_thresholds" also uses the thresholds found on that validation resample."""
    domain, groups = frame.domain, h1_groups(settings)
    resampled = resample_selection(val_frame, groups, validation_boot(val_frame, rules), rules.pair_tar)
    w, n_b = frame.wrong, boot.sums(frame.wrong)
    rows = np.flatnonzero(w)
    item_group = boot.group[rows]
    choice, full = {}, {}
    for g, items in groups.items():
        per_pair = np.stack([boot.sums(w & pair_accepts(frame, p, frozen(validation, domain, p.name, rules.pair_tar)))
                             for p in items], axis=1)
        winner = resampled[g]["winner"]
        choice[g] = per_pair[np.arange(boot.n), winner]
        scores = [(frame.member(p.a)[rows], frame.member(p.b)[rows]) for p in items]
        full[g] = np.array([boot.weights[b][item_group[accepts(*scores[k], t)]].sum()
                            for b, (k, t) in enumerate(zip(winner, resampled[g]["thresholds"]))], dtype=float)
    out = {"winners": {g: {p.name: float(np.mean(resampled[g]["winner"] == k)) for k, p in enumerate(items)}
                       for g, items in groups.items()}}
    for name, table in (("choice_only", choice), ("choice_and_thresholds", full)):
        reps = 100 * (table["ar_diffusion"] - table["ar_ar"]) / n_b
        out[name] = {"mean_points": float(reps.mean()), "ci95": interval(reps), "ci90": interval(reps, 0.90)}
    return out


def pseudo_check(frame: Frame, pair, validation: dict, settings: Settings, boot: Boot, rules) -> dict:
    """Does the pair accept fewer wrong answers than the T1+T2 pseudo-pair of its stronger member, at the same 90%
    pair TAR? Only then does a gain count as diversity (PLAN.md §5). "beats" needs a lower FAR (point estimate, with
    the interval reported beside it) and the two matched on test by the same limits as the final test
    (decisions/2026-11-21-mechanism-wording.md)."""
    info = validation["pseudo_of_stronger"][frame.domain][pair.name]
    if info["pseudo_pair"] is None:
        return {**info, "available": False, "beats": None}
    other = find_pair(settings, frame.domain, info["pseudo_pair"])
    acc = pair_accepts(frame, pair, frozen(validation, frame.domain, pair.name, rules.pair_tar))
    ref = pair_accepts(frame, other, frozen(validation, frame.domain, other.name, rules.pair_tar))
    d, c = delta_far(frame, acc, ref, boot, rules), frame.correct
    match = matching_check(float(acc[c].mean()), float(ref[c].mean()), rules)
    return {**info, "available": True, "delta_far": d, "lower_far": bool(d["points"] < 0), "matching": match,
            "beats": bool(d["points"] < 0 and match["passes"]), "interval_below_zero": bool(d["ci95"][1] < 0),
            "pair_tar": float(acc[c].mean()), "pseudo_tar": float(ref[c].mean())}


def ideal_thresholds(frame: Frame, pairs: dict, boot: Boot, rules) -> dict:
    """The same two pairs with thresholds tuned on test. Shows what frozen thresholds cost; no verdict uses it."""
    c, acc, rows = frame.correct, {}, {}
    for g, p in pairs.items():
        t = pair_thresholds(frame.member(p.a)[c], frame.member(p.b)[c], rules.pair_tar)[:2]
        acc[g] = pair_accepts(frame, p, t)
        rows[g] = {**rates(frame, acc[g], boot), "thresholds": list(t)}
    return {"rows": rows, "delta_far": delta_far(frame, acc["ar_diffusion"], acc["ar_ar"], boot, rules), "test_tuned": True}


def budget_matched(frame: Frame, pairs: dict, validation: dict, settings: Settings, boot: Boot, rules) -> dict | None:
    """Only when the selected pairs' cost ratio is above 1.5: the cheaper pair averages T1 and T2 (step 4, secondary)."""
    cost = validation["h1"]["cost"]
    if not cost["budget_matched_secondary"]:
        return None
    used = dict(pairs)
    p = pairs[cost["cheaper"]]
    used[cost["cheaper"]] = make_pair(*(m if m.column == "recon" else Member(m.verifier, "avg") for m in p.members), ())
    acc = {g: pair_accepts(frame, q, frozen(validation, frame.domain, q.name, rules.pair_tar)) for g, q in used.items()}
    rows = {g: rates(frame, a, boot) for g, a in acc.items()}
    return {"cheaper": cost["cheaper"], "pairs": {g: q.name for g, q in used.items()}, "rows": rows,
            "delta_far": delta_far(frame, acc["ar_diffusion"], acc["ar_ar"], boot, rules),
            "matching": matching_check(rows["ar_diffusion"]["tar"], rows["ar_ar"]["tar"], rules)}


def tradeoff(frame: Frame, validation: dict, settings: Settings) -> dict:
    """Test (TAR, FAR) of every final-test pair along validation thresholds. Shown with a failed matching check."""
    out = {}
    for items in h1_groups(settings).values():
        for p in items:
            points = []
            for key, t in validation["tradeoff"][p.name].items():
                acc = pair_accepts(frame, p, t)
                points.append({"target": float(key), "tar": float(acc[frame.correct].mean()), "far": float(acc[frame.wrong].mean())})
            out[p.name] = points
    return out


# ---------------------------------------------------------------- reading of the outcome (PLAN §8)

READINGS = {
    ("better_meaningful", BELOW): (
        "Tested diffusion verifiers add complementarity beyond strong cross-family AR pairs",
        "For the tested models, crossing the AR/diffusion boundary bought useful verification diversity (correlational)"),
    ("better_meaningful", ON): (
        "A competent, roughly independent extra verifier, but no more different than an equally competent AR model",
        "Practical yes, mechanism no"),
    ("no_meaningful_benefit", BELOW): (
        "Different mistakes, but too weak to turn into fewer false accepts at 90% TAR",
        "Diversity without competence; revisit as diffusion models improve"),
    ("no_meaningful_benefit", ON): (
        "No useful diversity for these models, tasks and interface", "Clean negative, scoped to the tested configurations"),
    ("worse", None): ("Adding the diffusion verifier as a veto hurts", "Scoped advice: do not add these models as vetoes"),
    ("better_small", None): ("A real but small gain", "A small benefit for the tested pairs, with the interval"),
    ("inconclusive", None): ("Underpowered", "Estimate and interval only"),
    ("not_matched", None): ("The pairs are not matched on test", "Estimate and interval only; trade-off curves shown"),
    ("better_meaningful", None): ("The AR+diffusion pair accepts fewer wrong answers, by a meaningful margin",
                                  "H1 stands alone: no mechanism claim"),
    ("no_meaningful_benefit", None): ("No meaningful benefit at 90% TAR", "H1 stands alone: no mechanism claim"),
}


def reading(verdict: str, h2_status: str, below_bar: bool, pseudo_beats, cost_claim: bool, repair: bool) -> dict:
    """Row of PLAN.md §8 for an H1 verdict and the H2 status of the selected AR+diffusion pair. An H2 result above
    the trend is read as "on the trend"; with H2 inconclusive, or LLaDA below gate A's bar on validation, H1 stands
    alone. A gain counts as diversity only if the pair beats the pseudo-pair of its stronger member (§5), so the
    diversity claim also needs that."""
    status = ON if h2_status == ABOVE else h2_status
    mechanism = status in (BELOW, ON) and not below_bar
    if verdict == "better_meaningful" and status == BELOW and not pseudo_beats:
        mechanism = False
    key = (verdict, status) if mechanism and (verdict, status) in READINGS else (verdict, None)
    text, claim = READINGS[key]
    caveats = []
    if repair:
        caveats.append("AR verdict versus diffusion reconstruction score; no comparable-cost claim")
    elif not cost_claim:
        caveats.append("no comparable-cost claim: the pairs' measured cost is not within 1.5x on the same GPU type")
    if below_bar:
        caveats.append("LLaDA is below gate A's bar on validation: no claim about diversity is made from H2")
    if h2_status == INCONCLUSIVE:
        caveats.append("H2 is inconclusive: no mechanism claim")
    if pseudo_beats is False:
        caveats.append("the AR+diffusion pair does not beat the T1+T2 pseudo-pair of its stronger member at a matched "
                       "TAR, so a gain does not count as diversity")
    elif pseudo_beats is None:
        caveats.append("no pseudo-pair for the stronger member (its score has no second wording), so the "
                       "'beats the pseudo-pair' condition is not checked")
    return {"h1": verdict, "h2": h2_status, "reading": text, "claim": claim, "mechanism_claim": key[1] is not None,
            "caveats": caveats}


# ---------------------------------------------------------------- everything else that is reported

def pair_rows(frame: Frame, validation: dict, settings: Settings, scores: Scores, boot: Boot, rules) -> list[dict]:
    out = []
    for p in plan_pairs(settings, frame.domain):
        for tar in rules.tars:
            acc = pair_accepts(frame, p, frozen(validation, frame.domain, p.name, tar))
            out.append({"domain": frame.domain, "pair": p.name, "kind": p.kind, "roles": list(p.roles), "target_tar": tar,
                        **rates(frame, acc, boot), "seconds": pair_seconds(scores, frame.domain, p),
                        "gpu_types": gpu_types(scores, frame.domain, p.members), "median_mass": {m.label: float(np.median(frame.mass[(m.verifier, m.column)]))
                                        for m in p.members if (m.verifier, m.column) in frame.mass}})
    return out


def reliability(p: np.ndarray, labels: np.ndarray, bins: int = 10) -> list[dict]:
    idx = np.minimum((p * bins).astype(int), bins - 1)
    return [{"bin": b, "n": int((idx == b).sum()), "confidence": float(p[idx == b].mean()),
             "accuracy": float(labels[idx == b].mean())} for b in range(bins) if (idx == b).any()]


def roc_points(x: np.ndarray, labels: np.ndarray, steps: int = 101) -> list[list[float]]:
    """(FPR, TPR) at every distinct threshold, thinned to about `steps` points."""
    cuts = np.unique(x)[::-1]
    pos, neg = np.sort(x[labels == 1]), np.sort(x[labels == 0])
    tpr = 1 - np.searchsorted(pos, cuts, side="left") / len(pos)
    fpr = 1 - np.searchsorted(neg, cuts, side="left") / len(neg)
    keep = np.unique(np.linspace(0, len(cuts) - 1, min(steps, len(cuts))).astype(int))
    return [[0.0, 0.0]] + [[float(fpr[i]), float(tpr[i])] for i in keep]


def single_rows(frames: dict, validation: dict, settings: Settings, scores: Scores, boots: dict, pooled_boot, rules) -> list[dict]:
    """Discrimination, calibration and cost of every verifier, per domain and pooled."""
    out = []
    for domain, frame in frames.items():
        for v, c in needed_scores(settings, domain):
            m = Member(v, c)
            x, entry = frame.member(m), validation["singles"][domain][m.label]
            acc = x >= entry["threshold"]
            verdict = (v, c) in frame.p_yes
            out.append({"domain": domain, "label": m.label, "verifier": v, "column": c, "kind": VERIFIERS[v].kind,
                        "auroc": auroc(x, frame.labels), "auroc_ci": interval(auroc_boot(boots[domain], x, frame.labels)),
                        "ece": ece(x, frame.labels) if verdict else None,
                        "reliability": reliability(frame.p_yes[(v, c)], frame.labels) if verdict else None,
                        "median_mass": float(np.median(frame.mass[(v, c)])) if verdict else None,
                        "roc": roc_points(x, frame.labels) if c in ("T1", "recon") else None,
                        "seconds": scores.seconds(domain, v, c), "gpu": " / ".join(sorted(scores.gpus(domain, v, c))),
                        "gpu_types": gpu_types(scores, domain, [m]),
                        "at_90_tar": {**rates(frame, acc, boots[domain]), "threshold": entry["threshold"]}})
    if len(frames) > 1:
        order = list(frames)
        for v, c in needed_scores(settings, order[-1]):
            if any((v, c) not in f.scores for f in frames.values()) or c == "recon":
                continue
            x = np.concatenate([frames[d].scores[(v, c)] for d in order])
            y = np.concatenate([frames[d].labels for d in order])
            p = np.concatenate([frames[d].p_yes[(v, c)] for d in order])
            out.append({"domain": "pooled", "label": Member(v, c).label, "verifier": v, "column": c, "kind": VERIFIERS[v].kind,
                        "auroc": auroc(x, y), "auroc_ci": interval(auroc_boot(pooled_boot, x, y)), "ece": ece(x, y),
                        "reliability": reliability(p, y), "median_mass": None, "roc": roc_points(x, y) if c == "T1" else None,
                        "seconds": None, "gpu": None, "at_90_tar": None})
    return out


def pooled_rows(frames: dict, validation: dict, settings: Settings, weights: dict, pooled_boot: Boot, rules) -> list[dict]:
    """Secondary: pairs present in both domains, with each domain weighted by its number of wrong test answers
    (fixed by the bank before unblinding) and domains resampled separately."""
    if len(frames) < 2:
        return []
    order = list(frames)
    total = sum(weights[d] for d in order)
    sizes = [len(frames[d].rows) for d in order]
    starts = np.cumsum([0, *sizes])
    by_name = {d: {p.name: p for p in plan_pairs(settings, d)} for d in order}
    out, far_b = [], {}
    for name in [n for n in by_name[order[0]] if all(n in by_name[d] for d in order)]:
        row = {"domain": "pooled", "pair": name, "kind": by_name[order[0]][name].kind,
               "roles": list(by_name[H1_DOMAIN if H1_DOMAIN in by_name else order[0]][name].roles), "weights": weights}
        point, reps = {"tar": 0.0, "far": 0.0}, {"tar": 0.0, "far": 0.0}
        for i, d in enumerate(order):
            f = frames[d]
            acc = pair_accepts(f, by_name[d][name], frozen(validation, d, name, rules.pair_tar))
            for key, mask in (("tar", f.correct), ("far", f.wrong)):
                num, den = np.zeros(starts[-1]), np.zeros(starts[-1])
                num[starts[i]:starts[i + 1]], den[starts[i]:starts[i + 1]] = acc & mask, mask
                point[key] += weights[d] / total * (acc & mask).sum() / mask.sum()
                reps[key] = reps[key] + weights[d] / total * pooled_boot.ratio(num, den)
        far_b[name] = (point["far"], reps["far"])
        out.append({**row, "tar": float(point["tar"]), "tar_ci": interval(reps["tar"]), "far": float(point["far"]),
                    "far_ci": interval(reps["far"])})
    chosen = (validation.get("h1") or {}).get("selected")
    if chosen and all(n in far_b for n in chosen.values()):
        d, b = (far_b[chosen["ar_diffusion"]][k] - far_b[chosen["ar_ar"]][k] for k in (0, 1))
        out.append({"domain": "pooled", "pair": "selected AR+diffusion minus selected AR+AR", "kind": "difference",
                    "roles": [], "delta_far_points": float(100 * d), "ci95": [100 * v for v in interval(b)], "weights": weights})
    return out


def breakdown_rows(frame: Frame, pairs: dict, validation: dict, boot: Boot, rules) -> list[dict]:
    """FAR of the selected pairs by source, generator and error type (descriptive; small groups are marked)."""
    thresholds = {g: frozen(validation, frame.domain, p.name, rules.pair_tar) for g, p in pairs.items()}
    acc = {g: pair_accepts(frame, p, thresholds[g]) for g, p in pairs.items()}
    out = []
    groupings = {"all": np.full(len(frame.rows), "all", dtype=object), "source": frame.column("source"),
                 "generator": frame.column("generator")}
    if frame.error_types is not None:
        groupings["error_type"] = frame.error_types
    for grouping, values in groupings.items():
        for value in sorted(set(values[frame.wrong]) - {""}):
            m = values == value
            w, c = frame.wrong & m, frame.correct & m
            row = {"domain": frame.domain, "grouping": grouping, "group": value, "n_wrong": int(w.sum()), "n_correct": int(c.sum()),
                   "small": bool(w.sum() < rules.small_group)}
            for g in pairs:
                row[f"far_{g}"] = float(acc[g][w].mean())
                row[f"tar_{g}"] = float(acc[g][c].mean()) if c.any() and grouping != "error_type" else None
                row[f"kappa_{g}"], row[f"kappa_{g}_ci"] = kappa_row(frame, pairs[g], thresholds[g], boot, m)
            if set(pairs) == {"ar_ar", "ar_diffusion"}:
                d = delta_far(frame, acc["ar_diffusion"], acc["ar_ar"], boot, rules, mask=m)
                row.update(delta_far_points=d["points"], ci95=d["ci95"])
            out.append(row)
    return out


def h2_block(frame: Frame, validation: dict, settings: Settings, boot: Boot, rules) -> dict:
    """Overlap of every pair of the final test, E1 and the H2 trend, and the competence trend, overall and within each
    error type (descriptive; groups with fewer than rules.small_group wrong answers are only counted)."""
    pairs = [p for p in plan_pairs(settings, frame.domain) if p.kind != "pseudo" and "template_avg" not in p.roles]
    thresholds = {p.name: frozen(validation, frame.domain, p.name, rules.pair_tar) for p in pairs}
    trend_ar = [p for p in pairs if "h2_trend" in p.roles]
    targets = [p for p in pairs if p.kind == "AR+diffusion"]
    aucs = auroc_tables(frame, trend_ar + targets, boot)
    trend = competence_trend(frame, trend_ar, targets, thresholds, boot, rules, aucs)
    by_type = {}
    if frame.error_types is not None:
        for value in sorted(set(frame.error_types[frame.wrong]) - {""}):
            mask = frame.error_types == value
            n = int((frame.wrong & mask).sum())
            by_type[value] = {"n_wrong": n, "trend": competence_trend(frame, trend_ar, targets, thresholds, boot, rules, aucs, mask)
                              if n >= rules.small_group else None}
    return {"domain": frame.domain, "overlap": [overlap_row(frame, p, thresholds[p.name], boot) for p in pairs],
            "trend": trend, "by_error_type": by_type}


def e1_block(frame: Frame, validation: dict, settings: Settings, boot: Boot, rules) -> list[dict]:
    pairs = with_role(plan_pairs(settings, frame.domain), "e1")
    thresholds = {p.name: frozen(validation, frame.domain, p.name, rules.pair_tar) for p in pairs}
    anchor = next(p for p in pairs if {m.verifier for m in p.members} == {E1_ANCHOR, E1_DIFFUSION})
    return e1_rows(frame, pairs, anchor, thresholds, boot)


def reference_rows(frame: Frame, validation: dict, settings: Settings, scores: Scores, boot: Boot, rules) -> list[dict]:
    """The rows beside the final test in the main table, all at validation thresholds (PLAN.md §5)."""
    domain, h1 = frame.domain, validation["h1"]
    single = validation["singles"][domain][h1["best_single"]]
    member = Member(single["verifier"], single["column"])
    acc_single = frame.member(member) >= single["threshold"]
    chosen = {g: find_pair(settings, domain, n) for g, n in h1["selected"].items()}
    acc = {g: pair_accepts(frame, p, frozen(validation, domain, p.name, rules.pair_tar)) for g, p in chosen.items()}
    out = [{"row": "best single verifier (validation threshold)", "name": h1["best_single"], **rates(frame, acc_single, boot),
            "seconds": scores.seconds(domain, member.verifier, member.column), "gpu_types": gpu_types(scores, domain, [member]),
            "pair_minus_single": {g: delta_far(frame, a, acc_single, boot, rules) for g, a in acc.items()}}]
    for m in final_test_members(settings, domain):
        if m.column != "recon":
            name = pseudo_pair_name(m.verifier)
            p = find_pair(settings, domain, name)
            out.append({"row": "same-model pseudo-pair (T1 + T2)", "name": name,
                        "median_mass_t2": float(np.median(frame.mass[(m.verifier, "T2")])),
                        **rates(frame, pair_accepts(frame, p, frozen(validation, domain, name, rules.pair_tar)), boot),
                        "seconds": pair_seconds(scores, domain, p), "gpu_types": gpu_types(scores, domain, p.members)})
    for kind, name in (validation.get("qwen3") or {}).get(domain, {}).items():
        p = find_pair(settings, domain, name)
        out.append({"row": f"strong-baseline check, {kind}", "name": name,
                    **rates(frame, pair_accepts(frame, p, frozen(validation, domain, name, rules.pair_tar)), boot),
                    "seconds": pair_seconds(scores, domain, p), "gpu_types": gpu_types(scores, domain, p.members)})
    return out


def ar_family_block(frame: Frame, validation: dict, settings: Settings, boot: Boot, rules) -> dict:
    """The AR-family study (PLAN.md §7 pivot): FAR and overlap of the six pairs, and how far each gets from the best
    single verifier toward a perfect verifier, which accepts no wrong answer."""
    domain, info = frame.domain, validation["ar_family"][frame.domain]
    single = validation["singles"][domain][info["best_single"]]
    far_single = float(np.mean(frame.member(Member(single["verifier"], single["column"]))[frame.wrong] >= single["threshold"]))
    rows = []
    for name in info["pairs"]:
        p = find_pair(settings, domain, name)
        t = frozen(validation, domain, name, rules.pair_tar)
        r = rates(frame, pair_accepts(frame, p, t), boot)
        o = overlap_row(frame, p, t, boot)
        rows.append({"pair": name, "selected": name == info["selected"], **r, "kappa": o["kappa"], "kappa_ci": o["kappa_ci"],
                     "spearman": o["spearman"], "share_of_single_false_accepts_removed":
                     float(1 - r["far"] / far_single) if far_single > 0 else float("nan")})
    return {"domain": domain, "best_single": info["best_single"], "best_single_far": far_single, "perfect_verifier_far": 0.0,
            "pairs": sorted(rows, key=lambda r: r["far"])}


def side_runs(root, side: dict, banks: dict) -> tuple:
    """(thresholds, scores) of the reconstruction side experiment, loaded before the join so a bad file is found first."""
    from analysis.reconstruction import load_runs
    frozen = json.loads((Path(root) / side["file"]["path"]).read_text(encoding="utf-8"))
    return frozen, load_runs([Path(root) / p for p in side["scores"]], banks[side["domain"]].sha256)


def side_results(frozen: dict, runs, bank, labels, rules) -> dict:
    """The side experiment on its drawn test answers, at the frozen thresholds (analysis.reconstruction.test_stage)."""
    from analysis.reconstruction import test_stage as apply_frozen
    label_of = dict(zip(bank.ids("test"), map(int, labels)))
    return apply_frozen(frozen, bank.rows, bank.sha256, label_of, runs, n_boot=rules.bootstrap)


def evaluate_step(frames: dict, val_frames: dict, validation: dict, settings: Settings, scores: Scores, weights: dict,
              rules=RULES, status="test split, frozen thresholds") -> dict:
    """Everything computed once test labels are in. frames: domain -> test Frame with labels."""
    for d, f in frames.items():
        if int(f.wrong.sum()) != weights[d]:
            raise DataError(f"{d}: {int(f.wrong.sum())} wrong test answers, but the manifest's weight is {weights[d]}")
    boots = {d: Boot(f.problems, n=rules.bootstrap, seed=rules.seed) for d, f in frames.items()}
    pooled_boot = None
    if len(frames) > 1:
        order = list(frames)
        pooled_boot = Boot([p for d in order for p in frames[d].problems], n=rules.bootstrap, seed=rules.seed,
                           strata=[d for d in order for _ in frames[d].problems])
    out = {"status": status, "settings": settings.to_dict(), "rules": {"bootstrap": rules.bootstrap, "seed": rules.seed},
           "pairs": [r for d, f in frames.items() for r in pair_rows(f, validation, settings, scores, boots[d], rules)],
           "singles": single_rows(frames, validation, settings, scores, boots, pooled_boot, rules),
           "pooled": pooled_rows(frames, validation, settings, weights, pooled_boot, rules),
           "h2": {d: h2_block(f, validation, settings, boots[d], rules) for d, f in frames.items()},
           "e1": {d: e1_block(f, validation, settings, boots[d], rules) for d, f in frames.items()},
           "h1": None, "breakdowns": []}
    if settings.runs_h1:
        f, boot = frames[H1_DOMAIN], boots[H1_DOMAIN]
        h1 = out["h1"] = h1_test(f, val_frames[H1_DOMAIN], validation, settings, boot, rules)
        out["reference_rows"] = reference_rows(f, validation, settings, scores, boot, rules)
        h1["h2_status"] = status_of(out["h2"][H1_DOMAIN]["trend"], h1["pairs"]["ar_diffusion"])
        h1["reading"] = reading(h1["verdict"], h1["h2_status"], h1["llada_below_bar_on_validation"],
                                h1["pseudo_check"]["ar_diffusion"]["beats"], h1["comparable_cost_claim"],
                                settings.mode == "repair")
        chosen = {g: find_pair(settings, H1_DOMAIN, n) for g, n in validation["h1"]["selected"].items()}
        for d, frame in frames.items():     # on code, the same verifiers with the scores they have there
            pairs = {g: same_verifiers(settings, p, d) for g, p in chosen.items()}
            out["breakdowns"] += breakdown_rows(frame, pairs, validation, boots[d], rules)
        if settings.mode == "repair" and len(frames) > 1:
            out["pooled_note"] = ("no pooled difference of the selected pairs: LLaDA has its reconstruction score on math and "
                                  "its verdict score on code")
    else:
        out["ar_family"] = {d: ar_family_block(f, validation, settings, boots[d], rules) for d, f in frames.items()}
        for d, f in frames.items():
            best = validation["ar_family"][d]["selected"]
            out["breakdowns"] += breakdown_rows(f, {"ar_ar": find_pair(settings, d, best)}, validation, boots[d], rules)
    return out


# ---------------------------------------------------------------- command line

def label_arguments(items) -> dict:
    out = {}
    for item in items:
        domain, _, path = item.partition("=")
        if not path or domain in out:
            raise DataError(f"--test-labels takes domain=path once per domain, got {item!r}")
        out[domain] = Path(path)
    return out


def run(root, manifest_path, test_labels: dict, out_dir, rules=RULES) -> dict:
    root, manifest_path, out_dir = Path(root), Path(manifest_path), Path(out_dir)
    if not (root / manifest_path).is_file():
        raise NotFrozen(f"{manifest_path}: no analysis manifest; the validation step and its commit come first")
    manifest = json.loads((root / manifest_path).read_text(encoding="utf-8"))
    side = manifest.get("side")
    commit = require_committed(root, [manifest_path.as_posix(), manifest["settings"]["path"], manifest["record"]["path"],
                                      *CODE_FILES, *([side["file"]["path"], *(c["path"] for c in side["code"])] if side else [])])
    problems = check_manifest(root / manifest_path, root, rules)
    if problems:
        raise NotFrozen("the manifest does not match its inputs:\n  " + "\n  ".join(problems[:20]))
    domains = tuple(manifest["data"])
    if set(test_labels) != set(domains):
        raise DataError(f"test labels are needed for exactly {sorted(domains)}, got {sorted(test_labels)}")
    if (root / out_dir).exists() and any((root / out_dir).iterdir()):
        raise DataError(f"{out_dir} already holds results; a reproduction needs its own --out")

    # Everything that can fail without labels fails here, before the join is logged.
    settings = parse_settings(json.loads((root / manifest["settings"]["path"]).read_text(encoding="utf-8")))
    banks_dir = Path(manifest["data"][domains[0]]["bank"]["path"]).parent.parent
    banks = load_banks(root / banks_dir, domains)
    for d, bank in banks.items():
        if bank.sha256 != manifest["data"][d]["bank"]["sha256"] or bank.split_sha256() != manifest["data"][d]["split_sha256"]:
            raise NotFrozen(f"{d}: the bank or its split is not the one in the manifest")
    scores = load_scores([root / p for p in manifest["scores"]["files"]], banks, {d: needed_scores(settings, d) for d in domains},
                         rules=rules)
    val_frames, test_frames = {}, {}
    for d, bank in banks.items():
        labels, _ = load_validation_labels(root / banks_dir / d / VALIDATION_LABELS, bank)
        members = all_members(settings, d)
        val_frames[d] = make_frame(bank, scores, "validation", members, labels)
        test_frames[d] = make_frame(bank, scores, "test", members)

    loaded_side = side_runs(root, side, banks) if side else None

    shas = {d: sha256_file(root / p) for d, p in test_labels.items()}
    for d, p in test_labels.items():
        recorded = banks[d].recorded.get(TEST_LABELS)      # whatever the file is called where F keeps it
        if recorded is not None and shas[d] not in recorded:
            raise NotFrozen(f"{p}: does not match the hash recorded at the bank freeze")
    manifest_sha = sha256_file(root / manifest_path)
    entry = {"event": join_event(read_log(root), manifest_sha, shas), "utc": now(), "manifest": manifest_path.as_posix(),
             "manifest_sha256": manifest_sha, "git_commit": commit, "code_sha256": manifest["code"]["sha256"],
             "out": out_dir.as_posix(), "labels": {d: {"sha256": s, "matches_bank_manifest": TEST_LABELS in banks[d].recorded}
                                                   for d, s in sorted(shas.items())}}
    write_log(root, entry)

    for d, bank in banks.items():
        test_frames[d].labels, test_frames[d].error_types = load_test_labels(root / test_labels[d], bank)
    results = evaluate_step(test_frames, val_frames, manifest["validation"], settings, scores, manifest["weights"], rules)
    results["join"], results["manifest_sha256"] = entry, manifest_sha
    if side:    # last, and its own failure never takes the main result with it
        try:
            results["reconstruction_side"] = side_results(*loaded_side, banks[side["domain"]], test_frames[side["domain"]].labels, rules)
        except (Exception, SystemExit) as exc:
            results["reconstruction_side"] = {"error": f"{type(exc).__name__}: {exc}"}
    text = json.dumps(results, indent=1, default=float) + "\n"
    (root / out_dir).mkdir(parents=True, exist_ok=True)
    (root / out_dir / "results.json").write_text(text, encoding="utf-8", newline="\n")
    write_log(root, {"event": "completed", "utc": now(), "out": out_dir.as_posix(), "manifest_sha256": manifest_sha, "labels": {}})
    return results


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--root", default=".")
    ap.add_argument("--manifest", default="results/final/analysis_manifest.json")
    ap.add_argument("--test-labels", nargs="+", required=True, metavar="DOMAIN=PATH")
    ap.add_argument("--out", default="results/final/test")
    args = ap.parse_args(argv)
    try:
        results = run(args.root, args.manifest, label_arguments(args.test_labels), args.out)
    except (DataError, ValueError, OSError) as exc:
        print(f"error: {exc}")
        return 2
    from analysis.final_report import write_report
    written = write_report(results, Path(args.root) / args.out)
    print(f"{results['join']['event']}: wrote {', '.join(written)} to {args.out}. Commit {JOIN_LOG} now.")
    if results["h1"]:
        print(f"H1: {results['h1']['verdict_text']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
