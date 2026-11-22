"""Backup score: the reconstruction score on the pilot, and its side experiment (PLAN.md §4, §7).

    # 1. sample count on the development bank (two seeds); labels are never read
    python -m analysis.reconstruction samples --bank bank/dev/dev_bank.jsonl \\
        --scores results/reconstruction/dev --out results/reconstruction/dev/samples.json
    # 2. does LLaDA's reconstruction score pass as the repair? (the pilot's math answers)
    python -m analysis.reconstruction pilot --bank bank/pilot/pilot_bank.jsonl \\
        --labels bank/pilot/pilot_labels.jsonl --scores results/reconstruction/pilot \\
        --verdicts results/pilot/ar results/pilot/llada results/pilot/dream \\
        --out results/reconstruction/pilot/analysis
    # 3. side experiment: 500 validation and 500 test answers, then thresholds from validation
    python -m analysis.reconstruction draw --bank bank/full/math/bank.jsonl --out subset.json
    python -m analysis.reconstruction validation --bank bank/full/math/bank.jsonl --subset subset.json \\
        --labels bank/full/math/labels_validation.jsonl --scores results/reconstruction/side --out side.json

Scores are files or folders of rows written by harness.diffusion.run_reconstruction and
harness.ar_logprob. The score is the per-token log-likelihood, higher = more plausible.

**Repair rule (PLAN.md §7).** LLaDA's 128-sample, seed-544 score passes if, on the pilot's math
answers, it meets gate A (AUROC at least 0.60, 95% lower bound above 0.50, no more than 0.10 below
Llama-3.2-3B's T1 verdict AUROC) and a seed-545 estimate has a rank correlation of at least 0.95
with it. Test 4 is not applied. Every answer must be scored unbatched by the pinned model, all on
one GPU type; until then the outcome is "not finished". Dream and the AR models are reported as
information only, and code is descriptive.

**Corrected labels.** If the audit corrects pilot labels, pass the corrected label file with the same
bank file (the scores are tied to its hash): pairs that no longer hold one correct and one wrong
answer are dropped and counted (PLAN.md §7), and nothing has to be scored again.

**Test labels.** Nothing here opens a test-label file: there is no test command, and the
validation command refuses a label file that names a test answer. ``test_stage`` takes the
labels as an argument, from the final-analysis code that joins them.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
import re
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

from analysis.metrics import ProblemBootstrap, auroc, spearman, threshold_at_tar
from analysis.pilot import (GATE_A_MIN, GATE_A_SMALL_GAP, SMALL_REFERENCE, VERIFIERS, domain_mask, fmt,
                            load_scores, pair_ranking, read_jsonl, score_files)

REPAIR_MODEL, REPAIR_DOMAIN = "LLaDA-8B", "math"
REPAIR_REVISION = "08b83a6feb34df1a6011b80c3c00c7563e963b07"  # harness/diffusion/scorer.py
N_SAMPLES, SEED, REPEAT_SEED, MC_GROUP, MICRO_BATCH = 128, 544, 545, 16, 1
RHO_MIN = 0.95
METHODS = ("diffusion_mc", "ar_exact")
SIDE_N, TARS = 500, (0.80, 0.90, 0.95)
# what has to be the same in every row of one run
SETTINGS = ("bank_hash", "model", "revision", "method", "n_samples", "seed", "mc_group", "micro_batch", "attention")


def sha256_of(path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


# ---------------------------------------------------------------- loading

def load_runs(paths, bank_hash: str) -> dict:
    """(verifier, method, n_samples, seed) -> {"settings", "rows": {candidate_id: row}}, for the
    rows scored on this bank. Rows of one run must share every setting."""
    files = []
    for p in map(Path, paths):
        files += sorted(p.rglob("*.jsonl")) if p.is_dir() else [p]
    runs = {}
    for path in files:
        if path.name.endswith(".manifest.jsonl"):
            continue
        for r in read_jsonl(path):
            if r.get("method") not in METHODS or r["bank_hash"] != bank_hash:
                continue
            if r["model"] not in VERIFIERS:
                raise SystemExit(f"{path}: unknown model {r['model']!r}")
            key = (VERIFIERS[r["model"]][0], r["method"], r["n_samples"], r["seed"])
            settings = {k: r[k] for k in SETTINGS}
            run = runs.setdefault(key, {"settings": settings, "rows": {}})
            if run["settings"] != settings:
                raise SystemExit(f"{path}: {key} mixes settings {run['settings']} and {settings}")
            if r["candidate_id"] in run["rows"]:
                raise SystemExit(f"{path}: {key} scores {r['candidate_id']} twice")
            if r["status"] == "ok" and not (
                    r["n_tokens"] >= 1 and math.isfinite(r["loglik"])
                    and math.isclose(r["score"], r["loglik"] / r["n_tokens"], rel_tol=1e-9, abs_tol=1e-12)):
                raise SystemExit(f"{path}: {r['candidate_id']} is not a per-token log-likelihood")
            run["rows"][r["candidate_id"]] = r
    return runs


def aligned(run: dict, bank: list[dict], field: str = "score") -> np.ndarray:
    """One value per bank row (nan where the answer has no ok score)."""
    out = np.full(len(bank), np.nan)
    for i, b in enumerate(bank):
        r = run["rows"].get(b["candidate_id"])
        if r is not None and r["status"] == "ok" and r.get(field) is not None:
            out[i] = r[field]
    return out


def intact_pairs(bank: list[dict], label_of: dict) -> tuple:
    """(bank rows of the pairs that hold one correct and one wrong answer, their labels, the pair
    ids dropped). A corrected label file can leave a pair with two equal labels or without one."""
    by_pair = defaultdict(list)
    for b in bank:
        by_pair[b["pair_id"]].append(label_of.get(b["candidate_id"]))
    dropped = sorted(p for p, ls in by_pair.items() if len(ls) != 2 or None in ls or sorted(ls) != [0, 1])
    kept = [b for b in bank if b["pair_id"] not in set(dropped)]
    return kept, np.array([label_of[b["candidate_id"]] for b in kept], dtype=float), dropped


def gpu_type(name):
    """"NVIDIA A100-SXM4-40GB" -> "A100" (as in harness/diffusion/run_reconstruction.py)."""
    found = re.search(r"\b[A-Z]{1,3}\d+\b", name or "")
    return found.group(0) if found else name


# ---------------------------------------------------------------- sample count (development bank)

def sample_count(first: dict, second: dict) -> dict:
    """The smallest sample count at which two independent estimates (two seeds) reach the rank
    correlation (PLAN.md §4). Only the side experiment uses it: the repair always uses 128."""
    ids = sorted(first["rows"])
    if ids != sorted(second["rows"]) or any(run["rows"][c]["status"] != "ok" for run in (first, second) for c in ids):
        raise SystemExit("the two seeds must score the same answers, all of them ok")

    def per_token(run, n):
        return np.array([run["rows"][c]["loglik_at"][n] / run["rows"][c]["n_tokens"] for c in ids])

    counts = sorted(set(first["rows"][ids[0]]["loglik_at"]) & set(second["rows"][ids[0]]["loglik_at"]), key=int)
    rho = {n: spearman(per_token(first, n), per_token(second, n)) for n in counts}
    passing = [int(n) for n in counts if math.isfinite(rho[n]) and rho[n] >= RHO_MIN]
    return {"items": len(ids), "spearman": rho, "minimum": RHO_MIN, "chosen_n_samples": min(passing, default=None)}


def samples_report(runs: dict) -> dict:
    out = {}
    for (name, method, n, seed) in sorted(k for k in runs if k[1] == "diffusion_mc" and k[3] == SEED):
        second = runs.get((name, method, n, REPEAT_SEED))
        if second is not None:
            out[f"{name} (up to {n})"] = sample_count(runs[(name, method, n, seed)], second)
    return out


# ---------------------------------------------------------------- pilot: gate A and the repeat check

def run_metrics(bank, labels, run, domain, n_boot) -> dict:
    in_domain = domain_mask(bank, domain)
    x = aligned(run, bank)
    idx = np.flatnonzero(in_domain & np.isfinite(x))
    statuses = Counter(run["rows"].get(b["candidate_id"], {"status": "missing"})["status"]
                       for b, keep in zip(bank, in_domain) if keep)
    out = {"domain": domain, "n": int(in_domain.sum()), "n_ok": len(idx), "statuses": dict(statuses)}
    if len(idx) == 0:
        return out
    s, y = x[idx], labels[idx]
    lo, hi = ProblemBootstrap([bank[i]["problem_id"] for i in idx], n=n_boot).interval(lambda ix: auroc(s[ix], y[ix]))
    seconds = aligned(run, bank, "runtime_s")[idx]
    rows = [run["rows"][bank[i]["candidate_id"]] for i in idx]
    return {**out, "auroc": auroc(s, y), "auroc_lo": lo, "auroc_hi": hi,
            "pair_ranking_accuracy": pair_ranking(bank, labels, x, in_domain),
            "median_answer_tokens": float(np.median(aligned(run, bank, "n_tokens")[idx])),
            "seconds_per_candidate": float(np.nanmean(seconds)) if np.isfinite(seconds).any() else None,
            "boundary_separate": sum(r["boundary"] == "separate" for r in rows),
            "gpus": sorted({str(r["gpu"]) for r in rows})}


def gate_a(entry: dict, small_auroc) -> dict:
    """PLAN.md §7 gate A for a diffusion score in one domain. ``passes`` is None until every answer
    is scored and the small AR reference is known."""
    out = {"small_reference": small_auroc, "minimum": GATE_A_MIN, "allowed_gap": GATE_A_SMALL_GAP}
    if entry["n_ok"] < entry["n"] or entry["n"] == 0 or small_auroc is None or not math.isfinite(small_auroc):
        return {**out, "passes": None}
    out["auroc_at_least_0.60"] = entry["auroc"] >= GATE_A_MIN
    out["lower_bound_above_0.50"] = entry["auroc_lo"] > 0.5
    out["within_0.10_of_small_ar"] = entry["auroc"] >= small_auroc - GATE_A_SMALL_GAP
    out["passes"] = bool(out["auroc_at_least_0.60"] and out["lower_bound_above_0.50"] and out["within_0.10_of_small_ar"])
    return out


def repeat_check(bank, first, second, domain) -> dict:
    """Rank correlation of two estimates of the same score (the seed-544 and seed-545 runs)."""
    if any(first["settings"][k] != second["settings"][k] for k in SETTINGS if k != "seed"):
        raise SystemExit("the two estimates were scored with different settings")
    m = domain_mask(bank, domain)
    a, b = aligned(first, bank), aligned(second, bank)
    both = m & np.isfinite(a) & np.isfinite(b)
    rho = spearman(a[both], b[both])
    complete = bool(m.any()) and int(both.sum()) == int(m.sum())
    return {"domain": domain, "n": int(m.sum()), "n_both": int(both.sum()), "spearman": rho, "minimum": RHO_MIN,
            "passes": bool(rho >= RHO_MIN) if complete and math.isfinite(rho) else None}


def repair_problems(runs, bank) -> list[str]:
    """Settings PLAN.md fixes for the repair that the two runs do not have (an empty list if none)."""
    pair = [runs.get((REPAIR_MODEL, "diffusion_mc", N_SAMPLES, s)) for s in (SEED, REPEAT_SEED)]
    problems = [f"no seed-{s} run" for s, run in zip((SEED, REPEAT_SEED), pair) if run is None]
    for run in filter(None, pair):
        s = run["settings"]
        for field, wanted in (("revision", REPAIR_REVISION), ("micro_batch", MICRO_BATCH), ("mc_group", MC_GROUP)):
            if s[field] != wanted:
                problems.append(f"seed {s['seed']}: {field} {s[field]}, not {wanted}")
    ids = {b["candidate_id"] for b in bank if b["domain"] == REPAIR_DOMAIN}
    types = {str(gpu_type(r["gpu"])) for run in filter(None, pair) for c, r in run["rows"].items()
             if c in ids and r["status"] == "ok"}
    if len(types) > 1:
        problems.append(f"scored on more than one GPU type: {', '.join(sorted(types))}")
    return problems


def repair_verdict(gate, repeat, problems) -> str:
    checks = [c["passes"] if c is not None else None for c in (gate, repeat)]
    if None in checks or problems:
        return "not finished"
    return "pass" if all(checks) else "fail"


def pilot_report(bank, labels, runs, verdicts, n_boot=2000) -> dict:
    """``verdicts``: {(verifier, template): {"score": array}} from analysis.pilot.load_scores."""
    def verdict_auroc(name, domain):
        s = verdicts.get((name, "T1"))
        if s is None:
            return None
        m = domain_mask(bank, domain) & np.isfinite(s["score"])
        # a half-scored reference is no reference
        return auroc(s["score"][m], labels[m]) if m.sum() == domain_mask(bank, domain).sum() else None

    domains = [d for d in ("math", "code") if domain_mask(bank, d).any()]
    models, repeats, gates = [], {}, {}
    for key in sorted(runs, key=str):
        name, method, n_samples, seed = key
        run = runs[key]
        for domain in domains:
            e = run_metrics(bank, labels, run, domain, n_boot)
            if e["n_ok"] == 0:
                continue
            models.append({"verifier": name, "class": next(v[2] for v in VERIFIERS.values() if v[0] == name),
                           "method": method, "n_samples": n_samples, "seed": seed,
                           "micro_batch": run["settings"]["micro_batch"], "revision": run["settings"]["revision"],
                           **e, "verdict_auroc_t1": verdict_auroc(name, domain),
                           "role": "repair" if (name, domain, method, n_samples) == (
                               REPAIR_MODEL, REPAIR_DOMAIN, "diffusion_mc", N_SAMPLES) else "information only"})
            if (method, n_samples, seed) == ("diffusion_mc", N_SAMPLES, SEED):
                gates.setdefault(name, {})[domain] = gate_a(e, verdict_auroc(SMALL_REFERENCE, domain))
                second = runs.get((name, method, n_samples, REPEAT_SEED))
                if second is not None:
                    repeats.setdefault(name, {})[domain] = repeat_check(bank, run, second, domain)
    gate = gates.get(REPAIR_MODEL, {}).get(REPAIR_DOMAIN)
    repeat = repeats.get(REPAIR_MODEL, {}).get(REPAIR_DOMAIN)
    problems = repair_problems(runs, bank)
    return {"repair": {"model": REPAIR_MODEL, "domain": REPAIR_DOMAIN, "n_samples": N_SAMPLES, "seed": SEED,
                       "repeat_seed": REPEAT_SEED, "gate_a": gate, "repeat_check": repeat,
                       "problems": problems, "verdict": repair_verdict(gate, repeat, problems)},
            "gate_a": gates, "repeat": repeats, "models": models}


def pilot_markdown(report: dict, bank_info: str, dropped: list = ()) -> str:
    def mark(p):
        return "not finished" if p is None else ("pass" if p else "**fail**")

    r = report["repair"]
    L = ["# Reconstruction score on the pilot", "", f"Bank: {bank_info}", "",
         "Pilot numbers: exploratory. The score is the per-token masked-reconstruction log-likelihood "
         "(AR models: exact per-token log-probability). Test 4 is not applied to it.", "",
         f"Pairs dropped because their labels are no longer one correct and one wrong answer: {len(dropped)}"
         + (f" ({', '.join(dropped)})" if dropped else ""), "",
         f"## Repair rule (PLAN.md §7): LLaDA, math, {r['n_samples']} samples, seed {r['seed']}", "",
         f"- **Outcome: {r['verdict']}**",
         f"- Gate A: {mark(r['gate_a'] and r['gate_a']['passes'])}",
         f"- Repeat check (seed {r['repeat_seed']}, Spearman at least {RHO_MIN}): "
         f"{mark(r['repeat_check'] and r['repeat_check']['passes'])}"
         + (f" ({fmt(r['repeat_check']['spearman'], 4)})" if r["repeat_check"] else ""),
         *(f"- Not the plan's settings: {p}" for p in r["problems"]), "",
         "## All runs", "",
         "Math decides the repair; every other row is information only, and code is descriptive. Gate A and the "
         "repeat check are shown for the 128-sample seed-544 diffusion runs.", "",
         "| Verifier | Method | Samples | Seed | Domain | Scored | AUROC [95% CI] | Verdict AUROC (T1) | Pair rank acc. "
         "| Gate A | Repeat (ρ) | s / answer |", "|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for m in sorted(report["models"], key=lambda m: (m["domain"] != "math", m["class"] != "diffusion",
                                                     m["verifier"], str(m["n_samples"]), str(m["seed"]))):
        main_run = (m["method"], m["n_samples"], m["seed"]) == ("diffusion_mc", N_SAMPLES, SEED)
        gate = report["gate_a"].get(m["verifier"], {}).get(m["domain"]) if main_run else None
        rep = report["repeat"].get(m["verifier"], {}).get(m["domain"]) if main_run else None
        L.append(f"| {m['verifier']} | {m['method']} | {m['n_samples'] or '–'} | {m['seed'] or '–'} | {m['domain']} "
                 f"| {m['n_ok']} / {m['n']} | {fmt(m['auroc'])} [{fmt(m['auroc_lo'])}, {fmt(m['auroc_hi'])}] "
                 f"| {fmt(m['verdict_auroc_t1'])} | {fmt(m['pair_ranking_accuracy'])} "
                 f"| {mark(gate['passes']) if gate else '–'} | {fmt(rep['spearman'], 4) if rep else '–'} "
                 f"| {fmt(m['seconds_per_candidate'])} |")
    return "\n".join(L) + "\n"


# ---------------------------------------------------------------- side experiment

def one_domain(bank: list[dict]) -> str:
    """Thresholds are per domain (PLAN.md §5), so a bank file must hold exactly one."""
    domains = sorted({r["domain"] for r in bank})
    if len(domains) != 1:
        raise SystemExit(f"the side experiment takes one domain at a time, not {domains}")
    return domains[0]


def draw_subset(bank: list[dict], n: int = SIDE_N, seed: int = SEED) -> dict:
    """``n`` answers of one domain from each split, as whole problems in seeded random order. A
    problem that would go over ``n`` is skipped."""
    one_domain(bank)
    out = {}
    for split in ("validation", "test"):
        problems = defaultdict(list)
        for r in bank:
            if r["split"] == split:
                problems[r["problem_id"]].append(r["candidate_id"])
        order = sorted(problems)
        random.Random(f"{seed}:{split}").shuffle(order)
        ids = []
        for pid in order:
            if len(ids) + len(problems[pid]) <= n:
                ids += problems[pid]
        out[split] = sorted(ids)
    return out


def validation_labels(path, bank: list[dict]) -> dict:
    """candidate_id -> label from a validation label file. A file that is called a test file, or
    that names anything outside the validation split, is refused."""
    path = Path(path)
    if "test" in path.name:
        raise SystemExit(f"{path}: test labels are never read here")
    allowed = {r["candidate_id"] for r in bank if r["split"] == "validation"}
    labels = {}
    for r in read_jsonl(path):
        if r["candidate_id"] not in allowed:
            raise SystemExit(f"{path} names {r['candidate_id']}, which is not a validation answer of this bank")
        labels[r["candidate_id"]] = r["label"]
    return labels


def _subset_arrays(bank, ids, label_of, run, label):
    """(bank rows, labels, scores) of the drawn answers; every one must be labeled and scored."""
    wanted = set(ids)
    rows = [r for r in bank if r["candidate_id"] in wanted]
    missing = [r["candidate_id"] for r in rows if r["candidate_id"] not in label_of]
    if len(rows) != len(wanted) or missing:
        raise SystemExit(f"{label}: {len(wanted) - len(rows)} drawn answers are not in the bank and {len(missing)} have no label")
    x = aligned(run, rows)
    if not np.isfinite(x).all():
        raise SystemExit(f"{label}: {int((~np.isfinite(x)).sum())} of {len(rows)} drawn answers have no ok score")
    return rows, np.array([label_of[r["candidate_id"]] for r in rows], dtype=float), x


def _auroc_entry(rows, y, x, n_boot):
    lo, hi = ProblemBootstrap([r["problem_id"] for r in rows], n=n_boot).interval(lambda ix: auroc(x[ix], y[ix]))
    return {"n_correct": int((y == 1).sum()), "n_wrong": int((y == 0).sum()),
            "auroc": auroc(x, y), "auroc_lo": lo, "auroc_hi": hi}


def run_label(key) -> str:
    name, method, n_samples, seed = key
    return f"{name} {method}" + ("" if n_samples is None else f" n{n_samples} s{seed}")


def validation_stage(bank, bank_hash, subset, label_of, runs, n_boot=2000) -> dict:
    """Validation AUROC and the thresholds at each TAR, per run. This is what the preregistration
    commit freezes for the side experiment; ``test_stage`` applies it unchanged."""
    out = {"bank_hash": bank_hash, "domain": one_domain(bank), "subset": subset, "runs": {}}
    for key in sorted(runs, key=str):
        label = run_label(key)
        rows, y, x = _subset_arrays(bank, subset["validation"], label_of, runs[key], label)
        points = {}
        for tar in TARS:
            t = threshold_at_tar(x[y == 1], tar)
            points[f"{tar:.2f}"] = {"threshold": t, "tar": float(np.mean(x[y == 1] >= t)),
                                    "far": float(np.mean(x[y == 0] >= t))}
        out["runs"][label] = {"settings": runs[key]["settings"], **_auroc_entry(rows, y, x, n_boot), "points": points}
    return out


def test_stage(frozen: dict, bank, bank_hash, test_label_of: dict, runs, n_boot=10000) -> dict:
    """AUROC, TAR and FAR on the drawn test answers at the frozen validation thresholds. The labels
    are given by the final-analysis code after its single join; nothing is read or chosen here."""
    if frozen["bank_hash"] != bank_hash or frozen["domain"] != one_domain(bank):
        raise SystemExit("the thresholds were set on another bank")
    split_of = {r["candidate_id"]: r["split"] for r in bank}
    if any(split_of.get(c) != "test" for c in frozen["subset"]["test"]):
        raise SystemExit("the drawn test answers are not all in the test split")
    by_label = {run_label(k): run for k, run in runs.items()}
    out = {"bank_hash": bank_hash, "runs": {}}
    for label, entry in frozen["runs"].items():
        run = by_label.get(label)
        if run is None or run["settings"] != entry["settings"]:
            raise SystemExit(f"{label}: no test scores with the settings the thresholds were set with")
        rows, y, x = _subset_arrays(bank, frozen["subset"]["test"], test_label_of, run, label)
        boot = ProblemBootstrap([r["problem_id"] for r in rows], n=n_boot)
        points = {}
        for tar, point in entry["points"].items():
            accepted = x >= point["threshold"]
            points[tar] = {"threshold": point["threshold"]}
            for name, value in (("tar", 1), ("far", 0)):
                chosen = y == value
                lo, hi = boot.interval(lambda ix: accepted[ix][chosen[ix]].mean() if chosen[ix].any() else np.nan)
                points[tar].update({name: float(accepted[chosen].mean()), f"{name}_count": int(accepted[chosen].sum()),
                                    f"{name}_lo": lo, f"{name}_hi": hi})
        out["runs"][label] = {**_auroc_entry(rows, y, x, n_boot), "points": points}
    return out


# ---------------------------------------------------------------- command line

def write_json(path, value) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(value, indent=1) + "\n", encoding="utf-8", newline="\n")


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = ap.add_subparsers(dest="command", required=True)
    for name in ("samples", "pilot", "draw", "validation"):
        p = sub.add_parser(name)
        p.add_argument("--bank", required=True)
        p.add_argument("--out", required=True)
        if name != "draw":
            p.add_argument("--scores", nargs="+", required=True)
        if name in ("pilot", "validation"):
            p.add_argument("--labels", required=True)
            p.add_argument("--bootstrap", type=int, default=2000)
        if name == "pilot":
            p.add_argument("--verdicts", nargs="+", required=True, help="verdict score folders, for the T1 AUROCs")
        if name == "validation":
            p.add_argument("--subset", required=True)
        if name == "draw":
            p.add_argument("--n", type=int, default=SIDE_N)
    args = ap.parse_args(argv)
    bank, bank_hash = read_jsonl(args.bank), sha256_of(args.bank)
    runs = load_runs(args.scores, bank_hash) if args.command != "draw" else {}
    if args.command != "draw" and not runs:
        raise SystemExit(f"no backup scores for bank {bank_hash[:12]} in {args.scores}")

    if args.command == "samples":
        report = {"bank_hash": bank_hash, "models": samples_report(runs)}
        write_json(args.out, report)
        print(json.dumps(report["models"], indent=1))
    elif args.command == "pilot":
        if any(r.get("split") != "pilot" for r in bank):
            raise SystemExit(f"{args.bank} is not the pilot bank; its labels are not read here")
        kept, labels, dropped = intact_pairs(bank, {r["candidate_id"]: r["label"] for r in read_jsonl(args.labels)})
        if not kept:
            raise SystemExit(f"{args.labels} leaves no pair with one correct and one wrong answer")
        verdicts = load_scores(score_files(args.verdicts), {r["candidate_id"]: i for i, r in enumerate(kept)})
        report = {"bank_hash": bank_hash, "labels_sha256": sha256_of(args.labels), "dropped_pairs": dropped,
                  **pilot_report(kept, labels, runs, verdicts, args.bootstrap)}
        out = Path(args.out)
        write_json(out / "reconstruction.json", report)
        text = pilot_markdown(report, f"`{args.bank}` (sha256 {bank_hash[:12]}…)", dropped)
        (out / "reconstruction.md").write_text(text, encoding="utf-8", newline="\n")
        print(text)
    elif args.command == "draw":
        splits = draw_subset(bank, args.n)
        write_json(args.out, {"bank": args.bank, "bank_hash": bank_hash, "seed": SEED, "requested": args.n,
                              "counts": {k: len(v) for k, v in splits.items()}, "splits": splits})
        for split, ids in splits.items():
            print(f"{split}: {len(ids)} answers drawn" + ("" if len(ids) == args.n else f" (fewer than the {args.n} asked for)"))
    else:
        subset = json.loads(Path(args.subset).read_text(encoding="utf-8"))
        if subset["bank_hash"] != bank_hash:
            raise SystemExit(f"{args.subset} was drawn from another bank")
        report = validation_stage(bank, bank_hash, subset["splits"], validation_labels(args.labels, bank), runs,
                                  args.bootstrap)
        write_json(args.out, {**report, "labels_sha256": sha256_of(args.labels)})
        print(json.dumps({k: {"auroc": v["auroc"], "points": v["points"]} for k, v in report["runs"].items()}, indent=1))


if __name__ == "__main__":
    main()
