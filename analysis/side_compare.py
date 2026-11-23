"""Side experiment, exploratory: likelihood score versus verdict score on the same 500 validation answers.

The likelihood score is LLaDA's reconstruction score (128 samples, seed 544) or an AR model's exact
log-probability (notebook 12, ``results/reconstruction/side``). The verdict score is the same model's
T1 Yes/No score on the full math bank (notebook 13, ``results/full/math``). Both are compared on the
side experiment's 500 drawn validation answers: AUROC of each, the paired difference with a 95%
interval from resampling whole problems, and the rank correlation between the two scores. Only the
side experiment's runs count (LLaDA at 128 samples and seed 544, and each AR model's exact
log-probability); a second run of one model is refused rather than silently replacing the first.

Only validation labels are read; ``validation_labels`` refuses a test-label file or a test answer.
The test half is not evaluated here (decisions/2026-11-22-side-experiment-after-test.md).

    python -m analysis.side_compare --out results/reconstruction/side/verdict_compare.json
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from analysis.metrics import ProblemBootstrap, auroc, spearman
from analysis.pilot import load_scores, read_jsonl, score_files
from analysis.reconstruction import (N_SAMPLES, SEED, aligned, load_runs, run_label, sha256_of, validation_labels,
                                      write_json)

MODELS = ("LLaDA-8B", "Qwen2.5-7B", "Llama-3.1-8B", "Gemma-2-9B", "Mistral-7B")
TEMPLATE = "T1"


def likelihood_by_model(runs, rows) -> dict:
    """Model name -> likelihood scores aligned to ``rows``, from the side experiment's runs only."""
    out = {}
    for key, run in runs.items():
        _, method, n_samples, seed = key
        if method == "diffusion_mc" and (n_samples, seed) != (N_SAMPLES, SEED):
            continue
        name = run_label(key).split()[0]
        if name in out:
            raise SystemExit(f"two side-experiment runs of {name}; keep one in the likelihood folder")
        out[name] = aligned(run, rows)
    return out


def compare(rows, y, verdict, likelihood, n_boot) -> dict:
    boot = ProblemBootstrap([r["problem_id"] for r in rows], n=n_boot)
    out = {}
    for name in MODELS:
        v, x = verdict[name], likelihood[name]
        if not (np.isfinite(v).all() and np.isfinite(x).all()):
            raise SystemExit(f"{name}: a drawn validation answer has no ok score")
        out[name] = {
            "verdict_auroc": auroc(v, y),
            "verdict_auroc_ci": boot.interval(lambda ix: auroc(v[ix], y[ix])),
            "likelihood_auroc": auroc(x, y),
            "likelihood_auroc_ci": boot.interval(lambda ix: auroc(x[ix], y[ix])),
            "likelihood_minus_verdict": auroc(x, y) - auroc(v, y),
            "likelihood_minus_verdict_ci": boot.interval(lambda ix: auroc(x[ix], y[ix]) - auroc(v[ix], y[ix])),
            "rank_correlation": spearman(v, x),
        }
    return out


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--bank", default="bank/full/math/bank.jsonl")
    ap.add_argument("--subset", default="results/reconstruction/side/subset.json")
    ap.add_argument("--labels", default="bank/full/math/labels_validation.jsonl")
    ap.add_argument("--likelihood", default="results/reconstruction/side")
    ap.add_argument("--verdicts", nargs="+",
                    default=["results/full/math/ar", "results/full/math/llada/full_T1_n1.jsonl"])
    ap.add_argument("--bootstrap", type=int, default=10000)
    ap.add_argument("--out", required=True)
    args = ap.parse_args(argv)

    bank, bank_hash = read_jsonl(args.bank), sha256_of(args.bank)
    subset = json.loads(Path(args.subset).read_text(encoding="utf-8"))
    if subset["bank_hash"] != bank_hash:
        raise SystemExit(f"{args.subset} was drawn from another bank")
    wanted = set(subset["splits"]["validation"])
    labels = validation_labels(args.labels, bank)
    rows = [r for r in bank if r["candidate_id"] in wanted]
    if len(rows) != len(wanted) or any(r["candidate_id"] not in labels for r in rows):
        raise SystemExit("a drawn validation answer is missing from the bank or has no label")
    y = np.array([labels[r["candidate_id"]] for r in rows], dtype=float)

    index = {r["candidate_id"]: i for i, r in enumerate(rows)}
    scores = load_scores(score_files(args.verdicts), index)
    verdict = {name: scores[(name, TEMPLATE)]["score"] for name in MODELS if (name, TEMPLATE) in scores}
    likelihood = likelihood_by_model(load_runs([args.likelihood], bank_hash), rows)
    missing = [n for n in MODELS if n not in verdict or n not in likelihood]
    if missing:
        raise SystemExit(f"no verdict or likelihood scores for {missing}")

    report = {"bank_hash": bank_hash, "labels_sha256": sha256_of(args.labels), "template": TEMPLATE,
              "n_correct": int((y == 1).sum()), "n_wrong": int((y == 0).sum()), "bootstrap": args.bootstrap,
              "models": compare(rows, y, verdict, likelihood, args.bootstrap)}
    write_json(args.out, report)
    print(f"{report['n_correct']} correct and {report['n_wrong']} wrong validation answers")
    print("| Model | Verdict AUROC | Likelihood AUROC | Likelihood − verdict [95% CI] | Rank corr. |")
    print("|---|---|---|---|---|")
    for name, m in report["models"].items():
        lo, hi = m["likelihood_minus_verdict_ci"]
        print(f"| {name} | {m['verdict_auroc']:.3f} | {m['likelihood_auroc']:.3f} | "
              f"{m['likelihood_minus_verdict']:+.3f} [{lo:+.3f}, {hi:+.3f}] | {m['rank_correlation']:.2f} |")


if __name__ == "__main__":
    main()
