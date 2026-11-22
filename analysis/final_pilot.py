"""Smoke test of the whole final analysis on the pilot (500 answers): pilot, exploratory, in-sample.

    python -m analysis.final_pilot --out results/final/pilot_smoke

Thresholds and pairs are chosen on the pilot's answers and evaluated on the same answers, so nothing here is an
H1 result. It runs the same functions the real run uses (loaders, validation step, test step, report) on real
score files, with the models adopted after the pilot (configs/final_models.example.json, the "go" row).
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from analysis.final_data import align_labels, load_bank, load_scores, make_frame, read_jsonl
from analysis.final_evaluate import evaluate_step
from analysis.final_report import write_report
from analysis.final_settings import DOMAINS, RULES, load_settings, needed_scores
from analysis.final_validation import all_members, summary, validation_step

STATUS = "pilot, exploratory, in-sample (code smoke test, not an H1 result)"
PILOT_SCORES = ("results/pilot/ar", "results/pilot/llada", "results/pilot/dream")


def pilot_inputs(root=".", settings_path="configs/final_models.example.json", rules=RULES):
    """(settings, banks, scores, labeled frames per domain) of the pilot."""
    root = Path(root)
    settings = load_settings(root / settings_path)
    banks = {d: load_bank(root / "bank/pilot/pilot_bank.jsonl", d, splits=("pilot",)) for d in DOMAINS}
    scores = load_scores([root / p for p in PILOT_SCORES], banks, {d: needed_scores(settings, d) for d in DOMAINS},
                         ignore_templates=("T3",), rules=rules)
    labels = read_jsonl(root / "bank/pilot/pilot_labels.jsonl")
    frames = {d: make_frame(b, scores, "pilot", all_members(settings, d),
                            align_labels([r for r in labels if r["candidate_id"] in b.index], b, "pilot", "pilot_labels.jsonl"))
              for d, b in banks.items()}
    return settings, banks, scores, frames


def run_pilot(root=".", settings_path="configs/final_models.example.json", rules=RULES) -> tuple[dict, dict]:
    settings, banks, scores, frames = pilot_inputs(root, settings_path, rules)
    validation = validation_step(frames, scores, settings, rules)
    weights = {d: b.pairs("pilot") for d, b in banks.items()}
    return validation, evaluate_step(frames, frames, validation, settings, scores, weights, rules, status=STATUS)


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--out", required=True)
    ap.add_argument("--settings", default="configs/final_models.example.json")
    args = ap.parse_args(argv)
    validation, results = run_pilot(".", args.settings)
    out = Path(args.out)
    written = write_report(results, out)
    (out / "validation.md").write_text(summary({"validation": validation}, f"Validation step on the pilot ({STATUS})"),
                                       encoding="utf-8", newline="\n")
    (out / "results.json").write_text(json.dumps(results, indent=1, default=float) + "\n", encoding="utf-8", newline="\n")
    print(f"{STATUS}: wrote {', '.join(written)}")


if __name__ == "__main__":
    main()
