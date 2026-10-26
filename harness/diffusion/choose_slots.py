"""Choose a diffusion model's answer-slot count on the development bank (PLAN.md §4).

Takes the run_score outputs for 1, 8 and 32 slots on the same items and picks the count with the
highest median Yes+No mass, ties going to fewer slots. Also reports the usable-scores gate
(PLAN.md §7: median Yes+No mass >= 0.5 under T1). Labels are never read.

    python -m harness.diffusion.choose_slots results/stage3/illada/dev_n1.jsonl \\
        results/stage3/illada/dev_n8.jsonl results/stage3/illada/dev_n32.jsonl
"""
from __future__ import annotations

import argparse
import json
import statistics
import sys

from .interface import read_jsonl

USABLE_MASS = 0.5


def choose(files: dict) -> dict:
    """files: {n_slots: [score rows]} for the same model, template, bank and items."""
    if set(files) != {1, 8, 32}:
        raise ValueError(f"need score files for 1, 8 and 32 slots, got {sorted(files)}")
    settings, ids, medians = set(), None, {}
    for n, rows in files.items():
        if not rows or any(r["status"] != "ok" or r["n_slots"] != n for r in rows):
            raise ValueError(f"{n} slots: every row must be ok and scored with {n} slots")
        settings |= {(r["model"], r["template"], r["bank_hash"], r["revision"]) for r in rows}
        these = sorted(r["candidate_id"] for r in rows)
        if ids is not None and these != ids:
            raise ValueError("the three files must cover the same items")
        ids = these
        medians[n] = statistics.median(r["yes_no_mass"] for r in rows)
    if len(settings) != 1:
        raise ValueError(f"files mix models, templates, banks or revisions: {sorted(settings)}")
    model, template, bank_hash, revision = settings.pop()
    chosen = min(medians, key=lambda n: (-medians[n], n))
    return {"model": model, "template": template, "bank_hash": bank_hash, "revision": revision,
            "items": len(ids), "median_yes_no_mass": medians, "chosen_n_slots": chosen,
            "rule": "highest median Yes+No mass; ties go to fewer slots",
            "usable_scores_gate": {"threshold": USABLE_MASS, "median": medians[chosen],
                                   "passed": medians[chosen] >= USABLE_MASS}}


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("files", nargs=3, help="run_score outputs for 1, 8 and 32 slots")
    ap.add_argument("--report", default=None, help="also write the JSON result here")
    args = ap.parse_args(argv)
    files = {}
    for path in args.files:
        rows = list(read_jsonl(path))
        n = {r["n_slots"] for r in rows}
        if len(n) != 1:
            raise SystemExit(f"{path}: mixed n_slots {sorted(n)}")
        files[n.pop()] = rows
    result = choose(files)
    text = json.dumps(result, indent=2)
    print(text)
    if args.report:
        with open(args.report, "w", encoding="utf-8") as f:
            f.write(text + "\n")
    return 0 if result["usable_scores_gate"]["passed"] else 1


if __name__ == "__main__":
    sys.exit(main())
