"""Choose a diffusion model's answer-slot count (and answer prefix) on the development bank.

PLAN.md §4: the slot count with the highest median Yes+No mass, ties going to fewer slots.
When run_score outputs with several answer prefixes are given (iLLaDA: thinking off, see
decisions/2026-10-25-illada-thinking-off.md), the prefix is chosen jointly by the same rule:
the (prefix, slots) combination with the highest median mass, ties going to fewer slots and
then to the prefix listed first in scorer.ANSWER_PREFIXES. Also reports the usable-scores gate
(PLAN.md §7: median Yes+No mass >= 0.5 under T1). Labels are never read.

    python -m harness.diffusion.choose_slots <the run_score outputs> --report slots.json

Files ending in .manifest.jsonl (run_score's run records, written next to its outputs) are
skipped, so a shell glob over the output folder is safe.
"""
from __future__ import annotations

import argparse
import json
import statistics
import sys

from .interface import read_jsonl
from .scorer import ANSWER_PREFIXES

USABLE_MASS = 0.5
PREFIX_ORDER = list(ANSWER_PREFIXES)


def choose_combination(groups: dict) -> dict:
    """groups: {(answer_prefix, n_slots): [score rows]} for one model, template, bank and revision."""
    if not groups:
        raise ValueError("no score files")
    settings, ids, medians = set(), None, {}
    for (prefix, n), rows in groups.items():
        if prefix not in ANSWER_PREFIXES or n not in (1, 8, 32):
            raise ValueError(f"unknown combination {(prefix, n)}")
        if not rows or any(r["status"] != "ok" or r["n_slots"] != n or r.get("answer_prefix", "none") != prefix
                           for r in rows):
            raise ValueError(f"{prefix}, {n} slots: every row must be ok and scored with those settings")
        settings |= {(r["model"], r["template"], r["bank_hash"], r["revision"]) for r in rows}
        these = sorted(r["candidate_id"] for r in rows)
        if ids is not None and these != ids:
            raise ValueError("every file must cover the same items")
        ids = these
        medians[(prefix, n)] = statistics.median(r["yes_no_mass"] for r in rows)
    if len(settings) != 1:
        raise ValueError(f"files mix models, templates, banks or revisions: {sorted(settings)}")
    model, template, bank_hash, revision = settings.pop()
    prefix, n = min(medians, key=lambda k: (-medians[k], k[1], PREFIX_ORDER.index(k[0])))
    table = {}
    for (p, m), value in sorted(medians.items(), key=lambda kv: (PREFIX_ORDER.index(kv[0][0]), kv[0][1])):
        table.setdefault(p, {})[str(m)] = value
    return {"model": model, "template": template, "bank_hash": bank_hash, "revision": revision,
            "items": len(ids), "median_yes_no_mass": table,
            "chosen_answer_prefix": prefix, "chosen_n_slots": n,
            "rule": "highest median Yes+No mass; ties go to fewer slots, then to the prefix listed first",
            "usable_scores_gate": {"threshold": USABLE_MASS, "median": medians[(prefix, n)],
                                   "passed": medians[(prefix, n)] >= USABLE_MASS}}


def choose(files: dict) -> dict:
    """files: {n_slots: [score rows]} for 1, 8 and 32 slots, all with the same answer prefix."""
    if set(files) != {1, 8, 32}:
        raise ValueError(f"need score files for 1, 8 and 32 slots, got {sorted(files)}")
    prefixes = {r.get("answer_prefix", "none") for rows in files.values() for r in rows}
    if len(prefixes) != 1:
        raise ValueError(f"files mix answer prefixes {sorted(prefixes)}; use choose_combination")
    prefix = prefixes.pop()
    return choose_combination({(prefix, n): rows for n, rows in files.items()})


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("files", nargs="+", help="run_score outputs: 1, 8 and 32 slots for each answer prefix")
    ap.add_argument("--report", default=None, help="also write the JSON result here")
    args = ap.parse_args(argv)
    groups = {}
    for path in args.files:
        if path.endswith(".manifest.jsonl"):
            continue
        rows = list(read_jsonl(path))
        keys = {(r.get("answer_prefix", "none"), r["n_slots"]) for r in rows}
        if len(keys) != 1:
            raise SystemExit(f"{path}: mixed answer prefixes or slot counts {sorted(keys)}")
        key = keys.pop()
        if key in groups:
            raise SystemExit(f"two files for {key}")
        groups[key] = rows
    for prefix in {p for p, _ in groups}:
        if {n for p, n in groups if p == prefix} != {1, 8, 32}:
            raise SystemExit(f"answer prefix {prefix!r}: need files for 1, 8 and 32 slots")
    result = choose_combination(groups)
    text = json.dumps(result, indent=2)
    print(text)
    if args.report:
        with open(args.report, "w", encoding="utf-8") as f:
            f.write(text + "\n")
    return 0 if result["usable_scores_gate"]["passed"] else 1


if __name__ == "__main__":
    sys.exit(main())
