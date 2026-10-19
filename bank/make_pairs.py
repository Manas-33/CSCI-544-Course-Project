"""Step 4: build matched pairs and write the bank and label files (PLAN.md §2, §9).

For each (question, generator) cell with at least one usable correct and one usable
wrong answer, keep one of each, drawn with a fixed seed. Near-duplicate answers are
collapsed first, within the correct answers and within the wrong answers separately. Labels go to a separate file from the bank, as PLAN.md §9
requires, so scoring code never sees them.

    python -m bank.make_pairs --questions bank/dev/questions.jsonl \\
        --generations bank/dev/generations.jsonl --labeled bank/dev/labeled.jsonl --out bank/dev
"""

from __future__ import annotations

import argparse
import json
import random
from collections import defaultdict
from pathlib import Path

from bank.common import read_jsonl, token_jaccard, write_jsonl

NEAR_DUPLICATE_JACCARD = 0.9


def dedupe(candidates: list[dict], threshold: float = NEAR_DUPLICATE_JACCARD) -> list[dict]:
    """Keep the first of any group of near-identical answers (token Jaccard above threshold)."""
    kept: list[dict] = []
    for c in candidates:
        if all(token_jaccard(c["text"], k["text"]) <= threshold for k in kept):
            kept.append(c)
    return kept


def build_pairs(generations: list[dict], labels: dict[str, dict], seed: int) -> tuple[list[tuple[dict, dict]], dict]:
    """Matched (correct, wrong) pairs and a summary of how many cells qualified."""
    cells: dict[tuple[str, str], list[dict]] = defaultdict(list)
    for g in generations:
        lab = labels[g["candidate_id"]]
        if lab["status"] == "ok":
            cells[(g["problem_id"], g["generator"])].append(g)

    rng = random.Random(seed)
    pairs = []
    n_all_correct = n_all_wrong = 0
    for key in sorted(cells):
        group = sorted(cells[key], key=lambda c: c["candidate_id"])
        # Collapse near-duplicates within each label only: a wrong answer that copies a
        # correct one except for a final slip is exactly the subtle error worth keeping.
        correct = dedupe([c for c in group if labels[c["candidate_id"]]["label"] == 1])
        wrong = dedupe([c for c in group if labels[c["candidate_id"]]["label"] == 0])
        if correct and wrong:
            pairs.append((rng.choice(correct), rng.choice(wrong)))
        elif correct:
            n_all_correct += 1
        elif wrong:
            n_all_wrong += 1

    summary = {
        "cells_with_usable_answers": len(cells),
        "cells_all_correct": n_all_correct,
        "cells_all_wrong": n_all_wrong,
        "pairs": len(pairs),
    }
    return pairs, summary


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--questions", default="bank/dev/questions.jsonl")
    ap.add_argument("--generations", default="bank/dev/generations.jsonl")
    ap.add_argument("--labeled", default="bank/dev/labeled.jsonl")
    ap.add_argument("--out", default="bank/dev")
    ap.add_argument("--seed", type=int, default=544)
    ap.add_argument("--split", default="dev")
    args = ap.parse_args()

    questions = {q["problem_id"]: q for q in read_jsonl(args.questions)}
    generations = read_jsonl(args.generations)
    labels = {r["candidate_id"]: r for r in read_jsonl(args.labeled)}
    pairs, summary = build_pairs(generations, labels, args.seed)

    bank_rows, label_rows = [], []
    for n, (good, bad) in enumerate(pairs):
        pair_id = f"{args.split}-pair-{n:04d}"
        for cand in (good, bad):
            q = questions[cand["problem_id"]]
            bank_rows.append(
                {
                    "candidate_id": cand["candidate_id"],
                    "pair_id": pair_id,
                    "problem_id": cand["problem_id"],
                    "source": q["source"],
                    "domain": q["domain"],
                    "question": q["question"],
                    "candidate": cand["text"],
                    "text_sha256": cand["text_sha256"],
                    "generator": cand["generator"],
                    "generator_revision": cand["generator_revision"],
                    "settings": cand["settings"],
                    "seed": cand["seed"],
                    "checker_version": labels[cand["candidate_id"]].get("checker_version"),
                    "split": args.split,
                }
            )
            lab = labels[cand["candidate_id"]]
            label_rows.append(
                {
                    "candidate_id": cand["candidate_id"],
                    "label": lab["label"],
                    "reference": lab["reference"],
                    "extracted": lab["extracted"],
                    "checker_version": lab.get("checker_version"),
                }
            )

    out = Path(args.out)
    write_jsonl(out / f"{args.split}_bank.jsonl", bank_rows)
    write_jsonl(out / f"{args.split}_labels.jsonl", label_rows)
    summary.update({"questions": len(questions), "generations": len(generations), "candidates": len(bank_rows)})
    (out / f"{args.split}_bank_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
