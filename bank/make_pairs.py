"""Step 4: build matched pairs and write the bank and label files (PLAN.md §2, §9).

For each (question, generator) cell with at least one usable correct and one usable
wrong answer, keep one of each. Each cell draws from its own seed (the string
"<seed>:<problem_id>:<generator>"), so excluding one question never changes another
question's pair, and each pair's ID comes from its question, so IDs never shift either.
Near-duplicate answers are collapsed first, within the correct answers and within the
wrong answers separately. The two rows of a pair are written in order of their answer
text's SHA-256, which has nothing to do with the label. Questions listed in the
exclusion file (broken or ambiguous, with a reason) are dropped entirely; no label is
ever set by hand. Labels go to a separate file from the bank, as PLAN.md §9 requires,
so scoring code never sees them.

    python -m bank.make_pairs --questions bank/dev/questions.jsonl \\
        --generations bank/dev/generations.jsonl --labeled bank/dev/labeled.jsonl --out bank/dev
"""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import random
from collections import Counter, defaultdict
from pathlib import Path

from bank.common import read_jsonl, token_jaccard, write_jsonl

NEAR_DUPLICATE_JACCARD = 0.9
EXCLUSIONS_FILENAME = "excluded_questions.txt"


def dedupe(candidates: list[dict], threshold: float = NEAR_DUPLICATE_JACCARD) -> list[dict]:
    """Keep the first of any group of near-identical answers (token Jaccard above threshold)."""
    kept: list[dict] = []
    for c in candidates:
        if all(token_jaccard(c["text"], k["text"]) <= threshold for k in kept):
            kept.append(c)
    return kept


def load_exclusions(path: str | Path) -> dict[str, str]:
    """problem_id -> reason from a file of "problem_id  # reason" lines. Missing file: none."""
    p = Path(path)
    if not p.exists():
        return {}
    out = {}
    for line in p.read_text().splitlines():
        body, _, reason = line.partition("#")
        if body.strip():
            out[body.strip()] = reason.strip()
    return out


def order_pair(a: dict, b: dict) -> tuple[dict, dict]:
    """The pair's two rows in order of answer-text SHA-256: the same whichever one is correct."""
    return tuple(sorted((a, b), key=lambda c: (c["text_sha256"], c["candidate_id"])))


def cell_tag(candidate: dict) -> str:
    """Short generator name used in candidate IDs ("<problem_id>__<tag>__<k>")."""
    parts = candidate["candidate_id"].rsplit("__", 2)
    return parts[1] if len(parts) == 3 else candidate["generator"].replace("/", "_")


def pair_tag(a: dict, b: dict) -> str:
    """The pair's generator tag, taken from both rows so the pair ID never depends on which is correct."""
    tag_a, tag_b = cell_tag(a), cell_tag(b)
    if tag_a != tag_b:
        raise SystemExit(
            f"Pair {a['candidate_id']} / {b['candidate_id']} mixes generator tags ({tag_a}, {tag_b}); "
            "give each generator run its own --tag"
        )
    return tag_a


def _require_unique(ids: list[str], what: str) -> None:
    seen, dupes = set(), set()
    for i in ids:
        (dupes if i in seen else seen).add(i)
    if dupes:
        raise SystemExit(f"Duplicate {what} (would silently overwrite data): {sorted(dupes)[:5]}")


def build_pairs(
    generations: list[dict],
    labels: dict[str, dict],
    seed: int,
    exclude: set[str] | frozenset[str] = frozenset(),
) -> tuple[list[tuple[dict, dict]], dict]:
    """Matched (correct, wrong) pairs and a summary of how many cells qualified."""
    cells: dict[tuple[str, str], list[dict]] = defaultdict(list)
    for g in generations:
        lab = labels[g["candidate_id"]]
        if lab["status"] == "ok":
            cells[(g["problem_id"], g["generator"])].append(g)

    pairs = []
    n_all_correct = n_all_wrong = n_excluded = 0
    for key in sorted(cells):
        if key[0] in exclude:
            n_excluded += 1
            continue
        # A string seed is hashed with SHA-512 by Python, so this is stable across runs.
        rng = random.Random(f"{seed}:{key[0]}:{key[1]}")
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
        "cells_excluded": n_excluded,
        "cells_all_correct": n_all_correct,
        "cells_all_wrong": n_all_wrong,
        "pairs": len(pairs),
    }
    return pairs, summary


def _sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--questions", default="bank/dev/questions.jsonl")
    ap.add_argument("--generations", default="bank/dev/generations.jsonl")
    ap.add_argument("--labeled", default="bank/dev/labeled.jsonl")
    ap.add_argument("--out", default="bank/dev")
    ap.add_argument("--seed", type=int, default=544)
    ap.add_argument("--split", default="dev")
    ap.add_argument(
        "--exclude",
        default=None,
        help=f'File of "problem_id  # reason" lines to drop. Default: {EXCLUSIONS_FILENAME} next to --questions.',
    )
    args = ap.parse_args(argv)
    out = Path(args.out)

    if args.exclude is None:
        # Exclusions belong to the question set, so look beside it, whatever --out is.
        exclude_path = Path(args.questions).parent / EXCLUSIONS_FILENAME
        if not exclude_path.is_file():
            print(f"No exclusion file at {exclude_path}; no questions excluded.")
    else:
        exclude_path = Path(args.exclude)
        if not exclude_path.is_file():
            raise SystemExit(f"Exclusion file not found: {exclude_path}")

    questions = {q["problem_id"]: q for q in read_jsonl(args.questions)}
    generations = read_jsonl(args.generations)
    labeled_rows = read_jsonl(args.labeled)
    _require_unique([g["candidate_id"] for g in generations], "candidate_id in generations")
    _require_unique([r["candidate_id"] for r in labeled_rows], "candidate_id in labels")
    labels = {r["candidate_id"]: r for r in labeled_rows}
    exclusions = load_exclusions(exclude_path)
    unknown = sorted(set(exclusions) - set(questions))
    if unknown:
        raise SystemExit(f"Excluded IDs not in the question set (typo?): {unknown}")
    pairs, summary = build_pairs(generations, labels, args.seed, exclude=set(exclusions))

    bank_rows, label_rows = [], []
    for good, bad in pairs:
        pair_id = f"{args.split}:{good['problem_id']}:{pair_tag(good, bad)}"
        for cand in order_pair(good, bad):
            q = questions[cand["problem_id"]]
            lab = labels[cand["candidate_id"]]
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
                    "checker_version": lab.get("checker_version"),
                    "split": args.split,
                }
            )
            label_rows.append(
                {
                    "candidate_id": cand["candidate_id"],
                    "label": lab["label"],
                    "reference": lab["reference"],
                    "extracted": lab["extracted"],
                    "status": lab["status"],
                    "checker1_boxed": lab.get("checker1_boxed"),
                    "checker2": lab.get("checker2"),
                    "checker2_name": lab.get("checker2_name"),
                    "checker_version": lab.get("checker_version"),
                }
            )

    pair_counts = Counter(r["pair_id"] for r in bank_rows)
    shared = sorted(p for p, n in pair_counts.items() if n != 2)
    if shared:
        raise SystemExit(f"Duplicate pair_id (two pairs share a question and generator tag): {shared[:5]}")
    bank_path, labels_path = out / f"{args.split}_bank.jsonl", out / f"{args.split}_labels.jsonl"
    write_jsonl(bank_path, bank_rows)
    write_jsonl(labels_path, label_rows)
    summary.update(
        {
            "questions": len(questions),
            "generations": len(generations),
            "candidates": len(bank_rows),
            "excluded_questions": exclusions,
            "exclusions_file": str(exclude_path) if exclusions else None,
            "exclusions_sha256": _sha256_file(exclude_path) if exclude_path.exists() else None,
            "inputs": {
                Path(p).name: _sha256_file(Path(p)) for p in (args.questions, args.generations, args.labeled)
            },
            "outputs": {bank_path.name: _sha256_file(bank_path), labels_path.name: _sha256_file(labels_path)},
            "seed": args.seed,
            "python": platform.python_version(),
        }
    )
    (out / f"{args.split}_bank_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps({k: v for k, v in summary.items() if k not in ("inputs", "outputs")}, indent=2))


if __name__ == "__main__":
    main()
