"""Step 3: label each generated answer with two independent checkers (PLAN.md §2).

Checker 1: the last \\boxed{...}, normalized as a number, compared with the GSM8K reference.
Checker 2: Math-Verify if installed (pip install math-verify); otherwise the last number
in the whole answer. Labels stay fully programmatic: an answer is usable only if
checker 1 found an answer, the generation was not cut off, and both checkers agree.

    python -m bank.label --questions bank/dev/questions.jsonl \\
        --generations bank/dev/generations.jsonl --out bank/dev/labeled.jsonl
"""

from __future__ import annotations

import argparse
from collections import Counter

from bank.common import (
    extract_boxed,
    extract_last_number,
    has_identity_string,
    normalize_number,
    parse_boxed_number,
    read_jsonl,
    write_jsonl,
)

# Bump when labeling logic changes; recorded with every label and bank row (PLAN.md §9).
LABELER_VERSION = "bank.label/2"


def checker_boxed(text: str, reference: str) -> tuple[bool | None, str | None]:
    """(correct?, extracted answer). None means no usable \\boxed{} answer."""
    extracted = parse_boxed_number(extract_boxed(text))
    if extracted is None:
        return None, None
    return extracted == reference, extracted


def _math_verify():
    try:
        from math_verify import parse, verify  # type: ignore

        return parse, verify
    except ImportError:
        return None


def checker_second(text: str, reference: str, use_math_verify: bool = True) -> tuple[bool | None, str]:
    """(correct?, checker name). Independent of checker 1's extraction logic."""
    mv = _math_verify() if use_math_verify else None
    if mv is not None:
        parse, verify = mv
        gold = parse(f"${reference}$")
        pred = parse(text)
        if not pred:
            return None, "math_verify"
        return bool(verify(gold, pred)), "math_verify"
    extracted = normalize_number(extract_last_number(text))
    if extracted is None:
        return None, "last_number"
    return extracted == reference, "last_number"


def label_row(gen: dict, reference: str, use_math_verify: bool = True) -> dict:
    c1, extracted = checker_boxed(gen["text"], reference)
    c2, c2_name = checker_second(gen["text"], reference, use_math_verify)
    if gen.get("finish_reason") == "length":
        status = "truncated"
    elif has_identity_string(gen["text"]):
        status = "identity_string"
    elif c1 is None:
        status = "no_answer"
    elif c2 is None or c1 != c2:
        status = "checker_disagree"
    else:
        status = "ok"
    return {
        "candidate_id": gen["candidate_id"],
        "problem_id": gen["problem_id"],
        "reference": reference,
        "extracted": extracted,
        "checker1_boxed": c1,
        "checker2": c2,
        "checker2_name": c2_name,
        "status": status,
        "label": int(c1) if status == "ok" else None,
        "checker_version": f"{LABELER_VERSION}+{c2_name}",
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--questions", default="bank/dev/questions.jsonl")
    ap.add_argument("--generations", default="bank/dev/generations.jsonl")
    ap.add_argument("--out", default="bank/dev/labeled.jsonl")
    ap.add_argument("--no-math-verify", action="store_true", help="Use the last-number checker instead")
    args = ap.parse_args()

    refs = {q["problem_id"]: q["reference"] for q in read_jsonl(args.questions)}
    gens = read_jsonl(args.generations)
    labeled = [label_row(g, refs[g["problem_id"]], not args.no_math_verify) for g in gens]
    write_jsonl(args.out, labeled)

    counts = Counter(r["status"] for r in labeled)
    correct = sum(1 for r in labeled if r["label"] == 1)
    wrong = sum(1 for r in labeled if r["label"] == 0)
    print(f"{len(labeled)} answers: {dict(counts)}; usable correct={correct}, wrong={wrong}")
    if labeled:
        print(f"Second checker: {labeled[0]['checker2_name']}")


if __name__ == "__main__":
    main()
