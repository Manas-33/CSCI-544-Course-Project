# The math bank adds problems from the rest of the MATH test set

**Date:** 2026-11-21 · **Decided by:** the team, by merging this record · **Seen when decided:** the full pilot results (October 27) and the midterm feedback, including the pilot's scores split by math source · **Post hoc, fixed before any new problem is generated**

## What happened

The pilot used GSM8K-Platinum and MATH-500. The unused problems (1,059 and 239) would give about 1,600 math answers at the pilot's uncapped yield, most of them from GSM8K.

## Decision

- Add numeric-answer problems from the MATH test set outside MATH-500 (`prm800k`'s `math_splits/train.jsonl`, ids starting `test/`).
- Keep the pilot's mix: half the math pairs from GSM8K-Platinum, half from MATH (MATH-500 first). This gives about 2,100 math answers, and the whole bank about 3,000, within the approved 2,000–5,000.
- PLAN.md §6 option (a): report the smallest detectable ΔFAR, about 3–5.5 points, with the verdict.

## Why the mix is fixed

On the pilot LLaDA's AUROC was 0.65 on GSM8K-Platinum (95% interval 0.57–0.75) and 0.55 on MATH-500 (0.46–0.64). Filling the bank with MATH would probably put LLaDA below gate A's bar, and filling it with GSM8K would put it above more easily. The gates were judged on an even mix, so the full bank keeps it. Results are reported per source.

## Known limits

PRM800K trained on these MATH problems. The file's fields and license, and the terms for redistributing MATH text, are to be confirmed at download; if unclear, the released bank lists these problems by id.
