# Sanity set (50 items)

The bug detector for scoring-code test 1 (PLAN.md §4): every verifier, AR and diffusion, must reach **AUROC ≥ 0.90** on these 50 items with T1 before any of its scores are used. A lower value almost always means a scoring bug (wrong mask id, missing Dream shift, wrong chat template, padding leak). It is a test fixture: never tune anything on it and never report it as a result.

- **Contents:** 25 obviously right (`label: 1`) and 25 obviously wrong (`label: 0`) answers. 26 are math (13 right, 13 wrong) and 24 are code (12 right, 12 wrong).
- **Wrong items** each have one clear error that changes the answer:
  - math: a wrong sum or product, or an added discount;
  - code: a flipped condition or a wrong operator.
- **Outside the bank:** new problems, not copied or paraphrased from GSM8K, MATH, HumanEval or MBPP.
- **Format:** one JSON object per line with `candidate_id`, `domain`, `problem`, `candidate`, `label` and `note` (why the item is right or wrong). It follows the spec in `harness/diffusion/README.md` and keeps its four examples (`sanity-m001`, `sanity-m002`, `sanity-c001`, `sanity-c002`) unchanged.
  - Math answers end with `The final answer is \boxed{...}`.
  - Code answers are a single Python function.
  - Lengths range from one line to a paragraph, so batching adds padding.
- **IDs** are `sanity-m001…m026` and `sanity-c001…c024`. They don't follow the labels, so an ID says nothing about whether an item is right.

## Checking the labels

`bank/tests/test_sanity_set.py` checks every label by machine:
- it compares each math answer with an independently worked-out true answer;
- it runs each code answer against test cases.

```bash
python -m pytest bank/tests/test_sanity_set.py -q
```

**A second person still has to read all 50 items** and confirm each label and note before any model is scored on the set (PLAN.md §4). Record who checked it, and when, in `decisions/`.

## Using it

Both scorers read `candidate_id`, `problem` and `candidate`. The sanity check reads `label` from this same file. The AR harness (`harness/run.py`, `harness/evaluate.py --sanity`) and the diffusion checks (`harness/diffusion/checks/`) both use it; the diffusion checks use it by default.
