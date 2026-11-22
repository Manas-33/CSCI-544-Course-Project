# The label check is judged per domain, and MBPP+ leaves the bank

**Date:** 2026-11-22 · **Decided by:** the team, by merging this record · **Seen when decided:** the full pilot results (October 27), the midterm feedback, the step 0 results, and the pilot label check as far as it has been read (136 of 200 items and the second look). The full-bank answers have been generated and are being labeled; none has been scored · **Post hoc** · **Replaces:** the pooled 3-in-200 mark of [the label check record](2026-11-21-label-check-codex-plus-one.md), PLAN.md §7's row for a failed audit, and the code sources in [the full bank's details](2026-11-21-full-bank-details.md)

## What happened

The pilot label check fails as the plan defines it. The reader has read 136 of the 200 items. Their blind verdicts differ from our labels on 9 (math 0 of 63, code 9 of 73), and after the second look 8 of the 9 are label errors. The limit is 3 in 200, and the 64 items not yet read can only add to the 8.

| | In the sample | Read | Label errors |
|---|---|---|---|
| Math | 100 | 63 | 0 |
| Code, HumanEval+ | 41 | 26 | 0 |
| Code, MBPP+ | 59 | 47 | 8, from 5 problems |

In the second look the reader judged that these problems' tests do not match the problem as written: they use inputs the problem does not allow, expect results it does not ask for, or miss allowed inputs on which the program fails; for one item the label rests on a time limit alone. The 5 problems are 5 of the 34 MBPP+ problems read. The read items are not a random sample of MBPP+: disagreements with Codex and GPT-written items were read first. The other MBPP+ problems had no second look.

Under the old rule this failure means: fix the checker, relabel the pilot, drop pairs that no longer hold one correct and one wrong answer, re-audit 200 fresh candidates, and recompute gates A and B and test 4 on the corrected labels. A second failure would cancel H1. This record replaces that for code and keeps it for math.

## Decision

1. **Math and code are judged separately.** H1 and the H2 claims are about math only, and code is descriptive (PLAN.md §8).
2. **Math on the pilot.** The reader, as before blind to our label, reads the 37 math items not yet read. The second look and the count of an error stay as in the earlier record. **It passes only if none of the 100 math items is a label error**: the one-sided 95% upper bound is then 2.95%, below the plan's 4%. One error gives 4.66% and fails.
3. **Math on the full bank.** The check before the freeze takes 200 math answers, stratified by label. Codex, the reader, the shortcut, the second look and the mark of at most 3 errors in 200 stay as they are. A failure means fixing the checker and checking a fresh sample, as before.
4. **If math fails on the pilot,** PLAN.md §7's row applies to math: fix the math checker, relabel the pilot's math answers, drop pairs that no longer hold one correct and one wrong answer, re-audit 100 fresh math answers of the pilot with the mark in point 2, and recompute gates A and B and test 4 in math on the corrected labels. A second failure means H1 is not run and the instructor is told. The fresh answers share pairs with the first sample, so a different person reads them.
5. **Code.** Every label error found is in MBPP+. MBPP+ is left out of the full bank and of the code results in the report, which use HumanEval+ only. Code has no pass mark. The reader also reads the 15 unread HumanEval+ items of the pilot sample, and the measured rate is reported with the code results (no error in 41 gives an upper bound of 7.0%). The full-bank check covers math only, so the full-bank code labels get no check of their own.
6. **The caps are counted on HumanEval+ alone.** MBPP+ answers are removed before the problem cap, the availability count and the generator cap. In the pilot's HumanEval+ pairs GPT-4.1-mini supplies 6 of 67 (9%), so three generators are available and the cap is 40%.
7. **No label is changed by hand** (PLAN.md §2). The 8 wrong labels are not corrected one by one; their answers leave with MBPP+.

## Why

- The pooled mark does not protect math. If math labels had 4% errors and code labels none, reading all 200 would still pass the bank 43% of the time. The math mark in point 2 passes such a bank 1.7% of the time.
- The pooled mark also fails math for code. A second failure would cancel H1 because of labels H1 does not use.
- HumanEval+ had no error in the 26 items read. That is encouraging and not proof, which is why its rate is reported with its bound.

## Known limits

- **The rule changed after this result was seen,** so the math check on the pilot is not independent of the decision to judge math alone. The report states the change and the result under the old rule (failed). The 37 math items are read only after this record is merged.
- **Those 37 are the easier items.** Codex agreed with our label on all of them, since disagreements and unusable verdicts were read first. If the true rate of label errors is 0.5%, 37 random items are all clean 83% of the time, and at 1%, 69%; agreement with Codex makes an error less likely, so these probably understate it. The reader can also work out that Codex agreed, and knows that one error fails the check.
- It is one person's reading, as in the earlier record.

## What this means

- Nothing about the models, pairs, thresholds or score changes. The pilot's math gates and test 4 in math stand if the 37 items are clean; gate B is also judged on the pooled pilot (PLAN.md §7), which includes MBPP+. Recomputed with MBPP+ left out, the T1–T2 spread is Llama 0.004, Gemma 0.013, Qwen 0.028 and LLaDA 0.016, all within the 0.05 limit, so the pooled gate is unchanged for the four verifiers in H1.
- The pilot's code results in the report use HumanEval+ only. The old numbers, with MBPP+, stay in the files as what was run; the report says they include MBPP+ labels found unreliable.
- The code that applies this: the sheet and the per-domain score (`bank/audit_sheet.py`, which notebook 11 does not use) are written now. The HumanEval+-only draw in `bank/draw_full.py` is changed after notebook 11 finishes, because notebook 11 runs from `main`.
- The label keeper's run continues. Math answers are unaffected. MBPP+ answers already generated and labeled stay in the private folder, unused.
- The change of scope (MBPP+ out, code without its own full-bank check) is raised with the instructor before the final test is run (PLAN.md §9).
