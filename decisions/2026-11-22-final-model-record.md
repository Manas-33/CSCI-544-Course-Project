# Final model record: go

**Date:** 2026-11-22 · **Decided by:** the team, by merging this record · **Seen when decided:** the full pilot results (October 27), the midterm feedback, the step 0 results (`results/step0/`, an L4), the reconstruction score on the pilot (`results/reconstruction/`) and the pilot label check with all 188 of its read items (`bank/pilot/audit/`; one HumanEval+ second look is outstanding). The full-bank answers have been generated, labeled, paired and split; none has been scored. The full-bank math label check had passed (below); it does not enter this decision

## Decision

**Go** (PLAN.md §7, the row "none of the four fails"). The final test is exactly the table in PLAN.md §5: math; LLaDA-8B with Qwen2.5-7B, Llama-3.1-8B and Gemma-2-9B against Qwen+Llama, Qwen+Gemma and Llama+Gemma. LLaDA is scored with its **verdict score** (T1 Yes/No). No verifier is dropped and Qwen3-8B is not scored. `configs/final_models.json` says the same.

From this merge on, nothing changes the models, pairs, domain, score, threshold procedure or verdict rule (PLAN.md §7).

## The checks

| Check (pilot, math) | LLaDA-8B | Qwen2.5-7B | Llama-3.1-8B | Gemma-2-9B |
|---|---|---|---|---|
| Gate A: T1 AUROC (95% interval) | 0.607 (0.545–0.668); Llama-3.2-3B 0.487 | 0.695 (0.645–0.745) | 0.659 (0.604–0.716) | 0.702 (0.650–0.754) |
| Gate B: T1–T2 spread, math (limit 0.05) | 0.009 | 0.046 | 0.016 | 0.014 |
| Gate B: T1–T2 spread, pooled as run | 0.014 | 0.015 | 0.000 | 0.018 |
| Gate B: pooled without MBPP+ | 0.016 | 0.028 | 0.004 | 0.013 |
| Usable scores: median Yes+No mass | 0.941 | 1.000 | 1.000 | 0.998 |
| Test 4, answer swap (80% needed) | 99 of 100 | 99 of 100 | 99 of 100 | 99 of 100 |
| Test 3, repeat runs | not needed (stage 3) | identical | identical | identical |

- **LLaDA passes gate A narrowly.** Its validation AUROC is reported next to the bar and does not change the test (PLAN.md §7).
- **Reconstruction score:** gate A passes on the pilot (AUROC 0.652), the repeat check fails (rank correlation 0.949, needs 0.95). It is not needed, since the verdict score passes.
- **Dynamic range, math:** the best AR verifier by FAR at 90% TAR is Qwen2.5-7B, 0.696 (PLAN.md §7).

## The pilot label check

By the per-domain rule ([record](2026-11-22-label-check-by-domain.md)); files and results in `bank/pilot/audit/`.

| Part | Read | Reader differs blind | Label errors | Result |
|---|---|---|---|---|
| Math | 100 of 100 | 0 | 0 | **Pass** (one-sided 95% bound 2.95%) |
| HumanEval+ | 41 of 41 | 2 | 1, counted as one until a second look | Rate 2.4%, bound 11.1%; no pass mark |
| MBPP+ | 47 of 59 | 8 | 8 | Not counted; MBPP+ is left out |

- Under the plan's pooled rule the check failed (8 errors in 136, all MBPP+); that is kept in `audit_result.json`.
- One person read all 188 items: 136 on the first sheet and 52 on the second. Both sheets say to work alone; they do not mention AI tools. The team reports the first sheet as read by hand, an answer given after its first score; the second sheet was sent with the instruction to use no AI tools. The same person did the second look at the 9 differences on the first sheet, whose sheet says to work alone and without AI tools (`second_look_*_1.*`).
- The second sheet was written after the per-domain record was merged, as that record requires. `audit_result.json` (the pooled rule) was made before it existed and reproduces only without `human_answers_2.csv`.
- One HumanEval+ difference from the second sheet has had no second look, so it counts as an error in the code rate. It cannot change the math result.
- **Codex:** 199 of 200 replies usable. `a016` was marked invalid by a false alarm about line endings (fixed in the code later) and was read in full like the items without a verdict. Codex used zero reasoning tokens on 100 of 200 items; a rerun of 16 items outside the check (not used, not in these files) showed the same pattern, and the run was kept.

## What follows

- The full bank, by notebook 11's printed counts (its files are committed at the bank freeze): 2,100 math answers in 1,050 pairs (700 validation answers, 1,400 test), and 350 HumanEval+ code answers in 175 pairs (110 validation, 240 test). Both generator caps came out at 40%.
- The full-bank math label check passed by the shortcut: 105 of 200 read, no label error, upper bound 2.88% (Codex agreed with 198 of 200 labels). Its files stay with the label keeper until the join. The math error tags come next, then the bank freeze. Scoring starts after the freeze, on one GPU type ([record](2026-11-22-scoring-one-gpu-type.md)).
