# The full bank is scored on one GPU type, A100 or L4

**Date:** 2026-11-22 · **Decided by:** the team, by merging this record · **Seen when decided:** the full pilot results (October 27) and the midterm feedback, and the step 0 results (`results/step0/`, an L4); no full-bank answer has been scored · **Replaces:** "on an A100" for the full-bank scoring in NEXT_STEPS.md (C.3, D.3) and notebook 12

## What happened

NEXT_STEPS.md asked for an A100 for all full-bank scoring. Colab often has no A100 free, and waiting for one would hold up the main experiment. PLAN.md asks for the same GPU type only where costs are compared (§5 step 4, §9), and the [unbatched-scoring record](2026-10-26-diffusion-scored-unbatched.md) for scores that are compared; neither needs an A100.

## Decision

1. The whole full bank, math and code, validation and test, T1 and T2, is scored on **one GPU type for every verifier: A100 or L4, whichever the first scoring run gets**. Any other type is refused afterwards, also for full-bank reconstruction scores. A100s with 40 and 80 GB count as one type.
2. Scores and costs are compared only within that type. BF16 scores differ between GPU types: in step 0 single pilot scores moved by up to 1.5 from the A100 to the L4 (Qwen2.5-7B), while AUROC and FAR at 90% TAR of the three H1 AR verifiers moved by at most 0.01. A100- and L4-seconds are never compared (PLAN.md §9).
3. The pilot's cost ratio (at most 1.45, A100) stays the basis for calling the pairs' costs comparable (PLAN.md §5 step 4). The ratio measured on the scoring GPU is reported with the result, and the 1.5 limit applies to it as written there.
4. Starting over on the other type means scoring everything again. It is allowed only before the preregistration commit (PLAN.md §5 step 5), and never because of what the scores show.

## What this means

- Nothing about H1 changes: not the models, pairs, domain, score, thresholds, matching check or verdict rule.
- All full-bank scoring runs from one Google account, because the type and the finished runs are kept in its Drive.
- `harness/full_run.py` keeps the type in `gpu.txt` next to the scores, checks it before anything is scored and again for every score file in the completeness check (notebook 13). Notebook 12 follows it for full-bank answers. The rule for the pilot's reconstruction scores is unchanged.
- The pilot gates come from A100 scores (Dream: L4) and are not recomputed on the scoring GPU.
- If the repair is chosen, LLaDA's reconstruction score of the whole math bank is made on the same type: about 4.5 hours on an A100, about 14 on an L4 (estimated).
