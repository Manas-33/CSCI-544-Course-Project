# The dynamic-range gate's "best" verifier is the one with the lowest FAR

**Date:** 2026-11-21 · **Decided by:** the team, by merging this record · **Seen when decided:** the full pilot results (October 27) and the midterm feedback · **Clarification of an undefined term, made after seeing both outcomes**

## What happened

PLAN.md §7 asks that "the best single AR verifier's pilot FAR at 90% TAR" lie between 0.10 and 0.70, without defining "best". In pilot math, the lowest-FAR verifier (Qwen2.5-7B) gives 0.696 and passes; the highest-AUROC verifier (Gemma-2-9B) gives 0.744 and fails. Code passes either way (Qwen, 0.616).

## Decision

"Best" means the lowest FAR at 90% TAR. PLAN.md §5 already selects pairs by lowest FAR.

## What this means

Math passes narrowly, so the bank is not rebalanced. The other reading is reported. `analysis/pilot.py` still prints the highest-AUROC reading; the final analysis uses this one.
