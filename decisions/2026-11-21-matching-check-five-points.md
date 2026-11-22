# The matching check allows 5 points, and 3 between the pairs

**Date:** 2026-11-21 · **Decided by:** the team, by merging this record · **Seen when decided:** the full pilot results (October 27), the midterm feedback and a simulation on pilot scores; no validation or test data exists · **Post hoc**

## What happened

PLAN.md §5 step 7 counted the final comparison as matched only if both pairs' test TAR (share of correct answers accepted) is within 3 points of 90% and within 3 points of each other. Thresholds are frozen on validation, so test TAR drifts. In a simulation on the pilot's math scores, the old rule called about 30% of comparisons "not matched" by chance alone, which would leave the test without a verdict.

| Each pair within | The two pairs within | Comparisons that count as matched |
|---|---|---|
| 3 points of 90% | 3 points of each other | about 70% |
| 5 points of 90% | 3 points of each other | about 85% |

The simulation draws whole problems again from the pilot's 125 correct and 125 wrong math answers, at the planned full-bank size. It is rough. The final-analysis branch recomputes it with committed code.

## Decision

The comparison counts as matched if **each pair's test TAR is within 5 points of 90% and the two pairs are within 3 points of each other** (point estimates; intervals reported). Otherwise the verdict is the decision rule's first row, as before. Nothing is retuned on test labels.

## Why

- The check looks only at how many correct answers are accepted, never at wrong ones.
- What could bias the comparison is a difference between the two pairs: on the pilot, 3 points more TAR moved a pair's FAR by about 2 to 9 points. So the limit between the pairs stays at 3 points.
- What the wider limit allows is both pairs sitting away from 90% together. Both TARs are reported with the verdict.

## What this means

- The change is made after the pilot and before any validation data exists. It is reported as post hoc, with the old rule's outcome beside it.
- About 15% of comparisons would still count as not matched by chance. The verdict is then "inconclusive (not matched)", with the trade-off curves.
