# The reconstruction side experiment uses 128 samples

**Date:** 2026-11-22 · **Decided by:** the team, by merging this record · **Seen when decided:** the reconstruction score on the pilot (`results/reconstruction/pilot/`) and the final model record; the development-bank run had not been made, and no full-bank answer had been scored · **Post hoc**

## What happened

PLAN.md §4 sets the reconstruction score's Monte Carlo sample count on the development bank: the smallest count whose two independent estimates reach a rank correlation of at least 0.95 (about 32 expected; default 128). On the pilot's 250 math answers, LLaDA's two estimates (seeds 544 and 545) reached:

| Samples | 8 | 16 | 32 | 64 | 128 |
|---|---|---|---|---|---|
| Spearman between the two seeds | 0.387 | 0.617 | 0.775 | 0.893 | 0.949 |

At 128 samples the measured value is 0.9489, below 0.95. The committed results call it a fail for the repair rule (`results/reconstruction/pilot/analysis/reconstruction.md`, the final model record). No count up to 128 reaches 0.95 on the pilot.

## Decision

The side experiment (PLAN.md §4: LLaDA's reconstruction score on 500 validation and 500 test answers of the math bank, [record](2026-11-21-mechanism-wording.md)) uses **128 samples**, seed 544, the plan's default. Dream's reconstruction score, if it is run, uses 128 as well. The 0.95 bar is **not** met; 0.9489 is accepted as within the sampling noise of it. A development-bank run, if made, is reported and does not change the count.

## Why

- 0.9489 is a measured rank correlation on 250 answers, and its own uncertainty is larger than the gap to 0.95. Resampling the 250 answers (10,000 times, seed 544) gives a 95% interval of 0.932 to 0.960; resampling the 97 problems gives 0.932 to 0.959.
- Reaching 0.95 for certain would take more samples, at about 34 s per answer for 128 on an L4, for a score that enters no decision.
- Nothing about H1 changes. The repair stays failed as committed, LLaDA keeps its verdict score (final model record), and the side experiment is reported as secondary.

## What this means

- The report gives the measured 0.9489 and says the count was kept after seeing it. It does not say the bar was met.
- The runs use the full bank's scoring GPU type ([record](2026-11-22-scoring-one-gpu-type.md)).
