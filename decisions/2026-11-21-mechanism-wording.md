# How the mechanism and side results are worded

**Date:** 2026-11-21 · **Decided by:** the team, by merging this record · **Seen when decided:** the full pilot results (October 27), including this analysis run on them; no validation or test data exists · **Clarifications, made after the pilot; none of them changes the H1 verdict**

## What happened

Four phrases in the plan could be read in more than one way. The final analysis needs one reading, fixed before the preregistration freeze.

## Decision

1. **H2 status of a pair** (PLAN.md §5, §8), from κ minus the AR trend:

| Status | When |
|---|---|
| Below the AR trend | The 95% interval is entirely under 0 |
| Above the AR trend (read as "on", PLAN.md §8) | The 95% interval is entirely over 0 |
| On the AR trend | Neither, and the 90% interval lies inside ±0.10 κ |
| Inconclusive | Otherwise; no mechanism claim |

2. **"LLaDA is below gate A's bar on validation"** (PLAN.md §7, §8): any of gate A's three parts fails there (AUROC at least 0.60, 95% lower bound above 0.50, at most 0.10 below Llama-3.2-3B).
3. **"Beats the pseudo-pair of its stronger member"** (PLAN.md §5): a lower test FAR (point estimate), with the two matched on test by the 5- and 3-point limits. The interval is reported beside it.
4. **The reconstruction side experiment** (PLAN.md §4) is on the math bank only: 500 validation and 500 test answers.

## Why 0.10, and what it does

The plan names "on the trend" and "inconclusive" as two outcomes and does not say how to tell them apart. Counting every interval that contains 0 as "on" would turn "not significant" into "no mechanism". 0.10 is a chosen number with no outside source. On synthetic data at the planned size the 90% interval reaches about 0.07–0.09 to each side, so "on" will rarely be met. It changes how the mechanism is worded, never the H1 verdict. On the pilot all math pairs are inconclusive.
