# The side experiment is not frozen with the preregistration

**Date:** 2026-11-22 · **Decided by:** the team, by merging this record · **Seen when decided:** the main validation step (`results/final/validation.md`); no test label joined; the side experiment not yet scored · **Post hoc** (a change of order)

## What happened

`analysis/README.md` freezes the reconstruction side experiment (PLAN.md §4: LLaDA's reconstruction score on 500 validation and 500 test math answers, [record](2026-11-22-reconstruction-128-samples.md)) in the same analysis manifest as the final test. Its scores do not exist yet; making them takes about 2½ hours on the scoring GPU. The team chose not to hold the preregistration and the final test for it.

## Decision

1. The analysis manifest is frozen without the side experiment (`"side": null`). Everything in it, H1 included, is unchanged by this record.
2. The side experiment is still run, later, with the settings already recorded (128 samples, seed 544, the same 500 + 500 draw, A100), and is reported as **exploratory**, not preregistered.
3. It is reported on its 500 validation answers. Its test half is evaluated only if a later record allows a second use of the test labels for it, and the report then says so.

## Why

It is a secondary result that enters no decision, and waiting for it would hold up the final test.

## Known limits

- The decision was made after the main validation step had been run. The side experiment does not depend on it, and nothing in the manifest changes.
- It loses its preregistered status. PLAN.md §9 lists dropping the reconstruction part entirely as a change of approved scope; this keeps it, as exploratory, and the report says so.
