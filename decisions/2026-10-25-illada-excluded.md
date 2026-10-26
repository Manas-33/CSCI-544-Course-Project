# iLLaDA is excluded: it fails the usable-scores gate with thinking off

**Date:** 2026-10-25 · **Decided by:** Manas Dalvi (role D, diffusion) · **Review:** roles A–F are not assigned yet, so no separate role C review (PLAN.md §11) · **Follows:** [iLLaDA is scored with thinking off](2026-10-25-illada-thinking-off.md), which fixed this outcome before the rerun

## What happened

Notebook `03_illada` rerun with thinking off (an L4; torch 2.11.0, transformers 4.57.1; iLLaDA-8B-Instruct at revision `5769f04`), results in `results/stage3/illada-thinking-off/`.

Median Yes+No mass on the 142 development answers (the usable-scores gate needs ≥ 0.5, PLAN.md §7):

| Answer prefix | 1 slot | 8 slots | 32 slots |
|---|---|---|---|
| `none` | 0.0002 | 0.0004 | 0.0001 |
| `think_nl` (`<think>\n</think>\n`) | 0.005 | 0.013 | 0.003 |
| `think_nlnl` (`<think>\n\n</think>\n\n`) | 0.136 | **0.284** | 0.267 |

- The rule chose `think_nlnl` with 8 slots. **The gate fails: 0.284 < 0.5.**
- On the sanity set with that setting, the median Yes+No mass is 0.227, and only 1 of 50 items reaches 0.5. The most likely first token is "The" on 40 items, "Yes" on 8, "No" on 1 and "Is" on 1. With thinking off, iLLaDA starts a sentence rather than answering Yes or No.
- Scoring-code test 1 (sanity AUROC ≥ 0.90) fails at 0.728 (math 0.533, code 0.889), the same as with thinking on (0.730).
- The scorer is correct: the smoke test, test 2 (parity with the official code, with the prefix) and test 3 (identical repeat runs) all pass.

## Decision

**iLLaDA is excluded from the verdict interface**, the primary interface of PLAN.md §4: it is not scored on the pilot and is not in any primary pair. This is the outcome the thinking-off record set in advance for a model that still fails the gate.

**Why not PLAN.md §7's own repair.** §7 says to switch the answer words to "Correct/Incorrect" for every verifier and every template, once, before excluding a verifier. That repair is not applied here, as the thinking-off record already noted:
- iLLaDA's remaining problem is that it opens with a sentence ("The…"), and changing the answer words does not change that;
- the repair would change the prompt of every verifier, including those that pass the gate.

## What this does not decide

- **iLLaDA can still come back through PLAN.md §7's one repair.** If no diffusion model passes gates A and B, the plan allows one time-boxed repair: the reconstruction score instead of a verdict. Which model gets it is a separate decision.
- **This is a gate outcome to report**, not a hidden failure. It goes in the reports as "we defined a go/no-go gate; here is what passed it" (PLAN.md §10).

## Where the diffusion verifiers stand

| Model | Usable-scores gate (dev bank) | Test 2 (parity) | Test 1 (sanity AUROC ≥ 0.90) | Status |
|---|---|---|---|---|
| LLaDA-8B-Instruct | passes (0.978, 1 slot) | passes | 0.886, fails | weak verifier, a gate A failure ([proposed](2026-10-26-llada-fails-sanity.md)) |
| iLLaDA-8B-Instruct | fails (0.284 with thinking off) | passes | 0.728, fails | **excluded** (this record) |
| Dream-v0-Instruct-7B | passes (0.941, 1 slot) | passes | 0.872, fails | weak by the same rule as LLaDA (no record yet) |

So **no diffusion model passes the sanity bar** under the verdict interface. PLAN.md §7's one repair, the reconstruction score for one diffusion model (time-boxed to a week), is now the next step; it needs its own decision. Results: `results/stage3/llada/`, `results/stage3/illada/`, `results/stage3/illada-thinking-off/`, `results/stage3/dream/`.
