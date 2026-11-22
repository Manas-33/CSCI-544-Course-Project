# One final test, settled before scoring

**Date:** 2026-11-21 · **Decided by:** the team, by merging this record · **Seen when decided:** the full pilot results (October 27) and the midterm feedback; no validation or test data exists · **Post hoc**

## What happened

After the midterm the eligibility rules could still be read in more than one way, and the readings led to different models and domains in the primary comparison. Under the rules before version 1.3, the primary test could still become LLaDA on math, Dream on code, a reconstruction-score comparison, or an AR-only study, partly depending on LLaDA's AUROC on validation, after the bank is scored. LLaDA's pilot value is 0.607 against a 0.60 bar, so that re-check is close to a coin flip.

## Decision

1. The final test is the table in PLAN.md §5: math, T1 scores, three AR+diffusion pairs (LLaDA with Qwen2.5, Llama-3.1, Gemma-2) against three AR+AR pairs among Qwen2.5, Llama-3.1 and Gemma-2.
2. Go/no-go is made once, on the pilot, before any full-bank answer is scored. What is left is the label audit, test 4 in math and the AR repeat-run check. The result goes in a final model record; after it is merged nothing changes the models, the domain or the score.
3. LLaDA's validation re-check is reported and not applied. If LLaDA is below the bar there, H1 is reported with that fact and no mechanism claim is made from H2.
4. H1 is math only. The code rule and the switch to code are removed.
5. Pair costs are judged comparable from the pilot (at most 1.45×, A100), so no cost matching can replace H1 (before: above 1.5× a cost-matched comparison replaced it).

## What this means

- The only outcome still open is the go/no-go, and it is fixed before scoring. Its possible results and what each does are listed in PLAN.md §7.
- Points 3 to 5 loosen or remove rules after seeing the pilot. They make the test settled. They are not expected to help the AR+diffusion pair: at the same acceptance of correct answers, a weaker verifier's pair accepts more wrong ones.
- Other readings (T1–T3, the 0.90 sanity bar, the highest-AUROC dynamic range) are reported as background and change no verdict.
