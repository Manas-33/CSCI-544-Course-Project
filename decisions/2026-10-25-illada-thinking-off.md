# iLLaDA is scored with thinking off

**Date:** 2026-10-25 · **Decided by:** Manas Dalvi (role D, diffusion) · **Review:** roles A–F are not assigned yet, so no separate role C review (PLAN.md §11) · **Status:** decided; the rerun is pending

## What happened

Notebook `03_illada` (an L4; torch 2.11.0, transformers 4.57.1; iLLaDA-8B-Instruct at revision `5769f04`), results in `results/stage3/illada/`:

- The scorer is correct: the tiny-model smoke test passed; scoring-code test 2 (our first-slot distribution against the official code's forward pass, 20 prompts) passed; test 3's repeat runs were identical.
- **The usable-scores gate failed** (PLAN.md §7: median Yes+No mass ≥ 0.5 under T1 on the development bank). The median Yes+No mass was 0.0002, 0.0004 and 0.0001 with 1, 8 and 32 answer slots.
- In the sanity run (8 slots, the count that run chose) the most likely token at the first answer slot is `<` on all 50 items. Test 1 (sanity AUROC ≥ 0.90) failed at 0.730.

**Cause.** iLLaDA opens its answers with a `<think>…</think>` block. In its official evaluation (`eval_illada.sh` and `eval_illada.py` in ML-GSAI/LLaDA, commit `9182493`), the math and reasoning tasks generate 2,048–4,096 tokens and boost the end-of-thinking text `</think>`, and the answer parsers keep the text after the last `</think>`. Its short-answer tasks (`mmlu_generative`, `mmlu_redux_generative`) instead generate 3–4 tokens after 5-shot multi-turn examples, with no think block. So reading Yes/No at the first slot of a zero-shot prompt reads the start of the thinking, not an answer.

## Decision

Score iLLaDA with **thinking off**: append an empty think block after the chat template's generation prompt and before the answer slots (`harness/diffusion/scorer.py`, `ANSWER_PREFIXES`). The user message, the T1–T3 wording, is unchanged and identical for every verifier; only the start of iLLaDA's own turn changes. The official evaluation has no example of an empty think block, so this is our construction, chosen to match how its parsers read answers (after `</think>`).

The exact text is chosen on the development bank, together with the answer-slot count, by the rule PLAN.md §4 uses for slots:

- candidates: `none` (""), `think_nl` (`<think>\n</think>\n`) and `think_nlnl` (`<think>\n\n</think>\n\n`), each with 1, 8 and 32 slots. Both think variants put a newline before `</think>`, so the closing tag tokenizes as the official end-of-thinking text does (`</`, `think`, `>`); `<think></think>` would not, so it is not a candidate;
- the combination with the highest median Yes+No mass wins; ties go to fewer slots, then to the prefix listed first;
- labels are never read (`harness/diffusion/choose_slots.py`).

Tests 1–3 then run with the chosen prefix (test 2's official prompt ids get the same prefix, so it checks the forward pass and slot reading; it cannot judge the prefix itself). The other diffusion models keep `none`.

## Why, and how it departs from PLAN.md

- **It follows the plan's intent.** PLAN.md turns thinking off wherever a model has a thinking mode: generators (§2, "Thinking modes are off") and the Qwen3-8B baseline (§3, "thinking off"). iLLaDA's chat template has no switch for it, so an empty think block is the way to do it.
- **Deviation 1: §7's stated repair.** For a usable-scores failure, §7 says to switch the answer words to "Correct/Incorrect" for every verifier and every template, once. That does not address a model that starts by thinking, and it would change the prompt for every model.
- **Deviation 2: a per-model piece of prompt text chosen on the development bank.** §2 and §4 list what the development bank may set, and §4 says it is "never used to choose a template per verifier". The prefix is not a template: the shared T1–T3 question is untouched, the prefix only opens the model's own turn, and it is chosen on the Yes+No mass format check alone, exactly as the answer-slot count is, never on labels or AUROC.
- **What had been seen.** The usable-scores result (a format check) and, in the same run, test 1's AUROC (0.730, computed from scores with almost no Yes/No mass). The prefix choice uses neither labels nor AUROC. The gate values (0.5 mass, 0.90 sanity AUROC) are unchanged.

## What happens next

1. Rerun notebook 03 with the prefix candidates. Results go to `results/stage3/illada-thinking-off/`; the first run stays in `results/stage3/illada/`.
2. If the chosen combination passes the usable-scores gate, iLLaDA goes on to tests 1–3 and, if it passes them, to the pilot, with the prefix fixed.
3. If it still fails, iLLaDA is excluded as PLAN.md §7 says for a verifier that stays unusable, unless the team instead applies §7's "Correct/Incorrect" repair to every verifier. That would be a separate decision record.
