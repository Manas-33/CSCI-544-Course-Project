# LLaDA fails the sanity bar with verified scoring code

**Date:** 2026-10-26 · **Proposed by:** Manas Dalvi (role D, diffusion) · **Review:** roles A–F are not assigned yet, so no separate role C sign-off (PLAN.md §4 asks C and D to sign off) · **Status:** proposed

## What happened

Notebook `02_llada` (an L4; its own Python 3.11.17 environment with torch 2.6.0+cu124 and transformers 4.38.2; LLaDA-8B-Instruct at revision `08b83a6`; 1 answer slot, chosen earlier on the development bank; T1), results in `results/stage3/llada/`:

- **The scoring code matches the official model.** The tiny-model smoke test passed. Scoring-code test 2 passed: on 20 prompts (10 math, 10 code) our first-slot distribution equals the official code's forward pass, same top token and score within 1e-3, with prompt ids identical to the official ones.
- **Test 1 failed: sanity AUROC 0.886** (required ≥ 0.90; math 0.787, code 0.965; median Yes+No mass 0.997). The earlier run through the other scorer, on an A100, gave 0.885. Two scorers, two GPUs, the same result.
- **Test 3: repeat runs identical, batched scores not.** Two unbatched runs gave bit-identical scores. Batched (8) against unbatched differed by up to 0.25 (bound 0.05), with every verdict at score 0 unchanged. See the separate decision on unbatched scoring.
- **Agreement with the saved run.** Scoring the 50 items unbatched with `harness/diffusion` and comparing them with the saved A100 run: same verdict on all 50, Spearman 0.9990, identical overall AUROC (0.8848), but scores differ by up to 0.19, above the 0.05 bound set for this check. Every difference is an exact multiple of 1/16, the BF16 spacing at these logit sizes, so the logits moved by one to three rounding steps between an A100 and an L4. A code difference would not be quantized this way.

## Decision (proposed)

Under PLAN.md §4, a verifier that fails test 1 after its code is signed off as matching the official reference (test 2) is **treated as weak, a gate A failure, rather than an open bug**. Test 2 passed, so **LLaDA counts as a weak verifier and is not eligible for the primary pairs.**

The 0.90 bar is not changed: LLaDA's 0.886 is close to it, but the bar was fixed before any model was scored, and moving it now would be setting a rule after seeing the result.

## What this means

- LLaDA's scores may still be reported descriptively (PLAN.md §5 secondary results), never in H1.
- If no diffusion model passes (iLLaDA's thinking-off rerun is pending; Dream failed test 1 at 0.872), PLAN.md §7's one repair applies: try the reconstruction score as a diffusion model's score, time-boxed to one week. That is a separate decision.
