# Diffusion models are scored unbatched

**Date:** 2026-10-26 · **Proposed by:** Manas Dalvi (role D, diffusion) · **Review:** roles A–F are not assigned yet · **Status:** proposed

## What happened

Scoring-code test 3 (PLAN.md §4) has two parts: repeat runs must agree (Spearman > 0.999) and batched scores must match unbatched ones (|Δs| < 0.05).

| Model | Repeat runs | Batched (8) vs unbatched | Results |
|---|---|---|---|
| LLaDA | identical (difference 0) | up to 0.250, no verdict at 0 changed | `results/stage3/llada/` |
| Dream | identical (difference 0) | up to 0.250 | `results/stage3/dream/` |
| iLLaDA | identical | not run: iLLaDA is always scored unbatched | `results/stage3/illada/` |

The batched differences have the signature of BF16 rounding, not of a padding leak:

- the batching code itself is exact: the tiny-model smoke test runs the official model code in FP32 with left padding and gets batched-vs-unbatched differences of about 1e-6 for LLaDA at 1, 8 and 32 slots;
- for LLaDA, every difference between scores of the same item is an exact multiple of 1/16, the BF16 spacing at these logit sizes: one to four rounding steps, which is what a different matrix shape (batch size, padding) changes;
- a leak of padded positions into the answer slot would produce irregular, larger differences.

## Decision (proposed)

Compute **every diffusion-model score unbatched** (batch size 1), as PLAN.md §3 already does for iLLaDA. Unbatched scores are bit-identical across repeat runs, so test 3's repeat part covers them; the batched part no longer applies because nothing is scored batched. The 0.05 bound is not changed.

## What this means

- `harness/diffusion/run_score.py` is run with `--batch-size 1` for LLaDA and Dream from now on; the notebooks' agreement step already does.
- Cost: unbatched scoring is slower per candidate. PLAN.md §5 step 4 already measures cost at batch size 1 for both pairs whenever a pair includes iLLaDA; this extends that to every diffusion model, so the cost comparison stays like for like if the AR models are also timed at batch size 1 (the AR harness scores at batch size 1 already).
- A score computed on a different GPU type can differ by a few BF16 rounding steps. Scores that are compared with each other (pairs, thresholds, test runs) must come from the same GPU type, as PLAN.md §9 already requires for costs.
