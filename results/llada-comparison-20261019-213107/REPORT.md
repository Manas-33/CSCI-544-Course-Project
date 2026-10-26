# AR + diffusion: comparison on the 50 sanity candidates

Qwen/Mistral: the saved BF16 + SDPA run `results/bf16/sdpa/ar-sanity-bf16-sdpa-20261026-000459`. LLaDA: its own saved run `results/llada-comparison-20261019-213107` (1 answer slot, T1). Nothing was rescored for this comparison.

| Verifier | Overall AUROC | Math AUROC | Code AUROC | Sanity AUROC ≥ 0.90 |
|---|---:|---:|---:|---|
| Qwen | 0.920 | 0.876 | 1.000 | PASS |
| Mistral | 0.752 | 0.604 | 0.944 | FAIL |
| LLaDA | 0.885 | 0.781 | 0.965 | FAIL |

| Pair (AND at score zero) | Correct accepted / 25 | Wrong accepted / 25 | TAR | FAR |
|---|---:|---:|---:|---:|
| Qwen+Mistral | 24 | 7 | 0.960 | 0.280 |
| Qwen+LLaDA | 25 | 7 | 1.000 | 0.280 |
| Mistral+LLaDA | 24 | 13 | 0.960 | 0.520 |

The zero threshold gives each pair a different TAR, so these FARs cannot be compared with each other and establish nothing about H1. The sanity fixture is a scoring check, not a validation or test set; no threshold or prompt was fitted to its labels.

LLaDA's answer-slot count was chosen on the development bank by median Yes+No mass. Its run's repeat check (the same forward pass run twice on 20 items) passed: True. Still pending: PLAN.md §4 test 2 (reference parity against the official model code, `harness/diffusion/checks/check_parity.py`), test 3's batch invariance, and agreement between this run and the shared scorer `harness/diffusion` (`--parity`).

See `single_metrics.csv`, `pair_metrics.csv` and `comparison_review.csv` for every count and score.
