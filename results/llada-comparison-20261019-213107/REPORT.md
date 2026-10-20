# AR + diffusion: comparison on the original 50 sanity candidates

Reused saved BF16 Qwen/Mistral scores; only LLaDA was run.

| Verifier | Overall AUROC | Sanity AUROC >= 0.90 |
|---|---:|---|
| Qwen | 0.7728 | False |
| Mistral | 0.7552 | False |
| LLaDA | 0.8848 | False |

| Pair (AND at score zero) | Correct accepted / 25 | Wrong accepted / 25 | TAR | FAR |
|---|---:|---:|---:|---:|
| Qwen+Mistral | 10 | 7 | 0.400 | 0.280 |
| Qwen+LLaDA | 11 | 7 | 0.440 | 0.280 |
| Mistral+LLaDA | 24 | 13 | 0.960 | 0.520 |

These exploratory thresholds do not match pair TAR and cannot establish superiority. The sanity fixture is not a validation or research test set. Thresholds and prompts were not fitted to its labels. The existing AR models already fail the overall sanity target.

LLaDA slots: 1, selected on separate dev candidates by median Yes+No mass.
Reference/repeat checks passed: True; full project gates remain pending.

See `single_metrics.csv` for domain AUROCs, `pair_metrics.csv` for all pair counts, and `comparison_review.csv` for aligned candidates, labels and three scores.

Scientific comparison still requires pilot/validation gates, validation-selected thresholds for matched TAR, and a held-out evaluation. No cost-matching claim is made.
  