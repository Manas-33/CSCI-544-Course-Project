# Validation step

Final test: AR+diffusion pair versus AR+AR pair (verdict scores), math. Selected on validation at 90% pair TAR:

| Group | Pair | Validation TAR | Validation FAR | Wins in resamples | Selected |
|---|---|---|---|---|---|
| ar_ar | Gemma-2-9B + Qwen2.5-7B | 0.900 | 0.649 (227/350) | 0.80 | **yes** |
| ar_ar | Llama-3.1-8B + Qwen2.5-7B | 0.900 | 0.671 (235/350) | 0.19 |  |
| ar_ar | Gemma-2-9B + Llama-3.1-8B | 0.903 | 0.717 (251/350) | 0.01 |  |
| ar_diffusion | LLaDA-8B + Qwen2.5-7B | 0.903 | 0.669 (234/350) | 0.81 | **yes** |
| ar_diffusion | Gemma-2-9B + LLaDA-8B | 0.900 | 0.697 (244/350) | 0.19 |  |
| ar_diffusion | LLaDA-8B + Llama-3.1-8B | 0.903 | 0.746 (261/350) | 0.00 |  |

- Best single verifier: Qwen2.5-7B
- Cost: ar_ar 0.1477 s, ar_diffusion 0.1150 s; ratio 1.28 (limit 1.5), GPU NVIDIA A100-SXM4-80GB; comparable-cost claim: True
- LLaDA re-check (reported, not applied): AUROC 0.622 [0.588, 0.657], Llama-3.2-3B 0.517; meets gate A's bar

## Single verifiers on validation (T1, or LLaDA's score in the final test)

| Domain | Score | AUROC [95% CI] | FAR at 90% TAR |
|---|---|---|---|
| math | Qwen2.5-7B | 0.701 [0.668, 0.734] | 0.666 |
| math | Gemma-2-9B | 0.685 [0.651, 0.719] | 0.720 |
| math | Qwen2.5-1.5B | 0.660 [0.625, 0.696] | 0.686 |
| math | Llama-3.1-8B | 0.645 [0.609, 0.682] | 0.777 |
| math | LLaDA-8B | 0.622 [0.588, 0.657] | 0.743 |
| math | Dream-7B | 0.598 [0.566, 0.629] | 0.740 |
| math | Mistral-7B | 0.592 [0.559, 0.625] | 0.783 |
| math | Llama-3.2-3B | 0.517 [0.481, 0.554] | 0.854 |
| code | Gemma-2-9B | 0.671 [0.567, 0.773] | 0.764 |
| code | Dream-7B | 0.668 [0.583, 0.755] | 0.727 |
| code | Qwen2.5-7B | 0.649 [0.577, 0.728] | 0.673 |
| code | Llama-3.1-8B | 0.648 [0.571, 0.732] | 0.800 |
| code | Llama-3.2-3B | 0.596 [0.520, 0.671] | 0.818 |
| code | Qwen2.5-1.5B | 0.564 [0.491, 0.641] | 0.855 |
| code | Mistral-7B | 0.556 [0.491, 0.630] | 0.891 |
| code | LLaDA-8B | 0.555 [0.509, 0.607] | 0.836 |
