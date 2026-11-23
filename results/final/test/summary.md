# Final analysis

**Status: test split, frozen thresholds.** Bootstrap over whole problems, 10,000 replicates, seed 544, percentile intervals, thresholds fixed.

## Final test: AR+diffusion pair versus AR+AR pair (verdict scores) (math)

**Verdict:** Inconclusive; report the interval and the smallest effect the data could have detected

| Row | Pair or verifier | TAR (counts) [95% CI] | FAR (counts) [95% CI] |
|---|---|---|---|
| selected AR+AR | Gemma-2-9B + Qwen2.5-7B | 0.877 (614/700) [0.851, 0.902] | 0.649 (454/700) [0.611, 0.685] |
| selected AR+diffusion | LLaDA-8B + Qwen2.5-7B | 0.884 (619/700) [0.859, 0.909] | 0.670 (469/700) [0.634, 0.705] |
| best single verifier (validation threshold) | Qwen2.5-7B | 0.886 (620/700) [0.860, 0.910] | 0.667 (467/700) [0.630, 0.703] |
| same-model pseudo-pair (T1 + T2) | Gemma-2-9B + Gemma-2-9B (T2) (T2 median Yes+No mass 1.00) | 0.870 (609/700) [0.841, 0.897] | 0.689 (482/700) [0.652, 0.724] |
| same-model pseudo-pair (T1 + T2) | Llama-3.1-8B + Llama-3.1-8B (T2) (T2 median Yes+No mass 1.00) | 0.853 (597/700) [0.825, 0.880] | 0.700 (490/700) [0.664, 0.735] |
| same-model pseudo-pair (T1 + T2) | Qwen2.5-7B + Qwen2.5-7B (T2) (T2 median Yes+No mass 1.00) | 0.886 (620/700) [0.861, 0.909] | 0.670 (469/700) [0.633, 0.706] |
| same-model pseudo-pair (T1 + T2) | LLaDA-8B + LLaDA-8B (T2) (T2 median Yes+No mass 0.55) | 0.906 (634/700) [0.882, 0.928] | 0.771 (540/700) [0.739, 0.803] |
| ideal thresholds (test-tuned), AR+AR | Gemma-2-9B + Qwen2.5-7B | 0.900 (630/700) [0.876, 0.922] | 0.706 (494/700) [0.671, 0.740] |
| ideal thresholds (test-tuned), AR+diffusion | LLaDA-8B + Qwen2.5-7B | 0.900 (630/700) [0.876, 0.923] | 0.700 (490/700) [0.664, 0.735] |

- **ΔFAR = FAR(AR+diffusion) − FAR(AR+AR):** +2.1 points, 95% [-0.3, 4.6], 90% [0.1, 4.2]
- **Smallest effect these data could have detected** (80% power, 5% two-sided): 3.5 points
- **Matching check:** off 90% by 1.6, 2.3 points (limit 5), gap 0.7 points (limit 3, 95% [-1.4, 2.7]): passes
- **Matching check under the rule before 2026-11-21** (3 points and 3 points; reported beside it, changes nothing): passes, which would give: Inconclusive; report the interval and the smallest effect the data could have detected
- **Ideal thresholds (test-tuned; no verdict uses it):** ΔFAR -0.6 points, 95% [-3.0, 1.8], 90% [-2.6, 1.4]
- **AR+AR pair against the best single verifier:** FAR difference -1.9 points, 95% [-4.3, 0.6], 90% [-3.9, 0.3]
- **AR+diffusion pair against the best single verifier:** FAR difference +0.3 points, 95% [-2.1, 2.7], 90% [-1.7, 2.3]
- **Selection repeated in the bootstrap (secondary):** choice only, 95% [-0.8, 5.9]; choice and thresholds, 95% [-3.3, 7.7] points
- **LLaDA below gate A's bar on validation:** False; **comparable-cost claim:** True
- **AR+AR pair against the pseudo-pair of its stronger member (Qwen2.5-7B):** FAR difference -2.1 points, 95% [-4.5, 0.1], 90% [-4.1, -0.1]; TAR 0.877 and 0.886 (matched); beats it: True
- **AR+diffusion pair against the pseudo-pair of its stronger member (Qwen2.5-7B):** FAR difference +0.0 points, 95% [-2.3, 2.3], 90% [-1.9, 1.9]; TAR 0.884 and 0.886 (matched); beats it: False

**Reading (PLAN.md §8):** H2 for the selected pair is *inconclusive*. Underpowered. Estimate and interval only.

- Caveat: H2 is inconclusive: no mechanism claim
- Caveat: the AR+diffusion pair does not beat the T1+T2 pseudo-pair of its stronger member at a matched TAR, so a gain does not count as diversity

| Pair TAR | ΔFAR (AR+diffusion − AR+AR) | TAR AR+diffusion / AR+AR |
|---|---|---|
| 80% | +2.1 points, 95% [-0.9, 5.1], 90% [-0.4, 4.7] | 0.789 / 0.773 |
| 90% | +2.1 points, 95% [-0.3, 4.6], 90% [0.1, 4.2] | 0.884 / 0.877 |
| 95% | +0.7 points, 95% [-1.5, 2.9], 90% [-1.1, 2.6] | 0.930 / 0.929 |

## Pairs at 90% pair TAR

| Domain | Pair | Kind | Roles | TAR | FAR [95% CI] | s / candidate | GPU |
|---|---|---|---|---|---|---|---|
| code | LLaDA-8B (T1/T2 mean) + Qwen2.5-7B (T1/T2 mean) | AR+diffusion | template_avg | 0.900 | 0.567 [0.473, 0.659] | 0.185 | A100 |
| code | Dream-7B + Qwen2.5-7B | AR+diffusion | e1 | 0.867 | 0.600 [0.500, 0.695] | 0.084 | A100 |
| code | Llama-3.1-8B (T1/T2 mean) + Qwen2.5-7B (T1/T2 mean) | AR+AR | template_avg | 0.875 | 0.625 [0.526, 0.720] | 0.175 | A100 |
| code | Llama-3.1-8B + Qwen2.5-7B | AR+AR | h1_ar, e1, h2_trend | 0.875 | 0.633 [0.541, 0.721] | 0.087 | A100 |
| code | Llama-3.2-3B + Qwen2.5-7B | AR+AR | h2_trend | 0.792 | 0.658 [0.571, 0.742] | 0.074 | A100 |
| code | Gemma-2-9B + Qwen2.5-7B | AR+AR | h1_ar, e1, h2_trend | 0.925 | 0.675 [0.587, 0.758] | 0.124 | A100 |
| code | Gemma-2-9B (T1/T2 mean) + Qwen2.5-7B (T1/T2 mean) | AR+AR | template_avg | 0.933 | 0.675 [0.577, 0.765] | 0.250 | A100 |
| code | Gemma-2-9B (T1/T2 mean) + Llama-3.1-8B (T1/T2 mean) | AR+AR | template_avg | 0.883 | 0.683 [0.583, 0.775] | 0.257 | A100 |
| code | Gemma-2-9B + Llama-3.1-8B | AR+AR | h1_ar, h2_trend | 0.892 | 0.692 [0.596, 0.781] | 0.128 | A100 |
| code | LLaDA-8B + Llama-3.1-8B | AR+diffusion | h1_mixed | 0.850 | 0.692 [0.602, 0.774] | 0.095 | A100 |
| code | LLaDA-8B + Qwen2.5-7B | AR+diffusion | h1_mixed, e1 | 0.933 | 0.692 [0.602, 0.776] | 0.092 | A100 |
| code | Gemma-2-9B + Llama-3.2-3B | AR+AR | h2_trend | 0.792 | 0.692 [0.603, 0.772] | 0.115 | A100 |
| code | LLaDA-8B (T1/T2 mean) + Llama-3.1-8B (T1/T2 mean) | AR+diffusion | template_avg | 0.850 | 0.717 [0.621, 0.805] | 0.192 | A100 |
| code | Dream-7B + Llama-3.1-8B | AR+diffusion | background | 0.883 | 0.717 [0.628, 0.800] | 0.088 | A100 |
| code | Mistral-7B + Qwen2.5-7B | AR+AR | e1, h2_trend | 0.958 | 0.725 [0.642, 0.803] | 0.089 | A100 |
| code | Llama-3.1-8B + Qwen2.5-1.5B | AR+AR | h2_trend | 0.858 | 0.733 [0.643, 0.815] | 0.077 | A100 |
| code | Dream-7B + Mistral-7B | AR+diffusion | background | 0.892 | 0.733 [0.644, 0.815] | 0.090 | A100 |
| code | Dream-7B + Gemma-2-9B | AR+diffusion | background | 0.908 | 0.742 [0.653, 0.821] | 0.126 | A100 |
| code | Gemma-2-9B + LLaDA-8B | AR+diffusion | h1_mixed | 0.917 | 0.750 [0.658, 0.832] | 0.133 | A100 |
| code | Gemma-2-9B + Qwen2.5-1.5B | AR+AR | h2_trend | 0.908 | 0.750 [0.660, 0.831] | 0.115 | A100 |
| code | Llama-3.2-3B + Qwen2.5-1.5B | AR+AR | h2_trend | 0.792 | 0.750 [0.664, 0.826] | 0.064 | A100 |
| code | Llama-3.1-8B + Mistral-7B | AR+AR | h2_trend | 0.892 | 0.758 [0.675, 0.836] | 0.092 | A100 |
| code | Gemma-2-9B + Mistral-7B | AR+AR | h2_trend | 0.950 | 0.775 [0.686, 0.854] | 0.130 | A100 |
| code | Llama-3.2-3B + Mistral-7B | AR+AR | h2_trend | 0.800 | 0.775 [0.693, 0.849] | 0.079 | A100 |
| code | Gemma-2-9B (T1/T2 mean) + LLaDA-8B (T1/T2 mean) | AR+diffusion | template_avg | 0.883 | 0.783 [0.692, 0.865] | 0.266 | A100 |
| code | Mistral-7B + Qwen2.5-1.5B | AR+AR | h2_trend | 0.908 | 0.842 [0.767, 0.906] | 0.079 | A100 |
| code | LLaDA-8B + Mistral-7B | AR+diffusion | background | 0.958 | 0.917 [0.858, 0.966] | 0.097 | A100 |
| math | Gemma-2-9B (T1/T2 mean) + Qwen2.5-7B (T1/T2 mean) | AR+AR | template_avg | 0.864 | 0.641 [0.604, 0.678] | 0.298 | A100 |
| math | Gemma-2-9B + Qwen2.5-1.5B | AR+AR | h2_trend | 0.873 | 0.647 [0.609, 0.683] | 0.127 | A100 |
| math | Gemma-2-9B + Qwen2.5-7B | AR+AR | h1_ar, e1, h2_trend | 0.877 | 0.649 [0.611, 0.685] | 0.148 | A100 |
| math | Llama-3.1-8B + Qwen2.5-7B | AR+AR | h1_ar, e1, h2_trend | 0.884 | 0.660 [0.623, 0.696] | 0.110 | A100 |
| math | Dream-7B + Qwen2.5-7B | AR+diffusion | e1 | 0.887 | 0.663 [0.626, 0.699] | 0.106 | A100 |
| math | LLaDA-8B + Qwen2.5-7B | AR+diffusion | h1_mixed, e1 | 0.884 | 0.670 [0.634, 0.705] | 0.115 | A100 |
| math | Llama-3.1-8B (T1/T2 mean) + Qwen2.5-7B (T1/T2 mean) | AR+AR | template_avg | 0.893 | 0.679 [0.643, 0.713] | 0.223 | A100 |
| math | Dream-7B + Gemma-2-9B | AR+diffusion | background | 0.884 | 0.679 [0.642, 0.714] | 0.148 | A100 |
| math | LLaDA-8B (T1/T2 mean) + Qwen2.5-7B (T1/T2 mean) | AR+diffusion | template_avg | 0.893 | 0.686 [0.649, 0.721] | 0.233 | A100 |
| math | Llama-3.1-8B + Qwen2.5-1.5B | AR+AR | h2_trend | 0.886 | 0.687 [0.651, 0.722] | 0.089 | A100 |
| math | Mistral-7B + Qwen2.5-7B | AR+AR | e1, h2_trend | 0.893 | 0.690 [0.653, 0.725] | 0.112 | A100 |
| math | Gemma-2-9B + LLaDA-8B | AR+diffusion | h1_mixed | 0.877 | 0.691 [0.655, 0.727] | 0.157 | A100 |
| math | Gemma-2-9B (T1/T2 mean) + LLaDA-8B (T1/T2 mean) | AR+diffusion | template_avg | 0.877 | 0.694 [0.657, 0.730] | 0.317 | A100 |
| math | Gemma-2-9B (T1/T2 mean) + Llama-3.1-8B (T1/T2 mean) | AR+AR | template_avg | 0.881 | 0.696 [0.659, 0.731] | 0.306 | A100 |
| math | Gemma-2-9B + Llama-3.1-8B | AR+AR | h1_ar, h2_trend | 0.880 | 0.703 [0.668, 0.737] | 0.152 | A100 |
| math | Dream-7B + Llama-3.1-8B | AR+diffusion | background | 0.894 | 0.706 [0.669, 0.740] | 0.110 | A100 |
| math | LLaDA-8B (T1/T2 mean) + Llama-3.1-8B (T1/T2 mean) | AR+diffusion | template_avg | 0.880 | 0.709 [0.672, 0.744] | 0.241 | A100 |
| math | Gemma-2-9B + Mistral-7B | AR+AR | h2_trend | 0.886 | 0.720 [0.685, 0.754] | 0.154 | A100 |
| math | Mistral-7B + Qwen2.5-1.5B | AR+AR | h2_trend | 0.904 | 0.723 [0.688, 0.757] | 0.091 | A100 |
| math | Llama-3.2-3B + Qwen2.5-7B | AR+AR | h2_trend | 0.894 | 0.726 [0.692, 0.758] | 0.087 | A100 |
| math | LLaDA-8B + Llama-3.1-8B | AR+diffusion | h1_mixed | 0.893 | 0.734 [0.699, 0.768] | 0.119 | A100 |
| math | Gemma-2-9B + Llama-3.2-3B | AR+AR | h2_trend | 0.884 | 0.734 [0.700, 0.767] | 0.129 | A100 |
| math | Llama-3.2-3B + Qwen2.5-1.5B | AR+AR | h2_trend | 0.899 | 0.734 [0.700, 0.767] | 0.066 | A100 |
| math | Dream-7B + Mistral-7B | AR+diffusion | background | 0.909 | 0.746 [0.709, 0.779] | 0.113 | A100 |
| math | Llama-3.1-8B + Mistral-7B | AR+AR | h2_trend | 0.904 | 0.754 [0.720, 0.787] | 0.116 | A100 |
| math | LLaDA-8B + Mistral-7B | AR+diffusion | background | 0.907 | 0.770 [0.737, 0.802] | 0.121 | A100 |
| math | Llama-3.2-3B + Mistral-7B | AR+AR | h2_trend | 0.914 | 0.819 [0.788, 0.848] | 0.093 | A100 |

Pooled (weights {'code': 120, 'math': 700}): selected AR+diffusion minus selected AR+AR: +2.1 points, 95% [-0.1, 4.3].

## Single verifiers

| Domain | Score | AUROC [95% CI] | ECE | Median Yes+No mass | FAR at 90% TAR | s / candidate | GPU |
|---|---|---|---|---|---|---|---|
| code | Qwen2.5-7B | 0.774 [0.716, 0.829] | 0.306 | 1.000 | 0.550 | 0.041 | NVIDIA A100-SXM4-80GB |
| code | Gemma-2-9B | 0.649 [0.586, 0.713] | 0.402 | 0.999 | 0.742 | 0.083 | NVIDIA A100-SXM4-80GB |
| code | Dream-7B | 0.632 [0.575, 0.691] | 0.048 | 0.911 | 0.708 | 0.043 | NVIDIA A100-SXM4-80GB |
| code | Llama-3.1-8B | 0.615 [0.559, 0.674] | 0.306 | 1.000 | 0.667 | 0.045 | NVIDIA A100-SXM4-80GB |
| code | Qwen2.5-1.5B | 0.586 [0.542, 0.632] | 0.408 | 0.997 | 0.833 | 0.032 | NVIDIA A100-SXM4-80GB |
| code | Llama-3.2-3B | 0.568 [0.512, 0.627] | 0.306 | 1.000 | 0.742 | 0.032 | NVIDIA A100-SXM4-80GB |
| code | Mistral-7B | 0.554 [0.493, 0.616] | 0.493 | 1.000 | 0.883 | 0.047 | NVIDIA A100-SXM4-80GB |
| code | LLaDA-8B | 0.521 [0.468, 0.574] | 0.229 | 0.108 | 0.900 | 0.050 | NVIDIA A100-SXM4-80GB |
| code | Qwen2.5-7B (T2) | 0.745 [0.685, 0.804] | 0.313 | 1.000 | 0.542 | 0.042 | NVIDIA A100-SXM4-80GB |
| code | Gemma-2-9B (T2) | 0.643 [0.586, 0.703] | 0.393 | 1.000 | 0.742 | 0.083 | NVIDIA A100-SXM4-80GB |
| code | Dream-7B (T2) | 0.625 [0.566, 0.686] | 0.065 | 0.881 | 0.708 | 0.043 | NVIDIA A100-SXM4-80GB |
| code | Llama-3.1-8B (T2) | 0.609 [0.550, 0.668] | 0.254 | 0.999 | 0.750 | 0.046 | NVIDIA A100-SXM4-80GB |
| code | Llama-3.2-3B (T2) | 0.567 [0.508, 0.628] | 0.456 | 1.000 | 0.750 | 0.032 | NVIDIA A100-SXM4-80GB |
| code | Qwen2.5-1.5B (T2) | 0.563 [0.510, 0.615] | 0.466 | 0.998 | 0.842 | 0.032 | NVIDIA A100-SXM4-80GB |
| code | Mistral-7B (T2) | 0.538 [0.478, 0.597] | 0.495 | 1.000 | 0.933 | 0.049 | NVIDIA A100-SXM4-80GB |
| code | LLaDA-8B (T2) | 0.536 [0.484, 0.592] | 0.157 | 0.095 | 0.800 | 0.051 | NVIDIA A100-SXM4-80GB |
| math | Qwen2.5-7B | 0.699 [0.676, 0.723] | 0.364 | 1.000 | 0.667 | 0.053 | NVIDIA A100-SXM4-80GB |
| math | Gemma-2-9B | 0.664 [0.640, 0.689] | 0.417 | 0.997 | 0.693 | 0.095 | NVIDIA A100-SXM4-80GB |
| math | Qwen2.5-1.5B | 0.622 [0.597, 0.648] | 0.441 | 0.984 | 0.709 | 0.032 | NVIDIA A100-SXM4-80GB |
| math | LLaDA-8B | 0.613 [0.590, 0.638] | 0.346 | 0.863 | 0.767 | 0.062 | NVIDIA A100-SXM4-80GB |
| math | Llama-3.1-8B | 0.613 [0.589, 0.638] | 0.227 | 1.000 | 0.750 | 0.057 | NVIDIA A100-SXM4-80GB |
| math | Dream-7B | 0.599 [0.578, 0.621] | 0.283 | 0.905 | 0.739 | 0.053 | NVIDIA A100-SXM4-80GB |
| math | Mistral-7B | 0.582 [0.559, 0.605] | 0.483 | 0.999 | 0.791 | 0.059 | NVIDIA A100-SXM4-80GB |
| math | Llama-3.2-3B | 0.524 [0.499, 0.550] | 0.302 | 1.000 | 0.853 | 0.034 | NVIDIA A100-SXM4-80GB |
| math | Qwen2.5-7B (T2) | 0.701 [0.677, 0.725] | 0.376 | 1.000 | 0.696 | 0.054 | NVIDIA A100-SXM4-80GB |
| math | Gemma-2-9B (T2) | 0.668 [0.644, 0.692] | 0.390 | 1.000 | 0.699 | 0.096 | NVIDIA A100-SXM4-80GB |
| math | Dream-7B (T2) | 0.630 [0.605, 0.655] | 0.203 | 0.992 | 0.663 | 0.055 | NVIDIA A100-SXM4-80GB |
| math | LLaDA-8B (T2) | 0.629 [0.605, 0.655] | 0.344 | 0.551 | 0.763 | 0.064 | NVIDIA A100-SXM4-80GB |
| math | Llama-3.1-8B (T2) | 0.614 [0.590, 0.638] | 0.195 | 1.000 | 0.697 | 0.058 | NVIDIA A100-SXM4-80GB |
| math | Qwen2.5-1.5B (T2) | 0.606 [0.581, 0.630] | 0.431 | 0.985 | 0.739 | 0.032 | NVIDIA A100-SXM4-80GB |
| math | Mistral-7B (T2) | 0.590 [0.567, 0.613] | 0.480 | 1.000 | 0.754 | 0.061 | NVIDIA A100-SXM4-80GB |
| math | Llama-3.2-3B (T2) | 0.506 [0.480, 0.532] | 0.456 | 1.000 | 0.871 | 0.034 | NVIDIA A100-SXM4-80GB |
| pooled | Qwen2.5-7B | 0.711 [0.689, 0.732] | 0.353 | – | – | – | – |
| pooled | Gemma-2-9B | 0.658 [0.636, 0.680] | 0.414 | – | – | – | – |
| pooled | Qwen2.5-1.5B | 0.616 [0.594, 0.638] | 0.436 | – | – | – | – |
| pooled | Llama-3.1-8B | 0.600 [0.580, 0.620] | 0.234 | – | – | – | – |
| pooled | LLaDA-8B | 0.597 [0.577, 0.618] | 0.321 | – | – | – | – |
| pooled | Dream-7B | 0.585 [0.568, 0.603] | 0.244 | – | – | – | – |
| pooled | Mistral-7B | 0.576 [0.555, 0.598] | 0.484 | – | – | – | – |
| pooled | Llama-3.2-3B | 0.528 [0.507, 0.548] | 0.301 | – | – | – | – |
| pooled | Qwen2.5-7B (T2) | 0.706 [0.684, 0.728] | 0.366 | – | – | – | – |
| pooled | Gemma-2-9B (T2) | 0.664 [0.641, 0.686] | 0.390 | – | – | – | – |
| pooled | Dream-7B (T2) | 0.616 [0.596, 0.637] | 0.174 | – | – | – | – |
| pooled | Llama-3.1-8B (T2) | 0.608 [0.587, 0.629] | 0.204 | – | – | – | – |
| pooled | LLaDA-8B (T2) | 0.603 [0.583, 0.624] | 0.311 | – | – | – | – |
| pooled | Qwen2.5-1.5B (T2) | 0.590 [0.570, 0.611] | 0.435 | – | – | – | – |
| pooled | Mistral-7B (T2) | 0.582 [0.560, 0.604] | 0.482 | – | – | – | – |
| pooled | Llama-3.2-3B (T2) | 0.515 [0.492, 0.539] | 0.455 | – | – | – | – |

## H2 (code): overlap on wrong answers

Trend over 13 cross-family AR+AR pairs: κ = -0.919 + 1.936 × mean AUROC (slope 95% [0.280, 3.470]). "On the trend" needs a 90% interval inside ±0.1.

| Pair | Kind | κ [95% CI] | Spearman | CAPA (wrong) | CAPA (all) | κ minus trend [95% CI] | Status |
|---|---|---|---|---|---|---|---|
| Mistral-7B + Qwen2.5-7B | AR+AR | 0.072 [-0.032, 0.197] | 0.247 | 0.032 | 0.374 | -0.295 [-0.395, -0.162] | in the fit |
| Llama-3.1-8B + Mistral-7B | AR+AR | 0.088 [-0.032, 0.232] | 0.402 | 0.011 | -0.473 | -0.125 [-0.216, -0.034] | in the fit |
| Gemma-2-9B + Mistral-7B | AR+AR | 0.097 [-0.031, 0.270] | 0.401 | 0.091 | 0.637 | -0.148 [-0.240, -0.046] | in the fit |
| Llama-3.2-3B + Mistral-7B | AR+AR | 0.097 [-0.031, 0.267] | 0.453 | 0.008 | -0.565 | -0.069 [-0.176, 0.032] | in the fit |
| Llama-3.1-8B + Qwen2.5-1.5B | AR+AR | 0.220 [0.035, 0.394] | 0.506 | 0.022 | -0.381 | -0.024 [-0.159, 0.110] | in the fit |
| Gemma-2-9B + Qwen2.5-1.5B | AR+AR | 0.242 [0.045, 0.439] | 0.549 | 0.138 | 0.571 | -0.034 [-0.165, 0.113] | in the fit |
| Llama-3.2-3B + Qwen2.5-1.5B | AR+AR | 0.242 [0.042, 0.435] | 0.472 | 0.017 | -0.462 | 0.044 [-0.102, 0.191] | in the fit |
| Mistral-7B + Qwen2.5-1.5B | AR+AR | 0.296 [0.039, 0.550] | 0.486 | 0.127 | 0.812 | 0.111 [-0.047, 0.281] | in the fit |
| Llama-3.1-8B + Qwen2.5-7B | AR+AR | 0.379 [0.191, 0.551] | 0.589 | 0.194 | -0.012 | -0.047 [-0.165, 0.094] | in the fit |
| Llama-3.2-3B + Qwen2.5-7B | AR+AR | 0.456 [0.246, 0.633] | 0.615 | 0.165 | -0.091 | 0.077 [-0.063, 0.219] | in the fit |
| Gemma-2-9B + Llama-3.2-3B | AR+AR | 0.460 [0.217, 0.655] | 0.529 | 0.079 | -0.273 | 0.202 [-0.007, 0.384] | in the fit |
| Gemma-2-9B + Llama-3.1-8B | AR+AR | 0.522 [0.338, 0.685] | 0.590 | 0.114 | -0.185 | 0.218 [0.048, 0.383] | in the fit |
| Gemma-2-9B + Qwen2.5-7B | AR+AR | 0.547 [0.360, 0.706] | 0.699 | 0.466 | 0.568 | 0.089 [-0.012, 0.209] | in the fit |
| LLaDA-8B + Qwen2.5-7B | AR+diffusion | 0.049 [-0.075, 0.193] | 0.130 | 0.048 | 0.107 | -0.286 [-0.432, -0.123] | below the AR trend |
| LLaDA-8B + Llama-3.1-8B | AR+diffusion | 0.079 [-0.069, 0.240] | 0.179 | 0.037 | -0.037 | -0.102 [-0.295, 0.092] | inconclusive |
| Gemma-2-9B + LLaDA-8B | AR+diffusion | 0.148 [-0.017, 0.337] | 0.155 | 0.018 | 0.138 | -0.064 [-0.255, 0.130] | inconclusive |
| LLaDA-8B + Mistral-7B | AR+diffusion | 0.151 [-0.045, 0.487] | 0.337 | 0.034 | 0.181 | 0.029 [-0.248, 0.356] | inconclusive |
| Dream-7B + Mistral-7B | AR+diffusion | 0.204 [0.041, 0.401] | 0.365 | 0.015 | 0.043 | -0.026 [-0.160, 0.148] | inconclusive |
| Dream-7B + Llama-3.1-8B | AR+diffusion | 0.515 [0.317, 0.689] | 0.447 | 0.039 | 0.020 | 0.227 [0.017, 0.428] | above the AR trend |
| Dream-7B + Qwen2.5-7B | AR+diffusion | 0.528 [0.369, 0.675] | 0.556 | 0.104 | 0.083 | 0.085 [-0.049, 0.242] | inconclusive |
| Dream-7B + Gemma-2-9B | AR+diffusion | 0.604 [0.393, 0.767] | 0.638 | 0.116 | 0.097 | 0.284 [0.097, 0.454] | above the AR trend |

Within error type *wrong_output* (120 wrong answers, descriptive):

| AR+diffusion pair | κ | κ minus trend [95% CI] | Status |
|---|---|---|---|
| Gemma-2-9B + LLaDA-8B | 0.148 | -0.064 [-0.255, 0.130] | inconclusive |
| LLaDA-8B + Llama-3.1-8B | 0.079 | -0.102 [-0.295, 0.092] | inconclusive |
| LLaDA-8B + Qwen2.5-7B | 0.049 | -0.286 [-0.432, -0.123] | below the AR trend |
| Dream-7B + Qwen2.5-7B | 0.528 | 0.085 [-0.049, 0.242] | inconclusive |
| Dream-7B + Gemma-2-9B | 0.604 | 0.284 [0.097, 0.454] | above the AR trend |
| Dream-7B + Llama-3.1-8B | 0.515 | 0.227 [0.017, 0.428] | above the AR trend |
| Dream-7B + Mistral-7B | 0.204 | -0.026 [-0.160, 0.148] | inconclusive |
| LLaDA-8B + Mistral-7B | 0.151 | 0.029 [-0.248, 0.356] | inconclusive |

## H2 (math): overlap on wrong answers

Trend over 13 cross-family AR+AR pairs: κ = -0.636 + 1.581 × mean AUROC (slope 95% [0.747, 2.405]). "On the trend" needs a 90% interval inside ±0.1.

| Pair | Kind | κ [95% CI] | Spearman | CAPA (wrong) | CAPA (all) | κ minus trend [95% CI] | Status |
|---|---|---|---|---|---|---|---|
| Llama-3.2-3B + Qwen2.5-1.5B | AR+AR | 0.234 [0.149, 0.316] | 0.217 | 0.051 | 0.361 | -0.036 [-0.100, 0.025] | in the fit |
| Llama-3.2-3B + Qwen2.5-7B | AR+AR | 0.240 [0.157, 0.324] | 0.290 | 0.210 | 0.331 | -0.091 [-0.154, -0.027] | in the fit |
| Gemma-2-9B + Llama-3.2-3B | AR+AR | 0.281 [0.196, 0.366] | 0.351 | 0.235 | 0.403 | -0.023 [-0.088, 0.043] | in the fit |
| Llama-3.2-3B + Mistral-7B | AR+AR | 0.285 [0.178, 0.390] | 0.274 | 0.092 | 0.407 | 0.047 [-0.017, 0.105] | in the fit |
| Mistral-7B + Qwen2.5-7B | AR+AR | 0.305 [0.217, 0.388] | 0.380 | 0.115 | 0.582 | -0.072 [-0.136, -0.006] | in the fit |
| Gemma-2-9B + Qwen2.5-1.5B | AR+AR | 0.328 [0.242, 0.412] | 0.425 | 0.109 | 0.696 | -0.053 [-0.120, 0.018] | in the fit |
| Gemma-2-9B + Mistral-7B | AR+AR | 0.333 [0.245, 0.416] | 0.458 | 0.215 | 0.774 | -0.017 [-0.082, 0.049] | in the fit |
| Llama-3.1-8B + Mistral-7B | AR+AR | 0.342 [0.247, 0.434] | 0.415 | 0.090 | 0.437 | 0.034 [-0.037, 0.105] | in the fit |
| Llama-3.1-8B + Qwen2.5-1.5B | AR+AR | 0.353 [0.266, 0.439] | 0.338 | 0.052 | 0.391 | 0.012 [-0.058, 0.083] | in the fit |
| Mistral-7B + Qwen2.5-1.5B | AR+AR | 0.366 [0.279, 0.448] | 0.398 | 0.141 | 0.864 | 0.050 [-0.023, 0.125] | in the fit |
| Llama-3.1-8B + Qwen2.5-7B | AR+AR | 0.368 [0.284, 0.447] | 0.405 | 0.218 | 0.376 | -0.033 [-0.089, 0.026] | in the fit |
| Gemma-2-9B + Qwen2.5-7B | AR+AR | 0.497 [0.421, 0.568] | 0.642 | 0.410 | 0.657 | 0.056 [0.002, 0.112] | in the fit |
| Gemma-2-9B + Llama-3.1-8B | AR+AR | 0.499 [0.416, 0.576] | 0.525 | 0.273 | 0.456 | 0.125 [0.060, 0.193] | in the fit |
| LLaDA-8B + Qwen2.5-7B | AR+diffusion | 0.338 [0.254, 0.419] | 0.379 | 0.173 | 0.476 | -0.063 [-0.139, 0.015] | inconclusive |
| Gemma-2-9B + LLaDA-8B | AR+diffusion | 0.356 [0.271, 0.438] | 0.397 | 0.212 | 0.598 | -0.017 [-0.090, 0.057] | on the AR trend |
| Dream-7B + Llama-3.1-8B | AR+diffusion | 0.362 [0.275, 0.447] | 0.381 | 0.103 | 0.292 | 0.040 [-0.040, 0.118] | inconclusive |
| LLaDA-8B + Llama-3.1-8B | AR+diffusion | 0.392 [0.304, 0.479] | 0.329 | 0.127 | 0.346 | 0.060 [-0.020, 0.141] | inconclusive |
| LLaDA-8B + Mistral-7B | AR+diffusion | 0.398 [0.307, 0.488] | 0.395 | 0.153 | 0.684 | 0.090 [0.007, 0.176] | above the AR trend |
| Dream-7B + Mistral-7B | AR+diffusion | 0.401 [0.312, 0.486] | 0.457 | 0.091 | 0.565 | 0.104 [0.019, 0.186] | above the AR trend |
| Dream-7B + Gemma-2-9B | AR+diffusion | 0.421 [0.336, 0.502] | 0.457 | 0.177 | 0.501 | 0.058 [-0.018, 0.136] | inconclusive |
| Dream-7B + Qwen2.5-7B | AR+diffusion | 0.445 [0.364, 0.522] | 0.484 | 0.184 | 0.418 | 0.055 [-0.017, 0.130] | inconclusive |

Within error type *arithmetic_slip* (58 wrong answers, descriptive):

| AR+diffusion pair | κ | κ minus trend [95% CI] | Status |
|---|---|---|---|
| Gemma-2-9B + LLaDA-8B | -0.048 | -0.108 [-0.224, 0.017] | inconclusive |
| LLaDA-8B + Llama-3.1-8B | 0.249 | 0.144 [-0.181, 0.598] | inconclusive |
| LLaDA-8B + Qwen2.5-7B | -0.055 | -0.084 [-0.250, 0.067] | inconclusive |
| Dream-7B + Qwen2.5-7B | 0.298 | 0.256 [-0.204, 0.666] | inconclusive |
| Dream-7B + Gemma-2-9B | -0.083 | -0.155 [-0.258, -0.021] | below the AR trend |
| Dream-7B + Llama-3.1-8B | 0.343 | 0.227 [-0.179, 0.625] | inconclusive |
| Dream-7B + Mistral-7B | 0.158 | 0.015 [-0.224, 0.364] | inconclusive |
| LLaDA-8B + Mistral-7B | 0.651 | 0.520 [-0.051, 0.850] | inconclusive |

Within error type *formatting*: 5 wrong answers, too few to report a trend.

Within error type *misread_problem* (70 wrong answers, descriptive):

| AR+diffusion pair | κ | κ minus trend [95% CI] | Status |
|---|---|---|---|
| Gemma-2-9B + LLaDA-8B | 0.317 | -0.057 [-0.255, 0.160] | inconclusive |
| LLaDA-8B + Llama-3.1-8B | 0.369 | -0.024 [-0.234, 0.218] | inconclusive |
| LLaDA-8B + Qwen2.5-7B | 0.530 | 0.169 [-0.109, 0.453] | inconclusive |
| Dream-7B + Qwen2.5-7B | 0.381 | 0.015 [-0.233, 0.267] | inconclusive |
| Dream-7B + Gemma-2-9B | 0.430 | 0.051 [-0.185, 0.324] | inconclusive |
| Dream-7B + Llama-3.1-8B | 0.094 | -0.304 [-0.449, -0.115] | below the AR trend |
| Dream-7B + Mistral-7B | 0.189 | -0.220 [-0.348, -0.072] | below the AR trend |
| LLaDA-8B + Mistral-7B | 0.367 | -0.037 [-0.245, 0.176] | inconclusive |

Within error type *wrong_method* (67 wrong answers, descriptive):

| AR+diffusion pair | κ | κ minus trend [95% CI] | Status |
|---|---|---|---|
| Gemma-2-9B + LLaDA-8B | 0.450 | -0.048 [-0.255, 0.155] | inconclusive |
| LLaDA-8B + Llama-3.1-8B | 0.420 | -0.007 [-0.240, 0.236] | inconclusive |
| LLaDA-8B + Qwen2.5-7B | 0.377 | -0.169 [-0.386, 0.042] | inconclusive |
| Dream-7B + Qwen2.5-7B | 0.574 | 0.047 [-0.188, 0.275] | inconclusive |
| Dream-7B + Gemma-2-9B | 0.382 | -0.098 [-0.325, 0.122] | inconclusive |
| Dream-7B + Llama-3.1-8B | 0.291 | -0.118 [-0.375, 0.141] | inconclusive |
| Dream-7B + Mistral-7B | 0.426 | 0.060 [-0.213, 0.318] | inconclusive |
| LLaDA-8B + Mistral-7B | 0.318 | -0.067 [-0.302, 0.178] | inconclusive |

## E1 (code, descriptive): Qwen2.5-7B with each partner

| Pair | FAR | FAR minus Qwen+Dream | κ | κ minus Qwen+Dream |
|---|---|---|---|---|
| Gemma-2-9B + Qwen2.5-7B | 0.675 [0.587, 0.758] | 0.075 [0.017, 0.134] | 0.547 [0.360, 0.706] | 0.019 [-0.148, 0.174] |
| Llama-3.1-8B + Qwen2.5-7B | 0.633 [0.541, 0.721] | 0.033 [-0.041, 0.105] | 0.379 [0.191, 0.551] | -0.149 [-0.320, 0.023] |
| LLaDA-8B + Qwen2.5-7B | 0.692 [0.602, 0.776] | 0.092 [0.034, 0.151] | 0.049 [-0.075, 0.193] | -0.479 [-0.676, -0.268] |
| Dream-7B + Qwen2.5-7B | 0.600 [0.500, 0.695] | 0.000 [0.000, 0.000] | 0.528 [0.369, 0.675] | 0.000 [0.000, 0.000] |
| Mistral-7B + Qwen2.5-7B | 0.725 [0.642, 0.803] | 0.125 [0.066, 0.186] | 0.072 [-0.032, 0.197] | -0.456 [-0.642, -0.261] |

## E1 (math, descriptive): Qwen2.5-7B with each partner

| Pair | FAR | FAR minus Qwen+Dream | κ | κ minus Qwen+Dream |
|---|---|---|---|---|
| Gemma-2-9B + Qwen2.5-7B | 0.649 [0.611, 0.685] | -0.014 [-0.039, 0.010] | 0.497 [0.421, 0.568] | 0.053 [-0.037, 0.141] |
| Llama-3.1-8B + Qwen2.5-7B | 0.660 [0.623, 0.696] | -0.003 [-0.028, 0.022] | 0.368 [0.284, 0.447] | -0.077 [-0.170, 0.017] |
| LLaDA-8B + Qwen2.5-7B | 0.670 [0.634, 0.705] | 0.007 [-0.019, 0.033] | 0.338 [0.254, 0.419] | -0.107 [-0.194, -0.019] |
| Dream-7B + Qwen2.5-7B | 0.663 [0.626, 0.699] | 0.000 [0.000, 0.000] | 0.445 [0.364, 0.522] | 0.000 [0.000, 0.000] |
| Mistral-7B + Qwen2.5-7B | 0.690 [0.653, 0.725] | 0.027 [0.006, 0.049] | 0.305 [0.217, 0.388] | -0.140 [-0.229, -0.052] |

## Breakdowns (descriptive; small groups marked *)

| Domain | Group | Wrong | FAR AR+diffusion | FAR AR+AR | ΔFAR points [95% CI] | κ AR+diffusion / AR+AR |
|---|---|---|---|---|---|---|
| code | all: all | 120 | 0.692 | 0.675 | +1.7 [-3.4, 6.9] | 0.049 / 0.547 |
| code | source: evalplus/humaneval+ | 120 | 0.692 | 0.675 | +1.7 [-3.4, 6.9] | 0.049 / 0.547 |
| code | generator: allenai/OLMo-2-1124-7B-Instruct | 30 | 0.433 | 0.467 | -3.3 [-11.1, 0.0] | 0.051 / 0.609 |
| code | generator: ibm-granite/granite-3.3-8b-instruct | 39 | 0.821 | 0.769 | +5.1 [-4.9, 15.8] | -0.046 / 0.260 |
| code | generator: microsoft/Phi-4-mini-instruct | 43 | 0.721 | 0.698 | +2.3 [-7.5, 12.8] | 0.029 / 0.527 |
| code | generator: openai/gpt-4.1-mini* | 8 | 0.875 | 0.875 | +0.0 [0.0, 0.0] | 0.000 / 1.000 |
| code | error_type: wrong_output | 120 | 0.692 | 0.675 | +1.7 [-3.4, 6.9] | 0.049 / 0.547 |
| math | all: all | 700 | 0.670 | 0.649 | +2.1 [-0.3, 4.6] | 0.338 / 0.497 |
| math | source: gsm8k-platinum/test | 344 | 0.695 | 0.657 | +3.8 [0.3, 7.3] | 0.424 / 0.542 |
| math | source: math-500/test | 138 | 0.609 | 0.580 | +2.9 [-2.8, 8.6] | 0.217 / 0.533 |
| math | source: math-rest/test | 218 | 0.670 | 0.679 | -0.9 [-5.0, 3.2] | 0.293 / 0.382 |
| math | generator: allenai/OLMo-2-1124-7B-Instruct | 267 | 0.715 | 0.719 | -0.4 [-4.5, 3.8] | 0.273 / 0.451 |
| math | generator: ibm-granite/granite-3.3-8b-instruct | 212 | 0.627 | 0.613 | +1.4 [-2.1, 5.1] | 0.380 / 0.612 |
| math | generator: microsoft/Phi-4-mini-instruct | 201 | 0.637 | 0.577 | +6.0 [0.5, 11.4] | 0.358 / 0.421 |
| math | generator: openai/gpt-4.1-mini* | 20 | 0.850 | 0.800 | +5.0 [0.0, 16.7] | 0.000 / -0.081 |
| math | error_type: arithmetic_slip | 58 | 0.862 | 0.845 | +1.7 [-5.9, 9.4] | -0.055 / 0.128 |
| math | error_type: formatting* | 5 | 0.400 | 0.400 | +0.0 [0.0, 0.0] | 1.000 / 0.615 |
| math | error_type: misread_problem | 70 | 0.743 | 0.686 | +5.7 [-1.6, 14.1] | 0.530 / 0.462 |
| math | error_type: wrong_method | 67 | 0.522 | 0.537 | -1.5 [-8.2, 5.2] | 0.377 / 0.653 |
