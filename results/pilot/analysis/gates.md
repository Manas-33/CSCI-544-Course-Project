# Pilot results

Bank: `bank/pilot/pilot_bank.jsonl` (sha256 aa4d8cc3ee3c…), math: 125 correct / 125 wrong, code: 125 correct / 125 wrong

All numbers are pilot (exploratory) numbers. Pair thresholds are set on the pilot itself, so pair FARs are in-sample and are not the H1 test.

## Verifiers (T1)

| Verifier | Class | Domain | AUROC [95% CI] | ECE | Yes+No mass | Pair rank acc. | s / candidate |
|---|---|---|---|---|---|---|---|
| Qwen2.5-7B | AR | code | 0.704 [0.647, 0.756] | 0.348 | 1.000 | 0.720 | 0.037 |
| Gemma-2-9B | AR | code | 0.655 [0.598, 0.711] | 0.390 | 0.999 | 0.680 | 0.077 |
| Dream-7B | diffusion | code | 0.617 [0.552, 0.672] | 0.057 | 0.841 | 0.560 | 0.113 |
| Llama-3.1-8B | AR | code | 0.612 [0.559, 0.662] | 0.348 | 1.000 | 0.688 | 0.040 |
| LLaDA-8B | diffusion | code | 0.594 [0.545, 0.642] | 0.278 | 0.985 | 0.672 | 0.045 |
| Mistral-7B | AR | code | 0.577 [0.525, 0.625] | 0.501 | 1.000 | 0.568 | 0.041 |
| Llama-3.2-3B | AR | code | 0.554 [0.500, 0.609] | 0.324 | 1.000 | 0.576 | 0.033 |
| Qwen2.5-1.5B | AR | code | 0.503 [0.460, 0.541] | 0.366 | 0.997 | 0.496 | 0.034 |
| Gemma-2-9B | AR | math | 0.702 [0.650, 0.754] | 0.423 | 0.997 | 0.736 | 0.095 |
| Qwen2.5-7B | AR | math | 0.695 [0.645, 0.745] | 0.393 | 1.000 | 0.728 | 0.053 |
| Llama-3.1-8B | AR | math | 0.659 [0.604, 0.716] | 0.244 | 1.000 | 0.672 | 0.057 |
| Qwen2.5-1.5B | AR | math | 0.652 [0.594, 0.710] | 0.449 | 0.985 | 0.664 | 0.034 |
| LLaDA-8B | diffusion | math | 0.607 [0.545, 0.668] | 0.370 | 0.891 | 0.592 | 0.063 |
| Dream-7B | diffusion | math | 0.592 [0.543, 0.640] | 0.294 | 0.890 | 0.664 | 0.185 |
| Mistral-7B | AR | math | 0.574 [0.523, 0.626] | 0.481 | 0.999 | 0.592 | 0.059 |
| Llama-3.2-3B | AR | math | 0.487 [0.424, 0.552] | 0.317 | 1.000 | 0.480 | 0.035 |
| Qwen2.5-7B | AR | pooled | 0.699 [0.660, 0.736] | 0.363 | 1.000 | 0.724 | 0.045 |
| Gemma-2-9B | AR | pooled | 0.662 [0.626, 0.699] | 0.402 | 0.998 | 0.708 | 0.086 |
| LLaDA-8B | diffusion | pooled | 0.595 [0.558, 0.628] | 0.321 | 0.941 | 0.632 | 0.054 |
| Llama-3.1-8B | AR | pooled | 0.590 [0.563, 0.619] | 0.284 | 1.000 | 0.680 | 0.048 |
| Mistral-7B | AR | pooled | 0.572 [0.538, 0.607] | 0.489 | 1.000 | 0.580 | 0.050 |
| Dream-7B | diffusion | pooled | 0.572 [0.546, 0.598] | 0.156 | 0.871 | 0.612 | 0.149 |
| Qwen2.5-1.5B | AR | pooled | 0.561 [0.530, 0.594] | 0.401 | 0.995 | 0.580 | 0.034 |
| Llama-3.2-3B | AR | pooled | 0.523 [0.492, 0.555] | 0.319 | 1.000 | 0.528 | 0.034 |

## Gates (PLAN.md §7)

| Verifier | Usable scores | Gate A math | Gate A code | Gate B math (T1–T3 / T1–T2) | Gate B code (T1–T3 / T1–T2) | Eligible (T1–T3 / T1–T2) |
|---|---|---|---|---|---|---|
| Dream-7B | pass | **fail** | pass | pass / pass (0.039 / 0.032) | **fail** / pass (0.140 / 0.038) | **fail** / **fail** |
| Gemma-2-9B | pass | pass | pass | pass / pass (0.030 / 0.014) | pass / pass (0.027 / 0.003) | pass / pass |
| LLaDA-8B | pass | pass | **fail** | **fail** / pass (0.098 / 0.009) | **fail** / pass (0.067 / 0.006) | **fail** / pass |
| Llama-3.1-8B | pass | pass | pass | **fail** / pass (0.103 / 0.016) | **fail** / pass (0.058 / 0.018) | **fail** / pass |
| Llama-3.2-3B | pass | **fail** | **fail** | **fail** / pass (0.092 / 0.034) | pass / pass (0.029 / 0.010) | reference only |
| Mistral-7B | pass | **fail** | **fail** | **fail** / pass (0.063 / 0.020) | pass / pass (0.047 / 0.020) | **fail** / **fail** |
| Qwen2.5-1.5B | pass | pass | **fail** | **fail** / pass (0.112 / 0.006) | pass / pass (0.045 / 0.033) | reference only |
| Qwen2.5-7B | pass | pass | pass | **fail** / pass (0.170 / 0.046) | **fail** / pass (0.146 / 0.004) | **fail** / pass |

- **Gate A small-AR reference (Llama-3.2-3B):** math 0.487, code 0.554
- **Eligibility, gate B over T1–T3 (as written):** repair: no diffusion verifier passes in any domain; try the reconstruction score (PLAN.md §7)
- **Eligibility, gate B over T1–T2:** continue on math (narrowed: no diffusion model is eligible in every domain); diffusion: LLaDA-8B; AR: Gemma-2-9B, Llama-3.1-8B, Qwen2.5-7B
- **Code rule:** {'ar_passing_code': ['Gemma-2-9B', 'Llama-3.1-8B', 'Qwen2.5-7B'], 'diffusion_passing_code': ['Dream-7B'], 'code_in_primary': True}
- **Dynamic range:** math: Gemma-2-9B FAR 0.744 at 90% TAR (fail); code: Qwen2.5-7B FAR 0.616 at 90% TAR (pass)
- **No shortcuts:** math: length AUROC 0.580, surface AUROC 0.548 (pass); code: length AUROC 0.554, surface AUROC – (pass)

## Pairs at 90% pair TAR (AND rule, T1)

| Pair | Kind | Domain | FAR [95% CI] | Pair TAR | κ false accepts | Spearman (wrong) |
|---|---|---|---|---|---|---|
| Llama-3.1-8B + Qwen2.5-7B | AR+AR | code | 0.632 [0.543, 0.719] | 0.912 | 0.129 | 0.511 |
| LLaDA-8B + Qwen2.5-7B | AR+diffusion | code | 0.656 [0.571, 0.738] | 0.912 | 0.075 | 0.243 |
| Gemma-2-9B + Qwen2.5-7B | AR+AR | code | 0.664 [0.585, 0.748] | 0.904 | 0.399 | 0.599 |
| Dream-7B + Qwen2.5-7B (E1) | AR+diffusion | code | 0.672 [0.593, 0.752] | 0.912 | 0.446 | 0.629 |
| Mistral-7B + Qwen2.5-7B | AR+AR | code | 0.688 [0.608, 0.770] | 0.904 | 0.137 | 0.339 |
| LLaDA-8B + Llama-3.1-8B | AR+diffusion | code | 0.744 [0.659, 0.820] | 0.904 | 0.038 | 0.317 |
| Dream-7B + Llama-3.1-8B | AR+diffusion | code | 0.760 [0.686, 0.832] | 0.904 | 0.221 | 0.440 |
| Gemma-2-9B + Llama-3.1-8B | AR+AR | code | 0.760 [0.681, 0.835] | 0.904 | 0.405 | 0.504 |
| Dream-7B + Gemma-2-9B | AR+diffusion | code | 0.768 [0.703, 0.835] | 0.904 | 0.465 | 0.571 |
| Gemma-2-9B + LLaDA-8B | AR+diffusion | code | 0.768 [0.688, 0.841] | 0.904 | 0.137 | 0.254 |
| Llama-3.1-8B + Mistral-7B | AR+AR | code | 0.808 [0.730, 0.878] | 0.904 | 0.145 | 0.522 |
| Gemma-2-9B + Mistral-7B | AR+AR | code | 0.824 [0.756, 0.887] | 0.912 | 0.430 | 0.454 |
| LLaDA-8B + Mistral-7B | AR+diffusion | code | 0.832 [0.762, 0.895] | 0.904 | 0.180 | 0.339 |
| Dream-7B + Mistral-7B | AR+diffusion | code | 0.840 [0.782, 0.900] | 0.912 | 0.409 | 0.303 |
| Gemma-2-9B + Qwen2.5-7B | AR+AR | math | 0.696 [0.611, 0.779] | 0.904 | 0.387 | 0.651 |
| Dream-7B + Qwen2.5-7B (E1) | AR+diffusion | math | 0.712 [0.628, 0.795] | 0.912 | 0.546 | 0.494 |
| Dream-7B + Gemma-2-9B | AR+diffusion | math | 0.720 [0.638, 0.798] | 0.904 | 0.361 | 0.412 |
| Llama-3.1-8B + Qwen2.5-7B | AR+AR | math | 0.720 [0.634, 0.808] | 0.912 | 0.471 | 0.479 |
| Mistral-7B + Qwen2.5-7B | AR+AR | math | 0.728 [0.648, 0.807] | 0.904 | 0.388 | 0.322 |
| Dream-7B + Llama-3.1-8B | AR+diffusion | math | 0.736 [0.650, 0.819] | 0.904 | 0.394 | 0.414 |
| Gemma-2-9B + Llama-3.1-8B | AR+AR | math | 0.736 [0.659, 0.817] | 0.904 | 0.393 | 0.464 |
| LLaDA-8B + Qwen2.5-7B | AR+diffusion | math | 0.736 [0.656, 0.813] | 0.912 | 0.406 | 0.328 |
| Gemma-2-9B + LLaDA-8B | AR+diffusion | math | 0.776 [0.697, 0.851] | 0.912 | 0.354 | 0.332 |
| LLaDA-8B + Llama-3.1-8B | AR+diffusion | math | 0.776 [0.700, 0.852] | 0.904 | 0.305 | 0.302 |
| Dream-7B + Mistral-7B | AR+diffusion | math | 0.792 [0.715, 0.864] | 0.904 | 0.448 | 0.339 |
| Llama-3.1-8B + Mistral-7B | AR+AR | math | 0.800 [0.731, 0.868] | 0.904 | 0.309 | 0.330 |
| Gemma-2-9B + Mistral-7B | AR+AR | math | 0.808 [0.734, 0.876] | 0.912 | 0.333 | 0.403 |
| LLaDA-8B + Mistral-7B | AR+diffusion | math | 0.848 [0.782, 0.908] | 0.904 | 0.486 | 0.222 |
| Llama-3.1-8B + Qwen2.5-7B | AR+AR | pooled | 0.676 [0.614, 0.736] | 0.912 | – | – |
| Gemma-2-9B + Qwen2.5-7B | AR+AR | pooled | 0.680 [0.622, 0.736] | 0.904 | – | – |
| Dream-7B + Qwen2.5-7B (E1) | AR+diffusion | pooled | 0.692 [0.635, 0.746] | 0.912 | – | – |
| LLaDA-8B + Qwen2.5-7B | AR+diffusion | pooled | 0.696 [0.639, 0.755] | 0.912 | – | – |
| Mistral-7B + Qwen2.5-7B | AR+AR | pooled | 0.708 [0.650, 0.762] | 0.904 | – | – |
| Dream-7B + Gemma-2-9B | AR+diffusion | pooled | 0.744 [0.690, 0.795] | 0.904 | – | – |
| Dream-7B + Llama-3.1-8B | AR+diffusion | pooled | 0.748 [0.690, 0.803] | 0.904 | – | – |
| Gemma-2-9B + Llama-3.1-8B | AR+AR | pooled | 0.748 [0.694, 0.800] | 0.904 | – | – |
| LLaDA-8B + Llama-3.1-8B | AR+diffusion | pooled | 0.760 [0.705, 0.815] | 0.904 | – | – |
| Gemma-2-9B + LLaDA-8B | AR+diffusion | pooled | 0.772 [0.717, 0.827] | 0.908 | – | – |
| Llama-3.1-8B + Mistral-7B | AR+AR | pooled | 0.804 [0.753, 0.853] | 0.904 | – | – |
| Dream-7B + Mistral-7B | AR+diffusion | pooled | 0.816 [0.766, 0.861] | 0.908 | – | – |
| Gemma-2-9B + Mistral-7B | AR+AR | pooled | 0.816 [0.769, 0.862] | 0.912 | – | – |
| LLaDA-8B + Mistral-7B | AR+diffusion | pooled | 0.840 [0.795, 0.886] | 0.904 | – | – |

**math:** best AR+AR Gemma-2-9B + Qwen2.5-7B FAR 0.696; best AR+diffusion Dream-7B + Gemma-2-9B FAR 0.720; ΔFAR +0.024 (in-sample, exploratory).

**code:** best AR+AR Llama-3.1-8B + Qwen2.5-7B FAR 0.632; best AR+diffusion LLaDA-8B + Qwen2.5-7B FAR 0.656; ΔFAR +0.024 (in-sample, exploratory).

**pooled:** best AR+AR Llama-3.1-8B + Qwen2.5-7B FAR 0.676; best AR+diffusion LLaDA-8B + Qwen2.5-7B FAR 0.696; ΔFAR +0.020 (in-sample, exploratory).
