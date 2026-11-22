# Precision and matching check from the pilot

Pilot numbers: exploratory planning estimates, made by `analysis/precision.py` (PLAN.md §6). Not H1 results.

## Math: 700 wrong test answers planned (2100 candidates)

The two pairs disagree on 8%–23% of the pilot's 125 wrong answers; design effect 0.93–1.19; detectable ΔFAR **3.0–5.6 points** (80% power, 5% two-sided).

| AR+diffusion pair | AR+AR pair | Disagree | Design effect | Detectable ΔFAR (points) |
|---|---|---|---|---|
| LLaDA-8B + Qwen2.5-7B | Llama-3.1-8B + Qwen2.5-7B | 8.0% | 0.99 | 3.0 |
| LLaDA-8B + Qwen2.5-7B | Gemma-2-9B + Qwen2.5-7B | 10.4% | 1.00 | 3.4 |
| Gemma-2-9B + LLaDA-8B | Gemma-2-9B + Llama-3.1-8B | 12.0% | 0.98 | 3.7 |
| LLaDA-8B + Llama-3.1-8B | Gemma-2-9B + Llama-3.1-8B | 12.0% | 1.00 | 3.7 |
| LLaDA-8B + Llama-3.1-8B | Llama-3.1-8B + Qwen2.5-7B | 12.0% | 1.12 | 3.9 |
| Gemma-2-9B + LLaDA-8B | Gemma-2-9B + Qwen2.5-7B | 14.4% | 1.09 | 4.2 |
| LLaDA-8B + Llama-3.1-8B | Gemma-2-9B + Qwen2.5-7B | 22.4% | 0.93 | 5.0 |
| LLaDA-8B + Qwen2.5-7B | Gemma-2-9B + Llama-3.1-8B | 22.4% | 0.93 | 5.0 |
| Gemma-2-9B + LLaDA-8B | Llama-3.1-8B + Qwen2.5-7B | 23.2% | 1.19 | 5.6 |

## Code: 283 wrong test answers planned (850 candidates)

The two pairs disagree on 7%–30% of the pilot's 125 wrong answers; design effect 0.81–1.01; detectable ΔFAR **4.5–9.2 points** (80% power, 5% two-sided).

| AR+diffusion pair | AR+AR pair | Disagree | Design effect | Detectable ΔFAR (points) |
|---|---|---|---|---|
| LLaDA-8B + Qwen2.5-7B | Gemma-2-9B + Qwen2.5-7B | 7.2% | 1.00 | 4.5 |
| Gemma-2-9B + LLaDA-8B | Gemma-2-9B + Llama-3.1-8B | 12.0% | 1.00 | 5.8 |
| LLaDA-8B + Qwen2.5-7B | Llama-3.1-8B + Qwen2.5-7B | 12.0% | 0.99 | 5.8 |
| LLaDA-8B + Llama-3.1-8B | Gemma-2-9B + Llama-3.1-8B | 12.8% | 1.01 | 6.0 |
| Gemma-2-9B + LLaDA-8B | Gemma-2-9B + Qwen2.5-7B | 18.4% | 0.86 | 7.1 |
| LLaDA-8B + Llama-3.1-8B | Llama-3.1-8B + Qwen2.5-7B | 25.6% | 0.81 | 8.4 |
| Gemma-2-9B + LLaDA-8B | Llama-3.1-8B + Qwen2.5-7B | 26.4% | 0.88 | 8.6 |
| LLaDA-8B + Qwen2.5-7B | Gemma-2-9B + Llama-3.1-8B | 28.0% | 0.99 | 8.8 |
| LLaDA-8B + Llama-3.1-8B | Gemma-2-9B + Qwen2.5-7B | 30.4% | 0.87 | 9.2 |

## Matching check: 3,000 runs, 350 validation and 700 test pairs, whole problems, seed 544

| Limit on TAR from 90% | Limit on the gap | Passes | Passes with score noise |
|---|---|---|---|
| 3 | 3 | 70.4% | 70.9% |
| 4 | 3 | 81.3% | 81.6% |
| 5 | 3 (adopted) | 84.8% | 85.9% |
| 6 | 3 | 86.3% | 87.9% |
| none | 3 | 86.9% | 88.9% |
| 4 | 4 | 87.5% | 87.0% |
| 5 | 5 | 95.1% | 94.4% |

Noise 0 SD: test TAR SD 2.1 (AR+AR) and 2.1 (AR+diffusion) points, gap SD 2.0; of the 3/3 failures, 44% break the gap limit. Monte Carlo error about 0.7 points.

Noise 0.15 SD: test TAR SD 2.0 (AR+AR) and 2.1 (AR+diffusion) points, gap SD 1.9; of the 3/3 failures, 38% break the gap limit. Monte Carlo error about 0.6 points.

## FAR by pair TAR on the pilot (in-sample)

| Pair | 85% | 87% | 90% | 93% | 95% |
|---|---|---|---|---|---|
| Gemma-2-9B + Llama-3.1-8B | 0.664 | 0.696 | 0.736 | 0.816 | 0.864 |
| Gemma-2-9B + Qwen2.5-7B | 0.608 | 0.648 | 0.696 | 0.736 | 0.792 |
| Llama-3.1-8B + Qwen2.5-7B | 0.664 | 0.696 | 0.720 | 0.752 | 0.776 |
| Gemma-2-9B + LLaDA-8B | 0.680 | 0.760 | 0.776 | 0.864 | 0.904 |
| LLaDA-8B + Llama-3.1-8B | 0.720 | 0.744 | 0.776 | 0.832 | 0.832 |
| LLaDA-8B + Qwen2.5-7B | 0.656 | 0.696 | 0.736 | 0.760 | 0.776 |
