# Step 0: answer swap and repeat runs

Pilot bank, T1, batch size 1. Math decides for the go/no-go (PLAN.md §7); the rest is reported.

## Answer swap, math (test 4, used in the go/no-go)

Pass: the swap scores strictly lower than the original on at least 80% of items. Tied and unscored items count as not lowered.

| Verifier | In H1 | Lowered / items | Share | 95% interval | Tied | Raised | Unscored | Result |
|---|---|---|---|---|---|---|---|---|
| Qwen2.5-7B | yes | 99 / 100 | 99.0% | 94.6%–99.8% | 0 | 1 | 0 | PASS |
| Llama-3.1-8B | yes | 99 / 100 | 99.0% | 94.6%–99.8% | 0 | 1 | 0 | PASS |
| Gemma-2-9B | yes | 99 / 100 | 99.0% | 94.6%–99.8% | 0 | 1 | 0 | PASS |
| LLaDA-8B | yes | 99 / 100 | 99.0% | 94.6%–99.8% | 0 | 1 | 0 | PASS |
| Mistral-7B | no | 98 / 100 | 98.0% | 93.0%–99.4% | 0 | 2 | 0 | PASS |
| Llama-3.2-3B | no | 99 / 100 | 99.0% | 94.6%–99.8% | 0 | 1 | 0 | PASS |
| Qwen2.5-1.5B | no | 100 / 100 | 100.0% | 96.3%–100.0% | 0 | 0 | 0 | PASS |
| Dream-7B | no | 98 / 100 | 98.0% | 93.0%–99.4% | 0 | 2 | 0 | PASS |

## Answer swap, code (reported only; expected to fail, PLAN.md §4)

Pass: the swap scores strictly lower than the original on at least 80% of items. Tied and unscored items count as not lowered.

| Verifier | In H1 | Lowered / items | Share | 95% interval | Tied | Raised | Unscored | Result |
|---|---|---|---|---|---|---|---|---|
| Qwen2.5-7B | yes | 71 / 100 | 71.0% | 61.5%–79.0% | 0 | 29 | 0 | FAIL |
| Llama-3.1-8B | yes | 68 / 100 | 68.0% | 58.3%–76.3% | 0 | 32 | 0 | FAIL |
| Gemma-2-9B | yes | 69 / 100 | 69.0% | 59.4%–77.2% | 0 | 31 | 0 | FAIL |
| LLaDA-8B | yes | 66 / 100 | 66.0% | 56.3%–74.5% | 0 | 34 | 0 | FAIL |
| Mistral-7B | no | 57 / 100 | 57.0% | 47.2%–66.3% | 0 | 43 | 0 | FAIL |
| Llama-3.2-3B | no | 58 / 100 | 58.0% | 48.2%–67.2% | 0 | 42 | 0 | FAIL |
| Qwen2.5-1.5B | no | 50 / 100 | 50.0% | 40.4%–59.6% | 0 | 50 | 0 | FAIL |
| Dream-7B | no | 54 / 100 | 54.0% | 44.3%–63.4% | 0 | 46 | 0 | FAIL |

## Repeat runs, AR models (test 3)

Pass: Spearman above 0.999 between two runs in one session, over the 500 pilot answers. Math, code and October 27 are information only. A failure is a scorer bug to fix, not a verdict on the model.

| Verifier | All | Math | Code | Largest difference | Identical | Result | October 27: all | Math | Code | Largest difference |
|---|---|---|---|---|---|---|---|---|---|---|
| Qwen2.5-7B | 1.00000 | 1.00000 | 1.00000 | 0.000 | 500 / 500 | PASS | 0.99834 | 0.99808 | 0.99823 | 1.500 |
| Llama-3.1-8B | 1.00000 | 1.00000 | 1.00000 | 0.000 | 500 / 500 | PASS | 0.99831 | 0.99375 | 0.99766 | 0.375 |
| Gemma-2-9B | 1.00000 | 1.00000 | 1.00000 | 0.000 | 500 / 500 | PASS | 0.99941 | 0.99914 | 0.99901 | 0.500 |
| Mistral-7B | 1.00000 | 1.00000 | 1.00000 | 0.000 | 500 / 500 | PASS | 0.99827 | 0.99791 | 0.99742 | 0.750 |
| Llama-3.2-3B | 1.00000 | 1.00000 | 1.00000 | 0.000 | 500 / 500 | PASS | 0.99668 | 0.99090 | 0.99631 | 1.125 |
| Qwen2.5-1.5B | 1.00000 | 1.00000 | 1.00000 | 0.000 | 500 / 500 | PASS | 0.99193 | 0.98144 | 0.99314 | 0.250 |

## For the final model record

| Verifier | Answer swap, math | Repeat runs |
|---|---|---|
| Qwen2.5-7B | pass | pass |
| Llama-3.1-8B | pass | pass |
| Gemma-2-9B | pass | pass |
| LLaDA-8B | pass | not needed (passed in stage 3) |

The label audit is the third pilot check and is not part of this file.
