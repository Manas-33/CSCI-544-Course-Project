# Step 0 results (November 22, 2026)

The answer-swap test (PLAN.md §4 test 4) and the AR repeat runs (test 3) on the pilot, from notebook 09. `step0.md` is
the summary. All scores: T1, batch size 1, BF16; LLaDA and Dream with 1 answer slot.

| Check | Outcome |
|---|---|
| Swap test, math (go/no-go) | All eight verifiers pass, 98–100 of 100; the four H1 verifiers 99 of 100 each |
| Swap test, code (reported only) | All fail, 50–71 of 100, as PLAN.md §4 expected |
| Repeat runs | The six AR models gave identical scores in both runs (500 of 500) |

- **Run:** NVIDIA L4, commit `8ff1bac`. The notebook at that commit accepted only an A100; that one line was removed by
  hand before the run (main now accepts both).
- **Compared with October 27 (A100):** single scores move (by up to 1.5 for Qwen2.5-7B), AUROC and FAR at 90% TAR of
  the three H1 AR verifiers by at most 0.01. Information only, but a reason to keep all full-bank scores on one GPU type.
- **What it shows:** the swap changes only the final answer, so the test shows that a score reacts to the answer. Gate A
  is the competence bar.
- **Still open for the go/no-go:** the label audit (PLAN.md §7).

`analysis.step0` on these files gives `step0.md` and `step0.json` again.
