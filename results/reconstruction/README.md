# Backup score on the pilot (November 22, 2026)

LLaDA's reconstruction score (PLAN.md §4, §7) on the pilot's 250 math answers, from notebook 12: 128 samples, seeds 544
and 545, unbatched, NVIDIA L4, commit `489426a`. It matches the official code (6 of 6), passes gate A (AUROC 0.652) and
fails the repeat check (0.949, needs 0.95), so it fails as the repair. The repair is not needed: LLaDA's verdict score
passed gate A and the swap test in math. The development-bank run for the sample count has not been done.

`analysis.reconstruction pilot` on these files gives the same report.

# Side experiment on the math bank (November 23, 2026), exploratory

From notebook 12, on the side-experiment draw in `side/subset.json`: 500 validation and 500 test answers of the math
bank, as whole problems, seed 544. Two kinds of score were run on it:

- LLaDA's reconstruction score: 128 samples, seed 544, unbatched.
- The exact log-probability of the four AR models, with the verdict runs' attention, BF16 and batch size 1.

The run used an NVIDIA A100-SXM4-40GB, the A100 type of notebook 13's scores, at commit `b206c92`. All 1,000 answers
are scored by every model. The AR commands were given `--device cuda:0` by hand: the old default, a bare `cuda`,
stopped before any model loaded. `cuda:0` is now the default.

It is reported on the 500 validation answers only (250 correct, 250 wrong), as an exploratory result
([record](../../decisions/2026-11-22-side-experiment-after-test.md)). The test half is not evaluated. The thresholds
below are set and measured on the same answers, so the FARs are in-sample.

| Score (higher = more plausible) | AUROC [95% CI] | FAR at 90% TAR |
|---|---|---|
| LLaDA-8B reconstruction | 0.654 [0.614, 0.695] | 0.780 |
| Qwen2.5-7B exact log-probability | 0.651 [0.616, 0.689] | 0.788 |
| Llama-3.1-8B exact log-probability | 0.625 [0.592, 0.661] | 0.788 |
| Mistral-7B exact log-probability | 0.602 [0.566, 0.640] | 0.812 |
| Gemma-2-9B exact log-probability | 0.587 [0.556, 0.619] | 0.828 |

The next table compares each likelihood score with the same model's T1 verdict score (notebook 13) on the same 500
answers. Intervals come from 10,000 resamples of whole problems.

| Model | Verdict AUROC | Likelihood AUROC | Likelihood − verdict [95% CI] | Rank correlation |
|---|---|---|---|---|
| LLaDA-8B | 0.626 | 0.654 | +0.028 [−0.023, +0.079] | 0.16 |
| Qwen2.5-7B | 0.699 | 0.651 | −0.047 [−0.094, +0.001] | 0.29 |
| Llama-3.1-8B | 0.627 | 0.625 | −0.002 [−0.052, +0.047] | 0.14 |
| Gemma-2-9B | 0.677 | 0.587 | −0.090 [−0.137, −0.044] | 0.03 |
| Mistral-7B | 0.563 | 0.602 | +0.039 [−0.010, +0.087] | 0.03 |

What it shows, on these 500 answers:

- LLaDA's reconstruction score is not clearly better than its verdict score: +0.028, with an interval from −0.023 to
  +0.079. There is no clear evidence that Yes/No prompting wastes LLaDA's diffusion objective, but a gain of up to
  about 0.08 AUROC is not ruled out.
- A likelihood score above the verdict is not special to diffusion here. Mistral's exact log-probability is ahead of
  its verdict by about as much (+0.039), and Qwen's reaches about the same AUROC as LLaDA's reconstruction score
  (0.651 against 0.654; this difference was not tested).
- Only for Gemma is the verdict score clearly better (by 0.090). Qwen's verdict is ahead by 0.047, but its interval
  just reaches zero.
- The likelihood and verdict scores rank answers very differently (rank correlation 0.03 to 0.29).

To reproduce:

```bash
python -m analysis.reconstruction validation --bank bank/full/math/bank.jsonl --subset results/reconstruction/side/subset.json \
    --labels bank/full/math/labels_validation.jsonl --scores results/reconstruction/side --out results/reconstruction/side/validation.json
python -m analysis.side_compare --out results/reconstruction/side/verdict_compare.json
```
