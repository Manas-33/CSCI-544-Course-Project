# The sanity set is a smoke test, not a gate

**Date:** 2026-11-21 · **Decided by:** the team, by merging this record · **Seen when decided:** the full pilot results (October 27) and the midterm feedback · **Post hoc** · **Supersedes:** the decision proposed in [LLaDA fails the sanity bar](2026-10-26-llada-fails-sanity.md), and its statement that the 0.90 bar is not changed

## What happened

PLAN.md §4 (before version 1.3) required AUROC ≥ 0.90 on 50 hand-written items, and counted a failure after the code was verified as a gate A failure. Only Qwen2.5-7B passed (0.920). LLaDA 0.886, Qwen2.5-1.5B 0.875, Gemma-2 0.872, Dream 0.872, Llama-3.2-3B 0.835, Llama-3.1 0.760, Mistral 0.752, iLLaDA 0.728. Read strictly, only one AR verifier would be left, so there would be no AR+AR pair.

## Decision

The set is kept as a smoke test for bugs. Gate A on the pilot decides competence, for every verifier alike.

## What this means

- The October 26 record argued against moving the bar after seeing results. That objection stands, so this is reported as a post hoc change. The midterm report already says so.
- The standard error of about 0.05 on 50 items explains the near misses (LLaDA, Gemma-2, Dream), but not Llama-3.1 or Mistral.
- The October 26 unbatched-scoring record is adopted: diffusion models are scored at batch size 1.
