# Pilot status, and changes already made in it

**Date:** 2026-11-21 · **Decided by:** the team, by merging this record · **Seen when decided:** the full pilot results (October 27) and the midterm feedback

## Status

The 500-candidate pilot ran on October 27 (`bank/pilot/`, `results/pilot/`) and its results have been seen. The gate values date from PLAN.md of October 19 (commit `7511f54`), before the pilot. The label audit and scoring-code test 4 have not run, so the data-and-pipeline gate is not passed and gate outcomes are provisional.

## Changes made in the pilot

| Planned | Done |
|---|---|
| Falcon3-1B as the weak generator | OLMo-2-7B; GPT-4.1-mini as the API model |
| Generator cap 30% (40% with three generators) | Counts only generators able to supply 10% of a domain's target; math has three, so 40% |
| Code labels in a container without network | EvalPlus's execution guard on a disposable cloud machine |
| AR attention unspecified | SDPA, except Gemma-2 (eager); Qwen scored 0.773 on the sanity set with eager and 0.920 with SDPA |
| iLLaDA as a verifier | Excluded ([record](2026-10-25-illada-excluded.md)) |
| Batched and unbatched diffusion scores agree | They differ by up to 0.25, so diffusion models are scored unbatched ([record](2026-10-26-diffusion-scored-unbatched.md)) |
| AR reference check | Mistral matches on 17 of 20 items (largest difference 0.125, one or two BF16 steps); it is not a primary verifier |
