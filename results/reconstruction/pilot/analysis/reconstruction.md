# Reconstruction score on the pilot

Bank: `bank/pilot/pilot_bank.jsonl` (sha256 aa4d8cc3ee3c…)

Pilot numbers: exploratory. The score is the per-token masked-reconstruction log-likelihood (AR models: exact per-token log-probability). Test 4 is not applied to it.

Pairs dropped because their labels are no longer one correct and one wrong answer: 0

## Repair rule (PLAN.md §7): LLaDA, math, 128 samples, seed 544

- **Outcome: fail**
- Gate A: pass
- Repeat check (seed 545, Spearman at least 0.95): **fail** (0.9489)

## All runs

Math decides the repair; every other row is information only, and code is descriptive. Gate A and the repeat check are shown for the 128-sample seed-544 diffusion runs.

| Verifier | Method | Samples | Seed | Domain | Scored | AUROC [95% CI] | Verdict AUROC (T1) | Pair rank acc. | Gate A | Repeat (ρ) | s / answer |
|---|---|---|---|---|---|---|---|---|---|---|---|
| LLaDA-8B | diffusion_mc | 128 | 544 | math | 250 / 250 | 0.652 [0.600, 0.703] | 0.607 | 0.688 | pass | 0.9489 | 33.828 |
| LLaDA-8B | diffusion_mc | 128 | 545 | math | 250 / 250 | 0.665 [0.617, 0.713] | 0.607 | 0.744 | – | – | 33.812 |
