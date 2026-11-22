# Backup score on the pilot (November 22, 2026)

LLaDA's reconstruction score (PLAN.md §4, §7) on the pilot's 250 math answers, from notebook 12: 128 samples, seeds 544
and 545, unbatched, NVIDIA L4, commit `489426a`. It matches the official code (6 of 6), passes gate A (AUROC 0.652) and
fails the repeat check (0.949, needs 0.95), so it fails as the repair. The repair is not needed: LLaDA's verdict score
passed gate A and the swap test in math. The development-bank run for the sample count has not been done.

`analysis.reconstruction pilot` on these files gives the same report.
