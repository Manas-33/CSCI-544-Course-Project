# Dream-v0-Instruct-7B: stage-3 scoring-code tests (October 25, 2026)

Run with `notebooks/04_dream.ipynb` on Colab: NVIDIA L4, Python 3.13, torch 2.5.1+cu124, transformers 4.46.2, revision `05334cb`, template T1. `torchvision` and `torchaudio` were uninstalled first: Colab's builds don't match torch 2.5.1, and `transformers` imports `torchvision` on load.

| Check (PLAN.md §4, §7) | Result | File |
|---|---|---|
| Smoke test on a tiny copy of the official code | PASS | `tiny.json` |
| Answer slots, chosen on the development bank (median Yes+No mass; ties to fewer) | **1 slot**: 0.941 (8 slots 0.714, 32 slots 0.802) | `slots.json`, `dev_T1_n*.jsonl` |
| Usable-scores gate (median Yes+No mass ≥ 0.5 on the development bank) | PASS (0.941) | `slots.json` |
| Test 2: parity with the official code's first forward pass (also the shift check) | PASS: 20/20 items (10 math, 10 code), identical prompt ids, score difference 0.0 | `test2_parity_n1.json` |
| Test 3: repeat runs | PASS (identical) | `test3_invariance_n1.json` |
| Test 3: batched (8) vs unbatched | FAIL: largest difference 0.25, bound 0.05 | `test3_invariance_n1.json` |
| Test 1: sanity AUROC ≥ 0.90 (pooled, as applied so far) | FAIL: 0.872 batched (the recorded run), 0.882 unbatched | `test1_sanity_n1.json`, `invariance_scores_n1.json` |

**Batching.** Every batched-vs-unbatched difference is a multiple of 0.0625, one BF16 rounding step at Dream's logit sizes (median 0.0625, 28 of 50 items above 0.05). That pattern is rounding, not a padding leak, which would give irregular differences. Scoring Dream one item at a time avoids it, as is already done for iLLaDA.

**Sanity AUROC by domain** (unbatched scores, the 50-item set):

| Pairs compared | AUROC |
|---|---|
| math right vs math wrong (169 pairs) | 0.970 |
| code right vs code wrong (144 pairs) | 0.993 |
| math right vs code wrong (156 pairs) | 1.000 |
| code right vs math wrong (156 pairs) | 0.564 |
| pooled (625 pairs) | 0.882 |

Dream scores math higher across the board (median scores: math right +1.88, math wrong +0.50, code right +0.63, code wrong −2.81), so a right code answer often scores below a wrong math answer. That cross-domain comparison is what pulls the pooled AUROC under 0.90. PLAN.md §5 sets thresholds per domain, so the main analysis never compares across domains.

**Open decision (not taken here).** PLAN.md §4 test 1 says "AUROC ≥ 0.90" without saying pooled or per domain. It has been applied pooled so far; read per domain, Dream passes both domains and Qwen2.5-7B fails math (0.876). The choice is the team's, and since it would be made after these results were seen, it must be recorded in `decisions/` as such and applied to every model.

At score 0 (unbatched), Dream accepts 25/25 right and 10/25 wrong sanity answers. On the sanity set its first slot's top token is often `<|endoftext|>` (27 of 50) and the median Yes+No mass is 0.49; the usable-scores gate is judged on the development bank, where it is 0.94.
