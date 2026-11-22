# Details of the full bank

**Date:** 2026-11-21 · **Decided by:** the team, by merging this record · **Seen when decided:** the full pilot results (October 27), the midterm feedback and the real problem selection; no full-bank answer has been generated · **Clarifications and facts found while building the bank**

## Code error types

PLAN.md §2 lists wrong output, exception and timeout. EvalPlus's check does not separate exceptions from wrong output, so code has two types: **wrong output** (exceptions included) and **timeout**. Code is descriptive.

## 390 code problems

392 HumanEval+ and MBPP+ problems are unused. `mbpp-plus-130` and `mbpp-plus-743` are near-duplicates of development problems and are left out, because near-duplicates stay in one set (PLAN.md §2). Expected at the pilot's yield: about 840 code answers and 2,090 math answers.

## Choices PLAN.md §2 leaves open

| Point | Choice |
|---|---|
| Order of the caps | Length filter; then the per-problem cap (seeded random order over all pairs); then G, M and the target; then the generator cap while drawing. A pair lost to the generator cap does not bring back one lost to the problem cap |
| "Available" generator | At least 10% of the target in qualifying pairs (exactly 10% counts), counted after the length filter and before the problem cap, as in the pilot |
| Generator cap | One number per domain. The two math halves take turns, one pair each. Fewer than four available generators means 40%; fewer than three is flagged |
| Expected math shortfall | About 22 pairs (8–37), nearly all in the GSM8K half: the target uses every GSM8K pair, so each one lost to OLMo-2's 40% cap is missing. Reported, not filled (PLAN.md §2) |
| Near-duplicate problem | Jaccard similarity above 0.9 on words, numbers and operators. Also applied to unused GSM8K, MATH-500 and code problems against the pilot and development problems. The sanity set is not part of it |
| Split | Validation is a third of each source's problems (nearest whole number). A near-duplicate group counts for the source of its first member and stays in one split |
| Seeds | The pilot's (544 plus position). They overlap with the pilot's numerically; no prompt is shared |
| Versions | vLLM 0.32.0, transformers 5.19.0, Math-Verify 0.9.0, EvalPlus 0.3.1, and the pilot's model commits. GPT-4.1-mini has no commit to pin |
| Error tags | Kept in the label keeper's private folder and added to `labels_test.jsonl` before the freeze |
| Raw answers | Not committed, only their manifests and hashes |

## One duplicated row

PRM800K lists `test/precalculus/24307.json` twice, with the same problem and answer and different worked solutions. The first is kept. Its 4,501 `test/` rows are 4,500 problems, which with MATH-500 make 5,000.

## Licenses found

| Source | License found |
|---|---|
| GSM8K, PRM800K, MATH's own repository | MIT |
| GSM8K-Platinum's added annotations | CC BY-SA 4.0 |
| EvalPlus (HumanEval+, MBPP+) | Apache-2.0 |

MATH-500's card states none and points to PRM800K. The team judged these to allow publication, so the question files go in the public repository, with attribution. All terms, GPT-4.1-mini's included, are still checked before the release ([record](2026-11-21-license-check-before-release.md)). The Hugging Face copy of MATH is disabled after a takedown notice (PLAN.md §2); that check looks at it again.
