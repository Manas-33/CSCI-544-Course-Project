# Terms that were undefined are settled

**Date:** 2026-11-21 · **Decided by:** the team, by merging this record · **Seen when decided:** the full pilot results (October 27) and the midterm feedback · **Clarifications, made after the pilot; test 4 has not been run**

| Term | Before | Now |
|---|---|---|
| Gate A "within 0.10 of the small AR verifier" | Could be read two-sided | No more than 0.10 **below** Llama-3.2-3B (the code already did this) |
| Test 4 items and scope | 200 pilot items, pooled or per domain unspecified | 100 math and 100 code items, judged per domain, built as in PLAN.md §4; a failure is a gate A failure in that domain |
| Reconstruction repair | Which model, domain and normalization unspecified | LLaDA on math, per token; Dream's is descriptive |
| Gate B for a score without wordings | Undefined | Two estimates (seeds 544 and 545, 128 samples) on the pilot's math answers, rank correlation at least 0.95 |
| Audit failure | Fix and re-audit, with no limit and no rule for changed gate results | One re-audit; gates and test 4 recomputed on corrected labels; the pool can only shrink; a second failure means H1 is not run |
| Matching check | Outside the verdict table | Its first row |
| Dynamic range after the pilot | "Rebalance" if it fails | Closed; the mix is fixed and later values are only reported |
| Selection ties, threshold rule, bootstrap details | Unspecified | PLAN.md §5 |
| Data-and-pipeline gate | Tests 1–3 pass for every model | Tests 2–3 pass and test 4 run for every verifier in H1; Mistral's 17/20 reference check is recorded |
| Cuts (1)–(4) | Applied if behind | Applied now; Qwen3-8B returns if the go/no-go leaves fewer than three AR verifiers |

**What this means for code.** In code, test 4 is the same as ranking a matched pair. On the pilot the best verifier (Qwen) does that for 72% of pairs, so test 4 is expected to fail in code for every verifier. This does not affect H1, which is math only ([record](2026-11-21-one-final-test.md)). The 80% rule itself is kept as written.
