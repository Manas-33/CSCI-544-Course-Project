# The license check comes before the release, not before generating

**Date:** 2026-11-21 · **Decided by:** the team, by merging this record · **Seen when decided:** the full pilot results (October 27) and the midterm feedback; no full-bank answer has been generated · **Replaces:** "generator and dataset terms are checked before the full bank is generated" (PLAN.md §9), and, in [the math-sources record](2026-11-21-math-bank-sources.md), that the license and the terms for redistributing MATH text are confirmed at download

## What happened

PLAN.md §9 asked for the terms of every dataset and generator before the full bank is generated, so that nothing is built that cannot be released. The check holds up the bank, which everything else waits for, and only the release depends on it.

## Decision

- The terms of every dataset and generator are checked **before the release**.
- At download, the source's fields, its license as stated there and the file's hash are still recorded.
- Math and code stay together; the check covers both.
- The sources' stated licenses were read at download, and the question files are public with attribution ([details](2026-11-21-full-bank-details.md)). The check before the release still covers every source and generator, GPT-4.1-mini's terms included.

## What this means

- The experiment does not depend on the answer. No problem, generator or answer is dropped from the bank because of its terms.
- If the terms of some text do not clearly allow release, the released bank lists those rows by id only, as already planned for the MATH problems (PLAN.md §2). A generator is no longer swapped for this reason; its outputs are left out of the release instead.
- The risk accepted: part of the bank may turn out not to be releasable. The question files are in the public repository before that check, and the bank freeze commits the answers, GPT-4.1-mini's included. Text committed there stays in the git history even if the release leaves it out, so the terms of whatever a commit publishes are best checked before that commit.
