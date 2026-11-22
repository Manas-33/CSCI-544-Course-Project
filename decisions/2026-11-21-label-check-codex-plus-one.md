# The label check is done by Codex plus one person

**Date:** 2026-11-21 · **Decided by:** the team, by merging this record · **Seen when decided:** the full pilot results (October 27) and the midterm feedback; no label check has been run · **Replaces:** "labeled independently by two people, with a third resolving disagreements" (PLAN.md §2)

## What happened

The go/no-go waits for a check of 200 pilot labels by three people, and the bank freeze for another 200. Neither has started.

## Decision

1. **Codex** (OpenAI's command-line tool, model `gpt-6.2-sol`; the model used is saved with the replies) judges all 200 candidates, one per call. It sees the problem and the candidate, and has no tools, files or web (`bank/codex_audit.py`). An item is invalid if its log shows anything but the model's own messages.
2. **One person** reads, blind to our label and to Codex's answer, in one shuffled sheet: every item where Codex disagrees with our label or gave no usable verdict, every item written by GPT-4.1-mini (Codex is a GPT model; 13 on the pilot's sheet), and 100 others drawn at random (`bank/audit_sheet.py`).
3. **The pass mark stays: at most 3 label errors in 200.** Every error found counts, whether among the disagreements, the GPT-written items or the others.
4. **The shortcut may pass the bank but never fail it.** It passes if there is no error among the 100 others, at most 3 errors were found in all, and the one-sided 95% upper bound on the bank's label-error rate is below 4%. In every other case the person reads all 200 and PLAN.md's rule decides.
5. **What counts as an error:** the reader's blind verdict differs from our label, or is "unsure". For those items only, and before any more reading is asked for, the reader looks once more with the reference answer or the failing test in view and writes whether the label is wrong, with a reason. Both verdicts are kept. Without a second look the item counts as an error.
6. **Full bank:** the same check on 200 new candidates before the freeze. The reader takes no part in selection or threshold work, so it is A or B (F runs the commands). Apart from the sheet, the files of that check give labels away, so they stay with F and are not committed before the test join.

## The bound

The read-in-full items (disagreements, no verdict, GPT-written) and the others are two groups with their own error rates. The estimate is the errors found in the first group plus the errors found among the others, scaled up to all of them. The bound is the largest bank error rate for which an estimate this low still has a 5% chance, taking the worst split of that rate between the two groups. If all 200 are read, or if no item is read in full, this is the ordinary exact bound (3 in 200 gives 3.83%; 0 in 100 gives 2.95%).

The bound with 100 others read and no error among them (from `python -m bank.audit_sheet table`; bold passes):

| Read in full | Items the person reads | 0 errors there | 1 | 2 | 3 |
|---|---|---|---|---|---|
| 13 (the GPT-written items alone) | 113 | **2.76%** | **2.98%** | 4.35% | 4.59% |
| 20 | 120 | **2.66%** | **2.89%** | 4.19% | 4.44% |
| 30 | 130 | **2.51%** | **2.77%** | **3.96%** | 4.24% |
| 40 | 140 | **2.36%** | **2.65%** | **3.72%** | 4.05% |
| 60 | 160 | **2.07%** | **2.45%** | **3.26%** | 4.31% |

## What checking the rule showed

The rule was computed exactly and simulated (`bank/tests/test_audit.py`).

- **The whole procedure keeps PLAN.md's 95%.** A bank with exactly 4% label errors passes a full reading 4.0% of the time. With the shortcut first, it passes at most 5.0% of the time once 10 good items are read in full (5.1% if none are; the pilot's sheet has 13 GPT-written items), however many of the wrong labels Codex catches.
- **It is usable.** With 20 good items read in full and Codex catching three of four label errors, the shortcut passes a bank with 0.5% label errors 73% of the time and one with 1% errors 43% of the time. Otherwise all 200 are read, so a good bank is never failed by it.
- **Rejected before any answer was read:** reading 75 others, then 118 after one error, then all 200. Each stop is sound alone, but three chances to pass let a 4% bank through 7–9% of the time.

## Known limits

Codex and the reader can make the same mistake. One reader replaces two, so a misread item is caught only by the second look. Codex judges programs without running them, so many code items may need reading. As in PLAN.md's rule, the bound treats the 200 as a plain sample of the bank.
