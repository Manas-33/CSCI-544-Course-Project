# Next steps

**Start here.** The plan is [PLAN.md](PLAN.md): read §1–5 for the design and §13 for what changed from our original idea in [PROJECT.md](PROJECT.md). This page is the to-do list. Tick items off in the repo as they get done.

## This week

| # | What | Who | By |
|---|---|---|---|
| 1 | **Assign roles A–F** (PLAN §11): A math bank, B code bank, C verifier harness, D diffusion specialist, E metrics and statistics, F infrastructure and writing | Everyone | Oct 20 |
| 2 | **Midterm presentation** (2 min, 2 slides): problem, method, how the instructor feedback was addressed (pairs and thresholds chosen on validation data and frozen before the test set), and gate status | Presenter + F | Oct 20–22 |
| 3 | **Record the pilot status** in `decisions/`: has the pilot run, and has anyone seen its results? If not, run the reduced pilot (PLAN §10) | F | Oct 21 |
| 4 | **Decide the bank size** (PLAN §6): stay within 5,000 answers and report the smallest detectable effect, or ask the instructor at the midterm about a bigger bank | Everyone, E leads | Oct 22 |
| 5 | **Falcon3 license:** confirm its outputs can be used and released, or pick another weak generator from an unused family (PLAN §2) | A | Oct 22 |

## Phase 0: setup before the pilot (owner F, target Oct 24)

- [ ] Request Hugging Face access for Llama-3.1-8B-Instruct, Llama-3.2-3B-Instruct and Gemma-2-9B-it (gated)
- [ ] Build the four environments: LLaDA (`transformers 4.38.2`), iLLaDA (`transformers 4.57.1`), Dream (`transformers 4.46.2`, `torch 2.5.1`), AR models (`transformers ≥ 4.51`) (F, with C and D)
- [ ] Measure peak memory on the actual GPU, especially Gemma-2-9B (about 18.5 GB of weights) (C)
- [ ] Confirm Colab compute units and the CARC allocation (F)
- [ ] Check data terms: GSM8K-Platinum annotations are CC BY-SA 4.0 (A)

## First tasks by role

| Role | Before the pilot |
|---|---|
| **A, math bank** | Math half of the ~150-answer development bank; generation defaults (PLAN §2); the two answer checkers; the matched-pair builder |
| **B, code bank** | EvalPlus sandbox (Docker, or Apptainer on CARC); code half of the development bank; flaky-test reruns |
| **C, verifier harness** | Shared T1–T3 wording; AR adapters reading P(Yes)/P(No) in one pass; Yes/No token sets; score-file format; usable-scores check |
| **D, diffusion** | LLaDA, iLLaDA and Dream adapters (mask ids, Dream's one-position shift); answer-slot choice; scoring-code tests 1–3, including the 50-item sanity set (PLAN §4) |
| **E, statistics** | Analysis code tested on synthetic data: pair thresholds on validation, AND rule, bootstrap over whole problems, decision rule (PLAN §5); the pilot precision check (PLAN §6) |
| **F, infrastructure and writing** | Environments, run ledger, the `decisions/` folder, the test-label custody setup (PLAN §2), the midterm report outline |

## Lock before the pilot (record in `decisions/`)

Confirm these before anyone sees pilot results, so no rule is set after the data:
- gate values (PLAN §7);
- T1–T3 wording and the answer-slot rule (PLAN §4);
- generation defaults (PLAN §2);
- the fallback if fewer than three AR verifiers survive the gates (PLAN §3).

## Suggested latest dates (adjust as a team)

| By | Milestone |
|---|---|
| Oct 24 | Phase 0 done |
| **Oct 27** | **Midterm report (4 pages):** how the feedback was addressed, pilot or reduced-pilot results with gate outcomes, scope changes for the instructor (PLAN §10, §12) |
| ~Nov 6 | Full bank built, audited and frozen |
| ~Nov 13 | Pairs and thresholds selected on validation; preregistration freeze |
| ~Nov 16 | Test labels joined once; primary analysis run |
| **Nov 20** | **Milestone 2:** full comparison, with generator and error-type breakdowns |
| Nov 20 – Dec 3 | Final presentation (6 min, 5 slides) |
| Dec 4 | Full report draft |
| **Dec 15, 6:30 PM** | **Final report** (8 pages + appendix), no late tokens |

## About this repo

Only `PLAN.md`, `PROJECT.md` and this file are tracked (see `.gitignore`). The proposal, the instructor feedback and the plan's review files are kept outside the repo. When code starts, add each new folder (for example `bank/`, `harness/`, `analysis/`) to the allow-list in `.gitignore`.
