# Math development bank

Matched pairs for checking the scoring pipeline, as described in [PLAN.md](../PLAN.md) §2. Each pair is one correct and one wrong answer to the same GSM8K question, written by Phi-4-mini, a generator outside the verifier families, with automatic labels.

These are **pipeline checks** (the midterm's "preliminary development results"), never evidence for H1. These questions are kept out of the pilot and the final bank.

## Current bank (October 19, 2026)

| | |
|---|---|
| Questions / answers | 500 GSM8K train questions / 2,000 answers (4 each) |
| Usable answers | 1,980 (20 dropped: no boxed number, or cut off) |
| Phi-4-mini correct | 90.2% |
| **Pairs** | **71** (142 answers) |

**Score `dev/dev_bank.jsonl` only.** `dev/dev_labels.jsonl` and `dev/labeled.jsonl` hold the labels and are for the analysis only.

Close-call labels were checked against the official solutions; none needed changing. One ambiguous question is excluded, with its reason, in `dev/excluded_questions.txt`. Run details (model revision, library versions, GPU) are in `dev/generations_provenance.json`.

## Rules

- **Labels are automatic.** Two checkers (the boxed answer, and Math-Verify) must agree. An answer is dropped if it is cut off, has no boxed number, or says "As an AI…". No label is ever set by hand.
- **One correct + one wrong answer per question,** drawn with fixed seeds. Near-identical answers are merged, but only within the same label, so a wrong answer that differs from a correct one by a final slip is kept.
- **Labels never appear in the bank file,** and row order inside a pair doesn't reveal which answer is correct.
- **Scorers read `candidate_id`, `problem` and `candidate`** from each bank row. (`questions.jsonl` keeps GSM8K's own `question` field; the bank renames it.)
- **Files are written with LF line endings on every OS,** and `.gitattributes` keeps them that way in git, so a file's SHA-256 is the same on every machine.
- **Exclusions:** add a broken or ambiguous question to `dev/excluded_questions.txt` as `problem_id  # reason`. It is removed, never relabeled.

## Rebuilding

Run from the repository root:

| Step | Command | Needs |
|---|---|---|
| 1. Pick questions | `python -m bank.select_questions` | — |
| 2. Generate answers | `python -m bank.generate` | GPU |
| 3. Label | `python -m bank.label` | — |
| 4. Build pairs | `python -m bank.make_pairs` | — |

On Colab, with an **L4** runtime (about 6 minutes for generation):

```
%cd /content
!git clone https://github.com/Manas-33/CSCI-544-Course-Project.git
%cd /content/CSCI-544-Course-Project
!pip install -q -r bank/requirements.txt
!python -m bank.generate && python -m bank.label && python -m bank.make_pairs
```

**GPU:** an L4 is the best value. A T4 works for generating but is too small for scoring the 7–9B verifiers. For scoring, avoid V100 and P100 GPUs (on CARC), since they lack bfloat16.

**Need more pairs?** Phi-4-mini gets about 90% right, so only about 1 question in 7 yields a pair. Enlarge the question set **before** anyone scores the bank, never after seeing results.

## Code development bank (`dev_code/`, in progress)

The code half of the development bank. It covers 150 MBPP problems, chosen with seed 544 from the 594 that are **not** in MBPP+. That keeps all 542 EvalPlus problems (HumanEval+ and MBPP+) free for the pilot and the final bank.
- **Tests:** MBPP's own three asserts per problem. These are weaker than EvalPlus's tests, which is fine for pipeline checks but never for results.
- **Problem text:** MBPP+'s layout (the task plus one example assert, in a docstring).

| Step | Command | Needs |
|---|---|---|
| 1. Pick problems | `python -m bank.select_code_questions` (done: `dev_code/questions.jsonl`) | — |
| 2. Generate | `python -m bank.generate --questions bank/dev_code/questions.jsonl --out bank/dev_code/generations.jsonl` | GPU |
| 3. Label | `python -m bank.label_code` | Docker |
| 4. Build pairs | `python -m bank.make_pairs --questions bank/dev_code/questions.jsonl --generations bank/dev_code/generations.jsonl --labeled bank/dev_code/labeled.jsonl --out bank/dev_code` | — |

- **Generation** uses the EvalPlus chat prompt. The reply is pre-filled up to an opening ` ```python ` fence, so the model writes the program directly. Settings follow PLAN.md §2: temperature 0.8, top-p 0.95, 8 answers per problem, at most 768 new tokens. These are picked automatically when the questions are code.
- **What verifiers see:** the tested program. Imports, definitions and the constants they use are kept; example calls, prints, asserts and `__main__` blocks are dropped, as EvalPlus's sanitizer does. An answer that doesn't parse or doesn't define the entry point is dropped as "no answer", not labeled wrong.
- **Labels:**
  - Every usable program runs against the tests **twice**, each run in a fresh process, inside a Docker container with no network, a 2 GB memory cap and a read-only filesystem. The timeout is 10 s per program per run.
  - A program that passes one run and fails the other is **flaky** and dropped.
  - Each failure gets an error type: `wrong_output`, `exception` or `timeout`.
  - Each problem's reference solution is run too, and a problem whose reference fails is dropped as broken.
  - `--sandbox none` runs the same jobs as plain subprocesses (on Colab, or in tests). The choice is recorded in every label's `checker_version`.
- **Where to run labeling:** on a machine with Docker running (`docker pull python:3.11-slim` first). Copy `generations.jsonl` back from Colab.

**Not tested yet:** the GPU generation step and the Docker sandbox. The CPU tests cover extraction, the runner, labels and pairs.

## Next: scoring

Score `dev/dev_bank.jsonl` with the same T1 prompt for every verifier:
- **LLaDA, iLLaDA, Dream:** `harness/diffusion/` on the `feature/diffusion-scorers` branch.
- **Qwen2.5-7B, Mistral-7B:** the AR scorer (role C).

Report AUROC per verifier with the number of pairs, and examples where one verifier catches an error another misses.

## Tests

`python -m pytest bank/tests -q`. CPU only; no downloads.
