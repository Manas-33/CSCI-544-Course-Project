# Math development bank

Matched pairs for checking the scoring pipeline, as described in [PLAN.md](../PLAN.md) §2. Each pair is one correct and one wrong answer to the same GSM8K question, written by Phi-4-mini, a generator outside the verifier families, with automatic labels.

These are **pipeline checks** for the scorers, never results: PLAN.md §2 says development-bank numbers are never reported. These questions are kept out of the pilot and the final bank.

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
- **What verifiers see:** the tested program. Imports, definitions and the module-level constants they need (read in their bodies or in their decorators, defaults, annotations and class bases, following chains such as `BASE` → `POW`) are kept; example calls, prints, asserts, `input()` lines and `__main__` blocks are dropped, as EvalPlus's sanitizer does. An answer counts as cut off only if the program itself is (no closing fence), not when only the explanation after it ran out of tokens. An answer that doesn't parse or doesn't define the entry point is dropped as "no answer", not labeled wrong.
- **Labels:**
  - Every usable program runs against the tests **twice**, each run in a fresh process, inside a Docker container with no network, a 2 GB memory cap and a read-only filesystem. The timeout is 10 s per program per run. The two runs use fixed, different hash seeds (0 and 1), so rebuilds give the same labels; a program whose result depends on string-set order is caught as flaky when the two seeds give different outcomes.
  - A program that passes one run and fails the other is **flaky** and dropped.
  - Each failure gets an error type: `wrong_output`, `exception` or `timeout`.
  - Each problem's reference solution is run too, and a problem whose reference fails is dropped as broken.
  - `--sandbox none` runs the same jobs as plain subprocesses, **for tests only**: it has no network isolation. The choice is recorded in every label's `checker_version`, and `bank.make_pairs` refuses such code labels unless given `--allow-unsandboxed-labels`.
- **Where to run labeling:** on a machine with Docker running (`docker pull python:3.11-slim` first). Copy `generations.jsonl` back from Colab.

**Not tested yet:** the GPU generation step, and the sandbox in the `python:3.11-slim` image itself. The Docker path was checked with another local Python 3.11 image (no network, the 2 GB memory cap, the 16 MB file limit, labels as expected). The CPU tests cover extraction, the runner, labels and pairs.

## Full bank (`full/`, not built yet)

About 2,100 math and 850 code answers for the final comparison (PLAN.md §2). Math and code are separate banks and can be built one at a time. Every step stops if its inputs are incomplete or not the expected ones.

| Step | Command | Needs |
|---|---|---|
| 1. Pick problems | `python -m bank.select_full --sources <folder>` → `full/questions_*.jsonl` | the five source files |
| 2. Generate | notebook `10_full_generate.ipynb` (`bank.generate_full`) | A100; GPT-4.1-mini needs only an API key |
| 3. Label, pair, draw, split | notebook `11_full_label_split.ipynb` | CPU; **label keeper only** |

Step 3 runs `bank.generate_full merge`, `bank.label` or `bank.label_evalplus`, `bank.make_pairs --split full`, `bank.token_lengths`, `bank.draw_full` and `bank.split_full`, per domain. The rules are in their docstrings: the draw rule and the order of its caps in `draw_full`, the split in `split_full`, what counts as a near-duplicate problem in `select_full`.

**Files per domain** (`full/math/`, `full/code/`):
- `bank.jsonl`: the pilot bank's fields, `split` is `validation` or `test`, no label. The bank is identified by its SHA-256.
- `labels_validation.jsonl`, and `labels_test.jsonl` in the label keeper's private folder: the pilot label file's fields, plus `error_type` for the tagged math answers. Code answers have one of EvalPlus's two error types: `wrong_output` (a crash counts here too) or `timeout`.
- `manifest.json`: SHA-256 of every file (`files`), counts per split, shortfalls, seeds, and the share of (problem, generator) cells that qualified.

Selected so far (`full/questions_manifest.json`): 1,059 GSM8K-Platinum, 239 MATH-500, 500 new MATH and 390 code problems. Two of the 392 unused code problems are near-duplicates of development problems and were left out.

**Sources and licences**, as found on November 21, 2026 (read on the linked pages; `licenses` in `full/questions_manifest.json` keeps PRM800K's text, the start of EvalPlus's, and notes on the two dataset cards):

| `source` | From | Licence found |
|---|---|---|
| `gsm8k-platinum/test` | [GSM8K-Platinum](https://huggingface.co/datasets/madrylab/gsm8k-platinum) (Vendrow et al.), built on [GSM8K](https://github.com/openai/grade-school-math) (Cobbe et al.) | GSM8K: MIT, as GSM8K-Platinum's card says. GSM8K-Platinum's added annotations: CC BY-SA 4.0 |
| `math-500/test` | [MATH-500](https://huggingface.co/datasets/HuggingFaceH4/MATH-500), from [MATH](https://github.com/hendrycks/math) (Hendrycks et al.) | MATH's repository: MIT. MATH-500's own card states none and points to PRM800K |
| `math-rest/test` (the new problems) | [PRM800K](https://github.com/openai/prm800k)'s `math_splits/train.jsonl` (Lightman et al.), from MATH | PRM800K's repository: MIT |
| `evalplus/humaneval+` | [HumanEval+](https://github.com/evalplus/humanevalplus_release) v0.1.10 (EvalPlus, Liu et al.) | Apache-2.0 |
| `evalplus/mbpp+` | [MBPP+](https://github.com/evalplus/mbppplus_release) v0.2.0 (EvalPlus, Liu et al.) | Apache-2.0 |

This records what the repositories and cards said; it is not a full review of the terms. All terms, including GPT-4.1-mini's for its answers, are checked again before the release.

**Label custody.** `bank/.gitignore` ignores everything in `full/` except the files listed in it, so labeled answers, test labels, execution logs and error tags in `full/` are not committed by accident (a test checks this). Labeling files never belong anywhere else in the repository. `bank.split_full` refuses to write test labels near the bank or inside a git checkout, or to write over a different bank or other test labels: a corrected bank is a new version, with new public and private folders (`VERSION` in notebook 11). The label keeper is the person who holds the test labels until the one join; nobody else keeps a copy. Labeling never runs on a machine or account used for scoring, selection or thresholds.

**Error tags.** `python -m bank.split_full error-sheet` draws 200 wrong math test answers into the private folder; `apply-tags` adds the filled-in tags (`arithmetic_slip`, `misread_problem`, `wrong_method`, `formatting`) to `labels_test.jsonl` before the bank freeze.

## Next: scoring

Score `dev/dev_bank.jsonl` with the same T1 prompt for every verifier:
- **LLaDA, iLLaDA, Dream:** `harness/diffusion/run_score.py`.
- **Qwen2.5-7B, Mistral-7B:** `harness/run.py` with `configs/ar_pair_dev.json`.

Use the scores for what PLAN.md §4 assigns to the development bank (answer-slot length, Yes/No token sets, the usable-scores gate and the scoring-code checks), not as results. LLaDA's development-bank scores at 1, 8 and 32 answer slots are in `results/llada-comparison-20261019-213107/` (1 slot was chosen).

## Tests

`python -m pytest bank/tests -q`. CPU only; no downloads.
