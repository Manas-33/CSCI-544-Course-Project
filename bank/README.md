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
!git clone -b feature/dev-bank https://github.com/Manas-33/CSCI-544-Course-Project.git
%cd /content/CSCI-544-Course-Project
!pip install -q -r bank/requirements.txt
!python -m bank.generate && python -m bank.label && python -m bank.make_pairs
```

**GPU:** an L4 is the best value. A T4 works for generating but is too small for scoring the 7–9B verifiers. For scoring, avoid V100 and P100 GPUs (on CARC), since they lack bfloat16.

**Need more pairs?** Phi-4-mini gets about 90% right, so only about 1 question in 7 yields a pair. Enlarge the question set **before** anyone scores the bank, never after seeing results.

## Next: scoring

Score `dev/dev_bank.jsonl` with the same T1 prompt for every verifier:
- **LLaDA, iLLaDA, Dream:** `harness/diffusion/` on the `feature/diffusion-scorers` branch.
- **Qwen2.5-7B, Mistral-7B:** the AR scorer (role C).

Report AUROC per verifier with the number of pairs, and examples where one verifier catches an error another misses.

## Tests

`python -m pytest bank/tests -q`. CPU only; no downloads.
