# Math development bank

Builds the development bank of matched correct/wrong answers described in [PLAN.md](../PLAN.md) §2: GSM8K questions, answers from a generator outside the verifier families, programmatic labels, one correct and one wrong answer per question.

These numbers are **pipeline checks** (the midterm's "preliminary development results"), never evidence for H1. The selected questions are excluded from the pilot and the final bank.

## Steps

| Step | Command | Where | Output |
|---|---|---|---|
| 1. Pick questions | `python -m bank.select_questions --n 200 --seed 544` | Anywhere (done: `dev/questions.jsonl`) | `dev/questions.jsonl`, `dev/question_ids.txt`, `dev/questions_manifest.json` |
| 2. Generate answers | `python -m bank.generate` | **GPU (Colab)** | `dev/generations.jsonl`, `dev/generations_manifest.json` |
| 3. Label | `python -m bank.label` | Anywhere | `dev/labeled.jsonl` |
| 4. Matched pairs | `python -m bank.make_pairs` | Anywhere | `dev/dev_bank.jsonl`, `dev/dev_labels.jsonl`, `dev/dev_bank_summary.json` |

Run every command from the repository root. The defaults point at `bank/dev/`.

## Settings (PLAN.md §2 defaults)

- **Questions:** 200 GSM8K **train** questions, seed 544. Train is used so the cleaner GSM8K-Platinum test set stays free for the final bank. The source file's SHA-256 is recorded in the manifest.
- **Generator:** `microsoft/Phi-4-mini-instruct` with a "solve step by step, put the final answer in `\boxed{}`" prompt, temperature 0.7, top-p 0.95, **4 answers per question**, at most 1,024 new tokens. Question *i* uses seed 544 + *i*, so reruns are reproducible. Pin the model with `--revision <commit>`.
- **Labels:** checker 1 reads the last `\boxed{}`; checker 2 is Math-Verify (`pip install math-verify`), or the last number in the answer if Math-Verify is missing. An answer is usable only if it isn't truncated, checker 1 found an answer, and both checkers agree.
- **Pairs:** per (question, generator), near-duplicate answers (token Jaccard > 0.9) are collapsed, then one correct and one wrong answer are drawn with seed 544. **Labels are written to a separate file** (`dev_labels.jsonl`), so scoring code never sees them.

## Which GPU

Generation uses Phi-4-mini (3.8B parameters, about 8 GB of weights), so **any GPU with 16 GB or more works**. An A100 is not required.

| GPU | Generation | Scoring with 7–9B verifiers |
|---|---|---|
| Colab **T4** (16 GB) | Works (float16, slower) | Too small without quantization, which the plan rules out |
| Colab **L4** (24 GB) | **Works well; best value** | **Works** (bfloat16) |
| Colab **A100** (40/80 GB) | Fastest | Works |
| CARC **L40S / A40** (48 GB), **A100** (40/80 GB) | Works | Works |
| CARC **V100** (32 GB), **P100** (16 GB) | Works (float16) | Avoid: no bfloat16, so scores would use a different precision from the other models |

`--dtype auto` (the default) picks bfloat16 where the GPU supports it and float16 otherwise, and records the choice.

## On Colab

```bash
git clone -b feature/dev-bank https://github.com/Manas-33/CSCI-544-Course-Project.git
cd CSCI-544-Course-Project
pip install -r bank/requirements.txt
python -m bank.generate            # about 800 answers; minutes with vLLM on an L4 or A100
python -m bank.label
python -m bank.make_pairs          # prints how many pairs you got
```

If vLLM won't install, use `python -m bank.generate --backend hf` (slower).

## On CARC (Discovery)

GPU jobs go to the `gpu` partition through Slurm. Request an L40S, A40 or A100, and check CARC's "Running Jobs" guide for the exact GPU-request syntax. You need a CARC account on a project allocation, usually added by a faculty PI. CARC OnDemand gives a browser-based Jupyter session if you prefer that to batch jobs. A minimal batch job:

```bash
#!/bin/bash
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1          # add a GPU type per CARC's guide, e.g. an L40S or A100
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=01:00:00
module load conda             # or your own Python environment
cd CSCI-544-Course-Project
python -m bank.generate && python -m bank.label && python -m bank.make_pairs
```

**Expected yield (est.):** Phi-4-mini solves most GSM8K questions, so only some questions produce both a correct and a wrong answer, perhaps 30–60 pairs from 200 questions. If you get fewer than about 30, select more questions with a **new** seed and exclude the IDs already used, rather than re-sampling the same questions.

## Next: scoring

Score `dev/dev_bank.jsonl` with the same T1 prompt for every verifier:
- **LLaDA, iLLaDA, Dream:** `harness/diffusion/` on the `feature/diffusion-scorers` branch.
- **Qwen2.5-7B, Mistral-7B:** the AR scorer (role C).

Report AUROC per verifier with the number of pairs, and examples where one verifier catches an error another misses.

## Tests

```bash
python -m pytest bank/tests -q
```

CPU only; no downloads.
