# Diffusion verifier scorers (role D)

Scores a candidate with LLaDA-8B-Instruct, iLLaDA-8B-Instruct or Dream-v0-Instruct-7B, as fixed in [PLAN.md](../../PLAN.md) §4. The prompt, in the model's own chat template, is followed by N masked answer slots (1, 8 or 32). One forward pass gives **s = log P(Yes) − log P(No)** at the first masked slot, plus P(Yes), P(No) and the Yes+No mass. Dream's logits are shifted right by one, so its first slot is read one position earlier. iLLaDA is scored unbatched.

| File | What it is |
|---|---|
| `scorer.py` | Model specs, Yes/No token sets, the scorer classes, `load_scorer()` |
| `interface.py` | Score row (PLAN §9) and the T1 prompt, the same text as `harness/prompts.py` (a test checks this) |
| `run_score.py` | Scores a JSONL of candidates into score rows; resumes by skipping ids already written |
| `reconstruction.py`, `run_reconstruction.py` | The reconstruction score and its runner (last section) |
| `checks/` | Scoring-code tests 1–3 (GPU) and a CPU smoke test on the real model code |
| `requirements/` | One file per model environment, plus `cpu-tests.txt` |
| `tests/` | CPU unit tests with tiny mocks; no downloads |

## Verified model facts (checked 2026-10-19)

| | LLaDA | iLLaDA | Dream |
|---|---|---|---|
| Hugging Face id | `GSAI-ML/LLaDA-8B-Instruct` | `GSAI-ML/iLLaDA-8B-Instruct` | `Dream-org/Dream-v0-Instruct-7B` |
| Pinned revision | `08b83a6` | `5769f04` | `05334cb` |
| Mask id | 126336 `<\|mdm_mask\|>` | 5 `<[MASK]>` | 151666 `<\|mask\|>` |
| First slot read at | slot | slot | slot − 1 (shift) |
| Batched | yes, left padding | no (official code asserts batch 1) | yes, official 4D mask and position ids |
| Context (prompt + slots) | 4096 | 8192 | 2048 |
| Environment | `transformers==4.38.2` | `transformers==4.57.1` | `transformers==4.46.2`, `torch==2.5.1` |

Sources: the model cards and `config.json` / `tokenizer_config.json` on Hugging Face; `README.md`, `generate.py` and `eval_illada.sh` in [ML-GSAI/LLaDA](https://github.com/ML-GSAI/LLaDA) (commit `9182493`); `README.md` and `generation_utils.py` in [HKUNLP/Dream](https://github.com/HKUNLP/Dream). All six Yes and six No spellings (`Yes`, ` Yes`, `yes`, ` yes`, `YES`, ` YES`, and the same for No) are single tokens in all three tokenizers. The revisions pin both the weights and the remote model code; `load_scorer()` refuses to run in the wrong environment unless `--allow-env-mismatch` is passed.

## Running on Colab

Use **one fresh runtime per model**: the three environments conflict. Each model needs about 16 GB of GPU memory in bf16, and Dream needs at least 20 GB, so use an L4 (24 GB) or an A100, not a T4.

```bash
git clone https://github.com/Manas-33/CSCI-544-Course-Project.git /content/project
cd /content/project
pip install -q -r harness/diffusion/requirements/llada.txt   # or illada.txt / dream.txt
# Dream only: restart the runtime now (its pin replaces Colab's torch), then cd back
git clone -q https://github.com/ML-GSAI/LLaDA /content/LLaDA && git -C /content/LLaDA checkout -q 9182493   # LLaDA and iLLaDA
```

In a notebook cell, prefix each line with `!` (and use `%cd` instead of `cd`). Then, in this order (`--llada-repo` is needed for LLaDA and iLLaDA only):

```bash
python -m pytest harness/diffusion/tests -q                                            # unit tests (CPU)
python -m harness.diffusion.checks.check_tiny --model llada --llada-repo /content/LLaDA  # smoke test (CPU, no weights)
python -m harness.diffusion.checks.check_parity --model llada --n-slots 1 --llada-repo /content/LLaDA --report reports/llada_parity_n1.json
python -m harness.diffusion.checks.check_sanity --model llada --n-slots 1 --report reports/llada_sanity_n1.json
python -m harness.diffusion.checks.check_invariance --model llada --n-slots 1 --save reports/llada_scores_n1.json
python -m harness.diffusion.run_score --model llada --n-slots 1 --items bank/dev/dev_bank.jsonl --bank-hash <sha256 of that file> --out results/llada_dev_T1_n1.jsonl
```

Each check prints PASS or FAIL, exits non-zero unless it passes, and writes a JSON report with `--report`. Save reports and scores to Google Drive (`/content/drive/MyDrive/...`) so they survive the runtime. For the stronger repeat-run check, rerun `check_invariance` in a fresh runtime with `--previous reports/llada_scores_n1.json`.

## Scoring-code tests (PLAN §4; all must pass before any score is used)

| Test | Script | Passes when |
|---|---|---|
| 1. Sanity set | `check_sanity.py` | AUROC ≥ 0.90 on the 50-item sanity set. Also prints the median Yes+No mass and the top token at the first slot |
| 2. Reference parity | `check_parity.py` | On 20 prompts: the prompt ids equal the official ones, and the official sampler's first forward pass (LLaDA/iLLaDA: `generate()`; Dream: `diffusion_generate()`, after its shift) gives the same top token and a score within 1e-3. For Dream this is also the shift check. Saves the rendered prompt for one example (PLAN §9) |
| 3. Invariance | `check_invariance.py` | Repeat runs: Spearman > 0.999. Batched vs unbatched: \|Δs\| < 0.05 (skipped for iLLaDA) |
| Smoke test | `check_tiny.py` | The same parity and batching checks on a 2-layer random-weight model built from the official code. Already passes locally for all three models in their pinned environments; it does not replace tests 1–3 |

All checks run on the official 50-item sanity set, `bank/sanity/sanity_set.jsonl`, by default (`--items` takes any JSONL with `candidate_id`, `problem`, `candidate`). Run the tests for each answer-slot length you use.

## The sanity set and the saved LLaDA run

The sanity set is `bank/sanity/sanity_set.jsonl`: 50 items (25 right, 25 wrong; 26 math, 24 code), shared with the AR harness. See `bank/sanity/README.md`. It is a bug detector: never tune anything on it.

LLaDA has already been scored once on it, by an earlier scorer: 1 answer slot, AUROC 0.885 (fails the 0.90 bar). Before relying on that run, check that this scorer gives the same scores; the commands are in `harness/README.md` under "Parity between the two LLaDA scorers".

## Answer prefixes (iLLaDA: thinking off)

`--answer-prefix` (on `run_score` and every check) appends a fixed text after the chat template's generation prompt and before the answer slots; the choices are in `scorer.ANSWER_PREFIXES` (`none`, `think_nl`, `think_nlnl`; each puts a newline before `</think>`, so the closing tag tokenizes like the official end-of-thinking text), and every score row and report records the one used. The user message is never changed. iLLaDA opens every answer with a `<think>` block, so without a prefix its first slot holds `<` and almost no Yes/No mass; `decisions/2026-10-25-illada-thinking-off.md` turns its thinking off and chooses the prefix together with the slot count on the development bank (`choose_slots.py` takes every file and picks the highest median Yes+No mass, ties to fewer slots, then the simpler prefix). LLaDA and Dream use `none`. `checks/check_parity.py` gives the official prompt the same prefix, so test 2 still compares like with like.

## Notes for role C and the team

- **T1 matches the AR harness.** `interface.t1_message()` produces the same text as `harness/prompts.py`'s T1. T2 and T3 are not here yet; when they are added, take them from `harness/prompts.py` and rerun tests 1–3.
- **Score rows** use the PLAN §9 fields listed in `interface.py`. The score is never sign-flipped there; T3's flip belongs to the analysis. Each `run_score` run also appends a small manifest line to `<out>.manifest.jsonl`.
- **iLLaDA may want to think first.** Its official evaluations of reasoning tasks close a `<think>…</think>` block (`end_think_logit_boost`), so its first answer slot may favour `<` over Yes/No. Test 1 prints the top token, and the usable-scores gate (median Yes+No mass ≥ 0.5) will show it. Any fix (such as a pre-filled empty think block) changes the prompt for one verifier, so it is a team decision for `decisions/`, not a scorer change.
- **Dream's template adds** "You are a helpful assistant." as a system message when none is given. This is the official template, kept as is.
- **Answer-slot length** is chosen per model on the development bank (PLAN §4): run `run_score` with 1, 8 and 32 slots and pick the highest median `yes_no_mass`, ties to fewer slots. Lock the tie rule before looking.
- **Loading** uses `device_map` (needs `accelerate`) so weights go straight to the GPU instead of through Colab's ~12 GB of system RAM.

## Unit tests (CPU, no downloads)

```bash
pip install -r harness/diffusion/requirements/cpu-tests.txt
python -m pytest harness/diffusion/tests -q
```

They cover the Yes/No sets (multi-token spellings dropped, overlap rejected), the verdict arithmetic, mask-slot indexing for 1, 8 and 32 slots, Dream's shift against the official shifted logits, Dream's official mask and position ids, iLLaDA's unbatched calls, left padding on tiny attention models (including a deliberately leaky scorer that the test catches), score rows, the scoring loop's `too_long` and `error` rows, and the AUROC and Spearman helpers.

## Reconstruction score (LLaDA and Dream; PLAN §4 and §7)

`reconstruction.py` estimates how likely the candidate is given the problem: the problem is a user turn in the model's own chat template, the candidate is the assistant's answer, and the score is the **per-token masked-reconstruction log-likelihood** (higher = more plausible). It follows LLaDA's official `get_log_likelihood` (the commit pinned above): each Monte Carlo sample masks k of the L answer tokens, never the prompt, and is weighted by L/k; mask counts are spread evenly within groups of 16. Dream uses the same estimate with its one-position shift. `harness/ar_logprob.py` is the AR arm: the exact log-probability of the same answer tokens, in the same row format. It is LLaDA's only allowed repair (math, 128 samples, seed 544; PLAN §7) and otherwise a side experiment.

```bash
python -m harness.diffusion.checks.check_reconstruction --model llada --llada-repo /content/LLaDA --report reports/reconstruction_llada.json   # GPU
python -m harness.diffusion.run_reconstruction --model llada --domain math --seed 544 --items bank/pilot/pilot_bank.jsonl --out results/reconstruction/pilot/llada_math_s544.jsonl
python -m harness.ar_logprob --config configs/pilot_qwen_mistral.json --which A --items bank/pilot/pilot_bank.jsonl --out results/reconstruction/pilot/ar_qwen25.jsonl
python -m analysis.reconstruction pilot --bank bank/pilot/pilot_bank.jsonl --labels bank/pilot/pilot_labels.jsonl --scores results/reconstruction/pilot --verdicts results/pilot/ar results/pilot/llada results/pilot/dream --out results/reconstruction/pilot/analysis
```

Notebook 12 runs all of it, and can also score the side experiment: 500 validation and 500 test answers of the math bank (whole problems, seed 544, drawn by `analysis.reconstruction draw`). It runs on an A100 or an L4, one type for all the pilot scores; full-bank answers are scored on one type too, the type of notebook 13's scores ([record](../../decisions/2026-11-22-scoring-one-gpu-type.md)), after the final model record is merged. What is fixed:

- **End-of-turn token:** neither shown nor scored, in every model. The official code scores the answer ids it is given and appends nothing.
- **Answer text:** the candidate without leading and trailing whitespace (LLaDA's, Llama's and Gemma's templates drop it anyway). The whole text is tokenized and cut after the prompt's tokens; if a token would span the boundary, the answer is tokenized on its own, and the row's `boundary` says `separate`.
- **Seeds:** each item's random numbers come from the seed and its candidate id, so the order of the items and resuming change nothing. Fewer samples are a prefix of more, and `loglik_at` keeps the estimates from 8 to 128 samples, for the development-bank choice (`analysis.reconstruction samples`).
- **`--micro-batch`** (samples per forward pass): 1. Samples of one item have equal length, so nothing is padded and FP32 results agree, but BF16 can round differently for another matrix shape (`decisions/2026-10-26-diffusion-scored-unbatched.md`). The GPU check reports the difference.
- **Rows** are described in `run_reconstruction.py`. A file scored with other settings is refused; `error` rows are retried; answers that do not fit the context (`too_long`) or have no text (`empty`) get a row without a score.
