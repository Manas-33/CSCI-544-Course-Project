# Diffusion verifier scorers (role D)

Scores a candidate with LLaDA-8B-Instruct, iLLaDA-8B-Instruct or Dream-v0-Instruct-7B, as fixed in [PLAN.md](../../PLAN.md) §4. The prompt, in the model's own chat template, is followed by N masked answer slots (1, 8 or 32). One forward pass gives **s = log P(Yes) − log P(No)** at the first masked slot, plus P(Yes), P(No) and the Yes+No mass. Dream's logits are shifted right by one, so its first slot is read one position earlier. iLLaDA is scored unbatched.

| File | What it is |
|---|---|
| `scorer.py` | Model specs, Yes/No token sets, the scorer classes, `load_scorer()` |
| `interface.py` | Score row (PLAN §9) and a **placeholder** T1 prompt. Role C owns both and will align them |
| `run_score.py` | Scores a JSONL of candidates into score rows; resumes by skipping ids already written |
| `checks/` | Scoring-code tests 1–3 (GPU) and a CPU smoke test on the real model code |
| `sanity_set.jsonl` | 4 example items; the team writes the full 50 (see below) |
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
git clone -b feature/diffusion-scorers https://github.com/Manas-33/CSCI-544-Course-Project.git /content/project
cd /content/project
pip install -q -r harness/diffusion/requirements/llada.txt   # or illada.txt / dream.txt
# Dream only: restart the runtime now (its pin replaces Colab's torch), then cd back
git clone -q https://github.com/ML-GSAI/LLaDA /content/LLaDA && git -C /content/LLaDA checkout -q 9182493   # LLaDA and iLLaDA
```

In a notebook cell, prefix each line with `!` (and use `%cd` instead of `cd`). Then, in this order (`--llada-repo` is needed for LLaDA and iLLaDA only):

```bash
python -m pytest harness/diffusion/tests -q                                            # unit tests (CPU)
python -m harness.diffusion.checks.check_tiny --model llada --llada-repo /content/LLaDA  # smoke test (CPU, no weights)
python -m harness.diffusion.checks.check_parity --model llada --n-slots 8 --llada-repo /content/LLaDA --report reports/llada_parity_n8.json
python -m harness.diffusion.checks.check_sanity --model llada --n-slots 8 --report reports/llada_sanity_n8.json
python -m harness.diffusion.checks.check_invariance --model llada --n-slots 8 --save reports/llada_scores_n8.json
python -m harness.diffusion.run_score --model llada --n-slots 8 --items dev_bank.jsonl --bank-hash <hash> --out results/llada_T1_n8.jsonl
```

Each check prints PASS or FAIL, exits non-zero unless it passes, and writes a JSON report with `--report`. Save reports and scores to Google Drive (`/content/drive/MyDrive/...`) so they survive the runtime. For the stronger repeat-run check, rerun `check_invariance` in a fresh runtime with `--previous reports/llada_scores_n8.json`.

## Scoring-code tests (PLAN §4; all must pass before any score is used)

| Test | Script | Passes when |
|---|---|---|
| 1. Sanity set | `check_sanity.py` | AUROC ≥ 0.90 on the 50-item sanity set. Also prints the median Yes+No mass and the top token at the first slot |
| 2. Reference parity | `check_parity.py` | On 20 prompts: the prompt ids equal the official ones, and the official sampler's first forward pass (LLaDA/iLLaDA: `generate()`; Dream: `diffusion_generate()`, after its shift) gives the same top token and a score within 1e-3. For Dream this is also the shift check. Saves the rendered prompt for one example (PLAN §9) |
| 3. Invariance | `check_invariance.py` | Repeat runs: Spearman > 0.999. Batched vs unbatched: \|Δs\| < 0.05 (skipped for iLLaDA) |
| Smoke test | `check_tiny.py` | The same parity and batching checks on a 2-layer random-weight model built from the official code. Already passes locally for all three models in their pinned environments; it does not replace tests 1–3 |

Tests 2 and 3 run on the sanity set by default (`--items` takes any JSONL with `candidate_id`, `problem`, `candidate`), so they report INCOMPLETE until it has 20+ items. Run the tests for each answer-slot length you use.

## The sanity set (team task: write 50 items)

`sanity_set.jsonl` holds 4 examples. Extend it to **50 items: 25 obviously right (`label: 1`) and 25 obviously wrong (`label: 0`)**, one JSON object per line with `candidate_id`, `domain`, `problem`, `candidate`, `label` and `note`.

- **Outside the bank.** Write new problems; do not copy or paraphrase GSM8K, MATH, HumanEval or MBPP.
- **Both domains, balanced.** About 13 math and 12 code items per label, so neither domain is all-correct.
- **Obvious.** Someone can judge it at a glance. A wrong item has one clear error that changes the final answer (math: a wrong sum in the last step; code: a flipped condition or wrong operator), not a subtle edge case.
- **Bank format.** Math candidates show short working and end with `The final answer is \boxed{...}`; code candidates are a single Python function.
- **Varied lengths**, from one line to a paragraph, so batching adds padding in test 3.
- **Ids** `sanity-m001…`, `sanity-c001…`; say in `note` why the item is right or wrong.
- **A second person checks every label.** Commit the set before any model is run on it, and never tune anything on it: it is a bug detector, and the AR adapters (role C) use the same 50 items.

## Notes for role C and the team

- **T1 is a placeholder.** `interface.t1_message()` uses PLAN's question line inside a layout that is not agreed. When C fixes T1–T3, replace `TEMPLATES` and rerun tests 1–3.
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
