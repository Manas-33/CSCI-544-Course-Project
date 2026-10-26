# AR+AR development verifier

Run commands from the repository root. Qwen2.5-7B-Instruct + Mistral-7B-Instruct-v0.3 is the initial cross-family baseline. T1 matches PLAN §4 and is identical across models; native chat templates differ. T2/T3 remain provisional (T3 asks whether the final answer is incorrect, not whether every reasoning step is valid).

The current bank is complete: 500 selected questions, 2,000 generations, **142 candidates in 71 matched pairs**, with separate labels. The official **50-item sanity fixture** is `bank/sanity/sanity_set.jsonl`: 25 correct/25 wrong (26 math, 24 code), outside the development bank. It supersedes the separate 12-example synthetic smoke fixture for the project's sanity check; neither is research evidence.

Sanity scoring strips `label` and `note` before prompt construction. After the second-person review required by `bank/sanity/README.md`, score all 50 with T1 and evaluate against the same file. AR models use SDPA attention; the earlier BF16 run used eager attention and must not be used as sanity-gated evidence. Each model must reach AUROC >= 0.90; partial runs cannot pass. No matched pairs or problem IDs are invented. Sanity evaluation writes `sanity.png`, null within-pair rankings, and a `sanity_check` in `metrics.json`; a failed AUROC check returns exit code 1. Threshold diagnostics are disabled for this fixture.

```bash
python -m harness.run --config configs/ar_pair_sanity.json --input bank/sanity/sanity_set.jsonl --validate-only
python -m harness.run --config configs/ar_pair_sanity.json --input bank/sanity/sanity_set.jsonl --output-dir results/ar-sanity-bf16-sdpa --device cuda:0 --dtype bf16 --no-load-in-4bit
python -m harness.evaluate --sanity --bank bank/sanity/sanity_set.jsonl --labels bank/sanity/sanity_set.jsonl --run-dir results/ar-sanity-bf16-sdpa --output-dir results/ar-sanity-analysis-bf16-sdpa
python -m harness.run --config configs/ar_pair_sanity.json --input bank/sanity/sanity_set.jsonl --output-dir results/ar-sanity-nf4-fp32 --device cuda:0 --dtype fp32 --load-in-4bit
python -m harness.evaluate --sanity --bank bank/sanity/sanity_set.jsonl --labels bank/sanity/sanity_set.jsonl --run-dir results/ar-sanity-nf4-fp32 --output-dir results/ar-sanity-analysis
```

`metrics.json` also records AUROC per domain (`auroc_by_domain`; `auroc_math`/`auroc_code` in `summary.csv`). The verified SDPA run is `results/bf16/sdpa/ar-sanity-bf16-sdpa-20261026-000459` with its analysis next to it. The slide package in `results/ppt_ar_sanity/` (summary, CSV/JSON, PNG and SVG figures) is generated from that analysis, never typed by hand:

```bash
python -m harness.sanity_package --run-dir results/bf16/sdpa/ar-sanity-bf16-sdpa-20261026-000459 --analysis-dir results/bf16/sdpa/ar-sanity-analysis-bf16-sdpa-20261026-000459 --previous-analysis results/bf16/ar-sanity-analysis-bf16-20261020-015535 --output-dir results/ppt_ar_sanity
```

The generation manifest records Phi-4-mini, temperature 0.7, top-p 0.95, 4 samples/question, 1024 new tokens, seed 544, vLLM and BF16. `generations_provenance.json` explicitly reconstructs the model revision, versions and L4 hardware afterwards; these are not contemporaneous run metadata. `bank.label/2+math_verify` compares correctness outcomes, not extracted answers. Label 1 means correct; 0 means wrong. Keep selected development questions excluded from pilot/final banks via `bank/dev/question_ids.txt`.

The dev adapter maps `candidate_id → id`, `problem → problem`, and full `candidate → candidate_answer`; it retains `problem_id` and `pair_id`. (`question` is only in the questions file, not the current paired bank.) It checks text hashes and pair structure. Only problem/solution text enters prompts. Labels are joined only by the evaluator. Offline validation does **not** check tokenizer lengths:

```bash
python -m harness.run --config configs/ar_pair_dev.json --input bank/dev/dev_bank.jsonl --validate-only
```

Colab: retain its working CUDA PyTorch; install only scoring dependencies in the AR environment (do not combine with vLLM/bank or diffusion environments). In a Colab Python cell, prefix shell commands with `!`.

```bash
python -m pip install -r requirements.txt bitsandbytes
python -m harness.run --config configs/ar_pair_dev.json --input bank/dev/dev_bank.jsonl --output-dir results/ar-dev-smoke-nf4-fp32 --limit 4 --device cuda:0 --dtype fp32 --load-in-4bit
python -m harness.run --config configs/ar_pair_dev.json --input bank/dev/dev_bank.jsonl --output-dir results/ar-dev-nf4-fp32 --device cuda:0 --dtype fp32 --load-in-4bit
```

NF4+FP32 is development only (12 synthetic cases passed per the team's prior smoke run, not research evidence). NF4 requires explicit FP32; NF4+FP16 is forbidden. The supplied config defaults to unquantized BF16. Exact formatted lengths are checked for the **whole bank with both tokenizers before weights**, even with `--limit`: 1760-token cap, 288 reserved within 2048, no truncation/filtering. The bank README's approximate length argument is not a substitute; bank owners must repair oversized pairs consistently. Both models score sequentially at batch size 1. No offload, retries, or precision fallback.

Offline analysis uses separate dependencies and requires complete scores for the exact bank bytes. A four-row smoke run cannot evaluate a larger bank. Use a new output directory each time:

```bash
python -m pip install -r requirements-analysis.txt
python -m harness.evaluate --bank bank/dev/dev_bank.jsonl --labels bank/dev/dev_labels.jsonl --run-dir results/ar-dev-nf4-fp32 --output-dir results/ar-dev-analysis
# Optional separate exploratory fixed-zero diagnostics; no fitting or saved-score changes:
python -m harness.evaluate --bank bank/dev/dev_bank.jsonl --labels bank/dev/dev_labels.jsonl --run-dir results/ar-dev-nf4-fp32 --output-dir results/ar-dev-zero-analysis --diagnostic-threshold-zero
```

CARC: no launcher exists in this checkout. Inside an allocated BF16-capable GPU job with CUDA PyTorch and `requirements.txt` installed:

```bash
python -m harness.run --config configs/ar_pair_dev.json --input bank/dev/dev_bank.jsonl --output-dir results/ar-dev-carc-bf16 --device cuda:0 --dtype bf16 --no-load-in-4bit
```

Model/local paths and revisions are configured under `models`; pin both model/tokenizer revisions for research. Honor normal Hugging Face authentication and `HF_HOME`. Both thresholds must be explicitly provided via config or `--threshold-a` plus `--threshold-b`; otherwise saved decisions stay null. Diagnostic zero thresholds are exploratory, never matched-TAR or validation-selected. Review categories use diagnostic thresholds when requested, otherwise the run's explicit thresholds if any.

Share `manifest.json`, `scores_A.jsonl`, `scores_B.jsonl`, `pairs.jsonl`, and analysis `metrics.json`, `summary.csv`, `review.csv`, `development.png`. Missing/failed scores produce coverage-only `metrics.json` and a nonzero exit, not subset metrics. Runtime excludes only explicitly cold forward calls; loading and peak allocated memory are separate. Figures show precision/quantization. Development evidence is not a test result, H1 test, strongest-pair selection, or proof of ensemble benefit. Final-answer correctness does not establish correct reasoning; Yes/No probabilities are uncalibrated token events. Real-bank tokenizer/GPU checks and figure rendering remain to be performed.


## LLaDA comparison using the existing AR results

This runs only LLaDA-8B-Instruct. It reuses the saved unquantized BF16 Qwen/Mistral
scores in `results/bf16/ar-sanity-bf16-20261020-015535/`, scoring the exact same 50
candidates in `bank/sanity/sanity_set.jsonl` (checked by hash and candidate ID).

After uploading the whole folder, run inside your CARC interactive A100 allocation:

```bash
bash run_llada_carc.sh
```

Or submit from the uploaded repository root on the login node:

```bash
sbatch --account=YOUR_CARC_PROJECT run_llada_carc.sh
```

The launcher creates a separate environment in `/scratch1/$USER/llada-comparison/`
with Transformers 4.38.2, CUDA PyTorch, SentencePiece and protobuf. It leaves the AR
environment unchanged. Conda packages, model weights, pip caches, remote model
code, temporary files and outputs use scratch to avoid home-directory quota errors.
Initial setup/downloads require network access. The official LLaDA implementation
is loaded with `trust_remote_code=True`, pinned to the resolved model commit.

The shared T1 wording is rendered with LLaDA's own chat template. A single
bidirectional forward pass reads the first masked answer position (mask ID 126336),
with no Dream-style shift. Slots 1, 8 and 32 are compared on the existing 142-row
development bank, without using its labels; highest median Yes+No mass wins, ties
favor fewer slots. LLaDA then scores the 50 original AR candidates. Twenty repeat
reference forwards check top tokens and an independent probability reduction.
Full project gates, including batch invariance, remain pending.

The launcher prints its output directory, normally
`/scratch1/$USER/llada-comparison/run-TIMESTAMP/`. Outputs are:

- `scores_llada.jsonl`: 50 label-free LLaDA score rows.
- `llada_manifest.json`: revision, settings, hashes, environment and checks.
- `dev_slots1.jsonl`, `dev_slots8.jsonl`, `dev_slots32.jsonl`: slot-selection scores.
- `reference_checks.jsonl`: 20 reference/repeat comparisons.
- `single_metrics.csv`: AUROC for all three models, overall and by domain.
- `pair_metrics.csv`: Qwen+Mistral, Qwen+LLaDA and Mistral+LLaDA AND-rule counts.
- `comparison_review.csv`: aligned candidates, labels and scores for all three.
- `comparison_metrics.json`, `REPORT.md`: the comparison and its limitations.

Pair metrics use a fixed zero threshold for exploratory integration checks. They
are not matched-TAR results, do not establish H1, and do not select thresholds on
the sanity fixture. The existing AR models fail the overall 0.90 sanity target.
Saved AR runs and new LLaDA runs use different environments, so there is no matched
cost claim. The later pilot/validation gates and matched-TAR analysis are still
needed for the scientific comparison. No LLaDA results exist until the GPU job runs.

Offline validation and tests (optional CPU Torch for the numerical adapter test):

```bash
python -B -m harness.llada_compare --validate-only
python -B -m unittest harness.test_llada_compare -v
```

To regenerate tables after downloading a completed run:

```bash
python -B -m harness.llada_compare --analyze-only --output-dir results/YOUR_DOWNLOADED_RUN
```

Official references: [LLaDA repository](https://github.com/ML-GSAI/LLaDA),
[masked forward/generation](https://github.com/ML-GSAI/LLaDA/blob/main/generate.py).
