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
