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

The committed NF4 run (`results/fp32/ar-sanity-nf4-fp32`, a T4) also predates the SDPA switch and used eager attention, so like the eager BF16 run it is not evidence (both models failed there: Qwen 0.854, Mistral 0.720).

Analyses made with the current evaluator also record AUROC per domain (`auroc_by_domain` in `metrics.json`; `auroc_math`/`auroc_code` in `summary.csv`); the older eager and NF4 analyses predate this. The verified SDPA run is `results/bf16/sdpa/ar-sanity-bf16-sdpa-20261026-000459` with its analysis next to it. The slide package in `results/ppt_ar_sanity/` (summary, CSV/JSON, PNG and SVG figures) is generated from that analysis and its run manifest; every number and setting in it is read from them:

```bash
python -m harness.sanity_package --run-dir results/bf16/sdpa/ar-sanity-bf16-sdpa-20261026-000459 --analysis-dir results/bf16/sdpa/ar-sanity-analysis-bf16-sdpa-20261026-000459 --previous-run results/bf16/ar-sanity-bf16-20261020-015535 --previous-analysis results/bf16/ar-sanity-analysis-bf16-20261020-015535 --output-dir results/ppt_ar_sanity
```

**Attention and the reference check.** Each model in a config may set `attn_implementation` (`sdpa`, the default, or `eager`); the manifest records the attention each model actually loaded. Gemma-2 must use `eager`: SDPA skips its attention-logit soft-capping, and `harness/ar.py` refuses that combination. `configs/ar_sanity_llama_gemma.json` and `configs/ar_sanity_small.json` are the sanity configs for Llama-3.1-8B + Gemma-2-9B and Llama-3.2-3B + Qwen2.5-1.5B. `python -m harness.ar_reference --config <config> --input bank/sanity/sanity_set.jsonl --output <report.json>` saves each model's rendered prompt (checking for a doubled BOS) and checks, on 10 math and 10 code items, that our one-pass readout matches the first step of the model's own greedy `generate()` without a KV cache (same top raw logit, score within 0.05). That catches a wrong readout position, not a template or precision bug, so `--fp32-check` also rescores the items with FP32 weights and reports how far the BF16 scores move (the check that would have caught the eager-attention bug). Together they are the evidence for signing off a sanity failure as a weak verifier rather than a bug (PLAN.md §4). The Colab notebooks in `notebooks/` run all of this.

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

Share `manifest.json`, `scores_A.jsonl`, `scores_B.jsonl`, `pairs.jsonl`, and analysis `metrics.json`, `summary.csv`, `review.csv`, `development.png`. Missing/failed scores produce coverage-only `metrics.json` and a nonzero exit, not subset metrics. Runtime excludes only explicitly cold forward calls; loading and peak allocated memory are separate. Figures show precision/quantization. Development evidence is not a test result, H1 test, strongest-pair selection, or proof of ensemble benefit. Final-answer correctness does not establish correct reasoning; Yes/No probabilities are uncalibrated token events. The development bank has not been scored by the AR models yet, so its exact token lengths on the real tokenizers are still unchecked.


## Diffusion verifiers (LLaDA, iLLaDA, Dream)

Scoring goes through **`harness/diffusion/`** for all three models: see its README for the environments, the scoring-code checks and `run_score`. The same T1 wording and the same six Yes and six No spellings are used as for the AR models (a test checks that the T1 text matches `harness/prompts.py`).

**The saved LLaDA run** is `results/llada-comparison-20261019-213107/`: LLaDA-8B-Instruct at revision `08b83a6`, BF16, 1 answer slot (chosen on the development bank by median Yes+No mass, which was 0.978), T1, all 50 sanity items, on an A100. It was scored by an earlier version of `harness/llada_compare.py` (commit `030a84a`), and `llada_manifest.json` records how. That run's "reference check" ran the same forward pass twice; it is a repeat check, not the PLAN.md §4 parity test.

`harness/llada_compare.py` no longer runs any model. It compares the saved LLaDA scores with the BF16 + SDPA Qwen/Mistral run (single and AND-pair metrics at the exploratory zero threshold) and refuses the eager-attention AR run:

```bash
python -m harness.llada_compare --validate-only   # check the saved inputs
python -m harness.llada_compare                    # rewrite the comparison files in the LLaDA run folder
```

Pair metrics at a fixed zero threshold are not matched-TAR results: each pair ends up at a different TAR, so their FARs cannot be compared, and they establish nothing about H1.

**Parity between the two LLaDA scorers.** Before relying on the saved run, score the same 50 items with `harness/diffusion` (unbatched, 1 slot, the same pinned revision) and compare. It passes when every score is within 0.05 of the saved one and the Spearman correlation is above 0.999 (PLAN.md §4 test 3's batched-vs-unbatched and repeat-run bounds). Use the saved run's environment where possible: an A100 with torch 2.6.0, transformers 4.38.2 and tokenizers 0.15.2. Different GPUs or torch builds change BF16 kernels and can move scores slightly without any bug, so record any difference:

```bash
python -m harness.diffusion.run_score --model llada --n-slots 1 --batch-size 1 --items bank/sanity/sanity_set.jsonl --bank-hash f431f8d60448eb9324f1fa6afb9a5d121003da0d2ef1f5e4b52d76813d841ecf --out results/llada_sanity_T1_n1.jsonl
python -m harness.llada_compare --parity results/llada_sanity_T1_n1.jsonl
```

Offline tests: `python -m pytest harness -q` (no models, no GPU).
