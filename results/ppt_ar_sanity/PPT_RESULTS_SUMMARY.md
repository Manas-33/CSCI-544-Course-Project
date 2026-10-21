# AR Verifier Sanity Check ? PPT Results

## Headline
Qwen passes the sanity check after switching to BF16 + SDPA attention (overall AUROC **0.920**). Mistral remains below the required threshold (overall AUROC **0.752**), so the overall verifier sanity gate **fails**.

## Experimental setup
- Official 50-item T1 sanity fixture
- 25 correct and 25 incorrect candidates
- 26 math items and 24 code items
- BF16 precision with SDPA attention
- NVIDIA A100-SXM4-40GB
- Full coverage: 50/50 items completed for both models
- No failed or missing scores

## Results

| Model | Overall AUROC | Math AUROC | Code AUROC | Gate |
|---|---:|---:|---:|---|
| Qwen2.5-7B-Instruct | 0.920 | 0.880 | 1.000 | PASS |
| Mistral-7B-Instruct-v0.3 | 0.752 | 0.600 | 0.940 | FAIL |

## Interpretation
1. The Qwen BF16 + eager-attention failure was a harness/runtime issue: changing to SDPA raised Qwen from the earlier ~0.773 result to 0.920.
2. Qwen clears the required 0.90 sanity AUROC threshold.
3. Mistral is strong on code relative to math but has weak math discrimination (0.600 math AUROC).
4. Because every verifier must pass the sanity gate, the Qwen+Mistral setup is not approved as a research-evidence pair.
5. Test Llama-3.1 and Gemma-2 under the same BF16 + SDPA + T1 protocol before selecting the AR verifier set.

## Caveat
This fixture is a scoring sanity check and is explicitly **not research evidence**. The math/code values above are the reported domain breakdowns; the current evaluator `metrics.json` stores aggregate AUROC and does not currently serialize domain AUROCs.

## Suggested slide takeaway
**?SDPA fixes Qwen (0.773 ? 0.920), but Mistral remains a weak math verifier (0.600 math AUROC); evaluate Llama and Gemma next.?**
