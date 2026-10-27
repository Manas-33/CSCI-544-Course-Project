# Pilot scores (October 27, 2026)

The 500-answer pilot bank (`bank/pilot/pilot_bank.jsonl`, sha256 `aa4d8cc3…`; 125 math and 125 code
matched pairs) scored with T1, T2 and T3 by notebooks 06–08. Every run scored all 500 answers.

| Folder | Verifiers | GPU | Notes |
|---|---|---|---|
| `ar/pilot_qwen_mistral_T*` | Qwen2.5-7B, Mistral-7B | A100 40 GB | BF16, SDPA, batch 1 |
| `ar/pilot_llama_gemma_T*` | Llama-3.1-8B, Gemma-2-9B | A100 40 GB | Gemma with eager attention |
| `ar/pilot_small_T*` | Llama-3.2-3B, Qwen2.5-1.5B | A100 40 GB | gate A's small-AR reference |
| `llada/` | LLaDA-8B-Instruct | A100 40 GB | 1 answer slot, unbatched, transformers 4.38.2 |
| `dream/` | Dream-v0-Instruct-7B | **L4** | 1 answer slot, unbatched; seconds per answer not comparable with the A100 runs |

T3 asks whether the answer is *incorrect*: AR scores are already sign-flipped by the harness;
diffusion T3 scores are flipped by `analysis.pilot`.

`analysis/` is produced by:

    python -m analysis.pilot --bank bank/pilot/pilot_bank.jsonl --labels bank/pilot/pilot_labels.jsonl \
        --scores results/pilot/ar results/pilot/llada results/pilot/dream --out results/pilot/analysis

`gates.md` is the summary. Pair FARs use thresholds set on the pilot itself (in-sample), so they
are exploratory, not the H1 test.
