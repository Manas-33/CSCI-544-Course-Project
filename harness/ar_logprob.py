"""Exact log-probability of a candidate given the problem under an AR model (PLAN.md §4).

The AR arm of the reconstruction score: the problem is a user turn in the model's own chat
template, the candidate is the assistant's answer, and the score is the log-probability of
the answer's tokens divided by their number. The prompt and answer tokens, the score rows,
the resume rules and the decision not to score the end-of-turn token are the reconstruction
score's (harness/diffusion/reconstruction.py, run_reconstruction.py); ``method`` is
``ar_exact``. Models, attention (SDPA, eager for Gemma-2), BF16 and batch size 1 are those of
the verdict runs: the model comes from the same config files and is loaded with the same
checks as in harness/ar.py. Labels are never read.

    python -m harness.ar_logprob --config configs/pilot_qwen_mistral.json --which A \\
        --items bank/pilot/pilot_bank.jsonl --out results/reconstruction/pilot/ar_qwen25.jsonl
"""
from __future__ import annotations

import argparse
import math
import sys
import traceback
from pathlib import Path

from .data import parse_json
from .diffusion.reconstruction import EmptyAnswer, encode_pair
from .diffusion.run_reconstruction import backup_row, gpu_name, load_items, run, timed
from .run import attention_of, validate_config

METHOD = "ar_exact"
CHUNK = 128        # answer positions per float32 log-softmax, so Gemma's 256k vocabulary fits a 24 GB GPU
MAX_TOKENS = 2048  # prompt + answer: the shared budget of the verdict runs (harness/run.py)


def answer_logprobs(model, prompt, answer, device) -> list:
    """log p(token | prompt and the answer so far) for every answer token, in one forward pass."""
    import torch

    ids = list(prompt) + list(answer)
    with torch.inference_mode():
        input_ids = torch.tensor([ids], dtype=torch.long, device=device)
        logits = model(input_ids=input_ids, attention_mask=torch.ones_like(input_ids), use_cache=False).logits[0]
        rows = logits[len(prompt) - 1:len(ids) - 1]  # position i predicts token i + 1
        targets = input_ids[0, len(prompt):]
        picked = []
        for part, chosen in zip(rows.split(CHUNK), targets.split(CHUNK)):
            picked += torch.log_softmax(part.float(), dim=-1).gather(1, chosen[:, None])[:, 0].double().tolist()
    if len(picked) != len(answer) or not all(math.isfinite(x) for x in picked):
        raise ValueError("nonfinite or missing token log probabilities")
    return picked


def score_items(model, tokenizer, items, settings, device):
    """Yield one score row per item, in input order."""
    gpu = gpu_name(device)
    for item in items:
        cid = item["candidate_id"]
        lengths = {"gpu": gpu}
        try:
            pair = encode_pair(tokenizer, item["problem"], item["candidate"])
            lengths.update(n_prompt_tokens=len(pair.prompt), n_tokens=len(pair.answer), boundary=pair.boundary)
            if len(pair.prompt) + len(pair.answer) > MAX_TOKENS:
                yield backup_row(settings, cid, "too_long", **lengths)
                continue
            picked, seconds, peak = timed(device, lambda: answer_logprobs(model, pair.prompt, pair.answer, device))
        except EmptyAnswer:
            yield backup_row(settings, cid, "empty", **lengths)
            continue
        except Exception:
            traceback.print_exc(file=sys.stderr)
            yield backup_row(settings, cid, "error", **lengths)
            continue
        loglik = math.fsum(picked)
        yield backup_row(settings, cid, loglik=loglik, score=loglik / len(picked),
                         runtime_s=seconds, peak_mem_gb=peak, **lengths)


def load_model(spec, config, revision, device, dtype):
    """The model as harness/ar.py loads it for verdict scores, with the same checks."""
    from transformers import AutoConfig, AutoModelForCausalLM

    model_config = AutoConfig.from_pretrained(spec["model_id"], revision=revision, trust_remote_code=False)
    if getattr(model_config, "quantization_config", None) or config["load_in_4bit"]:
        raise ValueError("quantized models are unsupported here")
    attention = attention_of(spec)
    if getattr(model_config, "attn_logit_softcapping", None) and attention != "eager":
        raise ValueError("this model soft-caps attention logits (Gemma-2); set attn_implementation to eager")
    model = AutoModelForCausalLM.from_pretrained(
        spec["model_id"], revision=revision, config=model_config, torch_dtype=dtype,
        device_map={"": str(device)}, low_cpu_mem_usage=True, trust_remote_code=False,
        attn_implementation=attention).eval()
    loaded = getattr(model.config, "_attn_implementation", None)
    if loaded != attention:
        raise ValueError(f"loaded {loaded} attention, not the requested {attention}")
    if model.dtype != dtype or any(p.device != device for p in model.parameters()):
        raise ValueError("unexpected precision or parameter placement; offloading is forbidden")
    return model


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--config", required=True, help="a verdict config (configs/pilot_*.json); only its models, "
                                                    "device and dtype are used")
    ap.add_argument("--which", required=True, choices=["A", "B"], help="which of the config's two models")
    ap.add_argument("--items", required=True, help="bank JSONL with candidate_id, problem, candidate")
    ap.add_argument("--out", required=True, help="output JSONL of score rows (appended)")
    ap.add_argument("--domain", default=None, choices=["math", "code"], help="score only this domain")
    ap.add_argument("--subset", default=None, help="score only the candidates drawn by analysis.reconstruction draw")
    ap.add_argument("--bank-hash", default=None, help="refuse the bank file if its sha256 differs")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args(argv)

    config = parse_json(Path(args.config).read_text(encoding="utf-8"))
    validate_config(config)
    if config["dtype"] != "bf16":
        raise SystemExit("the AR arm is scored in BF16, as the verdict runs are")
    config["device"] = args.device
    spec = config["models"][args.which]

    import transformers
    from transformers import AutoConfig, AutoTokenizer
    from .ar import runtime_settings

    if transformers.__version__ != "4.51.3":
        raise SystemExit("execution requires transformers==4.51.3")
    device, dtype = runtime_settings(config)
    items, bank_hash = load_items(args.items, args.domain, args.subset, args.bank_hash)
    revision = getattr(AutoConfig.from_pretrained(spec["model_id"], revision=spec["revision"],
                                                  trust_remote_code=False), "_commit_hash", None) or spec["revision"]
    settings = {"bank_hash": bank_hash, "model": spec["model_id"], "revision": revision, "method": METHOD,
                "n_samples": None, "seed": None, "mc_group": None, "micro_batch": 1,
                "attention": attention_of(spec)}

    def make_rows(todo):
        requested = spec.get("tokenizer_revision", spec["revision"])
        tokenizer = AutoTokenizer.from_pretrained(  # the same commit as the weights unless the config says otherwise
            spec["model_id"], revision=revision if requested == spec["revision"] else requested, trust_remote_code=False)
        model = load_model(spec, config, revision, device, dtype)
        yield from score_items(model, tokenizer, items=todo, settings=settings, device=device)

    return run(args, items, settings, make_rows, {"hf_id": spec["model_id"], "dtype": str(dtype),
                                                  "config": args.config, "which": args.which})


if __name__ == "__main__":
    raise SystemExit(main())
