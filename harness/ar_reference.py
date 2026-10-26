"""Reference checks for the AR scorer: evidence for the PLAN.md §4 sign-off of an AR model.

For each model in a config, on N items (half math, half code when the input has domains):
- the rendered prompt (the model's own chat template) is saved for one example (PLAN.md §9),
  and the encoded prompt must start with at most one BOS token;
- readout: our one-pass score (last-position logits) against the first step of the model's own
  greedy generate() on the same ids, without a KV cache: same top raw logit, and
  log P(Yes) - log P(No) within 0.05 (PLAN.md §4 test 3's batched-vs-unbatched bound; BF16
  kernels for different shapes can differ by a rounding step). This catches a wrong readout
  position; it shares the prompt, token sets and loaded model with the scorer, so it cannot
  catch a template, token-set or precision bug.
- with --fp32-check: the same items scored again with FP32 weights and compute, reported as
  the largest score change, Spearman correlation and verdict agreement at 0. This is the
  check that would have exposed the eager-attention BF16 bug; it is descriptive (no pass bar).

    python -m harness.ar_reference --config configs/ar_pair_sanity.json \\
        --input bank/sanity/sanity_set.jsonl --output results/stage3/ar/reference_qwen_mistral.json --fp32-check

The report is written even when a model fails to load. Exit code 0 only if every model passes.
"""
import argparse
import gc
import json
import math
from pathlib import Path

from .ar import aggregate_events, runtime_settings, token_sets
from .data import load_records
from .prompts import encode_prompt, messages
from .run import attention_of, validate_config

TOLERANCE = 0.05


def balanced(records, n):
    domains = sorted({r.get("domain") for r in records})
    if len(domains) < 2:
        return records[:n]
    per = n // len(domains)
    chosen = [r for d in domains for r in [x for x in records if x.get("domain") == d][:per]]
    return chosen + [r for r in records if r not in chosen][:n - len(chosen)]


def yes_no_score(logits, sets):
    import torch

    lp = torch.log_softmax(logits.float(), dim=-1)
    return aggregate_events(lp[sets["yes"]["ids"]].tolist(), lp[sets["no"]["ids"]].tolist())["score"]


def compare_step(our_logits, reference_logits, sets):
    """Our readout against the reference first-step logits (both 1-D, full vocabulary)."""
    ours, ref = yes_no_score(our_logits, sets), yes_no_score(reference_logits, sets)
    same_top = int(our_logits.argmax()) == int(reference_logits.argmax())
    delta = abs(ours - ref)
    return {"score": ours, "reference_score": ref, "score_delta": delta,
            "top_token": int(our_logits.argmax()), "same_top_raw_logit": same_top,
            "passed": same_top and delta < TOLERANCE and math.isfinite(delta)}


def single_bos(ids, tokenizer):
    bos = tokenizer.bos_token_id
    return bos is None or ids[:2].count(bos) <= 1


def spearman(xs, ys):
    def ranks(values):
        order = sorted(range(len(values)), key=lambda i: values[i])
        result = [0.0] * len(values)
        i = 0
        while i < len(order):
            j = i
            while j + 1 < len(order) and values[order[j + 1]] == values[order[i]]:
                j += 1
            for k in range(i, j + 1):
                result[order[k]] = (i + j) / 2 + 1
            i = j + 1
        return result
    rx, ry = ranks(xs), ranks(ys)
    mx, my = sum(rx) / len(rx), sum(ry) / len(ry)
    spread = math.sqrt(sum((a - mx) ** 2 for a in rx) * sum((b - my) ** 2 for b in ry))
    return sum((a - mx) * (b - my) for a, b in zip(rx, ry)) / spread if spread else float("nan")


def load_model(spec, device, dtype):
    from transformers import AutoModelForCausalLM

    attention = attention_of(spec)
    model = AutoModelForCausalLM.from_pretrained(
        spec["model_id"], revision=spec["revision"], torch_dtype=dtype, device_map={"": str(device)},
        low_cpu_mem_usage=True, attn_implementation=attention).eval()
    if getattr(model.config, "attn_logit_softcapping", None) and attention != "eager":
        raise ValueError("soft-capped model (Gemma-2) with SDPA; set attn_implementation to eager")
    if model.config._attn_implementation != attention:
        raise ValueError(f"loaded {model.config._attn_implementation} attention, not {attention}")
    return model


def free(model):
    import torch

    del model
    gc.collect()
    torch.cuda.empty_cache()


def check_model(spec, records, template, device, dtype, fp32_check):
    import torch
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(spec["model_id"], revision=spec.get("tokenizer_revision", spec["revision"]))
    sets = token_sets(tokenizer)
    prompts = [encode_prompt(tokenizer, r, template) for r in records]
    result = {"model_id": spec["model_id"], "requested_attention": attention_of(spec),
              "rendered_prompt_example": tokenizer.apply_chat_template(
                  messages(records[0], template), tokenize=False, add_generation_prompt=True),
              "single_bos": all(single_bos(ids, tokenizer) for ids in prompts), "token_sets": sets}
    model = load_model(spec, device, dtype)
    result["revision"] = getattr(model.config, "_commit_hash", None)
    result["attention_implementation"] = model.config._attn_implementation
    items = []
    with torch.inference_mode():
        for record, ids in zip(records, prompts):
            x = torch.tensor([ids], device=device)
            ours = model(input_ids=x, attention_mask=torch.ones_like(x), use_cache=False).logits[0, -1]
            gen = model.generate(x, attention_mask=torch.ones_like(x), max_new_tokens=1, do_sample=False,
                                 use_cache=False, output_logits=True, return_dict_in_generate=True,
                                 pad_token_id=tokenizer.pad_token_id if tokenizer.pad_token_id is not None
                                 else tokenizer.eos_token_id)
            items.append({"id": record["id"], **compare_step(ours, gen.logits[0][0], sets)})
    free(model)
    result["items"] = items
    result["passed"] = result["single_bos"] and all(i["passed"] for i in items)
    if fp32_check:
        model = load_model(spec, device, torch.float32)
        with torch.inference_mode():
            fp32 = [yes_no_score(model(input_ids=torch.tensor([ids], device=device), use_cache=False).logits[0, -1], sets)
                    for ids in prompts]
        free(model)
        bf16 = [i["score"] for i in items]
        result["fp32_check"] = {
            "max_abs_score_change": max(abs(a - b) for a, b in zip(bf16, fp32)),
            "spearman": spearman(bf16, fp32),
            "same_verdict_at_zero": sum((a >= 0) == (b >= 0) for a, b in zip(bf16, fp32)), "items": len(fp32),
            "fp32_scores": dict(zip([r["id"] for r in records], fp32)), "note": "descriptive; no pass bar"}
    return result


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--config", required=True)
    p.add_argument("--input", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--n", type=int, default=20)
    p.add_argument("--fp32-check", action="store_true", help="also score the items with FP32 weights (needs ~4 bytes/parameter)")
    args = p.parse_args(argv)
    config = json.loads(Path(args.config).read_text())
    validate_config(config)
    from importlib.metadata import version
    if version("transformers") != "4.51.3":
        raise SystemExit("use the AR environment (requirements.txt): transformers==4.51.3, as harness.run does")
    # validate_config ties the sanity and dev_bank formats to their exact field mappings.
    records, input_hash = load_records(args.input, config["field_mapping"], config.get("input_format", "mapped"))
    records = balanced(records, args.n)
    device, dtype = runtime_settings(config)
    report = {"config": args.config, "input": args.input, "input_sha256": input_hash, "template": config["template"],
              "tolerance": TOLERANCE, "items": [r["id"] for r in records], "models": {}}
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    for name, spec in config["models"].items():
        try:
            report["models"][name] = check_model(spec, records, config["template"], device, dtype, args.fp32_check)
        except Exception as exc:  # keep the other model's result and say what failed
            report["models"][name] = {"model_id": spec["model_id"], "passed": False, "error": f"{type(exc).__name__}: {exc}"}
        m = report["models"][name]
        detail = m.get("error") or f"max score delta {max(i['score_delta'] for i in m['items']):.3f}"
        print(f"{name} {spec['model_id']}: {'PASS' if m['passed'] else 'FAIL'} ({detail})", flush=True)
        out.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    report["passed"] = all(m["passed"] for m in report["models"].values())
    out.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
