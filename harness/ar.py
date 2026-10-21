"""Shared Transformers implementation. Heavy dependencies are execution-only imports."""

import gc
import math
import time

from .prompts import encode_prompt


def token_sets(tokenizer):
    result = {}
    special = set(tokenizer.all_special_ids)
    for name, variants in (("yes", ("Yes", "yes", "YES")), ("no", ("No", "no", "NO"))):
        accepted = {}
        for word in variants:
            for spelling in (word, " " + word):
                ids = tokenizer.encode(spelling, add_special_tokens=False)
                if len(ids) == 1 and ids[0] not in special:
                    accepted[spelling] = ids[0]
        result[name] = {"spellings": accepted, "ids": sorted(set(accepted.values()))}
        if not result[name]["ids"]:
            raise ValueError(f"empty {name} single-token set")
    if set(result["yes"]["ids"]) & set(result["no"]["ids"]):
        raise ValueError("Yes and No token sets overlap")
    return result


def aggregate_events(yes_log_probs, no_log_probs, inverted=False):
    """Stable logsumexp of selected float32 full-vocabulary log_softmax values.

    The tiny set reductions use Python floats; no Yes/No renormalization occurs.
    Returned probabilities describe token events, not calibrated correctness.
    """
    def logsumexp(values):
        if not values or not all(math.isfinite(x) for x in values):
            raise ValueError("empty or nonfinite token log probabilities")
        maximum = max(values)
        return maximum + math.log(math.fsum(math.exp(x - maximum) for x in values))

    log_yes, log_no = logsumexp(yes_log_probs), logsumexp(no_log_probs)
    p_yes, p_no = math.exp(log_yes), math.exp(log_no)
    score = log_yes - log_no
    result = {"score": -score if inverted else score, "log_p_yes": log_yes,
              "log_p_no": log_no, "p_yes": p_yes, "p_no": p_no,
              "yes_no_mass": p_yes + p_no}
    if not all(math.isfinite(x) for x in result.values()):
        raise ValueError("nonfinite score/probability")
    return result


def runtime_settings(config):
    if config["load_in_4bit"] and config["dtype"] != "fp32":
        raise ValueError("NF4 development runs require explicit FP32 compute")
    import torch

    device = torch.device(config["device"])
    if device.type == "cuda":
        if not torch.cuda.is_available():
            raise ValueError("CUDA requested but unavailable; no CPU fallback")
        torch.cuda.set_device(device)
        device = torch.device("cuda", torch.cuda.current_device())
        bf16 = torch.cuda.is_bf16_supported(including_emulation=False)
    else:
        bf16 = False
    name = config["dtype"]
    if name == "auto":
        name = ("bf16" if bf16 else "fp16") if device.type == "cuda" else "fp32"
    if name == "bf16" and not bf16:
        raise ValueError("bf16 requested without native CUDA bf16 support")
    if device.type == "cpu" and (name != "fp32" or config["load_in_4bit"]):
        raise ValueError("explicit CPU execution requires fp32 without quantization")
    dtype = {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}[name]
    return device, dtype


def forward_record(model, ids, sets, device, inverted, timing):
    import torch

    with torch.inference_mode():
        input_ids = torch.tensor([ids], dtype=torch.long, device=device)
        attention_mask = torch.ones_like(input_ids)
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        start = time.perf_counter()
        output = model(input_ids=input_ids, attention_mask=attention_mask, use_cache=False)
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        timing["forward_seconds"] = time.perf_counter() - start
        log_probs = torch.log_softmax(output.logits[0, -1, :].float(), dim=-1)
        if not torch.isfinite(log_probs).all().item():
            raise ValueError("nonfinite full-vocabulary log probabilities")
        result = aggregate_events(log_probs[sets["yes"]["ids"]].tolist(),
                                  log_probs[sets["no"]["ids"]].tolist(), inverted)
    return result


def run_model(spec, tokenizer, records, config, device, dtype, metadata, emit):
    """Stop this model on its first failure, release weights, and never retry."""
    import torch
    from transformers import AutoConfig, AutoModelForCausalLM

    model = None
    is_cuda = device.type == "cuda"
    if is_cuda:
        torch.cuda.reset_peak_memory_stats(device)
    start = time.perf_counter()
    try:
        model_config = AutoConfig.from_pretrained(
            spec["model_id"], revision=spec["revision"], trust_remote_code=False)
        metadata["resolved_model_revision"] = getattr(model_config, "_commit_hash", None)
        if getattr(model_config, "quantization_config", None):
            raise ValueError("prequantized checkpoints are unsupported; use explicit NF4 mode")
        kwargs = {}
        if config["load_in_4bit"]:
            import bitsandbytes  # noqa: F401 -- optional, only in explicitly selected mode
            from transformers import BitsAndBytesConfig

            quantization = BitsAndBytesConfig(
                load_in_4bit=True, bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=dtype, bnb_4bit_use_double_quant=False,
            )
            kwargs["quantization_config"] = quantization
            metadata["quantization"] = quantization.to_dict()
        model = AutoModelForCausalLM.from_pretrained(
            spec["model_id"], revision=metadata["resolved_model_revision"] or spec["revision"],
            config=model_config, torch_dtype=dtype, device_map={"": str(device)},
            low_cpu_mem_usage=True, trust_remote_code=False, attn_implementation="sdpa", **kwargs,
        )
        model.eval()
        if is_cuda:
            torch.cuda.synchronize(device)
        metadata["loading_seconds"] = time.perf_counter() - start
        metadata["actual_dtype"] = str(model.dtype)
        metadata["actual_quantization"] = "nf4" if getattr(model, "is_loaded_in_4bit", False) else None
        if bool(metadata["actual_quantization"]) != config["load_in_4bit"]:
            raise ValueError("loaded quantization does not match the requested setting")
        metadata["parameter_dtypes"] = sorted({str(p.dtype) for p in model.parameters()})
        if any(p.device != device for p in model.parameters()):
            raise ValueError("unexpected parameter placement; offloading is forbidden")
        for index, record in enumerate(records):
            metadata["active_candidate_id"] = record["id"]
            ids = encode_prompt(tokenizer, record, config["template"])
            if len(ids) > config["max_prompt_tokens"]:
                raise ValueError(f"{record['id']!r}: {len(ids)} tokens exceeds prompt budget")
            timing = {"forward_seconds": None, "input_tokens": len(ids), "cold": index == 0}
            try:
                result = forward_record(model, ids, metadata["token_sets"], device,
                                        config["template"] == "T3", timing)
            except Exception as exc:
                emit({"id": record["id"], "status": "failed", "score": None,
                      "p_yes": None, "p_no": None, "yes_no_mass": None, **timing,
                      "error": f"{type(exc).__name__}: {exc}"})
                raise
            emit({"id": record["id"], "status": "completed", **timing, **result})
        metadata.pop("active_candidate_id", None)
    finally:
        if metadata.get("loading_seconds") is None:
            metadata["loading_seconds"] = time.perf_counter() - start
        metadata["peak_gpu_allocated_bytes"] = torch.cuda.max_memory_allocated(device) if is_cuda else None
        del model
        gc.collect()
        if is_cuda:
            torch.cuda.empty_cache()
