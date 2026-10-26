"""CLI orchestration; --help and --validate-only need only the standard library."""

import argparse
from datetime import datetime, timezone
import gc
import hashlib
from importlib.metadata import PackageNotFoundError, version
import json
import math
from pathlib import Path
import platform
import re
import sys

from .data import BANK_MAPPING, SANITY_MAPPING, load_records, parse_json, validate_mapping
from .prompts import TEMPLATES, encode_prompt, messages


def digest(value):
    return hashlib.sha256(value).hexdigest()


def json_bytes(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True, allow_nan=False).encode("utf-8")


def finite_number(value):
    return type(value) in (int, float) and math.isfinite(value)


# SDPA by default: eager attention gave wrong BF16 scores for Qwen2.5. Gemma-2 needs eager, because
# SDPA silently skips its attention-logit soft-capping (ar.py refuses that combination).
ATTENTION = ("sdpa", "eager")


def attention_of(spec):
    return spec.get("attn_implementation", "sdpa")


def attention_summary(config):
    """One value when both models use the same attention, else "per_model" (see models.X)."""
    values = {attention_of(spec) for spec in config["models"].values()}
    return values.pop() if len(values) == 1 else "per_model"


def validate_config(config):
    keys = {"models", "field_mapping", "template", "max_prompt_tokens", "device",
            "dtype", "load_in_4bit", "thresholds"}
    if not isinstance(config, dict) or not keys <= config.keys() or config.keys() - keys - {"input_format"}:
        raise ValueError(f"config requires {', '.join(sorted(keys))}; input_format is optional")
    validate_mapping(config["field_mapping"])
    if config.get("input_format", "mapped") not in ("mapped", "dev_bank", "sanity"):
        raise ValueError("input_format must be mapped, dev_bank, or sanity")
    if config.get("input_format") == "dev_bank" and config["field_mapping"] != BANK_MAPPING:
        raise ValueError("dev_bank requires the exact make_pairs.py field mapping")
    if config.get("input_format") == "sanity":
        if config["field_mapping"] != SANITY_MAPPING or config["template"] != "T1":
            raise ValueError("sanity requires the official fixture mapping and shared T1")
    if not isinstance(config["template"], str) or config["template"] not in TEMPLATES:
        raise ValueError("template must be T1, T2, or T3")
    maximum = config["max_prompt_tokens"]
    if type(maximum) is not int or not 1 <= maximum <= 2048 - 288:
        raise ValueError("max_prompt_tokens must be 1..1760 (2048 budget, 288 reserved)")
    if config["dtype"] not in ("auto", "bf16", "fp16", "fp32"):
        raise ValueError("dtype must be auto, bf16, fp16, or fp32")
    if not isinstance(config["device"], str) or not re.fullmatch(r"cpu|cuda(?::\d+)?", config["device"]):
        raise ValueError("device must be cpu, cuda, or cuda:N")
    if type(config["load_in_4bit"]) is not bool:
        raise ValueError("load_in_4bit must be boolean")
    if config["load_in_4bit"] and config["dtype"] != "fp32":
        raise ValueError("NF4 development mode requires explicit --dtype fp32; no FP16 fallback")
    if config["device"] == "cpu" and (config["load_in_4bit"] or config["dtype"] not in ("auto", "fp32")):
        raise ValueError("CPU requires auto/fp32 and no 4-bit quantization")
    models = config["models"]
    if not isinstance(models, dict) or set(models) != {"A", "B"}:
        raise ValueError("models must contain exactly A and B")
    for name, spec in models.items():
        if (not isinstance(spec, dict) or not {"model_id", "revision"} <= spec.keys()
                or spec.keys() - {"model_id", "revision", "tokenizer_revision", "attn_implementation"}):
            raise ValueError(f"model {name} requires model_id/revision; tokenizer_revision and attn_implementation are optional")
        if any(not isinstance(v, str) or not v.strip() for v in spec.values()):
            raise ValueError(f"model {name} IDs/paths and revisions must be nonempty strings")
        if spec.get("attn_implementation", "sdpa") not in ATTENTION:
            raise ValueError(f"model {name}: attn_implementation must be one of {', '.join(ATTENTION)}")
    thresholds = config["thresholds"]
    if thresholds is not None and (not isinstance(thresholds, dict)
            or set(thresholds) != {"A", "B"}
            or not all(finite_number(v) for v in thresholds.values())):
        raise ValueError("thresholds must be null or an object with finite A and B values")


def pair_decision(a, b, thresholds):
    if any(row.get("status") != "completed" or not finite_number(row.get("score")) for row in (a, b)):
        return None, "unavailable"
    if thresholds is None:
        return None, "unthresholded"
    accepted = a["score"] >= thresholds["A"] and b["score"] >= thresholds["B"]
    return accepted, "accept" if accepted else "reject"


def write_line(handle, value):
    handle.write(json_bytes(value).decode("utf-8") + "\n")
    handle.flush()


def write_manifest(directory, manifest):
    manifest["updated_at_utc"] = datetime.now(timezone.utc).isoformat()
    temporary = directory / "manifest.json.tmp"
    temporary.write_bytes(json_bytes(manifest) + b"\n")
    temporary.replace(directory / "manifest.json")


def join_scores(directory, records, thresholds):
    expected = {record["id"] for record in records}
    scores, counts = {}, {}
    for name in ("A", "B"):
        scores[name] = {}
        path = directory / f"scores_{name}.jsonl"
        if path.exists():
            with path.open(encoding="utf-8") as handle:
                for line in handle:
                    row = parse_json(line)
                    candidate_id = row["id"]
                    if candidate_id not in expected or candidate_id in scores[name]:
                        raise ValueError(f"unexpected or duplicate ID in {path.name}: {candidate_id!r}")
                    scores[name][candidate_id] = row
        counts[name] = {
            "completed": sum(row["status"] == "completed" for row in scores[name].values()),
            "failed": sum(row["status"] == "failed" for row in scores[name].values()),
            "missing": len(expected) - len(scores[name]),
        }
    pair_counts = {key: 0 for key in ("accept", "reject", "unthresholded", "unavailable")}
    with (directory / "pairs.jsonl").open("x", encoding="utf-8") as handle:
        for record in records:
            candidate_id = record["id"]
            missing = {"id": candidate_id, "status": "missing", "score": None,
                       "p_yes": None, "p_no": None, "yes_no_mass": None,
                       "forward_seconds": None, "input_tokens": None, "cold": None}
            a, b = (scores[name].get(candidate_id, missing) for name in ("A", "B"))
            decision, status = pair_decision(a, b, thresholds)
            pair_counts[status] += 1
            row = {"id": candidate_id, "A": a, "B": b, "decision": decision, "status": status}
            for key in ("problem_id", "pair_id", "domain"):
                if key in record:
                    row[key] = record[key]
            write_line(handle, row)
    return counts, pair_counts


def execute(config, records, all_records, input_hash, config_raw, args):
    directory = Path(args.output_dir)
    directory.mkdir(parents=True, exist_ok=False)
    manifest = {
        "score_contract_version": 2, "command": sys.argv,
        "input_kind": config.get("input_format", "mapped"),
        "status": "running", "started_at_utc": datetime.now(timezone.utc).isoformat(),
        "input_sha256": input_hash, "config_file_sha256": digest(config_raw),
        "effective_config_sha256": digest(json_bytes(config)), "effective_config": config,
        "input_path": str(Path(args.input).resolve()), "limit": args.limit,
        "validated_records": len(all_records), "selected_records": len(records),
        "template": config["template"], "template_text": TEMPLATES[config["template"]],
        "template_sha256": digest(TEMPLATES[config["template"]].encode("utf-8")),
        "prompt_sha256": digest(json_bytes([messages(r, config["template"]) for r in records])),
        "wording_status": "provisional", "pair_role": "initial cross-family baseline; not a proven best pair",
        "probability_semantics": "raw token events; not calibrated correctness probabilities",
        "run_kind": "development_4bit" if config["load_in_4bit"] else "unquantized_baseline",
        "planned_research_precision": "bf16", "batch_size": 1, "use_cache": False,
        "shared_token_budget": 2048, "reserved_tokens": 288, "attention_implementation": attention_summary(config),
        "python": platform.python_version(), "platform": platform.platform(), "packages": {},
        "source_sha256": {p.name: digest(p.read_bytes()) for p in Path(__file__).parent.glob("*.py")},
        "models": {name: {"requested": spec, "quantization": None, "actual_dtype": None,
                           "loading_seconds": None, "peak_gpu_allocated_bytes": None}
                   for name, spec in config["models"].items()},
    }
    for package in ("torch", "transformers", "accelerate", "tokenizers", "huggingface-hub", "sentencepiece", "bitsandbytes"):
        try:
            manifest["packages"][package] = version(package)
        except PackageNotFoundError:
            manifest["packages"][package] = None
    write_manifest(directory, manifest)
    try:
        if manifest["packages"]["transformers"] != "4.51.3":
            raise ValueError("execution requires transformers==4.51.3")
        import torch
        from transformers import AutoConfig, AutoTokenizer
        from .ar import run_model, runtime_settings, token_sets

        device, dtype = runtime_settings(config)
        manifest.update(device=str(device), compute_dtype=str(dtype), cuda_version=torch.version.cuda)
        manifest["gpu"] = ({"name": torch.cuda.get_device_name(device),
                            "capability": list(torch.cuda.get_device_capability(device)),
                            "total_memory_bytes": torch.cuda.get_device_properties(device).total_memory}
                           if device.type == "cuda" else None)
        tokenizers = {}
        # Check the entire bank with both tokenizers, even for --limit smoke runs.
        for name, spec in config["models"].items():
            tokenizer_revision = spec.get("tokenizer_revision", spec["revision"])
            tokenizer_config = AutoConfig.from_pretrained(
                spec["model_id"], revision=tokenizer_revision, trust_remote_code=False)
            resolved_revision = getattr(tokenizer_config, "_commit_hash", None)
            tokenizer = AutoTokenizer.from_pretrained(
                spec["model_id"], revision=resolved_revision or tokenizer_revision,
                config=tokenizer_config, trust_remote_code=False)
            tokenizers[name] = tokenizer
            meta = manifest["models"][name]
            meta.update(tokenizer_revision=tokenizer_revision,
                        resolved_tokenizer_revision=resolved_revision,
                        tokenizer_class=type(tokenizer).__name__,
                        chat_template=tokenizer.get_chat_template(), token_sets=token_sets(tokenizer))
            meta["tokenizer_sha256"] = digest(tokenizer.backend_tokenizer.to_str().encode("utf-8")) if tokenizer.is_fast else digest(json_bytes(tokenizer.get_vocab()))
            prompt_hash = hashlib.sha256()
            maximum = 0
            for record in all_records:
                ids = encode_prompt(tokenizer, record, config["template"])
                if not ids or len(ids) > config["max_prompt_tokens"]:
                    raise ValueError(f"model {name}, candidate {record['id']!r}: {len(ids)} formatted tokens; "
                                     f"allowed 1..{config['max_prompt_tokens']}; no truncation")
                maximum = max(maximum, len(ids))
                prompt_hash.update(json_bytes({"id": record["id"], "tokens": ids}) + b"\n")
            meta.update(preflight_max_tokens=maximum, formatted_bank_sha256=prompt_hash.hexdigest())
            write_manifest(directory, manifest)
        for name in ("A", "B"):
            meta = manifest["models"][name]
            written = set()
            by_id = {record["id"]: record for record in records}
            with (directory / f"scores_{name}.jsonl").open("x", encoding="utf-8") as handle:
                def emit(row):
                    record = by_id[row["id"]]
                    row.update(candidate_id=row["id"], bank_sha256=input_hash, model=name,
                               model_id=config["models"][name]["model_id"],
                               model_revision=meta.get("resolved_model_revision"),
                               template=config["template"])
                    row.update({key: record[key] for key in ("problem_id", "pair_id", "domain") if key in record})
                    write_line(handle, row)
                    written.add(row["id"])

                try:
                    run_model(config["models"][name], tokenizers[name], records, config,
                              device, dtype, meta, emit)
                    meta["status"] = "completed"
                except Exception as exc:
                    meta.update(status="failed", error=f"{type(exc).__name__}: {exc}")
                    for record in records:
                        if record["id"] not in written:
                            emit({"id": record["id"], "status": "failed", "score": None,
                                  "p_yes": None, "p_no": None, "yes_no_mass": None,
                                  "forward_seconds": None, "input_tokens": None, "cold": None,
                                  "error": meta["error"], "reason": "model run aborted; no retry"})
            # Exception tracebacks can retain tensors until the except block exits.
            gc.collect()
            if device.type == "cuda":
                torch.cuda.empty_cache()
            write_manifest(directory, manifest)
        manifest["status"] = "completed" if all(m["status"] == "completed" for m in manifest["models"].values()) else "failed"
    except Exception as exc:
        manifest.update(status="failed", error=f"{type(exc).__name__}: {exc}")
    finally:
        counts, pair_counts = join_scores(directory, records, config["thresholds"])
        for name, count in counts.items():
            manifest["models"][name]["counts"] = count
        manifest["pair_counts"] = pair_counts
        manifest["outputs_sha256"] = {p.name: digest(p.read_bytes()) for p in directory.glob("*.jsonl")}
        manifest["finished_at_utc"] = datetime.now(timezone.utc).isoformat()
        write_manifest(directory, manifest)
    print(f"Run {manifest['status']}: {directory}")
    for error in [manifest.get("error"), *(m.get("error") for m in manifest["models"].values())]:
        if error:
            print(error, file=sys.stderr)
    return 0 if manifest["status"] == "completed" else 1


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--input", required=True)
    parser.add_argument("--output-dir", help="new directory; required except with --validate-only")
    parser.add_argument("--validate-only", action="store_true", help="offline schema checks only; no tokenizer/model imports")
    parser.add_argument("--limit", type=int, help="score first N records; validate/preflight the entire bank")
    parser.add_argument("--device", help="cuda, cuda:N, or explicit cpu")
    parser.add_argument("--dtype", choices=("auto", "bf16", "fp16", "fp32"))
    parser.add_argument("--load-in-4bit", action=argparse.BooleanOptionalAction, default=None,
                        help="explicit NF4 development mode (requires bitsandbytes)")
    parser.add_argument("--threshold-a", type=float)
    parser.add_argument("--threshold-b", type=float)
    args = parser.parse_args(argv)
    try:
        if args.limit is not None and args.limit <= 0:
            raise ValueError("--limit must be positive")
        if not args.validate_only and not args.output_dir:
            raise ValueError("--output-dir is required for execution")
        raw = Path(args.config).read_bytes()
        config = parse_json(raw.decode("utf-8"))
        validate_config(config)
        for key in ("device", "dtype", "load_in_4bit"):
            if getattr(args, key) is not None:
                config[key] = getattr(args, key)
        if (args.threshold_a is None) != (args.threshold_b is None):
            raise ValueError("--threshold-a and --threshold-b must be supplied together")
        if args.threshold_a is not None:
            config["thresholds"] = {"A": args.threshold_a, "B": args.threshold_b}
        validate_config(config)
        records, input_hash = load_records(args.input, config["field_mapping"], config.get("input_format", "mapped"))
        if args.validate_only:
            print(f"Validated {len(records)} records and config offline. Token lengths/model compatibility remain unchecked.")
            return 0
        return execute(config, records[:args.limit], records, input_hash, raw, args)
    except (ValueError, OSError) as exc:
        parser.exit(2, f"error: {exc}\n")


if __name__ == "__main__":
    raise SystemExit(main())
