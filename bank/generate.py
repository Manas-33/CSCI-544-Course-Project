"""Step 2: generate candidate answers with a generator outside the verifier families (GPU; run on Colab).

Defaults follow PLAN.md §2: Phi-4-mini-instruct, temperature 0.7, top-p 0.95,
4 answers per question, at most 1,024 new tokens. Each question gets its own seed
(base seed + question position), so a rerun reproduces the same answers.

    python -m bank.generate --questions bank/dev/questions.jsonl --out bank/dev/generations.jsonl
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

from bank.common import read_jsonl, sha256_text, write_jsonl

DEFAULT_GENERATOR = "microsoft/Phi-4-mini-instruct"
PROMPT = (
    "Solve the following math problem step by step. "
    "Put your final answer in \\boxed{{}}.\n\n{question}"
)


def build_messages(question: str) -> list[dict]:
    return [{"role": "user", "content": PROMPT.format(question=question)}]


def resolve_dtype(choice: str) -> str:
    """bfloat16 where the GPU supports it (A100, L4, L40S, A40), else float16 (T4, V100, P100)."""
    if choice != "auto":
        return choice
    import torch

    if torch.cuda.is_available() and torch.cuda.is_bf16_supported():
        return "bfloat16"
    return "float16"


def generate_vllm(questions, args) -> list[dict]:
    from vllm import LLM, SamplingParams

    llm = LLM(
        model=args.model,
        revision=args.revision,
        seed=args.seed,
        dtype=args.dtype,
        max_model_len=args.max_model_len,
    )
    params = [
        SamplingParams(
            n=args.n_samples,
            temperature=args.temperature,
            top_p=args.top_p,
            max_tokens=args.max_new_tokens,
            seed=args.seed + i,
        )
        for i in range(len(questions))
    ]
    outputs = llm.chat([build_messages(q["question"]) for q in questions], params)
    rows = []
    for i, (q, out) in enumerate(zip(questions, outputs)):
        for k, comp in enumerate(out.outputs):
            rows.append(_row(q, k, comp.text, comp.finish_reason, len(comp.token_ids), args, args.seed + i))
    return rows


def generate_hf(questions, args) -> list[dict]:
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tok = AutoTokenizer.from_pretrained(args.model, revision=args.revision)
    model = AutoModelForCausalLM.from_pretrained(
        args.model, revision=args.revision, torch_dtype=getattr(torch, args.dtype), device_map="auto"
    )
    eos_ids = _as_set(tok.eos_token_id) | _as_set(model.generation_config.eos_token_id)
    rows = []
    for i, q in enumerate(questions):
        torch.manual_seed(args.seed + i)
        ids = tok.apply_chat_template(
            build_messages(q["question"]), add_generation_prompt=True, return_tensors="pt"
        ).to(model.device)
        out = model.generate(
            ids,
            do_sample=True,
            temperature=args.temperature,
            top_p=args.top_p,
            max_new_tokens=args.max_new_tokens,
            num_return_sequences=args.n_samples,
            pad_token_id=tok.eos_token_id,
        )
        for k, seq in enumerate(out):
            new = seq[ids.shape[1]:].tolist()
            n_tok, finish = hf_finish(new, eos_ids, args.max_new_tokens)
            text = tok.decode(new[:n_tok], skip_special_tokens=True)
            rows.append(_row(q, k, text, finish, n_tok, args, args.seed + i))
    return rows


def _as_set(ids) -> set[int]:
    if ids is None:
        return set()
    return set(ids) if isinstance(ids, (list, tuple, set)) else {ids}


def hf_finish(new_ids: list[int], eos_ids: set[int], max_new_tokens: int) -> tuple[int, str]:
    """(answer length, finish reason) for one sampled sequence.

    Sequences in a batch are padded to the longest one, so the answer ends at its first
    end-of-text token. With no end token, it was cut off if it reached max_new_tokens.
    """
    for pos, token in enumerate(new_ids):
        if token in eos_ids:
            return pos, "stop"
    return len(new_ids), "length" if len(new_ids) >= max_new_tokens else "stop"


def _row(q, k, text, finish_reason, n_tokens, args, seed) -> dict:
    return {
        "candidate_id": f"{q['problem_id']}__{args.tag}__{k}",
        "problem_id": q["problem_id"],
        "generator": args.model,
        "generator_revision": args.revision,
        "settings": {
            "temperature": args.temperature,
            "top_p": args.top_p,
            "max_new_tokens": args.max_new_tokens,
            "n_samples": args.n_samples,
            "backend": args.backend,
            "dtype": args.dtype,
        },
        "seed": seed,
        "sample_index": k,
        "text": text,
        "text_sha256": sha256_text(text),
        "finish_reason": finish_reason,
        "n_tokens": n_tokens,
    }


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--questions", default="bank/dev/questions.jsonl")
    ap.add_argument("--out", default="bank/dev/generations.jsonl")
    ap.add_argument("--model", default=DEFAULT_GENERATOR)
    ap.add_argument("--revision", default=None, help="Pin a Hugging Face commit for reproducibility")
    ap.add_argument("--tag", default="phi4mini", help="Short generator name used in candidate IDs")
    ap.add_argument("--n-samples", type=int, default=4)
    ap.add_argument("--temperature", type=float, default=0.7)
    ap.add_argument("--top-p", type=float, default=0.95)
    ap.add_argument("--max-new-tokens", type=int, default=1024)
    ap.add_argument("--seed", type=int, default=544)
    ap.add_argument("--backend", choices=["vllm", "hf"], default="vllm")
    ap.add_argument(
        "--dtype",
        choices=["auto", "bfloat16", "float16"],
        default="auto",
        help="auto: bfloat16 if the GPU supports it, else float16 (T4, V100, P100)",
    )
    ap.add_argument(
        "--max-model-len",
        type=int,
        default=4096,
        help="vLLM context cap. Phi-4-mini declares 131,072 tokens, whose KV cache (~16 GB) "
        "does not fit an L4 or T4; prompt plus answer here is under 1,500 tokens.",
    )
    return ap


def resolve_revision(model: str, revision: str | None) -> str | None:
    """The exact Hugging Face commit SHA to load, so every run is pinned and recorded."""
    try:
        from huggingface_hub import HfApi

        return HfApi().model_info(model, revision=revision).sha
    except Exception as exc:  # offline or hub unavailable: run unpinned, but say so
        print(f"WARNING: could not resolve a commit SHA for {model} ({exc}); the run is unpinned")
        return revision


def environment_info() -> dict:
    """Code revision, library versions and GPU, recorded in the manifest (PLAN.md §9)."""
    import platform
    import subprocess
    from importlib import metadata

    def version(pkg: str) -> str | None:
        try:
            return metadata.version(pkg)
        except metadata.PackageNotFoundError:
            return None

    try:
        code_revision = subprocess.run(
            ["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=True
        ).stdout.strip()
    except Exception:
        code_revision = None
    gpu = None
    try:
        import torch

        if torch.cuda.is_available():
            gpu = torch.cuda.get_device_name(0)
    except Exception:
        pass
    return {
        "code_revision": code_revision,
        "python": platform.python_version(),
        "versions": {p: version(p) for p in ("torch", "transformers", "vllm", "huggingface-hub")},
        "gpu": gpu,
    }


def main() -> None:
    from datetime import datetime, timezone

    args = build_parser().parse_args()
    args.dtype = resolve_dtype(args.dtype)
    args.revision = resolve_revision(args.model, args.revision)
    print(f"Using dtype {args.dtype}, {args.model} at revision {args.revision}")

    questions = read_jsonl(args.questions)
    started = datetime.now(timezone.utc).isoformat(timespec="seconds")
    start = time.time()
    rows = generate_vllm(questions, args) if args.backend == "vllm" else generate_hf(questions, args)
    elapsed = time.time() - start
    write_jsonl(args.out, rows)

    manifest = {
        "step": "generate",
        "questions": args.questions,
        "n_questions": len(questions),
        "n_generations": len(rows),
        "generator": args.model,
        "generator_revision": args.revision,
        "prompt": PROMPT,
        "settings": {k: v for k, v in vars(args).items() if k not in ("questions", "out")},
        "seconds": round(elapsed, 1),
        "started_utc": started,
        "finished_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "environment": environment_info(),
    }
    Path(args.out).with_name("generations_manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8", newline="\n"
    )
    print(f"Wrote {len(rows)} answers for {len(questions)} questions in {elapsed:.0f}s to {args.out}")


if __name__ == "__main__":
    main()
