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


def generate_vllm(questions, args) -> list[dict]:
    from vllm import LLM, SamplingParams

    llm = LLM(model=args.model, revision=args.revision, seed=args.seed, dtype="bfloat16")
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
        args.model, revision=args.revision, torch_dtype=torch.bfloat16, device_map="auto"
    )
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
            new = seq[ids.shape[1]:]
            n_tok = int((new != tok.pad_token_id).sum()) if tok.pad_token_id is not None else len(new)
            finish = "length" if n_tok >= args.max_new_tokens else "stop"
            text = tok.decode(new, skip_special_tokens=True)
            rows.append(_row(q, k, text, finish, n_tok, args, args.seed + i))
    return rows


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
        },
        "seed": seed,
        "sample_index": k,
        "text": text,
        "text_sha256": sha256_text(text),
        "finish_reason": finish_reason,
        "n_tokens": n_tokens,
    }


def main() -> None:
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
    args = ap.parse_args()

    questions = read_jsonl(args.questions)
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
    }
    Path(args.out).with_name("generations_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(f"Wrote {len(rows)} answers for {len(questions)} questions in {elapsed:.0f}s to {args.out}")


if __name__ == "__main__":
    main()
