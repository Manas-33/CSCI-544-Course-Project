"""Find candidates whose verifier prompt is too long for any verifier (PLAN.md §2 filters).

For every candidate in the given bank files and every verifier tokenizer, the formatted
T1, T2 and T3 prompts are measured in that verifier's chat template. A candidate whose
longest prompt exceeds --max-tokens (1,760: Dream's 2,048-token context minus 32 answer
slots and 256 critique tokens) for any verifier is listed, and bank.sample_pilot drops
its whole matched pair. Tokenizers that fail to load are reported, not silently skipped.

    python -m bank.token_lengths bank/pilot/math/pilot_bank.jsonl bank/pilot/code/pilot_bank.jsonl \\
        --out bank/pilot/too_long.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from bank.common import read_jsonl
from harness.prompts import TEMPLATES, messages

TOKENIZERS = {
    "Qwen/Qwen2.5-7B-Instruct": False,
    "mistralai/Mistral-7B-Instruct-v0.3": False,
    "meta-llama/Llama-3.1-8B-Instruct": False,
    "google/gemma-2-9b-it": False,
    "meta-llama/Llama-3.2-3B-Instruct": False,
    "GSAI-ML/LLaDA-8B-Instruct": True,   # trust_remote_code
    "Dream-org/Dream-v0-Instruct-7B": True,
}


def n_tokens(tok, msgs) -> int:
    """Prompt length in tokens. Newer transformers return a BatchEncoding (a dict) here, whose
    len() is its number of keys, so read input_ids explicitly."""
    ids = tok.apply_chat_template(msgs, tokenize=True, add_generation_prompt=True)
    if hasattr(ids, "keys"):
        ids = ids["input_ids"]
    if ids and isinstance(ids[0], list):
        ids = ids[0]
    return len(ids)


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("banks", nargs="+")
    ap.add_argument("--out", required=True)
    ap.add_argument("--max-tokens", type=int, default=1760)
    args = ap.parse_args(argv)
    from transformers import AutoTokenizer

    rows = [r for path in args.banks for r in read_jsonl(path)]
    longest: dict[str, dict[str, int]] = {r["candidate_id"]: {} for r in rows}
    loaded, failed = [], {}
    for model_id, remote in TOKENIZERS.items():
        try:
            tok = AutoTokenizer.from_pretrained(model_id, trust_remote_code=remote)
        except Exception as exc:  # noqa: BLE001 - report and carry on with the others
            failed[model_id] = f"{type(exc).__name__}: {exc}"[:300]
            print(f"could not load {model_id}: {failed[model_id]}", file=sys.stderr)
            continue
        loaded.append(model_id)
        for r in rows:
            record = {"problem": r["problem"], "candidate_answer": r["candidate"]}
            n = max(n_tokens(tok, messages(record, t)) for t in TEMPLATES)
            longest[r["candidate_id"]][model_id] = n
        print(f"{model_id}: longest prompt {max(v[model_id] for v in longest.values())} tokens")
    too_long = sorted(c for c, by_model in longest.items() if any(n > args.max_tokens for n in by_model.values()))
    Path(args.out).write_text(json.dumps({
        "max_tokens": args.max_tokens, "templates": sorted(TEMPLATES), "tokenizers_loaded": loaded,
        "tokenizers_failed": failed, "too_long": too_long,
        "longest_overall": max(max(v.values(), default=0) for v in longest.values()),
    }, indent=2) + "\n", encoding="utf-8", newline="\n")
    print(f"{len(too_long)} of {len(rows)} candidates exceed {args.max_tokens} tokens for some verifier")
    if failed:
        print(f"WARNING: {len(failed)} tokenizers failed to load: {sorted(failed)}", file=sys.stderr)


if __name__ == "__main__":
    main()
