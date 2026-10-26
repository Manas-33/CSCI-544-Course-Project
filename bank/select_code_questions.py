"""Step 1 (code): pick development-bank code problems from MBPP with a fixed seed.

Only MBPP problems that are NOT in MBPP+ are used, so all 542 EvalPlus problems
(HumanEval+ and MBPP+) stay free for the pilot and the final bank, the way the math
development bank uses GSM8K train. These problems are tested with MBPP's own three
asserts, which are weaker than EvalPlus's tests: fine for pipeline checks, never
for results.

The problem text follows MBPP+'s layout (the task, then one example assert, in a
docstring), so development and pilot answers come from the same kind of prompt.

    python -m bank.select_code_questions --n 150 --seed 544 --out bank/dev_code
"""

from __future__ import annotations

import argparse
import ast
import gzip
import json
import random
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

from bank.common import normalize_text, sha256_bytes, write_jsonl

MBPP_URL = "https://raw.githubusercontent.com/google-research/google-research/master/mbpp/mbpp.jsonl"
MBPP_SIZE = 974
MBPP_PLUS_URL = "https://github.com/evalplus/mbppplus_release/releases/download/v0.2.0/MbppPlus.jsonl.gz"
MBPP_PLUS_SIZE = 378


def _fetch(source: str) -> bytes:
    if source.startswith("http"):
        with urllib.request.urlopen(source, timeout=60) as resp:
            return resp.read()
    return Path(source).read_bytes()


def load_jsonl_bytes(raw: bytes) -> list[dict]:
    if raw[:2] == b"\x1f\x8b":
        raw = gzip.decompress(raw)
    return [json.loads(line) for line in raw.decode("utf-8").splitlines() if line.strip()]


def mbpp_prompt(text: str, first_test: str) -> str:
    """MBPP+'s problem layout: the task and one example assert inside a docstring."""
    return f'"""\n{text.strip()}\n{first_test.strip()}\n"""\n'


def entry_point(reference: str, tests: list[str]) -> str | None:
    """The reference solution's top-level function that the tests call, if exactly one."""
    try:
        tree = ast.parse(reference)
    except SyntaxError:
        return None
    names = {n.name for n in tree.body if isinstance(n, ast.FunctionDef)}
    called = {
        n.func.id for t in tests for n in ast.walk(ast.parse(t))
        if isinstance(n, ast.Call) and isinstance(n.func, ast.Name)
    }
    found = names & called
    return found.pop() if len(found) == 1 else None


def eligible(mbpp: list[dict], mbpp_plus: list[dict]) -> list[dict]:
    """MBPP problems outside MBPP+ (by id and by exact task text), with a usable entry point."""
    plus_ids = {int(r["task_id"].split("/")[1]) for r in mbpp_plus}
    plus_texts = {normalize_text(r["prompt"].strip().strip('"').strip().splitlines()[0]) for r in mbpp_plus}
    out = []
    for row in mbpp:
        if row["task_id"] in plus_ids or normalize_text(row["text"]) in plus_texts:
            continue
        name = entry_point(row["code"], row["test_list"])
        if name is None:
            continue
        out.append(
            {
                "problem_id": f"mbpp-{row['task_id']:03d}",
                "source": "google-research/mbpp",
                "source_index": row["task_id"],
                "domain": "code",
                "question": mbpp_prompt(row["text"], row["test_list"][0]),
                "entry_point": name,
                "tests": row["test_list"],
                "setup": row.get("test_setup_code") or "",
                "reference_solution": row["code"],
            }
        )
    return out


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--n", type=int, default=150)
    ap.add_argument("--seed", type=int, default=544)
    ap.add_argument("--mbpp", default=MBPP_URL, help="URL or local path of mbpp.jsonl")
    ap.add_argument("--mbpp-plus", default=MBPP_PLUS_URL, help="URL or local path of MbppPlus.jsonl.gz")
    ap.add_argument("--out", default="bank/dev_code")
    args = ap.parse_args(argv)

    mbpp_raw, plus_raw = _fetch(args.mbpp), _fetch(args.mbpp_plus)
    mbpp, mbpp_plus = load_jsonl_bytes(mbpp_raw), load_jsonl_bytes(plus_raw)
    if len(mbpp) != MBPP_SIZE or len(mbpp_plus) != MBPP_PLUS_SIZE:
        raise SystemExit(f"Expected {MBPP_SIZE} MBPP and {MBPP_PLUS_SIZE} MBPP+ rows, got {len(mbpp)} and {len(mbpp_plus)}")

    pool = eligible(mbpp, mbpp_plus)
    picked = sorted(random.Random(args.seed).sample(range(len(pool)), args.n))
    questions = [pool[i] for i in picked]

    out = Path(args.out)
    write_jsonl(out / "questions.jsonl", questions)
    (out / "question_ids.txt").write_text(
        "\n".join(q["problem_id"] for q in questions) + "\n", encoding="utf-8", newline="\n"
    )
    manifest = {
        "step": "select_code_questions",
        "source_url": args.mbpp,
        "source_sha256": sha256_bytes(mbpp_raw),
        "excluded_source_url": args.mbpp_plus,
        "excluded_source_sha256": sha256_bytes(plus_raw),
        "eligible": len(pool),
        "n": args.n,
        "seed": args.seed,
        "created_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }
    (out / "questions_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8", newline="\n")
    print(f"Wrote {len(questions)} of {len(pool)} eligible MBPP problems to {out / 'questions.jsonl'}")


if __name__ == "__main__":
    main()
