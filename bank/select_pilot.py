"""Pick the pilot's problems with a fixed seed (PLAN.md §2, §10).

Math: GSM8K-Platinum test (labels with the Platinum corrections) and MATH-500, keeping
only MATH problems whose reference answer is a rational number, which the two checkers
read reliably. Code: HumanEval+ and MBPP+ from EvalPlus. Development problems never
overlap these: the math development bank uses GSM8K train, and the code one uses MBPP
problems outside MBPP+.

    python -m bank.select_pilot --data-dir <folder with the downloaded files> --out bank/pilot

The folder holds gsm8k_platinum_test.parquet, math500_test.jsonl, HumanEvalPlus.jsonl
and MbppPlus.jsonl (the URLs are in SOURCES); their SHA-256 hashes go in the manifest.
"""

from __future__ import annotations

import argparse
import json
import random
from datetime import datetime, timezone
from pathlib import Path

from bank.common import gsm8k_reference, parse_rational, sha256_bytes, write_jsonl

SOURCES = {
    "gsm8k_platinum_test.parquet":
        "https://huggingface.co/datasets/madrylab/gsm8k-platinum/resolve/main/main/test-00000-of-00001.parquet",
    "math500_test.jsonl": "https://huggingface.co/datasets/HuggingFaceH4/MATH-500/resolve/main/test.jsonl",
    "HumanEvalPlus.jsonl":
        "https://github.com/evalplus/humanevalplus_release/releases/download/v0.1.10/HumanEvalPlus.jsonl.gz",
    "MbppPlus.jsonl": "https://github.com/evalplus/mbppplus_release/releases/download/v0.2.0/MbppPlus.jsonl.gz",
}


def _jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def gsm8k_platinum(path: Path) -> list[dict]:
    import pyarrow.parquet as pq

    out = []
    for i, row in enumerate(pq.read_table(path).to_pylist()):
        ref = gsm8k_reference(row["answer"])
        if ref is None:
            raise ValueError(f"GSM8K-Platinum row {i} has no '####' answer")
        out.append({"problem_id": f"gsm8kp-{i:04d}", "source": "gsm8k-platinum/test", "source_index": i,
                    "domain": "math", "question": row["question"], "reference": ref,
                    "cleaning_status": row.get("cleaning_status")})
    return out


def math500(path: Path) -> list[dict]:
    out = []
    for i, row in enumerate(_jsonl(path)):
        ref = parse_rational(row["answer"])
        if ref is None:
            continue
        out.append({"problem_id": f"math500-{i:03d}", "source": "math-500/test", "source_index": i,
                    "source_id": row["unique_id"], "domain": "math", "question": row["problem"],
                    "reference": ref, "reference_latex": row["answer"], "answer_kind": "rational",
                    "subject": row["subject"], "level": row["level"]})
    return out


def evalplus(path: Path, dataset: str) -> list[dict]:
    out = []
    for i, row in enumerate(_jsonl(path)):
        num = row["task_id"].split("/")[1]
        out.append({"problem_id": f"{dataset}-plus-{int(num):03d}", "source": f"evalplus/{dataset}+",
                    "source_index": i, "task_id": row["task_id"], "dataset": dataset, "domain": "code",
                    "question": row["prompt"].strip(), "entry_point": row["entry_point"]})
    return out


def sample(rows: list[dict], n: int, seed: int, salt: str) -> list[dict]:
    if n > len(rows):
        raise SystemExit(f"{salt}: asked for {n} problems, only {len(rows)} available")
    picked = sorted(random.Random(f"{seed}:{salt}").sample(range(len(rows)), n))
    return [rows[i] for i in picked]


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--data-dir", required=True)
    ap.add_argument("--out", default="bank/pilot")
    ap.add_argument("--seed", type=int, default=544)
    ap.add_argument("--n-gsm8k", type=int, default=150)
    ap.add_argument("--n-math", type=int, default=150)
    ap.add_argument("--n-humaneval", type=int, default=50)
    ap.add_argument("--n-mbpp", type=int, default=100)
    args = ap.parse_args(argv)
    data, out = Path(args.data_dir), Path(args.out)

    pools = {
        "gsm8k": gsm8k_platinum(data / "gsm8k_platinum_test.parquet"),
        "math": math500(data / "math500_test.jsonl"),
        "humaneval": evalplus(data / "HumanEvalPlus.jsonl", "humaneval"),
        "mbpp": evalplus(data / "MbppPlus.jsonl", "mbpp"),
    }
    math_q = sample(pools["gsm8k"], args.n_gsm8k, args.seed, "gsm8k") + \
        sample(pools["math"], args.n_math, args.seed, "math")
    code_q = sample(pools["humaneval"], args.n_humaneval, args.seed, "humaneval") + \
        sample(pools["mbpp"], args.n_mbpp, args.seed, "mbpp")
    write_jsonl(out / "questions_math.jsonl", math_q)
    write_jsonl(out / "questions_code.jsonl", code_q)
    (out / "question_ids.txt").write_text(
        "\n".join(q["problem_id"] for q in math_q + code_q) + "\n", encoding="utf-8", newline="\n")
    manifest = {
        "step": "select_pilot",
        "seed": args.seed,
        "sources": {name: {"url": url, "sha256": sha256_bytes((data / name).read_bytes())}
                    for name, url in SOURCES.items()},
        "pool_sizes": {k: len(v) for k, v in pools.items()},
        "selected": {"gsm8k": args.n_gsm8k, "math": args.n_math,
                     "humaneval": args.n_humaneval, "mbpp": args.n_mbpp},
        "math500_rule": "reference answer parses as a rational number (bank.common.parse_rational)",
        "created_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }
    (out / "questions_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n",
                                                 encoding="utf-8", newline="\n")
    print(json.dumps(manifest["pool_sizes"]), f"-> {len(math_q)} math, {len(code_q)} code questions in {out}")


if __name__ == "__main__":
    main()
