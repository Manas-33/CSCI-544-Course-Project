"""Step 1: pick development-bank questions from GSM8K train with a fixed seed.

Train questions are used so the clean GSM8K-Platinum test set stays free for the
final bank. The selected IDs are saved so later banks can exclude them (PLAN.md §2).

    python -m bank.select_questions --n 500 --seed 544 --out bank/dev
"""

from __future__ import annotations

import argparse
import json
import random
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

from bank.common import (
    GSM8K_TRAIN_SIZE,
    GSM8K_TRAIN_URL,
    gsm8k_reference,
    sha256_bytes,
    write_jsonl,
)


def load_gsm8k_train(source: str) -> tuple[list[dict], str]:
    """Rows of GSM8K train and the SHA-256 of the raw file (from a URL or a local path)."""
    if source.startswith("http"):
        with urllib.request.urlopen(source, timeout=60) as resp:
            raw = resp.read()
    else:
        raw = Path(source).read_bytes()
    rows = [json.loads(line) for line in raw.decode("utf-8").splitlines() if line.strip()]
    return rows, sha256_bytes(raw)


def select(rows: list[dict], n: int, seed: int) -> list[dict]:
    """Seeded sample of n questions, returned in their original order."""
    picked = sorted(random.Random(seed).sample(range(len(rows)), n))
    out = []
    for idx in picked:
        ref = gsm8k_reference(rows[idx]["answer"])
        if ref is None:
            raise ValueError(f"GSM8K train line {idx} has no '####' answer")
        out.append(
            {
                "problem_id": f"gsm8k-train-{idx:05d}",
                "source": "gsm8k/train",
                "source_index": idx,
                "domain": "math",
                "question": rows[idx]["question"],
                "reference": ref,
            }
        )
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--n", type=int, default=500)
    ap.add_argument("--seed", type=int, default=544)
    ap.add_argument("--source", default=GSM8K_TRAIN_URL, help="URL or local path of GSM8K train.jsonl")
    ap.add_argument("--out", default="bank/dev")
    args = ap.parse_args()

    rows, file_sha = load_gsm8k_train(args.source)
    if len(rows) != GSM8K_TRAIN_SIZE:
        raise SystemExit(f"Expected {GSM8K_TRAIN_SIZE} GSM8K train rows, got {len(rows)}")

    questions = select(rows, args.n, args.seed)
    out = Path(args.out)
    write_jsonl(out / "questions.jsonl", questions)
    (out / "question_ids.txt").write_text(
        "\n".join(q["problem_id"] for q in questions) + "\n", encoding="utf-8", newline="\n"
    )
    manifest = {
        "step": "select_questions",
        "source_url": args.source,
        "source_sha256": file_sha,
        "source_rows": len(rows),
        "n": args.n,
        "seed": args.seed,
        "created_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }
    (out / "questions_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8", newline="\n")
    print(f"Wrote {len(questions)} questions to {out / 'questions.jsonl'}")


if __name__ == "__main__":
    main()
