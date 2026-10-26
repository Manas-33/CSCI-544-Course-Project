"""Score rows and the T1 prompt, as the diffusion scorers use them.

Role C owns the common interface (PLAN.md §11) and will align these names with
the AR adapters. Until then, this is the contract the diffusion code writes.

Score row (PLAN.md §9), one JSON object per line:

    bank_hash     bank version the candidate comes from ("sanity" for the sanity set)
    candidate_id  candidate id from the bank row
    model         short model key: llada | illada | dream
    revision      Hugging Face commit of the weights and remote code
    template      prompt template name: T1 | T2 | T3
    n_slots       number of masked answer slots (1, 8 or 32)
    answer_prefix name of the text after the generation prompt (scorer.ANSWER_PREFIXES; "none" by default)
    score         log P(Yes) - log P(No) at the first answer slot (null unless ok)
    p_yes         P(Yes), summed over the single-token Yes spellings (null unless ok)
    p_no          P(No), likewise (null unless ok)
    yes_no_mass   p_yes + p_no, the format check (null unless ok)
    status        ok | too_long | error
    runtime_s     synchronized wall time of the forward pass / batch size
    peak_mem_gb   peak GPU memory allocated during the batch (null on CPU)

The score is never sign-flipped here: T3's flip belongs to the analysis.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable, Iterator, Optional

SCORE_ROW_FIELDS = (
    "bank_hash", "candidate_id", "model", "revision", "template", "n_slots",
    "answer_prefix", "score", "p_yes", "p_no", "yes_no_mass", "status", "runtime_s", "peak_mem_gb",
)
STATUSES = ("ok", "too_long", "error")

# T1, the same text as harness/prompts.py (harness/test_llada_compare.py checks this).
# T2 and T3 are added from that file when the diffusion models need them.
T1_QUESTION = "Is the final answer correct? Answer Yes or No."


def t1_message(problem: str, candidate: str) -> str:
    return f"Problem:\n{problem}\n\nCandidate solution:\n{candidate}\n\n{T1_QUESTION}"


TEMPLATES = {"T1": t1_message}


def score_row(*, bank_hash: str, candidate_id: str, model: str, revision: str,
              template: str, n_slots: int, verdict=None, status: str = "ok",
              answer_prefix: str = "none",
              runtime_s: Optional[float] = None,
              peak_mem_gb: Optional[float] = None) -> dict:
    """Build one score row; ``verdict`` is a scorer.Verdict (None unless ok)."""
    if status not in STATUSES:
        raise ValueError(f"unknown status {status!r}")
    if (status == "ok") != (verdict is not None):
        raise ValueError("a verdict is required exactly when status is ok")
    return {
        "bank_hash": bank_hash,
        "candidate_id": candidate_id,
        "model": model,
        "revision": revision,
        "template": template,
        "n_slots": n_slots,
        "answer_prefix": answer_prefix,
        "score": verdict.score if verdict else None,
        "p_yes": verdict.p_yes if verdict else None,
        "p_no": verdict.p_no if verdict else None,
        "yes_no_mass": verdict.mass if verdict else None,
        "status": status,
        "runtime_s": runtime_s,
        "peak_mem_gb": peak_mem_gb,
    }


def read_jsonl(path) -> Iterator[dict]:
    with open(path) as f:
        for line in f:
            if line.strip():
                yield json.loads(line)


def write_jsonl(path, rows: Iterable[dict], append: bool = False) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a" if append else "w") as f:
        for row in rows:
            f.write(json.dumps(row) + "\n")
