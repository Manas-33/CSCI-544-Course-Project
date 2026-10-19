"""Shared helpers for the math bank: file I/O, GSM8K reference answers, answer extraction.

Everything here is pure Python so it can be unit-tested on a CPU without downloads.
"""

from __future__ import annotations

import hashlib
import json
import re
from decimal import Decimal, InvalidOperation
from pathlib import Path
from typing import Iterable, Optional

# Official GSM8K release (MIT license). The file's SHA-256 is recorded in each manifest.
GSM8K_TRAIN_URL = (
    "https://raw.githubusercontent.com/openai/grade-school-math/master/"
    "grade_school_math/data/train.jsonl"
)
GSM8K_TRAIN_SIZE = 7473


# ---------------------------------------------------------------- files

def read_jsonl(path: str | Path) -> list[dict]:
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def write_jsonl(path: str | Path, rows: Iterable[dict]) -> int:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    n = 0
    with open(path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
            n += 1
    return n


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_text(text: str) -> str:
    return sha256_bytes(text.encode("utf-8"))


# ---------------------------------------------------------------- numbers

_NUMBER_RE = re.compile(r"-?\d[\d,]*(?:\.\d+)?|-?\.\d+")


def normalize_number(raw: Optional[str]) -> Optional[str]:
    """Turn an answer string into a canonical number string, or None if it is not a number.

    Strips $, %, commas, spaces, a trailing period and simple LaTeX wrappers
    (\\text{}, \\$, \\!, \\,), then prints integers without decimals ("18.0" -> "18").
    """
    if raw is None:
        return None
    s = raw.strip()
    s = re.sub(r"\\text\{([^}]*)\}", r"\1", s)
    for token in ("\\$", "\\%", "\\!", "\\,", "$", "%", " "):
        s = s.replace(token, "")
    s = s.replace(",", "").rstrip(".")
    if not s:
        return None
    try:
        value = Decimal(s)
    except InvalidOperation:
        return None
    if value == value.to_integral_value():
        return str(int(value))
    return format(value.normalize(), "f")


def gsm8k_reference(solution: str) -> Optional[str]:
    """GSM8K solutions end with '#### <answer>'; return that answer, normalized."""
    if "####" not in solution:
        return None
    return normalize_number(solution.rsplit("####", 1)[1])


# ---------------------------------------------------------------- extraction

def extract_boxed(text: str) -> Optional[str]:
    """Content of the last \\boxed{...} in the text, with nested braces handled."""
    start = text.rfind("\\boxed{")
    if start == -1:
        return None
    i = start + len("\\boxed{")
    depth = 1
    out = []
    while i < len(text):
        ch = text[i]
        if ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                return "".join(out).strip()
        out.append(ch)
        i += 1
    return None  # unbalanced braces: treat as no answer


def parse_boxed_number(content: Optional[str]) -> Optional[str]:
    """Read a number from \\boxed{} content that may carry units or an equation.

    "41" -> 41; "41 \\text{ dollars}" -> 41; "200 - 159 = 41" -> 41 (the part after the
    last "="). If several numbers remain and none is clearly the answer ("3 \\times 4"),
    return None rather than guess.
    """
    if content is None:
        return None
    direct = normalize_number(content)
    if direct is not None:
        return direct
    tail = content.rsplit("=", 1)[-1]
    direct = normalize_number(tail)
    if direct is not None:
        return direct
    numbers = _NUMBER_RE.findall(tail)
    return normalize_number(numbers[0]) if len(numbers) == 1 else None


# Model self-references that PLAN.md §2 filters out of the bank ("As an AI...").
_IDENTITY_RE = re.compile(
    r"\bas an ai\b|\bi am an ai\b|\bi'm an ai\b|\bas a language model\b|\bai language model\b",
    re.IGNORECASE,
)


def has_identity_string(text: str) -> bool:
    return bool(_IDENTITY_RE.search(text))


def extract_last_number(text: str) -> Optional[str]:
    """The last number written anywhere in the text (a looser, independent extraction)."""
    matches = _NUMBER_RE.findall(text)
    return matches[-1] if matches else None


def normalize_text(text: str) -> str:
    """Lower-cased text with whitespace collapsed, used for near-duplicate checks."""
    return " ".join(text.lower().split())


def token_jaccard(a: str, b: str) -> float:
    ta, tb = set(normalize_text(a).split()), set(normalize_text(b).split())
    if not ta and not tb:
        return 1.0
    return len(ta & tb) / len(ta | tb)
