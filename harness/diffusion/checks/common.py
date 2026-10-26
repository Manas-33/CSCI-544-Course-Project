"""Shared helpers for the GPU scoring-code checks (PLAN.md §4, tests 1-3)."""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

from ..interface import TEMPLATES, read_jsonl
from ..scorer import SPECS, load_scorer

# The official 50-item sanity set (bank/sanity/README.md), shared with the AR harness.
SANITY_SET = Path(__file__).resolve().parents[3] / "bank" / "sanity" / "sanity_set.jsonl"


def model_args(description: str) -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description=description)
    ap.add_argument("--model", required=True, choices=sorted(SPECS))
    ap.add_argument("--n-slots", type=int, default=8, choices=[1, 8, 32])
    ap.add_argument("--items", default=str(SANITY_SET),
                    help="JSONL with candidate_id, problem, candidate (default: the sanity set)")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--revision", default=None, help="override the pinned HF revision")
    ap.add_argument("--allow-env-mismatch", action="store_true")
    ap.add_argument("--report", default=None, help="write the JSON report here")
    return ap


def scorer_from_args(args):
    return load_scorer(args.model, args.n_slots, device=args.device,
                       revision=args.revision, strict_env=not args.allow_env_mismatch)


def load_items(path) -> list:
    return list(read_jsonl(path))


def message(item, template="T1") -> str:
    return TEMPLATES[template](item["problem"], item["candidate"])


def verdicts_for(scorer, prompts, batch_size=1) -> list:
    """Verdicts for prompt id lists, ``batch_size`` prompts per forward pass."""
    out = []
    for i in range(0, len(prompts), batch_size):
        out.extend(scorer.score_ids(prompts[i:i + batch_size]))
    return out


def auroc(scores, labels) -> float:
    """P(a random positive outscores a random negative); ties count one half."""
    pos = [s for s, y in zip(scores, labels) if y == 1]
    neg = [s for s, y in zip(scores, labels) if y == 0]
    if not pos or not neg:
        raise ValueError("AUROC needs both labels")
    wins = sum((p > n) + 0.5 * (p == n) for p in pos for n in neg)
    return wins / (len(pos) * len(neg))


def ranks(xs) -> list:
    """1-based ranks; tied values share their average rank."""
    order = sorted(range(len(xs)), key=lambda i: xs[i])
    r = [0.0] * len(xs)
    i = 0
    while i < len(order):
        j = i
        while j + 1 < len(order) and xs[order[j + 1]] == xs[order[i]]:
            j += 1
        for k in range(i, j + 1):
            r[order[k]] = (i + j) / 2 + 1
        i = j + 1
    return r


def spearman(a, b) -> float:
    """Spearman rank correlation (Pearson on average ranks); nan if one side is constant."""
    ra, rb = ranks(a), ranks(b)
    ma, mb = sum(ra) / len(ra), sum(rb) / len(rb)
    cov = sum((x - ma) * (y - mb) for x, y in zip(ra, rb))
    va = sum((x - ma) ** 2 for x in ra)
    vb = sum((y - mb) ** 2 for y in rb)
    return cov / math.sqrt(va * vb) if va and vb else float("nan")


def finish(name: str, status: str, report: dict, path=None) -> None:
    """Print the verdict, save the report, exit 0 only on PASS."""
    report = {"check": name, "status": status, **report}
    if path:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        Path(path).write_text(json.dumps(report, indent=2))
    print(f"\n{name}: {status}")
    sys.exit(0 if status == "PASS" else 1)
