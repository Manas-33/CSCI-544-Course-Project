"""Step 5 (full bank): draw one domain's matched pairs by PLAN.md §2's draw rule.

Input: the folder make_pairs wrote with --split full (every qualifying pair) and the matching
too_long.json from bank.token_lengths. Output, in the same folder: drawn_bank.jsonl,
drawn_labels.jsonl and draw_summary.json. The folder holds labels, so keep it out of git.

The rule, in order:
0. Code uses HumanEval+ only (decisions/2026-11-22-label-check-by-domain.md): MBPP+ pairs are removed before
   anything below is counted, and the number removed is in the summary ("left_out"). The counts and shares
   below are then about HumanEval+ alone; only longest_prompt_tokens still covers every pair measured.
   Any other source is refused.
1. A pair with an answer on the too-long list is dropped.
2. Per-problem cap: at most 2 pairs per math problem and 3 per code problem, the first ones in a
   seeded random order of all pairs. Each comes from a different generator, because make_pairs gives one
   pair per (problem, generator) cell.
3. Target. Math: with G the GSM8K-Platinum pairs and M the MATH pairs left after the cap, 2 x min(G, M),
   half from each. Code: every pair left after the cap.
4. Generator cap: no generator supplies more than 30% of the target, or 40% if fewer than four
   generators are available. A generator is available if it has at least 10% of the target in
   qualifying pairs, counted after steps 0 and 1 and before step 2, as bank.sample_pilot counts them.
   The cap is one number per domain, not one per half.
5. Draw. GSM8K pairs in seeded random order; MATH pairs with MATH-500 first, then the new problems, each in
   seeded random order; code pairs in seeded random order. A pair whose generator is at its
   cap is skipped. The two math halves take turns, one pair each, so that neither uses up a generator's
   cap first. (bank.sample_pilot splits evenly over sources, which would leave GSM8K a third here.)
6. A half, or the code target, that stays short is reported as a shortfall, never filled.

    python -m bank.draw_full --domain math --work work/math --questions bank/full/questions_math.jsonl \\
        --generations work/math/generations.jsonl
"""

from __future__ import annotations

import argparse
import json
import random
from collections import Counter, defaultdict
from pathlib import Path

from bank.common import read_jsonl, sha256_bytes, write_jsonl
from bank.token_lengths import TOKENIZERS

PER_PROBLEM = {"math": 2, "code": 3}
GSM8K, MATH500, MATH_REST = "gsm8k-platinum/test", "math-500/test", "math-rest/test"
# Each half of the math target, and the order its sources are taken in.
MATH_HALVES = {"gsm8k": (GSM8K,), "math": (MATH500, MATH_REST)}
CODE_SOURCES = ("evalplus/humaneval+",)
# Generated and labeled, but not part of the bank.
LEFT_OUT = {"math": (), "code": ("evalplus/mbpp+",)}
SECOND_CHECKER = {"math": "+math_verify==", "code": "+evalplus=="}


def seeded_order(pair_ids, seed: int, salt: str) -> list[str]:
    order = sorted(pair_ids)
    random.Random(f"{seed}:full:{salt}").shuffle(order)
    return order


def cap_per_problem(pairs: dict[str, dict], domain: str, seed: int) -> list[str]:
    """Pair ids left after the per-problem cap (step 2)."""
    kept, per_problem = [], Counter()
    for pid in seeded_order(pairs, seed, f"{domain}:problem-cap"):
        problem = pairs[pid]["problem_id"]
        if per_problem[problem] < PER_PROBLEM[domain]:
            per_problem[problem] += 1
            kept.append(pid)
    return sorted(kept)


def generator_cap(supply: Counter, target: int) -> tuple[int, int, list[str]]:
    """(pairs allowed per generator, the cap in percent, the generators that count as available)."""
    available = sorted(g for g, n in supply.items() if 10 * n >= target)
    percent = 30 if len(available) >= 4 else 40
    return max(1, target * percent // 100), percent, available


def take(queues: dict[str, list[str]], quota: dict[str, int], pairs: dict[str, dict],
         cap: int) -> tuple[dict[str, list[str]], Counter]:
    """Take turns over the queues, one pair each, skipping pairs whose generator is at its cap."""
    position = {name: 0 for name in queues}
    chosen: dict[str, list[str]] = {name: [] for name in queues}
    per_generator, skipped = Counter(), Counter()
    active = True
    while active:
        active = False
        for name, queue in queues.items():
            while len(chosen[name]) < quota[name] and position[name] < len(queue):
                pid = queue[position[name]]
                position[name] += 1
                if per_generator[pairs[pid]["generator"]] < cap:
                    per_generator[pairs[pid]["generator"]] += 1
                    chosen[name].append(pid)
                    active = True
                    break
                skipped[name] += 1
    return chosen, skipped


def draw(pairs: dict[str, dict], domain: str, seed: int = 544) -> tuple[list[str], dict]:
    """(chosen pair ids, summary) for one domain. `pairs` maps a pair id to its source, problem and generator."""
    known = {s for half in MATH_HALVES.values() for s in half} if domain == "math" else set(CODE_SOURCES)
    unknown = {p["source"] for p in pairs.values()} - known
    if unknown:
        raise SystemExit(f"{domain}: unexpected sources {sorted(unknown)}")
    supply = Counter(p["generator"] for p in pairs.values())
    kept = cap_per_problem(pairs, domain, seed)
    by_source = defaultdict(list)
    for pid in kept:
        by_source[pairs[pid]["source"]].append(pid)

    if domain == "math":
        queues = {half: [pid for source in sources for pid in seeded_order(by_source[source], seed, source)]
                  for half, sources in MATH_HALVES.items()}
        each = min(len(q) for q in queues.values())
        quota = {half: each for half in queues}
    else:
        queues = {"code": seeded_order(kept, seed, "code")}
        quota = {"code": len(kept)}
    target = sum(quota.values())
    cap, percent, available = generator_cap(supply, target)
    chosen, skipped = take(queues, quota, pairs, cap)
    ids = sorted(pid for part in chosen.values() for pid in part)
    count = lambda key: dict(sorted(Counter(pairs[pid][key] for pid in ids).items()))  # noqa: E731
    return ids, {
        "qualifying_pairs": len(pairs),
        "qualifying_by_generator": dict(sorted(supply.items())),
        "pairs_after_problem_cap": len(kept),
        "after_problem_cap_by_source": {s: len(v) for s, v in sorted(by_source.items())},
        "target": target,
        "quota": quota,
        "generator_cap": {"pairs": cap, "percent": percent, "available": available,
                          "fewer_than_three_available": len(available) < 3},
        "chosen": len(ids),
        "shortfall": {name: quota[name] - len(chosen[name]) for name in queues},
        "skipped_at_generator_cap": {name: skipped[name] for name in queues},
        "by_source": count("source"),
        "by_generator": count("generator"),
    }


def without_left_out(domain: str, rows: dict[str, list[dict]], questions: list[dict],
                     generations: list[dict]) -> tuple[dict[str, list[dict]], list[dict], list[dict], dict]:
    """(pairs, questions, answers) without the sources the domain leaves out, and how much each of them held."""
    sources = LEFT_OUT[domain]
    source = {q["problem_id"]: q["source"] for q in questions}
    removed = {}
    for s in sources:
        pairs = sum(members[0]["source"] == s for members in rows.values())
        removed[s] = {"problems": sum(q["source"] == s for q in questions),
                      "answers": sum(source[g["problem_id"]] == s for g in generations),
                      "qualifying_pairs": pairs, "answers_in_pairs": 2 * pairs}
    return ({pid: members for pid, members in rows.items() if members[0]["source"] not in sources},
            [q for q in questions if q["source"] not in sources],
            [g for g in generations if source[g["problem_id"]] not in sources], removed)


def cell_shares(questions: list[dict], cells: set[tuple[str, str]], pairs: dict[str, dict]) -> dict:
    """How many (problem, generator) cells gave a matched pair, overall and by source and generator (PLAN.md §2)."""
    source = {q["problem_id"]: q["source"] for q in questions}
    total, qualified = Counter(), Counter()
    for problem, generator in cells:
        for key in ("all", source[problem], generator):
            total[key] += 1
    for p in pairs.values():
        for key in ("all", p["source"], p["generator"]):
            qualified[key] += 1
    return {key: {"cells": total[key], "qualified": qualified[key], "share": round(qualified[key] / total[key], 4)}
            for key in sorted(total)}


def load_pairs(work: Path, domain: str) -> tuple[dict[str, list[dict]], dict[str, dict]]:
    """Pair id -> its two bank rows, and candidate id -> label row, checked to be one right and one wrong answer."""
    labels = {r["candidate_id"]: r for r in read_jsonl(work / "full_labels.jsonl")}
    pairs = defaultdict(list)
    for r in read_jsonl(work / "full_bank.jsonl"):
        pairs[r["pair_id"]].append(r)
    cells = set()
    for pid, members in pairs.items():
        cell = {(m["problem_id"], m["generator"], m["source"], m["domain"]) for m in members}
        if len(members) != 2 or len(cell) != 1 or {m["domain"] for m in members} != {domain}:
            raise SystemExit(f"{pid}: not two {domain} answers of one (problem, generator) cell")
        if sorted(labels[m["candidate_id"]]["label"] for m in members) != [0, 1]:
            raise SystemExit(f"{pid}: not one correct and one wrong answer")
        if cell & cells:
            raise SystemExit(f"{pid}: a second pair from the same (problem, generator) cell")
        cells |= cell
    checkers = {r["checker_version"] for r in labels.values()}
    if len(checkers) != 1 or SECOND_CHECKER[domain] not in checkers.pop():
        raise SystemExit(f"{domain}: labels must come from one checker version with {SECOND_CHECKER[domain][1:-2]}; "
                         f"found {sorted({r['checker_version'] for r in labels.values()})}")
    return dict(pairs), labels


def load_too_long(work: Path) -> set[str]:
    """Candidates too long for some verifier, refusing a list that is partial or from other answers."""
    data = json.loads((work / "too_long.json").read_text(encoding="utf-8"))
    missing = sorted(set(TOKENIZERS) - set(data["tokenizers_loaded"]))
    if missing or data["tokenizers_failed"]:
        raise SystemExit(f"{work / 'too_long.json'} is not from every verifier's tokenizer; missing {missing}")
    if sha256_bytes((work / "full_bank.jsonl").read_bytes()) not in data.get("banks_sha256", []):
        raise SystemExit(f"{work / 'too_long.json'} was made from other answers; rerun bank.token_lengths")
    return set(data["too_long"])


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--domain", required=True, choices=sorted(PER_PROBLEM))
    ap.add_argument("--work", required=True, help="make_pairs --split full output, with too_long.json")
    ap.add_argument("--questions", required=True)
    ap.add_argument("--generations", required=True)
    ap.add_argument("--seed", type=int, default=544)
    args = ap.parse_args(argv)
    work = Path(args.work)

    made = json.loads((work / "full_bank_summary.json").read_text(encoding="utf-8"))
    for path in (args.questions, args.generations):
        if made["inputs"].get(Path(path).name) != sha256_bytes(Path(path).read_bytes()):
            raise SystemExit(f"{path} is not the file the pairs in {work} were made from")
    for name, digest in made["outputs"].items():
        if sha256_bytes((work / name).read_bytes()) != digest:
            raise SystemExit(f"{work / name} is not the file bank.make_pairs wrote; rerun it")
    rows, labels = load_pairs(work, args.domain)
    rows, questions, generations, removed = without_left_out(args.domain, rows, read_jsonl(args.questions),
                                                             read_jsonl(args.generations))
    too_long = load_too_long(work)
    if removed:
        too_long &= {m["candidate_id"] for members in rows.values() for m in members}
    dropped = {pid for pid, members in rows.items() if any(m["candidate_id"] in too_long for m in members)}
    info = {pid: {k: members[0][k] for k in ("source", "problem_id", "generator")} for pid, members in rows.items()}
    chosen, summary = draw({p: i for p, i in info.items() if p not in dropped}, args.domain, args.seed)

    cells = {(g["problem_id"], g["generator"]) for g in generations}
    bank_rows = [r for pid in chosen for r in rows[pid]]
    write_jsonl(work / "drawn_bank.jsonl", bank_rows)
    write_jsonl(work / "drawn_labels.jsonl", [labels[r["candidate_id"]] for r in bank_rows])
    result = {
        "step": "draw_full",
        "domain": args.domain,
        "seed": args.seed,
        **({"left_out": removed} if removed else {}),
        "pairs_dropped_too_long": len(dropped),
        "too_long_candidates": sorted(too_long),
        "longest_prompt_tokens": json.loads((work / "too_long.json").read_text(encoding="utf-8"))["longest_overall"],
        **summary,
        "cells": cell_shares(questions, cells, info),
        "inputs": {**made["inputs"], **made["outputs"],
                   "too_long.json": sha256_bytes((work / "too_long.json").read_bytes())},
        "outputs": {n: sha256_bytes((work / n).read_bytes()) for n in ("drawn_bank.jsonl", "drawn_labels.jsonl")},
    }
    (work / "draw_summary.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8", newline="\n")
    print(json.dumps({k: result[k] for k in ("domain", "left_out", "qualifying_pairs", "pairs_dropped_too_long",
                                             "pairs_after_problem_cap", "target", "chosen", "shortfall",
                                             "generator_cap", "by_source", "by_generator") if k in result}, indent=2))


if __name__ == "__main__":
    main()
