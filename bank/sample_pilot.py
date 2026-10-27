"""Step 5 (pilot): draw the pilot's matched pairs from every qualifying pair (PLAN.md §2).

Input: one make_pairs output per domain (all qualifying pairs, any number per problem).
Output: bank/pilot/pilot_bank.jsonl and pilot_labels.jsonl with --pairs-per-domain pairs
in each domain, drawn in a seeded random order under the caps of PLAN.md §2:
- at most 2 pairs per math problem and 3 per code problem, each from a different generator
  (make_pairs gives one pair per (problem, generator) cell, so that holds by construction);
- no generator supplies more than 30% of a domain's pairs (40% with three generators, counting
  only generators that can supply at least 10% of the target);
- math is split evenly between GSM8K-Platinum and MATH-500.
Pairs with a candidate on the too-long list (bank.token_lengths) are dropped first. If the
caps leave the target unmet, the shortfall is reported, not filled by relaxing them.

    python -m bank.sample_pilot --math bank/pilot/math --code bank/pilot/code \\
        --too-long bank/pilot/too_long.json --out bank/pilot
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
from collections import Counter, defaultdict
from pathlib import Path

from bank.common import read_jsonl, write_jsonl

PER_PROBLEM = {"math": 2, "code": 3}


def generator_cap(n_generators: int, target: int) -> int:
    share = 0.30 if n_generators >= 4 else 0.40
    return max(1, int(share * target))


def draw(pairs: dict[str, list[dict]], domain: str, target: int, seed: int) -> tuple[list[str], dict]:
    """Pair ids chosen for one domain, and a summary of what limited the draw."""
    order = sorted(pairs)
    random.Random(f"{seed}:{domain}").shuffle(order)
    # A generator counts as available only if it can supply at least 10% of the target; one that
    # is almost always right (GPT-4.1-mini on math) yields too few pairs to share the load, and the
    # 30% cap would then leave the domain short. PLAN.md §2 sets 40% for three generators.
    supply = Counter(pairs[p][0]["generator"] for p in order)
    generators = {g for g, n in supply.items() if n >= 0.10 * target}
    gen_cap = generator_cap(len(generators), target)
    sources = sorted({pairs[p][0]["source"] for p in order})
    source_cap = target // len(sources) + (target % len(sources) > 0) if domain == "math" else target
    per_problem, per_gen, per_source = Counter(), Counter(), Counter()
    chosen, skipped = [], Counter()
    for pid in order:
        row = pairs[pid][0]
        if len(chosen) == target:
            break
        if per_problem[row["problem_id"]] >= PER_PROBLEM[domain]:
            skipped["problem_cap"] += 1
        elif per_gen[row["generator"]] >= gen_cap:
            skipped["generator_cap"] += 1
        elif per_source[row["source"]] >= source_cap:
            skipped["source_cap"] += 1
        else:
            chosen.append(pid)
            per_problem[row["problem_id"]] += 1
            per_gen[row["generator"]] += 1
            per_source[row["source"]] += 1
    return chosen, {
        "available_pairs": len(order), "target": target, "chosen": len(chosen),
        "shortfall": target - len(chosen), "generator_cap": gen_cap, "source_cap": source_cap,
        "available_by_generator": dict(supply), "generators_counted_for_cap": sorted(generators),
        "by_generator": dict(per_gen), "by_source": dict(per_source), "skipped": dict(skipped),
    }


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--math", required=True, help="make_pairs output folder for math (split pilot)")
    ap.add_argument("--code", required=True, help="make_pairs output folder for code (split pilot)")
    ap.add_argument("--too-long", required=True)
    ap.add_argument("--out", default="bank/pilot")
    ap.add_argument("--pairs-per-domain", type=int, default=125)
    ap.add_argument("--seed", type=int, default=544)
    args = ap.parse_args(argv)
    too_long = set(json.loads(Path(args.too_long).read_text())["too_long"])

    bank_rows, label_rows, summary = [], [], {"too_long_candidates": len(too_long)}
    for domain, folder in (("math", Path(args.math)), ("code", Path(args.code))):
        rows = read_jsonl(folder / "pilot_bank.jsonl")
        labels = {r["candidate_id"]: r for r in read_jsonl(folder / "pilot_labels.jsonl")}
        pairs = defaultdict(list)
        for r in rows:
            pairs[r["pair_id"]].append(r)
        dropped = {p for p, members in pairs.items() if any(m["candidate_id"] in too_long for m in members)}
        kept = {p: m for p, m in pairs.items() if p not in dropped}
        chosen, info = draw(kept, domain, args.pairs_per_domain, args.seed)
        info["pairs_dropped_too_long"] = len(dropped)
        info["make_pairs_summary"] = json.loads((folder / "pilot_bank_summary.json").read_text())
        summary[domain] = info
        for pid in sorted(chosen):
            for r in pairs[pid]:
                bank_rows.append(r)
                label_rows.append(labels[r["candidate_id"]])

    out = Path(args.out)
    write_jsonl(out / "pilot_bank.jsonl", bank_rows)
    write_jsonl(out / "pilot_labels.jsonl", label_rows)
    summary["outputs"] = {name: hashlib.sha256((out / name).read_bytes()).hexdigest()
                          for name in ("pilot_bank.jsonl", "pilot_labels.jsonl")}
    summary["seed"] = args.seed
    (out / "pilot_bank_summary.json").write_text(json.dumps(summary, indent=2) + "\n",
                                                 encoding="utf-8", newline="\n")
    print(json.dumps({d: {k: summary[d][k] for k in ("available_pairs", "chosen", "shortfall", "by_generator",
                                                     "by_source")} for d in ("math", "code")}, indent=2))


if __name__ == "__main__":
    main()
