"""Answer-swap items for scoring-code test 4 (PLAN.md §4).

Draws 100 correct math and 100 correct code pilot candidates (seed 544, at most one per
problem where possible) and shows each one twice: as it is, and with its answer swapped.
- Math: only the content of the last \\boxed{} is replaced, by the content of the last
  \\boxed{} of the matched wrong partner (the other answer with the same pair_id). Every
  other character stays the same.
- Code: the swap is the matched wrong partner's program.

Originals and swaps go to one file, so each verifier scores both in one run. The file loads
as a dev_bank input of harness.run (two rows per pair_id, in one problem/generator cell) and
as --items of harness.diffusion.run_score. Candidate IDs are hashes and the rows are sorted
by them, so neither the ID nor the position says which row is the original. Which is which
goes to a separate key file; no label is written to either.

A candidate is left out of the draw, with the reason in the manifest, if its swap would leave
the final answer unchanged, equal in value or equal to the reference answer, or if a prompt
could pass the 1,760-token limit. Lengths come from the pilot's own score rows (--scores):
the original's longest recorded prompt plus one token per byte of the swapped-in answer.
bank.token_lengths measures the finished file with every tokenizer before it is scored.

    python -m bank.answer_swap --bank bank/pilot/pilot_bank.jsonl \\
        --labels bank/pilot/pilot_labels.jsonl --scores results/pilot/ar --out bank/pilot/swap
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
from collections import Counter, defaultdict
from fractions import Fraction
from pathlib import Path

from bank.common import (extract_boxed, normalize_text, parse_boxed_number, parse_rational,
                         read_jsonl, sha256_text, write_jsonl)

SEED = 544
PER_DOMAIN = 100
MAX_PROMPT_TOKENS = 1760
ITEMS_FILENAME, KEY_FILENAME, MANIFEST_FILENAME = "swap_items.jsonl", "swap_key.jsonl", "swap_manifest.json"
COPIED_FIELDS = ("problem_id", "source", "domain", "problem", "entry_point", "generator", "generator_revision")


# ---------------------------------------------------------------- math: the last \boxed{}

def last_box_span(text: str) -> tuple[int, int] | None:
    """(start, end) of the content of the last \\boxed{...}, read as bank.common.extract_boxed
    reads it. None if there is no box or its braces do not close."""
    start = text.rfind("\\boxed{")
    if start == -1:
        return None
    start += len("\\boxed{")
    depth = 1
    for i in range(start, len(text)):
        depth += {"{": 1, "}": -1}.get(text[i], 0)
        if depth == 0:
            return start, i
    return None


def swap_last_box(text: str, answer: str) -> str:
    """The text with only the content of its last \\boxed{} replaced."""
    span = last_box_span(text)
    if span is None:
        raise ValueError("no closed \\boxed{} to replace")
    return text[:span[0]] + answer + text[span[1]:]


def answer_values(content: str) -> set[Fraction]:
    """The numbers the two label readers see in a box (GSM8K's and MATH's reading)."""
    return {Fraction(v) for v in (parse_boxed_number(content), parse_rational(content)) if v is not None}


def same_answer(a: str, b: str) -> bool:
    """The same text apart from spaces and case, or the same number under either reading."""
    if "".join(normalize_text(a).split()) == "".join(normalize_text(b).split()):
        return True
    return bool(answer_values(a) & answer_values(b))


def make_swap(original: dict, partner: dict, reference: str | None) -> tuple[str | None, str | None]:
    """(swapped text, None) or (None, why this candidate cannot be used)."""
    if original["domain"] == "code":
        if normalize_text(original["candidate"]) == normalize_text(partner["candidate"]):
            return None, "partner program is the same"
        return partner["candidate"], None
    old, new = extract_boxed(original["candidate"]), extract_boxed(partner["candidate"])
    if old is None or not new:
        return None, "no readable \\boxed{} answer"
    if same_answer(old, new):
        return None, "swapped answer equals the original"
    if reference is not None and same_answer(reference, new):
        return None, "swapped answer equals the reference"
    text = swap_last_box(original["candidate"], new)
    if extract_boxed(text) != new or same_answer(extract_boxed(text), old):
        return None, "final answer did not change"
    return text, None


# ---------------------------------------------------------------- the draw

def draw(candidates: list[dict], n: int, seed: int, domain: str) -> list[dict]:
    """n candidates, one per problem before any problem is used twice. Candidates of one problem
    come from different generators (a pilot cell holds one matched pair)."""
    by_problem = defaultdict(list)
    for c in sorted(candidates, key=lambda c: c["candidate_id"]):
        by_problem[c["problem_id"]].append(c)
    rng = random.Random(f"{seed}:swap:{domain}")  # a string seed is stable across Python versions
    for group in by_problem.values():
        rng.shuffle(group)
    chosen = []
    while len(chosen) < n:
        problems = sorted(p for p, group in by_problem.items() if group)
        if not problems:
            raise SystemExit(f"{domain}: only {len(chosen)} usable candidates, {n} needed")
        rng.shuffle(problems)
        chosen += [by_problem[p].pop() for p in problems[:n - len(chosen)]]
    return chosen


def recorded_lengths(paths: list[str]) -> dict[str, int]:
    """candidate_id -> longest recorded prompt, from score files or folders of scores_[AB].jsonl."""
    longest: dict[str, int] = {}
    for path in map(Path, paths):
        for f in sorted(path.rglob("scores_[AB].jsonl")) if path.is_dir() else [path]:
            for r in read_jsonl(f):
                if r.get("input_tokens"):
                    longest[r["candidate_id"]] = max(longest.get(r["candidate_id"], 0), r["input_tokens"])
    return longest


def item_id(seed: int, pair_id: str, text: str) -> str:
    return "swap-" + sha256_text(f"{seed}:{pair_id}:{sha256_text(text)}")[:16]


def build(bank: list[dict], labels: dict[str, dict], lengths: dict[str, int] | None, seed: int = SEED,
          per_domain: int = PER_DOMAIN, max_tokens: int = MAX_PROMPT_TOKENS) -> tuple[list[dict], list[dict], dict]:
    """(scored rows, key rows, summary)."""
    missing = [r["candidate_id"] for r in bank if labels.get(r["candidate_id"], {}).get("label") not in (0, 1)]
    if missing:
        raise SystemExit(f"{len(missing)} bank rows have no 0/1 label, e.g. {missing[:3]}")
    pairs: dict[str, dict[int, dict]] = defaultdict(dict)
    for r in bank:
        pairs[r["pair_id"]][labels[r["candidate_id"]]["label"]] = r
    broken = sorted(p for p, members in pairs.items() if set(members) != {0, 1})
    if broken:
        raise SystemExit(f"{len(broken)} pairs do not hold one correct and one wrong answer, e.g. {broken[:3]}")
    if lengths is not None:
        unknown = [r["candidate_id"] for r in bank if r["candidate_id"] not in lengths]
        if unknown:
            raise SystemExit(f"--scores has no prompt length for {len(unknown)} candidates, e.g. {unknown[:3]}")

    rows, keys, summary = [], [], {"dropped": {}, "domains": {}}
    for domain in ("math", "code"):
        usable, swaps, longest = [], {}, 0
        for pair_id in sorted(pairs):
            good, bad = pairs[pair_id][1], pairs[pair_id][0]
            if good["domain"] != domain:
                continue
            text, reason = make_swap(good, bad, labels[good["candidate_id"]].get("reference"))
            if reason is None and lengths is not None:
                swapped = (lengths[bad["candidate_id"]] if domain == "code" else lengths[good["candidate_id"]]
                           + len(extract_boxed(bad["candidate"]).encode("utf-8")))
                bound = max(lengths[good["candidate_id"]], swapped)
                longest = max(longest, bound)
                if bound > max_tokens:
                    reason = f"prompt could exceed {max_tokens} tokens"
            if reason is None:
                usable.append(good)
                swaps[pair_id] = text
            else:
                summary["dropped"][good["candidate_id"]] = reason
        chosen = draw(usable, per_domain, seed, domain)
        for good in chosen:
            bad, text = pairs[good["pair_id"]][0], swaps[good["pair_id"]]
            for role, shown in (("original", good["candidate"]), ("swap", text)):
                cid = item_id(seed, good["pair_id"], shown)
                rows.append({"candidate_id": cid, "pair_id": f"swap:{good['pair_id']}",
                             **{k: good[k] for k in COPIED_FIELDS if k in good},
                             "candidate": shown, "text_sha256": sha256_text(shown), "split": good["split"]})
                keys.append({"candidate_id": cid, "pair_id": f"swap:{good['pair_id']}", "domain": domain,
                             "role": role, "source_candidate_id": good["candidate_id"],
                             "partner_candidate_id": bad["candidate_id"],
                             "final_answer": extract_boxed(shown) if domain == "math" else None})
        uses = Counter(g["problem_id"] for g in chosen)
        summary["domains"][domain] = {
            "usable": len(usable), "items": len(chosen), "problems": len(uses),
            "problems_used_twice_or_more": sorted(p for p, k in uses.items() if k > 1),
            "longest_prompt_bound": longest,
            **({"items_with_several_boxes": sum(g["candidate"].count("\\boxed{") > 1 for g in chosen)}
               if domain == "math" else {}),
        }
    ids = [r["candidate_id"] for r in rows]
    if len(set(ids)) != len(ids):
        raise SystemExit("two items share an ID")
    rows.sort(key=lambda r: r["candidate_id"])
    keys.sort(key=lambda r: r["candidate_id"])
    return rows, keys, summary


def sha256_file(path: str | Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--bank", default="bank/pilot/pilot_bank.jsonl")
    ap.add_argument("--labels", default="bank/pilot/pilot_labels.jsonl")
    ap.add_argument("--scores", nargs="+", default=["results/pilot/ar"],
                    help="score files or folders whose rows record input_tokens for every bank row")
    ap.add_argument("--out", default="bank/pilot/swap")
    ap.add_argument("--seed", type=int, default=SEED)
    ap.add_argument("--per-domain", type=int, default=PER_DOMAIN)
    ap.add_argument("--max-tokens", type=int, default=MAX_PROMPT_TOKENS)
    args = ap.parse_args(argv)

    labels = {r["candidate_id"]: r for r in read_jsonl(args.labels)}
    lengths = recorded_lengths(args.scores)
    rows, keys, summary = build(read_jsonl(args.bank), labels, lengths, args.seed, args.per_domain, args.max_tokens)
    out = Path(args.out)
    write_jsonl(out / ITEMS_FILENAME, rows)
    write_jsonl(out / KEY_FILENAME, keys)
    summary.update({
        "seed": args.seed, "per_domain": args.per_domain, "max_prompt_tokens": args.max_tokens,
        "inputs": {Path(p).as_posix(): sha256_file(p) for p in (args.bank, args.labels)},
        "lengths_from": args.scores,
        "outputs": {f: sha256_file(out / f) for f in (ITEMS_FILENAME, KEY_FILENAME)},
    })
    (out / MANIFEST_FILENAME).write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8", newline="\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
