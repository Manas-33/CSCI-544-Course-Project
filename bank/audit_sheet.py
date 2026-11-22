"""Blind label audit by Codex plus one person: sheets, scoring and the rule (PLAN.md §2).

The rule is in decisions/2026-11-21-label-check-codex-plus-one.md. Four steps, for the pilot
or the full bank (give --bank and --labels as many files as needed):

    python -m bank.audit_sheet sample --bank bank/pilot/pilot_bank.jsonl \\
        --labels bank/pilot/pilot_labels.jsonl --out bank/pilot/audit
    python -m bank.codex_audit --sheet bank/pilot/audit/audit_sheet.jsonl --out bank/pilot/audit/codex
    python -m bank.audit_sheet human --bank ... --labels ... --audit bank/pilot/audit --read 100
    python -m bank.audit_sheet score --bank ... --labels ... --audit bank/pilot/audit

- sample draws 200 candidates: 50 for each of math/code x correct/wrong, seed 544, never both
  answers of a matched pair (seeing both would give the label away). The sheet holds an opaque
  ID, the problem and the candidate. The ID key is a separate file. --exclude-key leaves out
  the candidates of an earlier audit, for a re-audit on fresh candidates.
- human writes what the person reads: every item where Codex disagrees with our label or gave
  no usable verdict, every item written by a GPT model (Codex is one), and 100 others drawn at
  random, all shuffled together. --read all adds a sheet with everything not read yet. The first call keeps a copy of Codex's answers, so a later Codex run cannot change what
  is read. The person fills human_answers_<k>.csv.
- score applies the rule: PASS, FAIL, READ ALL, or that the person's second look comes first
  (--recheck; see the record). With 100 others read it passes only if none of them is a label
  error, at most 3 errors were found in all and the bound is below 4%.
- table prints the bound for typical cases and how often the rule passes a bad bank.

decisions/2026-11-22-label-check-by-domain.md judges the pilot per domain, and checks math only
on the full bank:

    python -m bank.audit_sheet human --bank ... --labels ... --audit bank/pilot/audit \\
        --only math code:evalplus/humaneval+
    python -m bank.audit_sheet score --bank ... --labels ... --audit bank/pilot/audit --by-domain
    python -m bank.audit_sheet sample --bank ... --labels ... --out ... --domain math --per-stratum 100

- human --only writes a sheet with every item of these domains (or domain:source) not read yet.
- score --by-domain writes audit_result_by_domain.json and leaves audit_result.json alone. Math
  passes only if all 100 items are read and none is a label error. HumanEval+ gets a rate and no
  verdict. MBPP+ is not counted.
- sample --domain draws from one domain; the steps above then apply to its 200 items unchanged.

On the full bank every file except the sheet (the key, Codex's replies, the answers and the
result) gives labels away. They stay with the label holder and are not committed before the
test join.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import random
from collections import defaultdict
from functools import lru_cache
from pathlib import Path

from bank.common import read_jsonl, write_jsonl

SEED = 544
PER_STRATUM = 50
DOMAINS = ("math", "code")
SHEET, KEY, MANIFEST = "audit_sheet.jsonl", "audit_key.jsonl", "audit_manifest.json"
CODEX_ANSWERS = "codex/codex_answers.jsonl"
FROZEN_ANSWERS = "codex_answers_used.jsonl"   # copied when the first sheet is written, and used from then on
SAME_FAMILY = "openai/"          # generators of Codex's own family; their items are always read
OTHERS = 100                     # others read, drawn at random, before the shortcut can pass
MAX_ERRORS, BAD_RATE, ALPHA = 3, 0.04, 0.05
VERDICTS = ("correct", "wrong", "unsure")
RESULT, RESULT_BY_DOMAIN = "audit_result.json", "audit_result_by_domain.json"
POOLED_ITEMS = 200                               # what MAX_ERRORS is counted in
MATH_ITEMS, MATH_MAX_ERRORS = 100, 0             # the per-domain mark: all read, none a label error
RATE_ONLY, NOT_COUNTED = "evalplus/humaneval+", "evalplus/mbpp+"     # the two code sources

QUESTIONS = {
    "math": ("Solve the problem yourself first. Then judge only the candidate's final answer, the content of its "
             "last \\boxed{}: \"correct\" if it equals the right answer in value (units, formatting and the steps "
             "before it do not matter), otherwise \"wrong\"."),
    "code": ("Judge whether the candidate program solves the problem: \"correct\" if it gives the right result for "
             "every input the problem allows, edge cases included, and \"wrong\" if some allowed input gives a wrong "
             "result, an exception or an endless run. Style and efficiency do not matter."),
}


def sha256_file(path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_json(path, value) -> None:
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8", newline="\n")


def load_bank(bank_paths: list[str], label_paths: list[str]) -> tuple[dict[str, dict], dict[str, int]]:
    """(candidate_id -> bank row, candidate_id -> 0/1 label) from one or more files of each."""
    rows = [r for p in bank_paths for r in read_jsonl(p)]
    labels = [r for p in label_paths for r in read_jsonl(p)]
    for name, items in (("bank", rows), ("label", labels)):
        if len({r["candidate_id"] for r in items}) != len(items):
            raise SystemExit(f"a candidate_id appears twice in the {name} files")
    by_id = {r["candidate_id"]: r for r in rows}
    label_of = {r["candidate_id"]: r["label"] for r in labels if r["candidate_id"] in by_id}
    missing = sorted(c for c in by_id if label_of.get(c) not in (0, 1))
    if missing:
        raise SystemExit(f"{len(missing)} bank rows have no 0/1 label, e.g. {missing[:3]}")
    return by_id, label_of


# ---------------------------------------------------------------- the 200-candidate sheet

def sample(bank: dict[str, dict], labels: dict[str, int], seed: int = SEED, per_stratum: int = PER_STRATUM,
           exclude: set[str] = frozenset(), domains: tuple[str, ...] = DOMAINS) -> list[str]:
    """Candidate IDs in sheet order. Within a domain, pairs come in a seeded random order and give
    one answer each, so every candidate is equally likely and no pair shows both its answers."""
    pairs: dict[tuple[str, str], dict[int, str]] = defaultdict(dict)
    for cid in sorted(bank):
        pairs[(bank[cid]["domain"], bank[cid]["pair_id"])][labels[cid]] = cid
    chosen = []
    for domain in domains:
        rng = random.Random(f"{seed}:audit:{domain}")
        order = sorted(k for k in pairs if k[0] == domain)
        rng.shuffle(order)
        left = {0: per_stratum, 1: per_stratum}
        for key in order:
            side = rng.randrange(2)
            for label in (side, 1 - side):
                cid = pairs[key].get(label)
                if left[label] and cid is not None and cid not in exclude:
                    chosen.append(cid)
                    left[label] -= 1
                    break
        if any(left.values()):
            raise SystemExit(f"{domain}: short of {left[1]} correct and {left[0]} wrong candidates")
    random.Random(f"{seed}:audit:order").shuffle(chosen)
    return chosen


def sheet_rows(bank: dict[str, dict], chosen: list[str]) -> tuple[list[dict], list[dict]]:
    sheet, key = [], []
    for i, cid in enumerate(chosen, 1):
        audit_id = f"a{i:03d}"
        row = bank[cid]
        sheet.append({"audit_id": audit_id, "domain": row["domain"], "problem": row["problem"],
                      "candidate": row["candidate"]})
        key.append({"audit_id": audit_id, "candidate_id": cid})
    return sheet, key


def cmd_sample(args) -> None:
    bank, labels = load_bank(args.bank, args.labels)
    exclude = {r["candidate_id"] for p in args.exclude_key for r in read_jsonl(p)}
    domains = (args.domain,) if args.domain else DOMAINS
    sheet, key = sheet_rows(bank, sample(bank, labels, args.seed, args.per_stratum, exclude, domains))
    out = Path(args.out)
    if (out / SHEET).exists() and not args.overwrite:
        raise SystemExit(f"{out / SHEET} exists; a new audit needs a new --out (or --overwrite)")
    write_jsonl(out / SHEET, sheet)
    write_jsonl(out / KEY, key)
    write_json(out / MANIFEST, {
        "seed": args.seed, "per_stratum": args.per_stratum, **({"domain": args.domain} if args.domain else {}),
        "items": len(sheet), "excluded_candidates": len(exclude),
        "inputs": {Path(p).as_posix(): sha256_file(p) for p in [*args.bank, *args.labels, *args.exclude_key]},
        "outputs": {f: sha256_file(out / f) for f in (SHEET, KEY)},
    })
    print(f"{len(sheet)} items in {out / SHEET}; the key is {KEY} (not for the readers)")


# ---------------------------------------------------------------- what the person reads

def load_audit(audit: Path) -> tuple[list[dict], dict[str, str]]:
    sheet, key = read_jsonl(audit / SHEET), {r["audit_id"]: r["candidate_id"] for r in read_jsonl(audit / KEY)}
    if [r["audit_id"] for r in sheet] != list(key):
        raise SystemExit("the sheet and its key do not list the same items")
    return sheet, key


def load_codex(path, sheet: list[dict]) -> dict[str, str | None]:
    """audit_id -> Codex's verdict, or None where it gave no usable one (failed, invalid or unsure)."""
    rows = {r["audit_id"]: r for r in read_jsonl(path)}
    missing = [r["audit_id"] for r in sheet if rows.get(r["audit_id"], {}).get("status", "missing") == "missing"]
    if missing or len(rows) != len(sheet):
        raise SystemExit(f"{path} does not cover the sheet ({len(missing)} items missing); finish bank.codex_audit first")
    return {a: r["verdict"] if r["status"] == "ok" and r["verdict"] in ("correct", "wrong") else None
            for a, r in rows.items()}


def codex_file(args, audit: Path) -> Path:
    frozen = audit / FROZEN_ANSWERS
    return frozen if frozen.exists() else Path(args.codex or audit / CODEX_ANSWERS)


def groups(sheet, key, bank, labels, codex, same_family: str, seed: int) -> dict[str, list[str]]:
    """read_all = disagreements, items without a Codex verdict and same-family items; rest = the
    others, in the seeded order they are read."""
    disagree, no_verdict, family, rest = [], [], [], []
    for row in sheet:
        a, cid = row["audit_id"], key[row["audit_id"]]
        if codex[a] is None:
            no_verdict.append(a)
        elif codex[a] != ("correct" if labels[cid] else "wrong"):
            disagree.append(a)
        elif same_family and bank[cid]["generator"].lower().startswith(same_family.lower()):
            family.append(a)
        else:
            rest.append(a)
    random.Random(f"{seed}:audit:rest").shuffle(rest)
    return {"disagree": disagree, "no_verdict": no_verdict, "same_family": family,
            "read_all": disagree + no_verdict + family, "rest": rest}


def fenced(text: str, language: str = "") -> str:
    """A code block whose fence is longer than any run of backticks in the text."""
    longest = run = 0
    for ch in text:
        run = run + 1 if ch == "`" else 0
        longest = max(longest, run)
    fence = "`" * max(3, longest + 1)
    return f"{fence}{language}\n{text}\n{fence}"


def human_sheet(rows: list[dict], number: int) -> str:
    out = [f"# Label check, sheet {number} ({len(rows)} items)", "",
           f"Write one verdict per item in `human_answers_{number}.csv`: correct, wrong or unsure. "
           "Work alone and do not open any other file of this audit, the bank or its labels. "
           "You may use a calculator and run a program you type or paste.", "",
           f"- **Math.** {QUESTIONS['math']}", f"- **Code.** {QUESTIONS['code']}", ""]
    for r in rows:
        out += [f"## {r['audit_id']} ({r['domain']})", "", "**Problem**", "", fenced(r["problem"]), "",
                "**Candidate**", "", fenced(r["candidate"], "python" if r["domain"] == "code" else ""), ""]
    return "\n".join(out)


def answer_files(audit: Path) -> list[Path]:
    return sorted(audit.glob("human_answers_*.csv"), key=lambda p: int(p.stem.rsplit("_", 1)[1]))


def read_answers(path: Path) -> list[dict]:
    with open(path, encoding="utf-8-sig", newline="") as f:
        return [{k.strip().lower(): (v or "").strip() for k, v in row.items() if k} for row in csv.DictReader(f)]


def chosen_items(sheet, key, bank, only: list[str]) -> list[str]:
    """audit_ids of the items of these domains, or domain:source, in sheet order."""
    names = {r["audit_id"]: (r["domain"], f"{r['domain']}:{bank[key[r['audit_id']]].get('source')}") for r in sheet}
    unknown = [o for o in only if not any(o in n for n in names.values())]
    if unknown:
        raise SystemExit(f"no item of the sheet is of {', '.join(unknown)}; "
                         f"it holds {', '.join(sorted({x for n in names.values() for x in n}))}")
    return [a for a, n in names.items() if set(n) & set(only)]


def cmd_human(args) -> None:
    audit = Path(args.audit)
    bank, labels = load_bank(args.bank, args.labels)
    sheet, key = load_audit(audit)
    source = codex_file(args, audit)
    codex = load_codex(source, sheet)
    if source.name != FROZEN_ANSWERS:
        failed = sum(r["status"] == "failed" for r in read_jsonl(source))
        if failed and not args.allow_failed:
            raise SystemExit(f"{failed} Codex calls failed; run bank.codex_audit again to retry them, "
                             "or use --allow-failed to have the person read those items")
    if args.only:
        wanted = chosen_items(sheet, key, bank, args.only)
    else:
        g = groups(sheet, key, bank, labels, codex, args.same_family, args.seed)
        n = len(g["rest"]) if args.read == "all" else min(int(args.read), len(g["rest"]))
        wanted = g["read_all"] + g["rest"][:n]
    earlier = [r["audit_id"] for f in answer_files(audit) for r in read_answers(f)]
    new = [a for a in wanted if a not in set(earlier)]
    if not new:
        raise SystemExit("nothing new to read")
    if source.name != FROZEN_ANSWERS:
        (audit / FROZEN_ANSWERS).write_bytes(source.read_bytes())
    number = len(answer_files(audit)) + 1
    random.Random(f"{args.seed}:audit:human:{number}").shuffle(new)
    by_id = {r["audit_id"]: r for r in sheet}
    (audit / f"human_sheet_{number}.md").write_text(human_sheet([by_id[a] for a in new], number),
                                                    encoding="utf-8", newline="\n")
    with open(audit / f"human_answers_{number}.csv", "w", encoding="utf-8", newline="") as f:
        w = csv.writer(f, lineterminator="\n")
        w.writerow(["audit_id", "verdict", "note"])
        w.writerows([a, "", ""] for a in new)
    print(f"sheet {number}: {len(new)} items ({len(earlier)} on earlier sheets)")  # no counts per group: the reader may see this


# ---------------------------------------------------------------- the bound

@lru_cache(maxsize=None)
def pmf(n: int, p: float, upto: int) -> tuple[float, ...]:
    """Binomial probabilities of 0..min(upto, n)."""
    upto = min(upto, n)
    if p <= 0 or p >= 1:
        hit = 0 if p <= 0 else n
        return tuple(1.0 if k == hit else 0.0 for k in range(upto + 1))
    lp, lq = math.log(p), math.log1p(-p)
    return tuple(math.exp(math.lgamma(n + 1) - math.lgamma(k + 1) - math.lgamma(n - k + 1) + k * lp + (n - k) * lq)
                 for k in range(upto + 1))


@lru_cache(maxsize=None)
def p_value(n_all: int, n_rest: int, read: int, x: int, k: int, rate: float = BAD_RATE, grid: int = 200) -> float:
    """The largest chance, over every bank with this error rate, of data that look this good.

    n_all items are read in full and hold x errors; `read` of the n_rest others are read and hold
    k. "Looks this good" means that the estimated number of errors in the sample, x + k *
    n_rest / read, is no larger than the one seen. The bank is free to put its errors in either
    group: the rates in the two groups only have to average to `rate`. With everything read, or
    with no read-in-full group, this is the ordinary exact (Clopper-Pearson) test."""
    n = n_all + n_rest
    scale = n_rest / read if read else 0.0
    if n_rest and not read:
        return 1.0
    seen = x + k * scale + 1e-9
    total = rate * n
    lo = max(0.0, (total - n_rest) / n_all) if n_all else 0.0
    hi = min(1.0, total / n_all) if n_all else 0.0
    best = 0.0
    for i in range(grid + 1 if n_all else 1):
        p_all = lo + (hi - lo) * i / grid
        p_rest = min(1.0, max(0.0, (total - n_all * p_all) / n_rest)) if n_rest else 0.0
        chance = 0.0
        rest = pmf(read, p_rest, int(seen / scale)) if n_rest else (1.0,)
        for j, a in enumerate(pmf(n_all, p_all, int(seen))):
            chance += a * sum(rest[:int((seen - j) / scale) + 1] if n_rest else rest)
        best = max(best, chance)
    return best


def upper_bound(n_all: int, n_rest: int, read: int, x: int, k: int, alpha: float = ALPHA) -> float:
    """One-sided 95% upper bound on the bank's label-error rate."""
    lo, hi = 0.0, 1.0
    for _ in range(30):
        mid = (lo + hi) / 2
        lo, hi = (mid, hi) if p_value(n_all, n_rest, read, x, k, mid) > alpha else (lo, mid)
    return hi


def bound_ok(n_all: int, n_rest: int, read: int, x: int, k: int) -> bool:
    return p_value(n_all, n_rest, read, x, k) <= ALPHA


# ---------------------------------------------------------------- the decision

def decide(n_all: int, n_rest: int, x: int, read: int, k: int, others: int = OTHERS) -> str:
    """The rule: PASS, FAIL or READ ALL, with x errors among the n_all items read in full and k
    among the `read` others. The shortcut can pass but never fail."""
    if read >= n_rest:                                   # all 200 read: PLAN.md's own rule
        return "PASS" if x + k <= MAX_ERRORS else "FAIL"
    if read == others and k == 0 and x <= MAX_ERRORS and bound_ok(n_all, n_rest, read, x, 0):
        return "PASS"
    return "READ ALL"


def hypergeometric(total: int, marked: int, draws: int, hit: int) -> float:
    if hit > marked or draws - hit > total - marked:
        return 0.0
    return math.comb(marked, hit) * math.comb(total - marked, draws - hit) / math.comb(total, draws)


@lru_cache(maxsize=None)
def outcome_chances(n_all: int, n_rest: int, x: int, e: int, others: int) -> tuple[float, float, float]:
    """(passes by the shortcut, passes at all, expected items read), with e errors among the others."""
    n, full = n_all + n_rest, float(x + e <= MAX_ERRORS)
    if n_rest <= others:
        return 0.0, full, float(n)
    short = hypergeometric(n_rest, e, others, 0) if decide(n_all, n_rest, x, others, 0, others) == "PASS" else 0.0
    return short, short + (1 - short) * full, short * (n_all + others) + (1 - short) * n


def operating(q_all: float, q_rest: float, ok_all: float, others: int = OTHERS, n: int = 200) -> dict[str, float]:
    """Exact chances for a bank where an item is a label error in the read-in-full group with
    probability q_all, an error among the others with q_rest, and a good label in the read-in-full
    group with ok_all (wrong Codex verdicts, GPT-written items)."""
    short = overall = read = plan = 0.0
    for x, a in enumerate(pmf(n, q_all, 12)):
        for good, b in enumerate(pmf(n - x, ok_all / (1 - q_all), n)):
            if a * b < 1e-9:
                continue
            n_rest = n - x - good
            for e, c in enumerate(pmf(n_rest, q_rest / (1 - q_all - ok_all), 25)):
                w = a * b * c
                if w < 1e-9:
                    continue
                s, o, r = outcome_chances(x + good, n_rest, x, e, others)
                short, overall, read, plan = short + w * s, overall + w * o, read + w * r, plan + w * (x + e <= MAX_ERRORS)
    return {"shortcut_pass": short, "pass": overall, "read": read, "plan_pass": plan}


def worst_false_pass(ok_all: float, others: int = OTHERS, rate: float = BAD_RATE) -> dict[str, float]:
    """The largest pass chances for a bank at exactly `rate`, over where its errors sit."""
    runs = [operating(q, rate - q, ok_all, others) for q in (0, .002, .004, .006, .008, .01, .015, .02, .03)]
    return {k: max(r[k] for r in runs) for k in ("shortcut_pass", "pass", "plan_pass")}


def table_text(others: int) -> str:
    lines = [f"95% upper bound on the label-error rate, {others} others read and no error among them (* passes, below 4%)", "",
             "| Read in full | Items read | 0 errors there | 1 | 2 | 3 |", "|---|---|---|---|---|---|"]
    for n_all in (13, 20, 30, 40, 60, 80):
        n_rest = 200 - n_all
        cells = [f"{upper_bound(n_all, n_rest, others, x, 0):.2%}" + ("*" if bound_ok(n_all, n_rest, others, x, 0) else "")
                 for x in range(4)]
        lines.append(f"| {n_all} | {n_all + others} | " + " | ".join(cells) + " |")
    lines += ["", f"Chance of passing a bank at exactly 4% label errors, worst placement of its errors ({others} others, then all)", "",
              "| Good items read in full | Shortcut passes | Shortcut or full read passes | PLAN's full read alone |", "|---|---|---|---|"]
    for ok in (0, .05, .10, .15, .20, .30):
        w = worst_false_pass(ok, others)
        lines.append(f"| {200 * ok:.0f} | {w['shortcut_pass']:.1%} | {w['pass']:.1%} | {w['plan_pass']:.1%} |")
    lines += ["", "Chance of passing, and items read, when 20 good items are read in full and Codex catches 3 of 4 label errors", "",
              "| True label-error rate | Shortcut passes | Passes at all | PLAN's full read alone | Average items read |", "|---|---|---|---|---|"]
    for rate in (0, .0025, .005, .01, .015, .02):
        o = operating(.75 * rate, .25 * rate, .10, others)
        lines.append(f"| {rate:.2%} | {o['shortcut_pass']:.0%} | {o['pass']:.0%} | {o['plan_pass']:.0%} | {o['read']:.0f} |")
    return "\n".join(lines)


# ---------------------------------------------------------------- scoring

def collect_answers(audit: Path, sheet: list[dict]) -> dict[str, dict]:
    files = answer_files(audit)
    if not files:
        raise SystemExit(f"no human_answers_*.csv in {audit}; run the human step first")
    known, out, problems = {r["audit_id"] for r in sheet}, {}, []
    for path in files:
        for r in read_answers(path):
            a, verdict = r.get("audit_id", ""), r.get("verdict", "").lower()
            if a not in known or a in out:
                problems.append(f"{path.name}: unknown or repeated item {a!r}")
            elif verdict not in VERDICTS:
                problems.append(f"{path.name}: {a} has no verdict (correct, wrong or unsure)")
            else:
                out[a] = {"verdict": verdict, "note": r.get("note", "")}
    if problems:
        raise SystemExit("answers are not complete:\n" + "\n".join(problems[:20]))
    return out


def apply_recheck(path, differ: set[str]) -> dict[str, dict]:
    """The reader's second look at items where their blind verdict differs from our label."""
    out = {}
    for r in read_answers(Path(path)) if path else []:
        a, error = r.get("audit_id", ""), r.get("label_error", "").lower()
        if a not in differ or error not in ("yes", "no") or not r.get("reason"):
            raise SystemExit(f"{path}: {a!r} must be an item the reader and our label differ on, "
                             "with label_error yes or no and a reason")
        out[a] = {"label_error": error == "yes", "reason": r["reason"]}
    return out


def score(sheet, key, bank, labels, codex, answers, recheck, same_family=SAME_FAMILY, seed=SEED) -> dict:
    g = groups(sheet, key, bank, labels, codex, same_family, seed)
    unread = [a for a in g["read_all"] if a not in answers]
    if unread:
        raise SystemExit(f"{len(unread)} items that must be read have no answer")
    read = sum(a in answers for a in g["rest"])
    if any(a not in answers for a in g["rest"][:read]):
        raise SystemExit("the others were not read in their drawn order")
    if read not in (OTHERS, len(g["rest"])):
        raise SystemExit(f"{read} others were read; the rule needs {OTHERS} or all {len(g['rest'])}")
    expected = {a: "correct" if labels[key[a]] else "wrong" for a in answers}
    differ = {a for a, r in answers.items() if r["verdict"] != expected[a]}
    second_look = recheck(differ)
    errors = {a for a in differ if second_look.get(a, {"label_error": True})["label_error"]}
    n_all, n_rest = len(g["read_all"]), len(g["rest"])
    x = sum(a in errors for a in g["read_all"])
    k = sum(a in errors for a in g["rest"][:read])
    decision = decide(n_all, n_rest, x, read, k)
    waiting = sorted(differ - set(second_look))
    if decision != "PASS" and waiting:                   # before more reading, or a fail, is called for
        decision = "SECOND LOOK"
    elif decision in ("PASS", "FAIL"):
        decision += " (all items read)" if read == n_rest else " (shortcut)"
    return {
        "decision": decision,
        "items": len(sheet), "read": len(answers), "read_in_full_group": n_all, "others": n_rest, "others_read": read,
        "groups": {name: len(g[name]) for name in ("disagree", "no_verdict", "same_family", "rest")},
        "label_errors_found": x + k,
        "errors_by_group": {**{name: sum(a in errors for a in g[name]) for name in ("disagree", "no_verdict", "same_family")},
                            "others_read": k},
        "reader_differs_blind": len(differ), "second_looks": len(second_look), "waiting_for_second_look": waiting,
        "upper_bound_95": upper_bound(n_all, n_rest, read, x, k),
        "rule": {"max_errors": MAX_ERRORS, "bound_below": BAD_RATE, "others_read_first": OTHERS},
        "differences": [{"audit_id": a, "candidate_id": key[a], "our_label": expected[a], "codex": codex[a],
                         "reader": answers[a]["verdict"], "note": answers[a]["note"],
                         "label_error": a in errors, "second_look": second_look.get(a, {}).get("reason")}
                        for a in sorted(differ)],
        "codex_agreement": {d: _agreement(sheet, key, labels, codex, d) for d in DOMAINS},
    }


def _agreement(sheet, key, labels, codex, domain) -> dict:
    ids = [r["audit_id"] for r in sheet if r["domain"] == domain]
    with_verdict = [a for a in ids if codex[a] is not None]
    return {"items": len(ids), "with_verdict": len(with_verdict),
            "agrees_with_label": sum(codex[a] == ("correct" if labels[key[a]] else "wrong") for a in with_verdict)}


def score_by_domain(sheet, key, bank, labels, answers, recheck) -> dict:
    """The per-domain rule. recheck may cover any item the reader and our label differ on."""
    part: dict[str, list[str]] = {"math": [], RATE_ONLY: [], NOT_COUNTED: []}
    for r in sheet:
        row = bank[key[r["audit_id"]]]
        name = "math" if row["domain"] == "math" else row.get("source")
        if name not in part:
            raise SystemExit(f"the per-domain rule has no place for source {name!r}")
        part[name].append(r["audit_id"])
    if len(part["math"]) != MATH_ITEMS:
        raise SystemExit(f"the sheet holds {len(part['math'])} math items; the per-domain mark is for {MATH_ITEMS}")
    unread = sum(a not in answers for a in part["math"])
    if unread:
        raise SystemExit(f"{unread} math items have no answer; the per-domain rule needs all {MATH_ITEMS}")
    expected = {a: "correct" if labels[key[a]] else "wrong" for a in answers}
    differ = {a for a, r in answers.items() if r["verdict"] != expected[a]}
    second_look = recheck(differ)
    errors = {a for a in differ if second_look.get(a, {"label_error": True})["label_error"]}

    def counts(ids: list[str]) -> dict:
        read = [a for a in ids if a in answers]
        found = sum(a in errors for a in read)
        # the bound is for a random sample: none while the items left out were not left out at random
        whole = bool(ids) and len(read) == len(ids)
        return {"items": len(ids), "read": len(read), "label_errors": found, "rate": found / len(read) if whole else None,
                "upper_bound_95": upper_bound(0, len(ids), len(ids), 0, found) if whole else None,
                "reader_differs_blind": sum(a in differ for a in read),
                "waiting_for_second_look": sorted(a for a in read if a in differ and a not in second_look)}

    math_part, code_part = counts(part["math"]), counts(part[RATE_ONLY])
    decision = "PASS" if math_part["label_errors"] <= MATH_MAX_ERRORS else "FAIL"
    if decision != "PASS" and math_part["waiting_for_second_look"]:          # before a fail is called
        decision = "SECOND LOOK"
    return {
        "math": {"decision": decision, **math_part},
        "code": {RATE_ONLY: {"decision": None, **code_part}, NOT_COUNTED: "not counted"},
        "rule": {"math_items": MATH_ITEMS, "math_max_errors": MATH_MAX_ERRORS},
        "differences": [{"audit_id": a, "candidate_id": key[a], "our_label": expected[a], "reader": answers[a]["verdict"],
                         "note": answers[a]["note"], "label_error": a in errors,
                         "second_look": second_look.get(a, {}).get("reason")}
                        for a in sorted(differ) if a not in part[NOT_COUNTED]],
    }


def cmd_score(args) -> None:
    audit = Path(args.audit)
    bank, labels = load_bank(args.bank, args.labels)
    sheet, key = load_audit(audit)
    if args.by_domain:
        result = score_by_domain(sheet, key, bank, labels, collect_answers(audit, sheet),
                                 lambda differ: apply_recheck(args.recheck, differ))
        write_json(audit / RESULT_BY_DOMAIN, result)
        print(json.dumps({k: v for k, v in result.items() if k != "differences"}, indent=2))
        if result["math"]["decision"] == "SECOND LOOK":
            print("Second look first: for each math item in waiting_for_second_look the reader writes label_error "
                  "(yes or no) and a reason, then score again with --recheck.")
        if result["code"][RATE_ONLY]["waiting_for_second_look"]:
            print(f"The {RATE_ONLY} rate counts the items waiting for a second look as errors.")
        return
    if len({r["domain"] for r in sheet}) == 1 and len(sheet) != POOLED_ITEMS:
        raise SystemExit(f"the sheet holds {len(sheet)} items of one domain; the rule is for {POOLED_ITEMS} "
                         "(for the pilot's math re-audit use --by-domain)")
    result = score(sheet, key, bank, labels, load_codex(codex_file(args, audit), sheet),
                   collect_answers(audit, sheet), lambda differ: apply_recheck(args.recheck, differ),
                   args.same_family, args.seed)
    write_json(audit / RESULT, result)
    print(json.dumps({k: v for k, v in result.items() if k != "differences"}, indent=2))
    if result["decision"] == "SECOND LOOK":
        print("Second look first: for each item in waiting_for_second_look the reader writes label_error (yes or no) "
              "and a reason, then score again with --recheck. No more reading until then.")
    if result["decision"] == "READ ALL":
        print("The shortcut does not pass: run the human step with --read all and read the rest.")


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = ap.add_subparsers(dest="command", required=True)
    for name, func, text in (("sample", cmd_sample, "draw the 200-candidate sheet"),
                             ("human", cmd_human, "write the next sheet for the person"),
                             ("score", cmd_score, "apply the rule to the answers")):
        p = sub.add_parser(name, help=text)
        p.set_defaults(func=func)
        p.add_argument("--bank", nargs="+", required=True)
        p.add_argument("--labels", nargs="+", required=True)
        p.add_argument("--seed", type=int, default=SEED)
        if name == "sample":
            p.add_argument("--out", required=True)
            p.add_argument("--per-stratum", type=int, default=PER_STRATUM)
            p.add_argument("--domain", choices=DOMAINS, default=None, help="draw from this domain only")
            p.add_argument("--exclude-key", nargs="*", default=[], help="audit_key.jsonl of earlier audits")
            p.add_argument("--overwrite", action="store_true")
        else:
            p.add_argument("--audit", required=True, help="the folder sample wrote")
            p.add_argument("--codex", default=None, help=f"default: <audit>/{CODEX_ANSWERS}")
            p.add_argument("--same-family", default=SAME_FAMILY, help="generator prefix whose items are always read")
        if name == "human":
            which = p.add_mutually_exclusive_group(required=True)
            which.add_argument("--read", choices=(str(OTHERS), "all"), help="others to read in total")
            which.add_argument("--only", nargs="+", metavar="DOMAIN[:SOURCE]", help="every unread item of these")
            p.add_argument("--allow-failed", action="store_true", help="have the person read items whose Codex call failed")
        if name == "score":
            p.add_argument("--by-domain", action="store_true", help=f"the per-domain rule; writes {RESULT_BY_DOMAIN}")
            p.add_argument("--recheck", default=None, help="CSV audit_id,label_error,reason: the reader's second look")
    p = sub.add_parser("table", help="the bound for typical cases, and how often the rule passes a bad bank")
    p.add_argument("--others", type=int, default=OTHERS, help="how many others are read before the shortcut can pass")
    p.set_defaults(func=lambda a: print(table_text(a.others)))
    args = ap.parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()
