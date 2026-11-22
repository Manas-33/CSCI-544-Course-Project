"""Step 0 of the go/no-go: the answer-swap test and the AR repeat-run check (PLAN.md §4, §7).

    python -m analysis.step0 --swap-scores results/step0/swap \\
        --rerun results/step0/rerun_1 results/step0/rerun_2 --out results/step0

Answer swap (scoring-code test 4, T1, per verifier and domain): the share of items whose swap
scores strictly lower than the original in the same run. A tie counts as not lowered. A missing
or failed score counts as not lowered too, and is reported; if such scores could still change
the outcome, the outcome is "incomplete" (rescore), not "fail". Pass: at least 80% of items.
Math decides for the go/no-go; code is reported only.

Repeat runs (scoring-code test 3, AR models): the pilot bank scored twice with T1 in the same
session. Pass: Spearman above 0.999 over the 500 answers. Math and code, and the agreement with
the October 27 pilot scores, are reported as information only.

--swap-scores, --rerun and --previous take score files or folders searched for scores_A.jsonl,
scores_B.jsonl (harness.run) and *_T1_n*.jsonl (harness.diffusion.run_score). Every file must
hold T1 scores for the exact item file or bank, and a verifier may appear in only one file per
run, so an original and its swap always come from the same run. Verifiers that have not been
scored yet are listed as "not run".

Writes step0.json and step0.md.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

from analysis.metrics import spearman
from analysis.pilot import VERIFIERS, read_jsonl

SWAP_PASS = (80, 100)   # at least 80 of 100 items, compared as integers
RERUN_PASS = 0.999
H1_VERIFIERS = ("Qwen2.5-7B", "Llama-3.1-8B", "Gemma-2-9B", "LLaDA-8B")
OTHERS = ("Mistral-7B", "Llama-3.2-3B", "Qwen2.5-1.5B", "Dream-7B")
DOMAINS = ("math", "code")
KIND = {name: kind for name, _, kind, _ in VERIFIERS.values()}
AR_VERIFIERS = tuple(v for v in H1_VERIFIERS + OTHERS if KIND[v] == "AR")


# ---------------------------------------------------------------- loading

def score_files(paths) -> list[Path]:
    out = []
    for p in map(Path, paths):
        if p.is_dir():
            out += sorted(p.rglob("scores_[AB].jsonl")) + sorted(p.rglob("*_T1_n*.jsonl"))
        elif p.is_file():
            out.append(p)
        else:
            raise SystemExit(f"no such score file or folder: {p}")
    return [p for p in out if not p.name.endswith(".manifest.jsonl")]


def load_run(paths, bank_hash: str, ids: set[str]) -> dict[str, dict]:
    """verifier -> {"scores": candidate_id -> float or None (failed), "revision", "file"} for one run."""
    out: dict[str, dict] = {}
    for path in score_files(paths):
        for number, r in enumerate(read_jsonl(path), 1):
            key = r.get("model_id") or r.get("model")
            if key not in VERIFIERS:
                raise SystemExit(f"{path}: unknown model {key!r}")
            name = VERIFIERS[key][0]
            where = f"{path}, line {number} ({name})"
            if r.get("template") != "T1":
                raise SystemExit(f"{where}: template {r.get('template')!r}, expected T1")
            if (r.get("bank_sha256") or r.get("bank_hash")) != bank_hash:
                raise SystemExit(f"{where}: scored on another file (hash {r.get('bank_sha256') or r.get('bank_hash')})")
            if KIND[name] == "diffusion" and (r.get("n_slots"), r.get("answer_prefix", "none")) != (1, "none"):
                raise SystemExit(f"{where}: expected 1 answer slot and no answer prefix")
            run = out.setdefault(name, {"scores": {}, "revision": None, "file": path})
            if run["file"] != path:
                raise SystemExit(f"{name} was scored in two files of one run: {run['file']} and {path}")
            cid = r.get("candidate_id") or r.get("id")
            if cid not in ids:
                raise SystemExit(f"{where}: unexpected candidate {cid!r}")
            if cid in run["scores"]:
                raise SystemExit(f"{where}: candidate {cid!r} scored twice")
            score = r.get("score")
            ok = r.get("status") in ("ok", "completed") and isinstance(score, (int, float)) and math.isfinite(score)
            run["scores"][cid] = float(score) if ok else None
            run["revision"] = run["revision"] or r.get("model_revision") or r.get("revision")
    return out


def sha256_file(path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


# ---------------------------------------------------------------- answer swap

def wilson(k: int, n: int, z: float = 1.959963984540054) -> tuple[float, float]:
    """95% Wilson interval for a share."""
    if n == 0:
        return float("nan"), float("nan")
    p, d = k / n, 1 + z * z / n
    mid, half = (p + z * z / (2 * n)) / d, z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return max(0.0, mid - half), min(1.0, mid + half)


def swap_items(key_rows: list[dict], item_rows: list[dict]) -> dict[str, list[dict]]:
    """domain -> items, each {"original": candidate_id, "swap": candidate_id}; checked against the item file."""
    pairs: dict[str, dict] = defaultdict(dict)
    for k in key_rows:
        if k["role"] not in ("original", "swap") or k["role"] in pairs[k["pair_id"]]:
            raise SystemExit(f"swap key: pair {k['pair_id']!r} needs one original and one swap")
        pairs[k["pair_id"]][k["role"]] = k["candidate_id"]
        pairs[k["pair_id"]].setdefault("domain", k["domain"])
    if any({"original", "swap"} - p.keys() for p in pairs.values()):
        raise SystemExit("swap key: every pair needs one original and one swap")
    if Counter(k["candidate_id"] for k in key_rows) != Counter(r["candidate_id"] for r in item_rows):
        raise SystemExit("swap key and swap items do not list the same candidates")
    out: dict[str, list[dict]] = {d: [] for d in DOMAINS}
    for p in pairs.values():
        out[p["domain"]].append(p)
    return out


def swap_result(items: list[dict], scores: dict[str, float | None]) -> dict:
    """Counts and the outcome for one verifier in one domain."""
    lowered = tied = raised = unscored = 0
    for item in items:
        a, b = scores.get(item["original"]), scores.get(item["swap"])
        if a is None or b is None:
            unscored += 1
        elif b < a:
            lowered += 1
        elif b == a:
            tied += 1
        else:
            raised += 1
    n = len(items)
    need = lambda k: k * SWAP_PASS[1] >= SWAP_PASS[0] * n  # noqa: E731 - exact, no rounding
    outcome = "pass" if n and need(lowered) else "incomplete" if n and need(lowered + unscored) else "fail"
    lo, hi = wilson(lowered, n)
    return {"items": n, "lowered": lowered, "tied": tied, "raised": raised, "unscored": unscored,
            "share_lowered": lowered / n if n else None, "ci95": [lo, hi], "outcome": outcome}


def swap_test(items: dict[str, list[dict]], runs: dict[str, dict]) -> dict:
    out = {}
    for name in H1_VERIFIERS + OTHERS:
        run = runs.get(name)
        out[name] = {d: swap_result(items[d], run["scores"]) if run else {"outcome": "not run"} for d in DOMAINS}
        out[name]["revision"] = run["revision"] if run else None
    return out


# ---------------------------------------------------------------- repeat runs

def agreement(a: dict, b: dict, ids: list[str]) -> dict | None:
    """Spearman and differences over ids scored in both, or None if any is unscored."""
    pairs = [(a.get(i), b.get(i)) for i in ids]
    if not ids or any(x is None or y is None for x, y in pairs):
        return None
    x, y = (np.array(v, dtype=float) for v in zip(*pairs))
    rho = spearman(x, y)
    return {"n": len(ids), "spearman": rho if math.isfinite(rho) else None,
            "max_abs_difference": float(np.max(np.abs(x - y))), "identical": int(np.sum(x == y))}


def rerun_check(bank: list[dict], first: dict, second: dict, previous: dict) -> dict:
    ids = {"all": [r["candidate_id"] for r in bank],
           **{d: [r["candidate_id"] for r in bank if r["domain"] == d] for d in DOMAINS}}
    out = {}
    for name in AR_VERIFIERS:
        a, b, p = first.get(name), second.get(name), previous.get(name)
        if a is None and b is None:
            out[name] = {"outcome": "not run"}
            continue
        sa, sb = (a or {}).get("scores", {}), (b or {}).get("scores", {})
        both = {k: agreement(sa, sb, v) for k, v in ids.items()}
        unscored = {run: sum(s.get(i) is None for i in ids["all"]) for run, s in (("run_1", sa), ("run_2", sb))}
        if both["all"] is None:
            outcome = "incomplete"
        else:
            rho = both["all"]["spearman"]
            outcome = "pass" if rho is not None and rho > RERUN_PASS else "fail"
        out[name] = {
            "outcome": outcome, "unscored": unscored, "run_1_vs_run_2": both,
            "run_1_vs_october_27": {k: agreement(sa, p["scores"], v) for k, v in ids.items()} if p else None,
            "revisions": {"run_1": (a or {}).get("revision"), "run_2": (b or {}).get("revision"),
                          "october_27": (p or {}).get("revision")},
        }
    return out


# ---------------------------------------------------------------- report

def go_no_go(swap: dict, rerun: dict) -> dict:
    """What step 0 says about the four verifiers that can be in H1 (PLAN.md §7)."""
    return {name: {"answer_swap_math": swap[name]["math"]["outcome"],
                   "repeat_runs": rerun[name]["outcome"] if name in rerun else "not needed (passed in stage 3)"}
            for name in H1_VERIFIERS}


def pct(x) -> str:
    return "—" if x is None else f"{100 * x:.1f}%"


def rho_text(entry) -> str:
    return "—" if not entry or entry["spearman"] is None else f"{entry['spearman']:.5f}"


def diff_text(entry) -> str:
    return "—" if not entry else f"{entry['max_abs_difference']:.3f}"


def markdown(result: dict) -> str:
    lines = ["# Step 0: answer swap and repeat runs", "",
             "Pilot bank, T1, batch size 1. Math decides for the go/no-go (PLAN.md §7); the rest is reported.", ""]
    for domain, title in (("math", "Answer swap, math (test 4, used in the go/no-go)"),
                          ("code", "Answer swap, code (reported only; expected to fail, PLAN.md §4)")):
        lines += [f"## {title}", "",
                  "Pass: the swap scores strictly lower than the original on at least 80% of items. "
                  "Tied and unscored items count as not lowered.", "",
                  "| Verifier | In H1 | Lowered / items | Share | 95% interval | Tied | Raised | Unscored | Result |",
                  "|---|---|---|---|---|---|---|---|---|"]
        for name, entry in result["swap"].items():
            e, h1 = entry[domain], "yes" if name in H1_VERIFIERS else "no"
            if "items" not in e:
                lines.append(f"| {name} | {h1} | — | — | — | — | — | — | not run |")
            else:
                lines.append(f"| {name} | {h1} | {e['lowered']} / {e['items']} | {pct(e['share_lowered'])} | "
                             f"{pct(e['ci95'][0])}–{pct(e['ci95'][1])} | {e['tied']} | {e['raised']} | "
                             f"{e['unscored']} | {e['outcome'].upper()} |")
        lines.append("")
    lines += ["## Repeat runs, AR models (test 3)", "",
              "Pass: Spearman above 0.999 between two runs in one session, over the 500 pilot answers. "
              "Math, code and October 27 are information only. A failure is a scorer bug to fix, not a verdict on the model.", "",
              "| Verifier | All | Math | Code | Largest difference | Identical | Result | October 27: all | Math | Code | Largest difference |",
              "|---|---|---|---|---|---|---|---|---|---|---|"]
    for name, e in result["rerun"].items():
        if "run_1_vs_run_2" not in e:
            lines.append(f"| {name} | — | — | — | — | — | not run | — | — | — | — |")
            continue
        r, p = e["run_1_vs_run_2"], e["run_1_vs_october_27"] or {}
        same = "—" if not r["all"] else f"{r['all']['identical']} / {r['all']['n']}"
        lines.append(f"| {name} | {rho_text(r['all'])} | {rho_text(r['math'])} | {rho_text(r['code'])} | "
                     f"{diff_text(r['all'])} | {same} | {e['outcome'].upper()} | {rho_text(p.get('all'))} | "
                     f"{rho_text(p.get('math'))} | {rho_text(p.get('code'))} | {diff_text(p.get('all'))} |")
    changed = [n for n, e in result["rerun"].items() if len({v for v in e.get("revisions", {}).values() if v}) > 1]
    if changed:
        lines += ["", f"**Model revision differs between runs or from October 27:** {', '.join(changed)} (see step0.json)."]
    lines += ["", "## For the final model record", "",
              "| Verifier | Answer swap, math | Repeat runs |", "|---|---|---|"]
    lines += [f"| {n} | {e['answer_swap_math']} | {e['repeat_runs']} |" for n, e in result["go_no_go"].items()]
    lines += ["", "The label audit is the third pilot check and is not part of this file."]
    return "\n".join(lines) + "\n"


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--swap-items", default="bank/pilot/swap/swap_items.jsonl")
    ap.add_argument("--swap-key", default="bank/pilot/swap/swap_key.jsonl")
    ap.add_argument("--swap-scores", nargs="+", required=True)
    ap.add_argument("--bank", default="bank/pilot/pilot_bank.jsonl")
    ap.add_argument("--rerun", nargs=2, required=True, metavar=("RUN_1", "RUN_2"))
    ap.add_argument("--previous", nargs="*", default=None,
                    help="October 27 T1 score folders or files (default: results/pilot/ar/*_T1)")
    ap.add_argument("--out", required=True)
    args = ap.parse_args(argv)

    item_rows, bank = read_jsonl(args.swap_items), read_jsonl(args.bank)
    items = swap_items(read_jsonl(args.swap_key), item_rows)
    swap_hash, bank_hash = sha256_file(args.swap_items), sha256_file(args.bank)
    bank_ids = {r["candidate_id"] for r in bank}
    previous = args.previous if args.previous is not None else sorted(map(str, Path("results/pilot/ar").glob("*_T1")))
    swap = swap_test(items, load_run(args.swap_scores, swap_hash, {r["candidate_id"] for r in item_rows}))
    rerun = rerun_check(bank, *(load_run([p], bank_hash, bank_ids) for p in args.rerun),
                        load_run(previous, bank_hash, bank_ids))
    result = {"swap_items_sha256": swap_hash, "bank_sha256": bank_hash, "template": "T1",
              "rules": {"swap_share_at_least": SWAP_PASS[0] / SWAP_PASS[1], "rerun_spearman_above": RERUN_PASS},
              "swap": swap, "rerun": rerun, "go_no_go": go_no_go(swap, rerun)}
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "step0.json").write_text(json.dumps(result, indent=2, allow_nan=False) + "\n", encoding="utf-8", newline="\n")
    text = markdown(result)
    (out / "step0.md").write_text(text, encoding="utf-8", newline="\n")
    print(text)


if __name__ == "__main__":
    main()
