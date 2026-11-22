"""Score a bank file with LLaDA's or Dream's reconstruction score and write score rows.

Input rows need ``candidate_id``, ``problem`` and ``candidate`` (``--domain`` also needs
``domain``); labels are never read. The bank hash is the sha256 of the file itself.
Each run appends a manifest line to ``<out>.manifest.jsonl``. Ids already in the output file
are skipped, so a rerun resumes; `error` rows are retried, and a file scored with other
settings is refused.

    python -m harness.diffusion.run_reconstruction --model llada --domain math --seed 544 \\
        --items bank/pilot/pilot_bank.jsonl --out results/reconstruction/pilot/llada_math_s544.jsonl

Score row, one JSON object per line (harness/ar_logprob.py writes the same fields):

    bank_hash        sha256 of the bank file
    candidate_id     candidate id from the bank row
    model            llada | dream, or the Hugging Face id of an AR model
    revision         Hugging Face commit of the weights
    method           diffusion_mc | ar_exact
    n_samples        Monte Carlo samples (null for ar_exact)
    seed             Monte Carlo seed (null for ar_exact)
    mc_group         samples whose mask counts are spread together (null for ar_exact)
    micro_batch      samples per forward pass (1 for ar_exact)
    attention        attention implementation of an AR model (null for diffusion)
    n_prompt_tokens  tokens of the user turn and the generation prompt
    n_tokens         answer tokens scored (no end-of-turn token)
    loglik           log-likelihood of the answer, summed over its tokens (null unless ok)
    score            loglik / n_tokens, higher = more plausible (null unless ok)
    loglik_at        the same estimate from fewer samples (null for ar_exact)
    boundary         clean | separate (reconstruction.PromptAnswer)
    status           ok | too_long | empty | error
    runtime_s        synchronized wall time of all forward passes for the item
    peak_mem_gb      peak GPU memory allocated for the item (null on CPU)
    gpu              GPU name (null on CPU)
"""
from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import re
import subprocess
import sys
import time
import traceback
from pathlib import Path

import torch

from .interface import write_jsonl
from .reconstruction import (MC_GROUP, METHOD, MODELS, N_SAMPLES, SAMPLE_COUNTS, SEED, EmptyAnswer,
                             encode_pair, fits, reconstruct)
from .run_score import environment_info
from .scorer import SPECS, load_scorer

ROW_FIELDS = (
    "bank_hash", "candidate_id", "model", "revision", "method", "n_samples", "seed", "mc_group",
    "micro_batch", "attention", "n_prompt_tokens", "n_tokens", "loglik", "score", "loglik_at",
    "boundary", "status", "runtime_s", "peak_mem_gb", "gpu",
)
SETTINGS = ("bank_hash", "model", "revision", "method", "n_samples", "seed", "mc_group", "micro_batch",
            "attention")
STATUSES = ("ok", "too_long", "empty", "error")


def backup_row(settings: dict, candidate_id: str, status: str = "ok", **values) -> dict:
    """One score row; values that do not apply stay null."""
    if status not in STATUSES:
        raise ValueError(f"unknown status {status!r}")
    if set(settings) != set(SETTINGS) or set(values) - set(ROW_FIELDS):
        raise ValueError("unknown or missing row fields")
    if (status == "ok") != (values.get("score") is not None and values.get("loglik") is not None):
        raise ValueError("a score is required exactly when status is ok")
    row = {field: None for field in ROW_FIELDS}
    row.update(settings, candidate_id=candidate_id, status=status, **values)
    return row


def timed(device, work):
    """(result, synchronized seconds, peak GPU GB) of one call."""
    cuda = str(device).startswith("cuda") and torch.cuda.is_available()
    if cuda:
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
    start = time.perf_counter()
    result = work()
    if cuda:
        torch.cuda.synchronize()
    seconds = time.perf_counter() - start
    return result, seconds, torch.cuda.max_memory_allocated() / 1e9 if cuda else None


def gpu_name(device):
    return torch.cuda.get_device_name(0) if str(device).startswith("cuda") and torch.cuda.is_available() else None


def gpu_type(name):
    """"NVIDIA A100-SXM4-40GB" -> "A100". Scores that are compared must come from one GPU type."""
    found = re.search(r"\b[A-Z]{1,3}\d+\b", name or "")
    return found.group(0) if found else name


def score_items(scorer, items, settings):
    """Yield one score row per item, in input order."""
    gpu = gpu_name(scorer.device)
    for item in items:
        cid = item["candidate_id"]
        lengths = {"gpu": gpu}
        try:
            pair = encode_pair(scorer.tokenizer, item["problem"], item["candidate"])
            lengths.update(n_prompt_tokens=len(pair.prompt), n_tokens=len(pair.answer), boundary=pair.boundary)
            if not fits(scorer, pair):
                yield backup_row(settings, cid, "too_long", **lengths)
                continue
            result, seconds, peak = timed(scorer.device, lambda: reconstruct(
                scorer, pair, cid, settings["seed"], settings["n_samples"],
                settings["micro_batch"], settings["mc_group"]))
        except EmptyAnswer:
            yield backup_row(settings, cid, "empty", **lengths)
            continue
        except Exception:
            traceback.print_exc(file=sys.stderr)
            yield backup_row(settings, cid, "error", **lengths)
            continue
        yield backup_row(settings, cid, loglik=result.loglik, score=result.score, loglik_at=result.loglik_at,
                         runtime_s=seconds, peak_mem_gb=peak, **lengths)


# ---------------------------------------------------------------- items, resume, manifest

def load_items(path, domain=None, subset=None, bank_hash=None):
    """(items to score, sha256 of the bank file). ``subset`` is a file written by
    ``analysis.reconstruction draw``: only its candidate ids are kept."""
    raw = Path(path).read_bytes()
    digest = hashlib.sha256(raw).hexdigest()
    if bank_hash and bank_hash != digest:
        raise SystemExit(f"{path} has sha256 {digest}, not {bank_hash}")
    items = [json.loads(line) for line in raw.splitlines() if line.strip()]
    if len({it["candidate_id"] for it in items}) != len(items):
        raise SystemExit(f"{path} repeats a candidate id")
    if domain:
        items = [it for it in items if it["domain"] == domain]
    if subset:
        drawn = json.loads(Path(subset).read_text(encoding="utf-8"))
        if drawn["bank_hash"] != digest:
            raise SystemExit(f"{subset} was drawn from bank {drawn['bank_hash']}, not {digest}")
        wanted = {c for ids in drawn["splits"].values() for c in ids}
        items = [it for it in items if it["candidate_id"] in wanted]
        if len(items) != len(wanted):
            raise SystemExit(f"{subset} names {len(wanted) - len(items)} candidates that are not selected")
    if not items:
        raise SystemExit(f"nothing to score in {path}")
    return items, digest


def previous_rows(out, settings: dict, gpu=None) -> list:
    """Rows a rerun keeps from ``out``: everything except ``error`` rows, which are rescored,
    and a last line cut off by an interrupted write. A file holding rows scored with other
    settings is refused, so a 32-sample file never passes for a 128-sample run; so is one
    scored on another GPU type."""
    out = Path(out)
    if not out.exists():
        return []
    raw = out.read_bytes()
    rows, lines = [], raw.splitlines()
    for i, line in enumerate(lines):
        if not line.strip():
            continue
        try:
            rows.append(json.loads(line))
        except ValueError:
            if i < len(lines) - 1:
                raise
    for row in rows:
        other = {k: row.get(k) for k in SETTINGS if row.get(k) != settings[k]}
        if other:
            raise SystemExit(f"{out} holds rows scored with other settings {other}; this run is "
                             f"{ {k: settings[k] for k in other} }. Use a new --out.")
    kept = [r for r in rows if r["status"] != "error"]
    if len({r["candidate_id"] for r in kept}) != len(kept):
        raise SystemExit(f"{out} repeats a candidate id")
    other = {gpu_type(r["gpu"]) for r in kept if r["status"] == "ok"} - {gpu_type(gpu)}
    if other:
        raise SystemExit(f"{out} holds rows scored on {sorted(map(str, other))}; this run is on {gpu}. Use a new --out.")
    if len(kept) != len(lines) or not raw.endswith(b"\n"):
        temporary = out.with_name(out.name + ".tmp")  # replaced in one step, so nothing is lost if this stops
        write_jsonl(temporary, kept)
        temporary.replace(out)
    return kept


def code_revision():
    try:
        return subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=True,
                              cwd=Path(__file__).resolve().parent).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def run(args, items, settings, make_rows, extra) -> int:
    """Score what ``args.out`` does not have yet and append a manifest line. ``make_rows(items)``
    loads the model, so nothing is loaded when the file is already complete."""
    out = Path(args.out)
    done = {r["candidate_id"] for r in previous_rows(out, settings, gpu_name(args.device))}
    todo = [it for it in items if it["candidate_id"] not in done]
    if args.limit is not None:
        todo = todo[:args.limit]
    print(f"{len(done)} already scored, {len(todo)} to score", file=sys.stderr)
    if not todo:
        return 0
    started, counts = dt.datetime.now(dt.timezone.utc).isoformat(), {}
    for row in make_rows(todo):
        write_jsonl(out, [row], append=True)
        counts[row["status"]] = counts.get(row["status"], 0) + 1
    manifest = {
        "command": sys.argv, "started": started, "finished": dt.datetime.now(dt.timezone.utc).isoformat(),
        "code_revision": code_revision(), **settings, "end_of_turn_scored": False,
        "items": args.items, "domain": args.domain, "subset": args.subset,
        "expected": len(todo), "completed": counts,
        "output_sha256": hashlib.sha256(out.read_bytes()).hexdigest(),
        "environment": environment_info(args.device), **extra,
    }
    write_jsonl(str(out) + ".manifest.jsonl", [manifest], append=True)  # one line per run
    print(json.dumps(counts), file=sys.stderr)
    return 1 if counts.get("error") else 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--model", required=True, choices=MODELS)
    ap.add_argument("--items", required=True, help="bank JSONL with candidate_id, problem, candidate")
    ap.add_argument("--out", required=True, help="output JSONL of score rows (appended)")
    ap.add_argument("--domain", default=None, choices=["math", "code"], help="score only this domain")
    ap.add_argument("--subset", default=None, help="score only the candidates drawn by analysis.reconstruction draw")
    ap.add_argument("--bank-hash", default=None, help="refuse the bank file if its sha256 differs")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--n-samples", type=int, default=N_SAMPLES, choices=SAMPLE_COUNTS,
                    help="Monte Carlo samples; the repair always uses 128 (PLAN.md §7)")
    ap.add_argument("--seed", type=int, default=SEED)
    ap.add_argument("--micro-batch", type=int, default=1,
                    help="samples of one item per forward pass; more than 1 can move BF16 scores by rounding")
    ap.add_argument("--revision", default=None, help="override the pinned HF revision")
    ap.add_argument("--allow-env-mismatch", action="store_true")
    args = ap.parse_args(argv)

    spec = SPECS[args.model]
    items, bank_hash = load_items(args.items, args.domain, args.subset, args.bank_hash)
    settings = {"bank_hash": bank_hash, "model": args.model, "revision": args.revision or spec.revision,
                "method": METHOD, "n_samples": args.n_samples, "seed": args.seed,
                "mc_group": min(MC_GROUP, args.n_samples), "micro_batch": args.micro_batch, "attention": None}

    def make_rows(todo):
        scorer = load_scorer(args.model, 1, device=args.device, revision=args.revision,
                             strict_env=not args.allow_env_mismatch)
        yield from score_items(scorer, todo, settings)

    return run(args, items, settings, make_rows, {"hf_id": spec.hf_id})


if __name__ == "__main__":
    raise SystemExit(main())
