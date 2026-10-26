"""Score a JSONL file of candidates with one diffusion model and write score rows.

Input rows need ``candidate_id``, ``problem`` and ``candidate``. Output rows
follow interface.SCORE_ROW_FIELDS; each run appends a small manifest line to
``<out>.manifest.jsonl``.
Ids already in the output file are skipped, so a rerun resumes; `error` rows are
retried, and a file scored with other settings is refused.

    python -m harness.diffusion.run_score --model dream --n-slots 8 \\
        --items dev_bank.jsonl --bank-hash <hash> --out results/dream_T1_n8.jsonl
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import platform
import sys
import time
import traceback
from pathlib import Path

import torch

from .interface import TEMPLATES, read_jsonl, score_row, write_jsonl
from .scorer import ANSWER_PREFIXES, SPECS, PromptTooLong, load_scorer


def timed_score(scorer, prompts):
    """Verdicts, seconds per prompt (synchronized) and peak GPU GB for one batch."""
    cuda = str(scorer.device).startswith("cuda") and torch.cuda.is_available()
    if cuda:
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
    t0 = time.perf_counter()
    verdicts = scorer.score_ids(prompts)
    if cuda:
        torch.cuda.synchronize()
    seconds = (time.perf_counter() - t0) / len(prompts)
    peak = torch.cuda.max_memory_allocated() / 1e9 if cuda else None
    return verdicts, seconds, peak


def score_items(scorer, items, template="T1", batch_size=8, bank_hash="unknown"):
    """Yield one score row per item, in input order within each batch."""
    render = TEMPLATES[template]
    if not scorer.spec.batched:
        batch_size = 1
    common = dict(bank_hash=bank_hash, model=scorer.spec.key, revision=scorer.revision,
                  template=template, n_slots=scorer.n_slots, answer_prefix=scorer.answer_prefix)

    def rows_for(batch):
        try:
            verdicts, seconds, peak = timed_score(scorer, [p for _, p in batch])
        except Exception:
            if len(batch) > 1:  # retry one by one so one bad item does not sink the batch
                for one in batch:
                    yield from rows_for([one])
                return
            traceback.print_exc(file=sys.stderr)
            yield score_row(candidate_id=batch[0][0]["candidate_id"], status="error", **common)
            return
        for (item, _), v in zip(batch, verdicts):
            yield score_row(candidate_id=item["candidate_id"], verdict=v,
                            runtime_s=seconds, peak_mem_gb=peak, **common)

    batch = []
    for item in items:
        prompt = scorer.prompt_ids(render(item["problem"], item["candidate"]))
        try:
            scorer.check_length(prompt)
        except PromptTooLong:
            yield score_row(candidate_id=item["candidate_id"], status="too_long", **common)
            continue
        batch.append((item, prompt))
        if len(batch) == batch_size:
            yield from rows_for(batch)
            batch = []
    if batch:
        yield from rows_for(batch)


def environment_info(device: str) -> dict:
    import transformers

    info = {"python": platform.python_version(), "torch": torch.__version__,
            "transformers": transformers.__version__}
    if device.startswith("cuda") and torch.cuda.is_available():
        info["gpu"] = torch.cuda.get_device_name(0)
    return info


def previous_rows(out, model, template, n_slots, bank_hash, answer_prefix="none") -> list:
    """Rows a rerun keeps from ``out``: everything except ``error`` rows, which are
    rescored (the file is rewritten without them). A file holding rows scored with
    other settings is refused, so a 1-slot file never passes for an 8-slot run."""
    out = Path(out)
    if not out.exists():
        return []
    rows = list(read_jsonl(out))
    wanted = (model, template, n_slots, bank_hash, answer_prefix)
    other = {(r["model"], r["template"], r["n_slots"], r["bank_hash"], r.get("answer_prefix", "none"))
             for r in rows} - {wanted}
    if other:
        raise SystemExit(f"{out} holds rows scored with other settings {sorted(other)}; "
                         f"this run is {wanted}. Use a new --out.")
    kept = [r for r in rows if r["status"] != "error"]
    if len(kept) != len(rows):
        write_jsonl(out, kept)
    return kept


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--model", required=True, choices=sorted(SPECS))
    ap.add_argument("--items", required=True, help="JSONL with candidate_id, problem, candidate")
    ap.add_argument("--out", required=True, help="output JSONL of score rows (appended)")
    ap.add_argument("--n-slots", type=int, required=True, choices=[1, 8, 32])
    ap.add_argument("--bank-hash", required=True)
    ap.add_argument("--template", default="T1", choices=sorted(TEMPLATES))
    ap.add_argument("--answer-prefix", default="none", choices=sorted(ANSWER_PREFIXES),
                    help="text after the generation prompt (scorer.ANSWER_PREFIXES); iLLaDA needs thinking off")
    ap.add_argument("--batch-size", type=int, default=8, help="ignored for iLLaDA (always 1)")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--revision", default=None, help="override the pinned HF revision")
    ap.add_argument("--allow-env-mismatch", action="store_true")
    args = ap.parse_args(argv)

    out = Path(args.out)
    done = {r["candidate_id"] for r in previous_rows(out, args.model, args.template,
                                                     args.n_slots, args.bank_hash, args.answer_prefix)}
    items = [it for it in read_jsonl(args.items) if it["candidate_id"] not in done]
    if args.limit is not None:
        items = items[:args.limit]
    print(f"{len(done)} already scored, {len(items)} to score", file=sys.stderr)
    if not items:
        return

    started = dt.datetime.now(dt.timezone.utc).isoformat()
    scorer = load_scorer(args.model, args.n_slots, device=args.device,
                         revision=args.revision, strict_env=not args.allow_env_mismatch,
                         answer_prefix=args.answer_prefix)
    counts = {}
    for row in score_items(scorer, items, args.template, args.batch_size, args.bank_hash):
        write_jsonl(out, [row], append=True)
        counts[row["status"]] = counts.get(row["status"], 0) + 1
    manifest = {
        "command": sys.argv, "started": started,
        "finished": dt.datetime.now(dt.timezone.utc).isoformat(),
        "model": args.model, "hf_id": scorer.spec.hf_id, "revision": scorer.revision,
        "n_slots": args.n_slots, "template": args.template, "bank_hash": args.bank_hash,
        "answer_prefix": args.answer_prefix, "answer_prefix_text": ANSWER_PREFIXES[args.answer_prefix],
        "batch_size": args.batch_size if scorer.spec.batched else 1,
        "items": args.items, "expected": len(items), "completed": counts,
        "environment": environment_info(args.device),
    }
    write_jsonl(str(out) + ".manifest.jsonl", [manifest], append=True)  # one line per run
    print(json.dumps(counts), file=sys.stderr)


if __name__ == "__main__":
    main()
