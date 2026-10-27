"""Step 2 (API generator): candidate answers from an API model through OpenRouter (PLAN.md §2).

Same prompts and sampling defaults as bank.generate, one request per sample. Code answers
are not pre-filled (the chat API cannot continue an assistant turn), so their rows carry
"prefilled": false and the labeler reads the first fenced block instead.

Rate limits (429) and server errors are retried with exponential backoff, honouring
Retry-After. Spend is tracked from each response's token usage at the given prices and
kept in a ledger shared by every run; a request that could push the total past
--budget-usd is not sent. Answers are appended as they arrive, so a rerun resumes.

    OPENROUTER_API_KEY=... python -m bank.generate_api --questions bank/pilot/questions_math.jsonl \\
        --out bank/pilot/gen/gpt41mini_math/generations.jsonl
"""

from __future__ import annotations

import argparse
import json
import os
import random
import sys
import threading
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path

from bank.common import read_jsonl, sha256_text, write_jsonl
from bank.generate import CODE_PROMPT, PROMPT, apply_domain_defaults, questions_domain

URL = "https://openrouter.ai/api/v1/chat/completions"
MAX_ATTEMPTS = 10


class Budget:
    """Running spend in USD, persisted to a ledger file shared across runs."""

    def __init__(self, ledger: Path, cap: float, price_in: float, price_out: float):
        self.ledger, self.cap = ledger, cap
        self.price_in, self.price_out = price_in / 1e6, price_out / 1e6
        self.lock = threading.Lock()
        self.data = json.loads(ledger.read_text()) if ledger.is_file() else {"spent_usd": 0.0, "runs": []}
        self.reserved = 0.0

    def cost(self, prompt_tokens: int, completion_tokens: int) -> float:
        return prompt_tokens * self.price_in + completion_tokens * self.price_out

    def reserve(self, worst_case: float) -> bool:
        with self.lock:
            if self.data["spent_usd"] + self.reserved + worst_case > self.cap:
                return False
            self.reserved += worst_case
            return True

    def settle(self, worst_case: float, actual: float) -> None:
        with self.lock:
            self.reserved -= worst_case
            self.data["spent_usd"] += actual

    def save(self) -> None:
        with self.lock:
            self.ledger.parent.mkdir(parents=True, exist_ok=True)
            self.ledger.write_text(json.dumps(self.data, indent=2) + "\n", encoding="utf-8", newline="\n")


def messages(question: str, domain: str) -> list[dict]:
    text = CODE_PROMPT.format(question=question.strip()) if domain == "code" else PROMPT.format(question=question)
    return [{"role": "user", "content": text}]


def post(body: dict, key: str) -> dict:
    """One completion, retried on 429, 5xx and network errors with exponential backoff."""
    data = json.dumps(body).encode("utf-8")
    for attempt in range(MAX_ATTEMPTS):
        req = urllib.request.Request(URL, data=data, method="POST", headers={
            "Authorization": f"Bearer {key}", "Content-Type": "application/json"})
        try:
            with urllib.request.urlopen(req, timeout=180) as resp:
                out = json.loads(resp.read().decode("utf-8"))
            if "error" in out:  # OpenRouter can return 200 with an upstream error body
                raise RuntimeError(f"upstream error: {out['error']}")
            return out
        except urllib.error.HTTPError as exc:
            if exc.code not in (408, 429, 500, 502, 503, 504):
                raise RuntimeError(f"HTTP {exc.code}: {exc.read()[:300]!r}") from exc
            wait = float(exc.headers.get("Retry-After") or 0) or min(60.0, 2 ** attempt)
        except (urllib.error.URLError, TimeoutError, RuntimeError, json.JSONDecodeError) as exc:
            if attempt == MAX_ATTEMPTS - 1:
                raise RuntimeError(f"gave up after {MAX_ATTEMPTS} attempts: {exc}") from exc
            wait = min(60.0, 2 ** attempt)
        time.sleep(wait + random.uniform(0, 1))
    raise RuntimeError(f"gave up after {MAX_ATTEMPTS} attempts (rate limited)")


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--questions", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--model", default="openai/gpt-4.1-mini")
    ap.add_argument("--tag", default="gpt41mini")
    ap.add_argument("--n-samples", type=int, default=None)
    ap.add_argument("--temperature", type=float, default=None)
    ap.add_argument("--top-p", type=float, default=None)
    ap.add_argument("--max-new-tokens", type=int, default=None)
    ap.add_argument("--seed", type=int, default=544)
    ap.add_argument("--workers", type=int, default=6, help="parallel requests; lower it if 429s persist")
    ap.add_argument("--budget-usd", type=float, default=5.0, help="hard cap on total spend across runs")
    ap.add_argument("--price-in", type=float, default=0.40, help="USD per million input tokens")
    ap.add_argument("--price-out", type=float, default=1.60, help="USD per million output tokens")
    ap.add_argument("--ledger", default="bank/pilot/api_spend.json")
    return ap


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    key = os.environ.get("OPENROUTER_API_KEY")
    if not key:
        raise SystemExit("Set OPENROUTER_API_KEY")
    questions = read_jsonl(args.questions)
    domain = questions_domain(questions)
    apply_domain_defaults(args, domain)
    out = Path(args.out)
    done = {r["candidate_id"] for r in read_jsonl(out)} if out.is_file() else set()
    budget = Budget(Path(args.ledger), args.budget_usd, args.price_in, args.price_out)
    write_lock = threading.Lock()
    counts = {"ok": 0, "failed": 0, "over_budget": 0}

    jobs = [(i, q, k) for i, q in enumerate(questions) for k in range(args.n_samples)
            if f"{q['problem_id']}__{args.tag}__{k}" not in done]
    print(f"{len(done)} answers already present, {len(jobs)} to request; "
          f"spent so far ${budget.data['spent_usd']:.3f} of ${args.budget_usd:.2f}")

    def one(i: int, q: dict, k: int):
        msgs = messages(q["question"], domain)
        worst = budget.cost(sum(len(m["content"]) for m in msgs), args.max_new_tokens)  # chars >= tokens
        if not budget.reserve(worst):
            return "over_budget"
        actual = 0.0
        try:
            seed = args.seed + i * 1000 + k
            resp = post({"model": args.model, "messages": msgs, "temperature": args.temperature,
                         "top_p": args.top_p, "max_tokens": args.max_new_tokens, "seed": seed,
                         "provider": {"require_parameters": True}}, key)
            usage = resp.get("usage") or {}
            actual = budget.cost(usage.get("prompt_tokens", 0), usage.get("completion_tokens", 0))
            choice = resp["choices"][0]
            text = choice["message"].get("content") or ""
            row = {
                "candidate_id": f"{q['problem_id']}__{args.tag}__{k}",
                "problem_id": q["problem_id"],
                "generator": args.model,
                "generator_revision": resp.get("model"),
                "settings": {"temperature": args.temperature, "top_p": args.top_p,
                             "max_new_tokens": args.max_new_tokens, "n_samples": args.n_samples,
                             "backend": "openrouter", "provider": resp.get("provider")},
                "seed": seed,
                "sample_index": k,
                "text": text,
                "text_sha256": sha256_text(text),
                "finish_reason": "length" if choice.get("finish_reason") == "length" else "stop",
                "n_tokens": usage.get("completion_tokens"),
                "prefilled": False,
            }
            with write_lock, open(out, "a", encoding="utf-8", newline="\n") as f:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
            return "ok"
        except Exception as exc:  # noqa: BLE001 - one failed sample must not stop the run
            print(f"{q['problem_id']} sample {k}: {exc}", file=sys.stderr)
            return "failed"
        finally:
            budget.settle(worst, actual)

    out.parent.mkdir(parents=True, exist_ok=True)
    started = datetime.now(timezone.utc).isoformat(timespec="seconds")
    with ThreadPoolExecutor(args.workers) as pool:
        futures = [pool.submit(one, *job) for job in jobs]
        for n, fut in enumerate(as_completed(futures), 1):
            counts[fut.result()] += 1
            if n % 100 == 0:
                budget.save()
                print(f"{n}/{len(jobs)} {counts} spent ${budget.data['spent_usd']:.3f}", flush=True)

    rows = sorted(read_jsonl(out), key=lambda r: r["candidate_id"]) if out.is_file() else []
    write_jsonl(out, rows)  # deterministic order for hashing
    budget.data["runs"].append({"questions": args.questions, "out": args.out, "model": args.model,
                                "started_utc": started, "counts": counts})
    budget.save()
    (out.parent / "generations_manifest.json").write_text(json.dumps({
        "step": "generate_api", "questions": args.questions, "n_questions": len(questions),
        "n_generations": len(rows), "generator": args.model, "domain": domain,
        "prompt": CODE_PROMPT if domain == "code" else PROMPT, "prefilled": False,
        "settings": {k: v for k, v in vars(args).items() if k not in ("questions", "out")},
        "started_utc": started, "finished_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "spent_usd_total": round(budget.data["spent_usd"], 4),
    }, indent=2) + "\n", encoding="utf-8", newline="\n")
    print(f"{counts}; {len(rows)} answers in {out}; total spend ${budget.data['spent_usd']:.3f}")
    if counts["over_budget"]:
        print(f"Stopped {counts['over_budget']} requests at the ${args.budget_usd:.2f} cap", file=sys.stderr)


if __name__ == "__main__":
    main()
