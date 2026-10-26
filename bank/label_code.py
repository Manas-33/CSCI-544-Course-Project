"""Step 3 (code): label each generated program by running the problem's tests (PLAN.md §2).

An answer is usable only if it was not cut off, has no model self-reference, and
contains a program that parses and defines the entry point. Usable programs are run
against the tests twice, each in a fresh process; a program that passes in one run
and fails in the other is flaky and excluded. Each problem's reference solution is
run too, and a problem whose reference fails is excluded as broken. Failures carry
an error type from the run: wrong_output, exception or timeout.

The runs happen in a container with no network (PLAN.md §2), via Docker by default:

    python -m bank.label_code --questions bank/dev_code/questions.jsonl \\
        --generations bank/dev_code/generations.jsonl --out bank/dev_code/labeled.jsonl

`--sandbox none` runs the same jobs as plain subprocesses, for tests only: there is no
network isolation. The choice is recorded in every label's checker_version, and
bank.make_pairs refuses such code labels unless told they are for a test.
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import tempfile
from collections import Counter
from pathlib import Path

from bank import code_exec
from bank.common import has_identity_string, read_jsonl, write_jsonl

LABELER_VERSION = "bank.label_code/2"
DEFAULT_IMAGE = "python:3.11-slim"
N_RUNS = 2


def prepare(gen: dict, question: dict) -> tuple[str, str | None]:
    """(status, program shown to verifiers) before any code runs."""
    # Cut off only if the program itself is: the reply is pre-filled up to an opening fence, so a
    # closing fence means the program ended and only the explanation after it ran out of tokens.
    if gen.get("finish_reason") == "length" and code_exec.FENCE not in gen["text"]:
        return "truncated", None
    if has_identity_string(gen["text"]):
        return "identity_string", None
    program = code_exec.sanitize(code_exec.extract_program(gen["text"]), question["entry_point"])
    return ("ok", program) if program is not None else ("no_answer", None)


def run_in_sandbox(jobs: list[dict], sandbox: str, image: str, timeout_s: int) -> dict[str, dict]:
    """job_id -> result, from one pass over all jobs."""
    if sandbox == "none":
        return {j["job_id"]: code_exec.run_job(j) for j in jobs}
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        (tmp / "in").mkdir()
        (tmp / "out").mkdir()
        shutil.copy(code_exec.__file__, tmp / "in" / "code_exec.py")
        write_jsonl(tmp / "in" / "jobs.jsonl", jobs)
        cmd = [
            "docker", "run", "--rm", "--network", "none", "--memory", "2g", "--cpus", "2",
            "--pids-limit", "512", "--read-only", "--tmpfs", "/tmp:rw,size=256m",
            "-v", f"{tmp / 'in'}:/job:ro", "-v", f"{tmp / 'out'}:/out", image,
            "python", "/job/code_exec.py", "run", "/job/jobs.jsonl", "/out/results.jsonl",
        ]
        subprocess.run(cmd, check=True, timeout=max(600, timeout_s * len(jobs) * 2))
        return {r["job_id"]: r for r in read_jsonl(tmp / "out" / "results.jsonl")}


def image_id(image: str) -> str | None:
    try:
        return subprocess.run(["docker", "image", "inspect", "--format", "{{.Id}}", image],
                              capture_output=True, text=True, check=True).stdout.strip()
    except Exception:
        return None


def combine(runs: list[dict]) -> tuple[str, int | None, str | None]:
    """(status, label, error type) from the repeated runs of one program."""
    outcomes = {r["status"] == "pass" for r in runs}
    if len(outcomes) > 1:
        return "flaky", None, None
    if outcomes == {True}:
        return "ok", 1, None
    first = runs[0]["status"]
    return "ok", 0, first if all(r["status"] == first for r in runs) else "mixed_failure"


def label_all(questions: dict[str, dict], generations: list[dict], sandbox: str,
              image: str, timeout_s: int) -> list[dict]:
    checker_version = f"{LABELER_VERSION}+{'docker:' + image if sandbox == 'docker' else 'subprocess'}"
    prepared = {g["candidate_id"]: prepare(g, questions[g["problem_id"]]) for g in generations}

    def job(job_id, program, q):
        return {"job_id": job_id, "program": program, "setup": q.get("setup", ""),
                "tests": q["tests"], "timeout_s": timeout_s}

    used = sorted({g["problem_id"] for g in generations})
    jobs = [job(f"reference::{pid}", questions[pid]["reference_solution"], questions[pid]) for pid in used]
    jobs += [job(g["candidate_id"], prepared[g["candidate_id"]][1], questions[g["problem_id"]])
             for g in generations if prepared[g["candidate_id"]][0] == "ok"]
    # Each run gets its own fixed hash seed: rebuilds give the same labels, and a program whose
    # result depends on set order differs between the runs and is dropped as flaky.
    passes = [run_in_sandbox([{**j, "hash_seed": seed} for j in jobs], sandbox, image, timeout_s)
              for seed in range(N_RUNS)]

    broken = {pid for pid in used if any(p[f"reference::{pid}"]["status"] != "pass" for p in passes)}
    rows = []
    for g in generations:
        cid, pid = g["candidate_id"], g["problem_id"]
        status, program = prepared[cid]
        label = error_type = None
        runs = [p[cid] for p in passes] if status == "ok" else []
        if status == "ok" and pid in broken:
            status = "broken_problem"
        elif status == "ok":
            status, label, error_type = combine(runs)
        rows.append({
            "candidate_id": cid,
            "problem_id": pid,
            "status": status,
            "label": label,
            "error_type": error_type,
            "run_statuses": [r["status"] for r in runs],
            "detail": runs[0]["detail"] if runs and label == 0 else None,
            "program": program,
            "checker_version": checker_version,
        })
    return rows


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--questions", default="bank/dev_code/questions.jsonl")
    ap.add_argument("--generations", default="bank/dev_code/generations.jsonl")
    ap.add_argument("--out", default="bank/dev_code/labeled.jsonl")
    ap.add_argument("--sandbox", choices=["docker", "none"], default="docker")
    ap.add_argument("--image", default=DEFAULT_IMAGE, help="Docker image with a plain Python")
    ap.add_argument("--timeout-s", type=int, default=10, help="Per program, per run")
    args = ap.parse_args(argv)

    questions = {q["problem_id"]: q for q in read_jsonl(args.questions)}
    generations = read_jsonl(args.generations)
    rows = label_all(questions, generations, args.sandbox, args.image, args.timeout_s)
    write_jsonl(args.out, rows)

    counts = Counter(r["status"] for r in rows)
    errors = Counter(r["error_type"] for r in rows if r["label"] == 0)
    manifest = {
        "step": "label_code",
        "labeler": LABELER_VERSION,
        "sandbox": args.sandbox,
        "image": args.image if args.sandbox == "docker" else None,
        "image_id": image_id(args.image) if args.sandbox == "docker" else None,
        "runs_per_program": N_RUNS,
        "timeout_s": args.timeout_s,
        "status_counts": dict(counts),
        "error_types": dict(errors),
        "broken_problems": sorted({r["problem_id"] for r in rows if r["status"] == "broken_problem"}),
    }
    Path(args.out).with_name("labeled_manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8", newline="\n"
    )
    print(f"{len(rows)} answers: {dict(counts)}")
    print(f"usable correct={sum(r['label'] == 1 for r in rows)}, wrong={sum(r['label'] == 0 for r in rows)}; "
          f"error types {dict(errors)}")


if __name__ == "__main__":
    main()
