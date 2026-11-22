"""Step 3 (code, pilot and full bank): label programs with EvalPlus's own tests (PLAN.md §2).

Programs are extracted and filtered exactly as in bank.label_code (cut off, self-reference,
no program or no entry point -> unusable). Each usable program is then checked against
the problem's HumanEval+ / MBPP+ base and plus tests with EvalPlus's check_correctness,
which compares outputs to the canonical solution's with EvalPlus's tolerances and special
cases, under its per-test time limits and its reliability guard. A program is correct
only if it passes both test sets; it runs twice, and one that passes once and fails once
is flaky and dropped.

A wrong program has one of two error types, the two EvalPlus tells apart: "timeout" (a time limit was hit,
in either test set or either run) or "wrong_output" (it failed otherwise; a crash counts here too).

EvalPlus's guard is Linux-only, so this runs on Colab (a throwaway VM) rather than in the
Docker sandbox bank.label_code uses; the checker version records that.

    python -m bank.label_evalplus --questions bank/pilot/questions_code.jsonl \\
        --generations bank/pilot/generations_code.jsonl --out bank/pilot/labeled_code.jsonl

--progress <file> saves the finished checks every 200 programs and skips them on a rerun
(the full bank takes hours on a small CPU runtime). That file is an execution log: keep it private.
"""

from __future__ import annotations

import argparse
import json
import multiprocessing
import os
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from importlib import metadata
from pathlib import Path

from bank.common import read_jsonl, sha256_text, write_jsonl
from bank.label_code import combine, prepare

# Bump when the checks change: --progress reuses saved checks only from the same version.
LABELER_VERSION = "bank.label_evalplus/1"
N_RUNS = 2
SAVE_EVERY = 200


def load_problems(datasets: set[str]):
    """task_id -> (dataset, problem, expected outputs), for the EvalPlus datasets used."""
    from evalplus.data import get_human_eval_plus, get_human_eval_plus_hash, get_mbpp_plus, get_mbpp_plus_hash
    from evalplus.eval._special_oracle import MBPP_OUTPUT_NOT_NONE_TASKS
    from evalplus.evaluate import get_groundtruth

    out = {}
    if "humaneval" in datasets:
        problems = get_human_eval_plus()
        expected = get_groundtruth(problems, get_human_eval_plus_hash(), [])
        out.update({t: ("humaneval", p, expected[t]) for t, p in problems.items()})
    if "mbpp" in datasets:
        problems = get_mbpp_plus()
        expected = get_groundtruth(problems, get_mbpp_plus_hash(), MBPP_OUTPUT_NOT_NONE_TASKS)
        out.update({t: ("mbpp", p, expected[t]) for t, p in problems.items()})
    return out


def run_once(dataset: str, problem: dict, program: str, expected: dict) -> dict:
    """One EvalPlus check of a program: pass only if both the base and plus tests pass."""
    from evalplus.evaluate import check_correctness

    r = check_correctness(dataset, 0, problem, program, expected, fast_check=True)
    statuses = (r["base"][0], r["plus"][0])
    if statuses == ("pass", "pass"):
        return {"status": "pass", "detail": None}
    status = "timeout" if "timeout" in statuses else "wrong_output"
    return {"status": status, "detail": f"base={statuses[0]} plus={statuses[1]}"}


def check_job(job: tuple) -> list[dict]:
    """All runs of one program, inside a single-threaded worker process (see main)."""
    dataset, problem, program, expected = job
    return [run_once(dataset, problem, program, expected) for _ in range(N_RUNS)]


def read_progress(path: Path, version: str) -> dict[tuple[str, str], list[dict]]:
    """(candidate id, program hash) -> runs already done by this checker. A half-written last line is ignored."""
    if not path.is_file():
        return {}
    done = {}
    for line in path.read_text(encoding="utf-8").split("\n"):
        try:
            row = json.loads(line)
        except json.JSONDecodeError:
            continue
        if row["checker_version"] == version:
            done[(row["candidate_id"], row["program_sha256"])] = row["runs"]
    return done


def check_all(usable: list[dict], programs: dict[str, str], jobs: list[tuple], version: str,
              progress: Path | None, workers: int) -> dict[str, list[dict]]:
    """candidate id -> runs for every usable program, resuming from the progress file if there is one."""
    ids = [(g["candidate_id"], sha256_text(programs[g["candidate_id"]])) for g in usable]
    done = read_progress(progress, version) if progress else {}
    todo = [(key, job) for key, job in zip(ids, jobs) if key not in done]
    if progress:
        print(f"{len(ids) - len(todo)} of {len(ids)} programs already checked", flush=True)
        progress.parent.mkdir(parents=True, exist_ok=True)
    # EvalPlus forks a process per check. Forking from a threaded process is refused by recent
    # filelock versions (and is unsafe anyway), so each worker is a spawned, single-threaded process.
    ctx = multiprocessing.get_context("spawn")
    unsaved = []

    def save() -> None:
        if progress and unsaved:
            with open(progress, "a", encoding="utf-8", newline="\n") as f:
                f.write("\n" + "".join(json.dumps(r) + "\n" for r in unsaved))  # "\n" ends a half-written line
            unsaved.clear()

    with ProcessPoolExecutor(workers, mp_context=ctx) as pool:
        results = pool.map(check_job, [job for _, job in todo], chunksize=8)
        for n, ((key, _), runs) in enumerate(zip(todo, results), 1):
            done[key] = runs
            unsaved.append({"candidate_id": key[0], "program_sha256": key[1], "runs": runs, "checker_version": version})
            if n % SAVE_EVERY == 0:
                save()
            if n % 500 == 0:
                print(f"checked {n}/{len(todo)} programs", flush=True)
    save()
    return {cid: done[(cid, sha)] for cid, sha in ids}


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--questions", required=True)
    ap.add_argument("--generations", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--progress", default=None, help="save finished checks here and resume from them")
    ap.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) // 2),
                    help="parallel checks; kept below the core count so time limits stay stable")
    args = ap.parse_args(argv)

    questions = {q["problem_id"]: q for q in read_jsonl(args.questions)}
    generations = read_jsonl(args.generations)
    problems = load_problems({q["dataset"] for q in questions.values()})
    version = f"{LABELER_VERSION}+evalplus=={metadata.version('evalplus')}+colab-guard"
    prepared = {g["candidate_id"]: prepare(g, questions[g["problem_id"]]) for g in generations}

    usable = [g for g in generations if prepared[g["candidate_id"]][0] == "ok"]
    jobs = []
    for g in usable:
        dataset, problem, expected = problems[questions[g["problem_id"]]["task_id"]]
        jobs.append((dataset, problem, prepared[g["candidate_id"]][1], expected))
    runs = check_all(usable, {c: p for c, (_, p) in prepared.items()}, jobs, version,
                     Path(args.progress) if args.progress else None, args.workers)

    rows = []
    for g in generations:
        cid = g["candidate_id"]
        status, program = prepared[cid]
        label = error_type = None
        these = runs.get(cid, [])
        if status == "ok":
            status, label, error_type = combine(these)
            if error_type == "mixed_failure":  # one run failed, the other hit a time limit
                error_type = "timeout"
        rows.append({
            "candidate_id": cid, "problem_id": g["problem_id"], "status": status, "label": label,
            "error_type": error_type, "run_statuses": [r["status"] for r in these],
            "detail": these[0]["detail"] if these and label == 0 else None,
            "program": program, "checker_version": version,
        })
    write_jsonl(args.out, rows)
    counts = Counter(r["status"] for r in rows)
    errors = Counter(r["error_type"] for r in rows if r["label"] == 0)
    Path(args.out).with_name(Path(args.out).stem + "_manifest.json").write_text(json.dumps({
        "step": "label_evalplus", "labeler": version, "runs_per_program": N_RUNS,
        "fast_check": True, "status_counts": dict(counts), "error_types": dict(errors),
    }, indent=2) + "\n", encoding="utf-8", newline="\n")
    print(f"{len(rows)} answers: {dict(counts)}")
    print(f"usable correct={sum(r['label'] == 1 for r in rows)}, wrong={sum(r['label'] == 0 for r in rows)}; "
          f"error types {dict(errors)}")


if __name__ == "__main__":
    main()
