"""Code answers: pull the program out of a response, and run it against a problem's tests.

Standard library only, so the same file runs inside the sandbox container:

    python code_exec.py run jobs.jsonl results.jsonl

Each job is {"job_id", "program", "setup", "tests", "timeout_s"}; each result is
{"job_id", "status", "detail", "seconds"} with status one of
pass | wrong_output | exception | timeout. Every job runs in its own fresh Python
process, so one program can't affect the next.
"""

from __future__ import annotations

import ast
import json
import os
import subprocess
import sys
import tempfile
import time

FENCE = "```"
RESULT_MARK = "__CODE_EXEC_RESULT__"

# Top-level statements kept from a generated script: imports, definitions, and the
# assignments those definitions use (constants such as MOD = 10**9 + 7). Everything else
# (example calls, prints, asserts, `if __name__ == "__main__":` blocks) is dropped, as
# EvalPlus's sanitizer does, so only the solution itself is tested and shown to verifiers.
_DEFS = (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)
_IMPORTS = (ast.Import, ast.ImportFrom)


def _assigned_names(node: ast.stmt) -> set[str]:
    targets = node.targets if isinstance(node, ast.Assign) else [node.target]
    return {n.id for t in targets for n in ast.walk(t) if isinstance(n, ast.Name)}


def extract_program(text: str, prefilled: bool = True) -> str:
    """The code a response contains.

    With the EvalPlus prompt the reply is pre-filled up to an opening ```python fence,
    so the generation *is* the code, up to the closing fence. Otherwise take the first
    fenced block that has one, or the whole text.
    """
    if prefilled:
        return text.split(FENCE, 1)[0].strip("\n")
    if FENCE in text:
        body = text.split(FENCE, 2)[1]
        first, _, rest = body.partition("\n")
        return (rest if first.strip().isidentifier() or not first.strip() else body).strip("\n")
    return text.strip("\n")


def sanitize(program: str, entry_point: str) -> str | None:
    """The program cut down to imports, definitions and top-level assignments.

    None if it doesn't parse or doesn't define `entry_point`: such an answer is
    dropped as "no answer" rather than labeled wrong (PLAN.md §2 filters).
    """
    try:
        tree = ast.parse(program)
    except SyntaxError:
        return None
    defs = [node for node in tree.body if isinstance(node, _DEFS)]
    if entry_point not in {n.name for n in defs if not isinstance(n, ast.ClassDef)}:
        return None
    used = {n.id for d in defs for n in ast.walk(d) if isinstance(n, ast.Name)}
    kept = [
        node for node in tree.body
        if isinstance(node, _DEFS + _IMPORTS)
        or (isinstance(node, (ast.Assign, ast.AnnAssign)) and _assigned_names(node) & used)
    ]
    lines = program.splitlines()
    pieces = []
    for node in kept:
        start = min([d.lineno for d in getattr(node, "decorator_list", [])] + [node.lineno])
        pieces.append("\n".join(lines[start - 1 : node.end_lineno]))
    return "\n\n".join(pieces) + "\n"


_HARNESS = """\
import json, sys
def _report(status, detail=""):
    sys.__stdout__.write("\\n{mark}" + json.dumps({{"status": status, "detail": detail[:300]}}) + "\\n")
    sys.__stdout__.flush()
    sys.exit(0)
_ns = {{"__name__": "__candidate__"}}
try:
    exec(compile({setup!r}, "<setup>", "exec"), _ns)
    exec(compile({program!r}, "<candidate>", "exec"), _ns)
except BaseException as e:
    _report("exception", "loading: " + type(e).__name__ + ": " + str(e))
for _t in {tests!r}:
    try:
        exec(compile(_t, "<test>", "exec"), _ns)
    except AssertionError:
        _report("wrong_output", _t)
    except BaseException as e:
        _report("exception", type(e).__name__ + ": " + str(e) + " | " + _t)
_report("pass")
"""


def _limit_resources() -> None:  # pragma: no cover - Linux only, inside the child
    import resource

    gib = 1024 ** 3
    resource.setrlimit(resource.RLIMIT_AS, (2 * gib, 2 * gib))
    resource.setrlimit(resource.RLIMIT_FSIZE, (16 * 1024 ** 2, 16 * 1024 ** 2))


def run_job(job: dict) -> dict:
    """Run one program against its tests in a fresh, isolated Python process."""
    source = _HARNESS.format(mark=RESULT_MARK, setup=job.get("setup") or "",
                             program=job["program"], tests=list(job["tests"]))
    start = time.time()
    with tempfile.TemporaryDirectory() as tmp:
        path = os.path.join(tmp, "harness.py")
        with open(path, "w", encoding="utf-8", newline="\n") as f:
            f.write(source)
        try:
            proc = subprocess.run(
                [sys.executable, "-I", path], cwd=tmp, capture_output=True, text=True,
                timeout=job.get("timeout_s", 10), env={"PATH": os.environ.get("PATH", "")},
                stdin=subprocess.DEVNULL,
                preexec_fn=_limit_resources if os.name == "posix" else None,
            )
        except subprocess.TimeoutExpired:
            return {"job_id": job["job_id"], "status": "timeout", "detail": "", "seconds": round(time.time() - start, 2)}
    result = {"status": "exception", "detail": (proc.stderr or "no result")[-300:]}
    for line in reversed(proc.stdout.splitlines()):
        if line.startswith(RESULT_MARK):
            result = json.loads(line[len(RESULT_MARK):])
            break
    return {"job_id": job["job_id"], **result, "seconds": round(time.time() - start, 2)}


def run_jobs(jobs_path: str, results_path: str) -> None:
    with open(jobs_path, encoding="utf-8") as f:
        jobs = [json.loads(line) for line in f if line.strip()]
    with open(results_path, "w", encoding="utf-8", newline="\n") as out:
        for job in jobs:
            out.write(json.dumps(run_job(job)) + "\n")


if __name__ == "__main__":
    if len(sys.argv) != 4 or sys.argv[1] != "run":
        raise SystemExit("usage: python code_exec.py run JOBS.jsonl RESULTS.jsonl")
    run_jobs(sys.argv[2], sys.argv[3])
