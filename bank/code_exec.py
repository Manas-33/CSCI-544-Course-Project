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
import symtable
import sys
import tempfile
import time
import warnings

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


def _module_names_read(program: str, tree: ast.Module) -> set[str]:
    """Module-level names the top-level definitions need.

    - Read inside a def or class body: symtable separates these from parameters and locals,
      so `def f(n)` does not keep a top-level `n = int(input())`. Only the tables of the
      top-level definitions are walked, so lambdas in example code don't count.
    - Read by the definition statement itself, which runs at module level: decorators,
      default values, annotations and class bases or keywords.
    Raises SyntaxError for programs symtable rejects (e.g. `global` on a parameter).
    """
    def walk(table):
        names = {s.get_name() for s in table.get_symbols() if s.is_global() and s.is_referenced()}
        for child in table.get_children():
            names |= walk(child)
        return names

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", SyntaxWarning)
        top = symtable.symtable(program, "<candidate>", "exec")
    defs = [node for node in tree.body if isinstance(node, _DEFS)]
    starts = {(node.name, node.lineno) for node in defs}
    names = set()
    for child in top.get_children():
        if (child.get_name(), child.get_lineno()) in starts:
            names |= walk(child)
    for node in defs:
        if isinstance(node, ast.ClassDef):
            parts = [*node.decorator_list, *node.bases, *(k.value for k in node.keywords)]
        else:
            a = node.args
            params = [*a.posonlyargs, *a.args, *a.kwonlyargs, a.vararg, a.kwarg]
            parts = [*node.decorator_list, *a.defaults, *a.kw_defaults, node.returns,
                     *(x.annotation for x in params if x is not None)]
        names |= {n.id for part in parts if part is not None
                  for n in ast.walk(part) if isinstance(n, ast.Name)}
    return names


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
        with warnings.catch_warnings():  # e.g. "invalid escape sequence" in regex strings
            warnings.simplefilter("ignore", SyntaxWarning)
            tree = ast.parse(program)
        needed = _module_names_read(program, tree)
    except SyntaxError:
        return None
    defs = [node for node in tree.body if isinstance(node, _DEFS)]
    if entry_point not in {n.name for n in defs if not isinstance(n, ast.ClassDef)}:
        return None
    assigns = [node for node in tree.body if isinstance(node, (ast.Assign, ast.AnnAssign))]
    kept_assigns, changed = set(), True
    while changed:  # follow chains such as BASE -> POW -> f
        changed = False
        for i, node in enumerate(assigns):
            if i not in kept_assigns and _assigned_names(node) & needed:
                kept_assigns.add(i)
                changed = True
                if node.value is not None:
                    needed |= {n.id for n in ast.walk(node.value) if isinstance(n, ast.Name)}
    kept_ids = {id(assigns[i]) for i in kept_assigns}
    kept = [node for node in tree.body if isinstance(node, _DEFS + _IMPORTS) or id(node) in kept_ids]
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
                # -s and -P keep the run isolated like -I, but unlike -I they honour the fixed
                # hash seed, so set-order-dependent programs give the same result on every rebuild.
                [sys.executable, "-s", *(["-P"] if sys.version_info >= (3, 11) else []), path],
                cwd=tmp, capture_output=True, text=True, timeout=job.get("timeout_s", 10),
                env={"PATH": os.environ.get("PATH", ""), "PYTHONHASHSEED": str(job.get("hash_seed", 0))},
                stdin=subprocess.DEVNULL,
                # The memory limit is Linux-only (macOS refuses RLIMIT_AS); Docker runs are Linux.
                preexec_fn=_limit_resources if sys.platform.startswith("linux") else None,
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
