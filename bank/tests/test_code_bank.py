"""CPU tests for the code bank: program extraction, the test runner, labels and pairs.

Programs run as plain subprocesses here (--sandbox none); the Docker sandbox runs the
same bank/code_exec.py inside a container.
"""

import json
from argparse import Namespace

import pytest

from bank import code_exec
from bank.common import sha256_text
from bank.generate import apply_domain_defaults, build_messages, questions_domain
from bank.label_code import combine, label_all
from bank.make_pairs import main as make_pairs_main
from bank.select_code_questions import eligible, entry_point, mbpp_prompt


# ---------------------------------------------------------------- extraction

def test_prefilled_reply_is_cut_at_the_closing_fence():
    text = "def f(x):\n    return x\n```\n\nThis function returns its input."
    assert code_exec.extract_program(text) == "def f(x):\n    return x"


def test_sanitize_keeps_the_solution_and_drops_example_code():
    program = (
        "import math\n"
        "MOD = 10 ** 9 + 7\n"
        "unused = print('side effect')\n"
        "def helper(x):\n    return x % MOD\n"
        "def solve(x):\n    return helper(math.floor(x))\n"
        "print(solve(3.5))\n"
        "assert solve(1) == 1\n"
        "if __name__ == '__main__':\n    solve(2)\n"
    )
    clean = code_exec.sanitize(program, "solve")
    assert "import math" in clean and "MOD = 10 ** 9 + 7" in clean and "def helper" in clean
    assert "print" not in clean and "assert" not in clean and "__main__" not in clean and "unused" not in clean


@pytest.mark.parametrize("program", ["def other(x):\n    return x\n", "def solve(x) return x\n", "solve = lambda x: x\n"])
def test_no_parseable_entry_point_means_no_answer(program):
    assert code_exec.sanitize(program, "solve") is None


# ---------------------------------------------------------------- the runner

def _job(program, tests, timeout_s=5, setup=""):
    return {"job_id": "j", "program": program, "setup": setup, "tests": tests, "timeout_s": timeout_s}


@pytest.mark.parametrize(
    "program, status",
    [
        ("def f(x):\n    return x + 1\n", "pass"),
        ("def f(x):\n    return x + 2\n", "wrong_output"),
        ("def f(x):\n    return undefined_name\n", "exception"),
        ("raise ValueError('at import')\n", "exception"),
        ("import sys\ndef f(x):\n    sys.exit(0)\n", "exception"),  # exiting early is not a pass
        ("def f(x):\n    while True:\n        pass\n", "timeout"),
    ],
)
def test_run_job_statuses(program, status):
    result = code_exec.run_job(_job(program, ["assert f(1) == 2", "assert f(5) == 6"], timeout_s=2))
    assert result["status"] == status


def test_setup_code_runs_before_the_tests():
    result = code_exec.run_job(_job("def f():\n    return VALUE\n", ["assert f() == 3"], setup="VALUE = 3"))
    assert result["status"] == "pass"


def test_combine_flaky_and_error_types():
    assert combine([{"status": "pass"}, {"status": "timeout"}]) == ("flaky", None, None)
    assert combine([{"status": "pass"}, {"status": "pass"}]) == ("ok", 1, None)
    assert combine([{"status": "wrong_output"}, {"status": "wrong_output"}]) == ("ok", 0, "wrong_output")
    assert combine([{"status": "exception"}, {"status": "timeout"}]) == ("ok", 0, "mixed_failure")


# ---------------------------------------------------------------- labels and pairs (end to end)

def _questions():
    return [
        {"problem_id": "mbpp-001", "source": "google-research/mbpp", "domain": "code",
         "question": mbpp_prompt("Add one to x.", "assert add_one(1) == 2"), "entry_point": "add_one",
         "tests": ["assert add_one(1) == 2", "assert add_one(-1) == 0"], "setup": "",
         "reference_solution": "def add_one(x):\n    return x + 1\n"},
        {"problem_id": "mbpp-002", "source": "google-research/mbpp", "domain": "code",
         "question": mbpp_prompt("Broken tests.", "assert two() == 3"), "entry_point": "two",
         "tests": ["assert two() == 3"], "setup": "", "reference_solution": "def two():\n    return 2\n"},
    ]


def _gen(pid, k, body, finish="stop"):
    text = body + "\n```\nExplanation follows."
    return {"candidate_id": f"{pid}__phi4mini__{k}", "problem_id": pid, "generator": "microsoft/Phi-4-mini-instruct",
            "generator_revision": "abc", "settings": {}, "seed": 544, "sample_index": k, "text": text,
            "text_sha256": sha256_text(text), "finish_reason": finish, "n_tokens": 20}


def _generations():
    return [
        _gen("mbpp-001", 0, "def add_one(x):\n    return x + 1\n\nprint(add_one(1))"),
        _gen("mbpp-001", 1, "def add_one(x):\n    return x - 1"),
        _gen("mbpp-001", 2, "def plus_one(x):\n    return x + 1"),
        _gen("mbpp-001", 3, "def add_one(x):\n    return x +", finish="length"),
        _gen("mbpp-002", 0, "def two():\n    return 3"),
    ]


def test_label_all_statuses():
    qs = {q["problem_id"]: q for q in _questions()}
    rows = {r["candidate_id"]: r for r in label_all(qs, _generations(), "none", "unused", timeout_s=5)}
    assert (rows["mbpp-001__phi4mini__0"]["status"], rows["mbpp-001__phi4mini__0"]["label"]) == ("ok", 1)
    assert "print" not in rows["mbpp-001__phi4mini__0"]["program"]
    assert (rows["mbpp-001__phi4mini__1"]["label"], rows["mbpp-001__phi4mini__1"]["error_type"]) == (0, "wrong_output")
    assert rows["mbpp-001__phi4mini__2"]["status"] == "no_answer"
    assert rows["mbpp-001__phi4mini__3"]["status"] == "truncated"
    # The reference fails its own tests, so the problem is excluded even though this answer "passes".
    assert rows["mbpp-002__phi4mini__0"]["status"] == "broken_problem"
    assert all(r["checker_version"] == "bank.label_code/1+subprocess" for r in rows.values())


def test_code_pairs_show_the_tested_program_and_keep_labels_out(tmp_path):
    qs, gens = _questions(), _generations()
    labels = label_all({q["problem_id"]: q for q in qs}, gens, "none", "unused", timeout_s=5)
    for name, rows in (("questions.jsonl", qs), ("generations.jsonl", gens), ("labeled.jsonl", labels)):
        (tmp_path / name).write_text("".join(json.dumps(r) + "\n" for r in rows))
    make_pairs_main(["--questions", str(tmp_path / "questions.jsonl"), "--generations", str(tmp_path / "generations.jsonl"),
                     "--labeled", str(tmp_path / "labeled.jsonl"), "--out", str(tmp_path)])
    bank = [json.loads(l) for l in (tmp_path / "dev_bank.jsonl").read_text().splitlines()]
    lab = {r["candidate_id"]: r for r in map(json.loads, (tmp_path / "dev_labels.jsonl").read_text().splitlines())}
    assert len(bank) == 2 and {r["problem_id"] for r in bank} == {"mbpp-001"}
    assert all(r["entry_point"] == "add_one" and "Explanation" not in r["candidate"] for r in bank)
    assert all(r["text_sha256"] == sha256_text(r["candidate"]) for r in bank)
    assert not {"label", "error_type", "run_statuses", "program"} & set().union(*bank)
    assert sorted(lab[r["candidate_id"]]["label"] for r in bank) == [0, 1]


# ---------------------------------------------------------------- generation and selection

def test_code_prompt_prefills_the_reply_and_uses_code_defaults():
    messages = build_messages('"""\nAdd one.\n"""\n', "code")
    assert messages[-1]["role"] == "assistant" and messages[-1]["content"].endswith("```python\n")
    args = Namespace(n_samples=None, temperature=None, top_p=None, max_new_tokens=None)
    apply_domain_defaults(args, "code")
    assert (args.n_samples, args.temperature, args.top_p, args.max_new_tokens) == (8, 0.8, 0.95, 768)
    assert build_messages("What is 2+2?") == build_messages("What is 2+2?", "math")
    with pytest.raises(SystemExit):
        questions_domain([{"domain": "math"}, {"domain": "code"}])


def test_entry_point_and_mbpp_plus_exclusion():
    assert entry_point("def f(a, b):\n    return a\n", ["assert set(f((1,), (2,))) == set()"]) == "f"
    assert entry_point("def f():\n    pass\ndef g():\n    pass\n", ["assert f() == g()"]) is None
    mbpp = [
        {"task_id": 2, "text": "Shared elements.", "code": "def f(a):\n    return a\n", "test_list": ["assert f(1) == 1"]},
        {"task_id": 7, "text": "Something new.", "code": "def g(a):\n    return a\n", "test_list": ["assert g(1) == 1"]},
        {"task_id": 9, "text": "Same text as a plus task.", "code": "def h(a):\n    return a\n", "test_list": ["assert h(1) == 1"]},
    ]
    plus = [{"task_id": "Mbpp/2", "prompt": '"""\nShared elements.\nassert f(1) == 1\n"""\n'},
            {"task_id": "Mbpp/50", "prompt": '"""\nSame text as a plus task.\nassert h(1) == 1\n"""\n'}]
    assert [q["problem_id"] for q in eligible(mbpp, plus)] == ["mbpp-007"]
