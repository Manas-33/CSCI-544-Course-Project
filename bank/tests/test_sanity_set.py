"""Checks every label in the 50-item sanity set (PLAN.md §4, scoring-code test 1).

This is the machine half of the second check on the labels: each math answer is compared
with the true answer below, and each code answer is run against test cases. A human
second check is still required before any model is scored on the set.
"""

import ast
import json
import re
from collections import Counter
from fractions import Fraction
from pathlib import Path

import pytest

SANITY = Path(__file__).resolve().parents[1] / "sanity" / "sanity_set.jsonl"
DEV_BANK = Path(__file__).resolve().parents[1] / "dev" / "dev_bank.jsonl"
FIELDS = {"candidate_id", "domain", "problem", "candidate", "label", "note"}

# The true answer to each math problem, worked out independently of the candidates.
MATH_ANSWERS = {
    "sanity-m001": "4", "sanity-m002": "12", "sanity-m003": "30", "sanity-m004": "40",
    "sanity-m005": "180", "sanity-m006": "15", "sanity-m007": "20", "sanity-m008": "84",
    "sanity-m009": "81", "sanity-m010": "30", "sanity-m011": "32", "sanity-m012": "7",
    "sanity-m013": "205", "sanity-m014": "550", "sanity-m015": "28", "sanity-m016": "36",
    "sanity-m017": "6", "sanity-m018": "18", "sanity-m019": "3/8", "sanity-m020": "13",
    "sanity-m021": "9", "sanity-m022": "7", "sanity-m023": "150", "sanity-m024": "20",
    "sanity-m025": "70", "sanity-m026": "3/4",
}

# (arguments, expected result) for each code problem, following the problem statement.
CODE_CASES = {
    "sanity-c001": [((2, 3), 5), ((-1, 1), 0)],
    "sanity-c002": [((4,), True), ((7,), False), ((0,), True)],
    "sanity-c003": [(([3, 9, 2],), 9), (([-5, -2],), -2), (([7],), 7)],
    "sanity-c004": [(([3, 9, 2],), 2), (([-5, -2],), -5)],
    "sanity-c005": [((3,), 9), ((-4,), 16), ((0,), 0)],
    "sanity-c006": [(("abc",), "cba"), (("",), "")],
    "sanity-c007": [((-3,), 3), ((5,), 5), ((0,), 0)],
    "sanity-c008": [(("Hello World",), 3), (("xyz",), 0), (("AEIOU",), 5)],
    "sanity-c009": [((0,), 1), ((1,), 1), ((5,), 120)],
    "sanity-c010": [(([1, -2, 3, 0],), 2), (([],), 0)],
    "sanity-c011": [(("racecar",), True), (("abca",), False), (("",), True)],
    "sanity-c012": [(([1, 2, 3],), 3), ((["a"],), "a")],
    "sanity-c013": [((3, 4), 12), ((2, 2), 4)],
    "sanity-c014": [((0,), 32), ((100,), 212), ((-40,), -40)],
    "sanity-c015": [((15,), "FizzBuzz"), ((9,), "Fizz"), ((10,), "Buzz"), ((7,), "7")],
    "sanity-c016": [(([1, 2, 3],), 2), (([5],), 5)],
    "sanity-c017": [(([1, 2, 3, 4],), 6), (([],), 0), (([-2, 3],), -2)],
    "sanity-c018": [(("apple",), True), (("banana",), False)],
    "sanity-c019": [((95,), "A"), ((85,), "B"), ((75,), "C"), ((50,), "F")],
    "sanity-c020": [(("hello big  world",), 3), (("",), 0)],
    "sanity-c021": [((1,), False), ((2,), True), ((9,), False), ((13,), True)],
    "sanity-c022": [((3,), [3, 2, 1]), ((1,), [1])],
    "sanity-c023": [(([1, 2, 1, 3, 2],), [1, 2, 3]), (([],), [])],
    "sanity-c024": [((10, 5), True), ((10, 3), False)],
}

ITEMS = [json.loads(line) for line in SANITY.read_text(encoding="utf-8").splitlines() if line.strip()]
BY_DOMAIN = {d: [it for it in ITEMS if it["domain"] == d] for d in ("math", "code")}


def _value(text):
    """'\\frac{3}{8}' -> Fraction(3, 8); '-5' -> Fraction(-5); '3/4' -> Fraction(3, 4)."""
    frac = re.fullmatch(r"\\frac\{(-?\d+)\}\{(\d+)\}", text)
    return Fraction(int(frac[1]), int(frac[2])) if frac else Fraction(text)


def _boxed(candidate):
    last = candidate.splitlines()[-1]
    match = re.fullmatch(r"The final answer is \\boxed\{(.*)\}\.", last)
    assert match, f"last line is not 'The final answer is \\boxed{{...}}.': {last!r}"
    return match[1]


def test_shape_counts_and_ids():
    assert len(ITEMS) == 50
    assert all(set(it) == FIELDS for it in ITEMS)
    assert len({it["candidate_id"] for it in ITEMS}) == 50
    assert all(it["note"].strip() for it in ITEMS)
    counts = Counter((it["domain"], it["label"]) for it in ITEMS)
    assert counts == {("math", 1): 13, ("math", 0): 13, ("code", 1): 12, ("code", 0): 12}
    for it in ITEMS:
        prefix = "sanity-m" if it["domain"] == "math" else "sanity-c"
        assert re.fullmatch(prefix + r"\d{3}", it["candidate_id"])
    assert set(MATH_ANSWERS) == {it["candidate_id"] for it in BY_DOMAIN["math"]}
    assert set(CODE_CASES) == {it["candidate_id"] for it in BY_DOMAIN["code"]}


def test_outside_the_bank():
    if not DEV_BANK.exists():
        pytest.skip("no development bank")
    bank_problems = {json.loads(l)["problem"] for l in DEV_BANK.read_text(encoding="utf-8").splitlines() if l}
    assert not bank_problems & {it["problem"] for it in ITEMS}


@pytest.mark.parametrize("item", BY_DOMAIN["math"], ids=lambda it: it["candidate_id"])
def test_math_label_matches_the_true_answer(item):
    is_right = _value(_boxed(item["candidate"])) == _value(MATH_ANSWERS[item["candidate_id"]])
    assert item["label"] == int(is_right), item["note"]


@pytest.mark.parametrize("item", BY_DOMAIN["code"], ids=lambda it: it["candidate_id"])
def test_code_label_matches_the_test_cases(item):
    tree = ast.parse(item["candidate"])
    assert len(tree.body) == 1 and isinstance(tree.body[0], ast.FunctionDef), "one function only"
    name = re.search(r"function (\w+)\(", item["problem"])[1]
    assert tree.body[0].name == name
    namespace = {}
    exec(compile(tree, item["candidate_id"], "exec"), namespace)
    passes = all(namespace[name](*args) == expected for args, expected in CODE_CASES[item["candidate_id"]])
    assert item["label"] == int(passes), item["note"]
