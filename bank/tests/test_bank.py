"""CPU tests for the math bank pipeline (no downloads, no GPU)."""

import pytest

from bank.generate import resolve_dtype
from bank.common import (
    extract_boxed,
    extract_last_number,
    gsm8k_reference,
    normalize_number,
    token_jaccard,
)
from bank.label import label_row
from bank.make_pairs import build_pairs, dedupe
from bank.select_questions import select


# ---------------------------------------------------------------- numbers

@pytest.mark.parametrize(
    "raw, expected",
    [
        ("41", "41"),
        (" 1,000 ", "1000"),
        ("$18.00", "18"),
        ("18.0.", "18"),
        ("\\$5", "5"),
        ("50\\%", "50"),
        ("\\text{12}", "12"),
        ("-3", "-3"),
        ("2.50", "2.5"),
        ("0.125", "0.125"),
        ("twelve", None),
        ("", None),
        (None, None),
    ],
)
def test_normalize_number(raw, expected):
    assert normalize_number(raw) == expected


def test_gsm8k_reference():
    assert gsm8k_reference("She pays 3*4=<<3*4=12>>12 dollars.\n#### 12") == "12"
    assert gsm8k_reference("Total is 1,200.\n#### 1,200") == "1200"
    assert gsm8k_reference("no marker here") is None


# ---------------------------------------------------------------- extraction

def test_extract_boxed_takes_last_and_handles_nesting():
    text = "First \\boxed{3}, then the answer is \\boxed{\\frac{1}{2}}."
    assert extract_boxed(text) == "\\frac{1}{2}"
    assert extract_boxed("The answer is \\boxed{ 41 }") == "41"


def test_extract_boxed_missing_or_unbalanced():
    assert extract_boxed("The answer is 41.") is None
    assert extract_boxed("\\boxed{41") is None


def test_extract_last_number():
    assert extract_last_number("We get 3 apples, then 1,200 total.") == "1,200"
    assert extract_last_number("Answer: -2.5") == "-2.5"
    assert extract_last_number("no numbers") is None


def test_token_jaccard():
    assert token_jaccard("a b c", "a b c") == 1.0
    assert token_jaccard("A  b", "a b") == 1.0
    assert token_jaccard("a b", "c d") == 0.0


# ---------------------------------------------------------------- labeling

def _gen(text, finish="stop", cid="q1__g__0", pid="q1", gen="g"):
    return {"candidate_id": cid, "problem_id": pid, "generator": gen, "text": text, "finish_reason": finish}


def test_label_correct_and_wrong():
    ok = label_row(_gen("2+2=4, so \\boxed{4}"), "4", use_math_verify=False)
    assert ok["status"] == "ok" and ok["label"] == 1
    bad = label_row(_gen("2+3=5, so \\boxed{5}"), "4", use_math_verify=False)
    assert bad["status"] == "ok" and bad["label"] == 0


def test_label_no_answer_and_truncated():
    assert label_row(_gen("The answer is 4."), "4", use_math_verify=False)["status"] == "no_answer"
    cut = label_row(_gen("\\boxed{4}", finish="length"), "4", use_math_verify=False)
    assert cut["status"] == "truncated" and cut["label"] is None


def test_label_checkers_disagree():
    # \boxed{4} is correct, but the last number written is 7, so checker 2 says wrong.
    row = label_row(_gen("\\boxed{4} (checked against 7)"), "4", use_math_verify=False)
    assert row["status"] == "checker_disagree" and row["label"] is None


# ---------------------------------------------------------------- pairing

def _cell(pid, texts_labels, gen="g"):
    gens, labels = [], {}
    for k, (text, label) in enumerate(texts_labels):
        cid = f"{pid}__{gen}__{k}"
        gens.append(_gen(text, cid=cid, pid=pid, gen=gen))
        labels[cid] = {"status": "ok" if label is not None else "no_answer", "label": label}
    return gens, labels


def test_pairs_only_from_mixed_cells():
    g1, l1 = _cell("q1", [("right one", 1), ("wrong one here", 0), ("right again alt", 1)])
    g2, l2 = _cell("q2", [("all right a", 1), ("all right b c", 1)])
    g3, l3 = _cell("q3", [("all wrong x", 0)])
    pairs, summary = build_pairs(g1 + g2 + g3, {**l1, **l2, **l3}, seed=544)
    assert summary == {"cells_with_usable_answers": 3, "cells_all_correct": 1, "cells_all_wrong": 1, "pairs": 1}
    good, bad = pairs[0]
    assert good["problem_id"] == bad["problem_id"] == "q1"


def test_pairs_ignore_unusable_answers():
    g, labels = _cell("q1", [("right", 1), ("no boxed answer", None)])
    pairs, summary = build_pairs(g, labels, seed=1)
    assert pairs == [] and summary["cells_all_correct"] == 1


def test_pairs_are_seeded():
    g, labels = _cell("q1", [("r a", 1), ("r b x", 1), ("r c y z", 1), ("w a", 0), ("w b q", 0)])
    first, _ = build_pairs(g, labels, seed=7)
    second, _ = build_pairs(g, labels, seed=7)
    assert [(a["candidate_id"], b["candidate_id"]) for a, b in first] == [
        (a["candidate_id"], b["candidate_id"]) for a, b in second
    ]


def test_dedupe_collapses_near_identical_answers():
    a = {"text": "so the total is 12 apples"}
    b = {"text": "So the total is 12   apples"}
    c = {"text": "a completely different line of reasoning"}
    assert dedupe([a, b, c]) == [a, c]


# ---------------------------------------------------------------- generation settings

def test_resolve_dtype_explicit_choice_passes_through():
    assert resolve_dtype("float16") == "float16"
    assert resolve_dtype("bfloat16") == "bfloat16"


def test_resolve_dtype_auto_falls_back_without_bf16_gpu():
    # On a machine without a CUDA GPU (like CI or a laptop), auto must not pick bfloat16.
    import torch

    if not torch.cuda.is_available():
        assert resolve_dtype("auto") == "float16"


# ---------------------------------------------------------------- selection

def test_select_is_seeded_sorted_and_labeled():
    rows = [{"question": f"q{i}", "answer": f"work\n#### {i}"} for i in range(50)]
    first = select(rows, 10, seed=544)
    assert first == select(rows, 10, seed=544)
    idx = [q["source_index"] for q in first]
    assert idx == sorted(idx) and len(set(idx)) == 10
    assert all(q["reference"] == str(q["source_index"]) for q in first)
    assert first[0]["problem_id"] == f"gsm8k-train-{idx[0]:05d}"
