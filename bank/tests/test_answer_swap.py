"""CPU tests for the answer-swap items (PLAN.md §4 test 4): the edit itself, the draw, what the file reveals."""

import json
import random
from collections import Counter, defaultdict

import pytest

from bank.answer_swap import build, draw, last_box_span, main, make_swap, recorded_lengths, same_answer, swap_last_box
from bank.common import extract_boxed, read_jsonl, sha256_text, write_jsonl
from harness.data import BANK_MAPPING, load_records
from harness.diffusion.interface import read_jsonl as diffusion_read

PILOT = "bank/pilot/swap"


# ---------------------------------------------------------------- the edit

@pytest.mark.parametrize("text, answer, expected", [
    ("so \\boxed{5}.", "7", "so \\boxed{7}."),
    ("\\boxed{3} then \\boxed{4}, done", "9", "\\boxed{3} then \\boxed{9}, done"),          # several boxes: the last
    ("\\boxed{\\frac{1}{2}}$ units", "\\frac{3}{4}", "\\boxed{\\frac{3}{4}}$ units"),      # nested braces
    ("\\boxed{200 \\text{ feet}}", "56 \\text{ kg}", "\\boxed{56 \\text{ kg}}"),
    ("\\boxed{\\boxed{5}}", "6", "\\boxed{\\boxed{6}}"),
    ("a {b} \\boxed{{x}_{1}} {c}", "\\{1\\}", "a {b} \\boxed{\\{1\\}} {c}"),
])
def test_only_the_last_box_changes(text, answer, expected):
    swapped = swap_last_box(text, answer)
    assert swapped == expected
    assert extract_boxed(swapped) == answer
    a, b = last_box_span(text)
    assert text[a:b].strip() == extract_boxed(text)
    assert (text[:a], text[b:]) == (swapped[:a], swapped[a + len(answer):])


@pytest.mark.parametrize("text", ["no box", "\\boxed{5", "\\boxed{\\frac{1}{2}", "\\boxed{1} and \\boxed{2"])
def test_text_without_a_closed_last_box_is_refused(text):
    assert last_box_span(text) is None and extract_boxed(text) is None
    with pytest.raises(ValueError):
        swap_last_box(text, "1")


def _row(text, domain="math", **kw):
    return {"candidate": text, "domain": domain, **kw}


@pytest.mark.parametrize("old, new", [
    ("7", "7"), ("7", "7.0"), ("1,000", "1000"), ("\\frac{1}{2}", "0.5"), ("1/2", "\\frac{2}{4}"),
    ("7", "7 \\text{ dollars}"), ("x = 7", "7"), ("120^\\circ", "120"), ("A B", "a  b"),
])
def test_equal_answers_are_not_a_swap(old, new):
    assert same_answer(old, new)
    text, reason = make_swap(_row(f"\\boxed{{{old}}}"), _row(f"\\boxed{{{new}}}"), None)
    assert text is None and "equals the original" in reason


def test_different_answers_are_a_swap():
    for old, new in (("7", "8"), ("\\frac{1}{2}", "\\frac{1}{3}"), ("16", "4R"), ("243", "336.00"), ("5.5", "\\frac{73}{22}")):
        assert not same_answer(old, new)
        text, reason = make_swap(_row(f"see \\boxed{{{old}}} ok"), _row(f"\\boxed{{{new}}}"), old)
        assert reason is None and text == f"see \\boxed{{{new}}} ok"


def test_swap_to_the_reference_or_from_no_answer_is_refused():
    assert "reference" in make_swap(_row("\\boxed{7}"), _row("\\boxed{9}"), "9")[1]
    assert "no readable" in make_swap(_row("\\boxed{7}"), _row("no box"), None)[1]
    assert "no readable" in make_swap(_row("\\boxed{7}"), _row("\\boxed{ }"), None)[1]
    same = make_swap(_row("def f():\n  return 1", "code"), _row("def f():\n    return 1", "code"), None)
    assert same[0] is None and "same" in same[1]


# ---------------------------------------------------------------- the draw

def _bank(problems=6, per_problem=2, domains=("math", "code")):
    bank, labels = [], {}
    for d in domains:
        for p in range(problems):
            for g in range(per_problem):
                for lab in (1, 0):
                    cid = f"{d}{p}__g{g}__{lab}"
                    body = f"work {p} \\boxed{{{p + 10 * (1 - lab) + 100}}}" if d == "math" else f"def f():\n    return {p}, {g}, {lab}"
                    bank.append({"candidate_id": cid, "pair_id": f"pilot:{d}{p}:g{g}", "problem_id": f"{d}{p}", "source": d,
                                 "domain": d, "problem": f"question {d} {p}", "candidate": body + " " * g,
                                 "text_sha256": sha256_text(body + " " * g), "generator": f"gen{g}",
                                 "generator_revision": "r", "split": "pilot"})
                    labels[cid] = {"candidate_id": cid, "label": lab, "reference": str(p + 100) if d == "math" else None}
    return bank, labels


def test_every_problem_is_used_once_before_any_is_used_twice():
    bank, labels = _bank(problems=6, per_problem=2)
    good = [r for r in bank if r["domain"] == "math" and labels[r["candidate_id"]]["label"] == 1]
    few = draw(good, 5, 544, "math")
    assert len({r["problem_id"] for r in few}) == 5
    more = draw(good, 8, 544, "math")
    uses = Counter(r["problem_id"] for r in more)
    assert len(uses) == 6 and sorted(uses.values()) == [1, 1, 1, 1, 2, 2]
    cells = [(r["problem_id"], r["generator"]) for r in more]
    assert len(set(cells)) == len(cells)
    with pytest.raises(SystemExit):
        draw(good, 13, 544, "math")


def test_the_draw_is_seeded_and_ignores_input_order():
    bank, labels = _bank()
    shuffled = list(bank)
    random.Random(1).shuffle(shuffled)
    first = build(bank, labels, None, per_domain=8)
    assert first == build(shuffled, labels, None, per_domain=8)
    assert first[0] != build(bank, labels, None, seed=545, per_domain=8)[0]


def test_unusable_candidates_are_left_out_with_a_reason():
    bank, labels = _bank(problems=4, per_problem=1, domains=("math", "code"))
    by_id = {r["candidate_id"]: r for r in bank}
    by_id["math0__g0__0"]["candidate"] = "\\boxed{100.0}"              # equal in value to the original's 100
    lengths = {r["candidate_id"]: 900 for r in bank}
    lengths["math1__g0__1"] = 1759                                     # fits alone; a 3-byte answer could not
    lengths["code2__g0__0"] = 1761                                     # the partner program is too long
    _, keys, summary = build(bank, labels, lengths, per_domain=2)
    assert set(summary["dropped"]) == {"math0__g0__1", "math1__g0__1", "code2__g0__1"}
    assert "equals the original" in summary["dropped"]["math0__g0__1"]
    assert "1760" in summary["dropped"]["math1__g0__1"]
    assert not {k["source_candidate_id"] for k in keys} & set(summary["dropped"])
    with pytest.raises(SystemExit, match="only 2 usable"):
        build(bank, labels, lengths, per_domain=3)


def test_missing_lengths_labels_and_broken_pairs_are_errors():
    bank, labels = _bank()
    with pytest.raises(SystemExit, match="no prompt length"):
        build(bank, labels, {"math0__g0__1": 5}, per_domain=2)
    with pytest.raises(SystemExit, match="no 0/1 label"):
        build(bank, {k: v for k, v in labels.items() if k != "math0__g0__1"}, None, per_domain=2)
    labels["math0__g0__0"]["label"] = 1
    with pytest.raises(SystemExit, match="one correct and one wrong"):
        build(bank, labels, None, per_domain=2)


def test_recorded_lengths_take_the_longest_model(tmp_path):
    write_jsonl(tmp_path / "a" / "scores_A.jsonl", [{"candidate_id": "x", "input_tokens": 10}])
    write_jsonl(tmp_path / "a" / "scores_B.jsonl", [{"candidate_id": "x", "input_tokens": 14}, {"candidate_id": "y", "input_tokens": None}])
    assert recorded_lengths([str(tmp_path)]) == {"x": 14}


# ---------------------------------------------------------------- the file

def test_file_loads_everywhere_and_holds_no_label(tmp_path):
    bank, labels = _bank()
    write_jsonl(tmp_path / "bank.jsonl", bank)
    write_jsonl(tmp_path / "labels.jsonl", labels.values())
    write_jsonl(tmp_path / "s" / "scores_A.jsonl", [{"candidate_id": r["candidate_id"], "input_tokens": 50} for r in bank])
    main(["--bank", str(tmp_path / "bank.jsonl"), "--labels", str(tmp_path / "labels.jsonl"),
          "--scores", str(tmp_path / "s"), "--out", str(tmp_path / "swap"), "--per-domain", "8"])
    path = tmp_path / "swap" / "swap_items.jsonl"
    records, _ = load_records(path, BANK_MAPPING, "dev_bank")
    rows = list(diffusion_read(path))
    assert len(records) == len(rows) == 32
    assert all(set(r) >= {"candidate_id", "problem", "candidate"} for r in rows)
    for r in rows:
        assert not {"label", "reference", "role", "extracted", "status", "seed"} & set(r)
    assert b"\r" not in path.read_bytes()


# ---------------------------------------------------------------- the committed pilot items

def test_pilot_items_follow_plan():
    items, keys = read_jsonl(f"{PILOT}/swap_items.jsonl"), {k["candidate_id"]: k for k in read_jsonl(f"{PILOT}/swap_key.jsonl")}
    bank = {r["candidate_id"]: r for r in read_jsonl("bank/pilot/pilot_bank.jsonl")}
    labels = {r["candidate_id"]: r for r in read_jsonl("bank/pilot/pilot_labels.jsonl")}
    load_records(f"{PILOT}/swap_items.jsonl", BANK_MAPPING, "dev_bank")
    assert Counter(r["domain"] for r in items) == {"math": 200, "code": 200}
    pairs = defaultdict(dict)
    for r in items:
        pairs[r["pair_id"]][keys[r["candidate_id"]]["role"]] = r
    problems = {d: {p["original"]["problem_id"] for p in pairs.values() if p["original"]["domain"] == d} for d in ("math", "code")}
    assert (len(problems["math"]), len(problems["code"])) == (97, 86)
    for p in pairs.values():
        original, swap = p["original"], p["swap"]
        k = keys[original["candidate_id"]]
        good, bad = bank[k["source_candidate_id"]], bank[k["partner_candidate_id"]]
        assert good["pair_id"] == bad["pair_id"]
        assert (labels[good["candidate_id"]]["label"], labels[bad["candidate_id"]]["label"]) == (1, 0)
        assert original["candidate"] == good["candidate"]
        if original["domain"] == "code":
            assert swap["candidate"] == bad["candidate"]
            continue
        assert extract_boxed(swap["candidate"]) == extract_boxed(bad["candidate"])
        assert not same_answer(extract_boxed(swap["candidate"]), labels[good["candidate_id"]]["reference"])
        a, b = last_box_span(original["candidate"])
        assert swap["candidate"].startswith(original["candidate"][:a]) and swap["candidate"].endswith(original["candidate"][b:])


def test_pilot_position_and_id_do_not_reveal_the_original():
    items, keys = read_jsonl(f"{PILOT}/swap_items.jsonl"), {k["candidate_id"]: k for k in read_jsonl(f"{PILOT}/swap_key.jsonl")}
    assert [r["candidate_id"] for r in items] == sorted(r["candidate_id"] for r in items)
    roles = [keys[r["candidate_id"]]["role"] for r in items]
    assert abs(roles[:200].count("original") - 100) <= 25          # no front-loading
    first = {}
    for r, role in zip(items, roles):
        first.setdefault(r["pair_id"], role)
    assert abs(Counter(first.values())["original"] - 100) <= 25    # nor within pairs
    assert len({json.dumps(sorted(r)) for r in items if r["domain"] == "math"}) == 1
