"""Essential CPU tests for the math bank (no downloads, no GPU).

Kept to the checks where a bug would silently corrupt the bank or break PLAN.md's rules:
correct labels, unusable answers dropped, checkers must agree, one correct + one wrong
answer per pair (subtle errors kept), labels kept out of the bank, exclusions applied.
"""

import json
from collections import defaultdict

import pytest

from bank.common import extract_boxed, gsm8k_reference, normalize_number, sha256_bytes, sha256_text
from bank.generate import hf_finish
from bank.label import label_row
from bank.make_pairs import build_pairs, main
from bank.select_questions import select


# ---------------------------------------------------------------- reading answers and references

@pytest.mark.parametrize(
    "raw, expected",
    [("41", "41"), (" 1,000 ", "1000"), ("$18.00", "18"), ("2.50", "2.5"),
     ("\\text{12}", "12"), ("twelve", None), (None, None)],
)
def test_normalize_number(raw, expected):
    assert normalize_number(raw) == expected


def test_gsm8k_reference_and_boxed_extraction():
    assert gsm8k_reference("Total is 1,200.\n#### 1,200") == "1200"
    assert extract_boxed("First \\boxed{3}, then \\boxed{\\frac{1}{2}}.") == "\\frac{1}{2}"
    assert extract_boxed("The answer is 41.") is None


def test_select_questions_is_seeded_with_correct_references():
    rows = [{"question": f"q{i}", "answer": f"work\n#### {i}"} for i in range(50)]
    first = select(rows, 10, seed=544)
    assert first == select(rows, 10, seed=544)
    assert all(q["reference"] == str(q["source_index"]) for q in first)


# ---------------------------------------------------------------- labeling

def _gen(text, finish="stop", cid="q1__g__0", pid="q1", gen="g"):
    return {"candidate_id": cid, "problem_id": pid, "generator": gen, "text": text, "finish_reason": finish}


def _label(text, ref, **kw):
    return label_row(_gen(text, **kw), ref, use_math_verify=False)


def test_labels_correct_and_wrong_answers():
    assert _label("2+2=4, so \\boxed{4}", "4")["label"] == 1
    assert _label("2+3=5, so \\boxed{5}", "4")["label"] == 0
    # Units or an equation in the box are still read; an ambiguous box is not guessed.
    assert _label("\\boxed{41 \\text{ dollars}}", "41")["label"] == 1
    assert _label("So \\boxed{200 - 159 = 41}", "41")["label"] == 1
    assert _label("\\boxed{3 \\times 4}", "12")["status"] == "no_answer"


def test_unusable_answers_get_no_label():
    assert _label("The answer is 4.", "4")["status"] == "no_answer"
    assert _label("\\boxed{4}", "4", finish="length")["status"] == "truncated"
    assert _label("As an AI language model, \\boxed{4}", "4")["status"] == "identity_string"
    # Checkers disagree: \boxed{4} is right, but the last number written is 7.
    assert _label("\\boxed{4} (checked against 7)", "4")["status"] == "checker_disagree"
    assert all(_label(t, "4")["label"] is None for t in ("The answer is 4.", "\\boxed{4} (checked against 7)"))


def test_hf_backend_does_not_mark_short_answers_truncated():
    eos = {9}
    assert hf_finish([5, 6, 9, 9, 9], eos, max_new_tokens=5) == (2, "stop")
    assert hf_finish([5, 6, 8, 4, 3], eos, max_new_tokens=5) == (5, "length")


# ---------------------------------------------------------------- pairing

def _cell(pid, texts_labels, gen="g"):
    gens, labels = [], {}
    for k, (text, label) in enumerate(texts_labels):
        cid = f"{pid}__{gen}__{k}"
        gens.append(_gen(text, cid=cid, pid=pid, gen=gen))
        labels[cid] = {"status": "ok" if label is not None else "no_answer", "label": label}
    return gens, labels


def test_pairs_only_from_questions_with_both_a_correct_and_a_wrong_answer():
    g1, l1 = _cell("q1", [("right one", 1), ("wrong one here", 0), ("no answer", None)])
    g2, l2 = _cell("q2", [("all right a", 1), ("all right b c", 1)])
    g3, l3 = _cell("q3", [("all wrong x", 0)])
    pairs, summary = build_pairs(g1 + g2 + g3, {**l1, **l2, **l3}, seed=544)
    assert summary["pairs"] == 1 and summary["cells_all_correct"] == 1 and summary["cells_all_wrong"] == 1
    good, bad = pairs[0]
    assert (good["candidate_id"], bad["candidate_id"]) == ("q1__g__0", "q1__g__1")


def test_subtle_wrong_answer_is_kept_not_treated_as_a_duplicate():
    shared = " ".join(f"step{i}" for i in range(40))
    g, labels = _cell("q1", [(f"{shared} so the shoes cost 41", 1), (f"{shared} so the shoes cost 42", 0)])
    assert build_pairs(g, labels, seed=1)[1]["pairs"] == 1


# ---------------------------------------------------------------- the written bank (end to end)

def _write_inputs(d, n_questions=6):
    """Synthetic questions/generations/labels: every question gets 2 right and 2 wrong answers."""
    qs, gens, labs = [], [], []
    for i in range(n_questions):
        pid = f"gsm8k-train-{i:05d}"
        qs.append({"problem_id": pid, "source": "gsm8k/train", "source_index": i, "domain": "math",
                   "question": f"question {i}", "reference": str(i)})
        for k in range(4):
            text = f"answer {i} variant {k} with distinct words {'x' * k} {i * 7 + k}"
            cid = f"{pid}__phi4mini__{k}"
            gens.append({"candidate_id": cid, "problem_id": pid, "generator": "microsoft/Phi-4-mini-instruct",
                         "generator_revision": None, "settings": {}, "seed": 544 + i, "sample_index": k,
                         "text": text, "text_sha256": sha256_text(text), "finish_reason": "stop", "n_tokens": 9})
            label = 1 if k < 2 else 0
            labs.append({"candidate_id": cid, "problem_id": pid, "reference": str(i),
                         "extracted": str(i if label else i + 1), "checker1_boxed": bool(label),
                         "checker2": bool(label), "checker2_name": "math_verify", "status": "ok",
                         "label": label, "checker_version": "bank.label/2+math_verify==0.9.0"})
    for name, rows in (("questions.jsonl", qs), ("generations.jsonl", gens), ("labeled.jsonl", labs)):
        (d / name).write_text("".join(json.dumps(r) + "\n" for r in rows))


def _run(d, out):
    main(["--questions", str(d / "questions.jsonl"), "--generations", str(d / "generations.jsonl"),
          "--labeled", str(d / "labeled.jsonl"), "--out", str(out)])
    rows = lambda f: [json.loads(l) for l in (out / f).read_text().splitlines()]
    return rows("dev_bank.jsonl"), rows("dev_labels.jsonl"), json.loads((out / "dev_bank_summary.json").read_text())


def test_bank_file_hides_labels_and_pairs_are_one_right_one_wrong(tmp_path):
    _write_inputs(tmp_path)
    bank, labels, summary = _run(tmp_path, tmp_path)
    assert summary["pairs"] == 6
    forbidden = {"label", "reference", "extracted", "status", "checker1_boxed", "checker2"}
    assert all(not (forbidden & set(row)) for row in bank)
    assert [r["candidate_id"] for r in labels] == [r["candidate_id"] for r in bank]
    lab = {r["candidate_id"]: r["label"] for r in labels}
    by_pair = defaultdict(list)
    for row in bank:
        by_pair[row["pair_id"]].append(row)
    assert all(sorted(lab[r["candidate_id"]] for r in v) == [0, 1] for v in by_pair.values())
    # Row order follows answer hashes, never the label (a seeded or label-based shuffle fails here).
    assert all(v[0]["text_sha256"] < v[1]["text_sha256"] for v in by_pair.values())


def test_bank_rows_have_the_fields_scorers_read(tmp_path):
    # Every scorer reads candidate_id, problem and candidate; "question" would crash them.
    _write_inputs(tmp_path)
    bank, _, _ = _run(tmp_path, tmp_path)
    assert all({"candidate_id", "problem", "candidate"} <= set(r) and "question" not in r for r in bank)
    assert bank[0]["problem"].startswith("question ")


def test_written_files_use_lf_so_hashes_match_on_every_platform(tmp_path):
    # Windows text mode would write CRLF, so the summary's hashes would describe bytes
    # that git (eol=lf) never stores and no other machine reproduces.
    _write_inputs(tmp_path)
    _, _, summary = _run(tmp_path, tmp_path)
    for name in ("dev_bank.jsonl", "dev_labels.jsonl", "dev_bank_summary.json"):
        assert b"\r" not in (tmp_path / name).read_bytes()
    for name, digest in summary["outputs"].items():
        assert sha256_bytes((tmp_path / name).read_bytes()) == digest


def test_missing_generator_revision_comes_from_provenance(tmp_path):
    _write_inputs(tmp_path)
    bank, _, summary = _run(tmp_path, tmp_path)
    assert all(r["generator_revision"] is None for r in bank)
    (tmp_path / "generations_provenance.json").write_text(
        json.dumps({"generator": "microsoft/Phi-4-mini-instruct", "generator_revision": "abc123"}))
    bank, _, summary = _run(tmp_path, tmp_path)
    assert all(r["generator_revision"] == "abc123" for r in bank)
    assert summary["generator_revisions_from_provenance"]["microsoft/Phi-4-mini-instruct"] == "abc123"


def test_exclusions_are_applied_and_typos_stop_the_run(tmp_path):
    _write_inputs(tmp_path)
    out = tmp_path / "out"
    out.mkdir()
    (tmp_path / "excluded_questions.txt").write_text("gsm8k-train-00002  # broken for the test\n")
    bank, _, summary = _run(tmp_path, out)  # exclusions live beside the questions, whatever --out is
    assert summary["cells_excluded"] == 1 and all(r["problem_id"] != "gsm8k-train-00002" for r in bank)
    (tmp_path / "excluded_questions.txt").write_text("gsm8k-train-2  # typo\n")
    with pytest.raises(SystemExit):
        _run(tmp_path, out)
