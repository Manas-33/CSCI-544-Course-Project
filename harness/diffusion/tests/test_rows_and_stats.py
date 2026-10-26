"""CPU unit tests for score rows, the scoring loop and the check statistics."""
import json
import math

import pytest

from harness.diffusion.checks.common import auroc, ranks, spearman
from harness.diffusion.interface import (
    SCORE_ROW_FIELDS, read_jsonl, score_row, t1_message, write_jsonl,
)
from harness.diffusion.checks.check_parity import balanced_items
from harness.diffusion.run_score import previous_rows, score_items
from harness.diffusion.scorer import SPECS, LLaDAScorer, Verdict

from .mocks import MASK, MockTokenizer, SlotMarkerModel

V = Verdict(score=1.5, p_yes=0.6, p_no=0.134, mass=0.734, top_id=10)


def test_score_row_has_the_plan_fields():
    row = score_row(bank_hash="abc", candidate_id="c1", model="llada", revision="r",
                    template="T1", n_slots=8, verdict=V, runtime_s=0.2, peak_mem_gb=17.1)
    assert tuple(row) == SCORE_ROW_FIELDS
    assert (row["score"], row["p_yes"], row["p_no"], row["yes_no_mass"]) == (1.5, 0.6, 0.134, 0.734)
    assert json.loads(json.dumps(row)) == row


def test_score_row_status_rules():
    common = dict(bank_hash="abc", candidate_id="c1", model="dream", revision="r",
                  template="T1", n_slots=1)
    row = score_row(status="too_long", **common)
    assert row["status"] == "too_long" and row["score"] is None
    with pytest.raises(ValueError):
        score_row(status="ok", **common)          # ok needs a verdict
    with pytest.raises(ValueError):
        score_row(status="error", verdict=V, **common)
    with pytest.raises(ValueError):
        score_row(status="oops", **common)


def test_jsonl_round_trip(tmp_path):
    path = tmp_path / "rows.jsonl"
    write_jsonl(path, [{"a": 1}])
    write_jsonl(path, [{"a": 2}], append=True)
    assert list(read_jsonl(path)) == [{"a": 1}, {"a": 2}]


def test_t1_message_contains_problem_candidate_and_question():
    msg = t1_message("What is 2 + 2?", "4")
    assert msg.startswith("Problem:\nWhat is 2 + 2?") and "4" in msg
    assert msg.endswith("Is the final answer correct? Answer Yes or No.")


def _scorer(model, max_len=4096):
    spec = type(SPECS["llada"])(**{**SPECS["llada"].__dict__, "mask_id": MASK, "max_len": max_len})
    return LLaDAScorer(model, MockTokenizer(), spec, 8, device="cpu")


ITEMS = [{"candidate_id": f"c{i}", "problem": "add two numbers " * (i + 1), "candidate": "four"}
         for i in range(5)]


def test_score_items_writes_one_row_per_item():
    rows = list(score_items(_scorer(SlotMarkerModel()), ITEMS, batch_size=2, bank_hash="h"))
    assert [r["candidate_id"] for r in rows] == [it["candidate_id"] for it in ITEMS]
    assert all(r["status"] == "ok" and r["score"] > 0 and r["bank_hash"] == "h" for r in rows)
    assert all(r["peak_mem_gb"] is None and r["runtime_s"] >= 0 for r in rows)  # CPU
    assert rows[0]["revision"] == SPECS["llada"].revision


def test_score_items_marks_too_long_and_isolates_errors():
    class FailsOnLongInputs(SlotMarkerModel):
        def __call__(self, input_ids, attention_mask=None, position_ids=None):
            if input_ids.shape[1] > 30:
                raise RuntimeError("boom")
            return super().__call__(input_ids, attention_mask, position_ids)

    scorer = _scorer(FailsOnLongInputs(), max_len=35)
    rows = {r["candidate_id"]: r["status"] for r in score_items(scorer, ITEMS, batch_size=4)}
    lengths = {it["candidate_id"]: len(scorer.prompt_ids(t1_message(it["problem"], it["candidate"])))
               for it in ITEMS}
    for cid, n in lengths.items():
        expected = "too_long" if n + 8 > 35 else ("error" if n + 8 > 30 else "ok")
        assert rows[cid] == expected, (cid, n)
    assert set(rows.values()) == {"ok", "error", "too_long"}


def test_auroc():
    assert auroc([0.9, 0.8, 0.3, 0.1], [1, 1, 0, 0]) == 1.0
    assert auroc([0.1, 0.3, 0.8, 0.9], [1, 1, 0, 0]) == 0.0
    assert auroc([0.5, 0.5], [1, 0]) == 0.5
    assert auroc([3, 1, 2], [1, 1, 0]) == 0.5
    with pytest.raises(ValueError):
        auroc([1, 2], [1, 1])


def test_spearman_with_ties():
    assert ranks([5, 6, 7, 8, 7]) == [1, 2, 3.5, 5, 3.5]
    assert spearman([1, 2, 3, 4, 5], [5, 6, 7, 8, 7]) == pytest.approx(8 / math.sqrt(95))
    assert spearman([1, 2, 3], [3, 2, 1]) == pytest.approx(-1.0)
    assert math.isnan(spearman([1, 2, 3], [4, 4, 4]))


def _rows(path, **settings):
    base = dict(bank_hash="h", model="llada", template="T1", n_slots=1)
    base.update(settings)
    return [score_row(candidate_id=c, revision="r", verdict=None if s != "ok" else V, status=s, **base)
            for c, s in (("a", "ok"), ("b", "error"), ("c", "too_long"))]


def test_resume_keeps_finished_rows_and_retries_errors(tmp_path):
    out = tmp_path / "scores.jsonl"
    write_jsonl(out, _rows(out))
    kept = previous_rows(out, model="llada", template="T1", n_slots=1, bank_hash="h")
    assert [r["candidate_id"] for r in kept] == ["a", "c"]
    assert [r["candidate_id"] for r in read_jsonl(out)] == ["a", "c"]  # the error row is gone, to be rescored


def test_resume_refuses_a_file_scored_with_other_settings(tmp_path):
    out = tmp_path / "scores.jsonl"
    write_jsonl(out, _rows(out, n_slots=1))
    with pytest.raises(SystemExit, match="other settings"):
        previous_rows(out, model="llada", template="T1", n_slots=8, bank_hash="h")
    assert previous_rows(tmp_path / "new.jsonl", model="llada", template="T1", n_slots=8, bank_hash="h") == []


def test_parity_items_cover_both_domains():
    items = [{"candidate_id": f"c{i}", "domain": "code"} for i in range(24)] + \
            [{"candidate_id": f"m{i}", "domain": "math"} for i in range(26)]
    chosen = balanced_items(items, 20)
    assert len(chosen) == 20
    assert sum(it["domain"] == "math" for it in chosen) == 10
