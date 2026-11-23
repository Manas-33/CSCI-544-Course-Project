"""CPU tests for the full bank (PLAN.md §2): selection, resumable generation, the draw rule, the split and label
custody. No downloads, models or API calls: sources, generators and checkers are small fakes."""

import copy
import csv
import json
import random
import re
import shutil
import subprocess
import sys
import types
from collections import Counter, defaultdict
from pathlib import Path

import pytest

from bank import draw_full, generate_api, generate_full, label, label_evalplus, make_pairs, reuse_labels, select_full
from bank import split_full, token_lengths
from bank.common import read_jsonl, sha256_bytes, sha256_text, write_jsonl
from bank.generate import _row
from bank.select_pilot import evalplus, gsm8k_platinum, math500
from harness.data import BANK_MAPPING, load_records

REPO = Path(__file__).resolve().parents[2]


def rows_of(path):
    return read_jsonl(path)


def dump_rows(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")


# ---------------------------------------------------------------- fixtures: sources and a pilot

def words(seed, n=14):
    """A problem text unlike the others."""
    rng = random.Random(f"words:{seed}")
    return " ".join("".join(rng.choice("bcdfglmnprst") + rng.choice("aeiou") for _ in range(3)) for _ in range(n))


def math_row(i, answer=None, uid=None, text=None):
    answer = str(i + 2) if answer is None else answer
    return {"problem": text or f"{words(f'math{i}')}?", "solution": f"so \\boxed{{{answer}}}", "answer": answer,
            "subject": "Algebra", "level": 3, "unique_id": uid or f"test/algebra/{i}.json"}


NUM_GSM, GSM_USED, M500_USED = 12, 3, 2


@pytest.fixture
def world(tmp_path, monkeypatch):
    """A folder of source files, a bank folder with pilot and development problems, and the expected counts."""
    sources, bank_dir = tmp_path / "sources", tmp_path / "bank"
    sources.mkdir()
    gsm = [{"question": f"{words(f'gsm{i}')} how many?", "answer": f"steps\n#### {i + 3}", "cleaning_status": "verified"}
           for i in range(NUM_GSM)]
    m500 = [math_row(i, uid=f"test/algebra/{i}.json", answer=("\\sqrt{2}" if i >= 8 else None)) for i in range(10)]
    rest = [math_row(100 + i) for i in range(9)]
    rest += [
        math_row(200, answer="x+1"),                                          # not numeric
        math_row(201, uid="test/algebra/1.json"),                             # MATH-500 by id
        math_row(202, text=m500[9]["problem"].upper() + " !"),                # MATH-500 by text (a non-numeric one)
        math_row(203, text="$" + gsm[0]["question"] + "$"),                   # a pilot problem by text
        math_row(204, text="Development problem number one about " + words("dev")),  # a development problem
        math_row(205, text=words("twin", 20) + " find it"),                   # two near-duplicates of each other
        math_row(206, text=words("twin", 20) + ", find it."),
        math_row(207, uid="train/algebra/9.json"),                            # MATH train
        {**math_row(100), "solution": "another way to \\boxed{102}"},          # listed twice, as in the real file
    ]
    human = [{"task_id": f"HumanEval/{i}", "prompt": f"def f{i}():\n    \"\"\"{words(f'he{i}')}\"\"\"", "entry_point": f"f{i}"}
             for i in range(5)]
    mbpp = [{"task_id": f"Mbpp/{i}", "prompt": f"\"\"\"{words(f'mb{i}')}\"\"\"", "entry_point": f"g{i}"} for i in range(6)]
    # the parquet file is JSON here, read by a fake pyarrow
    (sources / "gsm8k_platinum_test.parquet").write_text(json.dumps(gsm), encoding="utf-8")
    for name, rows in (("math500_test.jsonl", m500), (select_full.PRM800K, rest),
                       ("HumanEvalPlus.jsonl", human), ("MbppPlus.jsonl", mbpp)):
        dump_rows(sources / name, rows)
    table = lambda p: types.SimpleNamespace(to_pylist=lambda: json.loads(Path(p).read_text(encoding="utf-8")))  # noqa: E731
    parquet = types.SimpleNamespace(read_table=table)
    monkeypatch.setitem(sys.modules, "pyarrow", types.SimpleNamespace(parquet=parquet))
    monkeypatch.setitem(sys.modules, "pyarrow.parquet", parquet)

    pools = {"gsm8k": gsm8k_platinum(sources / "gsm8k_platinum_test.parquet"),
             "math": math500(sources / "math500_test.jsonl"),
             "humaneval": evalplus(sources / "HumanEvalPlus.jsonl", "humaneval"),
             "mbpp": evalplus(sources / "MbppPlus.jsonl", "mbpp")}
    pilot_math = pools["gsm8k"][:GSM_USED] + pools["math"][:M500_USED]
    pilot_code = pools["humaneval"][:2] + pools["mbpp"][:2]
    dump_rows(bank_dir / "pilot" / "questions_math.jsonl", pilot_math)
    dump_rows(bank_dir / "pilot" / "questions_code.jsonl", pilot_code)
    (bank_dir / "pilot" / "question_ids.txt").write_text(
        "\n".join(q["problem_id"] for q in pilot_math + pilot_code) + "\n", encoding="utf-8")
    (bank_dir / "pilot" / "questions_manifest.json").write_text(json.dumps({
        "pool_sizes": {k: len(v) for k, v in pools.items()},
        "sources": {n: {"sha256": sha256_bytes((sources / n).read_bytes())} for n in select_full.PILOT_SOURCES}}))
    dump_rows(bank_dir / "dev" / "questions.jsonl", [{"problem_id": "gsm8k-train-00001", "question": rest[13]["problem"]}])
    dump_rows(bank_dir / "dev_code" / "questions.jsonl", [{"problem_id": "mbpp-777", "question": words("devcode")}])
    expected = {"gsm8k": 9, "math500": 6, "humaneval": 3, "mbpp": 4, "math500_rows": 10, "math_rest": 16}
    monkeypatch.setattr(select_full, "EXPECTED", expected)
    monkeypatch.setattr(select_full, "NEW_MATH", 11)
    return types.SimpleNamespace(sources=sources, bank=bank_dir, tmp=tmp_path)


def run_select(world, seed=544):
    return select_full.select(world.sources, world.bank, seed)


# ---------------------------------------------------------------- selection

def test_expected_counts_are_the_plans():
    assert (select_full.EXPECTED["gsm8k"], select_full.EXPECTED["math500"]) == (1059, 239)
    assert select_full.EXPECTED["humaneval"] + select_full.EXPECTED["mbpp"] == 392
    manifest = json.loads((REPO / "bank/pilot/questions_manifest.json").read_text())
    used = Counter(i.rsplit("-", 1)[0] for i in (REPO / "bank/pilot/question_ids.txt").read_text().split())
    assert manifest["pool_sizes"]["gsm8k"] - used["gsm8kp"] == 1059
    assert manifest["pool_sizes"]["math"] - used["math500"] == 239
    assert manifest["pool_sizes"]["humaneval"] - used["humaneval-plus"] == select_full.EXPECTED["humaneval"]
    assert manifest["pool_sizes"]["mbpp"] - used["mbpp-plus"] == select_full.EXPECTED["mbpp"]
    assert "media.githubusercontent.com" in select_full.SOURCES[select_full.PRM800K]


def test_selection_takes_all_unused_and_excludes_by_id_and_text(world):
    math_q, code_q, manifest = run_select(world)
    assert Counter(q["source"] for q in math_q) == {"gsm8k-platinum/test": 9, "math-500/test": 6, "math-rest/test": 11}
    assert Counter(q["source"] for q in code_q) == {"evalplus/humaneval+": 3, "evalplus/mbpp+": 4}
    used = set((world.bank / "pilot" / "question_ids.txt").read_text().split())
    assert not used & {q["problem_id"] for q in math_q + code_q}
    new = {q["source_id"] for q in math_q if q["source"] == "math-rest/test"}
    assert new == {f"test/algebra/{i}.json" for i in [*range(100, 109), 205, 206]}
    info = manifest["new_math"]
    assert (info["problems"], info["repeated_ids"]) == (16, ["test/algebra/100.json"])
    assert (info["numeric"], info["also_in_math500_by_id"], info["eligible"], info["drawn"]) == (15, 1, 11, 11)
    assert [d["problem_id"] for d in info["dropped_near_duplicates"]] == ["mathrest-00011", "mathrest-00012", "mathrest-00013"]
    assert info["dropped_near_duplicates"][0]["like"] == "MATH-500 test/algebra/9.json"
    assert manifest["sources"][select_full.PRM800K]["fields"]["unique_id"] == 18
    # the labeler needs these fields on MATH rows (bank.label)
    assert all(q["answer_kind"] == "rational" and q["reference_latex"] for q in math_q if q["source"] != "gsm8k-platinum/test")


def test_the_new_draw_is_seeded_and_the_files_are_reproducible(world, monkeypatch):
    monkeypatch.setattr(select_full, "NEW_MATH", 5)
    first, again, other = (run_select(world, seed)[0] for seed in (544, 544, 545))
    pick = lambda qs: [q["problem_id"] for q in qs if q["source"] == "math-rest/test"]  # noqa: E731
    assert pick(first) == pick(again) != pick(other) and len(pick(first)) == 5
    (world.sources / "licenses").mkdir()
    (world.sources / "licenses" / "prm800k.txt").write_text("MIT License")
    for out in ("a", "b"):
        select_full.main(["--sources", str(world.sources), "--bank-dir", str(world.bank), "--out", str(world.tmp / out)])
    for name in ("questions_math.jsonl", "questions_code.jsonl", "question_ids.txt"):
        data = (world.tmp / "a" / name).read_bytes()
        assert data == (world.tmp / "b" / name).read_bytes() and b"\r" not in data
    manifest = json.loads((world.tmp / "a" / "questions_manifest.json").read_text())
    assert manifest["licenses"]["prm800k"]["text_start"] == "MIT License" and manifest["licenses"]["math-500"] is None
    assert manifest["outputs"]["questions_math.jsonl"] == sha256_bytes((world.tmp / "a" / "questions_math.jsonl").read_bytes())


@pytest.mark.skipif(not (REPO / "bank/full/questions_manifest.json").is_file(), reason="problems not selected yet")
def test_the_selected_problems_match_their_manifest_and_share_nothing_with_the_pilot_or_dev_banks():
    full = REPO / "bank/full"
    manifest = json.loads((full / "questions_manifest.json").read_text(encoding="utf-8"))
    assert {n: sha256_bytes((full / n).read_bytes()) for n in manifest["outputs"]} == manifest["outputs"]
    math_q, code_q = rows_of(full / "questions_math.jsonl"), rows_of(full / "questions_code.jsonl")
    assert Counter(q["source"] for q in math_q) == {draw_full.GSM8K: 1059, draw_full.MATH500: 239, draw_full.MATH_REST: 500}
    assert len(code_q) + len(manifest["dropped_near_duplicates"]) == 392
    for rows, names in ((math_q, ("pilot/questions_math.jsonl", "dev/questions.jsonl")),
                        (code_q, ("pilot/questions_code.jsonl", "dev_code/questions.jsonl"))):
        before = [select_full.problem_tokens(q["question"]) for n in names for q in rows_of(REPO / "bank" / n)]
        mine = [select_full.problem_tokens(q["question"]) for q in rows]
        assert not any(select_full.near_duplicate(a, b) for a in mine for b in before)


def test_bad_sources_stop_the_run(world, monkeypatch):
    prm = world.sources / select_full.PRM800K
    good = prm.read_bytes()
    prm.write_text("version https://git-lfs.github.com/spec/v1\noid sha256:" + "a" * 64 + "\nsize 10900000\n")
    with pytest.raises(SystemExit, match="Git LFS pointer"):
        run_select(world)
    rows = [json.loads(line) for line in good.decode().splitlines()]
    for change, message in (
        ([{k: v for k, v in r.items() if k != "unique_id"} for r in rows], "required field"),
        ([{**r, "answer": 7} for r in rows], "not text"),
        (rows[1:-1], "expected 16, found 15"),
        (rows + [math_row(300)], "expected 16, found 17"),
        (rows[:-1] + [{**rows[0], "answer": "7"}], "twice with different"),
    ):
        dump_rows(prm, change)
        with pytest.raises(SystemExit, match=message):
            run_select(world)
    prm.write_text("<html>not found</html>")
    with pytest.raises(SystemExit, match="line 1: not JSON"):
        run_select(world)
    prm.write_bytes(good)
    pointer = world.sources / "prm800k_math_splits_train.pointer"
    pointer.write_text(f"version https://git-lfs.github.com/spec/v1\noid sha256:{'b' * 64}\nsize {len(good)}\n")
    with pytest.raises(SystemExit, match="does not match its Git LFS pointer"):
        run_select(world)
    pointer.write_text(f"version https://git-lfs.github.com/spec/v1\noid sha256:{sha256_bytes(good)}\nsize {len(good)}\n")
    assert run_select(world)[2]["prm800k_lfs_pointer"] == "matches"
    monkeypatch.setattr(select_full, "NEW_MATH", 12)
    with pytest.raises(SystemExit, match="asked for 12"):
        run_select(world)
    with (world.sources / "MbppPlus.jsonl").open("a") as f:
        f.write("\n")
    with pytest.raises(SystemExit, match="MbppPlus"):
        run_select(world)
    (world.sources / "math500_test.jsonl").unlink()
    with pytest.raises(SystemExit, match="missing"):
        run_select(world)


def test_near_duplicates_ignore_layout_but_not_numbers_in_short_problems():
    near = lambda a, b: select_full.near_duplicate(select_full.problem_tokens(a), select_full.problem_tokens(b))  # noqa: E731
    assert near("Compute $\\dfrac{1}{2} + x$.", "compute \\frac{1}{2}+x")
    assert near("Find $\\left(3, 4\\right)$ and its norm", "Find $(3,4)$ and its norm")
    assert not near("Compute 3 + 4", "Compute 3 - 4")
    assert not near("What is the gcd of 39 and 91?", "What is the gcd of 39 and 52?")
    assert not near("", "")
    long = " ".join(f"word{chr(97 + i)}" for i in range(26)) + " "
    assert select_full.near_duplicate_groups({"a": long + "one", "b": long + "two", "c": "other text", "d": long + "two!"}) == \
        [["a", "b", "d"], ["c"]]


# ---------------------------------------------------------------- generation

def test_generators_settings_and_commits_are_the_pilots():
    for tag, (model, revision) in generate_full.GENERATORS.items():
        for domain in ("math", "code"):
            pilot = json.loads((REPO / f"bank/pilot/gen/{tag}_{domain}/generations_manifest.json").read_text())
            assert pilot["generator"] == model and pilot.get("generator_revision") == revision
            now = generate_full.settings(tag, domain)
            for key in ("model", "tag", "seed", "n_samples", "temperature", "top_p", "max_new_tokens"):
                assert now[key] == pilot["settings"][key], (tag, domain, key)
            for key in ("backend", "dtype", "max_model_len") if tag != "gpt41mini" else ():
                assert now[key] == pilot["settings"][key], (tag, domain, key)
            assert pilot["prompt"] == (generate_full.CODE_PROMPT if domain == "code" else generate_full.PROMPT)


SAMPLES = {"math": 4, "code": 8}


def fake_text(q, tag, k):
    """A deterministic answer. Open models are right on part of the samples; GPT-4.1-mini nearly always."""
    i = int(re.sub(r"\D", "", q["problem_id"].rsplit("-", 1)[1]))
    filler = words(f"{q['problem_id']}{tag}{k}", 12)
    right = {"phi4mini": k % 2 == 0, "granite33": k != 1, "olmo2": k == 2 and i % 3 != 0, "gpt41mini": k != 0 or i % 9 != 4}[tag]
    if q["domain"] == "math":
        value = q["reference"] if right else str(90000 + k)
        return f"{filler} so \\boxed{{{value}}}"
    body = f"def {q['entry_point']}():\n    return {int(right)}  # {filler}\n```\n"
    return body if tag != "gpt41mini" else f"Here:\n```python\n{body}"


class FakeModel:
    """Stands in for the vLLM engine; stops after `fail_after` chunks, as a disconnect would."""
    chunks = 0
    fail_after = None

    def __init__(self, tag):
        self.tag = tag

    def answers(self, questions, first, s):
        if FakeModel.fail_after is not None and FakeModel.chunks >= FakeModel.fail_after:
            raise KeyboardInterrupt
        FakeModel.chunks += 1
        args = types.SimpleNamespace(**s)
        return [_row(q, k, fake_text(q, self.tag, k), "stop", 9, args, s["seed"] + first + i)
                for i, q in enumerate(questions) for k in range(s["n_samples"])]


@pytest.fixture
def full(world, monkeypatch):
    """Question files in <tmp>/full, small chunks and a fake API."""
    out = world.tmp / "full"
    select_full.main(["--sources", str(world.sources), "--bank-dir", str(world.bank), "--out", str(out)])
    monkeypatch.setitem(generate_full.CHUNK, "open", 10)
    monkeypatch.setitem(generate_full.CHUNK, "api", 5)
    monkeypatch.setenv("OPENROUTER_API_KEY", "not-a-key")
    monkeypatch.setattr(FakeModel, "chunks", 0)
    monkeypatch.setattr(FakeModel, "fail_after", None)
    by_prompt = {}
    for d in SAMPLES:
        for i, q in enumerate(rows_of(out / f"questions_{d}.jsonl")):
            by_prompt[generate_api.messages(q["question"], d)[0]["content"]] = (i, q)
    calls = []

    def post(body, key):
        position, q = by_prompt[body["messages"][0]["content"]]
        k = body["seed"] - 544 - 1000 * position  # the seed a single run over the whole file would use
        assert 0 <= k < SAMPLES[q["domain"]]
        calls.append((q["problem_id"], k))
        if api.broken and len(calls) > api.broken:
            raise RuntimeError("gave up")
        return {"choices": [{"message": {"content": fake_text(q, "gpt41mini", k)}, "finish_reason": "stop"}],
                "usage": {"prompt_tokens": 100, "completion_tokens": 200}, "model": "openai/gpt-4.1-mini"}

    api = types.SimpleNamespace(calls=calls, broken=0)
    monkeypatch.setattr(generate_api, "post", post)
    return types.SimpleNamespace(out=out, drive=world.tmp / "drive", tmp=world.tmp, api=api)


def generate(full, tag, domains=("math", "code"), out=None, save=True):
    return generate_full.run(tag, list(domains), out or full.out, full.drive if save else None, 8.0, FakeModel)


def fresh_runtime(full, name):
    """A new Colab runtime: only the question files, nothing generated yet."""
    out = full.tmp / name
    out.mkdir()
    for d in SAMPLES:
        (out / f"questions_{d}.jsonl").write_bytes((full.out / f"questions_{d}.jsonl").read_bytes())
    return out


def test_open_model_resumes_from_drive_after_a_disconnect(full):
    generate(full, "phi4mini", ["math"], save=False)
    whole = (full.out / "gen/phi4mini_math/generations.jsonl").read_bytes()
    FakeModel.chunks, FakeModel.fail_after = 0, 2
    first = fresh_runtime(full, "run1")
    with pytest.raises(KeyboardInterrupt):
        generate(full, "phi4mini", out=first)
    assert sorted(p.name for p in (full.drive / "gen/phi4mini_math/chunks").iterdir()) == ["chunk_00000.jsonl", "chunk_00010.jsonl"]
    FakeModel.chunks, FakeModel.fail_after = 0, None
    second = fresh_runtime(full, "run2")
    assert generate(full, "phi4mini", out=second)
    assert FakeModel.chunks == 2  # 26 math problems: only the third chunk; 7 code problems: one chunk
    assert (full.drive / "gen/phi4mini_math/generations.jsonl").read_bytes() == whole
    rows = rows_of(full.drive / "gen/phi4mini_math/generations.jsonl")
    by_question = {q["problem_id"]: i for i, q in enumerate(rows_of(full.out / "questions_math.jsonl"))}
    assert all(r["seed"] == 544 + by_question[r["problem_id"]] for r in rows)  # the pilot's seeds, whatever the chunks
    # a chunk cut short on Drive is generated again, never taken for a finished one
    for cut in (b"", (full.drive / "gen/phi4mini_math/chunks/chunk_00010.jsonl").read_bytes()[:-30]):
        (full.drive / "gen/phi4mini_math/chunks/chunk_00010.jsonl").write_bytes(cut)
        FakeModel.chunks = 0
        assert generate(full, "phi4mini", ["math"], out=fresh_runtime(full, f"cut{len(cut)}")) and FakeModel.chunks == 1
        assert (full.drive / "gen/phi4mini_math/generations.jsonl").read_bytes() == whole
    # a complete run is left alone, manifest included
    before = (full.drive / "gen/phi4mini_math/generations_manifest.json").read_bytes()
    FakeModel.chunks = 0
    assert generate(full, "phi4mini", out=fresh_runtime(full, "run3")) and FakeModel.chunks == 0
    assert (full.drive / "gen/phi4mini_math/generations_manifest.json").read_bytes() == before


def test_other_questions_or_settings_are_refused(full, monkeypatch):
    generate(full, "olmo2", ["math"])
    other = fresh_runtime(full, "other")
    questions = rows_of(other / "questions_math.jsonl")
    write_jsonl(other / "questions_math.jsonl", questions[1:])
    with pytest.raises(SystemExit, match="other questions or settings"):
        generate(full, "olmo2", ["math"], out=other)
    monkeypatch.setitem(generate_full.DOMAIN_DEFAULTS["math"], "temperature", 0.9)
    with pytest.raises(SystemExit, match="settings"):
        generate(full, "olmo2", ["math"])


def test_api_resumes_asks_only_for_missing_answers_and_keeps_a_new_ledger(full, capsys):
    full.api.broken = 30
    with pytest.raises(SystemExit, match="no answer arrived"):
        generate(full, "gpt41mini", ["math"])
    done = {r["candidate_id"] for p in (full.drive / "gen/gpt41mini_math/chunks").glob("chunk_*.jsonl") for r in rows_of(p)}
    assert len(done) == 30
    full.api.broken, full.api.calls[:] = 0, []
    second = fresh_runtime(full, "run2")
    chunk = second / "gen/gpt41mini_math/chunks/chunk_00005.jsonl"
    generate_full.mirror(full.drive / "gen", second / "gen")
    with chunk.open("a") as f:
        f.write('{"candidate_id": "half')  # a line cut off by an interrupt
    assert generate(full, "gpt41mini", ["math"], out=second)
    assert len(full.api.calls) == 26 * 4 - 30 and not done & {f"{p}__gpt41mini__{k}" for p, k in full.api.calls}
    rows = rows_of(full.drive / "gen/gpt41mini_math/generations.jsonl")
    assert len({r["candidate_id"] for r in rows}) == len(rows) == 26 * 4
    position = {q["problem_id"]: i for i, q in enumerate(rows_of(full.out / "questions_math.jsonl"))}
    assert all(r["seed"] == 544 + 1000 * position[r["problem_id"]] + r["sample_index"] for r in rows)
    spent = json.loads((full.drive / "api_spend.json").read_text())["spent_usd"]
    assert spent == pytest.approx(26 * 4 * (100 * 0.40 + 200 * 1.60) / 1e6)


def test_one_answer_that_keeps_failing_does_not_block_the_rest(full, monkeypatch):
    works = generate_api.post

    def post(body, key):
        if body["seed"] == 544 + 1000 * 2 + 1:  # third problem, second sample
            raise RuntimeError("gave up")
        return works(body, key)

    monkeypatch.setattr(generate_api, "post", post)
    for _ in range(2):
        assert not generate(full, "gpt41mini")  # both domains run, and one answer of each is reported missing
    assert len(full.api.calls) == 26 * 4 + 7 * 8 - 2
    monkeypatch.setattr(generate_api, "post", works)
    assert generate(full, "gpt41mini")


def test_the_ledger_with_more_spend_wins_and_open_models_leave_it_alone(full):
    assert generate(full, "gpt41mini", ["math"])
    after_math = json.loads((full.drive / "api_spend.json").read_text())["spent_usd"]
    clone = fresh_runtime(full, "clone")  # a clone of the repository with an older committed ledger
    (clone / "api_spend.json").write_text(json.dumps({"spent_usd": 0.001, "runs": []}))
    assert generate(full, "olmo2", ["code"], out=clone)
    assert json.loads((full.drive / "api_spend.json").read_text())["spent_usd"] == after_math
    assert generate(full, "gpt41mini", ["code"], out=clone)
    total = json.loads((full.drive / "api_spend.json").read_text())["spent_usd"]
    assert total == pytest.approx(after_math + 7 * 8 * (100 * 0.40 + 200 * 1.60) / 1e6)


def test_api_stops_at_the_budget_cap(full):
    with pytest.raises(SystemExit, match="budget cap"):
        generate_full.run("gpt41mini", ["math"], full.out, full.drive, 0.002)
    assert json.loads((full.out / "api_spend.json").read_text())["spent_usd"] <= 0.002


def test_no_prompt_is_shared_with_the_pilot_or_development_banks(full, world):
    """Seeds overlap the pilot's, so an answer could only repeat if a prompt did."""
    for d, names in (("math", ("pilot/questions_math.jsonl", "dev/questions.jsonl")),
                     ("code", ("pilot/questions_code.jsonl", "dev_code/questions.jsonl"))):
        before = {q["question"].strip() for n in names for q in rows_of(world.bank / n)}
        assert not before & {q["question"].strip() for q in rows_of(full.out / f"questions_{d}.jsonl")}


def generate_all(full):
    for tag in generate_full.GENERATORS:
        assert generate(full, tag)


def merged(full, domain):
    work = full.tmp / "work" / domain
    generate_full.main(["merge", "--domain", domain, "--gen", str(full.drive / "gen"),
                        "--questions-dir", str(full.out), "--out", str(work / "generations.jsonl")])
    return work


def test_merge_refuses_missing_changed_or_foreign_answers(full, capsys):
    generate_all(full)
    generate_full.main(["status", "--domains", "math", "code", "--gen", str(full.drive / "gen"),
                        "--questions-dir", str(full.out)])
    assert "math gpt41mini: 104/104 answers, complete" in capsys.readouterr().out
    assert len(rows_of(merged(full, "math") / "generations.jsonl")) == 26 * 4 * 4
    folder = full.drive / "gen/granite33_math"
    good = (folder / "generations.jsonl").read_bytes()
    manifest = json.loads((folder / "generations_manifest.json").read_text())
    rows = rows_of(folder / "generations.jsonl")
    for change, message in ((rows[1:], "candidate ids"), ([{**rows[0], "generator_revision": "main"}] + rows[1:], "commit"),
                            ([{**rows[0], "text": "changed"}] + rows[1:], "text hash")):
        write_jsonl(folder / "generations.jsonl", change)
        with pytest.raises(SystemExit, match="does not match the hash"):
            merged(full, "math")
        digest = sha256_bytes((folder / "generations.jsonl").read_bytes())
        (folder / "generations_manifest.json").write_text(json.dumps({**manifest, "sha256": digest}))
        with pytest.raises(SystemExit, match=message):
            merged(full, "math")
    (folder / "generations.jsonl").write_bytes(good)
    (folder / "generations_manifest.json").unlink()
    with pytest.raises(SystemExit, match="granite33: not finished"):
        merged(full, "math")


# ---------------------------------------------------------------- code labels: resuming

class InlinePool:
    """ProcessPoolExecutor without processes."""

    def __init__(self, *a, **k):
        pass

    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False

    def map(self, fn, jobs, chunksize=1):
        return map(fn, jobs)


def fake_checker(monkeypatch, checked):
    monkeypatch.setattr(label_evalplus, "ProcessPoolExecutor", InlinePool)
    monkeypatch.setattr(label_evalplus, "SAVE_EVERY", 3)

    def check_job(job):
        checked.append(job[2])
        if checked and len(checked) == label_evalplus_stop[0]:
            raise KeyboardInterrupt
        ok = "return 1" in job[2]
        return [{"status": "pass" if ok else "wrong_output", "detail": None if ok else "base=fail plus=fail"}] * 2

    monkeypatch.setattr(label_evalplus, "check_job", check_job)
    monkeypatch.setattr(label_evalplus, "load_problems", lambda datasets: defaultdict(lambda: ("d", {}, {})))
    monkeypatch.setattr(label_evalplus.metadata, "version", lambda pkg: "0.3.1")


label_evalplus_stop = [0]


def run_label_code(full, work, progress=None):
    label_evalplus.main(["--questions", str(full.out / "questions_code.jsonl"), "--generations", str(work / "generations.jsonl"),
                         "--out", str(work / "labeled.jsonl"), *(["--progress", str(progress)] if progress else [])])
    return (work / "labeled.jsonl").read_bytes()


def test_code_labels_resume_from_the_progress_file_and_match_an_uninterrupted_run(full, monkeypatch):
    generate_all(full)
    work = merged(full, "code")
    checked = []
    fake_checker(monkeypatch, checked)
    whole = run_label_code(full, work)
    total = len(checked)
    progress = full.tmp / "private" / "checks.jsonl"
    label_evalplus_stop[0], checked[:] = 10, []
    with pytest.raises(KeyboardInterrupt):
        run_label_code(full, work, progress)
    with progress.open("a") as f:
        f.write('{"candidate_id": "half')
    label_evalplus_stop[0], checked[:] = 0, []
    assert run_label_code(full, work, progress) == whole
    assert len(checked) == total - 9  # nine checks were saved in threes before the stop
    checked[:] = []
    assert run_label_code(full, work, progress) == whole and not checked


def test_code_errors_have_evalplus_two_types_and_no_other(full, monkeypatch):
    generate_all(full)
    work = merged(full, "code")
    fake_checker(monkeypatch, [])
    seen = [0]

    def check_job(job):
        if "return 1" in job[2]:
            return [{"status": "pass", "detail": None}] * 2
        seen[0] += 1
        runs = (("wrong_output", "wrong_output"), ("timeout", "timeout"), ("wrong_output", "timeout"))[seen[0] % 3]
        return [{"status": s, "detail": s} for s in runs]

    monkeypatch.setattr(label_evalplus, "check_job", check_job)
    run_label_code(full, work)
    rows = rows_of(work / "labeled.jsonl")
    assert {r["error_type"] for r in rows if r["label"] == 0} == {"wrong_output", "timeout"}
    assert all(r["error_type"] == ("timeout" if "timeout" in r["run_statuses"] else "wrong_output") for r in rows if r["label"] == 0)
    assert all(r["error_type"] is None for r in rows if r["label"] != 0)


# ---------------------------------------------------------------- length filter

class FakeTokenizer:
    def __init__(self, name):
        self.name = name

    def apply_chat_template(self, msgs, tokenize, add_generation_prompt, **kwargs):
        assert (kwargs == {"enable_thinking": False}) == (self.name == "Qwen/Qwen3-8B")
        ids = msgs[0]["content"].split()
        return {"input_ids": [ids]} if "Dream" in self.name else ids


def fake_transformers(monkeypatch, missing=(), long_for=None):
    def from_pretrained(name, trust_remote_code):
        if name in missing:
            raise OSError("gated repo")
        return FakeTokenizer(name)

    monkeypatch.setitem(sys.modules, "transformers",
                        types.SimpleNamespace(AutoTokenizer=types.SimpleNamespace(from_pretrained=from_pretrained)))


def test_length_filter_needs_every_verifier_tokenizer_and_qwen3(tmp_path, monkeypatch):
    for needed in ("Qwen/Qwen2.5-7B-Instruct", "meta-llama/Llama-3.1-8B-Instruct", "google/gemma-2-9b-it",
                   "mistralai/Mistral-7B-Instruct-v0.3", "meta-llama/Llama-3.2-3B-Instruct", "Qwen/Qwen2.5-1.5B-Instruct",
                   "GSAI-ML/LLaDA-8B-Instruct", "Dream-org/Dream-v0-Instruct-7B", "Qwen/Qwen3-8B"):
        assert needed in token_lengths.TOKENIZERS
    bank, out = tmp_path / "full_bank.jsonl", tmp_path / "too_long.json"
    write_jsonl(bank, [{"candidate_id": "a", "problem": "p", "candidate": "short"},
                       {"candidate_id": "b", "problem": "p", "candidate": "long " * 60}])
    fake_transformers(monkeypatch, missing=("google/gemma-2-9b-it", "meta-llama/Llama-3.1-8B-Instruct"))
    with pytest.raises(SystemExit, match="gemma-2-9b-it"):
        token_lengths.main([str(bank), "--out", str(out)])
    assert not out.exists()
    fake_transformers(monkeypatch)
    token_lengths.main([str(bank), "--out", str(out), "--max-tokens", "60"])
    result = json.loads(out.read_text())
    assert result["too_long"] == ["b"] and set(result["tokenizers_loaded"]) == set(token_lengths.TOKENIZERS)
    assert result["banks_sha256"] == [sha256_bytes(bank.read_bytes())]


# ---------------------------------------------------------------- the draw rule

GENS = ["A", "B", "C", "D"]
HUMANEVAL, MBPP = "evalplus/humaneval+", "evalplus/mbpp+"


def make(pairs):
    """pair id -> info from (source, problem, generator) triples."""
    return {f"{p}:{g}": {"source": s, "problem_id": p, "generator": g} for s, p, g in pairs}


def many(source, n, gens=GENS, per_problem=1, prefix=None):
    """n problems of one source, generators in rotation, per_problem pairs each."""
    return [(source, f"{prefix or source}-{i:04d}", gens[(i + j) % len(gens)]) for i in range(n) for j in range(per_problem)]


def test_math_target_is_twice_the_smaller_half_and_gsm8k_is_not_a_third():
    pairs = make(many(draw_full.GSM8K, 100) + many(draw_full.MATH500, 40) + many(draw_full.MATH_REST, 120))
    ids, s = draw_full.draw(pairs, "math")
    assert (s["target"], s["quota"]) == (200, {"gsm8k": 100, "math": 100})
    assert s["by_source"] == {draw_full.GSM8K: 100, draw_full.MATH500: 40, draw_full.MATH_REST: 60}
    assert s["shortfall"] == {"gsm8k": 0, "math": 0} and len(ids) == len(set(ids)) == 200
    # MATH smaller than GSM8K: every MATH pair, and a seeded subset of GSM8K
    ids, s = draw_full.draw(make(many(draw_full.GSM8K, 100) + many(draw_full.MATH500, 12) + many(draw_full.MATH_REST, 20)), "math")
    assert s["by_source"] == {draw_full.GSM8K: 32, draw_full.MATH500: 12, draw_full.MATH_REST: 20}


def test_the_problem_cap_comes_before_the_target():
    # every problem has 4 pairs: 2 survive, so G = 2 x 10 and M = 2 x 30
    pairs = make(many(draw_full.GSM8K, 10, per_problem=4) + many(draw_full.MATH_REST, 30, per_problem=4))
    ids, s = draw_full.draw(pairs, "math")
    assert s["pairs_after_problem_cap"] == 80 and s["target"] == 40
    assert max(Counter(pairs[i]["problem_id"] for i in ids).values()) == 2
    assert all(len({pairs[i]["generator"] for i in ids if pairs[i]["problem_id"] == p}) == n
               for p, n in Counter(pairs[i]["problem_id"] for i in ids).items())
    code = make(many(HUMANEVAL, 10, per_problem=4) + many(HUMANEVAL, 5, prefix="other"))
    _, s = draw_full.draw(code, "code")
    assert (s["pairs_after_problem_cap"], s["target"]) == (35, 35)


def test_four_generators_are_capped_at_30_percent_and_the_shortfall_is_not_filled():
    # 100 code pairs, 50 from A: the cap is 30, so 20 are missing even though nothing else is left
    pairs = make([(draw_full.CODE_SOURCES[0], f"p{i}", "A" if i < 50 else GENS[1 + i % 3]) for i in range(100)])
    ids, s = draw_full.draw(pairs, "code")
    assert s["generator_cap"] == {"pairs": 30, "percent": 30, "available": GENS, "fewer_than_three_available": False}
    assert s["by_generator"]["A"] == 30 and s["shortfall"] == {"code": 20} and len(ids) == 80
    assert s["skipped_at_generator_cap"] == {"code": 20}


def test_a_generator_below_ten_percent_of_the_target_does_not_count():
    base = [(draw_full.CODE_SOURCES[0], f"p{i}", GENS[i % 3]) for i in range(91)]
    nine = draw_full.draw(make(base + [(draw_full.CODE_SOURCES[0], f"q{i}", "D") for i in range(9)]), "code")[1]
    assert nine["target"] == 100 and nine["generator_cap"]["percent"] == 40 and nine["generator_cap"]["available"] == ["A", "B", "C"]
    assert nine["by_generator"]["D"] == 9 and nine["shortfall"] == {"code": 0}  # its pairs can still be drawn
    ten = draw_full.draw(make(base[:90] + [(draw_full.CODE_SOURCES[0], f"q{i}", "D") for i in range(10)]), "code")[1]
    assert ten["generator_cap"]["percent"] == 30  # exactly 10% counts


def test_a_shortage_at_the_generator_cap_hits_both_halves_alike():
    # A supplies 70% of both halves; taking turns, each half loses the same number
    half = lambda source, n: [(source, f"{source}-{i}", "A" if i % 10 < 7 else GENS[1 + i % 3]) for i in range(n)]  # noqa: E731
    _, s = draw_full.draw(make(half(draw_full.GSM8K, 100) + half(draw_full.MATH_REST, 100)), "math")
    assert s["by_generator"]["A"] == s["generator_cap"]["pairs"] == 200 * s["generator_cap"]["percent"] // 100
    assert abs(s["shortfall"]["gsm8k"] - s["shortfall"]["math"]) <= 8 and s["chosen"] == 200 - sum(s["shortfall"].values())
    assert s["by_source"][draw_full.GSM8K] == 100 - s["shortfall"]["gsm8k"]


def test_the_draw_is_seeded_and_does_not_depend_on_input_order():
    pairs = make(many(draw_full.GSM8K, 60) + many(draw_full.MATH500, 90))
    shuffled = dict(random.Random(1).sample(sorted(pairs.items()), len(pairs)))
    assert draw_full.draw(pairs, "math")[0] == draw_full.draw(shuffled, "math")[0]
    assert draw_full.draw(pairs, "math", 545)[0] != draw_full.draw(pairs, "math")[0]
    with pytest.raises(SystemExit, match="unexpected sources"):
        draw_full.draw(make(many("gsm8k/train", 3)), "math")


def remove(pairs, extra_questions=0):
    """without_left_out on `make` pairs, with one bank row standing for each pair."""
    questions = [{"problem_id": p, "source": s} for p, s in sorted({(i["problem_id"], i["source"]) for i in pairs.values()})]
    questions += [{"problem_id": f"no-pair-{i}", "source": MBPP} for i in range(extra_questions)]
    answers = [{"problem_id": q["problem_id"], "generator": g} for q in questions for g in GENS]
    rows, left, answers_left, removed = draw_full.without_left_out("code", {p: [i] for p, i in pairs.items()}, questions, answers)
    assert {q["source"] for q in left} <= {HUMANEVAL} and len(answers_left) == len(GENS) * len(left)
    return {p: m[0] for p, m in rows.items()}, removed


def test_code_is_humaneval_only_and_mbpp_leaves_before_anything_is_counted():
    # D has 12 of 100 pairs overall but 3 of 40 in HumanEval+; A has 60 in all and 13 there
    human = [(HUMANEVAL, f"h{i}", "D" if i < 3 else GENS[i % 3]) for i in range(40)]
    mbpp = [(MBPP, f"m{i}", "D" if i < 9 else ("A" if i < 56 else "B")) for i in range(60)]
    alone = draw_full.draw(make(human), "code")
    kept, removed = remove(make(human + mbpp), extra_questions=7)
    assert removed == {MBPP: {"problems": 67, "answers": 67 * 4, "qualifying_pairs": 60, "answers_in_pairs": 120}}
    assert kept == make(human) and draw_full.draw(kept, "code") == alone
    s = alone[1]
    assert (s["qualifying_pairs"], s["target"], s["generator_cap"]["available"]) == (40, 40, ["A", "B", "C"])
    assert (s["generator_cap"]["pairs"], s["generator_cap"]["percent"]) == (16, 40) and set(s["by_source"]) == {HUMANEVAL}
    for source in (MBPP, "evalplus/humaneval", "math-500/test"):  # MBPP+ is removed before the draw, never drawn
        with pytest.raises(SystemExit, match="unexpected sources"):
            draw_full.draw(make(human + [(source, "x", "A")]), "code")
    assert draw_full.LEFT_OUT["math"] == () and not set(draw_full.CODE_SOURCES) & set(draw_full.LEFT_OUT["code"])


def test_the_pilots_humaneval_pairs_give_three_generators_and_a_40_percent_cap():
    """The counts in the decision record (point 6). The full bank's own counts decide there."""
    pairs = {r["pair_id"]: {k: r[k] for k in ("source", "problem_id", "generator")}
             for r in rows_of(REPO / "bank/pilot/code/pilot_bank.jsonl")}
    kept, removed = remove(pairs)
    s = draw_full.draw(kept, "code")[1]
    assert removed[MBPP]["qualifying_pairs"] == 99 and s["qualifying_by_generator"]["openai/gpt-4.1-mini"] == 6
    assert s["target"] == 67 and s["generator_cap"]["percent"] == 40 and len(s["generator_cap"]["available"]) == 3


# ---------------------------------------------------------------- the split

def test_the_split_and_the_near_duplicate_rule_on_the_real_humaneval_problems():
    problems = {q["problem_id"]: (q["source"], q["question"]) for q in rows_of(REPO / "bank/full/questions_code.jsonl")
                if q["source"] == HUMANEVAL}
    assignment, together = split_full.split_problems(problems, 544)
    assert together == [["humaneval-plus-056", "humaneval-plus-061"]]  # the two bracket problems
    assert assignment["humaneval-plus-056"] == assignment["humaneval-plus-061"]
    assert Counter(assignment.values()) == {"validation": 38, "test": 76}


def test_split_is_a_third_per_source_by_problem_with_near_duplicates_together():
    problems = {f"s{n}-{i:02d}": (f"s{n}", words(f"{n}-{i}")) for n, size in (("1", 9), ("2", 10), ("3", 11)) for i in range(size)}
    problems["s1-00"] = ("s1", words("twin", 20))
    problems["s1-05"] = ("s1", words("twin", 20).upper())
    problems["s2-03"] = ("s2", words("twin", 20) + "!")  # a near-duplicate across sources counts for s1
    assignment, together = split_full.split_problems(problems, 544)
    assert together == [["s1-00", "s1-05", "s2-03"]] and set(assignment) == set(problems)
    assert len({assignment[p] for p in together[0]}) == 1
    per = Counter((p[:2], s) for p, s in assignment.items() if p != "s2-03")
    spread = int(assignment["s2-03"] == "validation")
    assert per[("s1", "validation")] + spread == 3  # (9 + 1 from s2) / 3 -> 3
    assert per[("s2", "validation")] == 3  # 9 left -> 3
    assert per[("s3", "validation")] == 4  # 11 / 3 -> 4
    assert assignment == split_full.split_problems(dict(reversed(problems.items())), 544)[0]
    assert assignment != split_full.split_problems(problems, 545)[0]


# ---------------------------------------------------------------- everything together

@pytest.fixture
def built(full, monkeypatch, capsys):
    """Both domains, from selection to split, on the fake generators and checkers."""
    generate_all(full)
    monkeypatch.setattr(label, "_math_verify", lambda: (lambda s: s, lambda gold, pred: f"boxed{{{gold.strip('$')}}}" in pred))
    monkeypatch.setattr(label, "_checker2_version", lambda name: "math_verify==0.9.0")
    fake_checker(monkeypatch, [])
    fake_transformers(monkeypatch)
    for domain in SAMPLES:
        work = merged(full, domain)
        if domain == "math":
            monkeypatch.setattr(sys, "argv", ["label", "--questions", str(full.out / "questions_math.jsonl"),
                                              "--generations", str(work / "generations.jsonl"), "--out", str(work / "labeled.jsonl")])
            label.main()
        else:
            run_label_code(full, work, full.tmp / "private/code/checks.jsonl")
        build(full, domain)
    full.public, full.private = full.tmp / "commit/bank/full", full.tmp / "private"
    return full


def build(full, domain, out="commit/bank/full"):
    work = full.tmp / "work" / domain
    questions = str(full.out / f"questions_{domain}.jsonl")
    gens = str(work / "generations.jsonl")
    make_pairs.main(["--split", "full", "--questions", questions, "--generations", gens,
                     "--labeled", str(work / "labeled.jsonl"), "--out", str(work)])
    token_lengths.main([str(work / "full_bank.jsonl"), "--out", str(work / "too_long.json")])
    draw_full.main(["--domain", domain, "--work", str(work), "--questions", questions, "--generations", gens])
    split_full.main(["split", "--domain", domain, "--work", str(work), "--out", str(full.tmp / out / domain),
                     "--private", str(full.tmp / "private" / domain)])


def test_end_to_end_files_are_what_the_scorers_and_analysis_read(built):
    for domain in SAMPLES:
        public, private = built.public / domain, built.private / domain
        assert sorted(p.name for p in public.iterdir()) == ["bank.jsonl", "labels_validation.jsonl", "manifest.json"]
        bank = rows_of(public / "bank.jsonl")
        validation, test = rows_of(public / "labels_validation.jsonl"), rows_of(private / "labels_test.jsonl")
        manifest = json.loads((public / "manifest.json").read_text())
        # rows: the pilot bank's fields, no label, readable by the AR scorer
        pilot = rows_of(REPO / "bank/pilot/pilot_bank.jsonl")
        example = next(r for r in pilot if r["domain"] == domain)
        assert all(list(r) == list(example) for r in bank)
        assert all(not set(r) & split_full.LABEL_FIELDS for r in bank)
        records, digest = load_records(public / "bank.jsonl", BANK_MAPPING, "dev_bank")
        assert len(records) == len(bank) and digest == manifest["files"]["bank.jsonl"]
        # labels: the pilot label file's fields, one file per split, every pair one right and one wrong
        pilot_label = next(r for r in rows_of(REPO / "bank/pilot/pilot_labels.jsonl") if ("run_statuses" in r) == (domain == "code"))
        assert all(list(r) == list(pilot_label) for r in validation + test)
        of = {s: [r["candidate_id"] for r in bank if r["split"] == s] for s in ("validation", "test")}
        assert of == {"validation": [r["candidate_id"] for r in validation], "test": [r["candidate_id"] for r in test]}
        labels = {r["candidate_id"]: r["label"] for r in validation + test}
        members = defaultdict(list)
        for r in bank:
            members[r["pair_id"]].append(r)
        assert all(sorted(labels[r["candidate_id"]] for r in m) == [0, 1] for m in members.values())
        # order: pairs by id, the two answers by text hash, whatever the labels are
        assert [r["pair_id"] for r in bank] == sorted(r["pair_id"] for r in bank)
        assert all(m[0]["text_sha256"] < m[1]["text_sha256"] for m in members.values())
        # split: by problem, nothing shared
        split_of = {r["problem_id"]: r["split"] for r in bank}
        assert all(split_of[r["problem_id"]] == r["split"] for r in bank)
        # manifest: hashes of every file, counts, shortfalls, qualifying cells
        for name, digest in manifest["files"].items():
            folder = private if name == "labels_test.jsonl" else public
            assert sha256_bytes((folder / name).read_bytes()) == digest
            assert b"\r" not in (folder / name).read_bytes()
        assert manifest["counts"]["test"]["candidates"] == len(test) and manifest["counts"]["total"]["pairs"] == len(members)
        cells = manifest["share_of_cells_that_qualified"]["all"]
        questions = rows_of(built.out / f"questions_{domain}.jsonl")
        n_questions = sum(q["source"] != MBPP for q in questions)
        if domain == "math":
            assert "left_out" not in manifest["draw"] and n_questions == len(questions)
        else:
            left = manifest["draw"]["left_out"]
            assert list(left) == [MBPP] and (left[MBPP]["problems"], left[MBPP]["answers"]) == (4, 4 * 8 * 4)
            assert left[MBPP]["qualifying_pairs"] > 0 and left[MBPP]["answers_in_pairs"] == 2 * left[MBPP]["qualifying_pairs"]
            assert {r["source"] for r in bank} == {HUMANEVAL} and MBPP not in manifest["share_of_cells_that_qualified"]
        assert cells["cells"] == 4 * n_questions and cells["qualified"] == manifest["draw"]["qualifying_pairs"]
        assert manifest["seed"] == 544 and set(manifest["shortfall"]) == ({"gsm8k", "math"} if domain == "math" else {"code"})
        text = (public / "manifest.json").read_text()
        assert '"label"' not in text and "wrong_output" not in text
    math = json.loads((built.public / "math/manifest.json").read_text())
    assert math["draw"]["generator_cap"]["available"] == sorted(generate_full.GENERATORS[t][0] for t in ("phi4mini", "granite33", "olmo2"))
    assert math["draw"]["quota"] == {"gsm8k": 18, "math": 18}
    assert math["draw"]["by_source"][draw_full.MATH500] == 12  # all of MATH-500 before the new problems
    # the two near-duplicate new problems, if both were drawn, are in one split
    twins = {r["problem_id"]: r["split"] for r in rows_of(built.public / "math/bank.jsonl") if "twin" in r["problem"]}
    assert len(set(twins.values())) <= 1


def test_rebuilding_gives_the_same_bytes_and_a_different_bank_is_refused(built):
    before = {p: p.read_bytes() for p in [*built.public.rglob("*"), *built.private.glob("*/labels_test.jsonl")] if p.is_file()}
    for domain in SAMPLES:
        build(built, domain)
    assert before == {p: p.read_bytes() for p in before}
    work = built.tmp / "work/math"
    drawn = json.loads((work / "draw_summary.json").read_text())
    rows = rows_of(work / "drawn_bank.jsonl")
    write_jsonl(work / "drawn_bank.jsonl", rows[2:])
    with pytest.raises(SystemExit, match="rerun bank.draw_full"):
        split_full.split("math", work, built.public / "math", built.private / "math")
    write_jsonl(work / "drawn_labels.jsonl", rows_of(work / "drawn_labels.jsonl")[2:])
    drawn["outputs"] = {n: sha256_bytes((work / n).read_bytes()) for n in drawn["outputs"]}
    (work / "draw_summary.json").write_text(json.dumps(drawn))
    with pytest.raises(SystemExit, match="different bank"):
        split_full.split("math", work, built.public / "math", built.private / "math")
    for private in (built.public, built.public / "math", built.public / "math" / "private", built.public / "code",
                    built.tmp / "commit", built.tmp):
        with pytest.raises(SystemExit, match="outside"):
            split_full.split("math", work, built.public / "math", private)
    (built.tmp / "repo" / ".git").mkdir(parents=True)
    with pytest.raises(SystemExit, match="git checkout"):
        split_full.split("math", work, built.public / "math", built.tmp / "repo" / "results")
    # a corrected bank in a new public folder must not replace the old version's test labels
    v2 = built.tmp / "commit_v2/bank/full/math"
    with pytest.raises(SystemExit, match="never written over"):
        split_full.split("math", work, v2, built.private / "math")
    assert not v2.exists() and before == {p: p.read_bytes() for p in before}
    manifest = split_full.split("math", work, v2, built.tmp / "private_v2/math")
    assert manifest["files"] != json.loads((built.public / "math/manifest.json").read_text())["files"]
    assert before == {p: p.read_bytes() for p in before}


def test_a_pair_is_dropped_when_one_answer_is_too_long_and_stale_lists_are_refused(built):
    work = built.tmp / "work/code"
    pairs, _ = draw_full.load_pairs(work, "code")
    human = {p for p, m in pairs.items() if m[0]["source"] == HUMANEVAL}
    victim = sorted(human)[0]
    data = json.loads((work / "too_long.json").read_text())
    good = copy.deepcopy(data)
    args = ["--domain", "code", "--work", str(work), "--questions", str(built.out / "questions_code.jsonl"),
            "--generations", str(work / "generations.jsonl")]
    gone = next(m[0]["candidate_id"] for p, m in sorted(pairs.items()) if p not in human)
    data["too_long"] = [pairs[victim][1]["candidate_id"], gone]  # an MBPP+ one is already out, so it is not counted
    (work / "too_long.json").write_text(json.dumps(data))
    draw_full.main(args)
    summary = json.loads((work / "draw_summary.json").read_text())
    assert summary["pairs_dropped_too_long"] == 1 and victim not in {r["pair_id"] for r in rows_of(work / "drawn_bank.jsonl")}
    assert summary["qualifying_pairs"] == len(human) - 1 and summary["too_long_candidates"] == [data["too_long"][0]]
    assert summary["left_out"][MBPP]["qualifying_pairs"] == len(pairs) - len(human)
    for change, message in (({"tokenizers_loaded": data["tokenizers_loaded"][:-1]}, "every verifier"),
                            ({"tokenizers_failed": {"x": "y"}}, "every verifier"), ({"banks_sha256": ["0" * 64]}, "other answers")):
        (work / "too_long.json").write_text(json.dumps({**good, **change}))
        with pytest.raises(SystemExit, match=message):
            draw_full.main(args)
    (work / "too_long.json").write_text(json.dumps(good))
    write_jsonl(built.tmp / "other" / "generations.jsonl", rows_of(work / "generations.jsonl")[1:])
    with pytest.raises(SystemExit, match="not the file"):
        draw_full.main(args[:-1] + [str(built.tmp / "other" / "generations.jsonl")])
    labels = rows_of(work / "full_labels.jsonl")
    for change in ({"label": 1}, {"checker_version": "bank.label_code/2+subprocess"}):
        write_jsonl(work / "full_labels.jsonl", [{**r, **change} for r in labels])
        with pytest.raises(SystemExit, match="make_pairs wrote"):
            draw_full.main(args)
        with pytest.raises(SystemExit, match="one checker version|one correct and one wrong"):
            draw_full.load_pairs(work, "code")


def test_the_code_bank_is_rebuilt_from_finished_labels_without_running_anything(full, monkeypatch, request):
    """An earlier version drawn with MBPP+, then a new one from its labels: nothing is relabeled, the old one stays."""
    with monkeypatch.context() as before:  # the draw as it was when the labels were made
        before.setattr(draw_full, "CODE_SOURCES", (HUMANEVAL, MBPP))
        before.setitem(draw_full.LEFT_OUT, "code", ())
        built = request.getfixturevalue("built")
    v1 = built.tmp / "work/code"
    assert {r["source"] for r in rows_of(built.public / "code/bank.jsonl")} == {HUMANEVAL, MBPP}
    old = {p: p.read_bytes() for root in (v1, built.public, built.private) for p in root.rglob("*") if p.is_file()}
    with pytest.raises(SystemExit, match="rerun bank.draw_full"):  # the old draw cannot be split any more
        split_full.split("code", v1, built.tmp / "other/bank/full/code", built.tmp / "other_private/code")

    monkeypatch.setattr(label_evalplus, "check_job", lambda job: pytest.fail("a program was run"))
    v2 = built.tmp / "work_v2/code"
    generate_full.main(["merge", "--domain", "code", "--gen", str(built.drive / "gen"), "--questions-dir", str(built.out),
                        "--out", str(v2 / "generations.jsonl")])
    questions = built.out / "questions_code.jsonl"
    in_repo = built.tmp / "clone/bank/full/code"
    args = ["reuse", "--old", str(v1), "--work", str(v2), "--questions", str(questions),
            "--old-public", str(built.public / "code"), str(in_repo)]
    with pytest.raises(SystemExit, match="validation labels can be test labels"):  # the old public code bank is still there
        reuse_labels.main(args)
    assert not (v2 / "labeled.jsonl").exists()
    (built.public / "code").rename(built.tmp / "deleted")
    in_repo.mkdir(parents=True)
    with pytest.raises(SystemExit, match="validation labels can be test labels"):  # or it was committed
        reuse_labels.main(args)
    in_repo.rmdir()
    old = {p: data for p, data in old.items() if p.exists()}
    reuse_labels.main(args)
    assert (v2 / "labeled.jsonl").read_bytes() == (v1 / "labeled.jsonl").read_bytes()
    make_pairs.main(["--split", "full", "--questions", str(questions), "--generations", str(v2 / "generations.jsonl"),
                     "--labeled", str(v2 / "labeled.jsonl"), "--out", str(v2)])
    for name in ("full_bank.jsonl", "full_labels.jsonl"):
        assert (v2 / name).read_bytes() == (v1 / name).read_bytes()
    token_lengths.main([str(v2 / "full_bank.jsonl"), "--out", str(v2 / "too_long.json")])
    draw_full.main(["--domain", "code", "--work", str(v2), "--questions", str(questions), "--generations", str(v2 / "generations.jsonl")])
    public, private = built.tmp / "commit_v2/bank/full/code", built.tmp / "private_v2/code"
    split_full.main(["split", "--domain", "code", "--work", str(v2), "--out", str(public), "--private", str(private)])
    bank = rows_of(public / "bank.jsonl")
    assert {r["source"] for r in bank} == {HUMANEVAL}
    labels = {r["candidate_id"]: r for r in rows_of(v1 / "full_labels.jsonl")}
    assert all(labels[r["candidate_id"]] == r for r in rows_of(public / "labels_validation.jsonl") + rows_of(private / "labels_test.jsonl"))
    assert old == {p: p.read_bytes() for p in old}

    # anything that is not exactly the earlier version's input is refused, and nothing is copied
    def refused(match):
        (v2 / "labeled.jsonl").unlink(missing_ok=True)
        with pytest.raises(SystemExit, match=match):
            reuse_labels.main(args)
        assert not (v2 / "labeled.jsonl").exists()

    for path in (v2 / "generations.jsonl", questions):
        good = path.read_bytes()
        path.write_bytes(good + b"\n")
        refused("not the file the earlier version")
        path.write_bytes(good)
    for name in ("full_bank_summary.json", "labeled.jsonl"):  # the earlier version never got that far
        good = (v1 / name).read_bytes()
        (v1 / name).unlink()
        refused("no finished labels")
        for part in (b"", good[: len(good) // 2], good + b"\n" if name == "labeled.jsonl" else b"[]"):  # or a copy was cut off
            (v1 / name).write_bytes(part)
            refused("no finished labels")
        (v1 / name).write_bytes(good)
    summary, labeled = (v1 / "full_bank_summary.json").read_bytes(), rows_of(v1 / "labeled.jsonl")

    def relabel(rows):  # an earlier version whose summary fits other labels
        write_jsonl(v1 / "labeled.jsonl", rows)
        data = json.loads(summary)
        data["inputs"]["labeled.jsonl"] = sha256_bytes((v1 / "labeled.jsonl").read_bytes())
        (v1 / "full_bank_summary.json").write_text(json.dumps(data))

    for rows in (labeled[:-1], labeled[1:] + labeled[:1], [{**labeled[0], "candidate_id": "x"}] + labeled[1:]):
        relabel(rows)
        refused("does not cover exactly the answers")
    for version in ("bank.label_evalplus/1+evalplus==0.3.0+colab-guard", "bank.label_evalplus/0+evalplus==0.3.1+colab-guard", None):
        relabel(labeled[:-1] + [{**labeled[-1], "checker_version": version}])  # one row from another checker is enough
        refused("not labeled by the current checker")
    relabel(labeled)
    reuse_labels.main(args)


def test_reused_math_labels_must_come_from_the_current_math_checker(built, monkeypatch):
    v1, v2, questions = built.tmp / "work/math", built.tmp / "v2/math", built.out / "questions_math.jsonl"
    v2.mkdir(parents=True)
    (v2 / "generations.jsonl").write_bytes((v1 / "generations.jsonl").read_bytes())
    assert {r["checker_version"] for r in rows_of(v1 / "labeled.jsonl")} == {reuse_labels.CHECKERS["math"]}
    assert reuse_labels.reuse(v1, v2, questions, [built.tmp / "gone"]) == len(rows_of(v1 / "labeled.jsonl"))
    (v2 / "labeled.jsonl").unlink()
    for checker in ("bank.label/3+math_verify==0.9.0", "bank.label/2+math_verify==0.9.1"):  # the checker was fixed since
        monkeypatch.setitem(reuse_labels.CHECKERS, "math", checker)
        with pytest.raises(SystemExit, match="not labeled by the current checker"):
            reuse_labels.reuse(v1, v2, questions, [])
    assert not (v2 / "labeled.jsonl").exists()


def test_the_current_checkers_are_the_pilots_and_the_notebooks():
    for domain in SAMPLES:
        rows = rows_of(REPO / f"bank/pilot/{domain}/pilot_bank.jsonl")
        assert {r["checker_version"] for r in rows} == {reuse_labels.CHECKERS[domain]}
    text = "\n".join(notebook_lines("11_full_label_split"))
    for pin in ("math-verify==0.9.0", "evalplus==0.3.1"):
        assert pin in text and pin.split("==")[1] in "".join(reuse_labels.CHECKERS.values())


def test_labels_are_kept_before_the_split_and_never_replaced(built):
    work, keep = built.tmp / "work/math", built.tmp / "private/math/work"
    questions = built.out / "questions_math.jsonl"
    reuse_labels.main(["save", "--work", str(work), "--keep", str(keep)])
    held = lambda: {p.name: p.read_bytes() for p in keep.iterdir() if p.suffix != ".tmp"}  # noqa: E731
    kept = held()
    assert sorted(kept) == sorted(reuse_labels.KEPT) and reuse_labels.save(work, keep) == 0

    # a copy that was cut off is replaced, whichever file it stopped in, and is never taken for finished labels
    for name in reuse_labels.KEPT:
        for part in (b"", kept[name][: len(kept[name]) // 2], None):
            (keep / name).unlink()
            if part is not None:
                (keep / name).write_bytes(part)
            (keep / f"{name}.tmp").write_text("half")
            if name in ("labeled.jsonl", "full_bank_summary.json"):
                with pytest.raises(SystemExit, match="no finished labels"):
                    reuse_labels.reuse(keep, work, questions, [])
            assert reuse_labels.save(work, keep) >= 1 and held() == kept
    (keep / "full_bank_summary.json").write_text("[]")  # not a summary at all
    assert reuse_labels.save(work, keep) == 1 and held() == kept

    # finished labels are never replaced by different ones
    summary = json.loads((work / "full_bank_summary.json").read_text())
    (work / "full_bank_summary.json").write_text(json.dumps({**summary, "python": "3.99"}))  # another runtime
    assert reuse_labels.save(work, keep) == 0 and held() == kept
    for name, kind in (("labeled.jsonl", "inputs"), ("full_labels.jsonl", "outputs")):
        good = (work / name).read_bytes()
        write_jsonl(work / name, rows_of(work / name)[1:])
        with pytest.raises(SystemExit, match="make the pairs first"):  # a file the pairs were not made from
            reuse_labels.save(work, keep)
        digest = sha256_bytes((work / name).read_bytes())
        (work / "full_bank_summary.json").write_text(json.dumps({**summary, kind: {**summary[kind], name: digest}}))
        with pytest.raises(SystemExit, match="never replaced"):
            reuse_labels.save(work, keep)
        (work / name).write_bytes(good)
        (work / "full_bank_summary.json").write_text(json.dumps(summary))
    assert held() == kept
    (work / "full_bank.jsonl").unlink()
    with pytest.raises(SystemExit, match="make the pairs first"):
        reuse_labels.save(work, keep)


def test_an_interrupted_save_never_leaves_a_finished_looking_folder(built, monkeypatch):
    """The summary is written last: wherever the copying stops, the folder is not finished, and saving again completes it."""
    work, copy = built.tmp / "work/math", reuse_labels.copy_file
    for stop_after in range(len(reuse_labels.KEPT)):
        keep, done = built.tmp / f"kept{stop_after}", []

        def copy_then_stop(src, dst):
            if len(done) == stop_after:
                raise KeyboardInterrupt
            copy(src, dst)
            done.append(src.name)

        monkeypatch.setattr(reuse_labels, "copy_file", copy_then_stop)
        with pytest.raises(KeyboardInterrupt):
            reuse_labels.save(work, keep)
        assert reuse_labels.finished(keep) is None and "full_bank_summary.json" not in done
        monkeypatch.setattr(reuse_labels, "copy_file", copy)
        assert reuse_labels.save(work, keep) == len(reuse_labels.KEPT) - stop_after
        assert reuse_labels.finished(keep) is not None


def test_the_commit_cell_stops_at_a_bank_that_must_not_be_committed(built, capsys):
    cells = json.loads((REPO / "notebooks/11_full_label_split.ipynb").read_text(encoding="utf-8"))["cells"]
    cell = next("".join(c["source"]) for c in cells if c["cell_type"] == "code" and "private_files" in "".join(c["source"]))
    commit, public = built.tmp / "to_commit/v2", built.tmp / "to_commit/v2/bank/full"
    shutil.copytree(built.tmp / "commit", commit)

    def run(*domains):
        exec(cell, {"DOMAINS": list(domains), "SHARED": str(built.drive), "PRIVATE": str(built.private), "COMMIT": str(commit)})
        return capsys.readouterr().out

    assert "math/labels_validation.jsonl" in run("math", "code")
    bank, manifest = public / "code/bank.jsonl", public / "code/manifest.json"
    good, described = bank.read_bytes(), manifest.read_bytes()
    with bank.open("a") as f:
        f.write(json.dumps({**json.loads(good.splitlines()[0]), "source": MBPP}) + "\n")
    for domains in (("code",), ("math",)):  # the folder is checked whatever was rebuilt
        with pytest.raises(AssertionError, match="drawn with MBPP"):
            run(*domains)
    bank.write_bytes(good)
    for path, data in ((manifest, described), (bank, good)):  # a split that stopped partway
        path.unlink()
        with pytest.raises(AssertionError, match="not a finished code bank"):
            run("math")
        path.write_bytes(data)
    # without reused labels too: an older public version of a rebuilt domain stops the cell, manifest or not
    old = built.tmp / "to_commit/v1/bank/full/code"
    old.mkdir(parents=True)
    (old / "labels_validation.jsonl").write_text("{}")
    with pytest.raises(AssertionError, match="another version of the same bank"):
        run("code")
    assert "labels_validation" not in capsys.readouterr().out
    (old.parent / "math").mkdir()
    shutil.rmtree(old)
    assert "code/labels_validation.jsonl" in run("code")  # math may stay in v1 when only code was rebuilt


@pytest.mark.skipif(not (REPO / "bank/full/code/bank.jsonl").is_file(), reason="code bank not committed yet")
def test_the_committed_code_bank_has_no_mbpp():
    assert {r["source"] for r in rows_of(REPO / "bank/full/code/bank.jsonl")} == {HUMANEVAL}


def test_error_tags_are_drawn_from_wrong_test_answers_and_added_to_the_label_file(built):
    out, private = built.public / "math", built.private / "math"
    test = {r["candidate_id"]: r for r in rows_of(private / "labels_test.jsonl")}
    wrong = {c for c, r in test.items() if r["label"] == 0}
    n = split_full.error_sheet(out, private, n=5)
    path = private / split_full.TAG_SHEET
    with path.open(encoding="utf-8", newline="") as f:
        sheet = list(csv.DictReader(f))
    assert n == len(sheet) == 5 and {r["candidate_id"] for r in sheet} < wrong
    with pytest.raises(SystemExit, match="already exists"):
        split_full.error_sheet(out, private)
    with pytest.raises(SystemExit, match="missing or not one of"):
        split_full.apply_tags(out, private, n=5)
    for r, tag in zip(sheet, (*split_full.ERROR_TAGS, split_full.ERROR_TAGS[0])):
        r["error_type"] = tag
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, sheet[0].keys())
        writer.writeheader()
        writer.writerows(sheet)
    with path.open(encoding="utf-8", newline="") as f:
        whole_sheet = f.read()
    path.write_text("\n".join(whole_sheet.splitlines()[:-1]) + "\n", encoding="utf-8")
    with pytest.raises(SystemExit, match="exactly the answers"):  # a dropped row must not shrink the sample
        split_full.apply_tags(out, private, n=5)
    path.write_text(whole_sheet, encoding="utf-8")
    before_tags = (out / "manifest.json").read_bytes()
    assert split_full.apply_tags(out, private, n=5) == 5
    tagged_files = [(out / "manifest.json").read_bytes(), (private / "labels_test.jsonl").read_bytes()]
    (out / "manifest.json").write_bytes(before_tags)  # stopped between the two writes: running it again finishes
    assert split_full.apply_tags(out, private, n=5) == 5
    assert tagged_files == [(out / "manifest.json").read_bytes(), (private / "labels_test.jsonl").read_bytes()]
    tagged = {r["candidate_id"]: r for r in rows_of(private / "labels_test.jsonl")}
    assert {c: r["error_type"] for c, r in tagged.items() if "error_type" in r} == {r["candidate_id"]: r["error_type"] for r in sheet}
    assert all({k: v for k, v in r.items() if k != "error_type"} == test[c] for c, r in tagged.items())
    manifest = json.loads((out / "manifest.json").read_text())
    assert manifest["files"]["labels_test.jsonl"] == sha256_bytes((private / "labels_test.jsonl").read_bytes())
    for out_again in ("commit/bank/full", "elsewhere/bank/full"):  # a rebuild would silently drop the tags
        with pytest.raises(SystemExit, match="error tags"):
            build(built, "math", out_again)
    path.unlink()
    assert split_full.error_sheet(out, private, n=10**6) == len(wrong)  # fewer than asked for: all of them
    with pytest.raises(SystemExit, match="only math"):
        split_full.error_sheet(built.public / "code", built.private / "code")


# ---------------------------------------------------------------- custody and notebooks

def ignored(path):
    return subprocess.run(["git", "check-ignore", "--no-index", "-q", path], cwd=REPO).returncode == 0


@pytest.mark.skipif(subprocess.run(["git", "rev-parse"], cwd=REPO, capture_output=True).returncode != 0, reason="not a git checkout")
def test_gitignore_keeps_labels_logs_and_tags_out_and_lets_the_bank_in():
    for path in ("math/labeled.jsonl", "code/labeled.jsonl", "labeled.jsonl", "code/labeled_manifest.json", "math/labels_test.jsonl",
                 "labels_test.jsonl", "math/labels_test_tagged.jsonl", "code/evalplus_checks.jsonl", "code/execution.log",
                 "code/logs/run1.txt", "math/error_tags_math.csv", "error_tags.jsonl", "math/full_labels.jsonl",
                 "math/drawn_labels.jsonl", "work/math/labeled.jsonl", "private/math/labels_test.jsonl", "math/generations.jsonl",
                 "gen/phi4mini_math/generations.jsonl", "gen/phi4mini_math/chunks/chunk_00000.jsonl", "math/work/labeled.jsonl",
                 "audit/sheet.jsonl", "audit_math/key.jsonl", "audit/replies/item_001.json"):  # the label-check files
        for root in ("bank/full", "bank/full_v2"):
            assert ignored(f"{root}/{path}"), f"{root}/{path}"
    for path in ("labels_test.jsonl", "error_tags_math.csv", "evalplus_checks.jsonl", "work/math/labeled.jsonl"):
        assert ignored(f"bank/{path}"), path
    for path in ("math/bank.jsonl", "math/labels_validation.jsonl", "math/manifest.json", "code/bank.jsonl",
                 "code/labels_validation.jsonl", "code/manifest.json", "questions_math.jsonl", "questions_code.jsonl",
                 "question_ids.txt", "questions_manifest.json", "excluded_questions.txt", "api_spend.json",
                 "gen/olmo2_code/generations_manifest.json"):
        assert not ignored(f"bank/full/{path}"), path
    assert not ignored("bank/pilot/math/labeled.jsonl") and not ignored("bank/pilot/swap/items.jsonl")


def notebook_lines(name):
    for cell in json.loads((REPO / f"notebooks/{name}.ipynb").read_text(encoding="utf-8"))["cells"]:
        if cell["cell_type"] == "code":
            yield from "".join(cell["source"]).splitlines()


@pytest.mark.parametrize("name", ["10_full_generate", "11_full_label_split"])
def test_notebook_commands_use_flags_that_exist(name):
    called = 0
    for line in notebook_lines(name):
        match = re.match(r"\s*!.*\bpython -m (bank\.\w+)(.*)", line)
        if match:
            called += 1
            source = (REPO / Path(*match[1].split("."))).with_suffix(".py").read_text(encoding="utf-8")
            for flag in re.findall(r"(--[a-z0-9-]+)", match[2]):
                assert f'"{flag}"' in source, f"{name}: {flag} not in {match[1]}"
    assert called >= 2


def test_the_label_notebook_keeps_test_labels_in_the_private_folder_and_prints_none():
    lines = list(notebook_lines("11_full_label_split"))
    assert "I_AM_THE_LABEL_KEEPER = False" in "\n".join(lines)
    assert all("{COMMIT}" not in line for line in lines if re.search(r"labeled|labels_test|checks|{WORK}/{d}/\.", line))
    # labels and pairs go to the private work folder through `save` only; the rest is copied after the split
    written = [line for line in lines if "{PRIVATE}/{d}/work" in line]
    assert len(written) == 2 and "bank.reuse_labels save" in written[0] and "path.name" in written[1]
    assert "\n".join(lines).index("split_full split") < "\n".join(lines).index(written[1])
    copied = next(line for line in lines if "labeled_manifest.json" in line and "too_long.json" in line)
    assert not any(f'"{name}"' in copied for name in reuse_labels.KEPT)
    assert not any("cp -r" in line for line in lines)
    assert not any(re.match(r"\s*!(cat|head|tail|less)\b", line) for line in lines)
    cpu = json.loads((REPO / "notebooks/11_full_label_split.ipynb").read_text())["metadata"]
    assert "accelerator" not in cpu
    # reused labels: both labeling cells are skipped, and the reuse cell runs before them
    text = "\n".join(lines)
    assert 'if "math" in DOMAINS and not REUSE_LABELS_FROM:' in text and 'if "code" in DOMAINS and not REUSE_LABELS_FROM:' in text
    reuse = next(line for line in lines if "bank.reuse_labels reuse" in line)
    assert "--old-public {OLD_COMMIT}/bank/full/{d} bank/full/{d}" in reuse
    assert text.index("bank.make_pairs") < text.index("bank.reuse_labels save") < text.index("bank.token_lengths")
    assert text.index("bank.reuse_labels reuse") < text.index("bank.label ") and 'REUSE_LABELS_FROM = ""' in text
    # model-written programs run as another user, before the login; every bank step but the tag sheet is checked
    text = "\n".join(lines)
    assert text.index("!runuser -u nobody") < text.index("login(token=")
    for i, line in enumerate(lines):
        if re.match(r"\s*!python -m bank\.", line) and "error-sheet" not in line:
            assert "_exit_code == 0" in lines[i + 1], line
    for name in ("10_full_generate", "11_full_label_split"):
        text = "\n".join(notebook_lines(name))
        assert "drive.flush_and_unmount()" in text and "runtime.unassign()" in text and "rev-parse" in text
