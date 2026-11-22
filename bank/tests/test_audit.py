"""CPU tests for the blind label audit: the sheet, the Codex runner (with a stand-in CLI), the human sheet, the rule."""

import csv
import json
import os
import random
import re
import stat
import sys
from collections import Counter
from pathlib import Path

import pytest

from bank import audit_sheet, codex_audit
from bank.audit_sheet import (bound_ok, decide, fenced, groups, load_bank, operating, p_value, sample, score, sheet_rows,
                              upper_bound, worst_false_pass)
from bank.common import read_jsonl, write_jsonl

PILOT = ["--bank", "bank/pilot/pilot_bank.jsonl", "--labels", "bank/pilot/pilot_labels.jsonl"]


# ---------------------------------------------------------------- the sheet

def make_bank(pairs=12, gpt=2):
    bank, labels = {}, {}
    for d in ("math", "code"):
        for p in range(pairs):
            gen = "openai/gpt-4.1-mini" if p < gpt else "microsoft/Phi-4-mini-instruct"
            for lab in (1, 0):
                cid = f"{d}{p}__tag{p}__{lab}"
                bank[cid] = {"candidate_id": cid, "pair_id": f"{d}{p}:{gen}", "problem_id": f"{d}{p}", "domain": d,
                             "problem": f"question {d} {p}", "candidate": f"candidate {d} {p} {lab}", "generator": gen}
                labels[cid] = lab
    return bank, labels


def test_sample_is_stratified_seeded_and_never_shows_a_whole_pair():
    bank, labels = make_bank()
    chosen = sample(bank, labels, per_stratum=5)
    assert Counter((bank[c]["domain"], labels[c]) for c in chosen) == {(d, l): 5 for d in ("math", "code") for l in (0, 1)}
    assert len({bank[c]["pair_id"] for c in chosen}) == len(chosen) == 20
    reordered = dict(sorted(bank.items(), reverse=True))
    assert chosen == sample(reordered, labels, per_stratum=5) != sample(bank, labels, seed=545, per_stratum=5)


def test_a_re_audit_takes_fresh_candidates_and_shortage_is_an_error():
    bank, labels = make_bank()
    first = sample(bank, labels, per_stratum=4)
    second = sample(bank, labels, per_stratum=4, exclude=set(first))
    assert not set(first) & set(second)
    with pytest.raises(SystemExit, match="short of"):
        sample(bank, labels, per_stratum=7)


def test_sheet_shows_only_the_problem_and_the_candidate():
    bank, labels = make_bank()
    chosen = sample(bank, labels, per_stratum=5)
    sheet, key = sheet_rows(bank, chosen)
    assert all(set(r) == {"audit_id", "domain", "problem", "candidate"} for r in sheet)
    assert [k["candidate_id"] for k in key] == chosen and all(set(k) == {"audit_id", "candidate_id"} for k in key)
    text = json.dumps(sheet)
    assert "tag" not in text and "Phi" not in text and "gpt" not in text
    # the sheet order says nothing about the label
    order = [labels[c] for c in sample(bank, labels, per_stratum=6)]
    assert order != sorted(order) and order != sorted(order, reverse=True)


def test_bank_and_labels_can_come_in_several_files(tmp_path):
    bank, labels = make_bank()
    for d in ("math", "code"):
        write_jsonl(tmp_path / f"{d}.jsonl", [r for r in bank.values() if r["domain"] == d])
        for lab in (0, 1):
            write_jsonl(tmp_path / f"{d}_{lab}.jsonl", [{"candidate_id": c, "label": l} for c, l in labels.items()
                                                        if c.startswith(d) and l == lab])
    banks, label_files = [str(tmp_path / f"{d}.jsonl") for d in ("math", "code")], sorted(map(str, tmp_path.glob("*_[01].jsonl")))
    assert load_bank(banks, label_files) == (bank, labels)
    with pytest.raises(SystemExit, match="no 0/1 label"):
        load_bank(banks, label_files[1:])
    with pytest.raises(SystemExit, match="twice"):
        load_bank(banks + banks[:1], label_files)


def test_pilot_sheet_is_the_one_the_code_draws(tmp_path):
    audit_sheet.main(["sample", *PILOT, "--out", str(tmp_path)])
    for name in ("audit_sheet.jsonl", "audit_key.jsonl", "audit_manifest.json"):
        assert (tmp_path / name).read_bytes() == Path("bank/pilot/audit", name).read_bytes()
    bank, labels = load_bank(PILOT[1:2], PILOT[3:4])
    key = read_jsonl("bank/pilot/audit/audit_key.jsonl")
    assert Counter((bank[k["candidate_id"]]["domain"], labels[k["candidate_id"]]) for k in key) == {
        (d, l): 50 for d in ("math", "code") for l in (0, 1)}
    sheet = read_jsonl("bank/pilot/audit/audit_sheet.jsonl")
    ids = {k["candidate_id"] for k in key} | {bank[k["candidate_id"]]["generator"] for k in key}
    text = json.dumps([{**r, "audit_id": ""} for r in sheet])
    assert not [i for i in ids if i in text]
    with pytest.raises(SystemExit, match="exists"):
        audit_sheet.main(["sample", *PILOT, "--out", str(tmp_path)])


# ---------------------------------------------------------------- the person's sheets

@pytest.fixture
def audit(tmp_path):
    """A 200-item audit of a synthetic bank; Codex agrees everywhere except where told."""
    bank, labels = make_bank(pairs=120, gpt=10)
    write_jsonl(tmp_path / "bank.jsonl", bank.values())
    write_jsonl(tmp_path / "labels.jsonl", [{"candidate_id": c, "label": l} for c, l in labels.items()])
    folder = tmp_path / "audit"
    args = ["--bank", str(tmp_path / "bank.jsonl"), "--labels", str(tmp_path / "labels.jsonl")]
    audit_sheet.main(["sample", *args, "--out", str(folder)])
    sheet, key = audit_sheet.load_audit(folder)

    class Audit:
        pass

    a = Audit()
    a.folder, a.args, a.sheet, a.key, a.bank, a.labels = folder, [*args, "--audit", str(folder)], sheet, key, bank, labels
    a.truth = {r["audit_id"]: "correct" if labels[key[r["audit_id"]]] else "wrong" for r in sheet}
    a.family = [i for i in a.truth if bank[key[i]]["generator"].startswith("openai/")]

    def codex(flip=(), status=()):
        rows = []
        for i, v in a.truth.items():
            v = {"correct": "wrong", "wrong": "correct"}[v] if i in flip else v
            rows.append({"audit_id": i, "status": dict(status).get(i, "ok"), "verdict": v})
        write_jsonl(folder / "codex" / "codex_answers.jsonl", rows)

    def answer(differ=(), number=None):
        for path in audit_sheet.answer_files(folder):
            rows = audit_sheet.read_answers(path)
            if number is None or path.stem.endswith(f"_{number}"):
                with open(path, "w", newline="", encoding="utf-8-sig") as f:      # as a spreadsheet would save it
                    w = csv.writer(f)
                    w.writerow(["audit_id", "verdict", "note"])
                    for r in rows:
                        i = r["audit_id"]
                        w.writerow([i, {"correct": "Wrong", "wrong": "Correct"}[a.truth[i]] if i in differ else a.truth[i].title(), ""])

    def result(*extra, errors=(), cleared=()):
        """Score, with the reader's second look: these are label errors, those are not."""
        look = tmp_path / "recheck.csv"
        look.write_text("audit_id,label_error,reason\n" + "".join(f"{i},yes,checked\n" for i in errors)
                        + "".join(f"{i},no,misread it\n" for i in cleared))
        audit_sheet.main(["score", *a.args, "--recheck", str(look), *extra])
        return json.loads((folder / "audit_result.json").read_text())

    a.codex, a.answer, a.result = codex, answer, result
    a.human = lambda n, *extra: audit_sheet.main(["human", *a.args, "--read", str(n), *extra])
    a.groups = lambda: groups(sheet, key, bank, labels, audit_sheet.load_codex(folder / "codex/codex_answers.jsonl", sheet), "openai/", 544)
    return a


def test_the_person_reads_disagreements_unusable_verdicts_gpt_items_and_random_others(audit):
    others = [i for i in audit.truth if i not in audit.family]
    audit.codex(flip=others[:6], status=[(others[6], "invalid"), (others[7], "failed")])
    with pytest.raises(SystemExit, match="1 Codex calls failed"):       # retried first
        audit.human(100)
    audit.human(100, "--allow-failed")
    ids = [r["audit_id"] for r in audit_sheet.read_answers(audit.folder / "human_answers_1.csv")]
    must = set(others[:8]) | set(audit.family)
    assert must <= set(ids) and len(ids) == len(set(ids)) == len(must) + 100
    assert ids != sorted(ids)
    front = sum(i in must for i in ids[:len(ids) // 2])          # the two kinds are mixed
    assert 0 < front < len(must)
    text = (audit.folder / "human_sheet_1.md").read_text() + (audit.folder / "human_answers_1.csv").read_text()
    for leak in ("tag", "openai", "gpt", "Phi", "codex", "Codex", "disagree", "our label", "reference"):
        assert leak not in text, leak
    assert [re.sub(r" .*", "", line[3:]) for line in text.splitlines() if line.startswith("## ")] == ids


def test_later_sheets_hold_only_new_items_and_end_at_all_200(audit):
    audit.codex()
    for read in (100, "all"):
        audit.human(read)
    sheets = [[r["audit_id"] for r in audit_sheet.read_answers(p)] for p in audit_sheet.answer_files(audit.folder)]
    assert [len(s) for s in sheets] == [len(audit.family) + 100, 100 - len(audit.family)]
    assert sorted(sum(sheets, [])) == sorted(audit.truth)
    with pytest.raises(SystemExit, match="nothing new"):
        audit.human("all")


def test_a_codex_rerun_after_the_first_sheet_changes_nothing(audit):
    others = [i for i in audit.truth if i not in audit.family]
    audit.codex(status=[(others[0], "failed")])
    audit.human(100, "--allow-failed")
    audit.answer()
    before = audit.result()
    audit.codex(flip=others[5:9])                     # the failed call was retried, others changed their mind
    assert audit.result() == before and before["decision"] == "PASS (shortcut)" and before["groups"]["no_verdict"] == 1
    audit.human("all")
    ids = [r["audit_id"] for p in audit_sheet.answer_files(audit.folder) for r in audit_sheet.read_answers(p)]
    assert sorted(ids) == sorted(audit.truth)


def test_an_unfinished_codex_run_is_refused(audit):
    audit.codex()
    rows = read_jsonl(audit.folder / "codex/codex_answers.jsonl")
    write_jsonl(audit.folder / "codex/codex_answers.jsonl", rows[:-1])
    with pytest.raises(SystemExit, match="does not cover"):
        audit.human(100)
    write_jsonl(audit.folder / "codex/codex_answers.jsonl", rows[:-1] + [{**rows[-1], "status": "missing", "verdict": None}])
    with pytest.raises(SystemExit, match="finish"):
        audit.human(100)


def test_fence_is_longer_than_any_backticks_inside():
    assert fenced("a") == "```\na\n```"
    assert fenced("x ``` y ````", "python").startswith("`````python\n")


# ---------------------------------------------------------------- the bound

def test_bound_is_the_ordinary_exact_one_at_the_ends():
    assert p_value(0, 200, 75, 0, 0) == pytest.approx(0.96 ** 75)
    assert bound_ok(0, 200, 74, 0, 0) and not bound_ok(0, 200, 73, 0, 0)
    for n_all in (0, 50, 200):                                      # everything read: PLAN's 3 in 200
        assert p_value(n_all, 200 - n_all, 200 - n_all, min(3, n_all), 3 - min(3, n_all)) == pytest.approx(0.03953, abs=1e-4)
    assert upper_bound(0, 200, 200, 0, 3) == pytest.approx(0.0383, abs=1e-4)
    assert upper_bound(0, 200, 200, 0, 4) > 0.04


def test_errors_hurt_and_what_passes_with_100_others_read():
    for n_all in (13, 30, 60):
        values = [p_value(n_all, 200 - n_all, 100, x, 0) for x in range(3)]
        assert values == sorted(values)
    for n_all, most in ((0, 1), (13, 1), (20, 1), (28, 1), (30, 2), (40, 2), (60, 2), (80, 3)):   # the table in the record
        assert [bound_ok(n_all, 200 - n_all, 100, x, 0) for x in range(4)] == [x <= most for x in range(4)]
        assert upper_bound(n_all, 200 - n_all, 100, most, 0) < 0.04
    assert p_value(30, 170, 0, 0, 0) == 1.0 and p_value(200, 0, 0, 3, 0) < 0.05 < p_value(200, 0, 0, 4, 0)


def draws(rng, q_all, q_rest, ok_all, runs):
    """Whole audits: (items read in full, errors among them, the others in random order as 0/1)."""
    for _ in range(runs):
        x = n_all = e = 0
        for _ in range(200):
            u = rng.random()
            if u < q_all:
                x, n_all = x + 1, n_all + 1
            elif u < q_all + ok_all:
                n_all += 1
            elif u < q_all + ok_all + q_rest:
                e += 1
        others = [1] * e + [0] * (200 - n_all - e)
        rng.shuffle(others)
        yield n_all, x, others


@pytest.mark.parametrize("q_all, ok_all", [(0, .10), (.006, .10), (.010, .15), (.02, .02), (.04, 0)])
def test_simulation_passes_a_four_percent_bank_at_most_five_percent_of_the_time(q_all, ok_all):
    runs = 4000
    short = overall = 0
    for n_all, x, others in draws(random.Random(544), q_all, .04 - q_all, ok_all, runs):
        passes = decide(n_all, len(others), x, 100, sum(others[:100])) == "PASS"
        short += passes
        overall += passes or x + sum(others) <= 3                   # else all 200 are read
    noise = 3 * (0.05 * 0.95 / runs) ** 0.5
    assert overall / runs <= 0.05 + noise
    exact = operating(q_all, .04 - q_all, ok_all)                   # the exact sums match the simulation
    assert short / runs == pytest.approx(exact["shortcut_pass"], abs=noise)
    assert overall / runs == pytest.approx(exact["pass"], abs=noise)


def test_the_whole_procedure_passes_a_four_percent_bank_at_most_five_percent_of_the_time():
    for ok_all in (.05, .10, .15, .20, .30):
        w = worst_false_pass(ok_all)
        assert w["pass"] <= 0.05 and w["plan_pass"] == pytest.approx(0.0395, abs=1e-3)
    assert worst_false_pass(0)["pass"] == pytest.approx(0.051, abs=1e-3)     # no good item read in full at all
    assert worst_false_pass(.05, others=90)["pass"] > 0.05                   # 90 would not be enough


def test_75_then_118_was_rejected_because_it_passes_more_often():
    """The history in the decision record: 7-9% for a 4% bank, from three chances to pass."""
    for ok_all, low, high in ((0, .08, .10), (.10, .07, .09)):
        runs = passed = 0
        for n_all, x, others in draws(random.Random(544), .004 if ok_all else 0, .036 if ok_all else .04, ok_all, 6000):
            k1, k2 = sum(others[:75]), sum(others[:118])
            short = (k1 == 0 and x <= 3 and bound_ok(n_all, len(others), 75, x, 0)) or (
                k1 == 1 and x + k2 <= 3 and bound_ok(n_all, len(others), 118, x, k2))
            runs, passed = runs + 1, passed + (short or x + sum(others) <= 3)
        assert low < passed / runs < high


def test_the_rule_passes_good_banks_often():
    clean = operating(0, 0, .10)
    assert clean["shortcut_pass"] == pytest.approx(1, abs=1e-6) and clean["read"] == pytest.approx(120, abs=1)
    half = operating(.75 * .005, .25 * .005, .10)
    assert half["shortcut_pass"] > 0.7 and half["pass"] >= half["plan_pass"] - 1e-9


# ---------------------------------------------------------------- the decision

@pytest.mark.parametrize("n_all, x, read, k, expected", [
    (30, 0, 100, 0, "PASS"),
    (10, 1, 100, 0, "PASS"),
    (10, 2, 100, 0, "READ ALL"),            # no error among the others, but the bound is above 4%
    (30, 2, 100, 0, "PASS"),
    (80, 3, 100, 0, "PASS"),
    (60, 4, 100, 0, "READ ALL"),            # 4 found: the shortcut never fails a bank, the full read does
    (30, 0, 100, 1, "READ ALL"),            # any error among the others
    (80, 0, 100, 1, "READ ALL"),            # also where the bound alone would allow it
    (30, 0, 99, 0, "READ ALL"),
    (30, 0, 118, 0, "READ ALL"),            # there is no other stop
    (30, 0, 170, 3, "PASS"),
    (30, 3, 170, 1, "FAIL"),
    (130, 3, 70, 0, "PASS"),                # fewer than 100 others: all are read
])
def test_decisions(n_all, x, read, k, expected):
    assert decide(n_all, 200 - n_all, x, read, k) == expected


def test_score_passes_and_reports(audit):
    others = [i for i in audit.truth if i not in audit.family]
    audit.codex(flip=others[:20])
    audit.human(100)
    with pytest.raises(SystemExit, match="no verdict"):
        audit.result()
    audit.answer(differ=others[:1])
    r = audit.result()
    assert r["decision"] == "PASS (shortcut)" and r["label_errors_found"] == 1
    assert r["errors_by_group"] == {"disagree": 1, "no_verdict": 0, "same_family": 0, "others_read": 0}
    assert r["groups"] == {"disagree": 20, "no_verdict": 0, "same_family": len(audit.family), "rest": 180 - len(audit.family)}
    assert r["upper_bound_95"] < 0.04 and r["read"] == 120 + len(audit.family)
    assert [d["audit_id"] for d in r["differences"]] == others[:1]
    assert sum(v["agrees_with_label"] for v in r["codex_agreement"].values()) == 180


def test_one_error_among_the_others_means_reading_all(audit):
    others = [i for i in audit.truth if i not in audit.family]
    audit.codex(flip=others[:20])
    audit.human(100)
    one = audit.groups()["rest"][10:11]
    audit.answer(differ=one)
    r = audit.result()
    assert (r["decision"], r["waiting_for_second_look"]) == ("SECOND LOOK", one)     # before any more reading
    assert audit.result(cleared=one)["decision"] == "PASS (shortcut)"                # a misreading: nothing more to read
    assert audit.result(errors=one)["decision"] == "READ ALL"
    audit.human("all")
    audit.answer(differ=one)
    r = audit.result(errors=one)
    assert (r["decision"], r["read"]) == ("PASS (all items read)", 200)
    with pytest.raises(SystemExit):
        audit.human(118)


def test_when_the_shortcut_does_not_pass_all_200_are_read_and_plans_rule_decides(audit):
    audit.codex()
    audit.human(100)
    rest = audit.groups()["rest"]
    bad = rest[:2]
    audit.answer(differ=bad)
    assert audit.result()["decision"] == "SECOND LOOK"
    assert audit.result(errors=bad)["decision"] == "READ ALL"
    audit.human("all")
    audit.answer(differ=bad + rest[-1:])
    r = audit.result()                                # 3 in 200 passes even without a second look
    assert (r["decision"], r["label_errors_found"]) == ("PASS (all items read)", 3)
    audit.answer(differ=bad + rest[-2:])
    assert audit.result()["decision"] == "SECOND LOOK"                               # never a fail before it
    assert audit.result(errors=bad + rest[-2:])["decision"] == "FAIL (all items read)"
    assert audit.result(errors=bad + rest[-2:-1], cleared=rest[-1:])["decision"] == "PASS (all items read)"


def test_unsure_counts_as_an_error_until_the_second_look(audit, tmp_path):
    others = [i for i in audit.truth if i not in audit.family]
    audit.codex(flip=others[:20])
    audit.human(100)
    audit.answer(differ=others[:4])
    assert audit.result(errors=others[:4])["decision"] == "READ ALL"     # 4 errors
    r = audit.result(cleared=others[:3])
    assert (r["decision"], r["label_errors_found"], r["reader_differs_blind"], r["second_looks"]) == ("PASS (shortcut)", 1, 4, 3)
    for bad in (f"{others[5]},no,agreed anyway\n", f"{others[0]},maybe,x\n", f"{others[0]},no,\n"):
        look = tmp_path / "bad.csv"
        look.write_text("audit_id,label_error,reason\n" + bad)
        with pytest.raises(SystemExit, match="must be"):
            audit.result("--recheck", str(look))


def test_answers_outside_the_drawn_order_are_refused(audit):
    audit.codex()
    g = audit.groups()
    codex = audit_sheet.load_codex(audit.folder / "codex/codex_answers.jsonl", audit.sheet)
    def run(ids):
        answers = {i: {"verdict": audit.truth[i], "note": ""} for i in ids}
        return score(audit.sheet, audit.key, audit.bank, audit.labels, codex, answers, lambda d: {})
    assert run(g["read_all"] + g["rest"][:100])["decision"] == "PASS (shortcut)"
    for ids, why in ((g["read_all"] + g["rest"][1:101], "drawn order"), (g["read_all"] + g["rest"][:99], "the rule needs"),
                     (g["read_all"][1:] + g["rest"][:100], "must be read")):
        with pytest.raises(SystemExit, match=why):
            run(ids)


# ---------------------------------------------------------------- the Codex runner, with a stand-in CLI

FAKE = r'''#!{python}
import json, os, sys
if sys.argv[1:2] == ["--version"]:
    print("codex-cli 0.155.1"); sys.exit()
if sys.argv[1:3] == ["debug", "models"]:
    print(json.dumps({{"models": [{{"slug": "gpt-6.2-sol", "tool_mode": "code_mode_only", "shell_type": "unified_exec", "keep": 1}}]}})); sys.exit()
prompt = sys.stdin.buffer.read().decode("utf-8")
log = {{"argv": sys.argv[1:], "cwd": os.getcwd(), "files": sorted(os.listdir(".")), "item": open("item.md", newline="").read() == prompt,
       "env": sorted(os.environ), "home": os.environ.get("CODEX_HOME")}}
open({log!r}, "a").write(json.dumps(log) + "\n")
def event(**e): print(json.dumps(e))
event(type="thread.started", thread_id="t"); event(type="turn.started")
mode = open({mode!r}).read()
if mode == "fail": event(type="error", message="usage limit"); sys.exit(1)
if mode == "tool": event(type="item.completed", item={{"type": "command_execution", "command": "cat /etc/hosts"}})
if mode == "search": event(type="item.started", item={{"type": "web_search", "query": "q"}})
if mode == "hang":
    event(type="item.completed", item={{"type": "command_execution", "command": "ls"}}); sys.stdout.flush()
    import time; time.sleep(20)
if mode == "write": open("notes.txt", "w").write("x")
if mode == "new": event(type="brand.new.event")
text = "not json" if mode == "text" else json.dumps({{"verdict": "correct", "answer": os.getcwd(), "reason": "r"}})
event(type="item.completed", item={{"type": "reasoning", "text": "..."}})
event(type="item.completed", item={{"type": "agent_message", "text": text}}); event(type="turn.completed", usage={{}})
'''


@pytest.fixture
def runner(tmp_path):
    sheet = tmp_path / "audit_sheet.jsonl"
    write_jsonl(sheet, [{"audit_id": f"a00{i}", "domain": d, "problem": f"problem {i}", "candidate": "```py\r\nx = '{y}'\n```"}
                        for i, d in ((1, "math"), (2, "code"), (3, "math"))])
    mode, log, home = tmp_path / "mode", tmp_path / "log.jsonl", tmp_path / "home"
    home.mkdir()
    (home / "auth.json").write_text("{}")
    fake = tmp_path / "codex"
    fake.write_text(FAKE.format(python=sys.executable, mode=str(mode), log=str(log)))
    fake.chmod(fake.stat().st_mode | stat.S_IXUSR)

    class Runner:
        pass

    r = Runner()
    r.out, r.home, r.sheet = tmp_path / "codex_out", home, sheet
    r.calls = lambda: [json.loads(line) for line in log.read_text().splitlines()] if log.exists() else []
    r.answers = lambda: {a["audit_id"]: a for a in read_jsonl(r.out / "codex_answers.jsonl")}

    def run(*extra, how="ok"):
        mode.write_text(how)
        codex_audit.main(["--sheet", str(sheet), "--out", str(r.out), "--codex", str(fake), "--codex-home", str(home), *extra])
    r.run = run
    return r


def test_each_item_runs_alone_outside_the_project_with_no_tools(runner, monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "secret")
    runner.run()
    calls = runner.calls()
    assert len(calls) == 3 and len({c["cwd"] for c in calls}) == 3
    for c in calls:
        assert c["files"] == ["item.md"] and c["item"]
        assert not c["cwd"].startswith(os.getcwd()) and codex_audit.inside_project(Path(c["cwd"])) is None
        assert "OPENAI_API_KEY" not in c["env"] and c["home"] == str(runner.home.resolve())
        a = " ".join(c["argv"])
        for flag in ("--model gpt-6.2-sol", "--sandbox read-only", "--ephemeral", "--ignore-user-config", "--ignore-rules", "--json",
                     "--disable shell_tool", "--disable unified_exec", 'web_search="disabled"', "project_doc_max_bytes=0"):
            assert flag in a
        assert not os.path.exists(c["cwd"])
    catalog = json.loads((runner.home / codex_audit.CATALOG_FILE).read_text())
    assert catalog == {"models": [{"slug": "gpt-6.2-sol", "tool_mode": None, "shell_type": "disabled", "keep": 1}]}


def test_prompt_holds_only_the_question_the_problem_and_the_candidate(runner):
    item = read_jsonl(runner.sheet)[1]
    prompt = codex_audit.prompt_for(item)
    assert item["problem"] in prompt and item["candidate"] in prompt and "a002" not in prompt
    assert "every input" in prompt and "\\boxed" in codex_audit.prompt_for(read_jsonl(runner.sheet)[0])


def test_everything_is_saved_without_local_paths(runner):
    runner.run()
    settings = json.loads((runner.out / "codex_settings.json").read_text())["settings"]
    assert settings["model"] == "gpt-6.2-sol" and settings["prompt"] == codex_audit.PROMPT
    raw = json.loads((runner.out / "raw/a001.json").read_text())
    assert raw["prompt"] == codex_audit.prompt_for(read_jsonl(runner.sheet)[0]) and raw["codex_version"] == "codex-cli 0.155.1"
    assert runner.answers()["a001"]["verdict"] == "correct" and runner.answers()["a001"]["answer"] == "{WORKDIR}"
    for path in runner.out.rglob("*"):
        if path.is_file():
            text = path.read_text()
            assert str(Path.home()) not in text and str(runner.home) not in text and "label_audit_" not in text.replace("label_audit_model_catalog", "")


def test_it_resumes_and_retries_only_failed_calls(runner):
    runner.run("--limit", "1")
    assert len(runner.calls()) == 1 and runner.answers()["a002"]["status"] == "missing"
    with pytest.raises(SystemExit, match="2 items missing"):   # an unfinished run cannot be used
        audit_sheet.load_codex(runner.out / "codex_answers.jsonl", read_jsonl(runner.sheet))
    runner.run("--limit", "1", how="fail")
    assert runner.answers()["a002"]["status"] == "failed"
    runner.run("--limit", "1", how="tool")               # retried, and now invalid
    assert runner.answers()["a002"]["status"] == "invalid"
    runner.run()
    assert {a: r["status"] for a, r in runner.answers().items()} == {"a001": "ok", "a002": "invalid", "a003": "ok"}
    assert len(runner.calls()) == 4                      # the invalid item is not asked again


@pytest.mark.parametrize("how, problem", [("tool", "command_execution: cat /etc/hosts"), ("search", "web_search"),
                                          ("write", "notes.txt"), ("new", "unknown event")])
def test_any_tool_use_makes_the_item_invalid_and_hides_its_verdict(runner, how, problem):
    runner.run("--limit", "1", how=how)
    a = runner.answers()["a001"]
    assert a["status"] == "invalid" and a["verdict"] is None and problem in a["problems"][0]


def test_a_reply_without_a_verdict_is_failed_and_repeated_failures_stop_the_run(runner):
    runner.run("--limit", "1", how="text")
    assert runner.answers()["a001"]["status"] == "failed"
    runner.run("--max-failures", "2", how="fail")
    assert len(runner.calls()) == 3                      # 1 + 2, then stopped


def test_a_timeout_keeps_the_log_so_far_and_failed_attempts_stay_on_record(runner):
    runner.run("--limit", "1", "--timeout", "2", how="hang")
    a = runner.answers()["a001"]
    assert a["status"] == "invalid" and "command_execution" in a["problems"][0]      # tool use before the timeout
    runner.run("--limit", "1", how="fail")
    runner.run("--limit", "1")
    raw = json.loads((runner.out / "raw/a002.json").read_text())
    assert runner.answers()["a002"]["status"] == "ok" and [e["returncode"] for e in raw["earlier_attempts"]] == [1]


def test_judge_reads_the_last_message_and_flags_unreadable_logs():
    def log(*events):
        return "\n".join(e if isinstance(e, str) else json.dumps(e) for e in events)
    def say(verdict):
        return {"type": "item.completed", "item": {"type": "agent_message", "text": json.dumps({"verdict": verdict})}}
    assert codex_audit.judge(log(say("wrong"), say("correct")), 0, [])["verdict"] == "correct"
    assert codex_audit.judge(log(say("unsure")), 0, [])["verdict"] == "unsure"
    assert codex_audit.judge(log(say("maybe")), 0, [])["status"] == "failed"
    assert codex_audit.judge(log(say("wrong")), 1, [])["status"] == "failed"
    assert codex_audit.judge(log("not json", say("wrong")), 0, [])["status"] == "invalid"
    assert codex_audit.judge("", 0, [])["status"] == "failed"
    lost = {"type": "error", "message": "Reconnecting... 1/5 (stream disconnected before completion)"}
    again = codex_audit.judge(log(lost, say("wrong"), {"type": "turn.completed"}), 0, [])     # the connection came back
    assert (again["status"], again["verdict"], again["problems"]) == ("ok", "wrong", [])
    for events, code in (((lost, {"type": "turn.failed"}, say("wrong")), 0), ((say("wrong"), lost), 1), ((lost,), 0)):
        gone = codex_audit.judge(log(*events), code, [])
        assert gone["status"] == "failed" and gone["verdict"] is None and "Reconnecting" in gone["problems"][-1]
    for kind in ("file_change", "mcp_tool_call", "todo_list", "collab_tool_call", None):
        assert codex_audit.judge(log({"type": "item.completed", "item": {"type": kind}}, say("wrong")), 0, [])["status"] == "invalid"


def test_dry_run_calls_and_writes_nothing(runner, capsys):
    runner.run("--dry-run")
    assert not runner.calls() and not runner.out.exists()
    out = capsys.readouterr().out
    assert "3 calls would run" in out and "--disable shell_tool" in out and "problem 1" in out


def test_unsafe_setups_are_refused(runner, tmp_path):
    runner.run("--limit", "1")
    for extra in (["--model", "gpt-6.2-mini"], ["--effort", "low"]):
        with pytest.raises(SystemExit, match="other settings"):
            runner.run(*extra)
    (runner.home / "AGENTS.md").write_text("notes")
    with pytest.raises(SystemExit, match="AGENTS.md"):
        runner.run()
    (runner.home / "AGENTS.md").unlink()
    (runner.home / "auth.json").unlink()
    with pytest.raises(SystemExit, match="codex login"):
        runner.run()
    assert codex_audit.inside_project(Path.cwd()) is not None
    with pytest.raises(SystemExit, match="not in this account"):
        codex_audit.tool_free_catalog([{"slug": "other"}], "gpt-6.2-sol")
    write_jsonl(runner.sheet, [{**r, "label": 1} for r in read_jsonl(runner.sheet)])
    with pytest.raises(SystemExit, match="only audit_id"):
        runner.run()


def test_pilot_prompts_never_hold_the_key():
    key = {r["audit_id"]: r["candidate_id"] for r in read_jsonl("bank/pilot/audit/audit_key.jsonl")}
    for item in read_jsonl("bank/pilot/audit/audit_sheet.jsonl"):
        prompt = codex_audit.prompt_for(item)
        assert key[item["audit_id"]] not in prompt and prompt.count("===== CANDIDATE =====") == 1
