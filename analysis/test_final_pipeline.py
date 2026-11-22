"""The validation and test steps end to end on a synthetic study, with a real git repository. CPU only.

The bootstrap is cut to 200 replicates here (rules are passed in by the tests; the command line always uses the
fixed ones)."""
import contextlib
import hashlib
import io
import json
import subprocess
import sys
import tempfile
import types
import unittest
from dataclasses import replace
from pathlib import Path
from unittest import mock

import numpy as np

from analysis import final_evaluate, final_validation
from analysis.final_data import DataError, read_jsonl
from analysis.final_evaluate import NotFrozen, evaluate_step, join_event, label_arguments, reading, run
from analysis.final_report import write_report
from analysis.final_settings import RULES, parse_settings
from analysis.final_stats import decide
from analysis.final_synthetic import GO, SEPARATION, write_jsonl, write_study
from analysis.final_validation import (build_manifest, check_manifest, hash_problems, load_inputs, manifest_bytes, select,
                                       validation_step)
from analysis.metrics import pair_thresholds
from analysis.overlap import ABOVE, BELOW, INCONCLUSIVE, ON

FAST = replace(RULES, bootstrap=200)
SCORES = ["results/full/math", "results/full/code"]
LABELS = {d: Path(f"bank/full/{d}/labels_test.jsonl") for d in ("math", "code")}
MANIFEST = "results/final/analysis_manifest.json"


def git(root, *args):
    subprocess.run(["git", "-C", str(root), "-c", "user.name=t", "-c", "user.email=t@example.com", *args], check=True,
                   capture_output=True)


class Repo(unittest.TestCase):
    settings, study = GO, {}

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.truth = write_study(self.root, seed=3, problems={"math": 60, "code": 30}, settings=self.settings, **self.study)
        git(self.root, "init", "-q")
        (self.root / ".gitignore").write_text("labels_test.jsonl\n", encoding="utf-8")

    def validate(self, rules=FAST):
        manifest = build_manifest(self.root, "configs/final_models.json", "bank/full", SCORES, "results/final", rules=rules)
        (self.root / MANIFEST).parent.mkdir(parents=True, exist_ok=True)
        (self.root / MANIFEST).write_bytes(manifest_bytes(manifest))
        return manifest

    def commit(self):
        git(self.root, "add", "-A")
        git(self.root, "commit", "-q", "-m", "freeze")

    def freeze(self):
        manifest = self.validate()
        self.commit()
        return manifest

    def test_run(self, out="results/final/test", labels=LABELS):
        return run(self.root, MANIFEST, labels, out, rules=FAST)

    test_run.__test__ = False

    def log(self):
        path = self.root / "results/final/join_log.jsonl"
        return read_jsonl(path) if path.is_file() else []


class GateTests(Repo):
    """What stands in front of the test labels."""

    def refused(self, fragment, **kwargs):
        with self.assertRaises(DataError) as caught:
            self.test_run(**kwargs)
        self.assertIn(fragment, str(caught.exception))
        self.assertEqual(self.log(), [])

    def test_no_manifest(self):
        self.commit()
        self.refused("no analysis manifest")

    def test_manifest_not_committed(self):
        self.commit()
        self.validate()
        self.refused("analysis_manifest.json: not committed")

    def test_manifest_staged_only(self):
        self.commit()
        self.validate()
        git(self.root, "add", "-A")
        self.refused("differs from the committed version")

    def test_manifest_edited_after_the_commit(self):
        self.freeze()
        m = json.loads((self.root / MANIFEST).read_text())
        m["validation"]["thresholds"]["math"]["Gemma-2-9B + Qwen2.5-7B"]["0.90"]["a"] += 0.125
        (self.root / MANIFEST).write_bytes(manifest_bytes(m))
        self.refused("differs from the committed version")
        self.commit()       # a committed edit is still not what the validation step computes
        self.refused("Gemma-2-9B + Qwen2.5-7B.0.90.a")

    def test_code_changed_after_the_freeze(self):
        self.freeze()
        path = self.root / "analysis/final_stats.py"
        path.write_text(path.read_text() + "# edited\n", encoding="utf-8")
        self.refused("final_stats.py: differs from the committed version")
        self.commit()       # committing the edit does not help: the manifest holds the frozen code's hash
        self.refused("final_stats.py: changed since the manifest was written")

    def test_inputs_changed_after_the_freeze(self):
        self.freeze()
        for path, what in (("bank/full/math/labels_validation.jsonl", "labels_validation.jsonl: changed"),
                           ("results/full/math/llada/full_T1_n1.jsonl", "full_T1_n1.jsonl: changed"),
                           ("bank/full/math/bank.jsonl", "bank.jsonl: changed")):
            file = self.root / path
            before = file.read_bytes()
            file.write_bytes(before + b"\n")
            self.refused(what)
            file.write_bytes(before)

    def test_test_labels_must_be_the_ones_frozen_with_the_bank(self):
        self.freeze()
        path = self.root / LABELS["math"]
        write_jsonl(path, read_jsonl(path)[::-1])
        self.refused("does not match the hash recorded at the bank freeze")

    def test_test_labels_kept_outside_the_repository(self):
        self.freeze()
        with tempfile.TemporaryDirectory() as elsewhere:
            paths = {}
            for d, p in LABELS.items():
                paths[d] = Path(elsewhere) / f"{d}_custody.jsonl"
                (self.root / p).rename(paths[d])
            self.test_run(labels=paths)
            text = (self.root / "results/final/join_log.jsonl").read_text()
        self.assertNotIn(elsewhere, text)       # no local paths in a file that is committed
        self.assertTrue(self.log()[0]["labels"]["code"]["matches_bank_manifest"])

    def test_labels_of_every_domain_are_joined_together(self):
        self.freeze()
        self.refused("exactly ['code', 'math']", labels={"math": LABELS["math"]})

    def test_validation_labels_cannot_stand_in_for_test_labels(self):
        self.freeze()
        self.refused("does not match the hash recorded at the bank freeze",
                     labels={**LABELS, "math": Path("bank/full/math/labels_validation.jsonl")})

    def test_single_join_is_logged_and_only_an_identical_run_may_follow(self):
        manifest = self.freeze()
        first = self.test_run()
        entry, done = self.log()
        self.assertEqual((entry["event"], set(entry["labels"])), ("join", {"math", "code"}))
        self.assertEqual(done["event"], "completed")
        self.assertTrue(entry["labels"]["math"]["matches_bank_manifest"])
        self.assertEqual(len(entry["git_commit"]), 40)
        with self.assertRaises(DataError):      # the results folder is never overwritten
            self.test_run()
        again = self.test_run(out="results/final/reproduction")
        self.assertEqual([e["event"] for e in self.log()], ["join", "completed", "rerun", "completed"])
        self.assertEqual(json.dumps(first["h1"], default=float), json.dumps(again["h1"], default=float))
        self.assertEqual(first["h1"]["pairs"], manifest["validation"]["h1"]["selected"])
        other = {**manifest, "created_utc": "a later freeze"}      # recomputes, but it is another manifest
        (self.root / MANIFEST).write_bytes(manifest_bytes(other))
        self.commit()
        with self.assertRaises(NotFrozen) as caught:
            self.test_run(out="results/final/second")
        self.assertIn("already joined", str(caught.exception))
        self.assertEqual(len(self.log()), 4)

    def test_the_manifest_cannot_be_rewritten_after_the_join(self):
        self.freeze()
        self.test_run()
        with contextlib.redirect_stdout(io.StringIO()) as printed:
            code = final_validation.main(["--root", str(self.root), "--settings", "configs/final_models.json", "--scores", *SCORES])
        self.assertEqual(code, 2)
        self.assertIn("can no longer change", printed.getvalue())
        with contextlib.redirect_stdout(io.StringIO()) as printed:     # another output folder does not help
            code = final_validation.main(["--root", str(self.root), "--settings", "configs/final_models.json", "--scores", *SCORES,
                                          "--out", "results/final/v2"])
        self.assertEqual(code, 2)
        self.assertIn("can no longer change", printed.getvalue())

    def test_a_second_freeze_elsewhere_cannot_join_again(self):
        """One join log for the repository: a manifest in another folder, with other settings, is refused."""
        self.freeze()
        self.test_run()
        settings = {**GO, "mode": "pivot", "llada_score": None, "reason": "second thoughts", "dropped": {"LLaDA-8B": "x"}}
        (self.root / "configs/final_models.json").write_text(json.dumps(settings), encoding="utf-8")
        other = build_manifest(self.root, "configs/final_models.json", "bank/full", SCORES, "results/final/v2", rules=FAST)
        (self.root / "results/final/v2").mkdir()
        (self.root / "results/final/v2/analysis_manifest.json").write_bytes(manifest_bytes(other))
        self.commit()
        with self.assertRaises(NotFrozen) as caught:
            run(self.root, "results/final/v2/analysis_manifest.json", LABELS, "results/final/v2/test", rules=FAST)
        self.assertIn("already joined", str(caught.exception))

    def test_a_committed_join_log_cannot_be_removed(self):
        self.freeze()
        self.test_run()
        self.commit()
        (self.root / "results/final/join_log.jsonl").unlink()
        with self.assertRaises(NotFrozen) as caught:
            self.test_run(out="results/final/again")
        self.assertIn("is missing here", str(caught.exception))

    def test_switching_branches_or_damaging_the_log_does_not_hide_a_join(self):
        self.freeze()
        git(self.root, "branch", "-M", "main")
        git(self.root, "checkout", "-q", "-b", "test-step")
        self.test_run()
        log = self.root / "results/final/join_log.jsonl"
        joined = log.read_text()
        self.commit()
        git(self.root, "checkout", "-q", "main")         # the log is only on the other branch
        self.assertFalse(log.exists())
        with self.assertRaises(NotFrozen) as caught:
            self.test_run(out="results/final/again")
        self.assertIn("is missing here", str(caught.exception))
        with contextlib.redirect_stdout(io.StringIO()) as printed:
            code = final_validation.main(["--root", str(self.root), "--settings", "configs/final_models.json", "--scores", *SCORES])
        self.assertEqual(code, 2)
        self.assertIn("is missing here", printed.getvalue())
        log.parent.mkdir(parents=True, exist_ok=True)
        only_completed = "".join(l + "\n" for l in joined.splitlines() if '"completed"' in l)
        for damaged in ("", only_completed):
            log.write_text(damaged, encoding="utf-8")
            with self.assertRaises(NotFrozen):
                self.test_run(out="results/final/again")
        log.write_text(joined, encoding="utf-8")         # with the log brought over, only an identical run goes on
        self.test_run(out="results/final/again")
        self.assertEqual([e["event"] for e in self.log()], ["join", "completed", "rerun", "completed"])

    def test_a_damaged_log_that_was_never_committed(self):
        self.freeze()
        self.test_run()
        log = self.root / "results/final/join_log.jsonl"
        only_completed = "".join(l + "\n" for l in log.read_text().splitlines() if '"completed"' in l)
        for damaged in ("", only_completed):
            log.write_text(damaged, encoding="utf-8")
            with self.assertRaises(NotFrozen) as caught:
                self.test_run(out="results/final/again")
            self.assertIn("damaged", str(caught.exception))

    def test_join_event(self):
        log = [{"event": "join", "utc": "t", "manifest_sha256": "m", "labels": {"math": {"sha256": "a"}}}]
        self.assertEqual(join_event([], "m", {"math": "a"}), "join")
        self.assertEqual(join_event(log, "m", {"math": "a"}), "rerun")
        for manifest, labels in (("other", {"math": "a"}), ("m", {"math": "b"})):
            with self.assertRaises(NotFrozen):
                join_event(log, manifest, labels)

    def test_label_arguments(self):
        self.assertEqual(label_arguments(["math=a", "code=b"]), {"math": Path("a"), "code": Path("b")})
        for bad in (["math"], ["math=a", "math=b"]):
            with self.assertRaises(DataError):
                label_arguments(bad)


class ValidationTests(Repo):
    def test_validation_never_needs_or_reads_test_labels(self):
        with_labels = manifest_bytes(self.validate())
        for path in LABELS.values():
            (self.root / path).unlink()
        self.assertEqual(with_labels, manifest_bytes(self.validate()))

    def test_manifest_contents(self):
        m = self.validate()
        banks = {d: read_jsonl(self.root / f"bank/full/{d}/bank.jsonl") for d in ("math", "code")}
        self.assertEqual(m["weights"], {d: len({r["pair_id"] for r in rows if r["split"] == "test"}) for d, rows in banks.items()})
        self.assertEqual((m["bootstrap"]["replicates"], m["bootstrap"]["seed"]), (200, 544))
        self.assertEqual(m["rules"]["match_tar_points"], 5.0)
        self.assertEqual(m["scores"]["models"]["LLaDA-8B T1"]["n_slots"], [1])
        for d in ("math", "code"):
            for name, by_tar in m["validation"]["thresholds"][d].items():
                self.assertEqual(sorted(by_tar), ["0.80", "0.90", "0.95"], name)
                for key, e in by_tar.items():
                    self.assertGreaterEqual(e["tar"], float(key) - 1e-12, name)
        for key in ("bank", "bank_manifest", "labels_validation"):
            self.assertEqual(len(m["data"]["math"][key]["sha256"]), 64)

    def test_the_record_must_exist(self):
        (self.root / GO["record"]).unlink()
        with self.assertRaises(DataError):
            self.validate()

    def test_thresholds_and_selection_by_hand(self):
        m = self.validate()
        _, _, frames, _ = load_inputs(self.root, parse_settings(GO), "bank/full", SCORES, ("math", "code"), FAST)
        f, best = frames["math"], {}
        for group, names in m["validation"]["h1"]["candidates"].items():
            for name in names:
                a, b = name.split(" + ")
                x, y = f.scores[(a, "T1")], f.scores[(b, "T1")]
                ta, tb, _ = pair_thresholds(x[f.correct], y[f.correct], 0.9)
                e = m["validation"]["thresholds"]["math"][name]["0.90"]
                self.assertEqual((e["a"], e["b"]), (ta, tb))
                best.setdefault(group, []).append((np.mean((x[f.wrong] >= ta) & (y[f.wrong] >= tb)), name))
        for group, rows in best.items():
            lowest = min(v for v, _ in rows)
            self.assertIn(m["validation"]["h1"]["selected"][group], [n for v, n in rows if v == lowest])
        stability = m["validation"]["h1"]["selection_stability"]
        for group in stability.values():
            self.assertAlmostEqual(sum(group.values()), 1.0)

    def test_ties_go_to_the_higher_mean_auroc_then_alphabetical(self):
        from analysis.final_settings import Member, make_pair
        pairs = [make_pair(Member(a), Member(b), ()) for a, b in (("Qwen2.5-7B", "Llama-3.1-8B"), ("Gemma-2-9B", "Qwen2.5-7B"),
                                                                 ("Gemma-2-9B", "Llama-3.1-8B"))]
        entries = {p.name: {"0.90": {"accepted_wrong": 10}} for p in pairs}
        flat = {n: {"auroc": 0.7} for n in ("Qwen2.5-7B", "Llama-3.1-8B", "Gemma-2-9B")}
        self.assertEqual(select(pairs, entries, flat, 0.9).name, "Gemma-2-9B + Llama-3.1-8B")
        self.assertEqual(select(pairs, entries, {**flat, "Qwen2.5-7B": {"auroc": 0.8}}, 0.9).name, "Gemma-2-9B + Qwen2.5-7B")
        entries["Llama-3.1-8B + Qwen2.5-7B"]["0.90"]["accepted_wrong"] = 9
        self.assertEqual(select(pairs, entries, flat, 0.9).name, "Llama-3.1-8B + Qwen2.5-7B")

    def test_check_recomputes_the_manifest(self):
        self.freeze()
        self.assertEqual(check_manifest(self.root / MANIFEST, self.root, FAST), [])
        m = json.loads((self.root / MANIFEST).read_text())
        m["validation"]["thresholds"]["math"]["Gemma-2-9B + Qwen2.5-7B"]["0.90"]["a"] += 0.125
        (self.root / MANIFEST).write_bytes(manifest_bytes(m))
        self.assertEqual(len(check_manifest(self.root / MANIFEST, self.root, FAST)), 1)

    def test_other_rules_do_not_match_the_manifest(self):
        m = self.validate()
        self.assertIn("rules: the fixed numbers differ from the manifest", hash_problems(m, self.root, RULES))

    def test_llada_recheck_is_reported_and_not_applied(self):
        m = self.validate()
        weak = {**SEPARATION, "LLaDA-8B": -0.5}
        write_study(self.root, seed=3, problems={"math": 60, "code": 30}, separation=weak)
        n = self.validate()
        self.assertFalse(m["validation"]["h1"]["llada_recheck"]["applied"])
        self.assertTrue(n["validation"]["h1"]["llada_recheck"]["below_bar"])
        self.assertEqual(m["validation"]["h1"]["candidates"], n["validation"]["h1"]["candidates"])

    def test_cost_check(self):
        m = self.validate()["validation"]["h1"]["cost"]
        self.assertTrue(m["within_limit"] and m["comparable_cost_claim"])
        write_study(self.root, seed=3, problems={"math": 60, "code": 30}, gpus={("math", "LLaDA-8B"): "NVIDIA L4"})
        m = self.validate()["validation"]["h1"]["cost"]
        self.assertFalse(m["same_gpu_type"] or m["within_limit"] or m["comparable_cost_claim"])     # never compared directly
        self.assertTrue(np.isnan(m["ratio"]) and m["cheaper"] is None)
        self.assertFalse(m["budget_matched_secondary"])
        write_study(self.root, seed=3, problems={"math": 60, "code": 30}, gpus={("code", "LLaDA-8B"): "NVIDIA L4"})
        self.assertTrue(self.validate()["validation"]["h1"]["cost"]["comparable_cost_claim"])       # code's GPU is not math's

    def test_stopped_mode_analyzes_nothing(self):
        stopped = parse_settings({**GO, "mode": "stopped", "llada_score": None, "reason": "second audit failed"})
        with self.assertRaises(DataError):
            validation_step({}, None, stopped, FAST)


class TestStepTests(Repo):
    def test_outputs(self):
        manifest = self.freeze()
        r = self.test_run()
        h1, main = r["h1"], r["h1"]["by_tar"]["0.90"]
        # the decision comes from these numbers and no others
        self.assertEqual(h1["verdict"], decide(h1["matching"]["passes"], main["delta_far"]["points"], main["delta_far"]["ci95"],
                                               main["delta_far"]["ci90"]))
        old = h1["matching_old_rule"]
        self.assertEqual(old["limits_points"], {"off_target": 3.0, "gap": 3.0})
        self.assertEqual(old["row"], decide(old["passes"], main["delta_far"]["points"], main["delta_far"]["ci95"],
                                            main["delta_far"]["ci90"]))
        # the thresholds are the frozen ones, and the counts can be redone by hand
        label = {c: y for c, y in self.truth["labels"]["math"].items()}
        rows = [x for x in self.truth["rows"]["math"] if x["split"] == "test"]
        score = {}
        for path in list((self.root / "results/full/math/ar").rglob("scores_*.jsonl")) + [self.root / "results/full/math/llada/full_T1_n1.jsonl"]:
            for s in read_jsonl(path):
                if s["template"] == "T1":
                    score[(s.get("model_id") or s["model"], s["candidate_id"])] = s["score"]
        ids = {"Qwen2.5-7B": "Qwen/Qwen2.5-7B-Instruct", "Llama-3.1-8B": "meta-llama/Llama-3.1-8B-Instruct",
               "Gemma-2-9B": "google/gemma-2-9b-it", "LLaDA-8B": "llada"}
        for group, name in manifest["validation"]["h1"]["selected"].items():
            a, b = (ids[v] for v in name.split(" + "))
            e = manifest["validation"]["thresholds"]["math"][name]["0.90"]
            ok = [score[(a, x["candidate_id"])] >= e["a"] and score[(b, x["candidate_id"])] >= e["b"] for x in rows]
            y = [label[x["candidate_id"]] for x in rows]
            self.assertEqual(main["rows"][group]["accepted_wrong"], sum(o for o, l in zip(ok, y) if l == 0))
            self.assertEqual(main["rows"][group]["accepted_correct"], sum(o for o, l in zip(ok, y) if l == 1))
            self.assertEqual(main["rows"][group]["n_wrong"], manifest["weights"]["math"])
        x = main["rows"]
        self.assertAlmostEqual(main["delta_far"]["points"], 100 * (x["ar_diffusion"]["far"] - x["ar_ar"]["far"]))
        # everything the plan lists is there
        self.assertEqual([t["row"] for t in r["reference_rows"]].count("same-model pseudo-pair (T1 + T2)"), 4)
        self.assertTrue(h1["ideal_thresholds_test_tuned"]["test_tuned"])
        self.assertEqual(sorted(h1["by_tar"]), ["0.80", "0.90", "0.95"])
        self.assertEqual({b["grouping"] for b in r["breakdowns"] if b["domain"] == "math"}, {"all", "source", "generator", "error_type"})
        self.assertEqual({b["group"] for b in r["breakdowns"] if b["grouping"] == "error_type" and b["domain"] == "math"},
                         {"wrong method", "arithmetic slip"})
        self.assertEqual(r["h2"]["math"]["trend"]["n_ar_pairs"], 13)
        by_type = r["h2"]["code"]["by_error_type"]["wrong_output"]
        self.assertEqual(by_type["n_wrong"], manifest["weights"]["code"])       # every wrong code answer has this type here
        self.assertEqual(by_type["trend"] is None, manifest["weights"]["code"] < RULES.small_group)
        self.assertEqual(set(r["h2"]["math"]["by_error_type"]), {"wrong method", "arithmetic slip"})
        self.assertEqual({d for d in r["e1"]}, {"math", "code"})
        self.assertTrue(any(p["kind"] == "difference" for p in r["pooled"]))
        self.assertEqual({s["domain"] for s in r["singles"]}, {"math", "code", "pooled"})
        self.assertNotIn("labels", json.dumps(r["h1"]))
        # and the report can be built from the file alone
        saved = json.loads((self.root / "results/final/test/results.json").read_text())
        written = write_report(saved, self.root / "report")
        self.assertTrue({"summary.md", "pairs.csv", "h1_delta_far.png", "tradeoff.png", "cost_math.png"} <= set(written))
        self.assertIn("Final test", (self.root / "report/summary.md").read_text())

    def test_pooling_weights_are_the_wrong_test_answers(self):
        manifest = self.freeze()
        r = self.test_run()
        w = manifest["weights"]
        for p in r["pooled"]:
            if p["kind"] == "difference":
                continue
            far = {x["domain"]: x["far"] for x in r["pairs"] if x["pair"] == p["pair"] and x["target_tar"] == 0.9}
            self.assertAlmostEqual(p["far"], (w["math"] * far["math"] + w["code"] * far["code"]) / (w["math"] + w["code"]))

    def test_labels_that_break_the_weights_are_refused(self):
        self.freeze()
        settings = parse_settings(GO)
        _, scores, frames, _ = load_inputs(self.root, settings, "bank/full", SCORES, ("math", "code"), FAST)
        with self.assertRaises(DataError):
            evaluate_step(frames, frames, {}, settings, scores, {"math": 1, "code": 1}, FAST)

    def test_command_line(self):
        self.freeze()
        with mock.patch.object(final_evaluate, "run", side_effect=lambda *a, **k: run(*a, **{**k, "rules": FAST})), \
                contextlib.redirect_stdout(io.StringIO()) as printed:
            code = final_evaluate.main(["--root", str(self.root), "--test-labels", *(f"{d}={p}" for d, p in LABELS.items())])
        self.assertEqual(code, 0, printed.getvalue())
        self.assertTrue((self.root / "results/final/test/summary.md").is_file())
        with contextlib.redirect_stdout(io.StringIO()) as printed:
            code = final_evaluate.main(["--root", str(self.root), "--manifest", "nothing.json", "--test-labels", "math=x"])
        self.assertEqual(code, 2)


class ExpensivePairTests(Repo):
    """Above 1.5x the cheaper pair's T1/T2 average is reported as secondary, and H1 does not change."""

    def test_budget_matched_comparison(self):
        path = self.root / "results/full/math/llada"
        for column in ("T1", "T2"):
            file = path / f"full_{column}_n1.jsonl"
            write_jsonl(file, [{**r, "runtime_s": 1.0} for r in read_jsonl(file)])
        manifest = self.freeze()
        cost = manifest["validation"]["h1"]["cost"]
        self.assertTrue(cost["budget_matched_secondary"] and not cost["comparable_cost_claim"])
        self.assertEqual(cost["cheaper"], "ar_ar")
        r = self.test_run()
        b = r["h1"]["budget_matched"]
        self.assertIn("(T1/T2 mean)", b["pairs"]["ar_ar"])
        self.assertEqual(b["pairs"]["ar_diffusion"], r["h1"]["pairs"]["ar_diffusion"])
        self.assertIn("no comparable-cost claim", " ".join(r["h1"]["reading"]["caveats"]))


class RepairTests(Repo):
    settings = {**GO, "mode": "repair", "llada_score": "reconstruction", "reason": "LLaDA fails test 4"}
    study = {"with_recon": True}

    def test_repair(self):
        manifest = self.freeze()
        h1 = manifest["validation"]["h1"]
        self.assertEqual(h1["test_name"], "AR verdict versus diffusion reconstruction score")
        self.assertTrue(all("(reconstruction)" in n for n in h1["candidates"]["ar_diffusion"]))
        self.assertFalse(h1["cost"]["comparable_cost_claim"])
        self.assertEqual(h1["llada_recheck"]["score"], "recon")
        r = self.test_run()
        self.assertEqual(r["h1"]["test_name"], "AR verdict versus diffusion reconstruction score")
        self.assertIn("reconstruction score", r["h1"]["reading"]["caveats"][0])
        self.assertTrue(any(p["pair"].endswith("Qwen2.5-7B") and "LLaDA-8B +" in p["pair"] for p in r["pairs"]))   # verdict row, background
        self.assertEqual({b["domain"] for b in r["breakdowns"]}, {"math", "code"})      # code: the verdict-score pair
        self.assertIn("no pooled difference", r["pooled_note"])
        for p in r["pooled"]:       # the verdict-score pairs are in no final test here
            self.assertFalse("h1_mixed" in p["roles"] and "LLaDA" in p["pair"], p["pair"])
        write_report(r, self.root / "report")


class PivotTests(Repo):
    settings = {**GO, "mode": "pivot", "llada_score": None, "reason": "reconstruction score failed",
                "dropped": {"LLaDA-8B": "fails test 4 in math"}}

    def test_pivot_reports_the_ar_family_study_and_no_h1(self):
        manifest = self.freeze()
        self.assertIsNone(manifest["validation"]["h1"])
        r = self.test_run()
        self.assertIsNone(r["h1"])
        for d in ("math", "code"):
            f = r["ar_family"][d]
            self.assertEqual(len(f["pairs"]), 6)
            self.assertEqual(sum(p["selected"] for p in f["pairs"]), 1)
            self.assertEqual(f["perfect_verifier_far"], 0.0)
        write_report(r, self.root / "report")
        self.assertIn("H1 is not run", (self.root / "report/summary.md").read_text())


class TwoArTests(Repo):
    settings = {**GO, "h1_ar": ["Llama-3.1-8B", "Qwen2.5-7B"], "dropped": {"Gemma-2-9B": "fails test 4 in math"}, "qwen3_scored": True}

    def test_pool_of_two(self):
        manifest = self.freeze()
        c = manifest["validation"]["h1"]["candidates"]
        self.assertEqual(sorted(c["ar_ar"]), ["Llama-3.1-8B + Qwen2.5-7B", "Llama-3.1-8B + Qwen3-8B"])
        self.assertEqual(sorted(c["ar_diffusion"]), ["LLaDA-8B + Llama-3.1-8B", "LLaDA-8B + Qwen2.5-7B"])
        r = self.test_run()
        self.assertIn("strong-baseline check, AR+diffusion", [t["row"] for t in r["reference_rows"]])
        self.assertEqual(r["h2"]["math"]["trend"]["n_ar_pairs"], 13 + 4)    # Qwen3 with the four models of other families


class SideExperimentTests(Repo):
    """The reconstruction side experiment: frozen with the manifest, applied after the join, never in the way of H1.
    analysis.reconstruction belongs to another package, so it is a stand-in here."""

    def setUp(self):
        super().setUp()
        bank = read_jsonl(self.root / "bank/full/math/bank.jsonl")
        self.drawn = {s: [r["candidate_id"] for r in bank if r["split"] == s][:20] for s in ("validation", "test")}
        self.calls = calls = []
        stub = types.ModuleType("analysis.reconstruction")
        stub.run_label = lambda key: key

        def load_runs(paths, bank_hash):
            rows = [r for p in paths for r in read_jsonl(p) if r["bank_hash"] == bank_hash]
            return {"LLaDA-8B": {"settings": {"seed": 544}, "rows": {r["candidate_id"]: r["score"] for r in rows}}} if rows else {}

        def validation_stage(rows, bank_hash, subset, label_of, runs, n_boot):
            t = min(runs["LLaDA-8B"]["rows"][c] for c in subset["validation"] if label_of[c] == 1)
            return {"runs": {"LLaDA-8B": {"settings": {"seed": 544}, "points": {"0.90": {"threshold": t}}}}}

        def test_stage(frozen, rows, bank_hash, label_of, runs, n_boot):
            calls.append((frozen["bank_hash"] == bank_hash, set(self.drawn["test"]) <= set(label_of), n_boot))
            if self.broken:
                raise SystemExit("no test scores")
            return {"bank_hash": bank_hash, "runs": {"LLaDA-8B": {"auroc": 0.6, "auroc_lo": 0.5, "auroc_hi": 0.7,
                                                                  "points": {"0.90": {"tar": 0.9, "far": 0.8}}}}}
        stub.load_runs, stub.validation_stage, stub.test_stage, self.broken = load_runs, validation_stage, test_stage, False
        stub.validation_labels = lambda path, rows: {r["candidate_id"]: r["label"] for r in read_jsonl(path)}
        stub.aligned = lambda run, rows: np.array([run["rows"].get(r["candidate_id"], np.nan) for r in rows], dtype=float)
        patch = mock.patch.dict(sys.modules, {"analysis.reconstruction": stub})
        patch.start()
        self.addCleanup(patch.stop)
        for name in ("reconstruction.py", "pilot.py"):
            (self.root / "analysis" / name).write_text("# stand-in\n", encoding="utf-8")

        def sha(path):
            return hashlib.sha256((self.root / path).read_bytes()).hexdigest()
        rng = np.random.default_rng(0)
        self.side_rows = [{"method": "diffusion_mc", "bank_hash": sha("bank/full/math/bank.jsonl"), "candidate_id": c,
                           "score": float(rng.normal())} for ids in self.drawn.values() for c in ids]
        write_jsonl(self.root / "results/reconstruction/side/llada.jsonl", self.side_rows)
        score = {r["candidate_id"]: r["score"] for r in self.side_rows}
        self.threshold = min(score[c] for c in self.drawn["validation"] if self.truth["labels"]["math"][c] == 1)
        self.frozen = {"domain": "math", "bank_hash": sha("bank/full/math/bank.jsonl"), "subset": self.drawn,
                       "labels_sha256": sha("bank/full/math/labels_validation.jsonl"),
                       "runs": {"LLaDA-8B": {"settings": {"seed": 544}, "points": {"0.90": {"threshold": self.threshold}}}}}

    def freeze_with_side(self, **changes):
        side = self.root / "results/reconstruction/side.json"
        side.write_text(json.dumps({**self.frozen, **changes}), encoding="utf-8")
        m = build_manifest(self.root, "configs/final_models.json", "bank/full", SCORES, "results/final", rules=FAST,
                           side="results/reconstruction/side.json", side_scores=["results/reconstruction/side"])
        (self.root / MANIFEST).parent.mkdir(parents=True, exist_ok=True)
        (self.root / MANIFEST).write_bytes(manifest_bytes(m))
        self.commit()
        return m

    def test_applied_after_the_join(self):
        m = self.freeze_with_side()
        self.assertEqual(m["side"]["thresholds"], {"LLaDA-8B": {"0.90": self.threshold}})
        r = self.test_run()
        self.assertEqual(self.calls, [(True, True, 200)])
        self.assertIn("LLaDA-8B", r["reconstruction_side"]["runs"])
        write_report(r, self.root / "report")
        self.assertIn("side experiment", (self.root / "report/summary.md").read_text())

    def test_its_failure_never_takes_the_main_result_with_it(self):
        self.freeze_with_side()
        self.broken = True
        r = self.test_run()
        self.assertIn("no test scores", r["reconstruction_side"]["error"])
        self.assertIsNotNone(r["h1"]["verdict"])
        write_report(r, self.root / "report")

    def test_frozen_by_hash(self):
        self.freeze_with_side()
        side = self.root / "results/reconstruction/side.json"
        side.write_text(side.read_text() + " ", encoding="utf-8")
        with self.assertRaises(DataError) as caught:
            self.test_run()
        self.assertIn("side.json", str(caught.exception))
        self.assertEqual(self.log(), [])

    def test_its_code_is_frozen(self):
        self.freeze_with_side()
        for name in ("reconstruction.py", "pilot.py"):
            (self.root / "analysis" / name).write_text("# edited\n", encoding="utf-8")
            self.commit()
            with self.assertRaises(DataError) as caught:
                self.test_run()
            self.assertIn(name, str(caught.exception))
        self.assertEqual(self.log(), [])

    def test_thresholds_of_another_bank(self):
        with self.assertRaises(DataError):
            self.freeze_with_side(bank_hash="0" * 64)

    def test_anything_that_would_fail_after_the_join_is_refused_at_the_freeze(self):
        scores = self.root / "results/reconstruction/side/llada.jsonl"
        write_jsonl(scores, self.side_rows[:-1])                            # a drawn test answer without a score
        self.refused_side("no usable score")
        write_jsonl(scores, [{**r, "bank_hash": "0" * 64} for r in self.side_rows])
        self.refused_side("LLaDA-8B: no scores")
        write_jsonl(scores, self.side_rows)
        self.refused_side("do not recompute", runs={"LLaDA-8B": {"settings": {"seed": 544}, "points": {"0.90": {"threshold": 5.0}}}})
        self.refused_side("other validation labels", labels_sha256="0" * 64)
        self.refused_side("not all test answers", subset={**self.drawn, "test": self.drawn["validation"]})

    def refused_side(self, fragment, **changes):
        with self.assertRaises(DataError) as caught:
            self.freeze_with_side(**changes)
        self.assertIn(fragment, str(caught.exception))

    def test_math_only(self):
        with self.assertRaises(DataError) as caught:
            self.freeze_with_side(domain="code")
        self.assertIn("on math only", str(caught.exception))

    def test_no_code_domain_is_needed(self):
        """The side experiment and H1 both run on a math-only study."""
        self.frozen["domain"] = "math"
        side = self.root / "results/reconstruction/side.json"
        side.write_text(json.dumps(self.frozen), encoding="utf-8")
        m = build_manifest(self.root, "configs/final_models.json", "bank/full", SCORES[:1], "results/final", ("math",), FAST,
                           "results/reconstruction/side.json", ["results/reconstruction/side"])
        (self.root / MANIFEST).parent.mkdir(parents=True, exist_ok=True)
        (self.root / MANIFEST).write_bytes(manifest_bytes(m))
        self.commit()
        r = self.test_run(labels={"math": LABELS["math"]})
        self.assertIn("LLaDA-8B", r["reconstruction_side"]["runs"])
        self.assertEqual((r["pooled"], list(r["h2"])), ([], ["math"]))
        write_report(r, self.root / "report")


class ReadingTests(unittest.TestCase):
    def read(self, verdict, status, below=False, beats=True, cost=True, repair=False):
        return reading(verdict, status, below, beats, cost, repair)

    def test_every_row_of_plan_section_8(self):
        self.assertIn("useful verification diversity", self.read("better_meaningful", BELOW)["claim"])
        self.assertIn("mechanism no", self.read("better_meaningful", ON)["claim"])
        self.assertIn("without competence", self.read("no_meaningful_benefit", BELOW)["claim"])
        self.assertIn("Clean negative", self.read("no_meaningful_benefit", ON)["claim"])
        for status in (BELOW, ON, INCONCLUSIVE):
            self.assertIn("vetoes", self.read("worse", status)["claim"])
            self.assertIn("small benefit", self.read("better_small", status)["claim"])
            self.assertIn("interval only", self.read("inconclusive", status)["claim"])
            self.assertIn("trade-off", self.read("not_matched", status)["claim"])

    def test_above_is_read_as_on(self):
        self.assertEqual(self.read("better_meaningful", ABOVE)["claim"], self.read("better_meaningful", ON)["claim"])

    def test_no_mechanism_claim(self):
        for kwargs in ({"below": True}, {}):
            status = BELOW if kwargs else INCONCLUSIVE
            for verdict in ("better_meaningful", "no_meaningful_benefit"):
                r = self.read(verdict, status, **kwargs)
                self.assertFalse(r["mechanism_claim"])
                self.assertIn("no mechanism claim", r["claim"])

    def test_diversity_needs_to_beat_the_pseudo_pair(self):
        for beats in (False, None):
            r = self.read("better_meaningful", BELOW, beats=beats)
            self.assertFalse(r["mechanism_claim"])
            self.assertIn("pseudo-pair", " ".join(r["caveats"]))

    def test_caveats(self):
        self.assertIn("reconstruction", self.read("worse", ON, repair=True, cost=False)["caveats"][0])
        self.assertIn("comparable-cost", self.read("worse", ON, cost=False)["caveats"][0])
        self.assertEqual(self.read("worse", ON)["caveats"], [])


class PilotSmokeTests(unittest.TestCase):
    """The real pilot files through the same code. In-sample, exploratory: never an H1 result."""

    def test_pilot(self):
        from analysis.final_pilot import STATUS, run_pilot
        validation, results = run_pilot(rules=FAST)
        self.assertIn("not an H1 result", STATUS)
        self.assertEqual(results["status"], STATUS)
        far = {(p["domain"], p["pair"]): p["far"] for p in results["pairs"] if p["target_tar"] == 0.9}
        # the same in-sample FARs the pilot analysis reported (results/pilot/analysis/pairs.csv)
        self.assertAlmostEqual(far[("math", "Gemma-2-9B + Qwen2.5-7B")], 0.696)
        self.assertAlmostEqual(far[("math", "LLaDA-8B + Qwen2.5-7B")], 0.736)
        self.assertAlmostEqual(far[("code", "Llama-3.1-8B + Qwen2.5-7B")], 0.632)
        auc = {s["label"]: s["auroc"] for s in results["singles"] if s["domain"] == "math"}
        self.assertAlmostEqual(auc["LLaDA-8B"], 0.607, places=3)
        self.assertEqual(validation["h1"]["best_single"], "Qwen2.5-7B")     # lowest FAR, not highest AUROC
        with tempfile.TemporaryDirectory() as tmp:
            write_report(results, Path(tmp))
            text = (Path(tmp) / "summary.md").read_text(encoding="utf-8")
        self.assertIn("They are not H1 results", text)
        self.assertNotIn("**Verdict:**", text)


if __name__ == "__main__":
    unittest.main()
