"""Backup-score analysis on synthetic banks with known answers (PLAN.md §4, §7). CPU only."""
import contextlib
import io
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from analysis import reconstruction as R
from analysis.metrics import threshold_at_tar
from harness.diffusion import reconstruction as harness_side
from harness.diffusion.run_reconstruction import ROW_FIELDS, SETTINGS, backup_row, gpu_type
from harness.diffusion.scorer import SPECS

QWEN = "Qwen/Qwen2.5-7B-Instruct"


def make_bank(n_pairs=40, split="pilot", domain="math", start=0):
    """Two answers per pair, one pair per problem; the correct one first or second by turns."""
    bank, labels = [], {}
    for i in range(start, start + n_pairs):
        for j in range(2):
            cid = f"{domain}{split}{i}_{j}"
            bank.append({"candidate_id": cid, "pair_id": f"{domain}{split}{i}", "problem_id": f"{domain}{split}p{i}",
                         "domain": domain, "split": split, "problem": "p", "candidate": "c"})
            labels[cid] = int(j == i % 2)
    return bank, labels


def make_run(bank, scores, bank_hash="h", model="llada", method="diffusion_mc", n_samples=128, seed=544,
             statuses=None, gpu="NVIDIA A100-SXM4-40GB", **changes):
    settings = {"bank_hash": bank_hash, "model": model, "revision": R.REPAIR_REVISION, "method": method, "n_samples": n_samples,
                "seed": seed, "mc_group": None if n_samples is None else 16, "micro_batch": 1, "attention": None,
                **changes}
    rows = []
    for b, s in zip(bank, scores):
        status = (statuses or {}).get(b["candidate_id"], "ok")
        values = dict(loglik=float(s) * 10, score=float(s) * 10 / 10, n_tokens=10, boundary="clean", runtime_s=1.0,
                      gpu=gpu) if status == "ok" else {}
        rows.append(backup_row(settings, b["candidate_id"], status, **values))
    return rows


def write_rows(path, rows):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")


def labels_of(bank, labels):
    return np.array([labels[b["candidate_id"]] for b in bank], dtype=float)


def runs_from(*row_lists):
    with tempfile.TemporaryDirectory() as folder:
        for i, rows in enumerate(row_lists):
            write_rows(Path(folder) / f"f{i}.jsonl", rows)
        return R.load_runs([folder], "h")


class PilotTests(unittest.TestCase):
    def setUp(self):
        self.bank, self.labels = make_bank()
        self.y = labels_of(self.bank, self.labels)
        rng = np.random.default_rng(0)
        self.good = self.y + rng.normal(0, 0.8, len(self.y))     # clearly above chance
        self.chance = rng.normal(0, 1, len(self.y))
        self.small = {(R.SMALL_REFERENCE, "T1"): {"score": self.y + rng.normal(0, 2, len(self.y))}}

    def report(self, *rows, verdicts=None):
        return R.pilot_report(self.bank, self.y, runs_from(*rows), self.small if verdicts is None else verdicts, n_boot=300)

    def test_a_good_stable_score_passes(self):
        noisy = self.good + np.random.default_rng(1).normal(0, 0.01, len(self.y))
        r = self.report(make_run(self.bank, self.good), make_run(self.bank, noisy, seed=545))["repair"]
        self.assertEqual(r["verdict"], "pass")
        self.assertTrue(r["gate_a"]["passes"] and r["repeat_check"]["passes"])
        self.assertGreater(r["repeat_check"]["spearman"], 0.95)

    def test_chance_fails_gate_a(self):
        r = self.report(make_run(self.bank, self.chance), make_run(self.bank, self.chance, seed=545))["repair"]
        self.assertEqual((r["verdict"], r["gate_a"]["passes"], r["repeat_check"]["passes"]), ("fail", False, True))

    def test_an_unstable_estimate_fails_the_repeat_check(self):
        other = self.good + np.random.default_rng(2).normal(0, 2.0, len(self.y))
        r = self.report(make_run(self.bank, self.good), make_run(self.bank, other, seed=545))["repair"]
        self.assertEqual((r["verdict"], r["gate_a"]["passes"], r["repeat_check"]["passes"]), ("fail", True, False))

    def test_the_gap_to_the_small_verifier_is_one_sided(self):
        entry = {"n": 10, "n_ok": 10, "auroc": 0.70, "auroc_lo": 0.6}
        self.assertTrue(R.gate_a(entry, 0.80 - 1e-9)["passes"])
        self.assertFalse(R.gate_a(entry, 0.81)["passes"])
        self.assertTrue(R.gate_a(entry, 0.40)["passes"])
        self.assertFalse(R.gate_a({**entry, "auroc": 0.599}, 0.40)["passes"])
        self.assertTrue(R.gate_a({**entry, "auroc": 0.60}, 0.40)["passes"])
        self.assertFalse(R.gate_a({**entry, "auroc_lo": 0.50}, 0.40)["passes"])

    def test_missing_pieces_are_not_finished_rather_than_a_verdict(self):
        both = make_run(self.bank, self.good), make_run(self.bank, self.good, seed=545)
        self.assertEqual(self.report(both[0])["repair"]["verdict"], "not finished")            # no seed 545
        self.assertEqual(self.report(both[1])["repair"]["verdict"], "not finished")            # no seed 544
        self.assertEqual(self.report(*both, verdicts={})["repair"]["verdict"], "not finished")  # no small reference
        self.assertEqual(self.report(both[0][:-1], both[1])["repair"]["verdict"], "not finished")
        self.assertEqual(self.report(both[0], both[1][:-1])["repair"]["verdict"], "not finished")
        bad = make_run(self.bank, self.good, statuses={self.bank[0]["candidate_id"]: "error"})
        self.assertEqual(self.report(bad, both[1])["repair"]["verdict"], "not finished")
        half = {(R.SMALL_REFERENCE, "T1"): {"score": np.where(np.arange(len(self.y)) < 5, np.nan, self.chance)}}
        self.assertEqual(self.report(*both, verdicts=half)["repair"]["verdict"], "not finished")

    def test_only_128_samples_count_for_the_repair(self):
        r = self.report(make_run(self.bank, self.good, n_samples=32), make_run(self.bank, self.good, n_samples=32, seed=545))
        self.assertEqual(r["repair"]["verdict"], "not finished")
        self.assertTrue(all(m["role"] == "information only" for m in r["models"]))

    def test_the_repair_needs_the_plans_settings(self):
        for change, gpus in (({"micro_batch": 16}, ()), ({"revision": "x"}, ()), ({"mc_group": 8}, ()),
                             ({}, ("NVIDIA A100-SXM4-40GB", "NVIDIA L4"))):
            first, second = gpus or ("NVIDIA A100-SXM4-40GB",) * 2
            r = self.report(make_run(self.bank, self.good, gpu=first, **change),
                            make_run(self.bank, self.good, seed=545, gpu=second, **change))["repair"]
            self.assertEqual((r["verdict"], len(r["problems"]) > 0), ("not finished", True), change)
        # 40 GB and 80 GB A100s are one GPU type
        r = self.report(make_run(self.bank, self.good),
                        make_run(self.bank, self.good, seed=545, gpu="NVIDIA A100-SXM4-80GB"))["repair"]
        self.assertEqual((r["verdict"], r["problems"]), ("pass", []))
        self.assertEqual(R.gpu_type("NVIDIA L4"), gpu_type("NVIDIA L4"))

    def test_corrected_labels_drop_broken_pairs_and_keep_the_scores(self):
        corrected = dict(self.labels)
        corrected[self.bank[0]["candidate_id"]] = corrected[self.bank[1]["candidate_id"]]  # two equal labels
        del corrected[self.bank[4]["candidate_id"]]                                        # no label at all
        kept, y, dropped = R.intact_pairs(self.bank, corrected)
        self.assertEqual(dropped, sorted({self.bank[0]["pair_id"], self.bank[4]["pair_id"]}))
        self.assertEqual((len(kept), len(y)), (len(self.bank) - 4, len(self.bank) - 4))
        self.assertEqual(int(y.sum()) * 2, len(y))
        keep = [i for i, b in enumerate(self.bank) if b["pair_id"] not in dropped]
        small = {(R.SMALL_REFERENCE, "T1"): {"score": self.small[(R.SMALL_REFERENCE, "T1")]["score"][keep]}}
        runs = runs_from(make_run(self.bank, self.good), make_run(self.bank, self.good, seed=545))
        report = R.pilot_report(kept, y, runs, small, n_boot=100)
        self.assertEqual(report["models"][0]["n"], len(self.bank) - 4)
        self.assertEqual(report["repair"]["verdict"], "pass")
        self.assertEqual(R.intact_pairs(self.bank, self.labels)[2], [])

    def test_estimates_with_different_settings_are_not_compared(self):
        with self.assertRaises(SystemExit):
            self.report(make_run(self.bank, self.good), make_run(self.bank, self.good, seed=545, micro_batch=16))

    def test_dream_and_ar_are_information_only_and_code_never_decides(self):
        bank = self.bank + make_bank(domain="code")[0]
        labels = {**self.labels, **make_bank(domain="code")[1]}
        y = labels_of(bank, labels)
        verdicts = {(R.SMALL_REFERENCE, "T1"): {"score": np.zeros(len(y))}, ("Dream-7B", "T1"): {"score": y}}
        rng = np.random.default_rng(3)
        math_bad_code_good = np.where([b["domain"] == "code" for b in bank], y + rng.normal(0, 0.1, len(y)),
                                      rng.normal(0, 1, len(y)))
        runs = runs_from(make_run(bank, math_bad_code_good), make_run(bank, math_bad_code_good, seed=545),
                         make_run(bank, y + rng.normal(0, 0.1, len(y)), model="dream"),
                         make_run(bank, y + rng.normal(0, 0.1, len(y)), model=QWEN, method="ar_exact",
                                  n_samples=None, seed=None, mc_group=None, attention="sdpa"))
        report = R.pilot_report(bank, y, runs, verdicts, n_boot=100)
        self.assertEqual(report["repair"]["verdict"], "fail")
        self.assertTrue(report["gate_a"]["LLaDA-8B"]["code"]["passes"])
        roles = {(m["verifier"], m["domain"]): m["role"] for m in report["models"]}
        self.assertEqual({k for k, v in roles.items() if v == "repair"}, {("LLaDA-8B", "math")})
        self.assertIn(("Qwen2.5-7B", "code"), roles)
        self.assertNotIn("Qwen2.5-7B", report["gate_a"])
        dream = next(m for m in report["models"] if m["verifier"] == "Dream-7B" and m["domain"] == "math")
        self.assertEqual(dream["verdict_auroc_t1"], 1.0)
        text = R.pilot_markdown(report, "bank")
        self.assertIn("Outcome: fail", text)
        self.assertIn("ar_exact", text)


class LoadTests(unittest.TestCase):
    def setUp(self):
        self.bank, self.labels = make_bank(4)

    def test_other_banks_are_ignored_and_bad_rows_refused(self):
        x = np.arange(8.0)
        self.assertEqual(runs_from(make_run(self.bank, x, bank_hash="other")), {})
        with self.assertRaises(SystemExit):  # one answer scored twice
            runs_from(make_run(self.bank, x), make_run(self.bank[:1], x))
        with self.assertRaises(SystemExit):  # one run, two micro-batch sizes
            runs_from(make_run(self.bank[:4], x), make_run(self.bank[4:], x, micro_batch=16))
        rows = make_run(self.bank, x)
        rows[1]["score"] = 123.0  # not loglik / n_tokens
        with self.assertRaises(SystemExit):
            runs_from(rows)

    def test_constants_match_the_harness(self):
        self.assertEqual((R.N_SAMPLES, R.SEED, R.REPEAT_SEED),
                         (harness_side.N_SAMPLES, harness_side.SEED, harness_side.REPEAT_SEED))
        self.assertEqual(R.SETTINGS, SETTINGS)
        self.assertEqual((R.REPAIR_REVISION, R.MC_GROUP), (SPECS["llada"].revision, harness_side.MC_GROUP))
        self.assertTrue(set(R.SETTINGS) <= set(ROW_FIELDS))


class SampleCountTests(unittest.TestCase):
    def runs(self, noise_at):
        rng = np.random.default_rng(0)
        truth = rng.normal(0, 1, 60)
        bank, _ = make_bank(30, split="dev")
        out = {}
        for seed in (544, 545):
            rows = make_run(bank, truth, seed=seed)
            for r, t in zip(rows, truth):
                r["loglik_at"] = {str(n): 10 * (t + rng.normal(0, s)) for n, s in noise_at.items()}
            out[("LLaDA-8B", "diffusion_mc", 128, seed)] = {"settings": {}, "rows": {r["candidate_id"]: r for r in rows}}
        return out

    def test_smallest_count_that_reaches_095(self):
        runs = self.runs({8: 2.0, 16: 1.0, 32: 0.05, 64: 0.04, 128: 0.02})
        entry = R.samples_report(runs)["LLaDA-8B (up to 128)"]
        self.assertEqual(entry["chosen_n_samples"], 32)
        self.assertLess(entry["spearman"]["16"], 0.95)
        self.assertEqual(entry["items"], 60)

    def test_none_when_no_count_is_enough(self):
        entry = R.samples_report(self.runs({8: 3.0, 128: 3.0}))["LLaDA-8B (up to 128)"]
        self.assertIsNone(entry["chosen_n_samples"])

    def test_both_seeds_must_score_the_same_answers(self):
        runs = self.runs({8: 0.1})
        runs[("LLaDA-8B", "diffusion_mc", 128, 545)]["rows"].popitem()
        with self.assertRaises(SystemExit):
            R.samples_report(runs)


class SideExperimentTests(unittest.TestCase):
    def setUp(self):
        self.bank, self.labels = [], {}
        for split, n in (("validation", 40), ("test", 60)):
            bank, labels = make_bank(n, split=split)
            # every third problem has a second pair, so problems hold two or four answers
            for i in range(0, n, 3):
                for j in range(2):
                    cid = f"extra{split}{i}_{j}"
                    bank.append({**bank[0], "candidate_id": cid, "pair_id": f"extra{split}{i}",
                                 "problem_id": f"math{split}p{i}"})
                    labels[cid] = j
            self.bank += bank
            self.labels.update(labels)
        self.y = labels_of(self.bank, self.labels)
        self.x = self.y + np.random.default_rng(0).normal(0, 1, len(self.y))
        self.splits = R.draw_subset(self.bank, n=50)

    def test_draw_takes_whole_problems_once_and_is_seeded(self):
        problem_of = {b["candidate_id"]: b["problem_id"] for b in self.bank}
        split_of = {b["candidate_id"]: b["split"] for b in self.bank}
        for split, ids in self.splits.items():
            self.assertEqual(len(ids), 50)
            self.assertEqual({split_of[c] for c in ids}, {split})
            chosen = {problem_of[c] for c in ids}
            whole = [c for c in problem_of if problem_of[c] in chosen]
            self.assertEqual(sorted(whole), ids)
        self.assertEqual(R.draw_subset(self.bank[::-1], n=50), self.splits)  # order of the file does not matter
        self.assertNotEqual(R.draw_subset(self.bank, n=50, seed=545), self.splits)
        with self.assertRaisesRegex(SystemExit, "one domain"):
            R.draw_subset(self.bank + make_bank(2, split="test", domain="code")[0])
        everything = R.draw_subset(self.bank, n=10_000)
        self.assertEqual(sum(map(len, everything.values())), len(self.bank))

    def val_labels(self):
        ids = {b["candidate_id"] for b in self.bank if b["split"] == "validation"}
        return {c: l for c, l in self.labels.items() if c in ids}

    def test_thresholds_come_from_validation_and_are_applied_unchanged(self):
        runs = runs_from(make_run(self.bank, self.x))
        frozen = R.validation_stage(self.bank, "h", self.splits, self.val_labels(), runs, n_boot=50)
        index = {b["candidate_id"]: i for i, b in enumerate(self.bank)}
        val = [index[c] for c in self.splits["validation"]]
        test = [index[c] for c in self.splits["test"]]
        entry = frozen["runs"]["LLaDA-8B diffusion_mc n128 s544"]
        t = threshold_at_tar(self.x[val][self.y[val] == 1], 0.9)
        self.assertAlmostEqual(entry["points"]["0.90"]["threshold"], t)
        self.assertGreaterEqual(entry["points"]["0.90"]["tar"], 0.9)
        before = json.dumps(frozen, sort_keys=True)
        out = R.test_stage(frozen, self.bank, "h", self.labels, runs, n_boot=50)["runs"]["LLaDA-8B diffusion_mc n128 s544"]
        self.assertEqual(json.dumps(frozen, sort_keys=True), before)
        self.assertEqual(out["points"]["0.90"]["threshold"], entry["points"]["0.90"]["threshold"])
        self.assertAlmostEqual(out["points"]["0.90"]["far"], float(np.mean(self.x[test][self.y[test] == 0] >= t)))
        self.assertAlmostEqual(out["points"]["0.90"]["tar"], float(np.mean(self.x[test][self.y[test] == 1] >= t)))
        self.assertEqual(out["n_correct"] + out["n_wrong"], 50)

    def test_validation_stage_never_needs_test_labels(self):
        runs = runs_from(make_run(self.bank, self.x))
        a = R.validation_stage(self.bank, "h", self.splits, self.val_labels(), runs, n_boot=20)
        flipped = {c: (l if c in self.val_labels() else 1 - l) for c, l in self.labels.items()}
        b = R.validation_stage(self.bank, "h", self.splits, flipped, runs, n_boot=20)
        self.assertEqual(json.dumps(a), json.dumps(b))

    def test_test_label_files_are_refused(self):
        with tempfile.TemporaryDirectory() as folder:
            rows = [{"candidate_id": c, "label": l} for c, l in self.labels.items()]
            named = Path(folder) / "labels_test.jsonl"
            with self.assertRaisesRegex(SystemExit, "never read"):
                R.validation_labels(named, self.bank)  # refused by name: the file does not even exist
            mixed = Path(folder) / "labels_all.jsonl"
            write_rows(mixed, rows)
            with self.assertRaisesRegex(SystemExit, "not a validation answer"):
                R.validation_labels(mixed, self.bank)
            good = Path(folder) / "labels_validation.jsonl"
            write_rows(good, [r for r in rows if r["candidate_id"] in self.val_labels()])
            self.assertEqual(R.validation_labels(good, self.bank), self.val_labels())

    def test_incomplete_or_mismatched_inputs_stop_the_stages(self):
        runs = runs_from(make_run(self.bank, self.x))
        frozen = R.validation_stage(self.bank, "h", self.splits, self.val_labels(), runs, n_boot=20)
        with self.assertRaises(SystemExit):  # an unscored drawn answer
            gone = runs_from(make_run(self.bank, self.x, statuses={self.splits["validation"][0]: "error"}))
            R.validation_stage(self.bank, "h", self.splits, self.val_labels(), gone, n_boot=20)
        with self.assertRaises(SystemExit):
            R.test_stage(frozen, self.bank, "other", self.labels, runs, n_boot=20)
        with self.assertRaises(SystemExit):  # other scoring settings
            R.test_stage(frozen, self.bank, "h", self.labels, runs_from(make_run(self.bank, self.x, n_samples=32)), n_boot=20)
        with self.assertRaises(SystemExit):  # no labels given
            R.test_stage(frozen, self.bank, "h", {}, runs, n_boot=20)
        swapped = {**frozen, "subset": {"validation": self.splits["validation"], "test": self.splits["validation"]}}
        with self.assertRaises(SystemExit):
            R.test_stage(swapped, self.bank, "h", self.labels, runs, n_boot=20)


class CommandTests(unittest.TestCase):
    def run_command(self, *args):
        with contextlib.redirect_stdout(io.StringIO()) as out:
            R.main(list(map(str, args)))
        return out.getvalue()

    def test_pilot_command_end_to_end(self):
        with tempfile.TemporaryDirectory() as folder:
            folder = Path(folder)
            bank, labels = make_bank(20)
            write_rows(folder / "bank.jsonl", bank)
            write_rows(folder / "labels.jsonl", [{"candidate_id": c, "label": l} for c, l in labels.items()])
            h = R.sha256_of(folder / "bank.jsonl")
            y = labels_of(bank, labels)
            for seed in (544, 545):
                write_rows(folder / "scores" / f"llada_s{seed}.jsonl", make_run(bank, y, bank_hash=h, seed=seed))
            write_rows(folder / "scores" / "llada_s544.jsonl.manifest.jsonl", [{"not": "a row"}])
            write_rows(folder / "verdicts" / "scores_A.jsonl", [
                {"model_id": "meta-llama/Llama-3.2-3B-Instruct", "template": "T1", "candidate_id": b["candidate_id"],
                 "score": 0.1 * i, "status": "completed", "yes_no_mass": 1.0, "cold": False, "forward_seconds": 0.1}
                for i, b in enumerate(bank)])
            text = self.run_command("pilot", "--bank", folder / "bank.jsonl", "--labels", folder / "labels.jsonl",
                                    "--scores", folder / "scores", "--verdicts", folder / "verdicts",
                                    "--out", folder / "out", "--bootstrap", 50)
            self.assertIn("Outcome: pass", text)
            # the audit corrects one label: same bank file, same scores, one pair fewer
            write_rows(folder / "corrected.jsonl", [{"candidate_id": c, "label": 1 - l if c == bank[0]["candidate_id"] else l}
                                                    for c, l in labels.items()])
            again = self.run_command("pilot", "--bank", folder / "bank.jsonl", "--labels", folder / "corrected.jsonl",
                                     "--scores", folder / "scores", "--verdicts", folder / "verdicts",
                                     "--out", folder / "out2", "--bootstrap", 50)
            self.assertIn("Outcome: pass", again)
            self.assertIn("38 / 38", again)
            self.assertIn(bank[0]["pair_id"], again)
            report = json.loads((folder / "out" / "reconstruction.json").read_text())
            self.assertEqual((report["bank_hash"], report["repair"]["verdict"]), (h, "pass"))
            self.assertIn("Outcome: pass", (folder / "out" / "reconstruction.md").read_text(encoding="utf-8"))

    def test_pilot_command_refuses_any_other_bank(self):
        with tempfile.TemporaryDirectory() as folder:
            folder = Path(folder)
            bank, _ = make_bank(4, split="test")
            write_rows(folder / "bank.jsonl", bank)
            write_rows(folder / "scores.jsonl", make_run(bank, np.arange(8.0), bank_hash=R.sha256_of(folder / "bank.jsonl")))
            with self.assertRaisesRegex(SystemExit, "not the pilot bank"):
                self.run_command("pilot", "--bank", folder / "bank.jsonl", "--labels", folder / "missing.jsonl",
                                 "--scores", folder / "scores.jsonl", "--verdicts", folder, "--out", folder / "out")

    def test_draw_and_validation_commands(self):
        with tempfile.TemporaryDirectory() as folder:
            folder = Path(folder)
            val, val_labels = make_bank(20, split="validation")
            test, _ = make_bank(30, split="test")
            bank = val + test
            write_rows(folder / "bank.jsonl", bank)
            write_rows(folder / "labels_validation.jsonl", [{"candidate_id": c, "label": l} for c, l in val_labels.items()])
            self.run_command("draw", "--bank", folder / "bank.jsonl", "--out", folder / "subset.json", "--n", 30)
            subset = json.loads((folder / "subset.json").read_text())
            self.assertEqual(subset["counts"], {"validation": 30, "test": 30})
            drawn = [b for b in bank if b["candidate_id"] in sum(subset["splits"].values(), [])]
            x = np.random.default_rng(0).normal(size=len(drawn))
            write_rows(folder / "scores" / "s.jsonl", make_run(drawn, x, bank_hash=subset["bank_hash"]))
            self.run_command("validation", "--bank", folder / "bank.jsonl", "--subset", folder / "subset.json",
                             "--labels", folder / "labels_validation.jsonl", "--scores", folder / "scores",
                             "--out", folder / "side.json", "--bootstrap", 20)
            side = json.loads((folder / "side.json").read_text())
            self.assertEqual(side["subset"], subset["splits"])
            with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
                self.run_command("test", "--bank", folder / "bank.jsonl", "--out", folder / "x")  # no such command


if __name__ == "__main__":
    unittest.main()
