"""Loaders and label custody: everything that does not line up is an error. CPU only, synthetic files."""
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from analysis.final_data import (DataError, align_labels, gpu_type, load_bank, load_banks, load_scores, load_validation_labels,
                                 read_jsonl, recorded_hashes)
from analysis.final_settings import needed_scores, parse_settings
from analysis.final_synthetic import GO, sha, write_jsonl, write_study

SETTINGS = parse_settings(GO)
DOMAINS = ("math", "code")


class Study(unittest.TestCase):
    """A fresh synthetic study for every test."""

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        write_study(self.root, seed=1, problems={"math": 24, "code": 12})
        self.banks = load_banks(self.root / "bank/full")

    def load(self, paths=("results/full/math", "results/full/code"), **kwargs):
        return load_scores([self.root / p for p in paths], self.banks, {d: needed_scores(SETTINGS, d) for d in DOMAINS}, **kwargs)

    def edit(self, path, change):
        """Rewrite one JSONL file with change(rows)."""
        path = self.root / path
        rows = change(read_jsonl(path))
        write_jsonl(path, rows)
        run = path.parent / "manifest.json"
        beside = Path(str(path) + ".manifest.jsonl")
        if beside.is_file():                    # keep the run's own records right, so the test hits the check it is about
            write_jsonl(beside, [{**m, "completed": {"ok": sum(r.get("status") == "ok" for r in rows)}} for m in read_jsonl(beside)])
        if path.name.startswith("scores_"):
            import hashlib
            m = json.loads(run.read_text())
            m["outputs_sha256"][path.name] = hashlib.sha256(path.read_bytes()).hexdigest()
            run.write_text(json.dumps(m))

    def refused(self, fragment, call):
        with self.assertRaises(DataError) as caught:
            call()
        self.assertIn(fragment, str(caught.exception))


class ScoreTests(Study):
    LLADA = "results/full/math/llada/full_T1_n1.jsonl"
    QWEN = "results/full/math/ar/full_Qwen2.5-7B_Mistral-7B_T1/scores_A.jsonl"

    def test_complete_scores_load(self):
        scores = self.load()
        self.assertEqual(len(scores.columns), 2 * 8 * 2)
        self.assertEqual(scores.gpus("math", "LLaDA-8B", "T1"), {"NVIDIA A100-SXM4-40GB"})
        self.assertAlmostEqual(scores.seconds("math", "Qwen2.5-7B", "T1"), 0.05)     # the cold first row is left out
        self.assertAlmostEqual(scores.seconds("math", "Qwen2.5-7B", "avg"), 0.10)
        self.assertEqual(scores.info[("LLaDA-8B", "T1")]["n_slots"], {1})

    def test_row_order_does_not_matter(self):
        before = self.load().score("math", "LLaDA-8B", "T1")
        self.edit(self.LLADA, lambda rows: [rows[i] for i in np.random.default_rng(0).permutation(len(rows))])
        np.testing.assert_array_equal(before, self.load().score("math", "LLaDA-8B", "T1"))

    def test_missing_row(self):
        self.edit(self.LLADA, lambda rows: rows[:-1])
        self.refused("1 of", self.load)

    def test_missing_file(self):
        (self.root / self.LLADA).unlink()
        self.refused("LLaDA-8B T1", self.load)

    def test_failed_row(self):
        self.edit(self.LLADA, lambda rows: [{**rows[0], "status": "error", "score": None}, *rows[1:]])
        self.refused("status 'error'", self.load)
        self.edit(self.LLADA, lambda rows: [{**rows[0], "status": "ok", "score": float("inf")}, *rows[1:]])
        self.refused("score inf", self.load)

    def test_duplicate_row(self):
        self.edit(self.QWEN, lambda rows: rows + [rows[3]])
        self.refused("duplicate row", self.load)

    def test_unknown_candidate(self):
        self.edit(self.LLADA, lambda rows: rows + [{**rows[0], "candidate_id": "nobody"}])
        self.refused("not in the bank", self.load)

    def test_wrong_bank_hash(self):
        self.edit(self.LLADA, lambda rows: [{**r, "bank_hash": "0" * 64} for r in rows])
        self.refused("is not a bank of this analysis", self.load)

    def test_scores_of_the_other_domains_bank(self):
        self.edit(self.LLADA, lambda rows: [{**r, "bank_hash": self.banks["code"].sha256} for r in rows])
        self.refused("not in the bank", self.load)

    def test_two_revisions(self):
        self.edit(self.LLADA, lambda rows: [{**r, "revision": "other"} for r in rows])
        self.refused("more than one model revision", self.load)

    def test_other_diffusion_settings(self):
        self.edit(self.LLADA, lambda rows: [{**r, "n_slots": 8} for r in rows])
        self.refused("8 answer slots", self.load)
        self.edit(self.LLADA, lambda rows: [{**r, "n_slots": 1, "answer_prefix": "no_think"} for r in rows])
        self.refused("'no_think'", self.load)

    def test_other_ar_settings(self):
        run = self.root / self.QWEN.replace("scores_A.jsonl", "manifest.json")
        good = json.loads(run.read_text())
        for change in ({"compute_dtype": "torch.float32"}, {"batch_size": 8},
                       {"models": {**good["models"], "A": {"actual_quantization": "nf4", "attention_implementation": "sdpa"}}},
                       {"models": {**good["models"], "A": {"actual_quantization": None, "attention_implementation": "eager"}}}):
            run.write_text(json.dumps({**good, **change}))
            self.refused("Qwen2.5-7B was scored with", self.load)

    def test_the_gpu_is_known_per_domain_and_never_mixed(self):
        manifest = self.root / (self.LLADA + ".manifest.jsonl")
        n = len(read_jsonl(self.root / self.LLADA))
        for names, types, finite in ((["NVIDIA L4"], {"L4"}, True), (["NVIDIA A100-SXM4-40GB", "NVIDIA L4"], {"A100", "L4"}, False),
                                     ([None], {"unknown"}, True), (["NVIDIA L4", None], {"L4", "unknown"}, False)):
            done = [n - 3 * (len(names) - 1)] + [3] * (len(names) - 1)
            write_jsonl(manifest, [{"batch_size": 1, "completed": {"ok": d}, "environment": {"gpu": g}} for g, d in zip(names, done)])
            scores = self.load()
            self.assertEqual(scores.gpu_types("math", "LLaDA-8B", "T1"), types)
            self.assertEqual(scores.gpu_types("code", "LLaDA-8B", "T1"), {"A100"})
            self.assertEqual(np.isfinite(scores.seconds("math", "LLaDA-8B", "T1")), finite)     # a resumed run on another GPU

    def test_every_diffusion_row_needs_a_recorded_run_at_batch_size_1(self):
        manifest = self.root / (self.LLADA + ".manifest.jsonl")
        good, = read_jsonl(manifest)
        n = good["completed"]["ok"]
        write_jsonl(manifest, [{**good, "batch_size": 8}])
        self.refused("batch size 8, not 1", self.load)
        write_jsonl(manifest, [{**good, "completed": {"ok": 3}}])       # a run that died left rows without a record
        self.refused(f"record 3 scored rows, the file has {n}", self.load)
        write_jsonl(manifest, [{**good, "completed": {"ok": n - 2, "error": 2}}, {**good, "completed": {"ok": 2}}])
        self.load()                                                     # errors that a second run scored again
        manifest.unlink()
        self.refused("no run manifest", self.load)

    def test_ar_rows_outside_a_run_folder(self):
        path = self.root / self.QWEN
        rows = read_jsonl(path)
        self.edit(self.QWEN, lambda _: [])
        write_jsonl(self.root / "results/full/math/qwen.jsonl", rows)
        self.refused("read only from a run folder", self.load)

    def test_t3_is_refused_on_the_full_bank_and_skipped_only_when_asked(self):
        rows = read_jsonl(self.root / self.LLADA)
        write_jsonl(self.root / "results/full/math/llada/full_T3_n1.jsonl", [{**r, "template": "T3"} for r in rows])
        self.refused("'T3'", self.load)
        self.assertEqual(len(self.load(ignore_templates=("T3",)).columns), 32)

    def test_unknown_model_and_unknown_format(self):
        write_jsonl(self.root / "results/full/math/other.jsonl", [{"candidate_id": "x", "score": 1.0}])
        self.refused("unknown score row format", self.load)
        write_jsonl(self.root / "results/full/math/other.jsonl", [{**read_jsonl(self.root / self.LLADA)[0], "model": "illada"}])
        self.refused("unknown model", self.load)

    def test_unfinished_or_partial_ar_run(self):
        run = self.root / self.QWEN.replace("scores_A.jsonl", "manifest.json")
        m = json.loads(run.read_text())
        run.write_text(json.dumps({**m, "status": "failed"}))
        self.refused("run status is 'failed'", self.load)
        run.write_text(json.dumps({**m, "limit": 10}))
        self.refused("only the first 10 rows", self.load)

    def test_edited_ar_file(self):
        path = self.root / self.QWEN
        path.write_text(path.read_text().replace('"score": ', '"score":  ', 1), encoding="utf-8")
        self.refused("does not match the hash in its run manifest", self.load)

    def test_reconstruction_rows_are_ignored_unless_the_repair_needs_them(self):
        write_study(self.root, seed=1, problems={"math": 24, "code": 12}, with_recon=True)
        scores = self.load()
        self.assertTrue(any("reconstruction" in what for _, what in scores.unused))
        repair = parse_settings({**GO, "mode": "repair", "llada_score": "reconstruction"})
        needed = {d: needed_scores(repair, d) for d in DOMAINS}
        paths = [self.root / "results/full/math", self.root / "results/full/code"]
        self.assertEqual(load_scores(paths, self.banks, needed).score("math", "LLaDA-8B", "recon").shape,
                         (len(self.banks["math"].rows),))
        recon = "results/full/math/reconstruction/llada_128.jsonl"
        self.edit(recon, lambda rows: rows[:100] if len(rows) > 100 else rows[:-1])
        with self.assertRaises(DataError):
            load_scores(paths, self.banks, needed)
        write_study(self.root, seed=1, problems={"math": 24, "code": 12}, with_recon=True)
        self.edit(recon, lambda rows: [{**r, "micro_batch": 16} for r in rows])
        self.refused("micro_batch 16", lambda: load_scores(paths, self.banks, needed))
        self.edit(recon, lambda rows: [{**r, "micro_batch": 1} for r in rows])
        load_scores(paths, self.banks, needed)
        self.edit(recon, lambda rows: [{**r, "seed": 545} for r in rows])     # the second estimate is never the score
        with self.assertRaises(DataError):
            load_scores(paths, self.banks, needed)

    def test_gpu_types(self):
        self.assertEqual(gpu_type("NVIDIA A100-SXM4-80GB"), gpu_type("NVIDIA A100-SXM4-40GB"))
        self.assertNotEqual(gpu_type("NVIDIA A100-SXM4-40GB"), gpu_type("NVIDIA L4"))


class BankTests(Study):
    def rewrite(self, change):
        """Edit the math bank and its manifest together, so the content checks run."""
        import hashlib
        path = self.root / "bank/full/math/bank.jsonl"
        write_jsonl(path, change(read_jsonl(path)))
        manifest = self.root / "bank/full/math/manifest.json"
        m = json.loads(manifest.read_text())
        m["files"]["bank.jsonl"] = hashlib.sha256(path.read_bytes()).hexdigest()
        manifest.write_text(json.dumps(m))

    def load_math(self):
        folder = self.root / "bank/full/math"
        return load_bank(folder / "bank.jsonl", "math", manifest_path=folder / "manifest.json")

    def test_hash_must_match_the_bank_manifest(self):
        path = self.root / "bank/full/math/bank.jsonl"
        path.write_text(path.read_text() + "\n", encoding="utf-8")
        self.refused("does not match the bank manifest", self.load_math)

    def test_manifest_is_required(self):
        (self.root / "bank/full/math/manifest.json").unlink()
        self.refused("manifest is missing", self.load_math)

    def test_labels_in_the_bank_are_refused(self):
        self.rewrite(lambda rows: [{**rows[0], "label": 1}, *rows[1:]])
        self.refused("carries a label", self.load_math)

    def test_duplicate_candidate(self):
        self.rewrite(lambda rows: rows + [rows[0]])
        self.refused("duplicate candidate id", self.load_math)

    def test_problem_in_two_splits(self):
        self.rewrite(lambda rows: [{**r, "split": "test" if r["pair_id"] == rows[0]["pair_id"] else r["split"],
                                    "problem_id": rows[0]["problem_id"] if r["split"] == "validation" else r["problem_id"]}
                                   for r in rows])
        self.refused("two splits", self.load_math)

    def test_broken_pair(self):
        self.rewrite(lambda rows: rows[1:])
        self.refused("a pair must be two answers", self.load_math)

    def test_pilot_split_is_refused(self):
        self.rewrite(lambda rows: [{**r, "split": "pilot"} for r in rows])
        self.refused("split 'pilot'", self.load_math)

    def test_split_hash_ignores_row_order(self):
        before = self.load_math().split_sha256()
        self.rewrite(lambda rows: rows[::-1])
        self.assertEqual(before, self.load_math().split_sha256())

    def test_recorded_hash_layouts(self):
        for layout in ({"files": {"bank.jsonl": "a" * 64}}, {"outputs": {"bank/full/math/bank.jsonl": {"sha256": "a" * 64, "rows": 3}}},
                       {"files": [{"path": "math/bank.jsonl", "sha256": "a" * 64}]}, {"sha256": {"bank.jsonl": "a" * 64}}):
            self.assertEqual(recorded_hashes(layout)["bank.jsonl"], {"a" * 64})


class LabelTests(Study):
    def test_validation_labels_in_bank_order(self):
        bank = self.banks["math"]
        labels, _ = load_validation_labels(self.root / "bank/full/math/labels_validation.jsonl", bank)
        self.assertEqual(len(labels), int(bank.mask("validation").sum()))
        self.assertEqual(labels.sum() * 2, len(labels))

    def test_a_test_label_file_is_refused_by_name(self):
        self.refused("reads only labels_validation.jsonl",
                     lambda: load_validation_labels(self.root / "bank/full/math/labels_test.jsonl", self.banks["math"]))

    def test_test_rows_are_refused_under_any_name(self):
        path = self.root / "bank/full/math/labels_validation.jsonl"
        test_rows = read_jsonl(self.root / "bank/full/math/labels_test.jsonl")
        write_jsonl(path, read_jsonl(path) + test_rows[:1])
        self.banks["math"].recorded = {}        # not caught by the hash, so the content check has to catch it
        self.refused("candidate of another split", lambda: load_validation_labels(path, self.banks["math"]))

    def test_edited_validation_labels(self):
        path = self.root / "bank/full/math/labels_validation.jsonl"
        write_jsonl(path, read_jsonl(path)[::-1])
        self.refused("does not match the hash in the bank manifest", lambda: load_validation_labels(path, self.banks["math"]))

    def test_incomplete_duplicate_and_odd_labels(self):
        bank = self.banks["math"]
        rows = read_jsonl(self.root / "bank/full/math/labels_validation.jsonl")
        check = lambda changed: lambda: align_labels(changed, bank, "validation", "file")
        self.refused("no label", check(rows[1:]))
        self.refused("duplicate label", check(rows + rows[:1]))
        self.refused("expected 0 or 1", check([{**rows[0], "label": None}, *rows[1:]]))
        self.refused("expected 0 or 1", check([{**rows[0], "label": True}, *rows[1:]]))
        self.refused("expected one correct and one wrong", check([{**r, "label": 1} for r in rows]))
        self.refused("not in the bank", check(rows + [{"candidate_id": "nobody", "label": 1}]))

    def test_validation_code_has_no_path_to_test_labels(self):
        """The modules of the validation step neither import the test step nor name the test-label file."""
        for module in ("final_data", "final_settings", "final_stats", "final_validation", "overlap"):
            text = Path(f"analysis/{module}.py").read_text(encoding="utf-8")
            self.assertNotIn("labels_test", text, module)
            self.assertNotIn("import analysis.final_evaluate", text, module)
            self.assertNotIn("from analysis.final_evaluate", text, module)


class SmallTests(unittest.TestCase):
    def test_duplicate_json_keys(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "x.jsonl"
            path.write_text('{"score": 1, "score": 2}\n', encoding="utf-8")
            with self.assertRaises(DataError):
                read_jsonl(path)

    def test_sha(self):
        self.assertEqual(len(sha("x")), 64)


if __name__ == "__main__":
    unittest.main()
