"""Step 0 analysis on small synthetic runs with known answers. CPU only."""
import contextlib
import hashlib
import io
import json
import re
import tempfile
import unittest
from pathlib import Path

from analysis.step0 import agreement, load_run, main, rerun_check, swap_items, swap_result, wilson

QWEN, LLAMA = "Qwen/Qwen2.5-7B-Instruct", "meta-llama/Llama-3.1-8B-Instruct"


def write(path, rows):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")


def ar_row(cid, score, bank, model=QWEN, status="completed", template="T1"):
    return {"candidate_id": cid, "id": cid, "bank_sha256": bank, "model": "A", "model_id": model, "model_revision": "r1",
            "template": template, "score": score, "status": status}


def diffusion_row(cid, score, bank, status="ok", **kw):
    return {"candidate_id": cid, "bank_hash": bank, "model": "llada", "revision": "r2", "template": "T1", "n_slots": 1,
            "answer_prefix": "none", "score": score, "status": status, **kw}


def items(n):
    return [{"original": f"o{i}", "swap": f"s{i}", "domain": "math"} for i in range(n)]


def scores(n, lowered, tied=0):
    """First `lowered` swaps lower, then `tied` equal, the rest higher."""
    out = {f"o{i}": 1.0 for i in range(n)}
    out.update({f"s{i}": 0.5 if i < lowered else 1.0 if i < lowered + tied else 2.0 for i in range(n)})
    return out


class SwapTests(unittest.TestCase):
    def test_eighty_percent_passes_and_seventy_nine_fails(self):
        self.assertEqual(swap_result(items(100), scores(100, 80))["outcome"], "pass")
        r = swap_result(items(100), scores(100, 79))
        self.assertEqual((r["outcome"], r["lowered"], r["raised"]), ("fail", 79, 21))

    def test_a_tie_counts_as_not_lowered(self):
        r = swap_result(items(100), scores(100, 79, tied=21))
        self.assertEqual((r["outcome"], r["tied"], r["share_lowered"]), ("fail", 21, 0.79))

    def test_unscored_items_stay_in_the_denominator(self):
        s = scores(100, 85)
        s["s0"], s["o1"] = None, None
        del s["o2"]
        r = swap_result(items(100), s)
        self.assertEqual((r["items"], r["lowered"], r["unscored"], r["share_lowered"]), (100, 82, 3, 0.82))
        self.assertEqual(r["outcome"], "pass")

    def test_unscored_items_that_could_change_the_outcome_give_incomplete(self):
        s = scores(100, 81)
        for i in range(5):
            s[f"s{i}"] = None
        self.assertEqual(swap_result(items(100), s)["outcome"], "incomplete")   # 76 lowered, 81 possible
        s = scores(100, 60)
        s["s0"] = None
        self.assertEqual(swap_result(items(100), s)["outcome"], "fail")         # cannot reach 80
        self.assertEqual(swap_result(items(100), {})["outcome"], "incomplete")

    def test_wilson_interval(self):
        lo, hi = wilson(80, 100)
        self.assertAlmostEqual(lo, 0.7112, places=3)
        self.assertAlmostEqual(hi, 0.8666, places=3)
        self.assertEqual(wilson(0, 10)[0], 0.0)
        self.assertAlmostEqual(wilson(10, 10)[1], 1.0)

    def test_key_must_match_the_items(self):
        key = [{"candidate_id": "a", "pair_id": "p", "domain": "math", "role": "original"},
               {"candidate_id": "b", "pair_id": "p", "domain": "math", "role": "swap"}]
        rows = [{"candidate_id": "a"}, {"candidate_id": "b"}]
        self.assertEqual(swap_items(key, rows)["math"], [{"original": "a", "swap": "b", "domain": "math"}])
        for bad_key, bad_rows in ((key[:1], rows[:1]), (key, rows[:1]), ([key[0], {**key[1], "role": "original"}], rows)):
            with self.assertRaises(SystemExit):
                swap_items(bad_key, bad_rows)


class LoadTests(unittest.TestCase):
    def setUp(self):
        self.dir = tempfile.TemporaryDirectory()
        self.root = Path(self.dir.name)
        self.addCleanup(self.dir.cleanup)

    def load(self, ids=("a", "b")):
        return load_run([str(self.root)], "h", set(ids))

    def test_failed_error_and_non_finite_scores_are_kept_as_none(self):
        write(self.root / "x/scores_A.jsonl", [ar_row("a", None, "h", status="failed"), ar_row("b", 1.5, "h")])
        write(self.root / "llada_T1_n1.jsonl", [diffusion_row("a", None, "h", status="too_long"),
                                                diffusion_row("b", float("nan"), "h")])
        write(self.root / "llada_T1_n1.jsonl.manifest.jsonl", [{"model": "llada"}])
        run = self.load()
        self.assertEqual(run["Qwen2.5-7B"]["scores"], {"a": None, "b": 1.5})
        self.assertEqual(run["LLaDA-8B"]["scores"], {"a": None, "b": None})
        self.assertEqual(run["LLaDA-8B"]["revision"], "r2")

    def test_wrong_bank_template_slots_unknown_ids_and_duplicates_are_errors(self):
        for rows in ([ar_row("a", 1.0, "other")], [ar_row("a", 1.0, "h", template="T2")], [ar_row("zzz", 1.0, "h")],
                     [ar_row("a", 1.0, "h"), ar_row("a", 1.0, "h")], [ar_row("a", 1.0, "h", model="who/knows")]):
            write(self.root / "x/scores_A.jsonl", rows)
            with self.assertRaises(SystemExit):
                self.load()
        write(self.root / "x/scores_A.jsonl", [ar_row("a", 1.0, "h")])
        for kw in ({"n_slots": 8}, {"answer_prefix": "think_nl"}):
            write(self.root / "llada_T1_n1.jsonl", [diffusion_row("a", 1.0, "h", **kw)])
            with self.assertRaises(SystemExit):
                self.load()

    def test_one_verifier_in_two_files_is_an_error(self):
        write(self.root / "x/scores_A.jsonl", [ar_row("a", 1.0, "h")])
        write(self.root / "y/scores_A.jsonl", [ar_row("b", 1.0, "h")])
        with self.assertRaises(SystemExit):
            self.load()

    def test_a_missing_folder_is_an_error(self):
        with self.assertRaises(SystemExit):
            load_run([str(self.root / "nope")], "h", set())


class RerunTests(unittest.TestCase):
    bank = [{"candidate_id": f"c{i}", "domain": "math" if i < 10 else "code"} for i in range(20)]

    def run_of(self, values, name="Qwen2.5-7B"):
        return {name: {"scores": {f"c{i}": v for i, v in enumerate(values)}, "revision": "r", "file": None}}

    def check(self, a, b, prev=()):
        return rerun_check(self.bank, self.run_of(a), self.run_of(b), self.run_of(prev) if prev else {})["Qwen2.5-7B"]

    def test_identical_runs_pass_and_reordered_runs_fail(self):
        x = [float(i) for i in range(20)]
        r = self.check(x, x, x)
        self.assertEqual((r["outcome"], r["run_1_vs_run_2"]["all"]["identical"]), ("pass", 20))
        self.assertEqual(r["run_1_vs_october_27"]["math"]["n"], 10)
        y = x[:]
        y[0], y[1] = y[1], y[0]        # one swap of neighbours: 1 - 6*2/(20*399) = 0.9985
        r = self.check(x, y)
        self.assertEqual(r["outcome"], "fail")
        self.assertAlmostEqual(r["run_1_vs_run_2"]["all"]["spearman"], 1 - 12 / (20 * 399))
        self.assertAlmostEqual(r["run_1_vs_run_2"]["code"]["spearman"], 1.0)
        self.assertIsNone(r["run_1_vs_october_27"])

    def test_october_agreement_never_decides(self):
        x = [float(i) for i in range(20)]
        self.assertEqual(self.check(x, x, x[::-1])["outcome"], "pass")

    def test_missing_failed_or_constant_scores_never_pass(self):
        x = [float(i) for i in range(20)]
        self.assertEqual(self.check(x, x[:-1])["outcome"], "incomplete")
        r = self.check(x, x[:-1] + [None])
        self.assertEqual((r["outcome"], r["unscored"]), ("incomplete", {"run_1": 0, "run_2": 1}))
        self.assertEqual(self.check([1.0] * 20, [1.0] * 20)["outcome"], "fail")
        out = rerun_check(self.bank, self.run_of(x), {}, {})
        self.assertEqual((out["Qwen2.5-7B"]["outcome"], out["Gemma-2-9B"]["outcome"]), ("incomplete", "not run"))
        self.assertIsNone(agreement({}, {}, []))


class EndToEnd(unittest.TestCase):
    def test_report_lists_every_verifier_and_the_go_no_go(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            rows = [{"candidate_id": c} for i in range(10) for c in (f"o{i}", f"s{i}")]
            key = [{"candidate_id": f"{r}{i}", "pair_id": f"p{i}", "domain": "math" if i < 5 else "code",
                    "role": "original" if r == "o" else "swap"} for i in range(10) for r in "os"]
            bank = [{"candidate_id": f"c{i}", "domain": "math" if i % 2 else "code"} for i in range(12)]
            write(root / "items.jsonl", rows)
            write(root / "key.jsonl", key)
            write(root / "bank.jsonl", bank)
            sh, bh = (hashlib.sha256((root / f).read_bytes()).hexdigest() for f in ("items.jsonl", "bank.jsonl"))
            s = scores(10, 4)
            write(root / "swap/cfg/scores_A.jsonl", [ar_row(c, v, sh) for c, v in s.items()])
            write(root / "swap/llada_T1_n1.jsonl", [diffusion_row(c, v, sh) for c, v in scores(10, 2).items()])
            for run in ("rerun_1", "rerun_2"):
                write(root / run / "cfg/scores_A.jsonl", [ar_row(r["candidate_id"], float(i), bh) for i, r in enumerate(bank)])
            with contextlib.redirect_stdout(io.StringIO()):
                main(["--swap-items", str(root / "items.jsonl"), "--swap-key", str(root / "key.jsonl"),
                      "--swap-scores", str(root / "swap"), "--bank", str(root / "bank.jsonl"), "--previous",
                      "--rerun", str(root / "rerun_1"), str(root / "rerun_2"), "--out", str(root / "out")])
            result = json.loads((root / "out/step0.json").read_text(encoding="utf-8"))
            self.assertEqual(result["swap"]["Qwen2.5-7B"]["math"]["outcome"], "pass")      # 4 of 5
            self.assertEqual(result["swap"]["Qwen2.5-7B"]["code"]["outcome"], "fail")
            self.assertEqual(result["swap"]["Dream-7B"]["math"], {"outcome": "not run"})
            self.assertEqual(result["go_no_go"]["Qwen2.5-7B"], {"answer_swap_math": "pass", "repeat_runs": "pass"})
            self.assertEqual(result["go_no_go"]["LLaDA-8B"]["answer_swap_math"], "fail")   # 2 of 5
            self.assertEqual(result["go_no_go"]["Gemma-2-9B"], {"answer_swap_math": "not run", "repeat_runs": "not run"})
            text = (root / "out/step0.md").read_text(encoding="utf-8")
            for name in ("Llama-3.1-8B", "Mistral-7B", "Dream-7B"):
                self.assertIn(name, text)
            widths = {len(re.findall(r"(?<!\\)\|", line)) for line in text.splitlines() if line.startswith("| ")}
            self.assertEqual(widths, {10, 12, 4})  # every row has its header's columns

    def test_notebook_cells_compile_and_call_existing_flags(self):
        cells = [c for c in json.loads(Path("notebooks/09_step0.ipynb").read_text(encoding="utf-8"))["cells"] if c["cell_type"] == "code"]
        for module, path in (("analysis.step0", "analysis/step0.py"), ("bank.token_lengths", "bank/token_lengths.py")):
            lines = [line for c in cells for line in "".join(c["source"]).splitlines() if f"-m {module} " in line]
            self.assertTrue(lines)
            for flag in {f for line in lines for f in re.findall(r"--[a-z-]+", line)}:
                self.assertIn(f'"{flag}"', Path(path).read_text(encoding="utf-8"))
        try:
            from IPython.core.inputtransformer2 import TransformerManager
        except ImportError:
            self.skipTest("IPython not installed")
        for c in cells:
            compile(TransformerManager().transform_cell("".join(c["source"])), "cell", "exec")

    def test_pilot_swap_files_load(self):
        from analysis.pilot import read_jsonl
        items = swap_items(read_jsonl("bank/pilot/swap/swap_key.jsonl"), read_jsonl("bank/pilot/swap/swap_items.jsonl"))
        self.assertEqual({d: len(v) for d, v in items.items()}, {"math": 100, "code": 100})


if __name__ == "__main__":
    unittest.main()
