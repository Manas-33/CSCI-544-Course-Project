"""Guards, resume and completeness check of the full-bank scoring. CPU only: a synthetic study in a temporary git
repository, and a stand-in for the two scorers that writes their file formats."""
import contextlib
import hashlib
import io
import json
import math
import shutil
import subprocess
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

from analysis.final_data import load_banks, load_scores, read_jsonl
from analysis.final_settings import BY_KEY, VERIFIERS, needed_scores, parse_settings
from analysis.final_synthetic import GO, ar_row, diffusion_row, write_jsonl, write_study
from harness import full_run, run as ar_run
from harness.full_run import GROUPS, SCORING_CODE, Step, main, steps, verify

REPO = Path(__file__).resolve().parent.parent
A100, L4 = "NVIDIA A100-SXM4-40GB", "NVIDIA L4"
PROBLEMS = {"math": 12, "code": 6}


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def git(root, *args):
    return subprocess.run(["git", "-C", str(root), "-c", "user.name=t", "-c", "user.email=t@t", *args], capture_output=True,
                          text=True, check=True).stdout.strip()


def option(argv, name):
    return argv[argv.index(name) + 1]


class FakeScorer:
    """Stands in for harness.run and run_score: reads the command, writes the files a finished run leaves."""

    def __init__(self, gpu=A100, fail_at=None, change=None):
        self.gpu, self.fail_at, self.change, self.calls = gpu, fail_at, change, []

    def __call__(self, argv, cwd):
        self.calls.append(argv)
        assert "labels" not in " ".join(argv)
        died = self.fail_at is not None and len(self.calls) == self.fail_at
        (self.ar if "harness.run" in argv else self.diffusion)(argv, Path(cwd), died)
        return 1 if died else 0

    def ar(self, argv, root, died):
        out, bank, config = root / option(argv, "--output-dir"), root / option(argv, "--input"), root / option(argv, "--config")
        check = ["--config", str(config), "--input", str(bank), "--template", option(argv, "--template"), "--output-dir", str(out),
                 *argv[argv.index("--device"):], "--validate-only"]
        with contextlib.redirect_stdout(io.StringIO()):
            assert ar_run.main(check) == 0            # the real argument parser, config check and bank reader
        out.mkdir(parents=True)                       # harness.run refuses an existing folder
        rows, models = read_jsonl(bank), json.loads(config.read_text())["models"]
        info = {}
        for which, spec in models.items():
            scored = [{**ar_row(r, which, BY_KEY[spec["model_id"]], option(argv, "--template"), k % 7 - 3.0, sha(bank), k == 0),
                       "model_revision": spec["revision"]} for k, r in enumerate(rows)]
            write_jsonl(out / f"scores_{which}.jsonl", scored[:3] if died else scored)
            info[which] = {"requested": spec, "resolved_model_revision": spec["revision"],
                           "resolved_tokenizer_revision": spec["tokenizer_revision"], "actual_quantization": None,
                           "attention_implementation": spec.get("attn_implementation", "sdpa"), "peak_gpu_allocated_bytes": 16e9}
        manifest = {"status": "running" if died else "completed", "limit": None, "input_sha256": sha(bank),
                    "template": option(argv, "--template"), "compute_dtype": "torch.bfloat16", "batch_size": 1,
                    "config_file_sha256": sha(config), "models": info, "gpu": {"name": self.gpu},
                    "outputs_sha256": {p.name: sha(p) for p in out.glob("*.jsonl")}}
        if self.change:
            self.change(manifest)
        (out / "manifest.json").write_text(json.dumps(manifest))

    def diffusion(self, argv, root, died):
        out, items, key = root / option(argv, "--out"), root / option(argv, "--items"), option(argv, "--model")
        assert option(argv, "--bank-hash") == sha(items) and option(argv, "--batch-size") == "1"
        rows = [{**diffusion_row(r, BY_KEY[key], option(argv, "--template"), k % 5 - 2.0, sha(items)),
                 "revision": full_run.DIFFUSION_REVISIONS[key]} for k, r in enumerate(read_jsonl(items))]
        if died:
            write_jsonl(out, rows[:3])      # a run that died: rows without a run manifest
            return
        write_jsonl(out, rows)
        run = {"model": key, "revision": full_run.DIFFUSION_REVISIONS[key], "n_slots": 1, "template": option(argv, "--template"),
               "bank_hash": sha(items), "answer_prefix": "none", "batch_size": 1, "completed": {"ok": len(rows)},
               "environment": {"gpu": self.gpu}}
        if self.change:
            self.change(run)
        write_jsonl(Path(str(out) + ".manifest.jsonl"), [run])


class Study(unittest.TestCase):
    """A repository with a merged model record and two frozen banks, and an empty saved folder."""

    def setUp(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.root, self.saved = Path(tmp.name) / "repo", Path(tmp.name) / "drive/full"
        write_study(self.root, seed=2, problems=PROBLEMS)
        shutil.rmtree(self.root / "results")
        manifest = self.root / "bank/full/math/manifest.json"
        manifest.write_text(json.dumps({**json.loads(manifest.read_text()), "math_error_tags": 20}))
        for domain in PROBLEMS:
            for path in (self.root / "bank/full" / domain).glob("labels_*.jsonl"):
                path.unlink()               # nothing here may need a label
        for name in SCORING_CODE:
            (self.root / name).parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(REPO / name, self.root / name)
        self.saved.mkdir(parents=True)
        (self.root / ".gitignore").write_text("/results/\n")
        git(self.root, "init", "-q", "-b", "main")
        self.commit("banks and record")

    def commit(self, message, merged=True):
        git(self.root, "add", "-A")
        git(self.root, "commit", "-q", "-m", message)
        if merged:
            git(self.root, "update-ref", "refs/remotes/origin/main", "HEAD")

    def call(self, command, *args, gpu=A100, execute=None, domains=("math", "code"), saved=True):
        """(exit code, printed text) of one command."""
        argv = [command, "--root", str(self.root), "--domains", *domains, *(["--saved", str(self.saved)] if saved else []), *args]
        out = io.StringIO()
        with contextlib.redirect_stdout(out):
            code = main(argv, gpu_name=lambda: gpu, execute=execute or FakeScorer(gpu))
        return code, out.getvalue()

    def score_all(self, groups=GROUPS, **kwargs):
        for group in groups:
            code, text = self.call("score", "--group", group, **kwargs)
            self.assertEqual(code, 0, text)

    def refused(self, fragment, command="start", *args, **kwargs):
        for c in (command, *(("score",) if command == "start" else ())):     # score has the same guards as start
            code, text = self.call(c, *args, *(("--group", "small") if c == "score" else ()), **kwargs)
            self.assertEqual(code, 2, text)
            self.assertIn("REFUSED", text)
            self.assertIn(fragment, text)
        self.assertFalse(self.saved.exists() and any(p.name.startswith("scores_") for p in self.saved.rglob("*small*/*")))

    def edit_json(self, path, change):
        d = json.loads((self.root / path).read_text())
        change(d)
        (self.root / path).write_text(json.dumps(d))


class RecordTests(Study):
    def test_start_passes_and_lists_what_is_to_do(self):
        code, text = self.call("start")
        self.assertEqual(code, 0, text)
        self.assertEqual(text.count("to do"), len(GROUPS) * 2 * 2)
        self.assertIn("GPU type A100", text)
        self.assertFalse((self.saved / "gpu.txt").exists())     # a session that scores nothing fixes nothing

    def test_no_settings_file(self):
        (self.root / full_run.SETTINGS).unlink()
        self.refused("configs/final_models.json is missing")

    def test_invalid_settings(self):
        self.edit_json(full_run.SETTINGS, lambda d: d.update(h1_ar=["Mistral-7B"]))
        self.commit("bad settings")
        self.refused("h1_ar must be a subset")

    def test_missing_record(self):
        (self.root / GO["record"]).unlink()
        self.commit("no record")
        self.refused("is missing: no full-bank answer is scored before")

    def test_record_that_is_not_a_decision(self):
        for record in ("PLAN.md", "decisions/README.md", "decisions/../PLAN.md"):
            (self.root / "PLAN.md").write_text("plan")
            (self.root / "decisions/README.md").write_text("list")
            self.edit_json(full_run.SETTINGS, lambda d, r=record: d.update(record=r))
            self.commit("other record")
            self.refused("is not a decision record")

    def test_uncommitted_record_and_settings(self):
        git(self.root, "rm", "-q", "--cached", GO["record"])
        self.commit_index("record out of git")
        self.refused("is not committed")
        self.commit("record back")
        self.edit_json(full_run.SETTINGS, lambda d: d.update(note="edited"))
        self.refused("differs from the committed file")

    def commit_index(self, message):
        git(self.root, "commit", "-q", "-m", message)
        git(self.root, "update-ref", "refs/remotes/origin/main", "HEAD")

    def test_a_branch_that_is_not_merged(self):
        (self.root / "new.txt").write_text("x")
        self.commit("not merged", merged=False)
        self.refused("is not part of origin/main")

    def test_stopped(self):
        self.edit_json(full_run.SETTINGS, lambda d: d.update(mode="stopped", llada_score=None, reason="second audit failed"))
        self.commit("stopped")
        self.refused("'stopped'")

    def test_repair_and_qwen3_notes(self):
        self.edit_json(full_run.SETTINGS, lambda d: d.update(mode="repair", llada_score="reconstruction", qwen3_scored=True))
        self.commit("repair")
        code, text = self.call("start")
        self.assertEqual(code, 0, text)
        self.assertIn("notebook 12 (WHOLE_BANK = True)", text)
        self.assertIn("Qwen3-8B", text)


class BankTests(Study):
    BANK, MANIFEST = "bank/full/math/bank.jsonl", "bank/full/math/manifest.json"

    def test_missing_bank_and_the_other_one_alone(self):
        shutil.rmtree(self.root / "bank/full/code")
        self.commit("math only")
        self.refused("bank/full/code/bank.jsonl is missing")
        self.assertEqual(self.call("start", domains=("math",))[0], 0)

    def test_a_bank_merged_after_the_other_ones_commit(self):
        files = self.root / "bank/full/code"
        shutil.move(files, self.root.parent / "code")
        shutil.rmtree(self.root / ".git")               # a history in which the code bank comes later
        git(self.root, "init", "-q", "-b", "main")
        self.commit("math only")
        self.score_all(["small"], domains=("math",))
        first = git(self.root, "rev-parse", "HEAD")
        shutil.move(self.root.parent / "code", files)
        self.commit("code bank frozen")
        self.assertEqual(self.call("start", domains=("code",))[0], 0)
        git(self.root, "checkout", "-q", first)             # what the notebook does for both banks together
        code, text = self.call("start")
        self.assertEqual(code, 2)
        self.assertIn('DOMAINS = ["code"]', text)

    def test_missing_manifest(self):
        (self.root / self.MANIFEST).unlink()
        self.commit("no manifest")
        self.refused("manifest.json is missing")

    def test_hash_that_does_not_match(self):
        self.edit_json(self.MANIFEST, lambda d: d["files"].update({"bank.jsonl": "0" * 64}))
        self.commit("wrong hash")
        self.refused("does not match the bank manifest")

    def test_edited_bank(self):
        path = self.root / self.BANK
        path.write_text(path.read_text() + "\n")
        self.refused("differs from the committed file")

    def test_uncommitted_bank(self):
        git(self.root, "rm", "-q", "--cached", self.BANK)
        git(self.root, "commit", "-q", "-m", "bank out of git")
        git(self.root, "update-ref", "refs/remotes/origin/main", "HEAD")
        self.refused("is not committed")

    def test_unmerged_bank(self):
        git(self.root, "update-ref", "refs/remotes/origin/main", "HEAD")
        rows = read_jsonl(self.root / self.BANK)
        write_jsonl(self.root / self.BANK, rows[::-1])
        self.edit_json(self.MANIFEST, lambda d: d["files"].update({"bank.jsonl": sha(self.root / self.BANK)}))
        self.commit("new bank, not merged", merged=False)
        self.refused("is not part of origin/main")
        git(self.root, "update-ref", "refs/remotes/origin/main", "HEAD")
        self.refused("changed in 2 commits")                        # merged now, but a frozen bank never changes

    def test_bank_changed_on_main_after_the_fixed_commit(self):
        first = git(self.root, "rev-parse", "HEAD")
        write_jsonl(self.root / self.BANK, read_jsonl(self.root / self.BANK)[::-1])
        self.edit_json(self.MANIFEST, lambda d: d["files"].update({"bank.jsonl": sha(self.root / self.BANK)}))
        self.commit("bank changed")
        git(self.root, "checkout", "-q", first)
        self.refused("changed on origin/main after this commit")

    def test_math_without_error_tags_is_not_frozen(self):
        self.edit_json(self.MANIFEST, lambda d: d.pop("math_error_tags"))
        self.commit("no tags")
        self.refused("no error tags yet")

    def test_label_fields_in_the_bank(self):
        for field, fragment in (("label", "carries a label"), ("extracted", "label fields ['extracted']")):
            rows = [{**r, field: 1} for r in read_jsonl(self.root / self.BANK)]
            write_jsonl(self.root / self.BANK, rows)
            self.edit_json(self.MANIFEST, lambda d: d["files"].update({"bank.jsonl": sha(self.root / self.BANK)}))
            self.commit("labels in the bank")
            self.refused(fragment)
            write_jsonl(self.root / self.BANK, [{k: v for k, v in r.items() if k != field} for r in rows])

    def test_not_a_git_checkout(self):
        shutil.rmtree(self.root / ".git")
        self.refused("not a git checkout")


class GpuTests(Study):
    def test_only_a100_and_l4(self):
        for name in ("", "Tesla T4", "NVIDIA L40S", "Tesla V100-SXM2-16GB"):
            self.refused("pick an A100 or an L4", gpu=name)

    def test_the_first_saved_run_fixes_the_type_for_everything(self):
        self.score_all(["small"], gpu=L4, domains=("math",))
        self.assertEqual((self.saved / "gpu.txt").read_text(), "L4\n")
        for domains in (("math",), ("code",)):                      # the other verifiers, and the other bank
            code, text = self.call("start", gpu=A100, domains=domains)
            self.assertEqual(code, 2)
            self.assertIn("says L4", text)
            self.assertIn("Change runtime type", text)
            self.assertIn("start all the full-bank scoring over on A100", text)
            code, text = self.call("score", "--group", "dream", gpu=A100, domains=domains)
            self.assertEqual(code, 2)
        self.assertEqual(self.call("start", gpu="NVIDIA L4")[0], 0)

    def test_40_and_80_gb_a100_are_one_type(self):
        self.score_all(["small"], gpu=A100)
        self.assertEqual(self.call("start", gpu="NVIDIA A100-SXM4-80GB")[0], 0)

    def test_the_score_files_count_when_gpu_txt_is_gone(self):
        self.score_all(["small"], gpu=L4)
        (self.saved / "gpu.txt").unlink()
        self.assertIn("says L4", self.call("start", gpu=A100)[1])

    def test_gpu_txt_committed_on_main(self):
        (self.root / "results/full").mkdir(parents=True)
        (self.root / "results/full/gpu.txt").write_text("L4\n")
        git(self.root, "add", "-f", "results/full/gpu.txt")
        self.commit("scores of another account")
        self.refused("results/full/gpu.txt on origin/main says L4", gpu=A100)

    def test_notebook_12_scores_count_once_something_is_scored(self):
        other = self.saved.parent / "reconstruction/math/gpu.txt"
        other.parent.mkdir(parents=True)
        other.write_text("L4")
        also = ("--also-gpu", str(other))
        self.assertEqual(self.call("start", *also, gpu=A100)[0], 0)          # nothing scored there yet
        write_jsonl(other.parent / "whole/llada_n128_s544.jsonl", [{"x": 1}])
        self.refused("says L4", "start", *also, gpu=A100)

    def test_a_second_session_that_finishes_on_another_type_is_not_saved(self):
        def other_session(argv, cwd):
            result = FakeScorer(A100)(argv, cwd)
            self.saved.mkdir(parents=True, exist_ok=True)
            (self.saved / "gpu.txt").write_text("L4\n")
            return result
        code, text = self.call("score", "--group", "small", gpu=A100, execute=other_session)
        self.assertEqual(code, 2)
        self.assertIn("another session saved scores first", text)
        self.assertEqual([p.name for p in self.saved.rglob("*")], ["gpu.txt"])

    def test_a_run_that_reports_another_gpu_is_not_saved(self):
        for name in (L4, None):
            code, text = self.call("score", "--group", "llada", gpu=A100, execute=FakeScorer(name))
            self.assertEqual(code, 2)
            self.assertIn("it is not saved", text)
            shutil.rmtree(self.root / "results")


class CommitTests(Study):
    def test_settings_that_changed_on_main_are_announced(self):
        first = git(self.root, "rev-parse", "HEAD")
        self.edit_json(full_run.SETTINGS, lambda d: d.update(note="new"))
        self.commit("settings changed")
        git(self.root, "checkout", "-q", first)
        code, text = self.call("start")
        self.assertEqual(code, 0)
        self.assertIn("changed on origin/main after this commit", text)

    def test_a_resumed_bank_keeps_its_commit(self):
        self.score_all(["small"], domains=("math",))
        (self.root / "unrelated.txt").write_text("x")
        self.commit("unrelated")
        code, text = self.call("start", domains=("math",))
        self.assertEqual(code, 2)
        self.assertIn("the saved math scores were made at commit", text)
        self.assertEqual(self.call("start", domains=("code",))[0], 0)       # the other bank may use a later commit

    def test_but_both_banks_use_the_same_scoring_code(self):
        self.score_all(["small"], domains=("math",))
        path = self.root / "harness/prompts.py"
        path.write_text(path.read_text() + "\n# changed\n")
        code, text = self.call("start", domains=("code",))
        self.assertIn("uncommitted changes in the scoring code: harness/prompts.py", text)
        self.commit("scoring code changed")
        code, text = self.call("start", domains=("code",))
        self.assertEqual(code, 2)
        self.assertIn("the scoring code changed since the saved math scores", text)


class ResumeTests(Study):
    def test_finished_runs_are_skipped_after_a_disconnect(self):
        self.score_all(["small", "llada"])
        shutil.rmtree(self.root / "results")                    # a new runtime
        code, text = self.call("start")
        self.assertEqual(code, 0, text)
        self.assertEqual(text.count("  done "), 8)
        scorer = FakeScorer()
        self.score_all(["small", "llada"], execute=scorer)
        self.assertEqual(scorer.calls, [])
        self.score_all(["dream"], execute=scorer)
        self.assertEqual(len(scorer.calls), 4)

    def test_a_run_that_died_is_scored_again_from_the_start(self):
        for group in ("small", "llada"):
            code, text = self.call("score", "--group", group, execute=FakeScorer(fail_at=2))
            self.assertEqual(code, 1)
            self.assertIn("did not finish", text)
            saved = [p.name for p in self.saved.glob("math/*/full_*") if group in str(p.relative_to(self.saved))]
            self.assertEqual({n.split(".")[0] for n in saved}, {"full_small_T1" if group == "small" else "full_T1_n1"})  # only the finished run
            scorer = FakeScorer()
            self.score_all([group], execute=scorer)     # same runtime: the dead run's files are still there
            self.assertEqual(len(scorer.calls), 3)
            path = self.root / "results/full/math/llada/full_T2_n1.jsonl"
            if group == "llada":
                self.assertEqual(len(read_jsonl(path)), len(read_jsonl(str(path) + ".manifest.jsonl")) * len(
                    read_jsonl(self.root / "bank/full/math/bank.jsonl")))

    def test_a_copy_to_drive_that_was_cut_off_does_not_count(self):
        self.score_all(["small"], domains=("math",))
        step = self.saved / "math/ar/full_small_T1"
        step.rename(step.with_name(step.name + ".part"))
        code, text = self.call("start", domains=("math",))
        self.assertEqual(code, 0, text)
        self.assertIn("to do math/ar/full_small_T1", text)
        self.score_all(["small"], domains=("math",))
        self.assertTrue(step.is_dir() and not step.with_name(step.name + ".part").exists())

    def test_a_run_another_session_finished_meanwhile_is_taken_over(self):
        self.score_all(["small"], domains=("math",))
        shutil.rmtree(self.root / "results")
        scorer = FakeScorer()
        self.score_all(["small"], domains=("math",), execute=scorer)
        self.assertEqual(scorer.calls, [])

    def test_without_the_saved_folder_nothing_runs(self):
        self.score_all(["small"], domains=("math",), gpu=L4)
        self.saved.rename(self.saved.with_name("unmounted"))
        for command in (("start",), ("score", "--group", "llada")):
            code, text = self.call(*command, gpu=A100)
            self.assertEqual(code, 2)
            self.assertIn("mount Google Drive first", text)

    def test_a_saved_copy_that_went_bad_is_replaced(self):
        self.score_all(["small"], domains=("math",))
        (self.saved / "math/ar/full_small_T2/scores_A.jsonl").unlink()
        scorer = FakeScorer()
        self.score_all(["small"], domains=("math",), execute=scorer)
        self.assertEqual(scorer.calls, [])
        self.assertTrue((self.saved / "math/ar/full_small_T2/scores_A.jsonl").is_file())

    def test_groups_and_domains_run_on_their_own(self):
        scorer = FakeScorer()
        self.score_all(["llada"], domains=("code",), execute=scorer)
        self.assertEqual([option(c, "--out") for c in scorer.calls],
                         ["results/full/code/llada/full_T1_n1.jsonl", "results/full/code/llada/full_T2_n1.jsonl"])
        self.assertEqual(sorted(p.name for p in self.saved.iterdir()), ["code", "gpu.txt"])


class ForeignFileTests(Study):
    AR, LLADA = "math/ar/full_small_T1", "math/llada/full_T1_n1.jsonl"

    def setUp(self):
        super().setUp()
        self.score_all(["small", "llada"], domains=("math",))

    def rewrite(self, relative, change, saved=True):
        path = self.saved / relative
        write_jsonl(path, change(read_jsonl(path)))
        if path.name.startswith("scores_"):
            m = json.loads((path.parent / "manifest.json").read_text())
            m["outputs_sha256"][path.name] = sha(path)
            (path.parent / "manifest.json").write_text(json.dumps(m))

    def refuses(self, *fragments):
        code, text = self.call("start", domains=("math",))
        self.assertEqual(code, 2, text)
        for f in fragments:
            self.assertIn(f, text)
        self.assertIn("Delete these runs", text)
        code, text = self.call("check", domains=("math",))
        self.assertEqual(code, 1)

    def test_diffusion_rows_with_other_settings(self):
        path = self.saved / self.LLADA
        good = path.read_text()
        for key, value in (("bank_hash", "0" * 64), ("revision", "main"), ("template", "T2"), ("n_slots", 8),
                           ("answer_prefix", "think_nl"), ("model", "dream")):
            self.rewrite(self.LLADA, lambda rows: [{**r, key: value} for r in rows])
            self.refuses(f"{key} {value}, not")
            path.write_text(good)

    def test_diffusion_runs_at_another_batch_size_or_without_a_record(self):
        beside = self.saved / (self.LLADA + ".manifest.jsonl")
        good, = read_jsonl(beside)
        write_jsonl(beside, [{**good, "batch_size": 8}])
        self.refuses("batch_size 8, not 1")
        write_jsonl(beside, [{**good, "completed": {"ok": 2}}])
        self.refuses("record 2 scored rows")
        beside.unlink()
        code, text = self.call("start", domains=("math",))
        self.assertEqual(code, 2)
        self.assertIn("does not say that an A100 or an L4 made it (unknown)", text)

    def test_ar_runs_with_other_settings(self):
        path = self.saved / self.AR / "manifest.json"
        good = path.read_text()
        for change, fragment in ((lambda m: m.update(input_sha256="0" * 64), "input_sha256"),
                                 (lambda m: m.update(template="T3"), "template T3, not T1"),
                                 (lambda m: m.update(compute_dtype="torch.float32"), "compute_dtype torch.float32"),
                                 (lambda m: m.update(batch_size=8), "batch_size 8, not 1"),
                                 (lambda m: m.update(limit=5), "limit 5"),
                                 (lambda m: m.update(config_file_sha256="0" * 64), "config_file_sha256"),
                                 (lambda m: m["models"]["A"].update(attention_implementation="eager"), "attention_implementation eager, not sdpa"),
                                 (lambda m: m["models"]["A"].update(actual_quantization="nf4"), "actual_quantization nf4"),
                                 (lambda m: m["models"]["B"].update(resolved_model_revision="other"), "resolved_model_revision other"),
                                 (lambda m: m["models"]["B"].update(resolved_tokenizer_revision="other"), "resolved_tokenizer_revision other"),
                                 (lambda m: m.update(status="failed"), "the run is 'failed'")):
            m = json.loads(good)
            change(m)
            path.write_text(json.dumps(m))
            self.refuses(fragment)
        path.write_text(good)
        self.assertEqual(self.call("start", domains=("math",))[0], 0)

    def test_edited_mixed_and_partial_files(self):
        rows = self.saved / self.AR / "scores_A.jsonl"
        text = rows.read_text()
        rows.write_text(text.replace('"score": ', '"score":  ', 1))
        self.refuses("does not match the hash in its run manifest")
        rows.write_text(text)
        self.rewrite(self.LLADA, lambda r: r[:-1])
        self.refuses("1 of")
        self.rewrite(self.LLADA, lambda r: r + [r[0]])
        self.refuses("duplicate rows")
        self.rewrite(self.LLADA, lambda r: [{**r[0], "candidate_id": "code-0000__0__0"}, *r[1:]])     # the other bank
        self.refuses("not in the bank")

    def test_files_of_another_scoring(self):
        shutil.copytree(self.saved / self.AR, self.saved / "math/ar/full_small_T3")
        self.refuses("full_small_T3", "not a file of this scoring")

    def test_a_reconstruction_file_may_lie_beside_them(self):
        write_jsonl(self.saved / "math/reconstruction/llada.jsonl", [{"method": "diffusion_mc", "model": "llada", "gpu": A100}])
        write_jsonl(self.saved / "math/reconstruction/llada.jsonl.manifest.jsonl", [{"environment": {"gpu": A100}}])
        self.assertEqual(self.call("start", domains=("math",))[0], 0)

    def test_score_refuses_a_foreign_local_run(self):
        shutil.rmtree(self.saved / "math/llada")
        path = self.root / "results/full" / self.LLADA
        write_jsonl(path, [{**r, "n_slots": 8} for r in read_jsonl(path)])
        code, text = self.call("score", "--group", "llada", domains=("math",))
        self.assertEqual(code, 2)
        self.assertIn("was made with other settings", text)


class EndToEndTests(Study):
    def test_score_check_and_load(self):
        self.assertEqual(self.call("start")[0], 0)
        scorer = FakeScorer(L4)
        self.score_all(gpu=L4, execute=scorer)
        self.assertEqual(len(scorer.calls), len(GROUPS) * 2 * 2)
        shutil.rmtree(self.root / "results")
        code, text = self.call("check")
        self.assertEqual(code, 0, text)
        self.assertIn("COMPLETE: every verifier", text)
        self.assertEqual(text.count("NVIDIA L4"), 8 * 2 * 2)
        report = json.loads((self.saved / "completeness_math_code.json").read_text())
        self.assertTrue(report["complete"])
        self.assertEqual((report["gpu_types"], len(report["scores"])), (["L4"], 32))
        for row in report["scores"]:
            self.assertEqual(row["scored"], row["expected"])
            self.assertTrue(all(row[k] for k in ("mean_seconds", "peak_memory_gb", "median_mass", "revision")))
        self.assertEqual(json.loads((self.root / "results/full/completeness_math_code.json").read_text()), report)
        # and the final analysis loads exactly these files
        settings = parse_settings(GO)
        banks = load_banks(self.root / "bank/full")
        scores = load_scores([self.root / "results/full" / d for d in PROBLEMS], banks, {d: needed_scores(settings, d) for d in PROBLEMS})
        self.assertEqual(len(scores.columns), 32)
        self.assertEqual(scores.unused, [])
        self.assertEqual({t for d in PROBLEMS for v in VERIFIERS if v != "Qwen3-8B" for t in scores.gpu_types(d, v, "avg")}, {"L4"})
        for verifier, revision in (("LLaDA-8B", full_run.DIFFUSION_REVISIONS["llada"]), ("Gemma-2-9B", "11c9b309abf73637e4b6f9a3fa1e92e615547819")):
            self.assertEqual(scores.info[(verifier, "T1")]["revisions"], {revision})

    def test_one_bank_alone_is_complete_on_its_own(self):
        self.score_all(domains=("math",))
        self.assertEqual(self.call("check", domains=("math",))[0], 0)
        self.assertEqual(self.call("check")[0], 1)

    def test_check_reports_what_is_missing(self):
        self.score_all()
        path = self.saved / "math/dream/full_T2_n1.jsonl"
        rows = read_jsonl(path)
        for change, fragment, scored in (
                (lambda: write_jsonl(path, rows[2:]), "2 of", len(rows) - 2),
                (lambda: write_jsonl(path, [{**rows[0], "status": "too_long", "score": None}, *rows[1:]]), "status 'too_long'", len(rows) - 1),
                (lambda: write_jsonl(path, [{**rows[0], "status": "error", "score": None}, *rows[1:]]), "status 'error'", len(rows) - 1),
                (lambda: write_jsonl(path, [{**r, "peak_mem_gb": None} for r in rows]), "rows failed, too long or incomplete", len(rows)),
                (lambda: write_jsonl(path, [{**r, "runtime_s": None} for r in rows]), "rows failed, too long or incomplete", len(rows)),
                (lambda: write_jsonl(path, [{**r, "template": "T1"} for r in rows]), "template T1, not T2", len(rows)),
                (lambda: write_jsonl(path, [{**r, "bank_hash": "0" * 64} for r in rows]), "bank_hash", len(rows)),
                (lambda: path.unlink(), "missing", 0)):
            change()
            code, text = self.call("check")
            self.assertEqual(code, 1, text)
            self.assertIn("NOT COMPLETE", text)
            self.assertIn(fragment, text)
            report = json.loads((self.saved / "completeness_math_code.json").read_text())
            self.assertFalse(report["complete"])
            entry, = [r for r in report["scores"] if (r["domain"], r["verifier"], r["template"]) == ("math", "Dream-7B", "T2")]
            self.assertEqual(entry["scored"], scored)

    def test_check_wants_one_gpu_type_and_its_records(self):
        self.score_all()
        beside = self.saved / "code/llada/full_T1_n1.jsonl.manifest.jsonl"
        good, = read_jsonl(beside)
        write_jsonl(beside, [{**good, "environment": {"gpu": L4}}])
        code, text = self.call("check")
        self.assertEqual(code, 1)
        self.assertIn("GPU types A100, L4", text)
        write_jsonl(beside, [good])
        (self.saved / "code/commit.txt").unlink()
        self.assertIn("commit.txt: missing", self.call("check")[1])

    def test_check_works_on_a_copy_without_the_saved_folder(self):
        self.score_all()
        self.assertEqual(self.call("check", saved=False)[0], 0)
        (self.root / "results/full/gpu.txt").unlink()
        self.assertIn("gpu.txt: missing", self.call("check", saved=False)[1])

    def test_repair_is_complete_without_the_reconstruction_score_and_qwen3_is_not(self):
        self.edit_json(full_run.SETTINGS, lambda d: d.update(mode="repair", llada_score="reconstruction"))
        self.commit("repair")
        self.score_all()
        code, text = self.call("check")
        self.assertEqual(code, 0, text)
        self.assertIn("notebook 12", text)
        self.edit_json(full_run.SETTINGS, lambda d: d.update(qwen3_scored=True))
        code, text = self.call("check")
        self.assertEqual(code, 1)
        self.assertIn("Qwen3-8B T1: not scored by this notebook", text)


class FormatTests(unittest.TestCase):
    """The pins and the file formats, against the pilot's real files."""

    def test_pins_are_the_pilots(self):
        from harness.diffusion.scorer import SPECS
        self.assertEqual(full_run.DIFFUSION_REVISIONS, {k: SPECS[k].revision for k in full_run.DIFFUSION_REVISIONS})
        for group in full_run.AR_GROUPS:
            full, pilot = (json.loads((REPO / f"configs/{kind}_{group}.json").read_text()) for kind in ("full", "pilot"))
            ar_run.validate_config(full)
            for template in ("T1", "T2"):
                manifest = json.loads((REPO / f"results/pilot/ar/pilot_{group}_{template}/manifest.json").read_text())
                for which, spec in full["models"].items():
                    meta = manifest["models"][which]
                    self.assertEqual((spec["model_id"], spec["revision"], spec["tokenizer_revision"]),
                                     (meta["requested"]["model_id"], meta["resolved_model_revision"], meta["resolved_tokenizer_revision"]))
                    self.assertRegex(spec["revision"], r"^[0-9a-f]{40}$")
                    self.assertEqual(VERIFIERS[BY_KEY[spec["model_id"]]].attention, spec.get("attn_implementation", "sdpa"))
            for config in (full, pilot):                # nothing else differs from the pilot's configuration
                for spec in config["models"].values():
                    spec["revision"] = spec["tokenizer_revision"] = None
            self.assertEqual(full, pilot)

    def test_every_verifier_is_in_one_group(self):
        wanted = {v for v, _ in needed_scores(parse_settings(GO), "math")}
        found = {v: full_run.step_of(v, "T1", "math", REPO).group for v in wanted}
        self.assertEqual(set(found.values()), set(GROUPS))
        self.assertIsNone(full_run.step_of("Qwen3-8B", "T1", "math", REPO))

    def test_verify_reads_the_pilots_real_runs(self):
        """The pilot's own files differ from a full-bank run only in the config file, which had no pins."""
        rows = read_jsonl(REPO / "bank/pilot/pilot_bank.jsonl")
        bank = SimpleNamespace(sha256=sha(REPO / "bank/pilot/pilot_bank.jsonl"), rows=rows,
                               index={r["candidate_id"]: k for k, r in enumerate(rows)})
        with tempfile.TemporaryDirectory() as tmp:
            for step in steps(("math",)):
                target = Path(tmp) / step.path
                target.parent.mkdir(parents=True, exist_ok=True)
                if step.is_ar:
                    shutil.copytree(REPO / f"results/pilot/ar/pilot_{step.group}_{step.template}", target)
                else:
                    for end in ("", ".manifest.jsonl"):
                        shutil.copyfile(REPO / f"results/pilot/{step.group}/pilot_{step.template}_n1.jsonl{end}", str(target) + end)
                foreign, missing = verify(step, REPO, tmp, bank)
                self.assertEqual(missing, [], step)
                self.assertEqual(len(foreign), 1 if step.is_ar else 0, foreign)
                self.assertTrue(all("config_file_sha256" in f for f in foreign))

    def test_the_real_diffusion_runner_with_a_mock_model(self):
        from harness.diffusion import run_score
        from harness.diffusion.scorer import SPECS, LLaDAScorer
        from harness.diffusion.tests.mocks import MASK, MockTokenizer, SlotMarkerModel
        spec = type(SPECS["llada"])(**{**SPECS["llada"].__dict__, "mask_id": MASK})
        timed = run_score.timed_score
        with tempfile.TemporaryDirectory() as tmp, \
                mock.patch.object(run_score, "load_scorer", lambda *a, **k: LLaDAScorer(SlotMarkerModel(), MockTokenizer(), spec, 1, device="cpu")), \
                mock.patch.object(run_score, "timed_score", lambda *a: (*timed(*a)[:2], 16.0)), \
                mock.patch.object(run_score, "environment_info", lambda device: {"gpu": L4}), contextlib.redirect_stderr(io.StringIO()):
            root = Path(tmp)
            write_study(root, seed=3, problems={"math": 6})
            step, bank = Step("math", "llada", "T1"), load_banks(root / "bank/full", ("math",))["math"]
            argv = full_run.command(step, "python", bank.path, bank.sha256, root / "scored")
            run_score.main(argv[3:])
            self.assertEqual(verify(step, REPO, root / "scored", bank), ([], []))
            path = root / "scored" / step.path
            write_jsonl(path, read_jsonl(path)[:-2])            # resumed after a crash: one file, one run record too few
            run_score.main(argv[3:])
            foreign, missing = verify(step, REPO, root / "scored", bank)
            self.assertEqual(foreign, [])
            self.assertIn("score it again in one go", missing[0])

    def test_nothing_names_a_label_file(self):
        text = (REPO / "harness/full_run.py").read_text(encoding="utf-8")
        for word in ("labels_test", "labels_validation", "align_labels", "load_validation_labels"):
            self.assertNotIn(word, text)

    def test_notebook_13(self):
        cells = json.loads((REPO / "notebooks/13_score_full.ipynb").read_text(encoding="utf-8"))["cells"]
        code = ["".join(c["source"]) for c in cells if c["cell_type"] == "code"]
        self.assertNotIn("labels_", "".join(code))
        first = [next(i for i, c in enumerate(code) if f"harness.full_run {name}" in c) for name in ("start", "score", "check")]
        self.assertEqual(first, sorted(first))
        for group in GROUPS:
            cell, = [c for c in code if f"--group {group} " in c]
            self.assertIn(f'if "{group}" in GROUPS:', cell)
            self.assertIn("assert _exit_code == 0", cell)
        self.assertTrue(all("python -m harness.full_run" in line or "-m" not in line.split() for c in code for line in c.splitlines()))
        self.assertIn("runtime.unassign()", code[-1])

    def test_notebook_12_follows_the_gpu_rule_for_full_bank_answers(self):
        cells = json.loads((REPO / "notebooks/12_backup_score.ipynb").read_text(encoding="utf-8"))["cells"]
        cell, = ["".join(c["source"]) for c in cells if "KEPT = " in "".join(c["source"])]
        with tempfile.TemporaryDirectory() as tmp:
            source = "\n".join(line.replace("!", "pass #", 1) if line.strip().startswith("!") else line for line in cell.splitlines())
            source = source.replace("results/full/gpu.txt", tmp + "/clone_gpu.txt").replace("/content/drive/MyDrive/cs544_full_scoring/full", tmp + "/notebook13")
            (Path(tmp) / "notebook13").mkdir()
            (Path(tmp) / "math/side").mkdir(parents=True)

            def bank_after(gpu, bank="bank/full/math/bank.jsonl"):
                import glob
                import os
                space = {"BANK": bank, "GPU": gpu, "OUT": tmp, "WHOLE_BANK": False, "os": os, "glob": glob}
                with contextlib.redirect_stdout(io.StringIO()):
                    exec(source, space)
                return space["BANK"]

            self.assertEqual(bank_after("L4", "bank/full/code/bank.jsonl"), "")
            self.assertTrue(bank_after("A100") and bank_after("L4"))            # nothing scored yet
            self.assertEqual((Path(tmp) / "math/gpu.txt").read_text(), "L4")
            (Path(tmp) / "math/side/llada.jsonl").write_text("{}")
            self.assertEqual((bank_after("A100"), bool(bank_after("L4"))), ("", True))
            (Path(tmp) / "math/side/llada.jsonl").unlink()
            (Path(tmp) / "notebook13/gpu.txt").write_text("A100\n")
            self.assertEqual((bank_after("L4"), bool(bank_after("A100"))), ("", True))
            (Path(tmp) / "notebook13/gpu.txt").unlink()
            (Path(tmp) / "clone_gpu.txt").write_text("L4\n")                    # committed by another account
            self.assertEqual((bank_after("A100"), bool(bank_after("L4"))), ("", True))

    def test_scoring_code_exists(self):
        for name in SCORING_CODE:
            self.assertTrue((REPO / name).is_file(), name)

    def test_numbers(self):
        self.assertFalse(full_run._number(True) or full_run._number(math.nan) or full_run._number(None))
        self.assertTrue(full_run._number(0.0))


if __name__ == "__main__":
    unittest.main()
