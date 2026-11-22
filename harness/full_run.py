"""Score the full bank: the guards, one command per model group, and the completeness check (PLAN.md §4, §5, §7, §9).

    python -m harness.full_run start --domains math code --saved <Drive folder>
    python -m harness.full_run score --group llama_gemma --domains math code --saved <Drive folder>
    python -m harness.full_run check --domains math code

Notebook 13 runs these. Every command reads the banks and the score files only, never a label.

start and score refuse to do anything (exit code 2) unless:
- configs/final_models.json is valid (analysis.final_settings), its mode is not "stopped", and it and the final
  model record it names in decisions/ are committed and merged into main (PLAN.md §7);
- each bank is frozen (PLAN.md §2): bank.jsonl and manifest.json are committed, merged into main and unchanged, the
  manifest records bank.jsonl's hash, bank.jsonl has never changed since its first commit (a corrected bank is a new
  version) and, for math, the error tags are in (they go in before the freeze);
- the GPU is an A100 or an L4, and the type the first scores were made on (gpu.txt and the score files themselves), for
  every verifier and both banks (decisions/2026-11-22-scoring-one-gpu-type.md);
- the checkout is the commit a bank's first scores were made with, and the scoring code is the same for both banks;
- no saved score file has another bank hash, revision, template, slot count, prefix, precision, attention or batch size.

score runs harness.run or harness.diffusion.run_score with the pilot's settings, T1 then T2, and copies each finished
run to --saved. A finished run is skipped; an unfinished one is scored again from the start, because every diffusion
row must come from one recorded run (analysis/final_data.py).

check needs no GPU and no git. It exits with 1 unless every verifier x template x candidate of the given banks is
there, with seconds, memory and one GPU type, and analysis.final_data.load_scores accepts the files.
"""

from __future__ import annotations

import argparse
import json
import math
import shutil
import statistics
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

from analysis.final_data import (DataError, _file_gpus, classify, find_score_files, gpu_type, load_bank, load_banks,
                                 load_scores, read_jsonl, sha256_file)
from analysis.final_settings import ANSWER_PREFIX, ANSWER_SLOTS, BY_KEY, DOMAINS, load_settings, needed_scores
from bank.split_full import LABEL_FIELDS

SETTINGS = "configs/final_models.json"
MAIN = "origin/main"
GPU_TYPES = ("A100", "L4")
GPU_FILE, COMMIT_FILE = "gpu.txt", "commit.txt"     # gpu.txt: one for both banks; commit.txt: one per bank
TEMPLATES = ("T1", "T2")        # no T3 on the full bank (PLAN.md §4)
AR_GROUPS = ("llama_gemma", "qwen_mistral", "small")    # configs/full_<group>.json
DIFFUSION_REVISIONS = {         # the pins of harness/diffusion/scorer.py (a test compares them); no torch needed here
    "llada": "08b83a6feb34df1a6011b80c3c00c7563e963b07",
    "dream": "05334cb9faaf763692dcf9d8737c642be2b2a6ae",
}
GROUPS = AR_GROUPS + tuple(DIFFUSION_REVISIONS)
SCORING_CODE = ("harness/run.py", "harness/ar.py", "harness/data.py", "harness/prompts.py", "harness/diffusion/run_score.py",
                "harness/diffusion/scorer.py", "harness/diffusion/interface.py", "harness/diffusion/requirements/llada.txt",
                "harness/diffusion/requirements/dream.txt", "requirements.txt", *(f"configs/full_{g}.json" for g in AR_GROUPS))


class Refused(Exception):
    """Scoring must not start or go on."""


@dataclass(frozen=True)
class Step:
    """One run: a model group, a template and a bank. `path` is relative to the results folder."""
    domain: str
    group: str
    template: str

    @property
    def is_ar(self) -> bool:
        return self.group in AR_GROUPS

    @property
    def path(self) -> str:
        if self.is_ar:
            return f"{self.domain}/ar/full_{self.group}_{self.template}"
        return f"{self.domain}/{self.group}/full_{self.template}_n{ANSWER_SLOTS}.jsonl"

    @property
    def config(self) -> str:
        return f"configs/full_{self.group}.json"

    def score_files(self, results) -> list[Path]:
        path = Path(results) / self.path
        return [path / "scores_A.jsonl", path / "scores_B.jsonl"] if self.is_ar else [path]


def sidecar(path) -> Path:
    return Path(str(path) + ".manifest.jsonl")


def steps(domains, groups=GROUPS) -> list[Step]:
    return [Step(d, g, t) for g in groups for d in domains for t in TEMPLATES]


def command(step: Step, python, bank, bank_hash: str, out) -> list[str]:
    """The exact scoring command of a step, with the pilot's settings. Paths are relative to the repository root."""
    path = (Path(out) / step.path).as_posix()
    if step.is_ar:
        return [python, "-m", "harness.run", "--config", step.config, "--input", bank, "--template", step.template,
                "--output-dir", path, "--device", "cuda:0", "--dtype", "bf16", "--no-load-in-4bit"]
    return [python, "-m", "harness.diffusion.run_score", "--model", step.group, "--n-slots", str(ANSWER_SLOTS),
            "--batch-size", "1", "--answer-prefix", ANSWER_PREFIX, "--template", step.template, "--items", bank,
            "--bank-hash", bank_hash, "--out", path]


# ---------------------------------------------------------------- git: merged record, frozen bank, fixed commit

def git(root, *args) -> subprocess.CompletedProcess:
    return subprocess.run(["git", "-C", str(root), *args], capture_output=True, text=True)


def require_committed(root, path: str, what: str) -> None:
    if not (Path(root) / path).is_file():
        raise Refused(f"{path} is missing: {what}")
    if git(root, "cat-file", "-e", f"HEAD:{path}").returncode:
        raise Refused(f"{path} is not committed: {what}")
    if git(root, "diff", "--quiet", "HEAD", "--", path).returncode:
        raise Refused(f"{path} differs from the committed file")


def require_merged(root) -> str:
    """HEAD must be on main, so everything committed in it has been merged. Returns the commit."""
    head = git(root, "rev-parse", "HEAD")
    if head.returncode:
        raise Refused("this is not a git checkout, so nothing shows that the model record and the banks are merged")
    if git(root, "merge-base", "--is-ancestor", "HEAD", MAIN).returncode:
        raise Refused(f"the checked-out commit {head.stdout.strip()[:12]} is not part of {MAIN}: the final model record, "
                      "the settings and the banks must be merged first (PLAN.md §2, §7)")
    return head.stdout.strip()


def check_record(root):
    """The model settings, if they and the final model record are merged (PLAN.md §7)."""
    require_committed(root, SETTINGS, "copy configs/final_models.example.json and set what the final model record says "
                                      "(analysis/README.md), then merge it")
    try:
        settings = load_settings(Path(root) / SETTINGS)
    except (ValueError, TypeError) as exc:
        raise Refused(f"{SETTINGS}: {exc}") from exc
    record = Path(settings.record)
    if record.parent.as_posix() != "decisions" or record.suffix != ".md" or record.name == "README.md":
        raise Refused(f"{SETTINGS}: record {settings.record!r} is not a decision record in decisions/")
    require_committed(root, record.as_posix(), "no full-bank answer is scored before the final model record is merged "
                                               "(PLAN.md §7)")
    if settings.mode == "stopped":
        raise Refused(f"{SETTINGS}: mode 'stopped' ({settings.reason}): H1 is not run and nothing is analyzed (PLAN.md §7)")
    return settings


def check_bank(root, banks: str, domain: str):
    """The bank, if it is frozen: committed, merged, hashed, unchanged since its first commit, without labels."""
    folder = f"{Path(banks).as_posix()}/{domain}"
    path, manifest_path = f"{folder}/bank.jsonl", f"{folder}/manifest.json"
    for name in (path, manifest_path):
        if git(root, "cat-file", "-e", f"HEAD:{name}").returncode and git(root, "cat-file", "-e", f"{MAIN}:{name}").returncode == 0:
            raise Refused(f"{name} is on {MAIN} but not at the checked-out commit, which the other bank's saved scores keep: "
                          f"score {domain} on its own (DOMAINS = [\"{domain}\"])")
        require_committed(root, name, "score a bank only after its freeze is merged (PLAN.md §2); to score the other "
                                      "bank first, run it alone")
    try:
        bank = load_bank(Path(root) / path, domain, manifest_path=Path(root) / manifest_path)
        manifest = json.loads((Path(root) / manifest_path).read_text(encoding="utf-8"))
    except (DataError, ValueError) as exc:
        raise Refused(str(exc)) from exc
    leaked = {k for r in bank.rows for k in r} & LABEL_FIELDS
    if leaked or len(bank.rows) != len(bank.file_ids):
        raise Refused(f"{path}: label fields {sorted(leaked)}" if leaked else f"{path} holds rows of another domain")
    if domain == "math" and not manifest.get("math_error_tags"):
        raise Refused(f"{manifest_path}: no error tags yet. They are added before the bank freeze, so this bank is not "
                      "frozen (PLAN.md §2; notebook 11, section 5)")
    commits = git(root, "log", "--format=%H", "HEAD", "--", path).stdout.split()
    if len(commits) != 1:
        raise Refused(f"{path} was changed in {len(commits)} commits; a frozen bank is committed once and never "
                      "changes, and a corrected bank is a new version in a new folder (PLAN.md §2)")
    if git(root, "diff", "--quiet", "HEAD", MAIN, "--", path).returncode:
        raise Refused(f"{path} changed on {MAIN} after this commit; a corrected bank is a new version (PLAN.md §2)")
    return bank


def check_commit(root, saved, domains, head: str) -> None:
    """Resumed scores keep their commit, and both banks are scored with the same, committed scoring code."""
    edited = git(root, "diff", "--name-only", "HEAD", "--", *SCORING_CODE).stdout.split()
    if edited:
        raise Refused(f"uncommitted changes in the scoring code: {' '.join(edited)}")
    for domain in DOMAINS:
        path = Path(saved) / domain / COMMIT_FILE
        if not path.is_file():
            continue
        commit = path.read_text(encoding="utf-8").strip()
        if domain in domains and commit != head:
            raise Refused(f"the saved {domain} scores were made at commit {commit[:12]}, this is {head[:12]}. Check that "
                          f"commit out (notebook 13 does), or delete {saved} to start over")
        changed = git(root, "diff", "--name-only", commit, "HEAD", "--", *SCORING_CODE)
        if changed.returncode or changed.stdout.strip():
            raise Refused(f"the scoring code changed since the saved {domain} scores ({commit[:12]}): "
                          f"{' '.join(changed.stdout.split()) or changed.stderr.strip()}. Both banks are scored with the "
                          f"same code: undo that change on {MAIN}, or delete {saved} to start over")


# ---------------------------------------------------------------- one GPU type

def current_gpu() -> str:
    """Name of this machine's GPU, "" without one."""
    try:
        shown = subprocess.run(["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"], capture_output=True, text=True)
    except OSError:
        return ""
    return (shown.stdout.splitlines() or [""])[0].strip() if shown.returncode == 0 else ""


def score_paths(folder) -> list[Path]:
    """Score files under a folder, without copies to Drive that were cut off (.part)."""
    folder = Path(folder)
    found = find_score_files([folder]) if folder.is_dir() else []
    return [p for p in found if not any(part.endswith(".part") for part in p.parts)]


def file_types(path) -> set:
    return {gpu_type(name) for name in _file_gpus(Path(path))}


def recorded_gpus(folder) -> list[tuple]:
    """(where, GPU type) of gpu.txt and of every score file under a results folder."""
    note = Path(folder) / GPU_FILE
    out = [(note.as_posix(), note.read_text(encoding="utf-8").strip())] if note.is_file() else []
    return out + [(p.as_posix(), t) for p in score_paths(folder) for t in sorted(file_types(p))]


def check_gpu(root, saved, name: str, also=()) -> str:
    """The GPU type, if it may be used. `also`: gpu.txt files of other full-bank scores (notebook 12), which count once
    something has been scored beside them."""
    mine = gpu_type(name) if name else "no GPU"
    if mine not in GPU_TYPES:
        raise Refused(f"{name or 'no GPU'}: pick an A100 or an L4 (Runtime > Change runtime type)")
    recorded = recorded_gpus(saved)
    for path in map(Path, also):
        if path.is_file() and any(path.parent.rglob("*.jsonl")):
            recorded.append((str(path), path.read_text(encoding="utf-8").strip()))
    shown = git(root, "show", f"{MAIN}:results/full/{GPU_FILE}")
    if shown.returncode == 0:
        recorded.append((f"results/full/{GPU_FILE} on {MAIN}", shown.stdout.strip()))
    for where, other in recorded:
        if other not in GPU_TYPES:
            raise Refused(f"{where} does not say that an A100 or an L4 made it ({other}); delete that run to score it again")
        if other != mine:
            raise Refused(f"this runtime has an {mine}, but {where} says {other}. The full bank is scored on one GPU type "
                          "for every verifier and both banks (decisions/2026-11-22-scoring-one-gpu-type.md). Choose "
                          f"{other} (Runtime > Change runtime type) and run again, or start all the full-bank scoring over on "
                          f"{mine}: delete {saved} and any other full-bank scores named above")
    return mine


# ---------------------------------------------------------------- is one run complete, and made with the right settings?

def _number(x) -> bool:
    return isinstance(x, (int, float)) and not isinstance(x, bool) and math.isfinite(x)


def _rows(path: Path, missing: list):
    """Rows of a score file, or [] when it is missing or cut off."""
    if not path.is_file():
        missing.append(f"{path}: missing")
        return []
    try:
        return read_jsonl(path)
    except (DataError, UnicodeDecodeError) as exc:
        missing.append(f"{path}: not readable ({exc})")
        return []


def _check_rows(path, rows, wanted: dict, bank, done: str, numbers, foreign: list, missing: list) -> None:
    """Identity of every row (foreign if wrong), and one finished row per candidate (missing if not)."""
    for key, value in wanted.items():
        other = {str(r.get(key)) for r in rows} - {str(value)}
        if other:
            foreign.append(f"{path}: {key} {', '.join(sorted(other))}, not {value}")
    ids = [r.get("candidate_id") for r in rows]
    if set(ids) - set(bank.index):
        foreign.append(f"{path}: {len(set(ids) - set(bank.index))} candidates that are not in the bank")
    if len(set(ids)) != len(ids):
        foreign.append(f"{path}: {len(ids) - len(set(ids))} duplicate rows")
    absent = len(set(bank.index) - set(ids))
    if absent:
        missing.append(f"{path}: {absent} of {len(bank.rows)} candidates have no row")
    bad = [r for r in rows if r.get("status") != done or not all(_number(r.get(k)) for k in numbers)]
    if bad:
        missing.append(f"{path}: {len(bad)} rows failed, too long or incomplete (e.g. {bad[0].get('candidate_id')}: status "
                       f"{bad[0].get('status')!r})")


def verify(step: Step, root, results, bank) -> tuple[list, list]:
    """(foreign, missing) of one run: what was made with other settings, and what is not there or not finished.
    Both empty means the run is complete. A run that never finished is only `missing`."""
    foreign, missing = [], []
    (_verify_ar if step.is_ar else _verify_diffusion)(step, Path(root), Path(results), bank, foreign, missing)
    return foreign, missing


def _verify_ar(step, root, results, bank, foreign, missing) -> None:
    run = results / step.path
    try:
        m = json.loads((run / "manifest.json").read_text(encoding="utf-8"))
    except (OSError, ValueError):
        missing.append(f"{run}: no readable run manifest")
        return
    if m.get("status") != "completed":
        missing.append(f"{run}: the run is {m.get('status')!r}, not 'completed'")
        return
    config = json.loads((root / step.config).read_text(encoding="utf-8"))
    wanted = {"input_sha256": bank.sha256, "template": step.template, "limit": None, "compute_dtype": "torch.bfloat16",
              "batch_size": 1, "config_file_sha256": sha256_file(root / step.config)}
    foreign += [f"{run}: {k} {m.get(k)}, not {v}" for k, v in wanted.items() if m.get(k) != v]
    for which, spec in config["models"].items():
        model, path = (m.get("models") or {}).get(which) or {}, run / f"scores_{which}.jsonl"
        expected = {"resolved_model_revision": spec["revision"], "resolved_tokenizer_revision": spec["tokenizer_revision"],
                    "actual_quantization": None, "attention_implementation": spec.get("attn_implementation", "sdpa")}
        foreign += [f"{run}: {spec['model_id']}: {k} {model.get(k)}, not {v}" for k, v in expected.items()
                    if k not in model or model[k] != v]
        rows = _rows(path, missing)
        if path.is_file() and (m.get("outputs_sha256") or {}).get(path.name) != sha256_file(path):
            foreign.append(f"{path}: does not match the hash in its run manifest")
        _check_rows(path, rows, {"bank_sha256": bank.sha256, "model_id": spec["model_id"], "model_revision": spec["revision"],
                                 "template": step.template}, bank, "completed", ("score", "p_yes", "p_no", "forward_seconds"),
                    foreign, missing)
        if not _number(model.get("peak_gpu_allocated_bytes")):
            missing.append(f"{run}: {spec['model_id']}: no peak memory recorded")


def _verify_diffusion(step, root, results, bank, foreign, missing) -> None:
    path = results / step.path
    rows, runs = _rows(path, missing), _rows(sidecar(path), missing)
    wanted = {"bank_hash": bank.sha256, "model": step.group, "revision": DIFFUSION_REVISIONS[step.group],
              "template": step.template, "n_slots": ANSWER_SLOTS, "answer_prefix": ANSWER_PREFIX}
    _check_rows(path, rows, wanted, bank, "ok", ("score", "p_yes", "p_no", "runtime_s", "peak_mem_gb"), foreign, missing)
    for run in runs:
        foreign += [f"{sidecar(path)}: {k} {run.get(k)}, not {v}" for k, v in {**wanted, "batch_size": 1}.items()
                    if run.get(k) != v]
    recorded = sum((r.get("completed") or {}).get("ok", 0) for r in runs)
    if runs and recorded != sum(r.get("status") == "ok" for r in rows):
        missing.append(f"{path}: its runs record {recorded} scored rows, the file has another number; score it again in one go")


def unexpected_files(results, domains) -> list[str]:
    """Score files that no step of this scoring writes (T3, a smoke run, another model). Reconstruction scores may be here."""
    expected = {p for s in steps(domains) for p in s.score_files(results)}
    out = []
    for domain in domains:
        for path in score_paths(Path(results) / domain):
            if path in expected:
                continue
            try:
                kinds = {classify(r)[0] for r in read_jsonl(path)}
            except DataError as exc:
                kinds = {str(exc)}
            if kinds != {"recon"}:
                out.append(f"{path}: not a file of this scoring")
    return out


def check_saved(root, results, domains, banks, saved) -> None:
    """Every score file that is already there must be complete and made with the right settings. Only finished runs
    are ever saved, so anything else did not come from this notebook."""
    problems = unexpected_files(results, domains)
    for step in steps(domains):
        if step.is_ar and (Path(results) / step.path).exists() or any(p.exists() for p in step.score_files(results)):
            foreign, missing = verify(step, root, results, banks[step.domain])
            problems += foreign + missing
    if problems:
        raise Refused("saved score files that this run cannot use:\n  " + "\n  ".join(problems[:12])
                      + f"\nDelete these runs from {saved} to score them again, or the whole folder to start over")


# ---------------------------------------------------------------- saving (Google Drive) and resuming

def remove(path: Path) -> None:
    shutil.rmtree(path) if path.is_dir() else path.unlink(missing_ok=True)


def copy_whole(src: Path, dst: Path) -> None:
    """Copy a file or folder. It gets its real name only when the copy is complete."""
    part = dst.with_name(dst.name + ".part")
    dst.parent.mkdir(parents=True, exist_ok=True)
    remove(part)
    (shutil.copytree if src.is_dir() else shutil.copy2)(src, part)
    remove(dst)
    part.rename(dst)


def restore(saved, results) -> None:
    """Replace the local results with the saved ones: the saved folder is the record, not what came with the clone."""
    saved, results = Path(saved), Path(results)
    if results.exists():
        remove(results)
    if saved.is_dir():
        shutil.copytree(saved, results, ignore=shutil.ignore_patterns("*.part"))
    results.mkdir(parents=True, exist_ok=True)


def copy_step(step: Step, src, dst) -> None:
    """One finished run, its sidecar first, so a score file is never there without it."""
    if not step.is_ar:
        copy_whole(sidecar(Path(src) / step.path), sidecar(Path(dst) / step.path))
    copy_whole(Path(src) / step.path, Path(dst) / step.path)


def save(step: Step, results, saved, gpu: str, head: str) -> None:
    """Copy a finished run to the saved folder, after gpu.txt and commit.txt, which the first saved run writes."""
    for name, value in ((GPU_FILE, gpu), (f"{step.domain}/{COMMIT_FILE}", head)):
        for folder in (Path(saved), Path(results)):
            path = folder / name
            if path.is_file() and path.read_text(encoding="utf-8").strip() != value:
                raise Refused(f"{path} says {path.read_text(encoding='utf-8').strip()}, this run is {value}: another session "
                              f"saved scores first, so {step.path} is not saved")
            if not path.is_file():
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text(value + "\n", encoding="utf-8", newline="\n")
    copy_step(step, results, saved)
    print("saved:", step.path, flush=True)


# ---------------------------------------------------------------- commands

@dataclass
class Context:
    settings: object
    banks: dict
    gpu: str
    head: str


def guard(args, gpu_name: str) -> Context:
    """Every check that comes before a model is loaded. Raises Refused."""
    root = Path(args.root)
    if not Path(args.saved).is_dir():
        raise Refused(f"{args.saved} does not exist, so finished runs could not be saved there and earlier ones would not be "
                      "seen: mount Google Drive first (the setup cell)")
    head = require_merged(root)
    settings = check_record(root)
    banks = {d: check_bank(root, args.banks, d) for d in args.domains}
    gpu = check_gpu(root, args.saved, gpu_name, args.also_gpu)
    check_commit(root, args.saved, args.domains, head)
    return Context(settings, banks, gpu, head)


def notes(settings) -> list[str]:
    out = []
    if settings.llada_score == "reconstruction":
        out.append("LLaDA's score on math is the reconstruction score (repair). Its verdict scores are run here all the same; "
                   "the reconstruction score of the whole math bank comes from notebook 12 (WHOLE_BANK = True).")
    if settings.qwen3_scored:
        out.append("The record asks for Qwen3-8B. harness.run cannot turn its thinking off yet, so it is not scored here and the "
                   "completeness check will list it as missing.")
    return out


def start(args, gpu_name=current_gpu) -> int:
    root, results = Path(args.root), Path(args.root) / args.out
    c = guard(args, gpu_name())
    restore(args.saved, results)
    check_saved(root, results, args.domains, c.banks, args.saved)
    print(f"final model record: {c.settings.record} (mode {c.settings.mode})")
    for d, bank in c.banks.items():
        print(f"{d}: frozen, {len(bank.rows)} answers, sha256 {bank.sha256}")
    print(f"GPU type {c.gpu}, commit {c.head[:12]}")
    for note in notes(c.settings):
        print("NOTE:", note)
    if git(root, "diff", "--quiet", "HEAD", MAIN, "--", SETTINGS, c.settings.record).returncode:
        print(f"NOTE: {SETTINGS} or the model record changed on {MAIN} after this commit. Read what changed before going on.")
    for step in steps(args.domains):
        print("  done " if not any(verify(step, root, results, c.banks[step.domain])) else "  to do", step.path)
    return 0


def run_command(argv, cwd) -> int:
    return subprocess.run(argv, cwd=cwd).returncode


def score(args, gpu_name=current_gpu, execute=run_command) -> int:
    root, results = Path(args.root), Path(args.root) / args.out
    c = guard(args, gpu_name())
    for step in steps(args.domains, [args.group]):
        bank, local, kept = c.banks[step.domain], results / step.path, Path(args.saved) / step.path
        if kept.exists() and any(verify(step, root, results, bank)):    # another session finished it meanwhile
            copy_step(step, args.saved, results)
        foreign, missing = verify(step, root, results, bank)
        if foreign:
            raise Refused("\n  ".join([f"{step.path} was made with other settings; delete it to score it again:", *foreign[:8]]))
        if missing:
            remove(local)       # an unfinished run is never continued
            remove(sidecar(local))
            local.parent.mkdir(parents=True, exist_ok=True)
            print("scoring:", step.path, flush=True)
            code = execute(command(step, args.python, f"{args.banks}/{step.domain}/bank.jsonl", bank.sha256, args.out), root)
            foreign, missing = verify(step, root, results, bank)
            if foreign or missing:
                print("\n  ".join([f"{step.path} did not finish (exit code {code}) and is not saved:", *(foreign + missing)[:8]]))
                return 1
        else:
            print("already done:", step.path)
        others = set().union(*map(file_types, step.score_files(results))) - {c.gpu}
        if others:
            raise Refused(f"{step.path} was scored on {', '.join(sorted(others))}, not on {c.gpu}; it is not saved")
        if any(verify(step, root, args.saved, bank)):
            save(step, results, args.saved, c.gpu, c.head)
    return 0


# ---------------------------------------------------------------- completeness check

def column_entry(domain, verifier, template, step, results, bank) -> dict:
    """One row of the table: what one verifier's scores with one template on one bank look like."""
    entry = {"domain": domain, "verifier": verifier, "template": template, "expected": len(bank.rows), "scored": 0,
             "mean_seconds": None, "peak_memory_gb": None, "median_mass": None, "gpu": None, "revision": None}
    if step is None:
        return entry
    rows, path = [], None
    for path in step.score_files(results):
        rows = [r for r in _rows(path, []) if BY_KEY.get(classify(r)[1]) == verifier]
        if rows:
            break
    good = [r for r in rows if r.get("status") in ("ok", "completed") and _number(r.get("score"))]
    seconds = [s for i, r in enumerate(rows) for s in [r.get("forward_seconds") if step.is_ar else r.get("runtime_s")]
               if i and _number(s)]     # the first row of a run is cold
    mass = [r["yes_no_mass"] for r in good if _number(r.get("yes_no_mass"))]
    entry.update(scored=len({r["candidate_id"] for r in good} & set(bank.index)))
    if rows:
        if step.is_ar and (path.parent / "manifest.json").is_file():
            model = json.loads((path.parent / "manifest.json").read_text(encoding="utf-8"))["models"].get(path.stem[-1], {})
            peak = model.get("peak_gpu_allocated_bytes")
            entry["peak_memory_gb"] = peak / 1e9 if _number(peak) else None
        else:
            entry["peak_memory_gb"] = max((r["peak_mem_gb"] for r in good if _number(r.get("peak_mem_gb"))), default=None)
        entry.update(mean_seconds=statistics.fmean(seconds) if seconds else None, median_mass=statistics.median(mass) if mass else None,
                     gpu="/".join(sorted(_file_gpus(path))), revision=rows[0].get("model_revision") or rows[0].get("revision"))
    return entry


def step_of(verifier: str, template: str, domain: str, root) -> Step | None:
    """The step that scores a verifier; None if no group does (Qwen3-8B)."""
    for group in AR_GROUPS:
        config = json.loads((Path(root) / Step(domain, group, template).config).read_text(encoding="utf-8"))
        if verifier in {BY_KEY.get(s["model_id"]) for s in config["models"].values()}:
            return Step(domain, group, template)
    return next((Step(domain, g, template) for g in DIFFUSION_REVISIONS if BY_KEY[g] == verifier), None)


def completeness(root, banks_dir, out, domains) -> dict:
    """Everything the analysis needs from this scoring, for the given banks. Reads no label."""
    root, results = Path(root), Path(root) / out
    settings = load_settings(root / SETTINGS)
    banks = load_banks(root / banks_dir, domains)
    needed = {d: [(v, c) for v, c in needed_scores(settings, d) if c in TEMPLATES] for d in domains}
    problems, table = unexpected_files(results, domains), []
    for d in domains:
        if not (results / d / COMMIT_FILE).is_file():
            problems.append(f"{results / d / COMMIT_FILE}: missing")
        for v, t in needed[d]:
            step = step_of(v, t, d, root)
            table.append(column_entry(d, v, t, step, results, banks[d]))
            if step is None:
                problems.append(f"{d}: {v} {t}: not scored by this notebook")
    for step in steps(domains):
        foreign, missing = verify(step, root, results, banks[step.domain])
        problems += foreign + missing
    types = sorted({t for _, t in recorded_gpus(results)})
    if not (results / GPU_FILE).is_file():
        problems.append(f"{results / GPU_FILE}: missing")
    if len(types) != 1 or types[0] not in GPU_TYPES:
        problems.append(f"GPU types {', '.join(types) or 'none'}: one type, {' or '.join(GPU_TYPES)}, is required")
    try:
        load_scores([results / d for d in domains], banks, needed)
    except DataError as exc:
        problems.append(f"analysis.final_data does not accept the scores: {exc}")
    return {"complete": not problems, "domains": list(domains), "mode": settings.mode, "gpu_types": types,
            "banks": {d: {"sha256": b.sha256, "answers": len(b.rows)} for d, b in banks.items()},
            "commits": {d: (results / d / COMMIT_FILE).read_text(encoding="utf-8").strip() for d in domains
                        if (results / d / COMMIT_FILE).is_file()},
            "notes": notes(settings), "problems": problems, "scores": table}


def show(report: dict) -> str:
    def f(x, digits):
        return "" if x is None else f"{x:.{digits}f}"
    lines = ["domain  verifier       template  scored        s/answer  peak GB  median mass  GPU"]
    for r in report["scores"]:
        lines.append(f"{r['domain']:<7} {r['verifier']:<14} {r['template']:<9} {r['scored']:>5}/{r['expected']:<7} "
                     f"{f(r['mean_seconds'], 4):>8}  {f(r['peak_memory_gb'], 1):>7}  {f(r['median_mass'], 3):>11}  {r['gpu'] or ''}")
    lines += ["NOTE: " + n for n in report["notes"]]
    lines += ["PROBLEM: " + p for p in report["problems"][:30]]
    lines.append("COMPLETE: every verifier x template x candidate is there" if report["complete"] else
                 f"NOT COMPLETE: {len(report['problems'])} problems")
    return "\n".join(lines)


def check(args) -> int:
    results = Path(args.root) / args.out
    if args.saved and (Path(args.saved) / GPU_FILE).is_file():
        restore(args.saved, results)
    report = completeness(args.root, args.banks, args.out, tuple(args.domains))
    print(show(report))
    name = f"completeness_{'_'.join(args.domains)}.json"
    for folder in [results] + ([Path(args.saved)] if args.saved else []):
        folder.mkdir(parents=True, exist_ok=True)
        (folder / name).write_text(json.dumps(report, indent=1, sort_keys=True) + "\n", encoding="utf-8", newline="\n")
    return 0 if report["complete"] else 1


def main(argv=None, gpu_name=current_gpu, execute=run_command) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = ap.add_subparsers(dest="command", required=True)
    for name in ("start", "score", "check"):
        p = sub.add_parser(name)
        p.add_argument("--root", default=".", help="repository root; --banks and --out are relative to it")
        p.add_argument("--domains", nargs="+", default=list(DOMAINS), choices=DOMAINS)
        p.add_argument("--banks", default="bank/full", help="folder with one subfolder per domain")
        p.add_argument("--out", default="results/full", help="local results; start replaces them with the saved ones")
        p.add_argument("--saved", required=name != "check", help="folder finished runs are copied to (Google Drive)")
        if name != "check":
            p.add_argument("--also-gpu", nargs="*", default=[], help="gpu.txt of other full-bank scores that must be the same type")
        if name == "score":
            p.add_argument("--group", required=True, choices=GROUPS)
            p.add_argument("--python", default=sys.executable, help="Python of the group's environment (LLaDA, Dream)")
    args = ap.parse_args(argv)
    args.domains = [d for d in DOMAINS if d in args.domains]
    try:
        if args.command == "check":
            return check(args)
        return start(args, gpu_name) if args.command == "start" else score(args, gpu_name, execute)
    except (Refused, DataError, ValueError, OSError) as exc:
        print(f"REFUSED: {exc}")
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
