"""Validation step (PLAN.md §5 "On validation"): choose the pairs, set every threshold, write the analysis manifest.

    python -m analysis.final_validation --settings configs/final_models.json --banks bank/full \\
        --scores results/full/math results/full/code --out results/final
    python -m analysis.final_validation --check results/final/analysis_manifest.json

Reads validation labels only (labels_validation.jsonl of each domain); it has no way to open a test-label file.
The manifest is the preregistration: commit it, with the analysis code, before test labels are joined.
--check recomputes everything from the committed inputs and compares it with the manifest.

What it does, on math for the final test and per domain for everything else:
- thresholds at 80, 90 and 95% pair TAR for every pair of the plan (analysis.metrics.pair_thresholds), and at 90%
  for every single score; trade-off thresholds for the final test's pairs;
- the AR+AR and the AR+diffusion pair with the lowest validation FAR at 90% pair TAR (ties: higher mean AUROC of
  the two members, then alphabetical), and how often each pair wins when validation problems are resampled;
- the best single verifier, the cost check (1.5x), and LLaDA's gate A re-check, which is reported and never applied.
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import subprocess
from dataclasses import asdict
from importlib import import_module
from pathlib import Path

import numpy as np

from analysis.final_data import (DataError, Frame, Scores, bank_paths, fail, find_score_files, load_banks, load_scores,
                                 load_validation_labels, make_frame, read_jsonl, sha256_bytes, sha256_file)
from analysis.final_settings import (DIFFUSION, DOMAINS, H1_DOMAIN, RULES, SMALL_REFERENCE, TRADEOFF_TARS,
                                     Member, Settings, load_settings, needed_scores, plan_pairs, primary_members,
                                     pseudo_pair_name, single_members, with_role)
from analysis.final_stats import TOL, Boot, accepts, auroc_boot, fast_pair_thresholds, interval
from analysis.metrics import auroc, pair_thresholds, threshold_at_tar

VALIDATION_STREAM = 1       # validation resamples use the generator seeded [544, 1], test resamples 544, so the
                            # two are independent
JOINS = ("join", "rerun")                   # events of the join log that mean test labels were read
JOIN_LOG = "results/final/join_log.jsonl"   # one log for the whole repository, whatever --out a run used
SIDE_CODE = ("analysis/reconstruction.py", "analysis/pilot.py")     # frozen too when the side experiment is in the
                                                                    # manifest (the first imports the second)
# Frozen with the manifest. Figures and tables (analysis/final_report.py) only display results and are not frozen.
CODE_FILES = tuple(f"analysis/{name}.py" for name in (
    "metrics", "final_settings", "final_data", "final_stats", "overlap", "final_validation", "final_evaluate"))


def tar_key(tar: float) -> str:
    return f"{tar:.2f}"


def all_members(settings: Settings, domain: str) -> list[Member]:
    members = single_members(settings, domain) + [m for p in plan_pairs(settings, domain) for m in p.members]
    return sorted(set(members), key=lambda m: m.label)


# ---------------------------------------------------------------- thresholds

def pair_entry(frame: Frame, pair, tar: float) -> dict:
    """Thresholds of a pair at one pair TAR, with its validation TAR and FAR counts."""
    x, y, c, w = frame.member(pair.a), frame.member(pair.b), frame.correct, frame.wrong
    ta, tb, r = pair_thresholds(x[c], y[c], tar)
    acc = accepts(x, y, (ta, tb))
    return {"a": ta, "b": tb, "individual_tar": r, "n_correct": int(c.sum()), "accepted_correct": int(acc[c].sum()),
            "n_wrong": int(w.sum()), "accepted_wrong": int(acc[w].sum()),
            "tar": float(acc[c].mean()), "far": float(acc[w].mean())}


def single_entry(frame: Frame, member, boot: Boot, rules) -> dict:
    x, c, w = frame.member(member), frame.correct, frame.wrong
    t = threshold_at_tar(x[c], rules.pair_tar)
    return {"verifier": member.verifier, "column": member.column, "threshold": t,
            "tar": float(np.mean(x[c] >= t)), "far": float(np.mean(x[w] >= t)),
            "accepted_wrong": int((x[w] >= t).sum()), "n_wrong": int(w.sum()), "n_correct": int(c.sum()),
            "auroc": auroc(x, frame.labels), "auroc_ci": interval(auroc_boot(boot, x, frame.labels))}


# ---------------------------------------------------------------- selection

def mean_auroc(singles: dict, pair) -> float:
    return (singles[pair.a.label]["auroc"] + singles[pair.b.label]["auroc"]) / 2


def select(pairs, entries: dict, singles: dict, tar: float):
    """Lowest validation FAR at the pair TAR; ties to the higher mean AUROC, then alphabetical (PLAN.md §5 step 3)."""
    return min(pairs, key=lambda p: (entries[p.name][tar_key(tar)]["accepted_wrong"],
                                     -round(mean_auroc(singles, p), 12), p.name))


def best_single(members, singles: dict) -> str:
    """Same criterion for one verifier: lowest validation FAR at its own 90% TAR threshold, then AUROC, then name."""
    return min((m.label for m in members), key=lambda n: (singles[n]["accepted_wrong"], -round(singles[n]["auroc"], 12), n))


def resample_selection(frame: Frame, groups: dict, boot: Boot, tar: float) -> dict:
    """Repeat thresholds and selection on every validation resample of whole problems.

    groups: name -> list of pairs. Returns per group the index of the winning pair and its thresholds in every
    replicate. Ties are broken as in select(), with AUROC recomputed on the resample."""
    pairs = [p for g in groups.values() for p in g]
    members = sorted({m for p in pairs for m in p.members}, key=lambda m: m.label)
    auc = {m: auroc_boot(boot, frame.member(m), frame.labels) for m in members}
    rows_c, rows_w = np.flatnonzero(frame.correct), np.flatnonzero(frame.wrong)
    correct = {m: frame.member(m)[rows_c] for m in members}
    wrong = {m: frame.member(m)[rows_w] for m in members}
    group_c, group_w = boot.group[rows_c], boot.group[rows_w]
    out = {g: {"winner": np.empty(boot.n, dtype=int), "thresholds": np.empty((boot.n, 2))} for g in groups}
    for b in range(boot.n):
        w = boot.weights[b]
        take = np.repeat(np.arange(len(rows_c)), w[group_c].astype(int))
        item = w[group_w]
        drawn = {m: correct[m][take] for m in members}
        for g, items in groups.items():
            best = None
            for k, p in enumerate(items):
                ta, tb, _ = fast_pair_thresholds(drawn[p.a], drawn[p.b], tar)
                false_accepts = float(item[accepts(wrong[p.a], wrong[p.b], (ta, tb))].sum())
                key = (false_accepts, -round((auc[p.a][b] + auc[p.b][b]) / 2, 12), p.name)
                if best is None or key < best[0]:
                    best = (key, k, (ta, tb))
            out[g]["winner"][b], out[g]["thresholds"][b] = best[1], best[2]
    return out


def validation_boot(frame: Frame, rules) -> Boot:
    return Boot(frame.problems, n=rules.bootstrap, seed=[rules.seed, VALIDATION_STREAM])


def h1_groups(settings: Settings) -> dict:
    """The two groups of the final test: name -> pairs."""
    pairs = plan_pairs(settings, H1_DOMAIN)
    return {"ar_ar": with_role(pairs, "h1_ar"), "ar_diffusion": with_role(pairs, "h1_mixed")}


# ---------------------------------------------------------------- cost and LLaDA's re-check

def gpu_types(scores: Scores, domain: str, members) -> list[str]:
    return sorted(set().union(*(scores.gpu_types(domain, m.verifier, m.column) for m in members)))


def pair_seconds(scores: Scores, domain: str, pair) -> float:
    """Measured GPU-seconds per candidate of a pair; nan when its members ran on different GPU types, because
    A100- and L4-seconds are never added or compared (PLAN.md §9)."""
    if len(gpu_types(scores, domain, pair.members)) > 1:
        return float("nan")
    return sum(scores.seconds(domain, m.verifier, m.column) for m in pair.members)


def cost_check(scores: Scores, domain: str, selected: dict, settings: Settings, rules) -> dict:
    """Step 4: measured GPU-seconds per candidate of the two selected pairs, same GPU type, ratio at most 1.5."""
    seconds = {g: pair_seconds(scores, domain, p) for g, p in selected.items()}
    members = [m for p in selected.values() for m in p.members]
    gpus = gpu_types(scores, domain, members)
    names = sorted(set().union(*(scores.gpus(domain, m.verifier, m.column) for m in members)))
    known = all(np.isfinite(v) and v > 0 for v in seconds.values())
    same = len(gpus) == 1 and gpus != ["unknown"]
    known = known and same      # seconds on different or unknown GPU types are never compared
    ratio = max(seconds.values()) / min(seconds.values()) if known else float("nan")
    within = bool(known and ratio <= rules.cost_ratio + TOL)
    cheaper = min(seconds, key=lambda g: (seconds[g], g)) if known else None
    return {"seconds_per_candidate": seconds, "ratio": ratio, "limit": rules.cost_ratio, "gpu_types": gpus,
            "gpus": names, "same_gpu_type": same, "within_limit": within, "cheaper": cheaper,
            "comparable_cost_claim": within and settings.mode != "repair",
            "budget_matched_secondary": bool(known and ratio > rules.cost_ratio + TOL)}


def llada_recheck(frame: Frame, settings: Settings, singles: dict, rules) -> dict:
    """LLaDA's AUROC on validation next to gate A's bar (PLAN.md §7). Reported; it changes no model, pair or score.
    "below_bar": any of gate A's three parts fails (decisions/2026-11-21-mechanism-wording.md)."""
    column = settings.llada_column(frame.domain)
    mine, small = singles[Member(DIFFUSION, column).label], singles[Member(SMALL_REFERENCE, "T1").label]
    checks = {"auroc_at_least_0.60": bool(mine["auroc"] >= rules.gate_a_auroc),
              "lower_bound_above_0.50": bool(mine["auroc_ci"][0] > rules.gate_a_lower),
              "within_0.10_below_small_ar": bool(mine["auroc"] >= small["auroc"] - rules.gate_a_small_gap)}
    return {"score": column, "auroc": mine["auroc"], "auroc_ci": mine["auroc_ci"],
            "small_reference": small["auroc"], "checks": checks, "below_bar": not all(checks.values()),
            "applied": False}


# ---------------------------------------------------------------- the whole step

def validation_step(frames: dict, scores: Scores, settings: Settings, rules=RULES) -> dict:
    """Everything that is decided on validation. frames: domain -> Frame with validation labels."""
    if settings.mode == "stopped":
        raise DataError(f"mode 'stopped' ({settings.reason}): H1 is not run and nothing is analyzed (PLAN.md §7)")
    out = {"thresholds": {}, "singles": {}, "pair_order": {}}
    for domain, frame in frames.items():
        boot = validation_boot(frame, rules)
        pairs = plan_pairs(settings, domain)
        out["singles"][domain] = {m.label: single_entry(frame, m, boot, rules) for m in all_members(settings, domain)}
        out["thresholds"][domain] = {p.name: {tar_key(t): pair_entry(frame, p, t) for t in rules.tars} for p in pairs}
        out["pair_order"][domain] = [p.name for p in pairs]
    out["pseudo_of_stronger"] = {d: {p.name: stronger_pseudo(out["singles"][d], p)
                                     for p in plan_pairs(settings, d) if p.kind != "pseudo"} for d in frames}
    if settings.runs_h1:
        out.update(h1_validation(frames[H1_DOMAIN], scores, settings, out, rules))
    else:
        out["h1"] = None
        out["ar_family"] = {d: ar_family_validation(settings, d, out, rules) for d in frames}
    if settings.qwen3_scored:
        out["qwen3"] = {d: qwen3_validation(settings, d, out, rules) for d in frames}
    return out


def stronger_pseudo(singles: dict, pair) -> dict:
    """The T1+T2 pseudo-pair of a pair's stronger member, by validation AUROC (PLAN.md §5, reference rows).
    A reconstruction score has no second wording, so it has no pseudo-pair."""
    best = min(pair.members, key=lambda m: (-round(singles[m.label]["auroc"], 12), m.label))
    return {"stronger": best.label, "pseudo_pair": None if best.column == "recon" else pseudo_pair_name(best.verifier)}


def h1_validation(frame: Frame, scores: Scores, settings: Settings, out: dict, rules) -> dict:
    domain = frame.domain
    entries, singles = out["thresholds"][domain], out["singles"][domain]
    groups = h1_groups(settings)
    selected = {g: select(items, entries, singles, rules.pair_tar) for g, items in groups.items()}
    boot = validation_boot(frame, rules)
    resampled = resample_selection(frame, groups, boot, rules.pair_tar)
    stability = {g: {p.name: float(np.mean(resampled[g]["winner"] == k)) for k, p in enumerate(items)}
                 for g, items in groups.items()}
    c = frame.correct
    tradeoff = {p.name: {tar_key(t): list(fast_pair_thresholds(frame.member(p.a)[c], frame.member(p.b)[c], t)[:2])
                         for t in TRADEOFF_TARS} for items in groups.values() for p in items}
    return {"h1": {"domain": domain, "test_name": settings.test_name,
                   "candidates": {g: [p.name for p in items] for g, items in groups.items()},
                   "selected": {g: p.name for g, p in selected.items()},
                   "selection_stability": stability,
                   "best_single": best_single(primary_members(settings, domain), singles),
                   "cost": cost_check(scores, domain, selected, settings, rules),
                   "llada_recheck": llada_recheck(frame, settings, singles, rules)},
            "tradeoff": tradeoff}


def ar_family_validation(settings: Settings, domain: str, out: dict, rules) -> dict:
    """Pivot or no H1: the same selection among the six cross-family pairs of the four AR verifiers, descriptively."""
    pairs = with_role(plan_pairs(settings, domain), "ar_family")
    singles = out["singles"][domain]
    best = select(pairs, out["thresholds"][domain], singles, rules.pair_tar)
    members = sorted({m for p in pairs for m in p.members}, key=lambda m: m.label)
    return {"pairs": [p.name for p in pairs], "selected": best.name, "best_single": best_single(members, singles)}


def qwen3_validation(settings: Settings, domain: str, out: dict, rules) -> dict:
    """Qwen3-8B with the AR and the diffusion partner that give the lowest validation FAR (strong-baseline check)."""
    pairs = with_role(plan_pairs(settings, domain), "qwen3")
    result = {}
    for kind in ("AR+AR", "AR+diffusion"):
        mine = [p for p in pairs if p.kind == kind]
        if mine:
            result[kind] = select(mine, out["thresholds"][domain], out["singles"][domain], rules.pair_tar).name
    return result


# ---------------------------------------------------------------- the analysis manifest

def relative(path, root) -> str:
    path, root = Path(path).resolve(), Path(root).resolve()
    if root not in path.parents:
        raise DataError(f"{path} is outside the repository {root}")
    return path.relative_to(root).as_posix()


def code_hashes(root) -> dict:
    return {f: sha256_file(Path(root) / f) for f in CODE_FILES}


def combined_hash(hashes: dict) -> str:
    return sha256_bytes("\n".join(f"{k} {v}" for k, v in sorted(hashes.items())).encode("utf-8"))


def load_inputs(root, settings: Settings, banks_dir, score_paths, domains, rules=RULES):
    root = Path(root)
    banks = load_banks(root / banks_dir, domains)
    needed = {d: needed_scores(settings, d) for d in domains}
    scores = load_scores([root / p for p in score_paths], banks, needed, rules=rules)
    frames, label_files = {}, {}
    for d, bank in banks.items():
        path = bank_paths(root / banks_dir, d)["labels_validation"]
        labels, sha = load_validation_labels(path, bank)
        label_files[d] = {"path": relative(path, root), "sha256": sha}
        frames[d] = make_frame(bank, scores, "validation", all_members(settings, d), labels)
    return banks, scores, frames, label_files


class NotFrozen(DataError):
    """The preregistration is missing, uncommitted or no longer matches its inputs."""


def git(root, *args) -> subprocess.CompletedProcess:
    return subprocess.run(["git", "-C", str(root), *args], capture_output=True, text=True)


def read_log(root) -> list[dict]:
    """The join log of this repository. Every join or rerun line of every committed version, on any branch, must still
    be in the working file, so switching branches or damaging the file cannot hide a join."""
    path, lines = Path(root) / JOIN_LOG, set()
    for commit in git(root, "log", "--all", "--format=%H", "--", JOIN_LOG).stdout.split():
        shown = git(root, "show", f"{commit}:{JOIN_LOG}")
        if shown.returncode == 0:
            lines |= {l for l in shown.stdout.splitlines() if l.strip()}
    entries = read_jsonl(path) if path.is_file() else []
    here = {json.dumps(e, sort_keys=True) for e in entries}
    for line in sorted(lines):
        try:
            entry = json.loads(line)
        except ValueError as exc:
            raise NotFrozen(f"{JOIN_LOG}: a committed version has an unreadable line") from exc
        if entry.get("event") in JOINS and json.dumps(entry, sort_keys=True) not in here:
            raise NotFrozen(f"{JOIN_LOG}: a committed version records the {entry['event']} of {entry.get('utc')}, which is "
                            "missing here; bring the log from that branch")
    if (path.is_file() and not entries) or any(e.get("event") == "completed" for e in entries) and not any(
            e.get("event") in JOINS for e in entries):
        raise NotFrozen(f"{JOIN_LOG} is damaged: it exists without the join it was written for")
    return entries


def side_block(root, side, side_scores, banks: dict, label_files: dict) -> dict | None:
    """The reconstruction side experiment (PLAN.md §4), on the math bank only (decisions/2026-11-21-mechanism-wording.md):
    its validation thresholds were set by analysis.reconstruction; the manifest freezes them, the scores they apply to
    and the code, by hash. Everything the test step will need is checked here, while it can still be fixed: the
    thresholds recompute from the validation labels, and every drawn answer of both splits has a usable score."""
    if side is None:
        return None
    recon = import_module("analysis.reconstruction")
    frozen = json.loads((root / side).read_text(encoding="utf-8"))
    if frozen["domain"] != H1_DOMAIN:
        raise DataError(f"{side}: the side experiment is on {H1_DOMAIN} only, not {frozen['domain']}")
    bank = banks.get(frozen["domain"])
    if bank is None or frozen["bank_hash"] != bank.sha256:
        raise DataError(f"{side}: its thresholds were set on another bank")
    files = find_score_files([root / p for p in side_scores])
    if not files:
        raise DataError("the side experiment needs --side-scores")
    if frozen.get("labels_sha256") != label_files[frozen["domain"]]["sha256"]:
        raise DataError(f"{side}: its thresholds were set with other validation labels")
    try:
        runs = {recon.run_label(k): (k, r) for k, r in recon.load_runs(files, bank.sha256).items()}
        missing = sorted(set(frozen["runs"]) - set(runs))
        fail([f"{label}: no scores" for label in missing], f"{side}: incomplete")
        kept = dict(runs[label] for label in frozen["runs"])
        label_of = recon.validation_labels(root / label_files[frozen["domain"]]["path"], bank.rows)
        again = recon.validation_stage(bank.rows, bank.sha256, frozen["subset"], label_of, kept, n_boot=1)
        drawn = {split: set(ids) for split, ids in frozen["subset"].items()}
        rows = {split: [r for r in bank.rows if r["candidate_id"] in ids] for split, ids in drawn.items()}
        for label, entry in frozen["runs"].items():
            if entry["points"] != again["runs"][label]["points"] or entry["settings"] != again["runs"][label]["settings"]:
                raise DataError(f"{side}: {label}: its thresholds do not recompute from the validation labels and scores")
            bad = int((~np.isfinite(recon.aligned(runs[label][1], rows["test"]))).sum())
            if bad or len(rows["test"]) != len(drawn["test"]) or any(r["split"] != "test" for r in rows["test"]):
                raise DataError(f"{side}: {label}: {bad} of {len(drawn['test'])} drawn test answers have no usable score, or "
                                "they are not all test answers of this bank")
    except SystemExit as exc:
        raise DataError(f"{side}: {exc}") from exc
    return {"file": hashed(root / side, root), "domain": frozen["domain"], "code": [hashed(root / f, root) for f in SIDE_CODE],
            "score_paths": [relative(root / p, root) for p in side_scores],
            "scores": {relative(f, root): sha256_file(f) for f in files},
            "thresholds": {label: {tar: p["threshold"] for tar, p in run["points"].items()}
                           for label, run in frozen["runs"].items()}}


def build_manifest(root, settings_path, banks_dir, score_paths, out_dir, domains=DOMAINS, rules=RULES, side=None,
                   side_scores=()) -> dict:
    root = Path(root)
    settings = load_settings(root / settings_path)
    if settings.runs_h1 and H1_DOMAIN not in domains:
        raise DataError("H1 is on math, so the math bank is required")
    banks, scores, frames, label_files = load_inputs(root, settings, banks_dir, score_paths, domains, rules)
    result = validation_step(frames, scores, settings, rules)
    data = {}
    for d, bank in banks.items():
        data[d] = {"bank": {"path": relative(bank.path, root), "sha256": bank.sha256},
                   "bank_manifest": hashed(bank_paths(root / banks_dir, d)["manifest"], root),
                   "split_sha256": bank.split_sha256(), "labels_validation": label_files[d],
                   "validation_pairs": bank.pairs("validation"), "test_pairs": bank.pairs("test")}
    code = code_hashes(root)
    models = {f"{v} {c}": {"revisions": sorted(map(str, m["revisions"])), "n_slots": sorted(m["n_slots"]),
                           "gpus": {d: sorted(scores.gpus(d, v, c)) for d in banks if (d, v, c) in scores.columns}}
              for (v, c), m in sorted(scores.info.items())}
    return {
        "kind": "analysis manifest (preregistration)", "schema": 1,
        "rules": json.loads(json.dumps(asdict(rules))),
        "settings": {"path": relative(root / settings_path, root), "sha256": sha256_file(root / settings_path),
                     "content": settings.to_dict()},
        "record": record_hash(root, settings),
        "data": data,
        "weights": {d: bank.pairs("test") for d, bank in banks.items()},
        "scores": {"paths": [relative(root / p, root) for p in score_paths],
                   "files": {relative(p, root): sha for p, sha in sorted(scores.files.items())},
                   "models": models, "read_but_not_used": [f"{relative(path, root)}: {what}" for path, what in scores.unused],
                   "seconds_per_candidate": {d: {f"{v} {c}": scores.seconds(d, v, c) for v, c in needed_scores(settings, d)}
                                             for d in banks}},
        "code": {"files": code, "sha256": combined_hash(code)},
        "bootstrap": {"replicates": rules.bootstrap, "seed": rules.seed, "unit": "problem", "interval": "percentile"},
        "validation": result,
        "side": side_block(root, side, side_scores, banks, label_files),
        "outputs": {"test_results": relative(root / out_dir / "test", root), "join_log": JOIN_LOG},
    }


def hashed(path, root) -> dict:
    return {"path": relative(path, root), "sha256": sha256_file(path)}


def record_hash(root, settings: Settings) -> dict:
    """The final model record the settings come from. It must exist: no full-bank answer is analyzed before it."""
    path = Path(root) / settings.record
    if not path.is_file():
        raise DataError(f"{settings.record}: the final model record named in the settings does not exist (PLAN.md §7)")
    return hashed(path, root)


def manifest_bytes(manifest: dict) -> bytes:
    return (json.dumps(manifest, indent=1, sort_keys=True, ensure_ascii=False, allow_nan=True) + "\n").encode("utf-8")


def normalized(manifest: dict) -> dict:
    """A manifest as it looks after a round trip through JSON, without the fields that change between runs."""
    d = json.loads(manifest_bytes(manifest))
    d.pop("created_utc", None)
    return d


def differences(a, b, where="") -> list[str]:
    """Where two JSON values differ. The step is deterministic; the tolerance only allows for last-digit differences
    of the linear algebra on another machine."""
    if isinstance(a, dict) and isinstance(b, dict):
        out = []
        for k in sorted(set(a) | set(b)):
            here = f"{where}.{k}" if where else k
            out += [f"{here}: only in one"] if (k in a) != (k in b) else differences(a[k], b[k], here)
        return out
    if isinstance(a, list) and isinstance(b, list) and len(a) == len(b):
        return [d for i, (x, y) in enumerate(zip(a, b)) for d in differences(x, y, f"{where}[{i}]")]
    if isinstance(a, float) and isinstance(b, float) and (np.isnan(a) and np.isnan(b) or abs(a - b) <= 1e-9):
        return []
    return [] if a == b else [f"{where}: {str(a)[:60]} != {str(b)[:60]}"]


def hash_problems(manifest: dict, root, rules=RULES) -> list[str]:
    """Files whose hash no longer matches the manifest. Reads bytes only, never a label."""
    root, out = Path(root), []

    def check(entry):
        path = root / entry["path"]
        if not path.is_file():
            out.append(f"{entry['path']}: missing")
        elif sha256_file(path) != entry["sha256"]:
            out.append(f"{entry['path']}: changed since the manifest was written")

    check(manifest["settings"])
    check(manifest["record"])
    for d in manifest["data"].values():
        for key in ("bank", "bank_manifest", "labels_validation"):
            check(d[key])
    for path, sha in manifest["scores"]["files"].items():
        check({"path": path, "sha256": sha})
    if manifest["side"]:
        check(manifest["side"]["file"])
        for entry in manifest["side"]["code"]:
            check(entry)
        for path, sha in manifest["side"]["scores"].items():
            check({"path": path, "sha256": sha})
    for path, sha in manifest["code"]["files"].items():
        check({"path": path, "sha256": sha})
    for f in CODE_FILES:    # the code that is running, wherever it was imported from
        if sha256_file(Path(__file__).parent / Path(f).name) != manifest["code"]["files"].get(f):
            out.append(f"{f}: the running code is not the code in the manifest")
    if set(manifest["code"]["files"]) != set(CODE_FILES) or manifest["code"]["sha256"] != combined_hash(manifest["code"]["files"]):
        out.append("code: the manifest does not cover the current analysis code")
    if manifest["rules"] != json.loads(json.dumps(asdict(rules))):
        out.append("rules: the fixed numbers differ from the manifest")
    return out


def check_manifest(path, root=".", rules=RULES) -> list[str]:
    """Every way the committed manifest no longer matches its inputs, including a full recomputation."""
    manifest = json.loads(Path(path).read_text(encoding="utf-8"))
    problems = hash_problems(manifest, root, rules)
    if problems:
        return problems
    banks_dir = Path(manifest["data"][next(iter(manifest["data"]))]["bank"]["path"]).parent.parent
    side = manifest["side"] or {}
    again = build_manifest(root, manifest["settings"]["path"], banks_dir, manifest["scores"]["paths"],
                           Path(manifest["outputs"]["test_results"]).parent, tuple(manifest["data"]), rules,
                           side.get("file", {}).get("path"), side.get("score_paths", ()))
    return differences(normalized(manifest), normalized(again))


# ---------------------------------------------------------------- summary and command line

def fmt(x, nd=3):
    return "–" if x is None or (isinstance(x, float) and not np.isfinite(x)) else f"{x:.{nd}f}"


def summary(manifest: dict, title="Validation step") -> str:
    v, lines = manifest["validation"], [f"# {title}", ""]
    if v.get("h1"):
        h = v["h1"]
        d = h["domain"]
        lines += [f"Final test: {h['test_name']}, {d}. Selected on validation at 90% pair TAR:", "",
                  "| Group | Pair | Validation TAR | Validation FAR | Wins in resamples | Selected |", "|---|---|---|---|---|---|"]
        for g, names in h["candidates"].items():
            for n in sorted(names, key=lambda n: v["thresholds"][d][n]["0.90"]["far"]):
                e = v["thresholds"][d][n]["0.90"]
                lines.append(f"| {g} | {n} | {fmt(e['tar'])} | {fmt(e['far'])} ({e['accepted_wrong']}/{e['n_wrong']}) "
                             f"| {fmt(h['selection_stability'][g][n], 2)} | {'**yes**' if h['selected'][g] == n else ''} |")
        c, r = h["cost"], h["llada_recheck"]
        seconds = ", ".join(f"{g} {fmt(s, 4)} s" for g, s in c["seconds_per_candidate"].items())
        lines += ["", f"- Best single verifier: {h['best_single']}",
                  f"- Cost: {seconds}; ratio {fmt(c['ratio'], 2)} (limit {c['limit']}), GPU {' / '.join(c['gpus'])}; "
                  f"comparable-cost claim: {c['comparable_cost_claim']}",
                  f"- LLaDA re-check (reported, not applied): AUROC {fmt(r['auroc'])} [{fmt(r['auroc_ci'][0])}, "
                  f"{fmt(r['auroc_ci'][1])}], {SMALL_REFERENCE} {fmt(r['small_reference'])}; "
                  + ("**below gate A's bar**" if r["below_bar"] else "meets gate A's bar")]
    for d, f in (v.get("ar_family") or {}).items():
        lines.append(f"- H1 is not run. AR-family study, {d}: selected {f['selected']}, best single {f['best_single']}")
    lines += ["", "## Single verifiers on validation (T1, or LLaDA's score in the final test)", "",
              "| Domain | Score | AUROC [95% CI] | FAR at 90% TAR |", "|---|---|---|---|"]
    for d, singles in v["singles"].items():
        for label, s in sorted(singles.items(), key=lambda kv: -kv[1]["auroc"]):
            if s["column"] in ("T1", "recon"):
                lines.append(f"| {d} | {label} | {fmt(s['auroc'])} [{fmt(s['auroc_ci'][0])}, {fmt(s['auroc_ci'][1])}] | {fmt(s['far'])} |")
    return "\n".join(lines) + "\n"


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--root", default=".", help="repository root; every path is relative to it")
    ap.add_argument("--settings")
    ap.add_argument("--banks", default="bank/full", help="folder with one subfolder per domain")
    ap.add_argument("--scores", nargs="+")
    ap.add_argument("--domains", nargs="+", default=list(DOMAINS), choices=DOMAINS)
    ap.add_argument("--out", default="results/final")
    ap.add_argument("--side", help="thresholds of the reconstruction side experiment (analysis.reconstruction validation)")
    ap.add_argument("--side-scores", nargs="*", default=[], help="the reconstruction scores they apply to")
    ap.add_argument("--check", metavar="MANIFEST", help="verify a manifest instead of writing one")
    args = ap.parse_args(argv)
    try:
        if args.check:
            problems = check_manifest(args.check, args.root)
            print("\n".join(problems) or f"{args.check} matches its inputs and recomputes exactly")
            return 1 if problems else 0
        if not args.settings or not args.scores:
            ap.error("--settings and --scores are required")
        if read_log(args.root):
            raise DataError(f"{JOIN_LOG}: test labels have been joined; the manifest can no longer change (PLAN.md §5)")
        manifest = build_manifest(args.root, args.settings, args.banks, args.scores, args.out, tuple(args.domains),
                                  side=args.side, side_scores=args.side_scores)
        manifest["created_utc"] = dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds")
        out = Path(args.root) / args.out
        out.mkdir(parents=True, exist_ok=True)
        target = out / "analysis_manifest.json"
        target.write_bytes(manifest_bytes(manifest))
        (out / "validation.md").write_text(summary(manifest), encoding="utf-8", newline="\n")
    except (DataError, ValueError, OSError) as exc:
        print(f"error: {exc}")
        return 2
    print(f"wrote {target}. Commit it with the analysis code (preregistration) before test labels are joined.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
