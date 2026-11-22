"""Reading banks, scores and validation labels for the final analysis, with every check in one place.

Formats: the bank is one folder per domain (bank.jsonl with split = validation | test and no label,
labels_validation.jsonl, manifest.json). Scores are harness.run folders (scores_A.jsonl, scores_B.jsonl,
manifest.json), harness.diffusion.run_score files, and, for the repair, reconstruction-score files.

Anything that does not line up is an error, never a silent drop: a wrong bank hash, an unknown, duplicate,
missing or failed row, two model revisions for one verifier, scoring settings other than the pilot's (precision,
attention, batch size, answer slots) or not recorded, a problem in both splits, a pair that is not two rows. Rows are aligned to the bank by candidate id, so file order never matters.

This module can read validation labels only. Test labels are read in analysis/final_evaluate.py, which checks
the committed analysis manifest first.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from analysis.final_settings import ANSWER_PREFIX, ANSWER_SLOTS, BY_KEY, DOMAINS, RULES, VERIFIERS

VALIDATION_LABELS = "labels_validation.jsonl"


class DataError(ValueError):
    """An input file does not match the plan's contracts."""


def sha256_bytes(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def sha256_file(path) -> str:
    return sha256_bytes(Path(path).read_bytes())


def _no_duplicate_keys(pairs):
    out = dict(pairs)
    if len(out) != len(pairs):
        raise ValueError("duplicate JSON key")
    return out


def read_jsonl(path) -> list[dict]:
    rows = []
    with open(path, encoding="utf-8") as f:
        for number, line in enumerate(f, 1):
            if not line.strip():
                continue
            try:
                row = json.loads(line, object_pairs_hook=_no_duplicate_keys)
            except ValueError as exc:
                raise DataError(f"{path}, line {number}: {exc}") from exc
            if not isinstance(row, dict):
                raise DataError(f"{path}, line {number}: not a JSON object")
            rows.append(row)
    return rows


def fail(errors: list[str], what: str, limit: int = 8) -> None:
    if errors:
        more = f"\n  ... and {len(errors) - limit} more" if len(errors) > limit else ""
        raise DataError(f"{what}:\n  " + "\n  ".join(errors[:limit]) + more)


# ---------------------------------------------------------------- bank

@dataclass
class Bank:
    domain: str
    path: str
    sha256: str
    rows: list[dict]
    index: dict                 # candidate_id -> row number
    splits: np.ndarray
    file_ids: set               # every candidate id in the file, whatever its domain (the pilot's file has both)
    recorded: dict = field(default_factory=dict)    # file name -> sha256 values from the bank's manifest.json

    def mask(self, split: str) -> np.ndarray:
        return self.splits == split

    def ids(self, split: str) -> list[str]:
        return [r["candidate_id"] for r in self.rows if r["split"] == split]

    def split_sha256(self) -> str:
        """Hash of which candidate is in which split, independent of row order."""
        lines = sorted(f"{r['candidate_id']}\t{r['problem_id']}\t{r['split']}" for r in self.rows)
        return sha256_bytes("\n".join(lines).encode("utf-8"))

    def pairs(self, split: str) -> int:
        """Matched pairs in a split. Every pair holds one wrong answer, so this is also the number of wrong
        answers, known before any label is read (the domain weights of PLAN.md §5 step 9)."""
        return len({r["pair_id"] for r in self.rows if r["split"] == split})


def recorded_hashes(manifest) -> dict:
    """file name -> set of sha256 values, for every such entry anywhere in a bank manifest. Takes
    {"bank.jsonl": "<sha>"}, {"bank.jsonl": {"sha256": "<sha>"}} and {"path": "bank.jsonl", "sha256": "<sha>"}."""
    out: dict[str, set] = {}

    def is_sha(x):
        return isinstance(x, str) and re.fullmatch(r"[0-9a-f]{64}", x) is not None

    def base(x):
        return str(x).replace("\\", "/").rsplit("/", 1)[-1]

    def walk(node):
        if isinstance(node, dict):
            if isinstance(node.get("path"), str) and is_sha(node.get("sha256")):
                out.setdefault(base(node["path"]), set()).add(node["sha256"])
            for key, value in node.items():
                digest = value.get("sha256") if isinstance(value, dict) else value
                if "." in base(key) and is_sha(digest):
                    out.setdefault(base(key), set()).add(digest)
                walk(value)
        elif isinstance(node, list):
            for value in node:
                walk(value)

    walk(manifest)
    return out


def load_bank(path, domain: str, splits=("validation", "test"), manifest_path=None) -> Bank:
    """Read one bank file and keep one domain's rows. With manifest_path, the file must match the hash
    recorded there."""
    path = Path(path)
    sha = sha256_file(path)
    recorded = {}
    if manifest_path is not None:
        if not Path(manifest_path).is_file():
            raise DataError(f"{manifest_path}: the bank manifest is missing")
        recorded = recorded_hashes(json.loads(Path(manifest_path).read_text(encoding="utf-8")))
        if sha not in recorded.get(path.name, ()):
            raise DataError(f"{path}: sha256 {sha} does not match the bank manifest "
                            f"({sorted(recorded.get(path.name, ())) or 'no entry for ' + path.name})")
    everything = read_jsonl(path)
    rows = [r for r in everything if r.get("domain") == domain]
    errors, index, pairs, split_of = [], {}, {}, {}
    for number, r in enumerate(rows):
        cid = r.get("candidate_id")
        if "label" in r:
            errors.append(f"{cid}: the bank file carries a label; labels belong in their own files (PLAN.md §2)")
        missing = [k for k in ("candidate_id", "pair_id", "problem_id", "source", "generator", "split") if not r.get(k)]
        if missing:
            errors.append(f"row {number}: missing {missing}")
            continue
        if cid in index:
            errors.append(f"{cid}: duplicate candidate id")
        index[cid] = number
        if r["split"] not in splits:
            errors.append(f"{cid}: split {r['split']!r}, expected one of {splits}")
        if split_of.setdefault(r["problem_id"], r["split"]) != r["split"]:
            errors.append(f"{r['problem_id']}: the problem is in two splits")
        pairs.setdefault(r["pair_id"], []).append(r)
    for pair_id, members in pairs.items():
        if len(members) != 2 or len({(m["problem_id"], m["generator"], m["split"]) for m in members}) != 1:
            errors.append(f"{pair_id}: a pair must be two answers of one (problem, generator) cell in one split")
    if not rows:
        errors.append(f"no {domain} rows")
    fail(errors, f"invalid bank {path}")
    return Bank(domain, path.as_posix(), sha, rows, index, np.array([r["split"] for r in rows], dtype=object),
                {r.get("candidate_id") for r in everything}, recorded)


# ---------------------------------------------------------------- scores

@dataclass
class Column:
    """One verifier's scores with one template, aligned to a bank (nan = no row)."""
    score: np.ndarray
    p_yes: np.ndarray       # P(Yes) / (P(Yes) + P(No)), nan for scores that are not verdicts
    mass: np.ndarray        # P(Yes) + P(No)
    seconds: np.ndarray     # warm rows only: the first row of a run is cold and left out
    gpus: set = field(default_factory=set)      # device names its rows ran on


@dataclass
class Scores:
    columns: dict           # (domain, verifier, column) -> Column
    info: dict              # (verifier, column) -> {"revisions", "n_slots"}
    files: dict             # path -> sha256 of every file that was read
    unused: list            # (path, what) that was read but is not part of the analysis

    def score(self, domain: str, verifier: str, column: str) -> np.ndarray:
        """Scores aligned to the bank. "avg" is the mean of T1 and T2."""
        if column == "avg":
            return (self.columns[(domain, verifier, "T1")].score + self.columns[(domain, verifier, "T2")].score) / 2
        return self.columns[(domain, verifier, column)].score

    def seconds(self, domain: str, verifier: str, column: str) -> float:
        """Mean measured GPU-seconds per candidate, warm rows only. "avg" needs both templates, so it costs both. nan
        when the rows ran on more than one GPU type."""
        if len(self.gpu_types(domain, verifier, column)) > 1:
            return float("nan")
        if column == "avg":
            return self.seconds(domain, verifier, "T1") + self.seconds(domain, verifier, "T2")
        s = self.columns[(domain, verifier, column)].seconds
        return float(np.mean(s[np.isfinite(s)])) if np.isfinite(s).any() else float("nan")

    def gpus(self, domain: str, verifier: str, column: str) -> set:
        """Device names one verifier's scores of one domain ran on."""
        templates = ("T1", "T2") if column == "avg" else (column,)
        return set().union(*(self.columns[(domain, verifier, c)].gpus or {"unknown"} for c in templates))

    def gpu_types(self, domain: str, verifier: str, column: str) -> set:
        return {gpu_type(name) for name in self.gpus(domain, verifier, column)}


def gpu_type(name: str) -> str:
    """The GPU family in a device name ("NVIDIA A100-SXM4-40GB" -> "A100")."""
    found = re.search(r"\b(A100|H100|H200|L40S?|L4|T4|V100|P100|A10G?|RTX[ \w]*\d+)\b", name)
    return found.group(1) if found else name


def find_score_files(paths) -> list[Path]:
    """Score files under the given files and folders: every .jsonl except pair joins and run manifests."""
    out = []
    for p in map(Path, paths):
        if p.is_dir():
            out += [f for f in sorted(p.rglob("*.jsonl"))
                    if f.name != "pairs.jsonl" and not f.name.endswith(".manifest.jsonl")]
        elif p.is_file():
            out.append(p)
        else:
            raise DataError(f"{p}: no such score file or folder")
    return sorted(set(out))


def _finite(x) -> bool:
    return isinstance(x, (int, float)) and not isinstance(x, bool) and math.isfinite(x)


def _sidecar(path: Path) -> list[dict]:
    """The run manifests a diffusion score file has beside it, one per finished run."""
    beside = Path(str(path) + ".manifest.jsonl")
    return read_jsonl(beside) if beside.is_file() else []


def _file_gpus(path: Path) -> set:
    """GPU names recorded beside a score file; "unknown" for a run that does not name one."""
    run = path.parent / "manifest.json"
    if path.name.startswith("scores_") and run.is_file():
        names = {(json.loads(run.read_text(encoding="utf-8")).get("gpu") or {}).get("name")}
    else:
        names = {(m.get("environment") or {}).get("gpu") for m in _sidecar(path)}
    return {n or "unknown" for n in names} or {"unknown"}


def _check_diffusion_runs(path: Path, scored: int, errors: list) -> None:
    """Every row of a diffusion score file must come from a recorded run at batch size 1 (PLAN.md §4). A run that died
    leaves rows without a record, so their batch size and GPU are not known; they have to be scored again."""
    runs = _sidecar(path)
    if not runs:
        errors.append(f"{path}: no run manifest ({path.name}.manifest.jsonl)")
        return
    sizes = sorted({str(m.get("batch_size")) for m in runs})
    if sizes != ["1"]:
        errors.append(f"{path}: scored at batch size {', '.join(sizes)}, not 1")
    recorded = sum((m.get("completed") or {}).get("ok", 0) for m in runs)
    if recorded != scored:
        errors.append(f"{path}: its run manifests record {recorded} scored rows, the file has {scored}; score it again in one go")


def _check_run_manifest(path: Path, sha: str, verifiers: set, errors: list) -> None:
    """An AR run folder records the hash of every file it wrote, whether the run finished, and how it was scored.
    The settings must be the pilot's: unquantized BF16, batch size 1, SDPA attention (Gemma-2 eager)."""
    run = path.parent / "manifest.json"
    if path.name not in ("scores_A.jsonl", "scores_B.jsonl"):
        errors.append(f"{path}: AR scores are read only from a run folder's scores_A.jsonl and scores_B.jsonl")
        return
    if not run.is_file():
        errors.append(f"{path}: the run's manifest.json is missing")
        return
    m = json.loads(run.read_text(encoding="utf-8"))
    if m.get("status") != "completed":
        errors.append(f"{run}: run status is {m.get('status')!r}, not 'completed'")
    if m.get("limit") is not None:
        errors.append(f"{run}: the run scored only the first {m['limit']} rows")
    if (m.get("outputs_sha256") or {}).get(path.name) != sha:
        errors.append(f"{path}: the file does not match the hash in its run manifest")
    model = (m.get("models") or {}).get(path.stem[-1]) or {}
    got = {"compute_dtype": m.get("compute_dtype"), "batch_size": m.get("batch_size"),
           "quantization": model.get("actual_quantization"), "attention": model.get("attention_implementation")}
    for name in sorted(verifiers):
        wanted = {"compute_dtype": "torch.bfloat16", "batch_size": 1, "quantization": None, "attention": VERIFIERS[name].attention}
        if got != wanted:
            errors.append(f"{run}: {name} was scored with {got}, not {wanted}")


def classify(row: dict) -> tuple:
    """(kind, model key, column, bank hash) of a score row; raises DataError for an unknown row format."""
    if "method" in row:
        return "recon", row.get("model"), "recon", row.get("bank_hash")
    if "n_slots" in row:
        return "diffusion", row.get("model"), row.get("template"), row.get("bank_hash")
    if "model_id" in row:
        return "ar", row.get("model_id"), row.get("template"), row.get("bank_sha256")
    raise DataError("unknown score row format")


def load_scores(paths, banks: dict, needed: dict, ignore_templates=(), rules=RULES) -> Scores:
    """Read every score file, align it to the banks and check that each needed score is complete.

    banks: domain -> Bank. needed: domain -> list of (verifier, column). ignore_templates is only for the
    pilot, whose folders also hold T3; on the full bank a T3 row is an error (PLAN.md §4)."""
    by_hash: dict[str, list] = {}
    for bank in banks.values():
        by_hash.setdefault(bank.sha256, []).append(bank)
    wanted = {(v, c) for items in needed.values() for v, c in items}
    want_recon = any(c == "recon" for _, c in wanted)
    columns, info, files, errors, unused = {}, {}, {}, [], {}

    def slot(domain, verifier, column):
        key = (domain, verifier, column)
        if key not in columns:
            n = len(banks[domain].rows)
            columns[key] = Column(*(np.full(n, np.nan) for _ in range(4)))
        return columns[key]

    for path in find_score_files(paths):
        sha = files[path.as_posix()] = sha256_file(path)
        gpus, file_errors, first = _file_gpus(path), [], True
        used, scored = {"ar": set(), "diffusion": set(), "recon": set()}, 0
        for number, row in enumerate(read_jsonl(path), 1):
            where = f"{path}, line {number}"
            try:
                kind, key, column, bank_hash = classify(row)
            except DataError as exc:
                file_errors.append(f"{where}: {exc}")
                break
            cold, first = first, False
            scored += row.get("status") == "ok"
            if kind == "recon" and (not want_recon or row.get("method") != "diffusion_mc"
                                    or BY_KEY.get(key) not in {v for v, c in wanted if c == "recon"}
                                    or row.get("n_samples") != rules.recon_samples
                                    or row.get("seed") != rules.recon_seed):
                unused[(path.as_posix(), "reconstruction rows not used by the final test")] = True
                continue
            if column in ignore_templates:
                unused[(path.as_posix(), f"template {column}")] = True
                continue
            if kind != "recon" and column not in ("T1", "T2"):
                file_errors.append(f"{where}: template {column!r}; only T1 and T2 are run on the full bank")
                continue
            if key not in BY_KEY:
                file_errors.append(f"{where}: unknown model {key!r}")
                continue
            verifier = BY_KEY[key]
            if (verifier, column) not in wanted:
                unused[(path.as_posix(), f"{verifier} {column}")] = True
                continue
            used[kind].add(verifier)
            if bank_hash not in by_hash:
                file_errors.append(f"{where}: bank hash {bank_hash} is not a bank of this analysis "
                                   f"(expected {', '.join(sorted(h[:12] for h in by_hash))}…)")
                continue
            cid = row.get("candidate_id") or row.get("id")
            bank = next((b for b in by_hash[bank_hash] if cid in b.index), None)
            if bank is None:
                if not any(cid in b.file_ids for b in by_hash[bank_hash]):
                    file_errors.append(f"{where}: candidate {cid!r} is not in the bank")
                continue
            if (verifier, column) not in needed.get(bank.domain, ()):
                unused[(path.as_posix(), f"{verifier} {column} on {bank.domain}")] = True
                continue
            c, i = slot(bank.domain, verifier, column), bank.index[cid]
            if row.get("status") not in ("ok", "completed") or not _finite(row.get("score")):
                file_errors.append(f"{where}: {verifier} {column} {cid}: status {row.get('status')!r}, "
                                   f"score {row.get('score')!r}")
                continue
            if np.isfinite(c.score[i]):
                file_errors.append(f"{where}: {verifier} {column} {cid}: duplicate row")
                continue
            if kind == "diffusion" and (row.get("n_slots"), row.get("answer_prefix", ANSWER_PREFIX)) != (ANSWER_SLOTS, ANSWER_PREFIX):
                file_errors.append(f"{where}: {verifier} {column} {cid}: {row.get('n_slots')} answer slots and prefix "
                                   f"{row.get('answer_prefix')!r}, not {ANSWER_SLOTS} and {ANSWER_PREFIX!r}")
                continue
            if kind == "recon" and (row.get("mc_group"), row.get("micro_batch")) != (rules.recon_mc_group, rules.recon_micro_batch):
                file_errors.append(f"{where}: {verifier} {cid}: mc_group {row.get('mc_group')} and micro_batch "
                                   f"{row.get('micro_batch')}, not {rules.recon_mc_group} and {rules.recon_micro_batch}")
                continue
            c.score[i] = float(row["score"])
            if kind != "recon":
                yes, no = row.get("p_yes"), row.get("p_no")
                if not (_finite(yes) and _finite(no) and yes + no > 0):
                    file_errors.append(f"{where}: {verifier} {column} {cid}: no Yes/No probabilities")
                    continue
                c.p_yes[i], c.mass[i] = yes / (yes + no), yes + no
            seconds = row.get("forward_seconds") if kind == "ar" else row.get("runtime_s")
            if row.get("cold", cold) is False and _finite(seconds):
                c.seconds[i] = seconds
            meta = info.setdefault((verifier, column), {"revisions": set(), "n_slots": set()})
            meta["revisions"].add(row.get("model_revision") or row.get("revision"))
            if kind == "diffusion":
                meta["n_slots"].add(row.get("n_slots"))
            c.gpus |= {row["gpu"]} if isinstance(row.get("gpu"), str) else gpus
        if used["ar"]:
            _check_run_manifest(path, sha, used["ar"], file_errors)
        if used["diffusion"]:
            _check_diffusion_runs(path, scored, file_errors)
        errors += file_errors
    fail(errors, "invalid score files")

    for domain, items in needed.items():
        bank = banks[domain]
        for verifier, column in items:
            c = columns.get((domain, verifier, column))
            missing = len(bank.rows) if c is None else int((~np.isfinite(c.score)).sum())
            if missing:
                example = bank.rows[0 if c is None else int(np.flatnonzero(~np.isfinite(c.score))[0])]["candidate_id"]
                errors.append(f"{domain}: {verifier} {column}: {missing} of {len(bank.rows)} candidates have no score "
                              f"(e.g. {example})")
    fail(errors, "incomplete scores (every verifier x template x candidate is required)")
    for name in VERIFIERS:
        mine = [m for (v, _), m in info.items() if v == name]
        if len(set().union(*(m["revisions"] for m in mine)) if mine else ()) > 1:
            errors.append(f"{name}: scored with more than one model revision")
    fail(errors, "inconsistent scores")
    return Scores(columns, info, files, sorted(unused))


# ---------------------------------------------------------------- labels

def align_labels(label_rows: list[dict], bank: Bank, split: str, source: str) -> np.ndarray:
    """Labels of one split in bank order. Refuses rows of any other split before using a single label."""
    expected, other = set(bank.ids(split)), set(bank.index)
    errors, label_of = [], {}
    for r in label_rows:
        cid = r.get("candidate_id")
        if cid not in expected:
            errors.append(f"{cid}: " + ("a candidate of another split" if cid in other else "not in the bank"))
        elif cid in label_of:
            errors.append(f"{cid}: duplicate label")
        elif r.get("label") not in (0, 1) or isinstance(r.get("label"), bool):
            errors.append(f"{cid}: label {r.get('label')!r}, expected 0 or 1")
        else:
            label_of[cid] = r["label"]
    fail(errors, f"refused label file {source}")
    missing = sorted(expected - set(label_of))
    fail([f"{cid}: no label" for cid in missing], f"incomplete label file {source}")
    pairs: dict[str, list] = {}
    for r in bank.rows:
        if r["split"] == split:
            pairs.setdefault(r["pair_id"], []).append(label_of[r["candidate_id"]])
    fail([f"{p}: labels {sorted(v)}, expected one correct and one wrong" for p, v in pairs.items()
          if sorted(v) != [0, 1]], f"label file {source} breaks the matched pairs")
    return np.array([label_of[r["candidate_id"]] for r in bank.rows if r["split"] == split], dtype=int)


def load_validation_labels(path, bank: Bank) -> tuple[np.ndarray, str]:
    """(labels in bank order, sha256). Takes a file named labels_validation.jsonl and nothing else."""
    path = Path(path)
    if path.name != VALIDATION_LABELS:
        raise DataError(f"{path}: the validation step reads only {VALIDATION_LABELS} (PLAN.md §2, label custody)")
    sha = sha256_file(path)
    if path.name in bank.recorded and sha not in bank.recorded[path.name]:
        raise DataError(f"{path}: does not match the hash in the bank manifest")
    return align_labels(read_jsonl(path), bank, "validation", str(path)), sha


# ---------------------------------------------------------------- one split of one domain, ready to analyze

@dataclass
class Frame:
    """Scores (and, once known, labels) of one split of one domain, in bank order."""
    domain: str
    rows: list[dict]
    problems: list[str]
    scores: dict            # (verifier, column) -> array
    p_yes: dict             # (verifier, "T1"|"T2") -> P(Yes) / (P(Yes) + P(No))
    mass: dict              # (verifier, "T1"|"T2") -> P(Yes) + P(No)
    labels: np.ndarray | None = None
    error_types: np.ndarray | None = None   # "" where unknown; only on test, where wrong answers are tagged

    def member(self, m) -> np.ndarray:
        return self.scores[(m.verifier, m.column)]

    def column(self, key: str) -> np.ndarray:
        return np.array([r.get(key, "") for r in self.rows], dtype=object)

    @property
    def correct(self) -> np.ndarray:
        return self.labels == 1

    @property
    def wrong(self) -> np.ndarray:
        return self.labels == 0


def make_frame(bank: Bank, scores: Scores, split: str, members, labels=None) -> Frame:
    mask = bank.mask(split)
    rows = [r for r in bank.rows if r["split"] == split]
    out, probs, mass = {}, {}, {}
    for m in members:
        key = (m.verifier, m.column)
        out[key] = scores.score(bank.domain, *key)[mask]
        if m.column in ("T1", "T2"):
            probs[key] = scores.columns[(bank.domain, *key)].p_yes[mask]
            mass[key] = scores.columns[(bank.domain, *key)].mass[mask]
    return Frame(bank.domain, rows, [r["problem_id"] for r in rows], out, probs, mass, labels)


def bank_paths(root, domain: str) -> dict:
    folder = Path(root) / domain
    return {"bank": folder / "bank.jsonl", "manifest": folder / "manifest.json",
            "labels_validation": folder / VALIDATION_LABELS}


def load_banks(root, domains=DOMAINS) -> dict:
    out = {}
    for domain in domains:
        paths = bank_paths(root, domain)
        if not paths["bank"].is_file():
            raise DataError(f"{paths['bank']}: missing")
        out[domain] = load_bank(paths["bank"], domain, manifest_path=paths["manifest"])
    ids: set = set()
    for bank in out.values():
        if ids & set(bank.index):
            raise DataError("a candidate id appears in two domains")
        ids |= set(bank.index)
    return out
