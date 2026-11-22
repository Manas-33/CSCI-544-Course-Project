"""A synthetic full bank with scores, in the real file formats, for the tests of the final analysis.

    from analysis.final_synthetic import write_study
    write_study("/tmp/study", seed=0)

Writes bank/full/<domain>/, results/full/<domain>/, configs/final_models.json, the final model record and a copy of
the frozen analysis code under a root folder, so the validation and test steps run on it exactly as on the real
data. Nothing in it is a result.
"""

from __future__ import annotations

import hashlib
import json
import shutil
from pathlib import Path

import numpy as np

from analysis.final_settings import RULES, VERIFIERS
from analysis.final_validation import CODE_FILES

GENERATORS = ("org/gen-a", "org/gen-b", "org/gen-c")
SOURCES = {"math": ("gsm8k-platinum/test", "math-500/test"), "code": ("evalplus/mbpp+", "evalplus/humaneval+")}
# how far apart a verifier's correct and wrong answers are, in standard deviations
SEPARATION = {"Qwen2.5-7B": 0.8, "Llama-3.1-8B": 0.7, "Gemma-2-9B": 0.8, "Mistral-7B": 0.4, "Llama-3.2-3B": 0.2,
              "Qwen2.5-1.5B": 0.4, "Qwen3-8B": 1.0, "LLaDA-8B": 0.6, "Dream-7B": 0.5}
AR_RUNS = (("Qwen2.5-7B", "Mistral-7B"), ("Llama-3.1-8B", "Gemma-2-9B"), ("Llama-3.2-3B", "Qwen2.5-1.5B"))
GO = {"schema": 1, "record": "decisions/final-model-record.md", "mode": "go", "reason": "synthetic",
      "h1_ar": ["Gemma-2-9B", "Llama-3.1-8B", "Qwen2.5-7B"], "llada_score": "verdict", "dropped": {}, "qwen3_scored": False}
A100 = "NVIDIA A100-SXM4-40GB"


def sha(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8", newline="\n")


def make_bank(domain: str, problems: int, rng) -> tuple[list[dict], dict]:
    """Bank rows and candidate_id -> label. One third of the problems are validation; 1 or 2 pairs per problem."""
    rows, labels = [], {}
    for k in range(problems):
        pid = f"{domain}-{k:04d}"
        split = "validation" if k % 3 == 0 else "test"
        for g in rng.permutation(len(GENERATORS))[:rng.integers(1, 3)]:
            generator = GENERATORS[g]
            members = [(f"{pid}__{g}__{i}", i) for i in (0, 1)]
            rng.shuffle(members)
            for cid, label in members:
                text = f"answer {cid}"
                rows.append({"candidate_id": cid, "pair_id": f"{pid}:{g}", "problem_id": pid, "source": SOURCES[domain][k % 2],
                             "domain": domain, "problem": f"question {pid}", "candidate": text, "generator": generator,
                             "generator_revision": "r", "seed": 1, "settings": {}, "text_sha256": sha(text), "split": split,
                             "checker_version": "synthetic"})
                labels[cid] = int(label)
    return rows, labels


def make_scores(rows: list[dict], labels: dict, rng, separation=SEPARATION, shared=0.5) -> dict:
    """(verifier, column) -> scores. AR verifiers share part of their errors; diffusion verifiers share less.
    T1 and T2 of one verifier share most of theirs."""
    y = np.array([labels[r["candidate_id"]] for r in rows])
    n = len(rows)
    _, problem = np.unique([r["problem_id"] for r in rows], return_inverse=True)
    difficulty = rng.normal(size=problem.max() + 1)[problem]
    ar_common = rng.normal(size=n)
    out = {}
    for name, d in separation.items():
        weight = shared if VERIFIERS[name].kind == "AR" else shared / 2
        own = rng.normal(size=n)
        base = np.sqrt(1 - weight ** 2 - 0.09) * own + weight * ar_common + 0.3 * difficulty
        for column, extra in (("T1", 0.0), ("T2", 0.3)):
            x = d * y + base + extra * rng.normal(size=n) + 1.0
            out[(name, column)] = np.round(x * 8) / 8       # BF16-like steps, so there are ties
    out[("LLaDA-8B", "recon")] = separation["LLaDA-8B"] * y + rng.normal(size=n) - 2.0
    return out


def error_type(domain: str, k: int) -> str | None:
    """Code: from the execution logs. Math: hand tags on some of the wrong answers."""
    return "wrong_output" if domain == "code" else (None, "arithmetic slip", "wrong method")[k % 3]


def probabilities(score: float) -> tuple[float, float]:
    yes = 1 / (1 + np.exp(-score))
    return float(yes), float(1 - yes)


def ar_row(row, model, name, column, score, bank_sha, first) -> dict:
    yes, no = probabilities(score)
    return {"id": row["candidate_id"], "candidate_id": row["candidate_id"], "bank_sha256": bank_sha, "model": model,
            "model_id": VERIFIERS[name].keys[0], "model_revision": f"rev-{name}", "template": column, "status": "completed",
            "score": float(score), "p_yes": yes, "p_no": no, "yes_no_mass": yes + no, "forward_seconds": 0.5 if first else 0.05,
            "cold": first, "input_tokens": 10, "problem_id": row["problem_id"], "pair_id": row["pair_id"], "domain": row["domain"]}


def diffusion_row(row, name, column, score, bank_sha) -> dict:
    yes, no = probabilities(score)
    return {"bank_hash": bank_sha, "candidate_id": row["candidate_id"], "model": VERIFIERS[name].keys[0], "revision": f"rev-{name}",
            "template": column, "n_slots": 1, "answer_prefix": "none", "score": float(score), "p_yes": yes, "p_no": no,
            "yes_no_mass": yes + no, "status": "ok", "runtime_s": 0.06, "peak_mem_gb": 16.0}


def recon_row(row, name, score, bank_sha, seed=RULES.recon_seed, samples=RULES.recon_samples) -> dict:
    return {"bank_hash": bank_sha, "candidate_id": row["candidate_id"], "model": VERIFIERS[name].keys[0], "revision": f"rev-{name}",
            "method": "diffusion_mc", "n_samples": samples, "seed": seed, "n_tokens": 50, "loglik": float(score) * 50,
            "score": float(score), "status": "ok", "runtime_s": 5.0, "peak_mem_gb": 16.0, "mc_group": RULES.recon_mc_group,
            "micro_batch": RULES.recon_micro_batch, "gpu": A100}


def write_scores(root: Path, domain: str, rows, scores, bank_sha, with_qwen3=False, with_recon=False, gpus=None) -> None:
    folder, gpus = root / "results/full" / domain, gpus or {}
    runs = AR_RUNS + ((("Qwen3-8B", "Llama-3.2-3B"),) if with_qwen3 else ())
    written = set()
    for a, b in runs:
        for column in ("T1", "T2"):
            run = folder / "ar" / f"full_{a}_{b}_{column}"
            for model, name in (("A", a), ("B", b)):
                if (name, column) in written:
                    write_jsonl(run / f"scores_{model}.jsonl", [])
                    continue
                written.add((name, column))
                write_jsonl(run / f"scores_{model}.jsonl", [ar_row(r, model, name, column, s, bank_sha, i == 0) for i, (r, s)
                                                            in enumerate(zip(rows, scores[(name, column)]))])
            outputs = {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in run.glob("scores_*.jsonl")}
            models = {m: {"actual_quantization": None, "attention_implementation": VERIFIERS[name].attention}
                      for m, name in (("A", a), ("B", b))}
            (run / "manifest.json").write_text(json.dumps({
                "status": "completed", "limit": None, "outputs_sha256": outputs, "compute_dtype": "torch.bfloat16",
                "batch_size": 1, "models": models, "gpu": {"name": gpus.get((domain, a), A100)}}), encoding="utf-8")
            write_jsonl(run / "pairs.jsonl", [{"ignored": True}])
    for name in ("LLaDA-8B", "Dream-7B"):
        key = VERIFIERS[name].keys[0]
        for column in ("T1", "T2"):
            path = folder / key / f"full_{column}_n1.jsonl"
            write_jsonl(path, [diffusion_row(r, name, column, s, bank_sha) for r, s in zip(rows, scores[(name, column)])])
            write_jsonl(Path(str(path) + ".manifest.jsonl"), [{
                "model": key, "batch_size": 1, "completed": {"ok": len(rows)}, "environment": {"gpu": gpus.get((domain, name), A100)}}])
    if with_recon:
        write_jsonl(folder / "reconstruction/llada_128.jsonl",
                    [recon_row(r, "LLaDA-8B", s, bank_sha) for r, s in zip(rows, scores[("LLaDA-8B", "recon")])])


def write_study(root, seed=0, problems=None, settings=None, separation=SEPARATION, gpus=None, with_recon=False) -> dict:
    """Write everything; returns {"labels": domain -> {candidate_id: label}, "rows": domain -> bank rows}. gpus:
    (domain, verifier) -> device name of its runs (an AR run's device goes with its first model)."""
    root, rng = Path(root), np.random.default_rng(seed)
    settings = dict(GO if settings is None else settings)
    labels_of, rows_of = {}, {}
    for domain, count in (problems or {"math": 90, "code": 45}).items():
        rows, labels = make_bank(domain, count, rng)
        folder = root / "bank/full" / domain
        write_jsonl(folder / "bank.jsonl", rows)
        for split in ("validation", "test"):
            write_jsonl(folder / f"labels_{split}.jsonl", [
                {"candidate_id": r["candidate_id"], "label": labels[r["candidate_id"]], "status": "ok",
                 "error_type": error_type(domain, k) if labels[r["candidate_id"]] == 0 else None}
                for k, r in enumerate(rows) if r["split"] == split])
        files = {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(folder.glob("*.jsonl"))}
        (folder / "manifest.json").write_text(json.dumps({"seed": 544, "files": files}), encoding="utf-8")
        write_scores(root, domain, rows, make_scores(rows, labels, rng, separation), files["bank.jsonl"],
                     with_qwen3=settings.get("qwen3_scored", False), with_recon=with_recon and domain == "math", gpus=gpus)
        labels_of[domain], rows_of[domain] = labels, rows
    (root / "configs").mkdir(exist_ok=True)
    (root / "configs/final_models.json").write_text(json.dumps(settings), encoding="utf-8")
    (root / "decisions").mkdir(exist_ok=True)
    (root / settings["record"]).write_text("synthetic record\n", encoding="utf-8")
    (root / "analysis").mkdir(exist_ok=True)
    for f in CODE_FILES:
        shutil.copyfile(Path(__file__).parent / Path(f).name, root / f)
    return {"labels": labels_of, "rows": rows_of}
