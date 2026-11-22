"""Step 2 (full bank): the pilot's four generators over the full bank's questions, in resumable chunks (PLAN.md §2).

Prompts, sampling settings and model commits are the pilot's (bank.generate, bank.generate_api and the
pilot's generation manifests). Answers are written one chunk of problems at a time, and a finished
chunk is never generated again, so a rerun after a disconnect continues. With --save every
chunk is copied to a second folder (Google Drive) as soon as it exists, and copied back first when a
new runtime starts empty.

Seeds are the pilot's too: question i uses 544 + i (open models) or 544 + 1000 i + k for sample k (API).
The pilot used some of the same numbers, which is harmless: an answer can only be repeated by the
same prompt, and bank.select_full leaves out every pilot and development problem, by id and by text.

    python -m bank.generate_full run --generator phi4mini --domains math code --save <Drive folder>
    python -m bank.generate_full run --generator gpt41mini --domains math code --save <Drive folder> --budget-usd 8
    python -m bank.generate_full status --domains math code --gen <Drive folder>/gen
    python -m bank.generate_full merge --domain math --gen <Drive folder>/gen --out work/math/generations.jsonl

gpt41mini needs OPENROUTER_API_KEY and no GPU; the open models need a GPU and vLLM. Do not run the API
generator in two places at once: the spend ledger (api_spend.json) is one file. merge checks that
every answer is there, with the pinned model and settings, before anything is labeled.
"""

from __future__ import annotations

import argparse
import filecmp
import json
import os
import shutil
import tempfile
import time
from datetime import datetime, timezone
from pathlib import Path

from bank import generate_api
from bank.common import read_jsonl, sha256_bytes, sha256_text, write_jsonl
from bank.generate import (
    CODE_PROMPT,
    CODE_RESPONSE_PREFIX,
    DOMAIN_DEFAULTS,
    PROMPT,
    _row,
    build_messages,
    environment_info,
    questions_domain,
)

# tag -> (model, commit). The commits are the ones in bank/pilot/gen/*/generations_manifest.json.
GENERATORS = {
    "phi4mini": ("microsoft/Phi-4-mini-instruct", "cfbefacb99257ffa30c83adab238a50856ac3083"),
    "granite33": ("ibm-granite/granite-3.3-8b-instruct", "51dd4bc2ade4059a6bd87649d68aa11e4fb2529b"),
    "olmo2": ("allenai/OLMo-2-1124-7B-Instruct", "470b1fba1ae01581f270116362ee4aa1b97f4c84"),
    "gpt41mini": ("openai/gpt-4.1-mini", None),  # API model: no commit to pin
}
API = "gpt41mini"
SEED = 544
MAX_MODEL_LEN, DTYPE = 4096, "bfloat16"  # as in the pilot
CHUNK = {"open": 200, "api": 50}  # problems per chunk
DOMAINS = ("math", "code")


# ---------------------------------------------------------------- files

def questions_path(domain: str, bank_dir: Path) -> Path:
    return bank_dir / f"questions_{domain}.jsonl"


def folder_name(tag: str, domain: str) -> str:
    return f"{tag}_{domain}"


def chunk_path(folder: Path, first: int) -> Path:
    return folder / "chunks" / f"chunk_{first:05d}.jsonl"


def copy_file(src: Path, dst: Path) -> None:
    """Copy through a temporary name, so an interrupted copy never looks like a finished file."""
    dst.parent.mkdir(parents=True, exist_ok=True)
    tmp = dst.with_name(dst.name + ".tmp")
    shutil.copyfile(src, tmp)
    os.replace(tmp, dst)


def mirror(src: Path, dst: Path, only_missing: bool = False) -> None:
    """Copy the files under src to dst: all that differ, or only those dst does not have."""
    if not src.is_dir():
        return
    for path in sorted(p for p in src.rglob("*") if p.is_file() and p.suffix != ".tmp"):
        target = dst / path.relative_to(src)
        if not target.exists() or not (only_missing or filecmp.cmp(path, target, shallow=False)):
            copy_file(path, target)


def chunk_rows(path: Path) -> list[dict]:
    """The answers in a chunk file, after cutting a half-written last line left by an interruption."""
    if not path.is_file():
        return []
    data = path.read_bytes()
    if data and not data.endswith(b"\n"):
        path.write_bytes(data[: data.rfind(b"\n") + 1])
    return read_jsonl(path)


def spent_usd(ledger: Path) -> float:
    return json.loads(ledger.read_text(encoding="utf-8"))["spent_usd"] if ledger.is_file() else -1.0


# ---------------------------------------------------------------- one (generator, domain)

def settings(tag: str, domain: str) -> dict:
    """Everything that decides the answers besides the questions, fixed for the whole run."""
    model, revision = GENERATORS[tag]
    open_model = {"backend": "vllm", "dtype": DTYPE, "max_model_len": MAX_MODEL_LEN}
    return {"model": model, "revision": revision, "tag": tag, "domain": domain, "seed": SEED,
            **DOMAIN_DEFAULTS[domain], **({"backend": "openrouter"} if tag == API else open_model),
            "chunk_problems": CHUNK["api" if tag == API else "open"]}


def check_state(folder: Path, state: dict) -> None:
    """Record the questions and settings on the first run; refuse to continue with others."""
    path = folder / "state.json"
    if path.is_file():
        before = json.loads(path.read_text(encoding="utf-8"))
        if before != state:
            raise SystemExit(f"{folder} was started with other questions or settings: "
                             f"{sorted(k for k in state if before.get(k) != state[k])}. Use a new folder.")
        return
    folder.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(state, indent=2) + "\n", encoding="utf-8", newline="\n")


def expected_ids(questions: list[dict], tag: str, n_samples: int) -> list[str]:
    return [f"{q['problem_id']}__{tag}__{k}" for q in questions for k in range(n_samples)]


class OpenModel:
    """vLLM engine for one open generator, called as bank.generate.generate_vllm calls it."""

    def __init__(self, tag: str):
        from vllm import LLM

        model, revision = GENERATORS[tag]
        self.llm = LLM(model=model, revision=revision, seed=SEED, dtype=DTYPE, max_model_len=MAX_MODEL_LEN)

    def answers(self, questions: list[dict], first: int, s: dict) -> list[dict]:
        """Rows for questions[first:], where `first` is the position in the whole question file."""
        from vllm import SamplingParams

        params = [SamplingParams(n=s["n_samples"], temperature=s["temperature"], top_p=s["top_p"],
                                 max_tokens=s["max_new_tokens"], seed=s["seed"] + first + i)
                  for i in range(len(questions))]
        prefill = s["domain"] == "code"
        outputs = self.llm.chat([build_messages(q["question"], s["domain"]) for q in questions], params,
                                add_generation_prompt=not prefill, continue_final_message=prefill)
        args = argparse.Namespace(**s)
        return [_row(q, k, c.text, c.finish_reason, len(c.token_ids), args, s["seed"] + first + i)
                for i, (q, out) in enumerate(zip(questions, outputs)) for k, c in enumerate(out.outputs)]


def run_open(folder: Path, questions: list[dict], s: dict, engine, after_chunk) -> None:
    """Generate the missing chunks of one open model. `engine` is called only if a chunk is missing."""
    for first in range(0, len(questions), s["chunk_problems"]):
        part, path = questions[first:first + s["chunk_problems"]], chunk_path(folder, first)
        if len(chunk_rows(path)) == len(part) * s["n_samples"]:
            continue
        rows = engine().answers(part, first, s)
        tmp = path.with_name(path.name + ".tmp")
        write_jsonl(tmp, rows)
        os.replace(tmp, path)
        after_chunk()
        print(f"{folder.name}: {min(first + s['chunk_problems'], len(questions))}/{len(questions)} problems", flush=True)


def run_api(folder: Path, questions: list[dict], s: dict, ledger: Path, budget_usd: float, after_chunk) -> None:
    """Request the missing answers chunk by chunk with bank.generate_api, which resumes inside a chunk."""
    for first in range(0, len(questions), s["chunk_problems"]):
        part = questions[first:first + s["chunk_problems"]]
        path = chunk_path(folder, first)
        before = len(chunk_rows(path))
        if before == len(part) * s["n_samples"]:
            continue
        with tempfile.TemporaryDirectory() as tmp:
            chunk_questions = Path(tmp) / "questions.jsonl"
            write_jsonl(chunk_questions, part)
            # seed + 1000 * first keeps each sample's seed the one a single run over the whole file gives it.
            generate_api.main(["--questions", str(chunk_questions), "--out", str(path), "--model", s["model"],
                               "--tag", s["tag"], "--seed", str(s["seed"] + 1000 * first),
                               "--ledger", str(ledger), "--budget-usd", str(budget_usd)])
        after_chunk()
        # A whole chunk without one answer is a bad key, no credit or the cap; a few that keep failing are not.
        if not before and not chunk_rows(path):
            raise SystemExit(f"{folder.name}: no answer arrived for problems {first}-{first + len(part) - 1}; check "
                             "the key, the credit and the budget cap, then rerun to continue")


def finish(folder: Path, questions_file: Path, questions: list[dict], s: dict, seconds: float) -> bool:
    """Join the chunks into generations.jsonl and write its manifest, once every answer is there."""
    rows = [r for first in range(0, len(questions), s["chunk_problems"])
            if chunk_path(folder, first).is_file() for r in read_jsonl(chunk_path(folder, first))]
    missing = set(expected_ids(questions, s["tag"], s["n_samples"])) - {r["candidate_id"] for r in rows}
    if missing:
        print(f"{folder.name}: {len(missing)} of {len(questions) * s['n_samples']} answers still missing; "
              "rerun to continue")
        return False
    out = folder / "generations.jsonl"
    manifest_path = folder / "generations_manifest.json"
    write_jsonl(out, sorted(rows, key=lambda r: r["candidate_id"]))
    digest = sha256_bytes(out.read_bytes())
    if manifest_path.is_file() and json.loads(manifest_path.read_text(encoding="utf-8"))["sha256"] == digest:
        return True  # finished in an earlier session: keep that run's manifest
    manifest = {
        "step": "generate_full", "questions": questions_file.name,
        "questions_sha256": sha256_bytes(questions_file.read_bytes()),
        "n_questions": len(questions), "n_generations": len(rows), "sha256": digest,
        "generator": s["model"], "generator_revision": s["revision"],
        "prompt": CODE_PROMPT if s["domain"] == "code" else PROMPT,
        "response_prefix": CODE_RESPONSE_PREFIX if s["domain"] == "code" and s["tag"] != API else None,
        "settings": s, "seconds_last_session": round(seconds, 1),
        "finished_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "environment": environment_info(),
    }
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8", newline="\n")
    print(f"{folder.name}: {len(rows)} answers, sha256 {digest}")
    return True


def run(tag: str, domains: list[str], out: Path, save: Path | None, budget_usd: float,
        engine_factory=OpenModel) -> bool:
    """Generate every missing answer of one generator. True when every requested domain is complete."""
    ledger = out / "api_spend.json"
    if save is not None and tag == API and spent_usd(save / ledger.name) > spent_usd(ledger):
        copy_file(save / ledger.name, ledger)  # a fresh clone may hold an older ledger than Drive
    loaded = []

    def engine():  # one engine for all domains, loaded only when something is missing
        if not loaded:
            loaded.append(engine_factory(tag))
        return loaded[0]

    complete = True
    for domain in domains:
        questions_file = questions_path(domain, out)
        questions = read_jsonl(questions_file)
        if questions_domain(questions) != domain:
            raise SystemExit(f"{questions_file} does not hold {domain} questions")
        s = settings(tag, domain)
        folder = out / "gen" / folder_name(tag, domain)
        saved = save / "gen" / folder_name(tag, domain) if save is not None else None
        if saved is not None:
            mirror(saved, folder, only_missing=True)
        check_state(folder, {"questions_sha256": sha256_bytes(questions_file.read_bytes()), "settings": s})

        def after_chunk(folder=folder, saved=saved):
            if saved is not None:
                mirror(folder, saved)
                if tag == API and ledger.is_file():
                    copy_file(ledger, save / ledger.name)

        started = time.time()
        try:
            if tag == API:
                run_api(folder, questions, s, ledger, budget_usd, after_chunk)
            else:
                run_open(folder, questions, s, engine, after_chunk)
            complete &= finish(folder, questions_file, questions, s, time.time() - started)
        finally:
            after_chunk()
    return complete


# ---------------------------------------------------------------- status and merge

def status(domains: list[str], questions_dir: Path, gen: Path) -> None:
    for domain in domains:
        questions = read_jsonl(questions_path(domain, questions_dir))
        for tag in GENERATORS:
            folder = gen / folder_name(tag, domain)
            done = sum(len(read_jsonl(p)) for p in sorted((folder / "chunks").glob("chunk_*.jsonl")))
            state = "complete" if (folder / "generations_manifest.json").is_file() else "not finished"
            print(f"{domain} {tag}: {done}/{len(questions) * DOMAIN_DEFAULTS[domain]['n_samples']} answers, {state}")


def merge(domain: str, gen: Path, questions_file: Path, out: Path) -> dict[str, str]:
    """One file with all four generators' answers, after checking that each set is complete and as pinned.

    Returns generator tag -> sha256 of its generations.jsonl."""
    questions = read_jsonl(questions_file)
    digest = sha256_bytes(questions_file.read_bytes())
    problems, rows, hashes = [], [], {}
    for tag in GENERATORS:
        s, folder = settings(tag, domain), gen / folder_name(tag, domain)
        path, manifest_path = folder / "generations.jsonl", folder / "generations_manifest.json"
        if not path.is_file() or not manifest_path.is_file():
            problems.append(f"{tag}: not finished ({path} or its manifest is missing)")
            continue
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        hashes[tag] = sha256_bytes(path.read_bytes())
        if manifest["sha256"] != hashes[tag]:
            problems.append(f"{tag}: generations.jsonl does not match the hash in its manifest")
        if manifest["questions_sha256"] != digest:
            problems.append(f"{tag}: generated from other questions than {questions_file}")
        if manifest["settings"] != s:
            problems.append(f"{tag}: settings differ from the pinned ones")
        mine = read_jsonl(path)
        if sorted(r["candidate_id"] for r in mine) != sorted(expected_ids(questions, tag, s["n_samples"])):
            problems.append(f"{tag}: the candidate ids are not exactly one set of answers per question")
        if any(r["generator"] != s["model"] or (s["revision"] and r["generator_revision"] != s["revision"])
               or r["text_sha256"] != sha256_text(r["text"]) for r in mine):
            problems.append(f"{tag}: some rows have another model, commit or text hash")
        rows += mine
    if problems:
        raise SystemExit(f"{domain}: cannot merge:\n- " + "\n- ".join(problems))
    write_jsonl(out, sorted(rows, key=lambda r: r["candidate_id"]))
    print(f"{domain}: {len(rows)} answers from {len(hashes)} generators -> {out}")
    print(json.dumps(hashes, indent=2))
    return hashes


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = ap.add_subparsers(dest="command", required=True)
    p = sub.add_parser("run")
    p.add_argument("--generator", required=True, choices=sorted(GENERATORS))
    p.add_argument("--domains", nargs="+", required=True, choices=DOMAINS)
    p.add_argument("--out", default="bank/full", help="folder with the question files; answers go to <out>/gen")
    p.add_argument("--save", default=None, help="second folder (Drive) that every finished chunk is copied to")
    p.add_argument("--budget-usd", type=float, default=8.0, help="hard cap on the full bank's API spend")
    p = sub.add_parser("status")
    p.add_argument("--domains", nargs="+", required=True, choices=DOMAINS)
    p.add_argument("--gen", default="bank/full/gen", help="folder with one <tag>_<domain> folder per generator")
    p.add_argument("--questions-dir", default="bank/full")
    p = sub.add_parser("merge")
    p.add_argument("--domain", required=True, choices=DOMAINS)
    p.add_argument("--gen", required=True, help="folder with one <tag>_<domain> folder per generator")
    p.add_argument("--questions-dir", default="bank/full")
    p.add_argument("--out", required=True)
    args = ap.parse_args(argv)

    if args.command == "run":
        save = Path(args.save) if args.save else None
        if not run(args.generator, args.domains, Path(args.out), save, args.budget_usd):
            raise SystemExit(1)
    elif args.command == "status":
        status(args.domains, Path(args.questions_dir), Path(args.gen))
    else:
        merge(args.domain, Path(args.gen), questions_path(args.domain, Path(args.questions_dir)), Path(args.out))


if __name__ == "__main__":
    main()
