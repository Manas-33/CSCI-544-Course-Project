"""Step 6 (full bank): split one domain's drawn bank by problem, and draw the math error-tag sample (PLAN.md §2).

split writes, per domain:
- <out>/bank.jsonl: every answer, with split = "validation" or "test" and no label. Pairs are in order of
  pair id and the two answers of a pair in order of their text's SHA-256, so the order says nothing
  about which is correct;
- <out>/labels_validation.jsonl and <out>/manifest.json (hashes, counts, shortfalls, seeds);
- <private>/labels_test.jsonl, in a separate folder that only the label keeper has.

The split is by problem: one third of each source's problems go to validation, the rest to test, seed
544. Near-duplicate problems (bank.select_full.near_duplicate) stay together and count for the source of
the first of them in id order. A third of n problems is rounded to the nearest whole number.
Nothing is written over a different bank or other test labels: a corrected bank is a new version, with
new <out> and <private> folders.

    python -m bank.split_full split --domain math --work work/math --out bank/full/math --private <private>/math
    python -m bank.split_full error-sheet --out bank/full/math --private <private>/math
    python -m bank.split_full apply-tags --out bank/full/math --private <private>/math

error-sheet draws 200 wrong math test answers (uniform, seed 544; all of them if there are fewer) into
<private>/error_tags_math.csv, where people fill in error_type. apply-tags copies the tags
into labels_test.jsonl and updates its hash in the manifest; it runs before the bank freeze. Both are
for the label keeper only: the sheet lists wrong test answers.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import random
import subprocess
from collections import Counter, defaultdict
from pathlib import Path

from bank.common import read_jsonl, sha256_bytes
from bank.draw_full import LEFT_OUT
from bank.select_full import near_duplicate_groups

SPLITS = ("validation", "test")
# Nothing a verifier or a scorer may see.
LABEL_FIELDS = {"label", "reference", "extracted", "status", "checker1_boxed", "checker2", "checker2_name",
                "error_type", "run_statuses", "detail"}
ERROR_TAGS = ("arithmetic_slip", "misread_problem", "wrong_method", "formatting")  # PLAN.md §2
TAG_SHEET = "error_tags_math.csv"
N_TAGGED = 200


def split_problems(problems: dict[str, tuple[str, str]], seed: int) -> tuple[dict[str, str], list[list[str]]]:
    """problem id -> split, and the near-duplicate groups kept together. `problems` is id -> (source, text)."""
    groups = near_duplicate_groups({pid: text for pid, (_, text) in problems.items()})
    by_source = defaultdict(list)
    for group in groups:
        by_source[problems[group[0]][0]].append(group)
    assignment = {}
    for source in sorted(by_source):
        members = by_source[source]
        random.Random(f"{seed}:full-split:{source}").shuffle(members)
        quota = (sum(len(g) for g in members) + 1) // 3  # a third, rounded to the nearest whole number
        filled = 0
        for group in members:
            to_validation = filled + len(group) <= quota
            filled += len(group) if to_validation else 0
            assignment.update({pid: "validation" if to_validation else "test" for pid in group})
    return assignment, [g for g in groups if len(g) > 1]


def counts(rows: list[dict]) -> dict:
    """Problems, pairs and answers of one split, in total and by source and generator (pairs)."""
    pairs = {r["pair_id"]: r for r in rows}
    by = lambda key: dict(sorted(Counter(p[key] for p in pairs.values()).items()))  # noqa: E731
    return {"problems": len({r["problem_id"] for r in rows}), "pairs": len(pairs), "candidates": len(rows),
            "pairs_by_source": by("source"), "pairs_by_generator": by("generator")}


def code_revision() -> str | None:
    try:
        return subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=True).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def _write(path: Path, data: bytes) -> None:
    """Write through a temporary name, so an interruption never leaves half a file."""
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_bytes(data)
    os.replace(tmp, path)


def _dump(path: Path, data: dict) -> None:
    _write(path, (json.dumps(data, indent=2) + "\n").encode("utf-8"))


def _jsonl_bytes(rows: list[dict]) -> bytes:
    return "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows).encode("utf-8")


def split(domain: str, work: Path, out: Path, private: Path, seed: int = 544) -> dict:
    out, private = out.resolve(), private.resolve()
    if out.parent in (private, *private.parents) or private in out.parents:
        raise SystemExit(f"--private must be outside {out.parent} and not a folder above it: the test labels stay with "
                         "the label keeper (PLAN.md §2)")
    if any((folder / ".git").exists() for folder in (private, *private.parents)):
        raise SystemExit(f"--private {private} is inside a git checkout; the test labels are kept outside (PLAN.md §2)")
    drawn = json.loads((work / "draw_summary.json").read_text(encoding="utf-8"))
    for name, digest in drawn["outputs"].items():
        if sha256_bytes((work / name).read_bytes()) != digest:
            raise SystemExit(f"{work / name} is not the file draw_summary.json describes; rerun bank.draw_full")
    if drawn["domain"] != domain:
        raise SystemExit(f"{work} holds the {drawn['domain']} draw, not {domain}")
    rows = read_jsonl(work / "drawn_bank.jsonl")
    left_out = sorted({r["source"] for r in rows} & set(LEFT_OUT[domain]))
    if left_out:
        raise SystemExit(f"{work} was drawn before {left_out} was left out of the bank; rerun bank.draw_full")
    labels = {r["candidate_id"]: r for r in read_jsonl(work / "drawn_labels.jsonl")}
    if set(labels) != {r["candidate_id"] for r in rows} or len(labels) != len(rows):
        raise SystemExit("drawn_bank.jsonl and drawn_labels.jsonl do not hold the same answers")
    assignment, together = split_problems({r["problem_id"]: (r["source"], r["problem"]) for r in rows}, seed)
    bank = [{**r, "split": assignment[r["problem_id"]]}
            for r in sorted(rows, key=lambda r: (r["pair_id"], r["text_sha256"], r["candidate_id"]))]
    leaked = {k for r in bank for k in r} & LABEL_FIELDS
    if leaked:
        raise SystemExit(f"the bank rows carry label fields: {sorted(leaked)}")
    of = {s: [r for r in bank if r["split"] == s] for s in SPLITS}

    files = {"bank.jsonl": _jsonl_bytes(bank),
             "labels_validation.jsonl": _jsonl_bytes([labels[r["candidate_id"]] for r in of["validation"]]),
             "labels_test.jsonl": _jsonl_bytes([labels[r["candidate_id"]] for r in of["test"]])}
    hashes = {name: sha256_bytes(data) for name, data in files.items()}
    manifest_path = out / "manifest.json"
    if manifest_path.is_file() and json.loads(manifest_path.read_text(encoding="utf-8"))["files"] != hashes:
        raise SystemExit(f"{out} already holds a different bank, or test labels with error tags; a corrected bank "
                         "is a new version, with a new --out and a new --private folder (PLAN.md §2)")
    kept = private / "labels_test.jsonl"
    if kept.is_file() and sha256_bytes(kept.read_bytes()) != hashes["labels_test.jsonl"]:
        raise SystemExit(f"{kept} holds other test labels, or labels with error tags; they are never written over. "
                         "Use a new --private folder for a new version (PLAN.md §2)")
    manifest = {
        "bank": "full", "domain": domain, "seed": seed, "files": hashes,
        "labels_test": "held by the label keeper, not in the repository before the test join (PLAN.md §2)",
        "counts": {"total": counts(bank), **{s: counts(of[s]) for s in SPLITS}},
        "split_rule": "by problem, stratified by source, one third validation, near-duplicate problems together",
        "near_duplicate_problems_kept_together": together,
        "share_of_cells_that_qualified": drawn["cells"],
        "shortfall": drawn["shortfall"],
        "draw": {k: v for k, v in drawn.items()
                 if k not in ("cells", "shortfall", "inputs", "outputs", "step", "domain")},
        "inputs": {**drawn["inputs"], **drawn["outputs"]},
        "code_revision": code_revision(),
    }
    for folder in (out, private):
        folder.mkdir(parents=True, exist_ok=True)
    for name, data in files.items():
        _write((private if name == "labels_test.jsonl" else out) / name, data)
    _dump(manifest_path, manifest)
    return manifest


def read_test_labels(out: Path, private: Path) -> tuple[dict, list[dict]]:
    """(manifest, test label rows), checked against the manifest's hash."""
    manifest = json.loads((out / "manifest.json").read_text(encoding="utf-8"))
    path = private / "labels_test.jsonl"
    if sha256_bytes(path.read_bytes()) != manifest["files"]["labels_test.jsonl"]:
        raise SystemExit(f"{path} does not match the hash in {out / 'manifest.json'}")
    return manifest, read_jsonl(path)


def drawn_for_tagging(labels: list[dict], n: int, seed: int) -> list[str]:
    """Candidate ids of the wrong test answers to tag: a uniform seeded draw, or all of them if there are fewer."""
    wrong = sorted(r["candidate_id"] for r in labels if r["label"] == 0)
    return random.Random(f"{seed}:error-tags").sample(wrong, min(n, len(wrong)))


def error_sheet(out: Path, private: Path, n: int = N_TAGGED, seed: int = 544) -> int:
    """Write the sheet of wrong math test answers to tag. Returns how many were drawn."""
    manifest, labels = read_test_labels(out, private)
    if manifest["domain"] != "math":
        raise SystemExit("only math answers are tagged by hand; code error types (wrong_output, timeout) come from EvalPlus")
    path = private / TAG_SHEET
    if path.is_file():
        raise SystemExit(f"{path} already exists; it may hold tags")
    bank = {r["candidate_id"]: r for r in read_jsonl(out / "bank.jsonl")}
    by_id = {r["candidate_id"]: r for r in labels}
    picked = drawn_for_tagging(labels, n, seed)
    with open(path, "w", encoding="utf-8", newline="") as f:
        sheet = csv.writer(f, lineterminator="\n")
        sheet.writerow(["candidate_id", "problem", "candidate", "reference", "extracted", "error_type"])
        sheet.writerows([c, bank[c]["problem"], bank[c]["candidate"], by_id[c]["reference"], by_id[c]["extracted"], ""]
                        for c in picked)
    return len(picked)


def apply_tags(out: Path, private: Path, n: int = N_TAGGED, seed: int = 544) -> int:
    """Copy the filled-in tags into labels_test.jsonl and update its hash in the manifest. Can be repeated."""
    path = private / "labels_test.jsonl"
    manifest = json.loads((out / "manifest.json").read_text(encoding="utf-8"))
    labels = read_jsonl(path)
    untagged = [{k: v for k, v in r.items() if k != "error_type"} for r in labels]
    # The second case: an earlier apply-tags stopped after writing the labels and before the manifest.
    if manifest["files"]["labels_test.jsonl"] not in (sha256_bytes(path.read_bytes()), sha256_bytes(_jsonl_bytes(untagged))):
        raise SystemExit(f"{path} does not match the hash in {out / 'manifest.json'}")
    if manifest["domain"] != "math":
        raise SystemExit("only math answers are tagged by hand; code error types (wrong_output, timeout) come from EvalPlus")
    with open(private / TAG_SHEET, encoding="utf-8-sig", newline="") as f:
        sheet = list(csv.DictReader(f))
    tags = {r["candidate_id"]: r["error_type"].strip() for r in sheet}
    if len(tags) != len(sheet) or set(tags) != set(drawn_for_tagging(labels, n, seed)):
        raise SystemExit("the tag sheet does not hold exactly the answers that were drawn for tagging, once each")
    bad = sorted(c for c, t in tags.items() if t not in ERROR_TAGS)
    if bad:
        raise SystemExit(f"{len(bad)} tags are missing or not one of {ERROR_TAGS}")
    data = _jsonl_bytes([{**r, "error_type": tags[r["candidate_id"]]} if r["candidate_id"] in tags else r
                         for r in untagged])
    _write(path, data)
    manifest["files"]["labels_test.jsonl"] = sha256_bytes(data)
    manifest["math_error_tags"] = len(tags)
    _dump(out / "manifest.json", manifest)
    return len(tags)


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = ap.add_subparsers(dest="command", required=True)
    for name in ("split", "error-sheet", "apply-tags"):
        p = sub.add_parser(name)
        p.add_argument("--out", required=True, help="folder with bank.jsonl, labels_validation.jsonl, manifest.json")
        p.add_argument("--private", required=True, help="the label keeper's own folder for labels_test.jsonl")
        p.add_argument("--seed", type=int, default=544)
        if name == "split":
            p.add_argument("--domain", required=True, choices=("math", "code"))
            p.add_argument("--work", required=True, help="bank.draw_full output")
        else:
            p.add_argument("--n", type=int, default=N_TAGGED, help="wrong math test answers to tag")
    args = ap.parse_args(argv)
    out, private = Path(args.out), Path(args.private)

    if args.command == "split":
        manifest = split(args.domain, Path(args.work), out, private, args.seed)
        print(json.dumps({"domain": args.domain, "files": manifest["files"], "counts": manifest["counts"],
                          "shortfall": manifest["shortfall"],
                          "cells_that_qualified": manifest["share_of_cells_that_qualified"]["all"]}, indent=2))
    elif args.command == "error-sheet":
        print(f"{error_sheet(out, private, args.n, args.seed)} wrong math test answers to tag in "
              f"{private / TAG_SHEET}; tags: {', '.join(ERROR_TAGS)}")
    else:
        print(f"{apply_tags(out, private, args.n, args.seed)} tags added to labels_test.jsonl; the manifest has its new hash")


if __name__ == "__main__":
    main()
