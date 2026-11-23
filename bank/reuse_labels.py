"""Full bank: keep a domain's finished labels, and take them over into a new version instead of checking again.

For rebuilding a bank whose labels have not changed, such as the code bank without MBPP+. For the label
keeper only. Nothing is run.

save keeps the labels, and the pairs made from them, in the private folder as soon as the pairs exist, so a
later step that stops does not lose them. Finished labels already there (a summary that fits labeled.jsonl) are
never replaced by different ones; files left half copied by an interruption are.

reuse copies an earlier version's labeled.jsonl into the new version's work folder. It stops unless
- that version labeled the same answers to the same questions, and its pairs were built from these labels;
- every label comes from the current checker (CHECKERS), so a later fix to a checker is never skipped;
- the earlier version's public folder for the domain is gone. The new bank is split again, so an answer that
  was in the old validation split can be in the new test split: the old validation labels, which are meant
  to be committed, would give away test labels.
Pairs, length filter, draw and split are then redone from the labels.

    python -m bank.reuse_labels save --work work_v1/code --keep <private>/v1/code/work
    python -m bank.reuse_labels reuse --old <private>/v1/code/work --work work_v2/code \\
        --questions bank/full/questions_code.jsonl --old-public <to_commit>/v1/bank/full/code bank/full/code
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
from pathlib import Path

from bank import label, label_evalplus
from bank.common import read_jsonl, sha256_bytes
from bank.generate import questions_domain

# The labelers' versions with the checkers notebook 11 installs.
CHECKERS = {"math": f"{label.LABELER_VERSION}+math_verify==0.9.0",
            "code": f"{label_evalplus.LABELER_VERSION}+evalplus==0.3.1+colab-guard"}
# The summary stays last: it marks the labels as finished.
KEPT = ("generations.jsonl", "labeled.jsonl", "full_bank.jsonl", "full_labels.jsonl", "full_bank_summary.json")


def made_from(summary: Path) -> dict:
    """File name -> sha256 of the pairs' inputs and outputs in a make_pairs summary."""
    data = json.loads(summary.read_text(encoding="utf-8"))
    return {**data["inputs"], **data["outputs"]}


def copy_file(src: Path, dst: Path) -> None:
    """Copy through a temporary name, so an interrupted copy never looks like a finished file."""
    tmp = dst.with_name(dst.name + ".tmp")
    shutil.copyfile(src, tmp)
    os.replace(tmp, dst)


def finished(folder: Path) -> dict | None:
    """What a kept folder's labels and pairs were made from, or None if the labels are missing or half copied."""
    try:
        made = made_from(folder / "full_bank_summary.json")
        return made if made["labeled.jsonl"] == sha256_bytes((folder / "labeled.jsonl").read_bytes()) else None
    except (OSError, ValueError, KeyError, TypeError):
        return None


def save(work: Path, keep: Path) -> int:
    """Keep the labels and pairs in the private folder. Returns how many files were written there."""
    made = finished(work)
    if made is None or any(not (work / n).is_file() or sha256_bytes((work / n).read_bytes()) != made.get(n)
                           for n in KEPT[:-1]):
        raise SystemExit(f"{work} does not hold the pairs and the files they were made from; make the pairs first")
    kept = finished(keep)
    if kept is not None and kept != made:
        raise SystemExit(f"{keep} holds other finished labels or pairs; they are never replaced. Use a new VERSION")
    keep.mkdir(parents=True, exist_ok=True)
    # What is missing or half copied. The summary is last, and only if the folder was not finished: it marks it so.
    write = [n for n in KEPT[:-1] if not (keep / n).is_file() or (keep / n).read_bytes() != (work / n).read_bytes()]
    write += KEPT[-1:] if kept is None else ()
    for name in write:
        copy_file(work / name, keep / name)
    return len(write)


def reuse(old: Path, work: Path, questions: Path, old_public: list[Path]) -> int:
    """Copy the earlier labels into the work folder. Returns how many answers they cover."""
    present = [p for p in old_public if p.exists()]
    if present:
        raise SystemExit(f"the old bank is still at {', '.join(map(str, present))}. If you did not mean to rebuild this "
                         "domain, take it out of DOMAINS and delete nothing. If you did, delete it first and make sure "
                         "nobody has a copy: its validation labels can be test labels of the new bank. If it was shared "
                         "or committed, stop and tell the team")
    made = finished(old)
    if made is None:
        raise SystemExit(f"{old} holds no finished labels. Run that version again first, with REUSE_LABELS_FROM = \"\": its "
                         "saved checks are reused and its labels are kept as soon as the pairs are made. If it then stops "
                         "because a different bank or other test labels are already there, that is expected: come back here")
    for path in (questions, work / "generations.jsonl"):
        if made.get(path.name) != sha256_bytes(path.read_bytes()):
            raise SystemExit(f"{path} is not the file the earlier version in {old} was built from")
    labels = read_jsonl(old / "labeled.jsonl")
    if [r["candidate_id"] for r in labels] != [g["candidate_id"] for g in read_jsonl(work / "generations.jsonl")]:
        raise SystemExit(f"{old / 'labeled.jsonl'} does not cover exactly the answers in {work / 'generations.jsonl'}")
    checker = CHECKERS[questions_domain(read_jsonl(questions))]
    others = sorted({str(r.get("checker_version")) for r in labels} - {checker})
    if others:
        raise SystemExit(f"{old / 'labeled.jsonl'} was not labeled by the current checker {checker} but by {others}; "
                         "label again with REUSE_LABELS_FROM = \"\"")
    copy_file(old / "labeled.jsonl", work / "labeled.jsonl")
    return len(labels)


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = ap.add_subparsers(dest="command", required=True)
    p = sub.add_parser("save")
    p.add_argument("--work", required=True)
    p.add_argument("--keep", required=True, help="this version's private work folder for this domain")
    p = sub.add_parser("reuse")
    p.add_argument("--old", required=True, help="the earlier version's private work folder for this domain")
    p.add_argument("--work", required=True, help="this version's work folder, with the merged generations.jsonl")
    p.add_argument("--questions", required=True)
    p.add_argument("--old-public", required=True, nargs="+", help="folders of the old bank that must not exist any more")
    args = ap.parse_args(argv)
    if args.command == "save":
        print(f"{save(Path(args.work), Path(args.keep))} files kept in {args.keep}")
    else:
        n = reuse(Path(args.old), Path(args.work), Path(args.questions), [Path(p) for p in args.old_public])
        print(f"labels of {n} answers reused from {args.old}; nothing was run")


if __name__ == "__main__":
    main()
