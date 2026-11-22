"""Pick the full bank's problems (PLAN.md §2): all that the pilot left, plus 500 new MATH problems.

Math: every unused GSM8K-Platinum and MATH-500 problem (MATH-500: numeric answers only, as in the
pilot), and a uniform seeded draw of 500 numeric-answer problems from the rest of the MATH test
set (PRM800K's math_splits/train.jsonl, rows whose unique_id starts with "test/"). Code: every
unused HumanEval+ and MBPP+ problem.

Left out, by id and by text: pilot and development problems, and, for the new MATH problems, all
500 MATH-500 problems. "By text" covers near-duplicates: two problems whose sets of words,
numbers and operators have a Jaccard similarity above 0.9, the bar bank.make_pairs uses for
answers. Whatever this removes is listed in the manifest. The counts expected from the plan
(1,059, 239 and 392 unused problems, 4,500 new MATH problems) are checked before it. The PRM800K file
lists one test problem twice, with the same answer and two solutions; the first copy is kept.

    python -m bank.select_full --sources <folder with the downloaded files> --out bank/full

The folder holds the pilot's four source files (bank.select_pilot.SOURCES; the EvalPlus ones
unzipped), whose hashes must match bank/pilot/questions_manifest.json, and
prm800k_math_splits_train.jsonl. That one is stored with Git LFS, so the raw GitHub URL gives a
133-byte pointer; use the URL in SOURCES. If the pointer is saved next to it as
prm800k_math_splits_train.pointer, the file is checked against the pointer's hash. License texts
saved as licenses/<name>.txt (names in LICENSES) go into the manifest, long ones cut. A field, count or
hash that is not as expected stops the run.
"""

from __future__ import annotations

import argparse
import json
import re
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

from bank.common import extract_boxed, parse_rational, read_jsonl, sha256_bytes, write_jsonl
from bank.make_pairs import NEAR_DUPLICATE_JACCARD
from bank.select_pilot import SOURCES as PILOT_SOURCES
from bank.select_pilot import evalplus, gsm8k_platinum, math500, sample

PRM800K = "prm800k_math_splits_train.jsonl"
SOURCES = {
    **PILOT_SOURCES,
    PRM800K: "https://media.githubusercontent.com/media/openai/prm800k/main/prm800k/math_splits/train.jsonl",
}
REQUIRED_FIELDS = {
    "gsm8k_platinum_test.parquet": ("question", "answer"),
    "math500_test.jsonl": ("problem", "answer", "subject", "level", "unique_id"),
    "HumanEvalPlus.jsonl": ("task_id", "prompt", "entry_point"),
    "MbppPlus.jsonl": ("task_id", "prompt", "entry_point"),
    PRM800K: ("problem", "solution", "answer", "subject", "level", "unique_id"),
}
LICENSES = ("gsm8k-platinum", "math-500", "prm800k", "evalplus")
LICENSE_CHARS = 1500

NEW_MATH = 500
# Unused problems left by the pilot (PLAN.md §2, NEXT_STEPS.md A and B), and the MATH test problems
# outside MATH-500 (5,000 - 500).
EXPECTED = {"gsm8k": 1059, "math500": 239, "humaneval": 114, "mbpp": 278, "math500_rows": 500, "math_rest": 4500}

LFS_POINTER = b"version https://git-lfs.github.com/spec/v1"


# ---------------------------------------------------------------- near-duplicate problems

_TOKEN_RE = re.compile(r"[a-z]+|\d+(?:\.\d+)?|[+\-*/=<>^!%|]")


def problem_tokens(text: str) -> frozenset[str]:
    """Words, numbers and operators of a problem, ignoring case, spacing, punctuation and LaTeX layout."""
    s = re.sub(r"\\[dt]frac", r"\\frac", text.lower())
    s = re.sub(r"\\(?:left|right|displaystyle)\b", " ", s)
    return frozenset(_TOKEN_RE.findall(s))


def near_duplicate(a: frozenset[str], b: frozenset[str]) -> bool:
    small, large = sorted((len(a), len(b)))
    if not large or small <= NEAR_DUPLICATE_JACCARD * large:  # the Jaccard index is at most small / large
        return False
    return len(a & b) > NEAR_DUPLICATE_JACCARD * len(a | b)


def near_duplicate_groups(problems: dict[str, str]) -> list[list[str]]:
    """Problem ids grouped with their near-duplicates (chains included). Most groups hold one problem."""
    ids = sorted(problems)
    tokens = [problem_tokens(problems[i]) for i in ids]
    parent = list(range(len(ids)))

    def root(i: int) -> int:
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    for i in range(len(ids)):
        for j in range(i):
            if near_duplicate(tokens[i], tokens[j]):
                a, b = sorted((root(i), root(j)))
                parent[b] = a
    groups: dict[int, list[str]] = {}
    for i, pid in enumerate(ids):
        groups.setdefault(root(i), []).append(pid)
    return [groups[k] for k in sorted(groups)]


def drop_near_duplicates(rows: list[dict], known: dict[str, str]) -> tuple[list[dict], list[dict]]:
    """(rows kept, what was dropped) after removing near-duplicates of the known problems (id -> text)."""
    others = [(pid, problem_tokens(text)) for pid, text in sorted(known.items())]
    kept, dropped = [], []
    for row in rows:
        tokens = problem_tokens(row["question"])
        match = next((pid for pid, t in others if near_duplicate(tokens, t)), None)
        if match is None:
            kept.append(row)
        else:
            dropped.append({"problem_id": row["problem_id"], "source": row["source"], "like": match})
    return kept, dropped


# ---------------------------------------------------------------- sources

def read_source(path: Path) -> list[dict]:
    """Rows of a source file (.jsonl or .parquet), refusing a Git LFS pointer."""
    if not path.is_file():
        raise SystemExit(f"{path} is missing; download it from {SOURCES.get(path.name, 'its source')}")
    data = path.read_bytes()
    if data.startswith(LFS_POINTER):
        raise SystemExit(f"{path} is a Git LFS pointer ({len(data)} bytes), not the data. Download the real file "
                         f"from {SOURCES.get(path.name, 'media.githubusercontent.com')}")
    if path.suffix == ".parquet":
        import pyarrow.parquet as pq

        return pq.read_table(path).to_pylist()
    rows = []
    for n, line in enumerate(data.decode("utf-8").split("\n"), 1):
        if line.strip():
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise SystemExit(f"{path}, line {n}: not JSON ({exc}); is this the right file?") from exc
    return rows


def check_fields(name: str, rows: list[dict]) -> dict[str, int]:
    """Field -> number of rows that have it. Every required field must be in every row."""
    seen = Counter(k for r in rows for k in r)
    missing = {f: len(rows) - seen[f] for f in REQUIRED_FIELDS[name] if seen[f] != len(rows)}
    if not rows or missing:
        raise SystemExit(f"{name}: {len(rows)} rows, fields {sorted(seen)}; rows without a required field: {missing}")
    not_text = [f for f in REQUIRED_FIELDS[name] if f != "level" and any(not isinstance(r[f], str) for r in rows)]
    if not_text:
        raise SystemExit(f"{name}: fields that are not text in every row: {not_text}")
    return dict(sorted(seen.items()))


def check_pointer(path: Path, pointer: Path) -> str:
    """Compare a downloaded file with its Git LFS pointer, if the pointer was saved too."""
    if not pointer.is_file():
        return "no pointer saved, not checked"
    text = pointer.read_text(encoding="utf-8")
    oid, size = re.search(r"oid sha256:([0-9a-f]{64})", text), re.search(r"size (\d+)", text)
    data = path.read_bytes()
    if not oid or not size or oid[1] != sha256_bytes(data) or int(size[1]) != len(data):
        raise SystemExit(f"{path} does not match its Git LFS pointer {pointer}")
    return "matches"


def math_test_rows(rows: list[dict]) -> tuple[list[tuple[int, dict]], list[str]]:
    """((row number, row) of every MATH test problem, ids that were repeated), keeping the first copy of each.

    Two copies that differ in problem or answer stop the run."""
    first: dict[str, tuple[int, dict]] = {}
    repeated = []
    for i, row in enumerate(rows):
        uid = row["unique_id"]
        if not uid.startswith("test/"):
            continue
        if uid not in first:
            first[uid] = (i, row)
            continue
        repeated.append(uid)
        if any(row[k] != first[uid][1][k] for k in ("problem", "answer")):
            raise SystemExit(f"{PRM800K}: {uid} appears twice with different problems or answers")
    return list(first.values()), repeated


def math_rest(rows: list[tuple[int, dict]]) -> list[dict]:
    """Questions from the MATH test problems outside MATH-500 whose answer is numeric (the pilot's rule)."""
    out = []
    for i, row in rows:
        ref = parse_rational(row["answer"])
        if ref is None:
            continue
        out.append({"problem_id": f"mathrest-{i:05d}", "source": "math-rest/test", "source_index": i,
                    "source_id": row["unique_id"], "domain": "math", "question": row["problem"],
                    "reference": ref, "reference_latex": row["answer"], "answer_kind": "rational",
                    "subject": row["subject"], "level": row["level"]})
    return out


def expect(what: str, found: int, wanted: int) -> None:
    if found != wanted:
        raise SystemExit(f"{what}: expected {wanted}, found {found}. Settle why before going on (PLAN.md §2).")


# ---------------------------------------------------------------- selection

def select(sources: Path, bank_dir: Path, seed: int = 544) -> tuple[list[dict], list[dict], dict]:
    """(math questions, code questions, manifest without the output hashes)."""
    pilot = json.loads((bank_dir / "pilot" / "questions_manifest.json").read_text(encoding="utf-8"))
    raw = {name: read_source(sources / name) for name in SOURCES}
    fields = {name: check_fields(name, rows) for name, rows in raw.items()}
    hashes = {name: sha256_bytes((sources / name).read_bytes()) for name in SOURCES}
    changed = [n for n in PILOT_SOURCES if hashes[n] != pilot["sources"][n]["sha256"]]
    if changed:
        raise SystemExit(f"Not the files the pilot used (sha256 differs from its manifest): {changed}")
    pointer = check_pointer(sources / PRM800K, sources / f"{Path(PRM800K).stem}.pointer")

    pools = {
        "gsm8k": gsm8k_platinum(sources / "gsm8k_platinum_test.parquet"),
        "math500": math500(sources / "math500_test.jsonl"),
        "humaneval": evalplus(sources / "HumanEvalPlus.jsonl", "humaneval"),
        "mbpp": evalplus(sources / "MbppPlus.jsonl", "mbpp"),
    }
    for key, pilot_key in (("gsm8k", "gsm8k"), ("math500", "math"), ("humaneval", "humaneval"), ("mbpp", "mbpp")):
        expect(f"{key} pool", len(pools[key]), pilot["pool_sizes"][pilot_key])
    expect("MATH-500 rows", len(raw["math500_test.jsonl"]), EXPECTED["math500_rows"])

    # By id: what the pilot used, and (none expected) what the development banks used.
    used = set((bank_dir / "pilot" / "question_ids.txt").read_text(encoding="utf-8").split())
    unknown = used - {q["problem_id"] for rows in pools.values() for q in rows}
    if unknown:
        raise SystemExit(f"Pilot problems that are not in the source files: {sorted(unknown)[:5]}")
    dev_code = read_jsonl(bank_dir / "dev_code" / "questions.jsonl")
    dev_tasks = {f"mbpp-plus-{int(q['problem_id'].rsplit('-', 1)[1]):03d}" for q in dev_code}
    unused = {k: [q for q in rows if q["problem_id"] not in used | dev_tasks] for k, rows in pools.items()}
    for key, rows in unused.items():
        expect(f"unused {key} problems", len(rows), EXPECTED[key])

    rest_rows, repeated = math_test_rows(raw[PRM800K])
    expect("PRM800K problems whose unique_id starts with 'test/'", len(rest_rows), EXPECTED["math_rest"])
    math500_ids = {r["unique_id"] for r in raw["math500_test.jsonl"]}
    numeric = math_rest(rest_rows)
    by_id = [q for q in numeric if q["source_id"] in math500_ids]
    rest = [q for q in numeric if q["source_id"] not in math500_ids]

    # By text: pilot, development and (new MATH only) MATH-500 problems and their near-duplicates.
    texts = {"math": {}, "code": {}}
    for path, domain in ((bank_dir / "pilot" / "questions_math.jsonl", "math"),
                         (bank_dir / "dev" / "questions.jsonl", "math"),
                         (bank_dir / "pilot" / "questions_code.jsonl", "code"),
                         (bank_dir / "dev_code" / "questions.jsonl", "code")):
        texts[domain].update({q["problem_id"]: q["question"] for q in read_jsonl(path)})
    math500_texts = {f"MATH-500 {r['unique_id']}": r["problem"] for r in raw["math500_test.jsonl"]}
    dropped = []
    for key, known in (("gsm8k", texts["math"]), ("math500", texts["math"]),
                       ("humaneval", texts["code"]), ("mbpp", texts["code"])):
        unused[key], gone = drop_near_duplicates(unused[key], known)
        dropped += gone
    eligible, rest_dropped = drop_near_duplicates(rest, {**texts["math"], **math500_texts})
    new = sample(eligible, NEW_MATH, seed, "math-rest")

    math_q = unused["gsm8k"] + unused["math500"] + new
    code_q = unused["humaneval"] + unused["mbpp"]
    ids = [q["problem_id"] for q in math_q + code_q]
    if len(set(ids)) != len(ids) or set(ids) & used:
        raise SystemExit("Selected problem ids are not unique, or one was in the pilot")
    solutions = {r["unique_id"]: r["solution"] for _, r in rest_rows}
    unlike = [q["problem_id"] for q in new
              if parse_rational(extract_boxed(solutions[q["source_id"]])) != q["reference"]]

    licenses = {}
    for name in LICENSES:
        path = sources / "licenses" / f"{name}.txt"
        data = path.read_bytes() if path.is_file() else None
        licenses[name] = None if data is None else {"sha256": sha256_bytes(data), "bytes": len(data),
                                                    "text_start": data.decode("utf-8")[:LICENSE_CHARS]}
        if data is None:
            print(f"WARNING: no license text for {name} ({path} is missing)")
    manifest = {
        "step": "select_full",
        "seed": seed,
        "sources": {name: {"url": url, "sha256": hashes[name], "bytes": (sources / name).stat().st_size,
                           "rows": len(raw[name]), "fields": fields[name]} for name, url in SOURCES.items()},
        "prm800k_lfs_pointer": pointer,
        "licenses": licenses,
        "pool_sizes": {k: len(v) for k, v in pools.items()},
        "near_duplicate_rule": f"token Jaccard above {NEAR_DUPLICATE_JACCARD} (bank.select_full.problem_tokens)",
        "dropped_near_duplicates": dropped,
        "new_math": {"problems": len(rest_rows), "repeated_ids": repeated, "numeric": len(numeric),
                     "also_in_math500_by_id": len(by_id),
                     "dropped_near_duplicates": rest_dropped, "eligible": len(eligible), "drawn": len(new),
                     "answer_differs_from_boxed_solution": unlike},
        "selected": {"gsm8k": len(unused["gsm8k"]), "math500": len(unused["math500"]), "math_rest": len(new),
                     "humaneval": len(unused["humaneval"]), "mbpp": len(unused["mbpp"])},
        "math_rule": "reference answer parses as a rational number (bank.common.parse_rational)",
    }
    return math_q, code_q, manifest


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--sources", required=True, help="folder with the downloaded source files")
    ap.add_argument("--bank-dir", default="bank", help="folder with pilot/, dev/ and dev_code/")
    ap.add_argument("--out", default="bank/full")
    ap.add_argument("--seed", type=int, default=544)
    args = ap.parse_args(argv)
    out = Path(args.out)

    math_q, code_q, manifest = select(Path(args.sources), Path(args.bank_dir), args.seed)
    write_jsonl(out / "questions_math.jsonl", math_q)
    write_jsonl(out / "questions_code.jsonl", code_q)
    (out / "question_ids.txt").write_text(
        "\n".join(q["problem_id"] for q in math_q + code_q) + "\n", encoding="utf-8", newline="\n")
    manifest["outputs"] = {name: sha256_bytes((out / name).read_bytes())
                           for name in ("questions_math.jsonl", "questions_code.jsonl", "question_ids.txt")}
    manifest["created_utc"] = datetime.now(timezone.utc).isoformat(timespec="seconds")
    (out / "questions_manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n",
                                                 encoding="utf-8", newline="\n")
    print(json.dumps(manifest["selected"]), f"-> {len(math_q)} math, {len(code_q)} code questions in {out}")
    print(f"near-duplicates left out: {len(manifest['dropped_near_duplicates'])} unused problems, "
          f"{len(manifest['new_math']['dropped_near_duplicates'])} new MATH problems")


if __name__ == "__main__":
    main()
