"""Offline JSONL adapter. Mapping keys are canonical, values are source fields."""

import hashlib
import json
from pathlib import Path


SANITY_MAPPING = {"id": "candidate_id", "problem": "problem", "candidate_answer": "candidate"}
BANK_MAPPING = {"id": "candidate_id", "problem": "problem", "candidate_answer": "candidate",
                "problem_id": "problem_id", "pair_id": "pair_id"}


def _object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON key {key!r}")
        result[key] = value
    return result


def _invalid_constant(value):
    raise ValueError(f"invalid JSON constant {value}")


def parse_json(text):
    return json.loads(text, object_pairs_hook=_object, parse_constant=_invalid_constant)


def validate_mapping(mapping):
    required = {"id", "problem", "candidate_answer"}
    if not isinstance(mapping, dict) or not required <= mapping.keys():
        raise ValueError("field_mapping requires id, problem, candidate_answer")
    if mapping.keys() - (required | {"problem_id", "pair_id"}):
        raise ValueError("unknown canonical field in field_mapping")
    if any(not isinstance(v, str) or not v.strip() for v in mapping.values()):
        raise ValueError("field_mapping values must be nonempty source field names")
    if len(set(mapping.values())) != len(mapping):
        raise ValueError("field_mapping source fields must be distinct")


def read_jsonl(path):
    """Strict, offline JSONL reader shared with analysis; no blank-line skipping."""
    raw = Path(path).read_bytes()
    rows, errors = [], []
    for line_number, line in enumerate(raw.splitlines(), 1):
        try:
            source = parse_json(line.decode("utf-8"))
            if not isinstance(source, dict):
                raise ValueError("record must be a JSON object")
            rows.append(source)
        except (ValueError, UnicodeError) as exc:
            errors.append(f"line {line_number}: {exc}")
    if errors:
        raise ValueError(f"invalid {path}:\n" + "\n".join(errors))
    if not rows:
        raise ValueError(f"{path}, line 1: input contains no records")
    return rows, hashlib.sha256(raw).hexdigest()


def map_records(sources, mapping):
    """Discard all fields except explicitly mapped text and grouping identifiers."""
    validate_mapping(mapping)
    records, errors, seen = [], [], set()
    for line_number, source in enumerate(sources, 1):
        try:
            record = {}
            for canonical, field in mapping.items():
                if canonical == "problem_id" and "pair_id" not in mapping and field not in source:
                    continue
                if field not in source:
                    raise ValueError(f"missing field {field!r} for {canonical}")
                value = source[field]
                if not isinstance(value, str):
                    raise ValueError(f"{field!r} must be a string")
                if not value.strip():
                    raise ValueError(f"{field!r} must be nonempty")
                record[canonical] = value
            if record["id"] in seen:
                raise ValueError(f"duplicate candidate id {record['id']!r}")
            seen.add(record["id"])
            records.append(record)
        except ValueError as exc:
            errors.append(f"line {line_number}, ID {source.get(mapping['id'])!r}: {exc}")
    if errors:
        raise ValueError("invalid input:\n" + "\n".join(errors))
    if not records:
        raise ValueError("line 1: input contains no records")
    return records


def bank_records(sources):
    """Validate make_pairs.py's dev schema without reading any separate labels."""
    records = map_records(sources, BANK_MAPPING)
    pairs, problems, cells, errors = {}, {}, {}, []
    for number, (source, record) in enumerate(zip(sources, records), 1):
        try:
            if source.get("split") not in ("dev", "pilot"):
                raise ValueError("this adapter accepts only split='dev' or 'pilot'")
            if source.get("domain") not in ("math", "code"):
                raise ValueError("expected math/code domain")
            record["domain"] = source["domain"]  # for per-domain AUROC (PLAN.md §5)
            generator = source.get("generator")
            if not isinstance(generator, str) or not generator.strip():
                raise ValueError("generator must be a nonempty string for pair validation")
            actual_hash = hashlib.sha256(record["candidate_answer"].encode("utf-8")).hexdigest()
            if source.get("text_sha256") != actual_hash:
                raise ValueError("text_sha256 does not match the full candidate solution")
            pid, pair_id = record["problem_id"], record["pair_id"]
            if problems.setdefault(pid, record["problem"]) != record["problem"]:
                raise ValueError("inconsistent question text for problem_id")
            cell = (pid, generator)
            if cells.setdefault(cell, pair_id) != pair_id:
                raise ValueError("multiple pair IDs for one (problem, generator) cell")
            pairs.setdefault(pair_id, []).append((record["id"], cell))
        except ValueError as exc:
            errors.append(f"line {number}, ID {record['id']!r}: {exc}")
    for pair_id, members in pairs.items():
        if len(members) != 2 or len({cell for _, cell in members}) != 1:
            errors.append(f"malformed pair {pair_id!r}: {members!r}; expected two candidates in one cell")
    if errors:
        raise ValueError("invalid development bank:\n" + "\n".join(errors))
    return records


def load_records(path, mapping, input_format="mapped"):
    if input_format not in ("mapped", "dev_bank", "sanity"):
        raise ValueError("input_format must be mapped, dev_bank, or sanity")
    sources, input_hash = read_jsonl(path)
    if input_format == "dev_bank":
        if mapping != BANK_MAPPING:
            raise ValueError("dev_bank requires the exact make_pairs.py field mapping")
        records = bank_records(sources)
    elif input_format == "sanity":
        if mapping != SANITY_MAPPING:
            raise ValueError("sanity requires candidate_id, problem, and full candidate text")
        records = map_records(sources, mapping)
        for number, (source, record) in enumerate(zip(sources, records), 1):
            if source.get("domain") not in ("math", "code"):
                raise ValueError(f"line {number}, ID {record['id']!r}: expected math/code domain")
            record["domain"] = source["domain"]
        # Labels and explanatory notes are deliberately excluded from scoring records.
    else:
        records = map_records(sources, mapping)
    return records, input_hash
