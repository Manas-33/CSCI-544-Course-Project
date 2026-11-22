"""Offline tests for the bank adapter: which splits it accepts, and that its other checks still hold."""
import copy
import hashlib
import json
import tempfile
import unittest
from pathlib import Path

from .data import BANK_MAPPING, bank_records, load_records


def row(pid, k, split, generator="g", text=None):
    text = text or f"answer {k} to {pid}"
    return {"candidate_id": f"{pid}__{generator}__{k}", "pair_id": f"full:{pid}:{generator}", "problem_id": pid,
            "domain": "math", "problem": f"problem {pid}", "candidate": text, "generator": generator,
            "text_sha256": hashlib.sha256(text.encode("utf-8")).hexdigest(), "split": split}


def bank():
    return [row("q1", 0, "validation"), row("q1", 1, "validation"), row("q2", 0, "test"), row("q2", 1, "test")]


class FullBankSplitTests(unittest.TestCase):
    def test_validation_and_test_rows_load_from_one_file(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "bank.jsonl"
            path.write_text("".join(json.dumps(r) + "\n" for r in bank()), encoding="utf-8")
            records, digest = load_records(path, BANK_MAPPING, "dev_bank")
            self.assertEqual([r["id"] for r in records], [r["candidate_id"] for r in bank()])
            self.assertTrue(all("label" not in r and "split" not in r for r in records))
            self.assertEqual(digest, hashlib.sha256(path.read_bytes()).hexdigest())

    def test_dev_and_pilot_still_load_and_other_splits_do_not(self):
        for split in ("dev", "pilot"):
            self.assertEqual(len(bank_records([row("q1", 0, split), row("q1", 1, split)])), 2)
        for split in ("full", "train", "", None, "Test"):
            with self.assertRaisesRegex(ValueError, "accepts only split"):
                bank_records([row("q1", 0, split), row("q1", 1, split)])

    def test_a_problem_cannot_be_in_two_splits(self):
        rows = [row("q1", 0, "validation"), row("q1", 1, "validation"),
                row("q1", 0, "test", "h"), row("q1", 1, "test", "h")]
        with self.assertRaisesRegex(ValueError, "two splits"):
            bank_records(rows)

    def test_the_other_checks_still_apply_to_the_new_splits(self):
        changes = (("text_sha256", "0" * 64, "text_sha256"), ("domain", "physics", "domain"),
                   ("generator", "", "generator"), ("pair_id", "other", "malformed pair"),
                   ("problem", "another text", "inconsistent question text"), ("candidate", " ", "nonempty"))
        for field, value, message in changes:
            rows = copy.deepcopy(bank())
            rows[3][field] = value
            with self.assertRaisesRegex(ValueError, message):
                bank_records(rows)


if __name__ == "__main__":
    unittest.main()
