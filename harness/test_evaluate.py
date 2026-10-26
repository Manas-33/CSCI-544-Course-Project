"""Offline tests for the evaluator's metrics (no models, no GPU)."""
import unittest

from .data import BANK_MAPPING, load_records
from .evaluate import domain_aurocs, roc_curve


class DomainAurocTests(unittest.TestCase):
    def test_each_domain_is_scored_on_its_own_items(self):
        records = [{"id": str(i), "domain": d} for i, d in enumerate(["math"] * 4 + ["code"] * 4)]
        labels = [1, 1, 0, 0, 1, 0, 1, 0]
        scores = [3, 2, 1, 0, 0, 1, 2, 3]  # math perfectly ranked; code half inverted
        result = domain_aurocs(records, labels, scores)
        self.assertEqual(result, {"code": 0.25, "math": 1.0})
        self.assertAlmostEqual(roc_curve(labels, scores)[2], 0.625)  # pooled AUROC is not the mean of the two

    def test_records_without_domain_or_one_class_are_skipped(self):
        self.assertEqual(domain_aurocs([{"id": "a"}, {"id": "b"}], [1, 0], [1, 0]), {})
        records = [{"id": "a", "domain": "math"}, {"id": "b", "domain": "math"}, {"id": "c", "domain": "code"}]
        self.assertEqual(domain_aurocs(records, [1, 0, 1], [2, 1, 0]), {"math": 1.0})


class BankRecordTests(unittest.TestCase):
    def test_dev_bank_records_keep_their_domain(self):
        records, _ = load_records("bank/dev/dev_bank.jsonl", BANK_MAPPING, "dev_bank")
        self.assertEqual(len(records), 142)
        self.assertEqual({r["domain"] for r in records}, {"math"})
        self.assertTrue(all("label" not in r for r in records))


if __name__ == "__main__":
    unittest.main()
