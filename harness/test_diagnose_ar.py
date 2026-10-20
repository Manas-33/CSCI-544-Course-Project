"""Offline checks for diagnostic integrity and independent score reduction."""

import json
from pathlib import Path
import shutil
import tempfile
import unittest

from .diagnose_ar import DEFAULT_RUN, domain_metrics, load_inputs, reference_events


class DiagnosticTests(unittest.TestCase):
    def test_existing_run_is_complete_and_pinned(self):
        records, labels, _, previous = load_inputs("bank/sanity/sanity_set.jsonl", Path(DEFAULT_RUN))
        self.assertEqual(len(records), 50)
        self.assertEqual(sum(labels.values()), 25)
        self.assertEqual(set(previous["A"]), set(labels))
        self.assertTrue(all("label" not in r and "note" not in r for r in records))

    def test_corrupted_saved_scores_are_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            target = Path(tmp) / "run"
            shutil.copytree(DEFAULT_RUN, target)
            with (target / "scores_A.jsonl").open("a") as handle:
                handle.write("{}\n")
            with self.assertRaisesRegex(ValueError, "checksum mismatch"):
                load_inputs("bank/sanity/sanity_set.jsonl", target)

    def test_unpinned_revision_is_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            target = Path(tmp) / "run"
            shutil.copytree(DEFAULT_RUN, target)
            path = target / "manifest.json"
            manifest = json.loads(path.read_text())
            manifest["models"]["A"]["resolved_model_revision"] = "main"
            path.write_text(json.dumps(manifest))
            with self.assertRaisesRegex(ValueError, "commit revisions"):
                load_inputs("bank/sanity/sanity_set.jsonl", target)

    def test_independent_reference_matches_last_position_scoring(self):
        try:
            import torch
        except ImportError:
            self.skipTest("optional CPU PyTorch is unavailable")
        from types import SimpleNamespace
        from .ar import forward_record

        # First-position logits deliberately favor  the opposite answer.
        logits = torch.tensor([[[20., -20., -20., -20.], [-80., 12., 11., 10.]]])
        class Model:
            def __call__(self, **kwargs):
                return SimpleNamespace(logits=logits)
        sets = {"yes": {"ids": [1, 2]}, "no": {"ids": [0]}}
        harness = forward_record(Model(), [1, 2], sets, torch.device("cpu"), False, {})
        reference = reference_events(logits[0, -1], sets)
        for key in ("score", "p_yes", "p_no"):
            self.assertAlmostEqual(harness[key], reference[key], places=5)
        self.assertGreater(reference["score"], 90)

    def test_domain_breakdown_preserves_score_direction(self):
        rows = [
            {"candidate_id": f"{d}{y}", "domain": d, "label": y,
             "harness": {"score": (2 * y - 1) * (1 if d == "code" else -1)}}
            for d in ("math", "code") for y in (0, 1)
        ]
        metrics = domain_metrics(rows)
        self.assertEqual(metrics["code"]["auroc"], 1)
        self.assertEqual(metrics["math"]["auroc"], 0)
        self.assertEqual(metrics["math"]["false_accepts_at_zero"], ["math0"])


if __name__ == "__main__":
    unittest.main()
