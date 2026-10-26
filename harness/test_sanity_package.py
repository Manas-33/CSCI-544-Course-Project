"""Offline test: the slide package reads its numbers and settings from the committed runs."""
import tempfile
import unittest
from pathlib import Path

from .sanity_package import main

RUN = "results/bf16/sdpa/ar-sanity-bf16-sdpa-20261026-000459"
ANALYSIS = "results/bf16/sdpa/ar-sanity-analysis-bf16-sdpa-20261026-000459"
OLD_RUN = "results/bf16/ar-sanity-bf16-20261020-015535"
OLD_ANALYSIS = "results/bf16/ar-sanity-analysis-bf16-20261020-015535"


class SanityPackageTests(unittest.TestCase):
    def test_package_matches_the_committed_one(self):
        with tempfile.TemporaryDirectory() as out:
            main(["--run-dir", RUN, "--analysis-dir", ANALYSIS, "--previous-run", OLD_RUN,
                  "--previous-analysis", OLD_ANALYSIS, "--output-dir", out])
            for name in ("PPT_RESULTS_SUMMARY.md", "ar_sanity_results.csv", "ar_sanity_results.json"):
                self.assertEqual((Path(out) / name).read_bytes(), (Path("results/ppt_ar_sanity") / name).read_bytes(), name)
            summary = (Path(out) / "PPT_RESULTS_SUMMARY.md").read_text(encoding="utf-8")
        self.assertIn("26 math, 24 code", summary)
        self.assertIn("0.773 (eager) → 0.920 (SDPA)", summary)

    def test_previous_run_and_analysis_go_together(self):
        with self.assertRaises(SystemExit):
            main(["--run-dir", RUN, "--analysis-dir", ANALYSIS, "--previous-run", OLD_RUN, "--output-dir", "unused"])


if __name__ == "__main__":
    unittest.main()
