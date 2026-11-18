"""Exercise the report CLI without torch, ATK or device access."""

import csv
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

SCRIPT = (
    Path(__file__).resolve().parents[1]
    / "skills/ops/torch-op-validation/scripts/render_report.py"
)


class ReportTest(unittest.TestCase):
    def setUp(self):
        temp = tempfile.TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        self.root = Path(temp.name)
        self.config = self.root / "bundle.json"
        self.output = self.root / "artifacts" / "report"
        self.case = {
            "case_id": "add-001",
            "operator": "torch.add",
            "phase": "forward",
            "suite": "accuracy",
            "backend": "device-under-test",
            "reference": "reference-backend",
            "status": "FAIL",
            "detail": ' =HYPERLINK("https://example.invalid")\n测试 <script>alert(1)</script>',
            "metadata": {"dtype": "float32", "shape": [2, 3]},
            "metrics": [{"name": "signed_delta", "value": -0.125, "unit": "", "limit": 0.01}],
            "evidence": ["raw/run log #1.txt", "raw/missing.txt"],
            "reproduction": "replay/README.md",
        }
        self.bundle = {
            "schema_version": "1.0",
            "run": {
                "id": "verification-example",
                "standard_id": "example-policy",
                "evidence_class": "synthetic",
                "metadata": {"coverage": {"required": 3, "executed": 1}},
            },
            "results": [self.case],
        }

    def invoke(self, write=True):
        if write:
            self.config.write_text(json.dumps(self.bundle), encoding="utf-8")
        return subprocess.run(
            [sys.executable, str(SCRIPT), str(self.config), "--out", str(self.output)],
            capture_output=True, text=True, check=False,
        )

    def test_generic_outputs_and_safe_links(self):
        (self.root / "raw").mkdir()
        (self.root / "raw" / "run log #1.txt").write_text("log", encoding="utf-8")
        (self.root / "replay").mkdir()
        (self.root / "replay" / "README.md").write_text("reproduction", encoding="utf-8")
        self.bundle["results"].append({
            **self.case,
            "suite": "performance",
            "status": "MEASURED",
            "detail": "",
            "metrics": [],
        })
        completed = self.invoke()
        self.assertEqual(completed.returncode, 0, completed.stderr)
        self.assertEqual(
            {path.name for path in self.output.iterdir()},
            {"report.html", "summary.json", "results.csv", "metrics.csv"},
        )
        summary = json.loads((self.output / "summary.json").read_text(encoding="utf-8"))
        self.assertEqual(summary["result_count"], 2)
        self.assertEqual(summary["case_count"], 1)
        self.assertEqual(summary["suites"]["accuracy"]["FAIL"], 1)
        self.assertEqual(summary["suites"]["performance"]["MEASURED"], 1)
        self.assertEqual(summary["run"]["metadata"]["coverage"]["required"], 3)
        self.assertEqual(summary["missing_artifacts"], ["raw/missing.txt"])

        with (self.output / "results.csv").open(newline="", encoding="utf-8") as file:
            results = list(csv.DictReader(file))
        self.assertEqual(results[0]["detail"], "'" + self.case["detail"])
        self.assertEqual(json.loads(results[0]["metadata_json"]), self.case["metadata"])
        with (self.output / "metrics.csv").open(newline="", encoding="utf-8") as file:
            metrics = list(csv.DictReader(file))
        self.assertEqual([(row["value"], row["limit"]) for row in metrics], [("-0.125", "0.01")])

        rendered = (self.output / "report.html").read_text(encoding="utf-8")
        self.assertIn("合成示例", rendered)
        self.assertIn("&lt;script&gt;alert(1)&lt;/script&gt;", rendered)
        self.assertNotIn("<script>alert(1)</script>", rendered)
        self.assertIn('href="../../raw/run%20log%20%231.txt"', rendered)
        self.assertIn('href="../../replay/README.md"', rendered)
        self.assertIn("raw/missing.txt（缺失）", rendered)
        self.assertNotIn('href="raw/missing.txt"', rendered)

    def test_directories_are_reported_as_missing_files(self):
        (self.root / "raw").mkdir()
        (self.root / "replay").mkdir()
        self.case["evidence"] = ["raw"]
        self.case["reproduction"] = "replay"
        completed = self.invoke()
        self.assertEqual(completed.returncode, 0, completed.stderr)
        summary = json.loads((self.output / "summary.json").read_text(encoding="utf-8"))
        self.assertEqual(summary["missing_artifacts"], ["raw", "replay"])
        html = (self.output / "report.html").read_text(encoding="utf-8")
        self.assertIn("raw（缺失）", html)
        self.assertIn("replay（缺失）", html)
        self.assertNotIn('href="../../replay"', html)

    def test_csv_escape_keeps_distinct_case_ids_reversible(self):
        self.case["case_id"] = "=1+1"
        self.bundle["results"].append({**self.case, "case_id": "'=1+1"})
        completed = self.invoke()
        self.assertEqual(completed.returncode, 0, completed.stderr)
        with (self.output / "results.csv").open(newline="", encoding="utf-8") as file:
            keys = [row["case_id"] for row in csv.DictReader(file)]
        self.assertEqual(keys, ["'=1+1", "''=1+1"])
        self.assertEqual([key[1:] for key in keys], ["=1+1", "'=1+1"])

    def test_duplicate_result_key_is_rejected_before_writing(self):
        self.bundle["results"].append({**self.case, "operator": "torch.mul"})
        completed = self.invoke()
        self.assertNotEqual(completed.returncode, 0)
        self.assertIn("duplicates result key", completed.stderr)
        self.assertFalse(self.output.exists())

    def test_unsafe_paths_are_rejected(self):
        for path in ("../outside", "//example.invalid/report", "javascript:alert(1)", "a\\..\\b", "a\nb"):
            with self.subTest(path=path):
                self.case["reproduction"] = path
                completed = self.invoke()
                self.assertNotEqual(completed.returncode, 0)
                self.assertIn("relative path", completed.stderr)
                self.assertFalse(self.output.exists())

    def test_symlink_escape_is_rejected(self):
        with tempfile.TemporaryDirectory() as outside:
            (self.root / "escape").symlink_to(outside, target_is_directory=True)
            self.case["reproduction"] = "escape/README.md"
            completed = self.invoke()
        self.assertNotEqual(completed.returncode, 0)
        self.assertIn("resolves outside", completed.stderr)
        self.assertFalse(self.output.exists())

    def test_boolean_and_nonfinite_metrics_are_rejected(self):
        for value in (True, float("nan"), float("inf")):
            with self.subTest(value=value):
                self.case["metrics"][0]["value"] = value
                completed = self.invoke()
                self.assertNotEqual(completed.returncode, 0)
                self.assertIn("finite number", completed.stderr)

    def test_existing_report_is_preserved(self):
        self.output.mkdir(parents=True)
        earlier_report = self.output / "report.html"
        earlier_report.write_text("existing report", encoding="utf-8")
        completed = self.invoke()
        self.assertNotEqual(completed.returncode, 0)
        self.assertEqual(earlier_report.read_text(encoding="utf-8"), "existing report")


if __name__ == "__main__":
    unittest.main()
