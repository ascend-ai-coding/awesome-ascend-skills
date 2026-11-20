"""Exercise the report CLI without torch, ATK or device access."""

import csv
from html.parser import HTMLParser
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


class PrecisionCells(HTMLParser):
    """Read only marked table cells, without relying on HTML string spacing."""

    def __init__(self, source):
        super().__init__()
        self.cells = []
        self.active = None
        self.feed(source)

    def handle_starttag(self, tag, attrs):
        attrs = dict(attrs)
        if tag == "td" and "data-field" in attrs:
            self.active = {**attrs, "text": ""}

    def handle_data(self, data):
        if self.active is not None:
            self.active["text"] += data

    def handle_endtag(self, tag):
        if tag == "td" and self.active is not None:
            self.cells.append(self.active)
            self.active = None


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

    def comparison(self, **changes):
        row = {
            "case_id": "add-001", "operator": "torch.add", "torch_api": "torch.add",
            "phase": "forward", "api_type": "floating", "api_family": "elementwise",
            "api_layer": "torch", "output_name": "output[0]", "shape": [2, 3],
            "dtype": "float32", "candidate_name": "candidate", "peer_name": "peer",
            "golden_name": "golden", "standard21": {
                "level": "L0", "numerical_status": "PASS", "formal_status": "BLOCKED",
                "mare_ratio": 1.25, "mere_ratio": 1.1, "rmse_ratio": 1.05,
                "issue_fields": {"formal_status": "缺少正式覆盖证据"},
            },
            "diagnostics": {
                "candidate_golden": {"bitwise_equal": False, "max_abs": 0.001},
                "peer_golden": {},
                "candidate_peer": {"max_ulp": 1000000000, "max_rel": 10000,
                                   "row_max_mean_rel": 250},
            },
            "context": {"source": "synthetic fixture", "missing": "hardware/ATK"},
        }
        row.update(changes)
        return row

    def read_precision_csv(self):
        with (self.output / "precision_comparison.csv").open(newline="", encoding="utf-8") as file:
            reader = csv.DictReader(file)
            rows = list(reader)
            return reader.fieldnames, rows

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
        self.assertNotIn("precision_comparison_count", summary)
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

    def test_empty_precision_extension_keeps_existing_outputs_identical(self):
        completed = self.invoke()
        self.assertEqual(completed.returncode, 0, completed.stderr)
        original = {path.name: path.read_bytes() for path in self.output.iterdir()}
        self.output = self.output.with_name("empty-extension")
        self.bundle["precision_comparisons"] = []
        completed = self.invoke()
        self.assertEqual(completed.returncode, 0, completed.stderr)
        self.assertEqual(original, {path.name: path.read_bytes() for path in self.output.iterdir()})

    def test_file_import_from_arbitrary_cwd_supports_validate_and_render(self):
        self.bundle["precision_comparisons"] = [self.comparison()]
        self.config.write_text(json.dumps(self.bundle), encoding="utf-8")
        program = """
import importlib.util
import json
from pathlib import Path
import sys

spec = importlib.util.spec_from_file_location("external_report_consumer", sys.argv[1])
report = importlib.util.module_from_spec(spec)
spec.loader.exec_module(report)
bundle_path = Path(sys.argv[2]).resolve()
bundle = json.loads(bundle_path.read_text(encoding="utf-8"))
run, rows = report.validate(bundle, bundle_path.parent)
assert run["id"] == bundle["run"]["id"]
assert rows == bundle["results"]
report.render(bundle_path, Path(sys.argv[3]))
"""
        completed = subprocess.run(
            [sys.executable, "-I", "-c", program, str(SCRIPT), str(self.config), str(self.output)],
            cwd=self.root, capture_output=True, text=True, check=False,
        )
        self.assertEqual(completed.returncode, 0, completed.stderr)
        self.assertTrue((self.output / "report.html").is_file())
        self.assertEqual(len(self.read_precision_csv()[1]), 1)

    def test_precision_forward_backward_leaves_and_backend_roles_stay_separate(self):
        rows = [
            self.comparison(), self.comparison(output_name="output[1]", shape=[]),
            self.comparison(phase="backward", output_name="grad_input"),
            self.comparison(phase="backward", output_name="grad_weight", shape=[0, 3]),
            self.comparison(candidate_name="other-candidate"),
        ]
        self.bundle["precision_comparisons"] = rows
        completed = self.invoke()
        self.assertEqual(completed.returncode, 0, completed.stderr)
        self.assertEqual(len(list(self.output.iterdir())), 5)
        summary = json.loads((self.output / "summary.json").read_text(encoding="utf-8"))
        self.assertEqual(summary["case_count"], 1)
        self.assertEqual(summary["precision_comparison_count"], len(rows))
        fields, actual = self.read_precision_csv()
        self.assertEqual(len(actual), len(rows))
        self.assertEqual([row["output_name"] for row in actual], [row["output_name"] for row in rows])
        self.assertEqual([row["shape_json"] for row in actual], ["[2, 3]", "[]", "[2, 3]", "[0, 3]", "[2, 3]"])
        self.assertEqual(fields[:8], ["schema_version", "run_id", "standard_id", "evidence_class",
                                    "case_id", "operator", "torch_api", "backend_operator"])
        self.assertEqual(fields[-3:], ["candidate_peer_row_max_mean_rel", "diagnostic_definition_json", "context_json"])
        rendered = (self.output / "report.html").read_text(encoding="utf-8")
        self.assertIn('href="precision_comparison.csv"', rendered)
        self.assertIn('href="#precision-comparison-title"', rendered)
        self.assertIn("前向", rendered)
        self.assertIn("反向", rendered)
        self.assertIn("输出／梯度槽", rendered)
        self.assertIn("位模式不一致元素数", rendered)
        self.assertIn('<strong>5</strong>输出叶子', rendered)
        self.assertIn("浮点计算类", rendered)

    def test_duplicate_precision_leaf_is_rejected_before_writing(self):
        row = self.comparison()
        self.bundle["precision_comparisons"] = [row, {**row, "operator": "other-name"}]
        completed = self.invoke()
        self.assertNotEqual(completed.returncode, 0)
        self.assertIn("duplicates precision comparison key", completed.stderr)
        self.assertFalse(self.output.exists())

    def test_precision_missing_measurements_remain_blank_and_methods_are_preserved(self):
        row = self.comparison()
        row["standard21"]["small_value_ratio"] = None
        row["diagnostics"]["candidate_peer"]["excluded_count"] = None
        row["diagnostics"]["peer_golden"]["excluded_count"] = 0
        row["diagnostics"]["candidate_peer"]["bitwise_equal"] = None
        row.pop("torch_api")
        row["diagnostic_definition"] = {
            "ulp_dtype": "float32", "relative_denominator": "abs(peer)+1e-7",
            "relative_mask": "finite pairs", "row_max_positions": "peer row maximum; first tie",
            "excluded": "nonfinite positions", "source": "synthetic formula fixture",
        }
        self.bundle["precision_comparisons"] = [row]
        completed = self.invoke()
        self.assertEqual(completed.returncode, 0, completed.stderr)
        _, actual = self.read_precision_csv()
        for field in ("torch_api", "backend_operator", "standard21_small_value_ratio",
                      "peer_golden_max_ulp", "candidate_peer_excluded_count", "candidate_peer_bitwise_equal"):
            self.assertEqual(actual[0][field], "", field)
        self.assertEqual(actual[0]["peer_golden_excluded_count"], "0")
        self.assertEqual(actual[0]["candidate_golden_bitwise_equal"], "False")
        self.assertEqual(json.loads(actual[0]["diagnostic_definition_json"]), row["diagnostic_definition"])
        self.assertEqual(json.loads(actual[0]["context_json"]), row["context"])
        rendered = (self.output / "report.html").read_text(encoding="utf-8")
        self.assertIn("未提供", rendered)
        self.assertIn("缺测字段保留空值", rendered)
        self.assertIn("peer row maximum; first tie", rendered)

    def test_only_standard_issues_are_red_without_rejudging_diagnostics(self):
        row = self.comparison()
        row["standard21"]["issue_fields"]["ratio_denominator_err"] = "待核对分母"  # null may still be an explicit issue
        row["diagnostics"]["candidate_golden"]["max_ulp"] = 117
        row["diagnostics"]["peer_golden"]["max_ulp"] = 123456789.125
        self.bundle["precision_comparisons"] = [row]
        completed = self.invoke()
        self.assertEqual(completed.returncode, 0, completed.stderr)
        rendered = (self.output / "report.html").read_text(encoding="utf-8")
        cells = PrecisionCells(rendered).cells
        red = {cell["data-field"] for cell in cells if "standard21-issue" in cell.get("class", "")}
        self.assertEqual(red, {"standard21.formal_status", "standard21.ratio_denominator_err"})
        by_field = {cell["data-field"]: cell for cell in cells}
        self.assertEqual(by_field["standard21.numerical_status"]["text"], "PASS")
        self.assertEqual(by_field["candidate_peer.max_ulp"]["text"], "1000000000")
        self.assertEqual(by_field["candidate_golden.max_ulp"]["text"], "117")
        self.assertEqual(by_field["peer_golden.max_ulp"]["text"], "123456789.125")
        _, csv_rows = self.read_precision_csv()
        self.assertEqual(csv_rows[0]["candidate_golden_max_ulp"], "117")
        self.assertEqual(csv_rows[0]["peer_golden_max_ulp"], "123456789.125")
        self.assertEqual(by_field["candidate_peer.max_rel"]["text"], "1.000000e+4")
        self.assertNotIn("class", by_field["candidate_golden.bitwise_equal"])
        self.assertIn("不重新判定 2.1", rendered)
        self.assertIn("未提供 ULP 计算 dtype", rendered)

    def test_all_standard_problem_statuses_are_red_without_issue_fields(self):
        row = self.comparison()
        row["standard21"] = {
            "numerical_status": "FAIL", "formal_status": "BLOCKED", "nonfinite_status": "ERROR",
            "recheck_status": "TIMEOUT",
        }
        self.bundle["precision_comparisons"] = [row]
        completed = self.invoke()
        self.assertEqual(completed.returncode, 0, completed.stderr)
        cells = PrecisionCells((self.output / "report.html").read_text(encoding="utf-8")).cells
        red = {cell["data-field"] for cell in cells if "standard21-issue" in cell.get("class", "")}
        self.assertEqual(red, {"standard21." + field for field in row["standard21"]})

    def test_precision_html_and_csv_injection_are_escaped(self):
        attack = '<img src=x onerror="alert(1)"><script>alert(2)</script>'
        row = self.comparison(case_id="=1+1", operator=attack, torch_api="@cmd", backend_operator="javascript:alert(1)")
        row["standard21"]["detail"] = "\n=HYPERLINK(\"https://example.invalid\")"
        row["standard21"]["issue_fields"] = {"detail": attack}
        row["diagnostic_definition"] = attack
        row["context"] = {"source": attack}
        self.bundle["precision_comparisons"] = [row]
        completed = self.invoke()
        self.assertEqual(completed.returncode, 0, completed.stderr)
        _, actual = self.read_precision_csv()
        self.assertEqual(actual[0]["case_id"], "'=1+1")
        self.assertEqual(actual[0]["torch_api"], "'@cmd")
        self.assertEqual(actual[0]["standard21_detail"], "'" + row["standard21"]["detail"])
        rendered = (self.output / "report.html").read_text(encoding="utf-8")
        self.assertNotIn(attack, rendered)
        self.assertNotIn('<img src=x', rendered)
        self.assertNotIn('href="javascript:', rendered)
        self.assertIn("&lt;script&gt;alert(2)&lt;/script&gt;", rendered)

    def test_precision_invalid_values_are_rejected_before_writing(self):
        changes = [
            ("shape", None), ("shape", [-1]), ("shape", [True]), ("shape", [1.5]),
            ("api_type", "boolean"), ("api_type", "floating_point"),
            ("standard21.mare_ratio", True), ("standard21.mare_ratio", float("nan")),
            ("standard21.mare_ratio", float("inf")), ("standard21.mare_ratio", -1),
            ("standard21.peer_error_count", 1.5), ("standard21.peer_error_count", True),
            ("standard21.level", "P0"), ("standard21.numerical_status", "UNKNOWN"),
            ("diagnostics.candidate_peer.max_ulp", True),
            ("diagnostics.candidate_peer.max_rel", float("inf")),
            ("diagnostics.candidate_peer.bitwise_equal", 1),
            ("diagnostics.candidate_peer.excluded_count", -1),
            ("context", {"nested": {"value": float("nan")}}),
            ("diagnostic_definition", {"epsilon": float("inf")}),
            ("diagnostic_definition", ["unsupported"]),
        ]
        for path, value in changes:
            with self.subTest(field=path, value=value):
                row = self.comparison()
                node = row
                parts = path.split(".")
                for part in parts[:-1]:
                    node = node[part]
                node[parts[-1]] = value
                self.bundle["precision_comparisons"] = [row]
                completed = self.invoke()
                self.assertNotEqual(completed.returncode, 0, completed.stderr)
                self.assertIn("precision_comparisons", completed.stderr)
                self.assertFalse(self.output.exists())

    def test_precision_issue_fields_cannot_target_diagnostics(self):
        for field in ("max_ulp", "candidate_peer.max_ulp", "diagnostics", "issue_fields", "unknown"):
            with self.subTest(field=field):
                row = self.comparison()
                row["standard21"]["issue_fields"] = {field: "must not mark diagnostic cells"}
                self.bundle["precision_comparisons"] = [row]
                completed = self.invoke()
                self.assertNotEqual(completed.returncode, 0)
                self.assertIn("only standard21 fields", completed.stderr)
                self.assertFalse(self.output.exists())

    def test_precision_api_categories_and_optional_torch_mapping(self):
        rows = []
        for category in ("non_computational", "integer", "quantized", "floating", "random"):
            row = self.comparison(case_id=category, operator="aclnnExample", api_type=category)
            row.pop("torch_api")
            row["standard21"] = {}
            row["diagnostics"] = {}
            rows.append(row)
        self.bundle["precision_comparisons"] = rows
        completed = self.invoke()
        self.assertEqual(completed.returncode, 0, completed.stderr)
        _, actual = self.read_precision_csv()
        self.assertEqual([row["api_type"] for row in actual], [row["api_type"] for row in rows])
        self.assertTrue(all(row["torch_api"] == "" for row in actual))
        self.assertTrue(all(row["standard21_numerical_status"] == "" for row in actual))


if __name__ == "__main__":
    unittest.main()
