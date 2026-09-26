# ----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ----------------------------------------------------------------------------------------------------------
import csv
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


TEST_ROOT = Path(__file__).resolve().parent
REPOSITORY_ROOT = TEST_ROOT.parents[2]
LOOKUP_SCRIPT = (
    REPOSITORY_ROOT / "tools" / "tool-npu-compute" / "scripts" / "lookup_metric.py"
)
EXPECTED_PATH = TEST_ROOT / "expected" / "metric-lookups.json"
CSV_EXPECTED_PATH = TEST_ROOT / "expected" / "csv-inspection.json"
FIXTURE_ROOT = TEST_ROOT / "fixtures" / "real-capture"
INSPECT_CSV_SCRIPT = (
    REPOSITORY_ROOT / "tools" / "tool-npu-compute" / "scripts" / "inspect_csv.py"
)
HARDWARE_EXPECTED_PATH = TEST_ROOT / "expected" / "hardware-info-inspection.json"
HARDWARE_FIXTURE_PATH = FIXTURE_ROOT / "HardwareInfo.jsonl"
HARDWARE_CATALOG_PATH = (
    REPOSITORY_ROOT
    / "tools"
    / "tool-npu-compute"
    / "assets"
    / "hardware-info-catalog.json"
)
INSPECT_HARDWARE_SCRIPT = (
    REPOSITORY_ROOT
    / "tools"
    / "tool-npu-compute"
    / "scripts"
    / "inspect_hardware_info.py"
)
PIPE_TRACE_EXPECTED_PATH = TEST_ROOT / "expected" / "pipe-trace-inspection.json"
SECTION_EXPECTED_PATH = TEST_ROOT / "expected" / "section-lookups.json"
SUMMARY_EXPECTED_PATH = TEST_ROOT / "expected" / "summary-inspection.json"
COLLECTION_EXPECTED_PATH = TEST_ROOT / "expected" / "collection-inspection.json"
INSPECT_PIPE_TRACE_SCRIPT = (
    REPOSITORY_ROOT / "tools" / "tool-npu-compute" / "scripts" / "inspect_pipe_trace.py"
)
LOOKUP_SECTION_SCRIPT = (
    REPOSITORY_ROOT / "tools" / "tool-npu-compute" / "scripts" / "lookup_section.py"
)
INSPECT_SUMMARY_SCRIPT = (
    REPOSITORY_ROOT / "tools" / "tool-npu-compute" / "scripts" / "inspect_summary.py"
)
INSPECT_COLLECTION_SCRIPT = (
    REPOSITORY_ROOT / "tools" / "tool-npu-compute" / "scripts" / "inspect_collection.py"
)
VALIDATE_CATALOG_SCRIPT = (
    REPOSITORY_ROOT / "tools" / "tool-npu-compute" / "scripts" / "validate_catalog.py"
)


class LookupMetricTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.expected = json.loads(EXPECTED_PATH.read_text(encoding="utf-8"))

    def run_lookup(self, *arguments):
        return subprocess.run(
            [sys.executable, str(LOOKUP_SCRIPT), *arguments],
            cwd=TEST_ROOT,
            check=False,
            capture_output=True,
            text=True,
        )

    def test_help_describes_supported_arguments(self):
        result = self.run_lookup("--help")
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("--section", result.stdout)
        self.assertIn("--field", result.stdout)
        self.assertIn("--format", result.stdout)

    def test_exact_field_query_returns_expected_json(self):
        query = self.expected["field_query"]
        result = self.run_lookup(*query["arguments"])
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(json.loads(result.stdout), query["result"])
        self.assertEqual(result.stderr, "")
        self.assertNotRegex(result.stdout, r"0x[0-9A-Fa-f]+")

    def test_section_query_returns_all_fields_in_catalog_order(self):
        query = self.expected["section_query"]
        result = self.run_lookup(*query["arguments"])
        self.assertEqual(result.returncode, 0, result.stderr)
        output = json.loads(result.stdout)
        self.assertEqual(output["section"], query["section"])
        self.assertEqual(output["output_file"], query["output_file"])
        self.assertNotIn("events", output)
        self.assertEqual(len(output["fields"]), query["field_count"])
        self.assertEqual(output["fields"][0]["name"], query["first_field"])
        self.assertEqual(output["fields"][-1]["name"], query["last_field"])
        self.assertNotRegex(result.stdout, r"0x[0-9A-Fa-f]+")

    def test_section_markdown_does_not_expose_event_ids(self):
        result = self.run_lookup("--section", "MemoryUB", "--format", "markdown")
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertNotIn("Section Event", result.stdout)
        self.assertNotRegex(result.stdout, r"0x[0-9A-Fa-f]+")

    def test_markdown_field_query_contains_each_kind_of_metadata(self):
        query = self.expected["markdown_query"]
        result = self.run_lookup(*query["arguments"])
        self.assertEqual(result.returncode, 0, result.stderr)
        for fragment in query["required_fragments"]:
            self.assertIn(fragment, result.stdout)
        self.assertNotRegex(result.stdout, r"0x[0-9A-Fa-f]+")

    def test_unknown_section_lists_exact_section_names(self):
        result = self.run_lookup("--section", "memory", "--format", "json")
        self.assertEqual(result.returncode, 2)
        self.assertEqual(result.stdout, "")
        self.assertIn("未知 Section：memory", result.stderr)
        for section in (
            "ArithmeticUtilization",
            "PipeUtilization",
            "ResourceConflictRatio",
            "Memory",
            "MemoryL0",
            "MemoryUB",
            "L2Cache",
        ):
            self.assertIn(section, result.stderr)

    def test_new_section_field_queries_return_expected_json(self):
        for query in self.expected["new_field_queries"]:
            with self.subTest(section=query["result"]["section"]):
                result = self.run_lookup(*query["arguments"])
                self.assertEqual(result.returncode, 0, result.stderr)
                self.assertEqual(json.loads(result.stdout), query["result"])
                self.assertEqual(result.stderr, "")

    def test_unknown_field_lists_legal_fields(self):
        result = self.run_lookup("--section", "MemoryUB", "--field", "aiv_unknown")
        self.assertEqual(result.returncode, 2)
        self.assertEqual(result.stdout, "")
        self.assertIn("MemoryUB 中不存在字段：aiv_unknown", result.stderr)
        self.assertIn("block_id", result.stderr)
        self.assertIn("aiv_ub_write_bw_gm(GB/s)", result.stderr)


class InspectCsvTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.expected = json.loads(CSV_EXPECTED_PATH.read_text(encoding="utf-8"))

    def run_inspect(self, path, output_format="json"):
        return subprocess.run(
            [
                sys.executable,
                str(INSPECT_CSV_SCRIPT),
                str(path),
                "--format",
                output_format,
            ],
            cwd=TEST_ROOT,
            check=False,
            capture_output=True,
            text=True,
        )

    def write_csv(self, path, rows):
        with path.open("w", newline="", encoding="utf-8") as output_file:
            csv.writer(output_file).writerows(rows)

    def fixture_rows(self, name):
        with (FIXTURE_ROOT / name).open(newline="", encoding="utf-8") as input_file:
            return list(csv.reader(input_file))

    def test_help_describes_csv_path_and_output_format(self):
        result = subprocess.run(
            [sys.executable, str(INSPECT_CSV_SCRIPT), "--help"],
            cwd=TEST_ROOT,
            check=False,
            capture_output=True,
            text=True,
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("csv_path", result.stdout)
        self.assertIn("--format", result.stdout)

    def test_all_real_fixtures_match_expected_availability(self):
        for section, expected in self.expected["fixtures"].items():
            self.assert_fixture_availability(section, expected)

    def assert_fixture_availability(self, section, expected):
        with self.subTest(section=section):
            result = self.run_inspect(FIXTURE_ROOT / expected["file"])
            self.assertEqual(result.returncode, 0, result.stderr)
            output = json.loads(result.stdout)
            self.assertEqual(output["section"], section)
            self.assertGreater(output["row_count"], 0)
            self.assertEqual(len(output["columns"]), expected["field_count"])
            columns = {column["name"]: column for column in output["columns"]}
            for name, availability in expected["selected_fields"].items():
                self.assert_field_availability(
                    columns[name], availability, output["row_count"]
                )
                self.assertTrue(columns[name]["meaning"])
                self.assertTrue(columns[name]["unit"])

    def assert_field_availability(self, column, availability, row_count):
        if availability == "all_valid":
            self.assertEqual(column["valid_count"], row_count)
            self.assertEqual(column["na_count"], 0)
        elif availability == "all_na":
            self.assertEqual(column["valid_count"], 0)
            self.assertEqual(column["na_count"], row_count)
        else:
            self.fail(f"未知可用性断言：{availability}")

    def test_header_identifies_renamed_csv_and_row_count_is_variable(self):
        rows = self.fixture_rows("MemoryUB.csv")
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "renamed.csv"
            self.write_csv(path, [rows[0], rows[1], rows[1]])
            result = self.run_inspect(path)
        self.assertEqual(result.returncode, 0, result.stderr)
        output = json.loads(result.stdout)
        self.assertEqual(output["section"], "MemoryUB")
        self.assertEqual(output["row_count"], 2)
        columns = {column["name"]: column for column in output["columns"]}
        self.assertEqual(columns["aiv_time(us)"]["valid_count"], 2)
        self.assertEqual(columns["aic_time(us)"]["na_count"], 2)

    def test_markdown_output_reports_section_rows_and_columns(self):
        result = self.run_inspect(FIXTURE_ROOT / "MemoryUB.csv", "markdown")
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("# MemoryUB CSV 检查", result.stdout)
        self.assertIn("**数据行数：**", result.stdout)
        self.assertIn("`aiv_ub_read_bw_gm(GB/s)`", result.stdout)

    def test_unknown_and_missing_columns_are_reported(self):
        rows = self.fixture_rows("MemoryUB.csv")
        rows[0][-1] = "unknown_column"
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "MemoryUB.csv"
            self.write_csv(path, rows)
            result = self.run_inspect(path)
        self.assertEqual(result.returncode, 2)
        self.assertIn("未知字段：unknown_column", result.stderr)
        self.assertIn("缺少字段：aiv_ub_write_bw_gm(GB/s)", result.stderr)

    def test_duplicate_columns_are_reported(self):
        rows = self.fixture_rows("MemoryUB.csv")
        rows[0][-1] = rows[0][0]
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "MemoryUB.csv"
            self.write_csv(path, rows)
            result = self.run_inspect(path)
        self.assertEqual(result.returncode, 2)
        self.assertIn("重复字段：block_id", result.stderr)

    def test_column_order_error_is_reported(self):
        rows = self.fixture_rows("MemoryUB.csv")
        rows[0][0], rows[0][1] = rows[0][1], rows[0][0]
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "MemoryUB.csv"
            self.write_csv(path, rows)
            result = self.run_inspect(path)
        self.assertEqual(result.returncode, 2)
        self.assertIn("字段顺序与 MemoryUB 定义不一致", result.stderr)

    def test_data_row_with_wrong_column_count_is_reported(self):
        rows = self.fixture_rows("MemoryUB.csv")
        rows[1] = rows[1][:-1]
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "MemoryUB.csv"
            self.write_csv(path, rows)
            result = self.run_inspect(path)
        self.assertEqual(result.returncode, 2)
        self.assertIn("第 2 行列数为 9，预期为 10", result.stderr)

    def test_malformed_csv_is_reported(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "MemoryUB.csv"
            path.write_text('block_id,"unterminated\n', encoding="utf-8")
            result = self.run_inspect(path)
        self.assertEqual(result.returncode, 2)
        self.assertIn("CSV 格式错误", result.stderr)


class InspectHardwareInfoTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.expected = json.loads(HARDWARE_EXPECTED_PATH.read_text(encoding="utf-8"))
        cls.catalog = json.loads(HARDWARE_CATALOG_PATH.read_text(encoding="utf-8"))
        cls.fixture_records = [
            json.loads(line)
            for line in HARDWARE_FIXTURE_PATH.read_text(encoding="utf-8").splitlines()
        ]

    def run_inspect(self, path, output_format="json"):
        return subprocess.run(
            [
                sys.executable,
                str(INSPECT_HARDWARE_SCRIPT),
                str(path),
                "--format",
                output_format,
            ],
            cwd=TEST_ROOT,
            check=False,
            capture_output=True,
            text=True,
        )

    def write_records(self, path, records):
        path.write_text(
            "".join(
                json.dumps(record, ensure_ascii=False) + "\n" for record in records
            ),
            encoding="utf-8",
        )

    def inspect_records(self, records):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "HardwareInfo.jsonl"
            self.write_records(path, records)
            return self.run_inspect(path)

    def copied_records(self):
        return json.loads(json.dumps(self.fixture_records))

    def test_help_describes_jsonl_path_and_output_format(self):
        result = subprocess.run(
            [sys.executable, str(INSPECT_HARDWARE_SCRIPT), "--help"],
            cwd=TEST_ROOT,
            check=False,
            capture_output=True,
            text=True,
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("jsonl_path", result.stdout)
        self.assertIn("--format", result.stdout)

    def test_real_fixture_values_and_metadata_match_catalog(self):
        result = self.run_inspect(HARDWARE_FIXTURE_PATH)
        self.assertEqual(result.returncode, 0, result.stderr)
        output = json.loads(result.stdout)
        self.assertEqual(output["record_count"], self.expected["record_count"])
        self.assertEqual(
            [category["name"] for category in output["categories"]],
            [category["name"] for category in self.expected["categories"]],
        )
        catalog_categories = {
            category["name"]: category for category in self.catalog["categories"]
        }
        for actual, expected in zip(output["categories"], self.expected["categories"]):
            self.assertEqual(len(actual["fields"]), expected["field_count"])
            fields = {field["name"]: field for field in actual["fields"]}
            for name, value in expected["selected_values"].items():
                self.assertEqual(fields[name]["value"], value)
            catalog_fields = catalog_categories[actual["name"]]["fields"]
            for field, catalog_field in zip(actual["fields"], catalog_fields):
                self.assertEqual(field["name"], catalog_field["name"])
                for key in (
                    "json_type",
                    "meaning",
                    "unit",
                    "variability",
                    "analysis",
                    "notes",
                ):
                    self.assertEqual(field[key], catalog_field[key])

    def test_markdown_output_contains_values_and_explanations(self):
        result = self.run_inspect(HARDWARE_FIXTURE_PATH, "markdown")
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("# HardwareInfo 检查", result.stdout)
        self.assertIn("## Device Info", result.stdout)
        self.assertIn("Ascend950PR_9599 V100", result.stdout)
        self.assertIn("Runtime 当前可见的 NPU 设备数量", result.stdout)

    def test_malformed_json_is_reported_with_line_number(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "HardwareInfo.jsonl"
            path.write_text('{"category":"Host Info"}\n{bad}\n', encoding="utf-8")
            result = self.run_inspect(path)
        self.assertEqual(result.returncode, 2)
        self.assertIn("第 2 行 JSON 格式错误", result.stderr)

    def test_non_object_record_is_reported(self):
        records = self.copied_records()
        records[0] = ["Host Info"]
        result = self.inspect_records(records)
        self.assertEqual(result.returncode, 2)
        self.assertIn("第 1 行必须是 JSON 对象", result.stderr)

    def test_missing_category_field_is_reported(self):
        records = self.copied_records()
        records[0].pop("category")
        result = self.inspect_records(records)
        self.assertEqual(result.returncode, 2)
        self.assertIn("第 1 行缺少 category", result.stderr)

    def test_unknown_category_is_reported(self):
        records = self.copied_records()
        records[0]["category"] = "Unknown Info"
        result = self.inspect_records(records)
        self.assertEqual(result.returncode, 2)
        self.assertIn("未知类别：Unknown Info", result.stderr)

    def test_duplicate_category_is_reported(self):
        records = self.copied_records()
        records.append(records[0])
        result = self.inspect_records(records)
        self.assertEqual(result.returncode, 2)
        self.assertIn("重复类别：Host Info", result.stderr)

    def test_missing_category_is_reported(self):
        records = self.copied_records()[:-1]
        result = self.inspect_records(records)
        self.assertEqual(result.returncode, 2)
        self.assertIn("缺少类别：Memory Information", result.stderr)

    def test_missing_and_unknown_fields_are_reported(self):
        records = self.copied_records()
        records[0].pop("cpu logical count")
        records[0]["unknown field"] = 1
        result = self.inspect_records(records)
        self.assertEqual(result.returncode, 2)
        self.assertIn("Host Info 未知字段：unknown field", result.stderr)
        self.assertIn("Host Info 缺少字段：cpu logical count", result.stderr)

    def test_field_order_error_is_reported(self):
        records = self.copied_records()
        items = list(records[0].items())
        items[0], items[1] = items[1], items[0]
        records[0] = dict(items)
        result = self.inspect_records(records)
        self.assertEqual(result.returncode, 2)
        self.assertIn("Host Info 字段顺序不一致", result.stderr)

    def test_field_type_error_is_reported(self):
        records = self.copied_records()
        records[0]["cpu logical count"] = 88.5
        result = self.inspect_records(records)
        self.assertEqual(result.returncode, 2)
        self.assertIn("cpu logical count 类型应为 integer", result.stderr)

    def test_negative_number_is_reported(self):
        records = self.copied_records()
        records[-1]["hbm used(MB)"] = -1
        result = self.inspect_records(records)
        self.assertEqual(result.returncode, 2)
        self.assertIn("hbm used(MB) 必须是有限非负数", result.stderr)

    def test_empty_string_is_reported(self):
        records = self.copied_records()
        records[1]["chip info"] = ""
        result = self.inspect_records(records)
        self.assertEqual(result.returncode, 2)
        self.assertIn("chip info 不能为空", result.stderr)


class InspectPipeTraceTest(unittest.TestCase):
    COLORS = {
        "SCALAR": "startup",
        "VECTOR": "rail_idle",
        "CUBE": "rail_response",
        "MTE1": "thread_state_iowait",
        "MTE2": "yellow",
        "MTE3": "rail_animation",
        "FIXP": "thread_state_unknown",
    }

    @classmethod
    def setUpClass(cls):
        cls.expected = json.loads(PIPE_TRACE_EXPECTED_PATH.read_text(encoding="utf-8"))

    def run_inspect(self, path, output_format="json"):
        return subprocess.run(
            [
                sys.executable,
                str(INSPECT_PIPE_TRACE_SCRIPT),
                str(path),
                "--format",
                output_format,
            ],
            cwd=TEST_ROOT,
            check=False,
            capture_output=True,
            text=True,
        )

    def inspect_document(self, document):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "PipeTrace.json"
            path.write_text(json.dumps(document), encoding="utf-8")
            return self.run_inspect(path)

    def event(self, name="MTE2", pid="group0.veccore0", ts=1.0, dur=0.5):
        return {
            "cname": self.COLORS[name],
            "dur": dur,
            "name": name,
            "ph": "X",
            "pid": pid,
            "tid": name,
            "ts": ts,
        }

    def document(self, events=None):
        return {
            "displayTimeUnit": "ns",
            "profilingType": "op",
            "schemaVersion": 1,
            "traceEvents": [self.event()] if events is None else events,
        }

    def assert_trace_statistics(self, output, expected):
        for name in (
            "event_count",
            "track_count",
            "sampled_group_ids",
            "sampled_group_count",
            "sampled_core_tracks",
            "maximum_sampled_groups",
            "sampling_limited",
            "tracks_represent_all_enabled_cores",
            "trace_start_us",
            "trace_end_us",
            "trace_span_us",
        ):
            self.assertEqual(output[name], expected[name])
        pipelines = {entry["name"]: entry for entry in output["pipelines"]}
        self.assertEqual(
            {name: entry["event_count"] for name, entry in pipelines.items()},
            expected["pipeline_event_counts"],
        )
        self.assertEqual(
            {name: entry["duration_sum_us"] for name, entry in pipelines.items()},
            expected["pipeline_duration_sum_us"],
        )
        return {(entry["pid"], entry["tid"]): entry for entry in output["tracks"]}

    def test_help_describes_json_path_and_output_format(self):
        result = subprocess.run(
            [sys.executable, str(INSPECT_PIPE_TRACE_SCRIPT), "--help"],
            cwd=TEST_ROOT,
            check=False,
            capture_output=True,
            text=True,
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("json_path", result.stdout)
        self.assertIn("--format", result.stdout)

    def test_valid_trace_reports_pipeline_track_and_time_statistics(self):
        events = [
            self.event(pid="group0.veccore0", ts=3.0, dur=1.0),
            self.event(name="VECTOR", pid="group0.veccore0", ts=1.0, dur=0.5),
            self.event(pid="group1.veccore1", ts=2.0, dur=0.25),
        ]
        result = self.inspect_document(self.document(events))
        self.assertEqual(result.returncode, 0, result.stderr)
        output = json.loads(result.stdout)
        expected = self.expected["minimal_trace"]

        self.assertEqual(output["section"], "Pipeline")
        self.assertEqual(output["header"]["display_time_unit"], "ns")
        self.assertEqual(output["header"]["timestamp_unit"], "us")
        tracks = self.assert_trace_statistics(output, expected)
        track = tracks[("group0.veccore0", "MTE2")]
        self.assertIsNone(track["process_ordinal"])
        self.assertIsNone(track["result_sequence"])
        self.assertIsNone(track["device_id"])
        self.assertIsNone(track["replay_id"])
        self.assertEqual(track["group_id"], 0)
        self.assertEqual(track["core"], "veccore0")
        self.assertEqual(track["event_count"], 1)
        self.assertEqual(track["trace_start_us"], 3.0)
        self.assertEqual(track["trace_end_us"], 4.0)
        self.assertEqual(track["duration_sum_us"], 1.0)

    def test_empty_trace_is_valid_and_has_no_time_range(self):
        result = self.inspect_document(self.document([]))
        self.assertEqual(result.returncode, 0, result.stderr)
        output = json.loads(result.stdout)
        self.assertEqual(output["event_count"], 0)
        self.assertEqual(output["track_count"], 0)
        self.assertEqual(output["sampled_group_ids"], [])
        self.assertEqual(output["sampled_group_count"], 0)
        self.assertEqual(output["sampled_core_tracks"], {})
        self.assertEqual(output["maximum_sampled_groups"], 6)
        self.assertTrue(output["sampling_limited"])
        self.assertFalse(output["tracks_represent_all_enabled_cores"])
        self.assertEqual(output["pipelines"], [])
        self.assertEqual(output["tracks"], [])
        self.assertIsNone(output["trace_start_us"])
        self.assertIsNone(output["trace_end_us"])
        self.assertIsNone(output["trace_span_us"])

    def test_merged_result_pid_is_parsed(self):
        event = self.event(
            name="CUBE",
            pid="process1.result2.device3.replay4.group5.cubecore",
        )
        result = self.inspect_document(self.document([event]))
        self.assertEqual(result.returncode, 0, result.stderr)
        track = json.loads(result.stdout)["tracks"][0]
        self.assertEqual(track["process_ordinal"], 1)
        self.assertEqual(track["result_sequence"], 2)
        self.assertEqual(track["device_id"], 3)
        self.assertEqual(track["replay_id"], 4)
        self.assertEqual(track["group_id"], 5)
        self.assertEqual(track["core"], "cubecore")

    def test_real_fixture_matches_expected_summary(self):
        expected = self.expected["real_fixture"]
        path = FIXTURE_ROOT / expected["file"]
        self.assertTrue(path.is_file(), path.name)
        result = self.run_inspect(path)
        self.assertEqual(result.returncode, 0, result.stderr)
        output = json.loads(result.stdout)

        self.assertEqual(
            output["header"]["display_time_unit"], expected["display_time_unit"]
        )
        self.assertEqual(output["header"]["profiling_type"], expected["profiling_type"])
        self.assertEqual(output["header"]["schema_version"], expected["schema_version"])
        self.assertEqual(output["header"]["timestamp_unit"], expected["timestamp_unit"])
        tracks = self.assert_trace_statistics(output, expected)
        for key, selected in expected["selected_tracks"].items():
            pid, tid = key.split("|", maxsplit=1)
            actual = tracks[(pid, tid)]
            for name, value in selected.items():
                self.assertEqual(actual[name], value)

    def test_markdown_output_contains_summary_and_units(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "PipeTrace.json"
            path.write_text(json.dumps(self.document()), encoding="utf-8")
            result = self.run_inspect(path, "markdown")
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("# Pipeline 时间线检查", result.stdout)
        self.assertIn("时间单位：微秒（us）", result.stdout)
        self.assertIn("MTE2", result.stdout)

    def test_malformed_json_is_reported(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "PipeTrace.json"
            path.write_text("{bad}", encoding="utf-8")
            result = self.run_inspect(path)
        self.assertEqual(result.returncode, 2)
        self.assertIn("JSON 格式错误", result.stderr)

    def test_invalid_top_level_contract_is_reported(self):
        cases = [
            ("displayTimeUnit", "us", "displayTimeUnit"),
            ("profilingType", "kernel", "profilingType"),
            ("schemaVersion", 2, "schemaVersion"),
            ("traceEvents", {}, "traceEvents 必须是数组"),
        ]
        for field, value, expected_error in cases:
            with self.subTest(field=field):
                document = self.document()
                document[field] = value
                result = self.inspect_document(document)
                self.assertEqual(result.returncode, 2)
                self.assertIn(expected_error, result.stderr)

    def test_missing_event_field_is_reported(self):
        event = self.event()
        event.pop("dur")
        result = self.inspect_document(self.document([event]))
        self.assertEqual(result.returncode, 2)
        self.assertIn("第 1 个事件缺少字段：dur", result.stderr)

    def test_invalid_pipeline_semantics_are_reported(self):
        cases = []

        unknown = self.event()
        unknown["name"] = "UNKNOWN"
        unknown["tid"] = "UNKNOWN"
        cases.append((unknown, "未知流水线：UNKNOWN"))

        mismatched_name = self.event()
        mismatched_name["tid"] = "VECTOR"
        cases.append((mismatched_name, "name 与 tid 不一致"))

        mismatched_color = self.event()
        mismatched_color["cname"] = "startup"
        cases.append((mismatched_color, "cname 与 MTE2 不一致"))

        invalid_phase = self.event()
        invalid_phase["ph"] = "B"
        cases.append((invalid_phase, "ph 必须为 X"))

        for event, expected_error in cases:
            with self.subTest(expected_error=expected_error):
                result = self.inspect_document(self.document([event]))
                self.assertEqual(result.returncode, 2)
                self.assertIn(expected_error, result.stderr)

    def test_invalid_pid_is_reported(self):
        event = self.event(pid="group0.unknowncore")
        result = self.inspect_document(self.document([event]))
        self.assertEqual(result.returncode, 2)
        self.assertIn("pid 格式不受支持", result.stderr)

    def test_invalid_timing_value_is_reported(self):
        for field, value in (("ts", -1), ("dur", -1), ("ts", True)):
            with self.subTest(field=field, value=value):
                event = self.event()
                event[field] = value
                result = self.inspect_document(self.document([event]))
                self.assertEqual(result.returncode, 2)
                self.assertIn(f"{field} 必须是有限非负数", result.stderr)

    def test_sampling_summary_reports_groups_and_core_tracks(self):
        events = [
            self.event(pid="group0.veccore0"),
            self.event(name="VECTOR", pid="group0.veccore1"),
            self.event(pid="group5.cubecore"),
        ]
        result = self.inspect_document(self.document(events))
        self.assertEqual(result.returncode, 0, result.stderr)
        output = json.loads(result.stdout)
        self.assertEqual(output["sampled_group_ids"], [0, 5])
        self.assertEqual(output["sampled_group_count"], 2)
        self.assertEqual(
            output["sampled_core_tracks"],
            {
                "0": ["veccore0", "veccore1"],
                "5": ["cubecore"],
            },
        )
        self.assertEqual(output["maximum_sampled_groups"], 6)
        self.assertTrue(output["sampling_limited"])
        self.assertFalse(output["tracks_represent_all_enabled_cores"])

    def test_group_id_outside_sampling_range_is_reported(self):
        result = self.inspect_document(
            self.document([self.event(pid="group6.veccore0")])
        )
        self.assertEqual(result.returncode, 2)
        self.assertIn("Group ID 必须在 0 到 5 之间", result.stderr)

    def test_markdown_explains_sampling_does_not_equal_enabled_cores(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "PipeTrace.json"
            path.write_text(json.dumps(self.document()), encoding="utf-8")
            result = self.run_inspect(path, "markdown")
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("采样 Group 上限：** 6", result.stdout)
        self.assertIn("不能用于推断目标程序实际启用的 Core 数量", result.stdout)


class LookupSectionTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.expected = json.loads(SECTION_EXPECTED_PATH.read_text(encoding="utf-8"))
        cls.sections_by_name = {
            section["name"]: section for section in cls.expected["sections"]
        }

    def run_lookup(self, *arguments):
        self.assertTrue(
            LOOKUP_SECTION_SCRIPT.is_file(), "尚未实现 scripts/lookup_section.py"
        )
        return subprocess.run(
            [sys.executable, str(LOOKUP_SECTION_SCRIPT), *arguments],
            cwd=TEST_ROOT,
            check=False,
            capture_output=True,
            text=True,
        )

    def test_each_supported_section_returns_declared_mapping(self):
        for expected in self.expected["sections"]:
            with self.subTest(section=expected["name"]):
                result = self.run_lookup(
                    "--section", expected["name"], "--format", "json"
                )
                self.assertEqual(result.returncode, 0, result.stderr)
                self.assertEqual(json.loads(result.stdout), expected)

    def test_unknown_section_lists_supported_names(self):
        result = self.run_lookup("--section", "memory", "--format", "json")
        self.assertEqual(result.returncode, 2)
        self.assertIn("未知 Section：memory", result.stderr)
        for expected in self.expected["sections"]:
            self.assertIn(expected["name"], result.stderr)

    def test_list_query_returns_sections_and_default_results(self):
        result = self.run_lookup("--format", "json")
        self.assertEqual(result.returncode, 0, result.stderr)
        output = json.loads(result.stdout)
        self.assertEqual(output["sets"], self.expected["sets"])
        self.assertEqual(output["sections"], self.expected["sections"])
        self.assertEqual(output["default_results"], self.expected["default_results"])

    def test_help_describes_repeatable_selection_and_output_format(self):
        result = self.run_lookup("--help")
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("--set", result.stdout)
        self.assertIn("--section", result.stdout)
        self.assertIn("--format", result.stdout)

    def test_basic_set_returns_expanded_sections_in_order(self):
        result = self.run_lookup("--set", "basic", "--format", "json")
        self.assertEqual(result.returncode, 0, result.stderr)
        output = json.loads(result.stdout)
        self.assertEqual(
            output["sections"],
            [
                self.sections_by_name[section_name]
                for section_name in self.expected["sets"]["basic"]
            ],
        )

    def test_full_set_markdown_contains_members_and_analysis_routes(self):
        result = self.run_lookup("--set", "full", "--format", "markdown")
        self.assertEqual(result.returncode, 0, result.stderr)
        section_blocks = []
        for block in result.stdout.split("\n\n## ")[1:]:
            section_name, _, body = block.partition("\n")
            section_blocks.append((section_name, body))
        self.assertEqual(
            [section_name for section_name, _ in section_blocks],
            self.expected["sets"]["full"],
        )
        for section_name, body in section_blocks:
            section = self.sections_by_name[section_name]
            self.assertIn(f"`{section['result_file']}`", body)
            self.assertIn(f"`{section['inspector']}`", body)

    def test_section_then_set_preserves_first_occurrence_order(self):
        result = self.run_lookup(
            "--section", "Memory", "--set", "basic", "--format", "json"
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        output = json.loads(result.stdout)
        expected_names = [
            "Memory",
            "Pipeline",
            "PipeUtilization",
            "MemoryL0",
            "MemoryUB",
            "L2Cache",
            "ArithmeticUtilization",
        ]
        self.assertEqual(
            [section["name"] for section in output["sections"]], expected_names
        )

    def test_set_then_section_appends_new_section(self):
        result = self.run_lookup(
            "--set",
            "basic",
            "--section",
            "ResourceConflictRatio",
            "--format",
            "json",
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        output = json.loads(result.stdout)
        self.assertEqual(
            [section["name"] for section in output["sections"]],
            self.expected["sets"]["full"],
        )

    def test_repeated_sets_and_sections_are_deduplicated(self):
        result = self.run_lookup(
            "--set",
            "basic",
            "--section",
            "Memory",
            "--set",
            "full",
            "--section",
            "Pipeline",
            "--format",
            "json",
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        output = json.loads(result.stdout)
        self.assertEqual(
            [section["name"] for section in output["sections"]],
            self.expected["sets"]["full"],
        )

    def test_unknown_set_lists_supported_names(self):
        result = self.run_lookup("--set", "Basic", "--format", "json")
        self.assertEqual(result.returncode, 2)
        self.assertIn("未知 Set：Basic", result.stderr)
        for set_name in self.expected["sets"]:
            self.assertIn(set_name, result.stderr)

    def test_markdown_query_contains_result_and_routes(self):
        result = self.run_lookup("--section", "Pipeline", "--format", "markdown")
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("## Pipeline", result.stdout)
        self.assertIn("`PipeTrace.json`", result.stdout)
        self.assertIn("`pipeline_trace`", result.stdout)
        self.assertIn("`scripts/inspect_pipe_trace.py`", result.stdout)


class InspectSummaryTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.expected = json.loads(SUMMARY_EXPECTED_PATH.read_text(encoding="utf-8"))

    def run_inspect(self, path, output_format="json"):
        self.assertTrue(
            INSPECT_SUMMARY_SCRIPT.is_file(), "尚未实现 scripts/inspect_summary.py"
        )
        return subprocess.run(
            [
                sys.executable,
                str(INSPECT_SUMMARY_SCRIPT),
                str(path),
                "--format",
                output_format,
            ],
            cwd=TEST_ROOT,
            check=False,
            capture_output=True,
            text=True,
        )

    def copied_records(self):
        path = FIXTURE_ROOT / self.expected["real_fixture"]["file"]
        return [
            json.loads(line)
            for line in path.read_text(encoding="utf-8").splitlines()
            if line
        ]

    def inspect_records(self, records):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "summary.jsonl"
            path.write_text(
                "".join(json.dumps(record) + "\n" for record in records),
                encoding="utf-8",
            )
            return self.run_inspect(path)

    def test_help_describes_jsonl_path_and_output_format(self):
        result = subprocess.run(
            [sys.executable, str(INSPECT_SUMMARY_SCRIPT), "--help"],
            cwd=TEST_ROOT,
            check=False,
            capture_output=True,
            text=True,
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("jsonl_path", result.stdout)
        self.assertIn("--format", result.stdout)

    def test_real_summary_reports_sections_and_final_op_info(self):
        path = FIXTURE_ROOT / self.expected["real_fixture"]["file"]
        result = self.run_inspect(path)
        self.assertEqual(result.returncode, 0, result.stderr)
        output = json.loads(result.stdout)
        self.assertEqual(
            output["record_count"], self.expected["real_fixture"]["record_count"]
        )
        self.assertEqual(
            output["section_categories"],
            self.expected["real_fixture"]["section_categories"],
        )
        self.assertEqual(
            output["op_info_summary"]["selected_values"],
            self.expected["real_fixture"]["op_info_summary"]["selected_values"],
        )

    def test_malformed_jsonl_is_reported_with_line_number(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "summary.jsonl"
            path.write_text('{"category":"Memory"}\n{bad}\n', encoding="utf-8")
            result = self.run_inspect(path)
        self.assertEqual(result.returncode, 2)
        self.assertIn("第 2 行 JSON 格式错误", result.stderr)

    def test_op_info_summary_must_be_last_record(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "summary.jsonl"
            path.write_text(
                '{"category":"OpInfoSummary"}\n{"category":"Memory"}\n',
                encoding="utf-8",
            )
            result = self.run_inspect(path)
        self.assertEqual(result.returncode, 2)
        self.assertIn("OpInfoSummary 必须是最后一条记录", result.stderr)

    def test_markdown_output_explains_section_and_op_info(self):
        path = FIXTURE_ROOT / self.expected["real_fixture"]["file"]
        result = self.run_inspect(path, "markdown")
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("# summary.jsonl 检查", result.stdout)
        self.assertIn("## Memory", result.stdout)
        self.assertIn("## OpInfoSummary", result.stdout)
        self.assertIn("算子类型", result.stdout)

    def test_unknown_category_is_reported(self):
        records = self.copied_records()
        records[0]["category"] = "Unknown"
        result = self.inspect_records(records)
        self.assertEqual(result.returncode, 2)
        self.assertIn("未知类别：Unknown", result.stderr)

    def test_duplicate_section_category_is_reported(self):
        records = self.copied_records()
        records.insert(1, records[0])
        result = self.inspect_records(records)
        self.assertEqual(result.returncode, 2)
        self.assertIn("重复类别：PipeUtilization", result.stderr)

    def test_missing_final_op_info_summary_is_reported(self):
        result = self.inspect_records(self.copied_records()[:-1])
        self.assertEqual(result.returncode, 2)
        self.assertIn("最后一条记录必须是 OpInfoSummary", result.stderr)

    def test_section_field_order_error_is_reported(self):
        records = self.copied_records()
        items = list(records[1].items())
        items[1], items[2] = items[2], items[1]
        records[1] = dict(items)
        result = self.inspect_records(records)
        self.assertEqual(result.returncode, 2)
        self.assertIn("Memory 字段顺序不一致", result.stderr)

    def test_section_value_type_error_is_reported(self):
        records = self.copied_records()
        records[1]["aiv_total_cycles"] = "52798"
        result = self.inspect_records(records)
        self.assertEqual(result.returncode, 2)
        self.assertIn("Memory.aiv_total_cycles 类型应为 number 或 null", result.stderr)

    def test_op_info_null_value_is_accepted(self):
        records = self.copied_records()
        records[-1]["Op Name"] = None
        records[-1]["Current Freq"] = None
        result = self.inspect_records(records)
        self.assertEqual(result.returncode, 0, result.stderr)

    def test_op_info_unknown_and_missing_fields_are_reported(self):
        records = self.copied_records()
        records[-1].pop("Pid")
        records[-1]["unknown"] = 1
        result = self.inspect_records(records)
        self.assertEqual(result.returncode, 2)
        self.assertIn("OpInfoSummary 未知字段：unknown", result.stderr)
        self.assertIn("OpInfoSummary 缺少字段：Pid", result.stderr)


class InspectCollectionTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.expected = json.loads(COLLECTION_EXPECTED_PATH.read_text(encoding="utf-8"))

    def run_inspect(self, path, output_format="json"):
        self.assertTrue(
            INSPECT_COLLECTION_SCRIPT.is_file(),
            "尚未实现 scripts/inspect_collection.py",
        )
        return subprocess.run(
            [
                sys.executable,
                str(INSPECT_COLLECTION_SCRIPT),
                str(path),
                "--format",
                output_format,
            ],
            cwd=TEST_ROOT,
            check=False,
            capture_output=True,
            text=True,
        )

    def test_help_describes_result_path_and_output_format(self):
        result = subprocess.run(
            [sys.executable, str(INSPECT_COLLECTION_SCRIPT), "--help"],
            cwd=TEST_ROOT,
            check=False,
            capture_output=True,
            text=True,
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("result_path", result.stdout)
        self.assertIn("--format", result.stdout)

    def test_real_capture_files_are_classified(self):
        result = self.run_inspect(FIXTURE_ROOT)
        self.assertEqual(result.returncode, 0, result.stderr)
        output = json.loads(result.stdout)
        self.assertEqual(
            output["root_result"]["recognized_files"],
            self.expected["real_fixture"]["recognized_files"],
        )
        self.assertEqual(output["collection_results"], [])
        self.assertEqual(
            output["unknown_files"],
            self.expected["real_fixture"]["unknown_files"],
        )
        self.assertEqual(
            output["result_subdirectory_patterns"],
            self.expected["result_subdirectory_patterns"],
        )

    def test_collection_subdirectory_is_reported_separately(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            nested = root / "collection-p123-0001"
            nested.mkdir()
            (root / "HardwareInfo.jsonl").write_bytes(
                (FIXTURE_ROOT / "HardwareInfo.jsonl").read_bytes()
            )
            (nested / "Memory.csv").write_bytes(
                (FIXTURE_ROOT / "Memory.csv").read_bytes()
            )
            result = self.run_inspect(root)
        self.assertEqual(result.returncode, 0, result.stderr)
        output = json.loads(result.stdout)
        self.assertEqual(
            output["root_result"]["recognized_files"], ["HardwareInfo.jsonl"]
        )
        self.assertEqual(
            output["collection_results"],
            [
                {
                    "relative_path": "collection-p123-0001",
                    "recognized_files": ["Memory.csv"],
                }
            ],
        )

    def test_unknown_file_is_reported(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "unexpected.bin").write_bytes(b"unknown")
            result = self.run_inspect(root)
        self.assertEqual(result.returncode, 0, result.stderr)
        output = json.loads(result.stdout)
        self.assertEqual(output["unknown_files"], ["unexpected.bin"])

    def test_unknown_file_in_result_subdirectory_uses_relative_path(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            nested = root / "collection-p123-0001"
            nested.mkdir()
            (nested / "unexpected.bin").write_bytes(b"unknown")
            result = self.run_inspect(root)
        self.assertEqual(result.returncode, 0, result.stderr)
        output = json.loads(result.stdout)
        self.assertEqual(
            output["collection_results"],
            [
                {
                    "relative_path": "collection-p123-0001",
                    "recognized_files": [],
                }
            ],
        )
        self.assertEqual(
            output["unknown_files"], ["collection-p123-0001/unexpected.bin"]
        )

    def test_unknown_subdirectory_is_reported(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "other").mkdir()
            result = self.run_inspect(root)
        self.assertEqual(result.returncode, 0, result.stderr)
        output = json.loads(result.stdout)
        self.assertEqual(output["unknown_files"], ["other/"])

    def test_markdown_output_lists_recognized_files(self):
        result = self.run_inspect(FIXTURE_ROOT, "markdown")
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("# 解包结果检查", result.stdout)
        self.assertIn("`HardwareInfo.jsonl`", result.stdout)
        self.assertIn("`summary.jsonl`", result.stdout)

    def test_non_directory_input_is_reported(self):
        result = self.run_inspect(FIXTURE_ROOT / "Memory.csv")
        self.assertEqual(result.returncode, 2)
        self.assertIn("解包结果路径必须是目录", result.stderr)


class ValidateCatalogTest(unittest.TestCase):
    def run_validate(self, *arguments):
        return subprocess.run(
            [sys.executable, str(VALIDATE_CATALOG_SCRIPT), *arguments],
            cwd=TEST_ROOT,
            check=False,
            capture_output=True,
            text=True,
        )

    def test_catalog_fixtures_and_structure_validate(self):
        result = self.run_validate()
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(result.stdout, "tool-npu-compute 目录校验通过\n")
        self.assertEqual(result.stderr, "")


if __name__ == "__main__":
    unittest.main()
