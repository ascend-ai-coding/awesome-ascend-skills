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
import hashlib
import json
import unittest
from pathlib import Path


TEST_ROOT = Path(__file__).resolve().parent
REPOSITORY_ROOT = TEST_ROOT.parents[2]
FIXTURE_ROOT = TEST_ROOT / "fixtures" / "real-capture"
MANIFEST_PATH = FIXTURE_ROOT / "fixture-manifest.json"
REFERENCE_ROOT = REPOSITORY_ROOT / "tools" / "tool-npu-compute" / "references"
HARDWARE_INFO_CATALOG_PATH = (
    REPOSITORY_ROOT
    / "tools"
    / "tool-npu-compute"
    / "assets"
    / "hardware-info-catalog.json"
)
METRIC_CATALOG_PATH = (
    REPOSITORY_ROOT / "tools" / "tool-npu-compute" / "assets" / "metric-catalog.json"
)
PIPE_TRACE_CATALOG_PATH = (
    REPOSITORY_ROOT
    / "tools"
    / "tool-npu-compute"
    / "assets"
    / "pipe-trace-catalog.json"
)
SECTION_CATALOG_PATH = (
    REPOSITORY_ROOT / "tools" / "tool-npu-compute" / "assets" / "section-catalog.json"
)
SUMMARY_CATALOG_PATH = (
    REPOSITORY_ROOT / "tools" / "tool-npu-compute" / "assets" / "summary-catalog.json"
)
SECTION_EXPECTED_PATH = TEST_ROOT / "expected" / "section-lookups.json"
SUMMARY_EXPECTED_PATH = TEST_ROOT / "expected" / "summary-inspection.json"
SECTION_REFERENCES = {
    "ArithmeticUtilization": "arithmetic-utilization.md",
    "PipeUtilization": "pipe-utilization.md",
    "ResourceConflictRatio": "resource-conflict-ratio.md",
    "Memory": "memory.md",
    "MemoryL0": "memory-l0.md",
    "MemoryUB": "memory-ub.md",
    "L2Cache": "l2-cache.md",
}
BASIC_SECTION_SET = [
    "Pipeline",
    "PipeUtilization",
    "Memory",
    "MemoryL0",
    "MemoryUB",
    "L2Cache",
    "ArithmeticUtilization",
]
FULL_SECTION_SET = [*BASIC_SECTION_SET, "ResourceConflictRatio"]
EXPECTED_SECTION_SETS = {
    "basic": BASIC_SECTION_SET,
    "full": FULL_SECTION_SET,
}
USER_VISIBLE_METRIC_REFERENCES = (
    REFERENCE_ROOT / "metric-foundations.md",
    REFERENCE_ROOT / "data-availability.md",
    *(REFERENCE_ROOT / name for name in SECTION_REFERENCES.values()),
)


def load_json(path):
    with path.open(encoding="utf-8") as input_file:
        return json.load(input_file)


def reference_field_rows(path):
    rows = []
    for line_number, line in enumerate(
        path.read_text(encoding="utf-8").splitlines(), start=1
    ):
        cells = [cell.strip() for cell in line.strip().split("|")]
        if len(cells) < 4 or cells[0] or cells[-1]:
            continue
        name = cells[1]
        if len(name) < 2 or not name.startswith("`") or not name.endswith("`"):
            continue
        rows.append((line_number, [name[1:-1], *cells[2:-1]]))
    return rows


class CatalogContractTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.manifest = load_json(MANIFEST_PATH)
        cls.hardware_info_catalog = load_json(HARDWARE_INFO_CATALOG_PATH)
        cls.metric_catalog = load_json(METRIC_CATALOG_PATH)
        cls.pipe_trace_catalog = load_json(PIPE_TRACE_CATALOG_PATH)

    def test_metric_catalog_has_all_supported_csv_sections(self):
        self.assertEqual(
            list(self.metric_catalog["sections"]), list(SECTION_REFERENCES)
        )

    def test_fixture_manifest_matches_files(self):
        expected_names = {
            self.hardware_info_catalog["output_file"],
            self.pipe_trace_catalog["output_file"],
            "summary.jsonl",
        } | {f"{section}.csv" for section in SECTION_REFERENCES}
        manifest_files = {entry["name"]: entry for entry in self.manifest["files"]}
        self.assertEqual(set(manifest_files), expected_names)
        self.assertEqual(self.manifest["excluded_files"], [".hardware_info.lock"])
        self.assertFalse((FIXTURE_ROOT / ".hardware_info.lock").exists())

        for name, entry in manifest_files.items():
            path = FIXTURE_ROOT / name
            self.assertTrue(path.is_file(), name)
            self.assertEqual(path.stat().st_size, entry["size"])
            self.assertEqual(
                hashlib.sha256(path.read_bytes()).hexdigest(), entry["sha256"]
            )

    def test_csv_fixtures_match_catalog_and_have_nonempty_rows(self):
        sections = self.metric_catalog["sections"]
        self.assertEqual(list(sections), list(SECTION_REFERENCES))
        for section_name, reference_name in SECTION_REFERENCES.items():
            section = sections[section_name]
            path = FIXTURE_ROOT / section["output_file"]
            with path.open(newline="", encoding="utf-8") as input_file:
                rows = list(csv.reader(input_file))
            self.assertGreater(len(rows), 1, path.name)
            self.assertEqual(
                rows[0], [field["name"] for field in section["fields"]], path.name
            )
            for row_number, row in enumerate(rows[1:], start=2):
                self.assertEqual(len(row), len(rows[0]), f"{path.name}:{row_number}")
                self.assertTrue(
                    any(value != "" for value in row), f"{path.name}:{row_number}"
                )

            reference_rows = reference_field_rows(REFERENCE_ROOT / reference_name)
            self.assertEqual(
                [cells[0] for _, cells in reference_rows], rows[0], section_name
            )

    def test_hardware_info_fixture_matches_catalog_and_device(self):
        path = FIXTURE_ROOT / self.hardware_info_catalog["output_file"]
        records = [
            json.loads(line)
            for line in path.read_text(encoding="utf-8").splitlines()
            if line
        ]
        categories = self.hardware_info_catalog["categories"]
        self.assertEqual(
            [record["category"] for record in records],
            [category["name"] for category in categories],
        )
        for record, category in zip(records, categories):
            self.assertEqual(
                list(record), [field["name"] for field in category["fields"]]
            )

        device = next(
            record for record in records if record["category"] == "Device Info"
        )
        self.assertEqual(device["chip info"], self.manifest["device"]["chip_info"])
        self.assertEqual(device["arch info"], self.manifest["device"]["arch_info"])

    def test_section_references_cover_every_csv_field_in_catalog_order(self):
        sections = self.metric_catalog["sections"]
        self.assertEqual(list(sections), list(SECTION_REFERENCES))
        required_metadata = {
            "name",
            "meaning",
            "core_types",
            "unit",
            "dependencies",
            "formula",
            "na_conditions",
            "analysis_notes",
        }
        core_types = {
            "通用": ["common"],
            "AIC": ["AIC"],
            "AIV": ["AIV"],
            "AIC、AIV": ["AIC", "AIV"],
        }
        for section_name, reference_name in SECTION_REFERENCES.items():
            fields = sections[section_name]["fields"]
            reference_rows = reference_field_rows(REFERENCE_ROOT / reference_name)
            self.assertTrue(reference_rows, reference_name)
            self.assertEqual(
                [field["name"] for field in fields],
                [cells[0] for _, cells in reference_rows],
            )
            for field, (line_number, cells) in zip(fields, reference_rows):
                self.assertEqual(len(cells), 8, f"{reference_name}:{line_number}")
                self.assertTrue(all(cells), f"{reference_name}:{line_number}")
                self.assertEqual(set(field), required_metadata)
                self.assertEqual(field["meaning"], cells[1])
                expected_core_types = core_types.get(cells[2])
                self.assertIsNotNone(expected_core_types, cells[2])
                self.assertEqual(field["core_types"], expected_core_types)
                self.assertEqual(field["unit"], cells[3])
                self.assertEqual(field["dependencies"], [cells[4]])
                self.assertEqual(field["formula"], cells[5])
                self.assertEqual(field["na_conditions"], [cells[6]])
                self.assertEqual(field.get("analysis_notes"), cells[7])
                self.assertTrue(all(value for value in field.values()))

            field_names = [field["name"] for field in fields]
            self.assertEqual(len(field_names), len(set(field_names)), section_name)

    def test_user_visible_metric_data_does_not_expose_event_ids(self):
        paths = (METRIC_CATALOG_PATH, *USER_VISIBLE_METRIC_REFERENCES)
        for path in paths:
            with self.subTest(path=path.name):
                self.assertNotRegex(path.read_text(encoding="utf-8"), r"0x[0-9A-Fa-f]+")

    def test_hardware_info_catalog_has_complete_metadata(self):
        self.assertEqual(self.hardware_info_catalog["schema_version"], 1)
        actual_categories = self.hardware_info_catalog["categories"]
        self.assertEqual(
            self.hardware_info_catalog["format"]["record_order"],
            [category["name"] for category in actual_categories],
        )
        required_metadata = {
            "name",
            "json_type",
            "meaning",
            "unit",
            "source",
            "variability",
            "analysis",
            "notes",
        }
        for category in actual_categories:
            fields = category["fields"]
            for field in fields:
                self.assertEqual(set(field), required_metadata)
                self.assertTrue(all(value != "" for value in field.values()))
            field_names = [field["name"] for field in fields]
            self.assertEqual(len(field_names), len(set(field_names)), category["name"])

    def test_hardware_info_reference_covers_every_field_in_category_order(self):
        rows = reference_field_rows(REFERENCE_ROOT / "hardware-info.md")
        expected_fields = [
            field["name"]
            for category in self.hardware_info_catalog["categories"]
            for field in category["fields"]
        ]
        self.assertEqual([cells[0] for _, cells in rows], expected_fields)
        for line_number, cells in rows:
            self.assertEqual(len(cells), 6, f"hardware-info.md:{line_number}")
            self.assertTrue(all(cells), f"hardware-info.md:{line_number}")

    def test_pipe_trace_catalog_describes_current_format(self):
        self.assertTrue(PIPE_TRACE_CATALOG_PATH.is_file(), "pipe-trace-catalog.json")
        catalog = load_json(PIPE_TRACE_CATALOG_PATH)

        self.assertEqual(catalog["schema_version"], 1)
        self.assertEqual(catalog["section"], "Pipeline")
        self.assertEqual(catalog["output_file"], "PipeTrace.json")
        self.assertEqual(
            catalog["format"]["top_level_fields"],
            ["displayTimeUnit", "profilingType", "schemaVersion", "traceEvents"],
        )
        self.assertEqual(
            catalog["format"]["event_fields"],
            ["cname", "dur", "name", "ph", "pid", "tid", "ts"],
        )
        self.assertEqual(catalog["format"]["display_time_unit"], "ns")
        self.assertEqual(catalog["format"]["timestamp_unit"], "us")
        self.assertEqual(catalog["format"]["event_phase"], "X")
        self.assertEqual(
            catalog["pipelines"],
            {
                "SCALAR": {"color": "startup"},
                "VECTOR": {"color": "rail_idle"},
                "CUBE": {"color": "rail_response"},
                "MTE1": {"color": "thread_state_iowait"},
                "MTE2": {"color": "yellow"},
                "MTE3": {"color": "rail_animation"},
                "FIXP": {"color": "thread_state_unknown"},
            },
        )
        self.assertEqual(catalog["core_tracks"], ["cubecore", "veccore0", "veccore1"])
        self.assertEqual(
            set(catalog["pid_patterns"]), {"single_result", "merged_results"}
        )

    def test_pipe_trace_reference_covers_catalog_fields_and_pipelines(self):
        rows = reference_field_rows(REFERENCE_ROOT / "pipeline.md")
        documented_names = {cells[0] for _, cells in rows}
        catalog = load_json(PIPE_TRACE_CATALOG_PATH)
        required_names = {
            *catalog["format"]["top_level_fields"],
            *catalog["format"]["event_fields"],
            *catalog["pipelines"],
        }
        self.assertTrue(required_names.issubset(documented_names))

    def test_pipe_trace_fixture_matches_catalog(self):
        path = FIXTURE_ROOT / self.pipe_trace_catalog["output_file"]
        self.assertTrue(path.is_file(), path.name)
        document = load_json(path)
        self.assertEqual(
            list(document), self.pipe_trace_catalog["format"]["top_level_fields"]
        )
        self.assertEqual(
            document["displayTimeUnit"],
            self.pipe_trace_catalog["format"]["display_time_unit"],
        )
        self.assertEqual(
            document["profilingType"],
            self.pipe_trace_catalog["format"]["profiling_type"],
        )
        self.assertEqual(
            document["schemaVersion"],
            self.pipe_trace_catalog["format"]["trace_schema_version"],
        )
        self.assertTrue(document["traceEvents"])
        for event in document["traceEvents"]:
            self.assertEqual(
                list(event), self.pipe_trace_catalog["format"]["event_fields"]
            )
            self.assertEqual(
                event["cname"],
                self.pipe_trace_catalog["pipelines"][event["name"]]["color"],
            )
            self.assertEqual(event["tid"], event["name"])
            self.assertEqual(
                event["ph"], self.pipe_trace_catalog["format"]["event_phase"]
            )

    def test_section_catalog_matches_public_results(self):
        expected = load_json(SECTION_EXPECTED_PATH)
        self.assertTrue(
            SECTION_CATALOG_PATH.is_file(),
            "尚未实现 assets/section-catalog.json",
        )
        catalog = load_json(SECTION_CATALOG_PATH)
        self.assertEqual(catalog["schema_version"], expected["schema_version"])
        self.assertEqual(catalog["sets"], expected["sets"])
        self.assertEqual(catalog["sections"], expected["sections"])
        self.assertEqual(catalog["default_results"], expected["default_results"])

    def test_section_catalog_sets_match_cli_contract(self):
        catalog = load_json(SECTION_CATALOG_PATH)
        self.assertEqual(catalog["sets"], EXPECTED_SECTION_SETS)
        self.assertEqual(list(catalog["sets"]), ["basic", "full"])
        self.assertEqual(catalog["sets"]["full"][:-1], catalog["sets"]["basic"])
        self.assertEqual(catalog["sets"]["full"][-1], "ResourceConflictRatio")

    def test_section_catalog_set_members_are_known_and_unique(self):
        catalog = load_json(SECTION_CATALOG_PATH)
        known_sections = {section["name"] for section in catalog["sections"]}
        for set_name, members in catalog["sets"].items():
            with self.subTest(set_name=set_name):
                self.assertTrue(members)
                self.assertEqual(len(members), len(set(members)))
                self.assertTrue(set(members).issubset(known_sections))

    def test_summary_catalog_covers_declared_record_types(self):
        expected = load_json(SUMMARY_EXPECTED_PATH)
        self.assertTrue(
            SUMMARY_CATALOG_PATH.is_file(),
            "尚未实现 assets/summary-catalog.json",
        )
        catalog = load_json(SUMMARY_CATALOG_PATH)
        self.assertEqual(catalog["schema_version"], expected["schema_version"])
        self.assertEqual(list(catalog["record_types"]), expected["record_categories"])

        section_categories = expected["record_categories"][:-1]
        for category in section_categories:
            self.assert_summary_section(catalog["record_types"][category], category)

        op_info = catalog["record_types"]["OpInfoSummary"]
        self.assert_op_info_summary(op_info)

    def assert_summary_section(self, record_type, category):
        self.assertEqual(record_type["kind"], "pmu-section-summary")
        self.assertEqual(record_type["data_level"], "task-average")
        self.assertEqual(
            record_type["metric_fields"],
            {
                "catalog": "metric-catalog.json",
                "section": category,
                "excluded_fields": ["block_id", "sub_block_id"],
            },
        )

    def assert_op_info_summary(self, op_info):
        self.assertEqual(op_info["kind"], "operation-summary")
        self.assertEqual(op_info["data_level"], "kernel")
        expected_names = [
            "category",
            "Op Name",
            "Op Type",
            "Task Duration(us)",
            "Block Dim",
            "Mix Block Dim",
            "Device Id",
            "Pid",
            "Current Freq",
            "Rated Freq",
            "aicore_parallel_utilization",
            "aicore_parallel_balance",
            "aicore_gm_bw_theoretical(GB/s)",
            "aicore_gm_read_bw(GB/s)",
            "aicore_gm_write_bw(GB/s)",
            "aicore_gm_bw_usage_rate(%)",
        ]
        self.assertEqual([field["name"] for field in op_info["fields"]], expected_names)
        required_metadata = {
            "name",
            "json_types",
            "meaning",
            "unit",
            "nullable_when",
            "analysis",
        }
        for field in op_info["fields"]:
            self.assertEqual(set(field), required_metadata)
            self.assertTrue(all(value != "" for value in field.values()))

    def test_pipe_trace_catalog_declares_sampling_boundaries(self):
        catalog = load_json(PIPE_TRACE_CATALOG_PATH)
        self.assertIn("sampling", catalog, "PipeTrace 目录尚未声明 Pipeline 采样边界")
        self.assertEqual(catalog["sampling"]["maximum_sampled_groups"], 6)
        self.assertEqual(catalog["sampling"]["group_id_range"], [0, 5])
        self.assertEqual(
            catalog["sampling"]["core_tracks"],
            ["cubecore", "veccore0", "veccore1"],
        )
        self.assertFalse(catalog["sampling"]["tracks_represent_all_enabled_cores"])


if __name__ == "__main__":
    unittest.main()
