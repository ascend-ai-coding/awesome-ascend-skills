#!/usr/bin/env python3
# ----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ----------------------------------------------------------------------------------------------------------
"""校验 tool-npu-compute 的结构、目录表和真实样本。"""

import csv
import hashlib
import json
import logging
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from cli_support import configure_cli_loggers

from inspect_collection import InspectionError as CollectionInspectionError
from inspect_collection import inspect as inspect_collection
from inspect_pipe_trace import InspectionError as PipeTraceInspectionError
from inspect_pipe_trace import inspect as inspect_pipe_trace
from inspect_summary import InspectionError as SummaryInspectionError
from inspect_summary import inspect as inspect_summary


LOGGER, OUTPUT_LOGGER = configure_cli_loggers(__name__)

SKILL_ROOT = Path(__file__).resolve().parents[1]
TEST_ROOT = SKILL_ROOT / "tests"
FIXTURE_ROOT = TEST_ROOT / "fixtures" / "real-capture"
MANIFEST_PATH = FIXTURE_ROOT / "fixture-manifest.json"

SECTION_REFERENCES = {
    "ArithmeticUtilization": "arithmetic-utilization.md",
    "PipeUtilization": "pipe-utilization.md",
    "ResourceConflictRatio": "resource-conflict-ratio.md",
    "Memory": "memory.md",
    "MemoryL0": "memory-l0.md",
    "MemoryUB": "memory-ub.md",
    "L2Cache": "l2-cache.md",
}
METRIC_FIELD_KEYS = {
    "name",
    "meaning",
    "core_types",
    "unit",
    "dependencies",
    "formula",
    "na_conditions",
    "analysis_notes",
}
HARDWARE_FIELD_KEYS = {
    "name",
    "json_type",
    "meaning",
    "unit",
    "source",
    "variability",
    "analysis",
    "notes",
}
SUMMARY_FIELD_KEYS = {
    "name",
    "json_types",
    "meaning",
    "unit",
    "nullable_when",
    "analysis",
}
CORE_TYPES = {
    "通用": ["common"],
    "AIC": ["AIC"],
    "AIV": ["AIV"],
    "AIC、AIV": ["AIC", "AIV"],
}
SUPPORTED_SECTION_NAMES = [
    "PipeUtilization",
    "Memory",
    "MemoryL0",
    "MemoryUB",
    "L2Cache",
    "Pipeline",
    "ArithmeticUtilization",
    "ResourceConflictRatio",
]
SUMMARY_RECORD_TYPES = [
    "PipeUtilization",
    "Memory",
    "MemoryL0",
    "MemoryUB",
    "L2Cache",
    "ArithmeticUtilization",
    "ResourceConflictRatio",
    "OpInfoSummary",
]


class ValidationError(Exception):
    """表示 Skill 资源不一致。"""


def require(condition, message):
    if not condition:
        raise ValidationError(message)


def load_json(path):
    require(path.is_file(), f"文件不存在：{path}")
    try:
        with path.open(encoding="utf-8") as input_file:
            return json.load(input_file)
    except (OSError, json.JSONDecodeError) as error:
        raise ValidationError(f"JSON 读取失败：{path}: {error}") from error


def markdown_links(text):
    return re.findall(r"\[[^\]]+\]\(([^)]+)\)", text)


def reference_field_rows(path):
    require(path.is_file(), f"字段参考不存在：{path.name}")
    rows = []
    for line_number, line in enumerate(
        path.read_text(encoding="utf-8").splitlines(), start=1
    ):
        match = re.match(r"^\|\s*`([^`]+)`\s*\|(.*)\|\s*$", line)
        if match is None:
            continue
        cells = [match.group(1)] + [cell.strip() for cell in match.group(2).split("|")]
        rows.append((line_number, cells))
    return rows


def parse_frontmatter(text):
    lines = text.splitlines()
    require(lines and lines[0] == "---", "SKILL.md 必须以 YAML frontmatter 开始")
    try:
        end = lines.index("---", 1)
    except ValueError as error:
        raise ValidationError("SKILL.md 的 YAML frontmatter 未闭合") from error
    frontmatter = {}
    for line in lines[1:end]:
        if not line.strip():
            continue
        key, separator, value = line.partition(":")
        require(bool(separator), f"SKILL.md frontmatter 行格式错误：{line}")
        frontmatter[key.strip()] = value.strip()
    return frontmatter


def _validate_local_links(skill_text):
    links = set(markdown_links(skill_text))
    local_links = {
        link for link in links if "://" not in link and not link.startswith("#")
    }
    for link in local_links:
        target = SKILL_ROOT / link.split("#", 1)[0]
        require(target.is_file(), f"SKILL.md 链接的文件不存在：{link}")
    return links


def _validate_reference_links(links):
    references = sorted((SKILL_ROOT / "references").glob("*.md"))
    require(bool(references), "references 目录不能为空")
    reference_paths = {path.resolve() for path in references}
    for reference in references:
        relative = f"references/{reference.name}"
        require(relative in links, f"SKILL.md 未直接链接：{relative}")
        for link in markdown_links(reference.read_text(encoding="utf-8")):
            if "://" in link or link.startswith("#"):
                continue
            target = (reference.parent / link.split("#", 1)[0]).resolve()
            require(
                target not in reference_paths,
                f"reference 不得互相引用：{reference.name} -> {target.name}",
            )


def _validate_required_resources(links):
    required_resources = {
        "assets/analysis-report-template.md",
        "assets/metric-catalog.json",
        "assets/hardware-info-catalog.json",
        "assets/pipe-trace-catalog.json",
        "assets/section-catalog.json",
        "assets/summary-catalog.json",
        "references/cli-usage.md",
        "references/collection-workflow.md",
        "references/summary.md",
        "scripts/lookup_metric.py",
        "scripts/lookup_section.py",
        "scripts/inspect_csv.py",
        "scripts/inspect_collection.py",
        "scripts/inspect_hardware_info.py",
        "scripts/inspect_pipe_trace.py",
        "scripts/inspect_summary.py",
        "scripts/validate_catalog.py",
    }
    for resource in required_resources:
        require(resource in links, f"SKILL.md 未直接链接：{resource}")


def validate_structure():
    skill_path = SKILL_ROOT / "SKILL.md"
    require(skill_path.is_file(), f"文件不存在：{skill_path}")
    skill_text = skill_path.read_text(encoding="utf-8")
    frontmatter = parse_frontmatter(skill_text)
    require(
        set(frontmatter) == {"name", "description", "license"},
        "SKILL.md frontmatter 必须包含 name、description 和 license",
    )
    require(frontmatter["name"] == "tool-npu-compute", "SKILL.md name 不正确")
    require(bool(frontmatter["description"]), "SKILL.md description 不能为空")
    require(frontmatter["license"] == "CANN-2.0", "SKILL.md license 不正确")
    links = _validate_local_links(skill_text)
    _validate_reference_links(links)
    _validate_required_resources(links)


def _validate_metric_section(section_name, reference_name, section):
    require(
        section.get("output_file") == f"{section_name}.csv",
        f"{section_name} 输出文件不一致",
    )
    fields = section.get("fields", [])
    names = [field.get("name") for field in fields]
    require(bool(names), f"{section_name} 字段不能为空")
    require(len(names) == len(set(names)), f"{section_name} 存在重复字段")

    reference_rows = reference_field_rows(SKILL_ROOT / "references" / reference_name)
    require(
        [cells[0] for _, cells in reference_rows] == names,
        f"{section_name} 字段与 reference 不一致",
    )
    for field, (line_number, cells) in zip(fields, reference_rows):
        require(
            len(cells) == 8 and all(cells),
            f"{reference_name}:{line_number} 字段说明不完整",
        )
        require(
            set(field) == METRIC_FIELD_KEYS,
            f"{section_name}.{field.get('name')} 属性不完整",
        )
        require(
            all(value for value in field.values()),
            f"{section_name}.{field['name']} 存在空属性",
        )
        core_types = CORE_TYPES.get(cells[2])
        require(core_types is not None, f"{reference_name}:{line_number} 核类型无效")
        expected = {
            "name": cells[0],
            "meaning": cells[1],
            "core_types": core_types,
            "unit": cells[3],
            "dependencies": [cells[4]],
            "formula": cells[5],
            "na_conditions": [cells[6]],
            "analysis_notes": cells[7],
        }
        require(
            field == expected, f"{section_name}.{field['name']} 与 reference 不一致"
        )


def validate_metric_catalog():
    catalog = load_json(SKILL_ROOT / "assets" / "metric-catalog.json")
    require(catalog.get("schema_version") == 2, "CSV 指标目录 schema_version 必须为 2")
    sections = catalog.get("sections", {})
    require(
        list(sections) == list(SECTION_REFERENCES),
        "CSV 指标目录 Section 顺序不一致",
    )

    for section_name, reference_name in SECTION_REFERENCES.items():
        section = sections.get(section_name)
        require(section is not None, f"{section_name} Section 不存在")
        _validate_metric_section(section_name, reference_name, section)
    return catalog


def validate_hardware_catalog():
    catalog = load_json(SKILL_ROOT / "assets" / "hardware-info-catalog.json")
    require(
        catalog.get("schema_version") == 1, "HardwareInfo 目录 schema_version 必须为 1"
    )
    categories = catalog.get("categories", [])
    names = [category.get("name") for category in categories]
    require(bool(names), "HardwareInfo 类别不能为空")
    require(len(names) == len(set(names)), "HardwareInfo 存在重复类别")
    require(
        catalog.get("format", {}).get("record_order") == names,
        "HardwareInfo 记录顺序不一致",
    )
    for category in categories:
        fields = category.get("fields", [])
        field_names = [field.get("name") for field in fields]
        require(bool(field_names), f"HardwareInfo {category['name']} 字段不能为空")
        require(
            len(field_names) == len(set(field_names)),
            f"HardwareInfo {category['name']} 存在重复字段",
        )
        for field in fields:
            require(
                set(field) == HARDWARE_FIELD_KEYS,
                f"HardwareInfo {category['name']}.{field.get('name')} 属性不完整",
            )
            require(
                all(value != "" for value in field.values()),
                f"HardwareInfo {category['name']}.{field['name']} 存在空属性",
            )

    reference_rows = reference_field_rows(
        SKILL_ROOT / "references" / "hardware-info.md"
    )
    expected_fields = [
        field["name"] for category in categories for field in category["fields"]
    ]
    require(
        [cells[0] for _, cells in reference_rows] == expected_fields,
        "HardwareInfo 字段与 reference 不一致",
    )
    return catalog


def _validate_pipe_trace_format(trace_format):
    require(
        trace_format.get("top_level_fields")
        == ["displayTimeUnit", "profilingType", "schemaVersion", "traceEvents"],
        "PipeTrace 顶层字段不一致",
    )
    require(
        trace_format.get("event_fields")
        == ["cname", "dur", "name", "ph", "pid", "tid", "ts"],
        "PipeTrace 事件字段不一致",
    )
    require(
        trace_format.get("timestamp_unit") == "us",
        "PipeTrace 事件时间单位必须为 us",
    )


def _validate_pipe_trace_sampling(catalog, pipelines):
    require(bool(pipelines), "PipeTrace 流水线定义不能为空")
    require(
        all(
            isinstance(item, dict)
            and isinstance(item.get("color"), str)
            and item["color"]
            for item in pipelines.values()
        ),
        "PipeTrace 流水线颜色定义无效",
    )
    sampling = catalog.get("sampling", {})
    require(
        sampling.get("maximum_sampled_groups") == 6
        and sampling.get("group_id_range") == [0, 5],
        "PipeTrace 采样 Group 范围不一致",
    )
    require(
        sampling.get("core_tracks") == catalog.get("core_tracks"),
        "PipeTrace 采样 Core 轨道不一致",
    )
    require(
        sampling.get("tracks_represent_all_enabled_cores") is False,
        "PipeTrace 轨道边界说明不一致",
    )


def _validate_pipe_trace_reference(trace_format, pipelines):
    reference_rows = reference_field_rows(SKILL_ROOT / "references" / "pipeline.md")
    documented_names = {cells[0] for _, cells in reference_rows}
    required_names = {
        *trace_format["top_level_fields"],
        *trace_format["event_fields"],
        *pipelines,
    }
    require(
        required_names.issubset(documented_names),
        "PipeTrace 字段或流水线未在 reference 中完整说明",
    )


def validate_pipe_trace_catalog():
    catalog = load_json(SKILL_ROOT / "assets" / "pipe-trace-catalog.json")
    require(
        catalog.get("schema_version") == 1,
        "PipeTrace 目录 schema_version 必须为 1",
    )
    require(catalog.get("section") == "Pipeline", "PipeTrace Section 必须为 Pipeline")
    require(
        catalog.get("output_file") == "PipeTrace.json",
        "PipeTrace 输出文件必须为 PipeTrace.json",
    )
    trace_format = catalog.get("format", {})
    _validate_pipe_trace_format(trace_format)
    pipelines = catalog.get("pipelines", {})
    _validate_pipe_trace_sampling(catalog, pipelines)
    _validate_pipe_trace_reference(trace_format, pipelines)
    return catalog


def validate_section_catalog(metric_catalog, hardware_catalog, pipe_trace_catalog):
    catalog = load_json(SKILL_ROOT / "assets" / "section-catalog.json")
    require(catalog.get("schema_version") == 1, "Section 目录 schema_version 必须为 1")
    sections = catalog.get("sections", [])
    require(
        [section.get("name") for section in sections] == SUPPORTED_SECTION_NAMES,
        "Section 目录名称或顺序不一致",
    )
    require(
        set(metric_catalog["sections"]) | {pipe_trace_catalog["section"]}
        == set(SUPPORTED_SECTION_NAMES),
        "Section 目录与结果格式目录覆盖范围不一致",
    )
    for section in sections:
        name = section["name"]
        if name == "Pipeline":
            expected_file = pipe_trace_catalog["output_file"]
            expected_kind = "pipeline_trace"
            expected_inspector = "scripts/inspect_pipe_trace.py"
        else:
            expected_file = metric_catalog["sections"][name]["output_file"]
            expected_kind = "pmu_csv"
            expected_inspector = "scripts/inspect_csv.py"
        require(section.get("result_file") == expected_file, f"{name} 结果文件不一致")
        require(section.get("result_kind") == expected_kind, f"{name} 结果类型不一致")
        require(
            section.get("inspector") == expected_inspector,
            f"{name} 检查脚本不一致",
        )
        require(
            (SKILL_ROOT / section.get("reference", "")).is_file(),
            f"{name} reference 不存在",
        )

    defaults = catalog.get("default_results", [])
    require(
        [entry.get("result_file") for entry in defaults]
        == [hardware_catalog["output_file"], "summary.jsonl"],
        "Section 目录默认结果不一致",
    )
    return catalog


def validate_summary_section(category, record_type, metric_catalog):
    require(
        record_type.get("kind") == "pmu-section-summary"
        and record_type.get("data_level") == "task-average",
        f"summary {category} 记录属性不一致",
    )
    require(
        set(record_type.get("category_field", {})) == SUMMARY_FIELD_KEYS,
        f"summary {category}.category 元数据不完整",
    )
    require(
        record_type.get("metric_fields")
        == {
            "catalog": "metric-catalog.json",
            "section": category,
            "excluded_fields": ["block_id", "sub_block_id"],
        },
        f"summary {category} 指标字段来源不一致",
    )
    require(category in metric_catalog["sections"], f"summary {category} 没有指标定义")


def validate_op_info(op_info, names):
    require(
        op_info.get("kind") == "operation-summary"
        and op_info.get("data_level") == "kernel",
        "OpInfoSummary 记录属性不一致",
    )
    fields = op_info.get("fields", [])
    names.extend(field.get("name") for field in fields)
    require(bool(names), "OpInfoSummary 字段不能为空")
    require(len(names) == len(set(names)), "OpInfoSummary 存在重复字段")
    for field in fields:
        require(
            set(field) == SUMMARY_FIELD_KEYS,
            f"OpInfoSummary.{field.get('name')} 元数据不完整",
        )
        require(
            all(value != "" for value in field.values()),
            f"OpInfoSummary.{field['name']} 存在空属性",
        )
        require(
            isinstance(field["json_types"], list) and field["json_types"],
            f"OpInfoSummary.{field['name']} 类型定义无效",
        )

    return fields


def validate_summary_reference(fields, names):
    reference_rows = reference_field_rows(SKILL_ROOT / "references" / "summary.md")
    require(
        [cells[0] for _, cells in reference_rows] == names,
        "OpInfoSummary 字段与 reference 不一致",
    )
    for field, (line_number, cells) in zip(fields, reference_rows):
        require(
            len(cells) == 6 and all(cells),
            f"summary.md:{line_number} 字段说明不完整",
        )
        require(
            field["meaning"].replace("`", "") == cells[1].replace("`", ""),
            f"summary.md:{line_number} 含义不一致",
        )
        require(field["unit"] == cells[2], f"summary.md:{line_number} 单位不一致")
        require(
            field["json_types"] == cells[3].split("/"),
            f"summary.md:{line_number} 类型不一致",
        )
        require(
            field["nullable_when"].replace("`", "") == cells[4].replace("`", ""),
            f"summary.md:{line_number} null 条件不一致",
        )
        require(
            field["analysis"].replace("`", "") == cells[5].replace("`", ""),
            f"summary.md:{line_number} 分析说明不一致",
        )


def validate_summary_catalog(metric_catalog):
    catalog = load_json(SKILL_ROOT / "assets" / "summary-catalog.json")
    require(catalog.get("schema_version") == 1, "summary 目录 schema_version 必须为 1")
    require(catalog.get("output_file") == "summary.jsonl", "summary 输出文件名不一致")
    record_types = catalog.get("record_types", {})
    require(list(record_types) == SUMMARY_RECORD_TYPES, "summary 记录类别或顺序不一致")
    for category in SUMMARY_RECORD_TYPES[:-1]:
        validate_summary_section(category, record_types[category], metric_catalog)
    names = []
    fields = validate_op_info(record_types["OpInfoSummary"], names)
    selected_fields = catalog.get("op_info_selected_fields", [])
    require(
        selected_fields
        == [
            "Op Type",
            "Block Dim",
            "Mix Block Dim",
            "Device Id",
            "Current Freq",
            "Rated Freq",
        ],
        "OpInfoSummary 摘要字段不一致",
    )
    require(set(selected_fields).issubset(names), "OpInfoSummary 摘要字段不存在")

    validate_summary_reference(fields, names)
    return catalog


def validate_fixtures(
    metric_catalog, hardware_catalog, pipe_trace_catalog, summary_catalog
):
    expected_names = _validate_fixture_manifest(
        metric_catalog, hardware_catalog, pipe_trace_catalog, summary_catalog
    )
    _validate_metric_fixtures(metric_catalog)
    _validate_runtime_fixtures(
        metric_catalog,
        hardware_catalog,
        pipe_trace_catalog,
        summary_catalog,
        expected_names,
    )


def _validate_fixture_manifest(
    metric_catalog, hardware_catalog, pipe_trace_catalog, summary_catalog
):
    manifest = load_json(MANIFEST_PATH)
    expected_names = {
        hardware_catalog["output_file"],
        pipe_trace_catalog["output_file"],
        summary_catalog["output_file"],
    } | {section["output_file"] for section in metric_catalog["sections"].values()}
    manifest_files = {entry["name"]: entry for entry in manifest.get("files", [])}
    require(set(manifest_files) == expected_names, "真实样本文件列表不一致")
    require(
        manifest.get("excluded_files") == [".hardware_info.lock"],
        "真实样本排除文件列表不一致",
    )

    for name, entry in manifest_files.items():
        path = FIXTURE_ROOT / name
        require(path.is_file(), f"真实样本文件不存在：{name}")
        require(path.stat().st_size == entry.get("size"), f"真实样本大小不一致：{name}")
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        require(digest == entry.get("sha256"), f"真实样本 SHA-256 不一致：{name}")

    return expected_names


def _validate_metric_fixtures(metric_catalog):
    for section in metric_catalog["sections"].values():
        path = FIXTURE_ROOT / section["output_file"]
        with path.open(newline="", encoding="utf-8") as input_file:
            rows = list(csv.reader(input_file))
        expected_header = [field["name"] for field in section["fields"]]
        require(
            len(rows) > 1 and rows[0] == expected_header,
            f"真实样本表头或数据行不符合定义：{path.name}",
        )
        for line_number, row in enumerate(rows[1:], start=2):
            require(
                len(row) == len(rows[0]),
                f"真实样本列数不一致：{path.name}:{line_number}",
            )
            require(
                any(value != "" for value in row),
                f"真实样本存在空行：{path.name}:{line_number}",
            )


def _validate_hardware_fixture(hardware_catalog):
    hardware_path = FIXTURE_ROOT / hardware_catalog["output_file"]
    records = [
        json.loads(line)
        for line in hardware_path.read_text(encoding="utf-8").splitlines()
        if line
    ]
    categories = hardware_catalog["categories"]
    require(
        [record.get("category") for record in records]
        == [category["name"] for category in categories],
        "HardwareInfo 样本类别顺序不一致",
    )
    for record, category in zip(records, categories):
        require(
            list(record) == [field["name"] for field in category["fields"]],
            f"HardwareInfo 样本字段不一致：{category['name']}",
        )


def _validate_pipe_trace_fixture(pipe_trace_catalog):
    pipe_trace_path = FIXTURE_ROOT / pipe_trace_catalog["output_file"]
    pipe_trace_summary = inspect_pipe_trace(pipe_trace_path, pipe_trace_catalog)
    require(
        pipe_trace_summary["event_count"] > 0,
        "PipeTrace 真实样本必须包含事件",
    )
    require(
        pipe_trace_summary["track_count"] > 0,
        "PipeTrace 真实样本必须包含轨道",
    )
    require(
        pipe_trace_summary["sampled_group_ids"] == [0, 3]
        and pipe_trace_summary["sampled_group_count"] == 2
        and pipe_trace_summary["maximum_sampled_groups"] == 6
        and pipe_trace_summary["tracks_represent_all_enabled_cores"] is False,
        "PipeTrace 真实样本采样摘要不一致",
    )


def _validate_summary_fixture(metric_catalog, summary_catalog):
    summary_path = FIXTURE_ROOT / summary_catalog["output_file"]
    summary_result = inspect_summary(summary_path, summary_catalog, metric_catalog)
    require(
        summary_result["record_count"] > 1
        and bool(summary_result["section_categories"])
        and summary_result["op_info_summary"] is not None,
        "summary 真实样本缺少 Section 汇总或 OpInfoSummary",
    )


def _validate_collection_fixture(
    metric_catalog,
    hardware_catalog,
    pipe_trace_catalog,
    summary_catalog,
    expected_names,
):
    collection_result = inspect_collection(
        FIXTURE_ROOT,
        metric_catalog,
        hardware_catalog,
        pipe_trace_catalog,
        summary_catalog,
    )
    require(
        collection_result["root_result"]["recognized_files"] == sorted(expected_names),
        "完整解包结果检查未识别全部真实样本",
    )
    require(
        collection_result["collection_results"] == [],
        "真实样本不应包含子结果集合",
    )
    require(
        collection_result["unknown_files"] == ["fixture-manifest.json"],
        "完整解包结果检查的未知项目不一致",
    )


def _validate_runtime_fixtures(
    metric_catalog,
    hardware_catalog,
    pipe_trace_catalog,
    summary_catalog,
    expected_names,
):
    _validate_hardware_fixture(hardware_catalog)
    _validate_pipe_trace_fixture(pipe_trace_catalog)
    _validate_summary_fixture(metric_catalog, summary_catalog)
    _validate_collection_fixture(
        metric_catalog,
        hardware_catalog,
        pipe_trace_catalog,
        summary_catalog,
        expected_names,
    )


def main():
    logging.basicConfig(format="%(message)s")
    try:
        validate_structure()
        metric_catalog = validate_metric_catalog()
        hardware_catalog = validate_hardware_catalog()
        pipe_trace_catalog = validate_pipe_trace_catalog()
        validate_section_catalog(metric_catalog, hardware_catalog, pipe_trace_catalog)
        summary_catalog = validate_summary_catalog(metric_catalog)
        validate_fixtures(
            metric_catalog,
            hardware_catalog,
            pipe_trace_catalog,
            summary_catalog,
        )
    except (
        OSError,
        CollectionInspectionError,
        ValidationError,
        PipeTraceInspectionError,
        SummaryInspectionError,
        json.JSONDecodeError,
    ) as error:
        LOGGER.error("校验失败：%s", error)
        return 1

    OUTPUT_LOGGER.info("tool-npu-compute 目录校验通过")
    return 0


if __name__ == "__main__":
    sys.exit(main())
