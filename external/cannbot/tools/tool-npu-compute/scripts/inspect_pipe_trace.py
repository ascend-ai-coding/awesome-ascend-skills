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
import argparse
import json
import logging
import math
import re
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from cli_support import add_format_argument, configure_cli_loggers, emit_result

LOGGER, OUTPUT_LOGGER = configure_cli_loggers(__name__)

CATALOG_PATH = (
    Path(__file__).resolve().parents[1] / "assets" / "pipe-trace-catalog.json"
)
SINGLE_PID_PATTERN = re.compile(
    r"^group(?P<group_id>[0-9]+)\.(?P<core>cubecore|veccore0|veccore1)$"
)
MERGED_PID_PATTERN = re.compile(
    r"^process(?P<process_ordinal>[0-9]+)"
    r"\.result(?P<result_sequence>[0-9]+)"
    r"\.device(?P<device_id>[0-9]+)"
    r"\.replay(?P<replay_id>[0-9]+)"
    r"\.group(?P<group_id>[0-9]+)"
    r"\.(?P<core>cubecore|veccore0|veccore1)$"
)


class InspectionError(Exception):
    pass


def parse_arguments():
    parser = argparse.ArgumentParser(
        description="检查并汇总 npu-compute PipeTrace.json。"
    )
    parser.add_argument("json_path", type=Path, help="需要检查的 PipeTrace JSON 文件")
    add_format_argument(parser)
    return parser.parse_args()


def load_catalog():
    with CATALOG_PATH.open(encoding="utf-8") as input_file:
        return json.load(input_file)


def load_document(path):
    try:
        with path.open(encoding="utf-8") as input_file:
            document = json.load(input_file)
    except json.JSONDecodeError as error:
        raise InspectionError(f"JSON 格式错误：{error.msg}") from error
    except (OSError, UnicodeError) as error:
        raise InspectionError(f"读取 PipeTrace 失败：{error}") from error
    if not isinstance(document, dict):
        raise InspectionError("PipeTrace 顶层必须是 JSON 对象")
    return document


def validate_header(document, catalog):
    expected = {
        "displayTimeUnit": catalog["format"]["display_time_unit"],
        "profilingType": catalog["format"]["profiling_type"],
        "schemaVersion": catalog["format"]["trace_schema_version"],
    }
    for name, expected_value in expected.items():
        if name not in document:
            raise InspectionError(f"PipeTrace 缺少字段：{name}")
        if document[name] != expected_value:
            raise InspectionError(f"{name} 值应为 {expected_value}")
    if "traceEvents" not in document:
        raise InspectionError("PipeTrace 缺少字段：traceEvents")
    if not isinstance(document["traceEvents"], list):
        raise InspectionError("traceEvents 必须是数组")


def parse_pid(pid):
    match = MERGED_PID_PATTERN.fullmatch(pid)
    if match is not None:
        fields = {
            name: int(value)
            for name, value in match.groupdict().items()
            if name != "core"
        }
        fields["core"] = match.group("core")
        return fields

    match = SINGLE_PID_PATTERN.fullmatch(pid)
    if match is not None:
        return {
            "process_ordinal": None,
            "result_sequence": None,
            "device_id": None,
            "replay_id": None,
            "group_id": int(match.group("group_id")),
            "core": match.group("core"),
        }
    raise InspectionError(f"pid 格式不受支持：{pid}")


def require_string(event, name, event_number):
    value = event[name]
    if not isinstance(value, str) or not value:
        raise InspectionError(f"第 {event_number} 个事件的 {name} 必须是非空字符串")
    return value


def require_nonnegative_number(event, name, event_number):
    value = event[name]
    if not isinstance(value, (int, float)) or isinstance(value, bool):
        raise InspectionError(f"第 {event_number} 个事件的 {name} 必须是有限非负数")
    if not math.isfinite(value) or value < 0:
        raise InspectionError(f"第 {event_number} 个事件的 {name} 必须是有限非负数")
    return value


def validate_event(event, event_number, catalog):
    if not isinstance(event, dict):
        raise InspectionError(f"第 {event_number} 个事件必须是 JSON 对象")
    for name in catalog["format"]["event_fields"]:
        if name not in event:
            raise InspectionError(f"第 {event_number} 个事件缺少字段：{name}")

    name = require_string(event, "name", event_number)
    tid = require_string(event, "tid", event_number)
    if name not in catalog["pipelines"]:
        raise InspectionError(f"未知流水线：{name}")
    if tid != name:
        raise InspectionError(f"第 {event_number} 个事件 name 与 tid 不一致")

    color = require_string(event, "cname", event_number)
    if color != catalog["pipelines"][name]["color"]:
        raise InspectionError(f"第 {event_number} 个事件 cname 与 {name} 不一致")

    phase = require_string(event, "ph", event_number)
    if phase != catalog["format"]["event_phase"]:
        raise InspectionError(
            f"第 {event_number} 个事件 ph 必须为 {catalog['format']['event_phase']}"
        )

    pid = require_string(event, "pid", event_number)
    source = parse_pid(pid)
    first_group, last_group = catalog["sampling"]["group_id_range"]
    if not first_group <= source["group_id"] <= last_group:
        raise InspectionError(
            f"第 {event_number} 个事件的 Group ID 必须在 {first_group} 到 {last_group} 之间"
        )
    timestamp = require_nonnegative_number(event, "ts", event_number)
    duration = require_nonnegative_number(event, "dur", event_number)
    return {
        "name": name,
        "pid": pid,
        "tid": tid,
        "timestamp": timestamp,
        "duration": duration,
        "end": timestamp + duration,
        **source,
    }


def summarize_pipelines(events, catalog):
    grouped = defaultdict(list)
    for event in events:
        grouped[event["name"]].append(event)
    return [
        {
            "name": name,
            "event_count": len(grouped[name]),
            "duration_sum_us": sum(event["duration"] for event in grouped[name]),
        }
        for name in catalog["pipelines"]
        if name in grouped
    ]


def summarize_tracks(events):
    grouped = defaultdict(list)
    for event in events:
        grouped[(event["pid"], event["tid"])].append(event)

    tracks = []
    for (pid, tid), track_events in sorted(grouped.items()):
        source = track_events[0]
        tracks.append(
            {
                "pid": pid,
                "tid": tid,
                "process_ordinal": source["process_ordinal"],
                "result_sequence": source["result_sequence"],
                "device_id": source["device_id"],
                "replay_id": source["replay_id"],
                "group_id": source["group_id"],
                "core": source["core"],
                "event_count": len(track_events),
                "trace_start_us": min(event["timestamp"] for event in track_events),
                "trace_end_us": max(event["end"] for event in track_events),
                "duration_sum_us": sum(event["duration"] for event in track_events),
            }
        )
    return tracks


def summarize_sampling(events, catalog):
    sampled_core_tracks = defaultdict(set)
    for event in events:
        sampled_core_tracks[event["group_id"]].add(event["core"])

    core_order = {
        core: index for index, core in enumerate(catalog["sampling"]["core_tracks"])
    }
    group_ids = sorted(sampled_core_tracks)
    return {
        "sampled_group_ids": group_ids,
        "sampled_group_count": len(group_ids),
        "sampled_core_tracks": {
            str(group_id): sorted(
                sampled_core_tracks[group_id], key=core_order.__getitem__
            )
            for group_id in group_ids
        },
        "maximum_sampled_groups": catalog["sampling"]["maximum_sampled_groups"],
        "sampling_limited": True,
        "tracks_represent_all_enabled_cores": catalog["sampling"][
            "tracks_represent_all_enabled_cores"
        ],
    }


def inspect(path, catalog):
    document = load_document(path)
    validate_header(document, catalog)
    events = [
        validate_event(event, index, catalog)
        for index, event in enumerate(document["traceEvents"], start=1)
    ]

    if events:
        trace_start = min(event["timestamp"] for event in events)
        trace_end = max(event["end"] for event in events)
        trace_span = trace_end - trace_start
    else:
        trace_start = None
        trace_end = None
        trace_span = None

    tracks = summarize_tracks(events)
    return {
        "file": str(path),
        "section": catalog["section"],
        "header": {
            "display_time_unit": document["displayTimeUnit"],
            "profiling_type": document["profilingType"],
            "schema_version": document["schemaVersion"],
            "timestamp_unit": catalog["format"]["timestamp_unit"],
        },
        "event_count": len(events),
        "track_count": len(tracks),
        "trace_start_us": trace_start,
        "trace_end_us": trace_end,
        "trace_span_us": trace_span,
        **summarize_sampling(events, catalog),
        "pipelines": summarize_pipelines(events, catalog),
        "tracks": tracks,
    }


def format_number(value):
    return "无" if value is None else f"{value:.17g}"


def render_markdown(result):
    lines = [
        "# Pipeline 时间线检查",
        "",
        f"- **文件：** `{result['file']}`",
        f"- **事件数：** {result['event_count']}",
        f"- **轨道数：** {result['track_count']}",
        f"- **采样 Group：** {', '.join(map(str, result['sampled_group_ids'])) or '无'}",
        f"- **采样 Group 数：** {result['sampled_group_count']}",
        f"- **采样 Group 上限：** {result['maximum_sampled_groups']}",
        "- **轨道说明：** 轨道数量不能用于推断目标程序实际启用的 Core 数量。",
        "- **时间单位：微秒（us）**",
        f"- **开始时间：** {format_number(result['trace_start_us'])}",
        f"- **结束时间：** {format_number(result['trace_end_us'])}",
        f"- **时间跨度：** {format_number(result['trace_span_us'])}",
        "",
        "## 流水线摘要",
        "",
        "| 流水线 | 事件数 | 累计忙碌时间（us） |",
        "|---|---:|---:|",
    ]
    for pipeline in result["pipelines"]:
        lines.append(
            f"| `{pipeline['name']}` | {pipeline['event_count']} | "
            f"{format_number(pipeline['duration_sum_us'])} |"
        )
    if not result["pipelines"]:
        lines.append("| 无 | 0 | 0 |")

    lines.extend(
        (
            "",
            "## 轨道摘要",
            "",
            "| pid | tid | 事件数 | 开始时间（us） | 结束时间（us） | 累计忙碌时间（us） |",
            "|---|---|---:|---:|---:|---:|",
        )
    )
    for track in result["tracks"]:
        lines.append(
            f"| `{track['pid']}` | `{track['tid']}` | {track['event_count']} | "
            f"{format_number(track['trace_start_us'])} | "
            f"{format_number(track['trace_end_us'])} | "
            f"{format_number(track['duration_sum_us'])} |"
        )
    if not result["tracks"]:
        lines.append("| 无 | 无 | 0 | 无 | 无 | 0 |")
    return "\n".join(lines)


def main():
    logging.basicConfig(format="%(message)s")
    arguments = parse_arguments()
    try:
        result = inspect(arguments.json_path, load_catalog())
    except (InspectionError, json.JSONDecodeError, OSError) as error:
        LOGGER.error("错误：%s", error)
        return 2

    emit_result(OUTPUT_LOGGER, arguments.format, result, render_markdown)
    return 0


if __name__ == "__main__":
    sys.exit(main())
