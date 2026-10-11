#!/usr/bin/env python3
#
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
#
# ----------------------------------------------------------------------------------------------------------
"""report_session_stats.py — Session 统计模块（从 generate_report.py 拆分）。

职责：
- session JSONL 解析（时间戳、token 用量、模型名）
- main_session / evo_agent / subagent 耗时统计
- round 目录时间统计（birthtime、预创建修正）
- 资源消耗统计 HTML 区块（耗时表 + token 表）
"""

import json
import logging
import os
import shutil
import subprocess
from datetime import datetime
from pathlib import Path
from typing import Any

LOGGER = logging.getLogger(__name__)

STAT_BIN = shutil.which("stat") or "/usr/bin/stat"


def _parse_ts_value(ts) -> float | None:
    """把时间戳值（ISO 字符串或数值）解析为毫秒，失败返回 None。"""
    if not ts:
        return None
    try:
        if isinstance(ts, str):
            return datetime.fromisoformat(ts.replace("Z", "+00:00")).timestamp() * 1000
        if isinstance(ts, (int, float)):
            return ts
    except (ValueError, TypeError):
        return None
    return None


def _parse_jsonl_line(line: str):
    """解析单行 JSONL，失败返回 None。"""
    try:
        return json.loads(line)
    except json.JSONDecodeError:
        return None


def _parse_jsonl(filepath: str) -> list[dict]:
    """逐行解析 JSONL 文件，坏行跳过。"""
    try:
        with open(filepath, "r", encoding="utf-8") as f:
            return [r for r in (_parse_jsonl_line(line) for line in f) if r is not None]
    except FileNotFoundError:
        return []


def _extract_timestamps(records: list[dict]) -> tuple[float | None, float | None]:
    """从记录列表提取 (min_ts, max_ts) 毫秒时间戳。"""
    timestamps = [t for t in (_parse_ts_value(r.get("timestamp")) for r in records)
                  if t is not None]
    if timestamps:
        return min(timestamps), max(timestamps)
    return None, None


def _count_tokens(records: list[dict]) -> dict[str, int]:
    """汇总 assistant 记录的 token 用量。"""
    tokens = {"input": 0, "output": 0, "cache_read": 0, "cache_creation": 0}
    for r in records:
        if r.get("type") == "assistant":
            usage = r.get("message", {}).get("usage", {})
            tokens["input"] += usage.get("input_tokens", 0)
            tokens["output"] += usage.get("output_tokens", 0)
            tokens["cache_read"] += usage.get("cache_read_input_tokens", 0)
            tokens["cache_creation"] += usage.get("cache_creation_input_tokens", 0)
    return tokens


def _extract_model(records: list[dict]) -> str | None:
    """Extract model name from the first assistant message with a model field."""
    for r in records:
        if r.get("type") == "assistant":
            model = r.get("message", {}).get("model")
            if model:
                return model
    return None


def _find_first_user_message_ts(records: list[dict]) -> float | None:
    """Find timestamp of the first genuine user input (not tool_result wrapper)."""
    for r in records:
        if r.get("type") != "user":
            continue
        content = r.get("message", {}).get("content", "")
        # Genuine user input is a string; tool_result wrappers are lists
        if isinstance(content, str) and content.strip():
            ts = _parse_ts_value(r.get("timestamp"))
            if ts is not None:
                return ts
    return None


def _find_last_task_completion_ts(records: list[dict]) -> float | None:
    """Find timestamp of the last task-completion / agent-done notification.

    Returns the last task-notification timestamp (not the first), because
    earlier ones may be stale notifications from a resumed session.
    """
    last_ts = None
    for r in records:
        if r.get("type") != "user":
            continue
        content = str(r.get("message", {}).get("content", ""))
        if "task-notification" in content.lower() or "task_notification" in content.lower():
            ts = _parse_ts_value(r.get("timestamp"))
            if ts is not None:
                last_ts = ts
    return last_ts


def _calc_active_duration_ms(timestamps: list[float], gap_threshold_ms: float = 600_000) -> float:
    """Calculate active duration by summing intervals, completely excluding large gaps.

    Large gaps (> gap_threshold_ms) represent idle/wait time and are excluded
    entirely — no buffer is added. This prevents inflated timing when the session
    was left open or waiting for external processes.
    """
    if len(timestamps) < 2:
        return 0.0
    sorted_ts = sorted(timestamps)
    total = 0.0
    for i in range(1, len(sorted_ts)):
        gap = sorted_ts[i] - sorted_ts[i - 1]
        if gap <= gap_threshold_ms:
            total += gap
        # else: gap is idle time, exclude entirely
    return total


def _new_session_stats() -> dict[str, Any]:
    """构造 parse_session_stats 的初始统计结构。"""
    return {
        "timing": {
            "main_session": {"start": None, "end": None, "duration_minutes": 0},
            "evo_agent": {"start": None, "end": None, "duration_minutes": 0},
            "rounds": {},
            "total_duration_minutes": 0,
        },
        "tokens": {
            "main_session": {"input": 0, "output": 0, "cache_read": 0, "cache_creation": 0},
            "evo_agent": {"input": 0, "output": 0, "cache_read": 0, "cache_creation": 0},
            "aside_agents": [],
            "total": {"input": 0, "output": 0, "cache_read": 0, "cache_creation": 0},
        },
        "model": None,  # Extracted AI model name (e.g., "kimi-for-coding")
    }


def _resolve_main_end_ts(last_task_done_ts: float | None, raw_end: float | None):
    """确定 main session 结束时间。

    仅当 task-completion 时间与真实结束相差 30 min 内才采用；
    否则说明 agent 结束后 session 仍在继续（如报告重生成），取全程。
    """
    if last_task_done_ts and raw_end and (raw_end - last_task_done_ts) < 1_800_000:
        return last_task_done_ts
    return raw_end


def _fill_main_session_timing(timing: dict, main_records: list[dict]):
    """填充 main_session 的 start/end/duration。"""
    first_user_ts = _find_first_user_message_ts(main_records)
    last_task_done_ts = _find_last_task_completion_ts(main_records)
    raw_start, raw_end = _extract_timestamps(main_records)

    start = first_user_ts if first_user_ts is not None else raw_start
    end = _resolve_main_end_ts(last_task_done_ts, raw_end)

    if not (start and end):
        if raw_start and raw_end:
            timing["main_session"]["start"] = raw_start
            timing["main_session"]["end"] = raw_end
            timing["main_session"]["duration_minutes"] = (raw_end - raw_start) / 1000 / 60
        return

    timing["main_session"]["start"] = start
    timing["main_session"]["end"] = end
    # Calculate active duration: filter out large idle gaps (>10 min)
    all_ts = [t for t in (_parse_ts_value(r.get("timestamp")) for r in main_records)
              if t is not None]
    # Only count timestamps within [start, end]
    filtered_ts = [t for t in all_ts if start <= t <= end]
    active_ms = _calc_active_duration_ms(filtered_ts, gap_threshold_ms=600_000)
    range_ms = end - start
    # When the session was mostly waiting for a background agent
    # (active time < 15% of wall-clock span), the wall-clock span
    # is more representative of the user-facing duration.
    # Otherwise use active duration to exclude idle gaps.
    if range_ms > 0 and active_ms / range_ms < 0.15 and range_ms > 300_000:
        timing["main_session"]["duration_minutes"] = range_ms / 1000 / 60
    else:
        # Use active duration directly — no artificial floor.
        # This excludes post-task wait time and long idle gaps.
        timing["main_session"]["duration_minutes"] = active_ms / 1000 / 60


def _parse_subagent_stats(stats: dict, subagent_dir: Path | None) -> float | None:
    """解析 subagents 目录，识别 evo agent（按 token 用量最高者）。

    返回 evo agent 的结束时间戳（无则 None）。
    """
    if not (subagent_dir and subagent_dir.is_dir()):
        return None

    agent_stats = []
    for agent_file in subagent_dir.glob("*.jsonl"):
        records = _parse_jsonl(str(agent_file))
        tokens = _count_tokens(records)
        start, end = _extract_timestamps(records)
        agent_stats.append({
            "name": agent_file.stem,
            "records": records,
            "tokens": tokens,
            "total_tokens": sum(tokens.values()),
            "start": start,
            "end": end,
        })

    # Sort by total token usage descending
    agent_stats.sort(key=lambda x: x["total_tokens"], reverse=True)

    evo_agent_end_ts = None
    # The agent with highest token usage is the evo agent (main orchestrator)
    if agent_stats:
        evo = agent_stats[0]
        if evo["total_tokens"] > 1000:  # Must have meaningful token usage
            stats["timing"]["evo_agent"]["start"] = evo["start"]
            stats["timing"]["evo_agent"]["end"] = evo["end"]
            evo_agent_end_ts = evo["end"]
            if evo["start"] and evo["end"]:
                stats["timing"]["evo_agent"]["duration_minutes"] = (
                    evo["end"] - evo["start"]
                ) / 1000 / 60
            stats["tokens"]["evo_agent"] = evo["tokens"]
            # Also extract model from evo agent if main session didn't have it
            if not stats["model"]:
                stats["model"] = _extract_model(evo["records"])

        # Remaining agents are aside agents (if they have token usage)
        for aside in agent_stats[1:]:
            if aside["total_tokens"] > 1000:
                stats["tokens"]["aside_agents"].append(
                    {"name": aside["name"], **aside["tokens"]}
                )
    return evo_agent_end_ts


def _accumulate_token_totals(stats: dict):
    """汇总 main_session + evo_agent + aside_agents 的 token 总量。"""
    total_tokens = stats["tokens"]["total"]
    for component in ["main_session", "evo_agent"]:
        comp_tokens = stats["tokens"][component]
        for key in ["input", "output", "cache_read", "cache_creation"]:
            total_tokens[key] += comp_tokens[key]
    for aside in stats["tokens"]["aside_agents"]:
        for key in ["input", "output", "cache_read", "cache_creation"]:
            total_tokens[key] += aside[key]


def _compute_total_duration(stats: dict):
    """计算总时长：优先 evo_agent（代表实际优化工作），否则 main_session。"""
    evo_dur = stats["timing"]["evo_agent"]["duration_minutes"]
    main_dur = stats["timing"]["main_session"]["duration_minutes"]
    if evo_dur > 0:
        stats["timing"]["total_duration_minutes"] = evo_dur
    elif main_dur > 0:
        stats["timing"]["total_duration_minutes"] = main_dur
    else:
        stats["timing"]["total_duration_minutes"] = 0


def parse_session_stats(session_dir: str, session_jsonl: str | None = None) -> dict[str, Any]:
    """Parse session records to extract timing, token statistics, and model info.

    Args:
        session_dir: Path to .claude/projects directory containing session files
        session_jsonl: Path to the specific JSONL file for this session (optional)

    Returns:
        Dictionary with timing, token statistics, and model information
    """
    stats = _new_session_stats()

    # Find session files
    session_path = Path(session_dir)
    session_files = list(session_path.glob("*.jsonl"))

    # Determine subagent directory location
    subagent_dir = None
    if session_jsonl:
        jsonl_path = Path(session_jsonl)
        subagent_dir = jsonl_path.parent / jsonl_path.stem / "subagents"

    # Parse main session
    main_records: list[dict] = []
    if session_jsonl:
        main_records = _parse_jsonl(session_jsonl)
    else:
        for sf in session_files:
            main_records = _parse_jsonl(str(sf))
            if main_records:
                break

    if main_records:
        _fill_main_session_timing(stats["timing"], main_records)
        stats["tokens"]["main_session"] = _count_tokens(main_records)
        stats["model"] = _extract_model(main_records)

    # Parse subagents: identify evo agent by highest token usage (not by name)
    evo_agent_end_ts = _parse_subagent_stats(stats, subagent_dir)

    # If main session end was not found via task notification but we have evo agent end,
    # use evo agent end as a fallback for main session end (the main session waited for it)
    if stats["timing"]["main_session"]["end"] is None and evo_agent_end_ts:
        stats["timing"]["main_session"]["end"] = evo_agent_end_ts
        if stats["timing"]["main_session"]["start"]:
            start = stats["timing"]["main_session"]["start"]
            stats["timing"]["main_session"]["duration_minutes"] = (evo_agent_end_ts - start) / 1000 / 60

    # Calculate totals
    _accumulate_token_totals(stats)
    _compute_total_duration(stats)

    return stats


def _get_dir_birthtime(path: Path) -> float | None:
    """Get directory creation (birth) time.

    On Linux, os.stat().st_ctime is the inode change time (updated when files
    are written into the directory), NOT the creation time.  We use `stat -c '%W'`
    to obtain the true birth time.  Falls back to st_ctime if unavailable.
    Returns None only if the path does not exist or is inaccessible.
    """
    try:
        result = subprocess.run(
            [STAT_BIN, "-c", "%W", str(path)],
            capture_output=True, text=True, check=True,
        )
        ts = float(result.stdout.strip())
        if ts > 0:
            return ts
    except Exception as e:
        # stat 命令不可用时回退 st_ctime，仅记录不中断
        LOGGER.debug("stat -c %%W failed for %s: %s", path, e)
    # Fallback: st_ctime is wrong on Linux for this purpose, but better than nothing
    try:
        return os.stat(path).st_ctime
    except OSError:
        return None


def _ts_from_jsonl_line(line: str):
    """从单行 JSONL 提取时间戳（毫秒），无则 None。"""
    rec = _parse_jsonl_line(line)
    if rec is None:
        return None
    return _parse_ts_value(rec.get("timestamp"))


def _iter_session_timestamps(jsonl_path: Path) -> list:
    """从 session JSONL 提取全部有效时间戳（毫秒）。"""
    try:
        with open(jsonl_path, "r", encoding="utf-8") as f:
            return [t for t in (_ts_from_jsonl_line(line) for line in f) if t is not None]
    except FileNotFoundError:
        return []


def get_session_time_range(jsonl_path: Path) -> tuple[float | None, float | None]:
    """Return (start_ms, end_ms) timestamps from a session JSONL file."""
    timestamps = _iter_session_timestamps(jsonl_path)
    if timestamps:
        return min(timestamps), max(timestamps)
    return None, None


def _get_dir_times(path: Path) -> tuple[float, float]:
    """返回 (birth_or_mtime, mtime)。"""
    birth = _get_dir_birthtime(path)
    mtime = os.stat(path).st_mtime
    return birth if birth is not None else mtime, mtime


def _collect_variant_timing(rd: Path) -> list[dict]:
    """收集单个 round 目录下所有 parallel 变体的时间信息。"""
    variants = []
    parallel_dirs = sorted(
        [p for p in rd.iterdir() if p.is_dir() and p.name.startswith("parallel_")],
        key=lambda p: int(p.name.split("_")[1]),
    )
    for pd in parallel_dirs:
        par_num = int(pd.name.split("_")[1])
        # Use evaluation_results.json timestamp for end time (when eval finished)
        eval_json = pd / "evaluation_results.json"
        if eval_json.exists():
            var_end = eval_json.stat().st_mtime
        else:
            _, var_end = _get_dir_times(pd)
        # Try to get a more accurate build-end time from the evolved directory
        evolved_dir = pd / "evolved"
        if evolved_dir.exists():
            build_end = evolved_dir.stat().st_mtime
        else:
            build_end = var_end
        variants.append({
            "parallel": par_num,
            "build_end": build_end,
            "end": var_end,
            "duration_minutes": max(0, (var_end - build_end)) / 60,
        })
    return variants


def _calc_round_start(rd: Path, variants: list[dict]) -> float:
    """取最早变体目录的 birth 时间作为 round 开始时间。

    Birth time 反映目录实际创建时刻（subagent 启动），
    ctime 会随目录内文件写入更新，不可用。
    """
    variant_starts = []
    for v in variants:
        pd = rd / f"parallel_{v['parallel']}"
        if pd.exists():
            birth = _get_dir_birthtime(pd)
            if birth is not None:
                variant_starts.append(birth)
    if variant_starts:
        return min(variant_starts)
    return rd.stat().st_ctime


def _fix_precreated_rounds(rounds: dict):
    """修正预创建的 round 目录时间。

    若某轮开始时间早于或过接近上一轮结束时间，视为预创建，
    将其开始时间顺移到上一轮结束之后。
    """
    sorted_round_nums = sorted(rounds.keys())
    for i in range(1, len(sorted_round_nums)):
        prev_rn = sorted_round_nums[i - 1]
        curr_rn = sorted_round_nums[i]
        prev_end = rounds[prev_rn]["end"]
        curr_start = rounds[curr_rn]["start"]
        # If current round appears to start before previous round ended,
        # or if multiple rounds were created within a very short window (< 2 min),
        # treat as pre-created and shift start time.
        if curr_start is not None and prev_end is not None and curr_start < prev_end + 120:
            rounds[curr_rn]["start"] = prev_end + 60
            # Recalculate duration, ensure non-negative
            new_duration = max(0, rounds[curr_rn]["end"] - rounds[curr_rn]["start"])
            rounds[curr_rn]["duration_minutes"] = new_duration / 60


def parse_round_timing(output_dir: str) -> dict[str, Any]:
    """Parse round directories to extract timing information.

    Returns:
        Dictionary with round timing details
    """
    round_timing = {
        "baseline": {"start": None, "end": None, "duration_minutes": 0},
        "rounds": {},
        "total_evolution_minutes": 0,
    }

    # Baseline timing
    baseline_dir = Path(output_dir) / "baseline"
    if baseline_dir.exists():
        create_time, modify_time = _get_dir_times(baseline_dir)
        round_timing["baseline"]["start"] = create_time
        round_timing["baseline"]["end"] = modify_time
        round_timing["baseline"]["duration_minutes"] = max(0, (modify_time - create_time)) / 60

    # Round timing - use variant directory birthtimes for accurate round starts
    round_dirs = sorted(
        [d for d in Path(output_dir).iterdir() if d.is_dir() and d.name.startswith("round_")],
        key=lambda d: int(d.name.split("_")[1]),
    )

    for rd in round_dirs:
        round_num = int(rd.name.split("_")[1])
        variants = _collect_variant_timing(rd)

        if variants:
            round_end = max(v["end"] for v in variants)
            round_start = _calc_round_start(rd, variants)
            round_timing["rounds"][round_num] = {
                "start": round_start,
                "end": round_end,
                "duration_minutes": max(0, (round_end - round_start)) / 60,
                "variants": variants,
            }
        else:
            # Fallback to directory times
            round_create, round_modify = _get_dir_times(rd)
            round_timing["rounds"][round_num] = {
                "start": round_create,
                "end": round_modify,
                "duration_minutes": max(0, (round_modify - round_create)) / 60,
                "variants": [],
            }

    _fix_precreated_rounds(round_timing["rounds"])

    # Calculate total evolution time
    if round_timing["rounds"]:
        first_round_start = min(r["start"] for r in round_timing["rounds"].values())
        last_round_end = max(r["end"] for r in round_timing["rounds"].values())
        round_timing["total_evolution_minutes"] = max(0, (last_round_end - first_round_start)) / 60

    return round_timing


def _format_duration(minutes: float) -> str:
    """格式化分钟数为可读时长。"""
    if minutes < 60:
        return f"{minutes:.1f} 分钟"
    total_mins = round(minutes)
    hours = total_mins // 60
    mins = total_mins % 60
    if mins == 0:
        return f"{hours} 小时"
    return f"{hours} 小时 {mins} 分钟"


def _format_tokens(count: int) -> str:
    """格式化 token 数为 K/M 缩写。"""
    if count >= 1_000_000:
        return f"{count / 1_000_000:.2f}M"
    if count >= 1_000:
        return f"{count / 1_000:.1f}K"
    return str(count)


def _format_time_range(start_ms: float | None, end_ms: float | None) -> str:
    """格式化时间范围（处理跨天 session）。"""
    if not start_ms or not end_ms:
        return "- ~ -"
    start_dt = datetime.fromtimestamp(start_ms / 1000)
    end_dt = datetime.fromtimestamp(end_ms / 1000)
    if start_dt.date() == end_dt.date():
        # Same day: HH:MM:SS ~ HH:MM:SS
        return f"{start_dt.strftime('%H:%M:%S')} ~ {end_dt.strftime('%H:%M:%S')}"
    # Different days: MM-DD HH:MM ~ MM-DD HH:MM
    return f"{start_dt.strftime('%m-%d %H:%M')} ~ {end_dt.strftime('%m-%d %H:%M')}"


def _token_sum(t: dict) -> int:
    return t.get("input", 0) + t.get("output", 0) + t.get("cache_read", 0) + t.get("cache_creation", 0)


def _build_timing_rows(timing: dict, round_timing: dict, pipeline_type: str) -> str:
    """构建耗时统计表格行。"""
    timing_rows = []

    # Main session
    main_timing = timing.get("main_session", {})
    if main_timing.get("duration_minutes", 0) > 0:
        time_range = _format_time_range(main_timing.get("start"), main_timing.get("end"))
        timing_rows.append(
            f"<tr><td>主会话</td><td>{time_range}</td>"
            f"<td>{_format_duration(main_timing['duration_minutes'])}</td><td>串行</td></tr>"
        )

    # Evo agent
    evo_timing = timing.get("evo_agent", {})
    if evo_timing.get("duration_minutes", 0) > 0:
        time_range = _format_time_range(evo_timing.get("start"), evo_timing.get("end"))
        pipeline_label = "ops-evo" if pipeline_type == "ops-evo" else "lingxi-evo"
        timing_rows.append(
            f"<tr><td>{pipeline_label} 代理</td><td>{time_range}</td>"
            f"<td>{_format_duration(evo_timing['duration_minutes'])}</td><td>主控</td></tr>"
        )

    # Rounds
    for round_num, round_data in sorted(round_timing.get("rounds", {}).items()):
        start_str = datetime.fromtimestamp(round_data["start"]).strftime("%H:%M:%S") if round_data.get("start") else "-"
        end_str = datetime.fromtimestamp(round_data["end"]).strftime("%H:%M:%S") if round_data.get("end") else "-"
        variant_count = len(round_data.get("variants", []))
        timing_rows.append(
            f"<tr><td>轮次 {round_num}</td><td>{start_str} ~ {end_str}</td>"
            f"<td>{_format_duration(round_data['duration_minutes'])}</td>"
            f"<td>{variant_count} 变体并行</td></tr>"
        )

    return "\n".join(timing_rows) if timing_rows else "<tr><td colspan='4'>无时间数据</td></tr>"


def _build_one_token_row(name: str, token_dict: dict) -> str:
    """构建单行词元统计行。"""
    return (
        f"<tr><td>{name}</td>"
        f"<td>{_format_tokens(token_dict.get('input', 0))}</td>"
        f"<td>{_format_tokens(token_dict.get('output', 0))}</td>"
        f"<td>{_format_tokens(token_dict.get('cache_read', 0))}</td>"
        f"<td>{_format_tokens(token_dict.get('cache_creation', 0))}</td>"
        f"<td>{_format_tokens(_token_sum(token_dict))}</td></tr>"
    )


def _build_token_rows(tokens: dict, pipeline_type: str) -> str:
    """构建词元统计表格行。"""
    token_rows = []

    # Main session
    main_tokens = tokens.get("main_session", {})
    if main_tokens.get("input", 0) > 0 or main_tokens.get("output", 0) > 0:
        token_rows.append(_build_one_token_row("主会话", main_tokens))

    # Evo agent
    evo_tokens = tokens.get("evo_agent", {})
    if evo_tokens.get("input", 0) > 0 or evo_tokens.get("output", 0) > 0:
        pipeline_label = "ops-evo" if pipeline_type == "ops-evo" else "lingxi-evo"
        token_rows.append(_build_one_token_row(f"{pipeline_label} 代理", evo_tokens))

    # Aside agents
    for aside in tokens.get("aside_agents", []):
        if aside.get("input", 0) > 0 or aside.get("output", 0) > 0:
            token_rows.append(_build_one_token_row(f"{aside['name'][:20]}...", aside))

    return "\n".join(token_rows) if token_rows else "<tr><td colspan='6'>无 词元 数据</td></tr>"


def build_resource_stats_section(session_stats: dict, round_timing: dict, pipeline_type: str = "ops-evo") -> str:
    """Build HTML section for resource statistics (timing and tokens)."""
    timing = session_stats.get("timing", {})
    tokens = session_stats.get("tokens", {})

    timing_table = _build_timing_rows(timing, round_timing, pipeline_type)
    token_table = _build_token_rows(tokens, pipeline_type)

    # Calculate percentages
    total_tokens = tokens.get("total", {})
    total_all = _token_sum(total_tokens)
    cache_pct = (total_tokens.get("cache_read", 0) / total_all * 100) if total_all > 0 else 0
    cache_creation_pct = (total_tokens.get("cache_creation", 0) / total_all * 100) if total_all > 0 else 0
    input_pct = (total_tokens.get("input", 0) / total_all * 100) if total_all > 0 else 0
    output_pct = (total_tokens.get("output", 0) / total_all * 100) if total_all > 0 else 0

    html = f"""
<h2>资源消耗统计</h2>
<div class="two-col">
<div class="card">
<h3 style="margin-top:0">耗时统计</h3>
<table>
<tr><th>阶段</th><th>时间范围</th><th>持续时间</th><th>执行方式</th></tr>
{timing_table}
<tr style="font-weight:600;border-top:2px solid var(--border);">
<td>总计</td><td>-</td>
<td>{_format_duration(timing.get('total_duration_minutes', 0))}</td><td>-</td>
</tr>
</table>
</div>
<div class="card">
<h3 style="margin-top:0">词元用量统计</h3>
<table>
<tr><th>组件</th><th>Input</th><th>Output</th><th>Cache Read</th><th>Cache Creation</th><th>总计</th></tr>
{token_table}
<tr style="font-weight:600;border-top:2px solid var(--border);">
<td>总计</td>
<td>{_format_tokens(total_tokens.get('input', 0))}</td>
<td>{_format_tokens(total_tokens.get('output', 0))}</td>
<td>{_format_tokens(total_tokens.get('cache_read', 0))}</td>
<td>{_format_tokens(total_tokens.get('cache_creation', 0))}</td>
<td>{_format_tokens(total_all)}</td>
</tr>
</table>
<div style="margin-top:1rem;font-size:0.85rem;color:var(--text-muted);">
<strong>词元分布:</strong> Cache Read {cache_pct:.1f}% | Cache Creation {cache_creation_pct:.1f}%
| Input {input_pct:.1f}% | Output {output_pct:.1f}%
</div>
</div>
</div>
"""
    return html
