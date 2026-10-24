#!/usr/bin/env python3
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# Licensed under the CANN Open Software License Agreement Version 2.0.
"""Local configuration checks; no API calls, file writes, or model decisions."""

from pathlib import Path
from datetime import datetime, timezone
import math
import re
from urllib.parse import urlsplit

# Rules describe consumers' requirements, not runtime defaults (which stay in YAML).
SCHEMA = {
    "repo": "repo",
    "responsibility": {k: "strings" for k in ("handle", "list-only", "ignore")},
    "ignored_issue_ids": "ids",
    "auto-response": "boolean",
    "auto-assign": "boolean",
    "auto-assign-fallback-user": "login",
    "lookback_days": "positive_int",
    "gitcode_api": "url",
    "last_check_file": "path",
    "cache_dir": "path",
    "report_file": "path",
    "pr_fetch_pages": "positive_int",
    "pr_linkage_scan_mode": "scan_mode",
    "pr_linkage_api_budget": "nonnegative_int",
    "follow_up": {
        "lookback_days": "positive_int",
        "fetch_pages": "positive_int",
        "poll_hours": "positive_int",
        "stale_hours": "positive_int",
        "state_file": "path",
    },
    "auto_close": {
        "inactive_hours": "positive_number",
        "question_labels": "strings",
        "question_title_markers": "strings",
        "comment": "text",
        "pr_fetch_pages": "positive_int",
        "pr_linkage_api_budget": "nonnegative_int",
    },
}
DEPRECATED = {"follow_up.waiting_status", "follow_up.active_status"}


def _local_now():
    """Return local wall time without relying on a timezone-free now() call."""
    return datetime.now(timezone.utc).astimezone().replace(tzinfo=None)


def _number_rule(value, rule):
    is_int = value.__class__ is int
    if rule == "boolean":
        valid, hint = value.__class__ is bool, "必须是布尔值 true 或 false，不能加引号"
    elif rule in ("positive_int", "nonnegative_int"):
        valid = is_int and value >= (1 if rule == "positive_int" else 0)
        hint = "必须是正整数" if rule == "positive_int" else "必须是非负整数（0 表示无额外查询预算）"
    elif rule == "positive_number":
        valid = (value.__class__ is int or value.__class__ is float and math.isfinite(value)) and value > 0
        hint = "必须是有限的正数"
    return valid, hint


def _list_rule(value, rule):
    if rule in ("strings", "ids"):
        valid = isinstance(value, list) and all(
            (i.__class__ is int and i > 0) if rule == "ids" else isinstance(i, str) and bool(i.strip()) for i in value
        )
        hint = "必须是正整数列表，例如 [296, 301]" if rule == "ids" else "必须是非空字符串组成的列表；无条目时使用 []"
    return valid, hint


def _url_rule(value):
    valid = False
    if isinstance(value, str) and not re.search(r"\s", value):
        try:
            url = urlsplit(value)
            valid = (
                url.scheme in ("http", "https")
                and bool(url.hostname)
                and not (url.username or url.password or url.query or url.fragment)
            )
            _ = url.port
        except ValueError:
            valid = False
    return valid, "必须是有效的 HTTP(S) API 地址，不含凭据、查询参数或片段"


def _text_rule(value, rule):
    if rule == "repo":
        valid = isinstance(value, str) and (value == "" or bool(re.fullmatch(r"[\w.-]+/[\w.-]+", value)))
        hint = "必须是 owner/repo 格式（不要填 URL）；配置阶段允许空字符串"
    elif rule == "login":
        valid = isinstance(value, str) and (
            value == "" or bool(re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]*", value)) and value.casefold() != "direct"
        )
        hint = "必须是空字符串或 GitCode 账号，不填 URL、@前缀或 direct"
    elif rule == "scan_mode":
        valid, hint = isinstance(value, str) and value in ("ambiguous", "all"), "只能是 ambiguous 或 all"
    elif rule == "url":
        valid, hint = _url_rule(value)
    else:
        valid = isinstance(value, str) and bool(value.strip()) and "\x00" not in value
        hint = "必须是非空字符串且不含 NUL 字符"
        if rule == "path" and valid:
            valid = value == value.strip() and not value.startswith("~") and "$" not in value
            hint = "路径应为绝对路径或相对工作目录的路径；不要使用 ~、环境变量或首尾空格"
    return valid, hint


def _scalar_rule(value, rule):
    if rule in ("boolean", "positive_int", "nonnegative_int", "positive_number"):
        return _number_rule(value, rule)
    if rule in ("strings", "ids"):
        return _list_rule(value, rule)
    return _text_rule(value, rule)


def _scalar_problem(value, rule, field):
    valid, hint = _scalar_rule(value, rule)
    if valid and field.endswith("lookback_days"):
        valid = value <= (_local_now() - datetime.min).days
        hint = "回看天数超出日期可表示范围，会导致日期计算溢出"
    if valid and field.endswith(("poll_hours", "stale_hours", "inactive_hours")):
        valid = value < (datetime.max - _local_now()).total_seconds() / 3600
        hint = "小时数超出日期可表示范围，会导致日期计算溢出"
    return hint if not valid else None


def problems(document, *, complete=False):
    errors = []

    def error(field, message):
        errors.append({"field": field, "message": message})

    def check(value, rule, field):
        if isinstance(rule, dict):
            if not isinstance(value, dict):
                error(field, "必须是 YAML 映射，不能是 null、列表或字符串")
                return
            for key, item in value.items():
                path = f"{field}.{key}" if field else str(key)
                if path in DEPRECATED:
                    continue
                if key not in rule:
                    error(path, "未知配置项；请检查字段拼写或删除此项")
                else:
                    check(item, rule[key], path)
            # responsibility may intentionally be empty; consumers use .get().
            if complete and field in ("follow_up", "auto_close"):
                for key in rule.keys() - value.keys():
                    error(f"{field}.{key}", "缺少运行所需字段；删除空 {} 配置块可恢复模板默认值")
            return
        message = _scalar_problem(value, rule, field)
        if message:
            error(field, message)

    check(document, SCHEMA, "")
    return errors


def _checked_path(field, value, resolved, reserved):
    path = Path(value).resolve()
    expected_dir = field == "cache_dir"
    if path in reserved:
        raise ValueError("不能覆盖配置文件或流水线状态/锁文件")
    if path.exists() and (not path.is_dir() if expected_dir else not path.is_file()):
        raise ValueError("目标应为目录" if expected_dir else "目标应为普通文件")
    if any(parent.exists() and not parent.is_dir() for parent in path.parents):
        raise ValueError("路径的父级是文件，无法创建目标")
    for other, target in resolved.items():
        target_is_parent = other != "cache_dir" and target in path.parents
        path_is_parent = not expected_dir and path in target.parents
        if path == target or target_is_parent or path_is_parent:
            raise ValueError(f"与 {other} 路径冲突，会互相覆盖或阻止创建")
    return path


def path_problems(config, source):
    """Check known local collisions without creating files or probing credentials."""
    errors = []
    paths = {
        "last_check_file": config["last_check_file"],
        "report_file": config["report_file"],
        "cache_dir": config["cache_dir"],
        "follow_up.state_file": config["follow_up"]["state_file"],
    }
    resolved = {}
    reserved = {
        Path(source).resolve(),
        Path(".cannbot/gitcode-issue-handler/data/pipeline-state.json").resolve(),
        Path(".cannbot/gitcode-issue-handler/data/pipeline-state.previous.json").resolve(),
        Path(".cannbot/gitcode-issue-handler/data/pipeline-state.lock").resolve(),
    }
    for field, value in paths.items():
        try:
            resolved[field] = _checked_path(field, value, resolved, reserved)
        except (OSError, RuntimeError, ValueError) as exc:
            errors.append({"field": field, "message": str(exc)})
    return errors
