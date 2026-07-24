# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

"""retrieval.eval - small deterministic retrieval regression runner."""
from __future__ import annotations

import json
import os

from retrieval.errors import CliError
from retrieval.index import load_index
from retrieval.output import emit_stdout
from retrieval.plan import preflight_task


def _skill_root():
    return os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))


def default_cases_path():
    return os.path.join(_skill_root(), "eval", "retrieval_cases.json")


def _load_cases(path):
    with open(path, encoding="utf-8") as f:
        obj = json.load(f)
    cases = obj.get("cases", obj)
    if not isinstance(cases, list):
        raise ValueError("eval cases must be a list or an object with a cases list")
    return cases


def _position(paths, expected):
    expected = set(expected or [])
    for idx, path in enumerate(paths):
        if path in expected:
            return idx
    return None


def _case_expected(case):
    expected = case.get("expected") or case.get("expected_any") or []
    if isinstance(expected, str):
        return [expected]
    return list(expected)


def _case_forbidden(case):
    forbidden = case.get("forbidden") or case.get("forbidden_any") or []
    if isinstance(forbidden, str):
        return [forbidden]
    return list(forbidden)


def _evaluate_case(case, idx, k, grep_k, max_queries):
    task = case["task"]
    expected = _case_expected(case)
    forbidden = _case_forbidden(case)
    out = preflight_task(task, idx, max_queries=max_queries, k=k, grep_k=grep_k)
    result_paths = [item["path"] for item in out.get("results", [])][:k]
    suggested_paths = out.get("suggested_get", [])
    position = _position(result_paths, expected)
    suggested_position = _position(suggested_paths, expected)
    result_hit = position is not None
    suggested_hit = suggested_position is not None
    selected_paths = dict.fromkeys(result_paths + suggested_paths)
    forbidden_hits = [path for path in selected_paths if path in set(forbidden)]
    platform_guard_pass = not forbidden_hits
    case_pass = (not expected or result_hit or suggested_hit) and platform_guard_pass
    row = {
        "id": case.get("id", task),
        "task": task,
        "expected": expected,
        "forbidden": forbidden,
        "intent": out.get("plan", {}).get("intent", {}),
        "result_hit": result_hit,
        "result_position": position,
        "suggested_hit": suggested_hit,
        "suggested_position": suggested_position,
        "forbidden_hits": forbidden_hits,
        "platform_guard_pass": platform_guard_pass,
        "case_pass": case_pass,
        "top_results": result_paths,
        "suggested_get": suggested_paths[:max(k * 2, k)],
    }
    stats = {
        "expected": bool(expected),
        "result_hit": result_hit,
        "suggested_hit": suggested_hit,
        "reciprocal_rank": 1.0 / (position + 1) if position is not None else 0.0,
        "platform_guard_pass": platform_guard_pass,
        "case_pass": case_pass,
    }
    return row, stats


def _rate(numerator, denominator):
    return round(numerator / denominator, 4) if denominator else 1.0


def evaluate(cases, idx, k=5, grep_k=3, max_queries=6):
    rows = []
    stats = []
    for case in cases:
        row, case_stats = _evaluate_case(case, idx, k, grep_k, max_queries)
        rows.append(row)
        stats.append(case_stats)
    expected_stats = [item for item in stats if item["expected"]]
    n = len(rows)
    return {
        "case_count": n,
        "k": k,
        "grep_k": grep_k,
        "recall_at_k": _rate(sum(item["result_hit"] for item in expected_stats), len(expected_stats)),
        "suggested_hit_rate": _rate(sum(item["suggested_hit"] for item in expected_stats), len(expected_stats)),
        "mrr": _rate(sum(item["reciprocal_rank"] for item in expected_stats), len(expected_stats)),
        "platform_guard_pass_rate": _rate(sum(item["platform_guard_pass"] for item in stats), n),
        "case_pass_rate": _rate(sum(item["case_pass"] for item in stats), n),
        "cases": rows,
    }


def _fails_threshold(result, fail_under):
    if fail_under is None:
        return False
    rates = (result["recall_at_k"], result["case_pass_rate"])
    return min(rates) < fail_under or result["platform_guard_pass_rate"] < 1.0


def cmd_eval(cases_path=None, k=5, grep_k=3, max_queries=6, fail_under=None):
    cases_path = os.path.abspath(cases_path or default_cases_path())
    cases = _load_cases(cases_path)
    idx = load_index()
    result = evaluate(cases, idx, k=k, grep_k=grep_k, max_queries=max_queries)
    result["cases_path"] = cases_path
    emit_stdout(json.dumps(result, ensure_ascii=False, indent=2))
    if _fails_threshold(result, fail_under):
        raise CliError(code=1)
