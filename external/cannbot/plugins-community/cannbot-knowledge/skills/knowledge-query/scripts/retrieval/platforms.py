# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

"""Narrow platform policy for A3 retrieval.

The first-stage policy intentionally excludes only cards whose structured
``platforms`` value is exactly ``[950]``. Missing and multi-platform metadata
remain eligible and keep their existing scores.
"""
from __future__ import annotations

import re


PLATFORM_FILTER_RULE = "exclude_exact_platforms_950_only"

_A3_RE = re.compile(
    r"(?<![0-9A-Za-z_])(?:910\s*c|a3)(?![0-9A-Za-z_])|"
    r"(?<![0-9A-Za-z_])atlas\s*a3(?![0-9A-Za-z_])",
    re.I,
)
_ASCEND_950_RE = re.compile(
    r"(?<![0-9A-Za-z_])(?:950(?:pr|dt)?|a5|arch\s*35)(?![0-9A-Za-z_])",
    re.I,
)


def _normalized_explicit(value):
    value = str(value or "").strip()
    return "a3" if _A3_RE.search(value) else None


def platform_context(task="", explicit=None):
    """Return the platform-filter context for a task or explicit CLI target."""
    if explicit is not None:
        target = _normalized_explicit(explicit)
        return {
            "enabled": target == "a3",
            "target": target,
            "source": "explicit",
            "reason": "explicit_target" if target else "unsupported_explicit_target",
            "rule": PLATFORM_FILTER_RULE,
        }

    task = str(task or "")
    has_a3 = _A3_RE.search(task) is not None
    has_950 = _ASCEND_950_RE.search(task) is not None
    if has_a3 and has_950:
        return {
            "enabled": False,
            "target": None,
            "source": "task",
            "reason": "multi_platform_task",
            "rule": PLATFORM_FILTER_RULE,
        }
    if has_a3:
        return {
            "enabled": True,
            "target": "a3",
            "source": "task",
            "reason": "a3_detected",
            "rule": PLATFORM_FILTER_RULE,
        }
    return {
        "enabled": False,
        "target": None,
        "source": "task",
        "reason": "no_a3_target",
        "rule": PLATFORM_FILTER_RULE,
    }


def is_950_only(doc):
    """True only for a well-formed, single-value ``platforms: [950]`` list."""
    platforms = doc.get("platforms") if isinstance(doc, dict) else None
    if not isinstance(platforms, list) or len(platforms) != 1:
        return False
    return str(platforms[0]).strip().lower() == "950"


def filter_candidate_ids(idx, candidates, context):
    """Remove strict 950-only doc ids in A3 mode, preserving the input sentinel."""
    if not context or not context.get("enabled") or context.get("target") != "a3":
        return candidates
    doc_ids = candidates if candidates is not None else range(len(idx.get("docs", [])))
    return {doc_id for doc_id in doc_ids if not is_950_only(idx["docs"][doc_id])}


def platform_filter_output(context, result_count=None):
    """Return compact, stable metadata for platform-aware command output."""
    context = context or platform_context()
    output = {
        "enabled": bool(context.get("enabled")),
        "target": context.get("target"),
        "source": context.get("source"),
        "reason": context.get("reason"),
        "rule": context.get("rule", PLATFORM_FILTER_RULE),
    }
    if output["enabled"] and result_count == 0:
        output["warning"] = "no_results_after_filter"
    return output
