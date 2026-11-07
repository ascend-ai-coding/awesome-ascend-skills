# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

"""Deterministic relevance signals for retrieval outputs.

These scores are evidence-routing hints, not final truth. The Agent still reads
the card body and decides whether the evidence actually covers the task.
"""
from __future__ import annotations

from retrieval.tokenizer import tokenize


KIND_FIT = {
    "api_usage": {
        "api": 0.24,
        "example": 0.14,
        "guide": 0.08,
        "operator": 0.06,
        "glossary": -0.08,
        "index": -0.12,
    },
    "implementation": {
        "example": 0.22,
        "operator": 0.18,
        "api": 0.12,
        "guide": 0.10,
        "operator_optimization": 0.08,
        "implementation_trap": 0.06,
        "glossary": -0.10,
        "index": -0.10,
    },
    "similar_example": {
        "example": 0.28,
        "operator": 0.20,
        "guide": -0.04,
        "api": -0.04,
        "glossary": -0.12,
        "index": -0.12,
    },
    "debug": {
        "debugging_journey": 0.24,
        "implementation_trap": 0.22,
        "guide": 0.08,
        "example": 0.06,
        "api": 0.04,
        "glossary": -0.08,
        "index": -0.10,
    },
    "performance": {
        "operator_optimization": 0.28,
        "debugging_journey": 0.10,
        "guide": 0.10,
        "example": 0.08,
        "api": 0.04,
        "glossary": -0.08,
    },
    "overview": {
        "index": 0.24,
        "guide": 0.12,
        "operator_optimization": 0.08,
        "glossary": -0.08,
    },
    "migration": {
        "guide": 0.14,
        "api": 0.08,
        "implementation_trap": 0.08,
        "example": 0.04,
        "glossary": -0.06,
    },
    "validation": {
        "guide": 0.10,
        "example": 0.10,
        "debugging_journey": 0.08,
        "api": 0.06,
        "glossary": -0.06,
    },
    "concept": {
        "glossary": 0.26,
        "guide": 0.10,
        "index": 0.04,
        "api": -0.04,
        "example": -0.08,
    },
}

STRONG_FIELDS = {"title", "path", "tags", "names", "aliases", "signatures"}
MID_FIELDS = {"description", "headings", "body"}


def _dedupe(values):
    out = []
    seen = set()
    for value in values:
        if not value or value in seen:
            continue
        seen.add(value)
        out.append(value)
    return out


def _labels(intent):
    return set((intent or {}).get("labels", []) or [])


def _query_tokens(queries):
    toks = []
    for query in queries or []:
        toks.extend(tokenize(str(query)))
    return set(toks)


def _text_tokens(item, fields):
    parts = []
    for field in fields:
        value = item.get(field, "")
        if isinstance(value, list):
            value = " ".join(str(v) for v in value)
        parts.append(str(value))
    return set(tokenize(" ".join(parts)))


def _path_scope_bonus(item, labels, reasons):
    path = item.get("path", "")
    category = item.get("category", "")
    bonus = 0.0
    if "api_usage" in labels and path.startswith("asc-devkit/api/"):
        reasons.append("scope_fit:reference_api")
        bonus += 0.08
    implementation_path = any((path.startswith("ops/"), "examples/" in path, category.endswith("_api")))
    if labels & {"implementation", "similar_example"} and implementation_path:
        reasons.append("scope_fit:implementation_or_example")
        bonus += 0.08
    performance_path = any((path.startswith("runbooks/"), "operator_practice" in path, "profiling" in path))
    if "performance" in labels and performance_path:
        reasons.append("scope_fit:performance")
        bonus += 0.08
    if "debug" in labels and path.startswith("runbooks/field-notes/"):
        reasons.append("scope_fit:field_notes")
        bonus += 0.08
    return bonus


def _recommended_action(level, reasons, risks):
    if "generic_card_may_need_cross_check" in risks:
        return "expand"
    if level == "strong":
        return "read"
    if level == "medium" and "generic_card_may_need_cross_check" not in risks:
        return "read"
    if level == "weak":
        return "expand"
    if any(r.startswith("only_") for r in risks):
        return "expand"
    if any("grep" in r for r in reasons):
        return "read"
    return "expand"


def _matched_score(item, reasons):
    matched = item.get("matched", {}) or {}
    fields = set(matched.get("fields", []) or [])
    terms = set(matched.get("terms", []) or [])
    score = 0.0
    if fields & STRONG_FIELDS:
        reasons.append("exact_metadata_hit")
        score += 0.22
    if fields & MID_FIELDS:
        reasons.append("content_hit")
        score += 0.10
    return score + min(0.12, 0.03 * len(terms)) if terms else score


def _query_score(item, queries, reasons, risks):
    query_tokens = _query_tokens(queries)
    if not query_tokens:
        return 0.0
    title_hits = query_tokens & _text_tokens(item, ("title", "tags", "path"))
    body_hits = query_tokens & _text_tokens(item, ("description", "snippet"))
    score = 0.0
    if title_hits:
        reasons.append("title_alias_path_hit")
        score += min(0.20, 0.05 * len(title_hits))
    if body_hits:
        reasons.append("snippet_or_description_hit")
        score += min(0.12, 0.03 * len(body_hits))
    if not title_hits and not body_hits:
        risks.append("query_terms_not_visible_in_summary")
    return score


def _kind_score(kind, labels, reasons, risks):
    score = 0.0
    for label in labels:
        fit = KIND_FIT.get(label, {}).get(kind, 0.0)
        if fit > 0:
            reasons.append("kind_fit:%s" % kind)
        elif fit < 0:
            risks.append("kind_mismatch:%s_for_%s" % (kind, label))
        score += fit
    return score


def _source_score(source, reasons, risks):
    source_signals = {
        "precise_grep": ("precise_grep_hit", 0.16, None),
        "exact": ("exact_query_hit", 0.10, None),
        "broad_grep": ("broad_grep_hit", 0.05, "broad_grep_needs_context_check"),
        "hint": ("domain_hint_hit", 0.03, "hint_result_needs_confirmation"),
    }
    reason, score, risk = source_signals.get(source, (None, 0.0, None))
    if reason:
        reasons.append(reason)
    if risk:
        risks.append(risk)
    return score


def _relevance_level(score):
    if score >= 0.72:
        return "strong"
    if score >= 0.45:
        return "medium"
    return "weak"


def evaluate_relevance(item, intent=None, queries=None):
    labels = _labels(intent)
    reasons = []
    risks = []
    score = 0.18
    score += _matched_score(item, reasons)
    score += _query_score(item, queries, reasons, risks)
    kind = item.get("kind", "")
    score += _kind_score(kind, labels, reasons, risks)
    score += _path_scope_bonus(item, labels, reasons)
    score += _source_score(item.get("source"), reasons, risks)
    if labels and kind in {"glossary", "index"} and not (labels & {"concept", "overview"}):
        risks.append("generic_card_may_need_cross_check")
        score -= 0.04
    score = max(0.0, min(1.0, score))
    level = _relevance_level(score)
    return {
        "score": round(score, 3),
        "level": level,
        "reasons": _dedupe(reasons),
        "risks": _dedupe(risks),
        "recommended_action": _recommended_action(level, reasons, risks),
    }


def annotate_relevance(items, intent=None, queries=None):
    for item in items or []:
        if isinstance(item, dict):
            item["relevance"] = evaluate_relevance(item, intent=intent, queries=queries)
    return items


def relevance_bonus(item):
    relevance = item.get("relevance") or {}
    score = float(relevance.get("score") or 0.0)
    return round((score - 0.45) * 3.0, 4)


def read_purpose(item, intent=None):
    labels = _labels(intent)
    kind = item.get("kind", "")
    title = item.get("title") or item.get("path", "")
    if "api_usage" in labels and kind == "api":
        return "确认 %s 的接口语义、参数约束、支持范围和 resource 来源" % title
    if labels & {"implementation", "similar_example"} and kind in {"example", "operator"}:
        return "复用 %s 的实现模式、Tiling/数据搬运写法或相似样例证据" % title
    if "debug" in labels:
        return "核对 %s 是否覆盖报错关键字、触发条件、根因和修复路径" % title
    if "performance" in labels:
        return "核对 %s 是否提供性能瓶颈、profiling 现象或优化措施" % title
    if "overview" in labels:
        return "建立 %s 所在目录或主题的知识范围，再选择具体卡片深入" % title
    return "核对 %s 是否真正覆盖用户问题，而不是只做关键词命中" % title


def verify_after_read(item, intent=None, missing_signals=None):
    labels = _labels(intent)
    checks = [
        "正文是否直接覆盖用户问题，不只是标题或关键词命中",
        "结论是否能保留 doc-id/resource 作为依据",
    ]
    if labels & {"api_usage", "implementation"}:
        checks.append("是否说明 API/接口约束、参数含义、dtype/shape/format 或内存范围")
    if labels & {"implementation", "similar_example"}:
        checks.append("是否包含可复用实现路径、相似样例、Tiling 或数据搬运写法")
    if "debug" in labels:
        checks.append("是否覆盖错误码/报错片段、触发条件、根因和修复动作")
    if "performance" in labels:
        checks.append("是否给出 profiling 指标、性能瓶颈或优化前后依据")
    if missing_signals:
        checks.append("是否仍缺少这些关键信号：%s" % "；".join(missing_signals[:3]))
    relevance = item.get("relevance") or {}
    if relevance.get("level") == "weak":
        checks.append("该卡 relevance=weak，读取后若无直接证据应标记为弱证据并继续扩展")
    return checks
