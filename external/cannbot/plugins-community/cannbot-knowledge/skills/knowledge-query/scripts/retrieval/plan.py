# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

"""retrieval.plan - deterministic query planning for knowledge-query preflight.

The planner turns a task/error/debug sentence into short retrieval probes. It is
intentionally stdlib-only and conservative: exact symbols and error codes win;
knowledge base index terms and configurable knowledge-need hints provide compact context.
"""
from __future__ import annotations

import json
import os
import re
import shlex

from retrieval.grep import grep_results
from retrieval.paths import attach_local_paths, local_path_for_doc
from retrieval.platforms import platform_context, platform_filter_output
from retrieval.relevance import (annotate_relevance, evaluate_relevance, read_purpose,
                                 relevance_bonus, verify_after_read)
from retrieval.search import query_warnings, search_results
from retrieval.config import ROOT


ERROR_RE = re.compile(r"\b[1-9]\d{4,8}\b")
BACKTICK_RE = re.compile(r"`([^`\n]{2,80})`")
IDENT_RE = re.compile(r"\b[A-Za-z_][A-Za-z0-9_]*\b")
ACL_MACRO_RE = re.compile(r"\b[A-Z][A-Z0-9_]{3,}\b")

LOW_VALUE_WORDS = {
    "a", "an", "and", "api", "bug", "case", "code", "custom", "debug",
    "developer", "doc", "error", "failed", "failure", "fix", "for", "grep",
    "host", "impl", "implementation", "issue", "kernel", "log", "op",
    "operator", "project", "query", "run", "script", "search", "session",
    "task", "test", "the", "trace", "workflow", "design", "get", "tiling",
    "soc", "ascend",
}
LOW_VALUE_METADATA = {
    "api", "guide", "example", "index", "other", "reference", "doc", "docs",
    "task", "workflow", "project", "design", "implementation", "host",
}

INTENT_PATTERNS = [
    (
        "overview",
        r"(overview|summary|\u603b\u89c8|\u6c47\u603b)",
    ),
    (
        "similar_example",
        r"(similar example|sample|example|\u76f8\u4f3c\u6837\u4f8b|\u6837\u4f8b|\u627e\u4e00\u4e2a)",
    ),
    (
        "concept",
        r"(what is|concept|definition|\u662f\u4ec0\u4e48|\u6982\u5ff5|\u5b9a\u4e49|\u542b\u4e49|\u8bed\u4e49)",
    ),
    (
        "api_usage",
        r"(api|interface|usage|call|signature|parameter|param|constraint|"
        r"\u63a5\u53e3|\u51fd\u6570|\u53c2\u6570|\u8c03\u7528|\u4f7f\u7528|"
        r"\u600e\u4e48\u5199|\u600e\u4e48\u7528|\u5982\u4f55)",
    ),
    (
        "implementation",
        r"(implement|implementation|example|sample|coding|code|tiling|kernel|operator|"
        r"\u5b9e\u73b0|\u5f00\u53d1|\u4ee3\u7801|\u6837\u4f8b|\u7b97\u5b50|"
        r"\u5185\u6838|\u5207\u5206|\u591a\u6838)",
    ),
    (
        "debug",
        r"(error|exception|fail|failed|failure|crash|debug|build|compile|runtime|log|trace|"
        r"\u9519\u8bef|\u62a5\u9519|\u5f02\u5e38|\u5931\u8d25|\u5d29\u6e83|"
        r"\u8c03\u8bd5|\u8bca\u65ad|\u65e5\u5fd7|\u6784\u5efa|\u7f16\u8bd1|\u8fd0\u884c)",
    ),
    (
        "performance",
        r"(performance|optimize|optimization|profiling|profile|latency|throughput|bandwidth|"
        r"\u6027\u80fd|\u4f18\u5316|\u8c03\u4f18|\u8017\u65f6|\u541e\u5410|\u5e26\u5bbd)",
    ),
    (
        "validation",
        r"(test|verify|validation|acceptance|accuracy|precision|coverage|regression|"
        r"\u6d4b\u8bd5|\u9a8c\u8bc1|\u9a8c\u6536|\u7cbe\u5ea6|\u8986\u76d6|\u56de\u5f52)",
    ),
    (
        "migration",
        r"(migration|migrate|version|compatibility|upgrade|porting|platform|hardware|"
        r"\u8fc1\u79fb|\u7248\u672c|\u517c\u5bb9|\u5347\u7ea7|\u9002\u914d|\u5e73\u53f0|\u786c\u4ef6)",
    ),
]

SIGNAL_INTENTS = {
    "api_semantics": "api_usage",
    "design_implementation": "implementation",
    "parallel_partitioning": "implementation",
    "debugging": "debug",
    "performance": "performance",
    "testing_validation": "validation",
    "migration_compatibility": "migration",
}

QUERY_ROUTE_TRIGGERS = re.compile(
    r"(overview|summary|catalog|list|browse|all|scope|map|"
    r"\u603b\u89c8|\u6c47\u603b|\u76ee\u5f55|\u5217\u8868|\u6709\u54ea\u4e9b|"
    r"\u6574\u4f53|\u8303\u56f4|\u4f53\u7cfb|\u8def\u7ebf|\u5206\u7c7b)"
)

SOC_RE = re.compile(
    r"\b(ascend\s*)?(310p?|610|710|910[abc]?|920|930|950)\b|"
    r"\bAtlas\b|\u6607\u817e\s*(310p?|610|710|910[abc]?|920|930|950)|"
    r"(?<![0-9A-Za-z_])a3(?![0-9A-Za-z_])",
    re.I,
)
DTYPE_RE = re.compile(
    r"\b(dtype|data\s*type|float16|float32|float|half|fp16|fp32|bf16|"
    r"int8|int16|int32|int64|uint8|uint16|uint32|uint64|bool)\b|"
    r"\u6570\u636e\u7c7b\u578b|\u534a\u7cbe\u5ea6|\u5355\u7cbe\u5ea6",
    re.I,
)
SHAPE_RE = re.compile(
    r"\b(shape|rank|dim|dims?|axis|axes|size|sizes?|[0-9]+\s*[xX*]\s*[0-9]+)\b|"
    r"\u5f62\u72b6|\u7ef4\u5ea6|\u8f74|\u8f93\u5165\u89c4\u6a21|\u5c3a\u5bf8",
    re.I,
)
OP_RE = re.compile(r"\b(op|operator|kernel|算子|内核)\b", re.I)

INTENT_KIND_BOOSTS = {
    "concept": {
        "glossary": 2.4,
        "guide": 0.7,
        "index": -1.0,
        "api": -0.5,
        "example": -0.8,
    },
    "api_usage": {
        "api": 2.2,
        "example": 1.8,
        "guide": 1.0,
        "operator": 0.4,
        "glossary": -2.0,
        "index": -2.0,
    },
    "similar_example": {
        "example": 3.4,
        "operator": 1.0,
        "guide": -0.3,
        "api": -0.6,
        "glossary": -1.4,
        "index": -2.0,
    },
    "overview": {
        "guide": 1.2,
        "operator_optimization": 1.2,
        "index": 0.5,
        "api": -0.5,
        "glossary": -1.0,
    },
    "implementation": {
        "example": 2.0,
        "guide": 1.6,
        "operator": 1.0,
        "api": 0.9,
        "implementation_trap": 0.7,
        "glossary": -1.2,
        "index": -2.0,
    },
    "debug": {
        "implementation_trap": 2.2,
        "debugging_journey": 2.0,
        "cross_skill_gap": 1.2,
        "guide": 0.8,
        "api": 0.4,
        "index": -2.0,
    },
    "performance": {
        "operator_optimization": 2.3,
        "guide": 1.5,
        "example": 0.9,
        "operator": 0.6,
        "api": 0.3,
        "index": -2.0,
    },
    "validation": {
        "guide": 1.2,
        "example": 1.0,
        "implementation_trap": 0.8,
        "api": 0.3,
        "index": -1.5,
    },
    "migration": {
        "guide": 1.5,
        "api": 0.8,
        "implementation_trap": 0.5,
        "index": -1.5,
    },
}

INTENT_BUNDLE_BOOSTS = {
    "debug": {"runbooks": 1.0},
    "performance": {"runbooks": 0.9, "ascend-c-profiling": 0.9},
    "implementation": {"ops": 0.6},
}


def _dedupe(items):
    seen = set()
    out = []
    for item in items:
        value = " ".join(str(item).strip().split())
        if not value:
            continue
        key = value.lower()
        if key in seen:
            continue
        seen.add(key)
        out.append(value)
    return out


def _skill_root():
    return os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))


def _hint_candidates():
    env_path = os.environ.get("CANNBOT_KNOWLEDGE_QUERY_HINTS_PATH")
    if env_path:
        yield os.path.abspath(env_path)
    yield os.path.join(ROOT, ".cannbot", "knowledge-query-hints.json")
    yield os.path.join(ROOT, "search", "knowledge-query-hints.json")
    yield os.path.join(_skill_root(), "domain_hints.json")


def _load_hint_config():
    warnings = []
    for path in _hint_candidates():
        if not path or not os.path.exists(path):
            continue
        try:
            with open(path, encoding="utf-8") as f:
                data = json.load(f)
            triggers = data.get("triggers", [])
            if not isinstance(triggers, list):
                warnings.append("hint file ignored, triggers is not a list: %s" % path)
                continue
            return {"source": path, "triggers": triggers, "warnings": warnings}
        except (OSError, ValueError) as e:
            warnings.append("hint file ignored: %s (%s)" % (path, e))
    return {"source": None, "triggers": [], "warnings": warnings}


def _matches_hint(task, hint):
    low = task.lower()
    for item in hint.get("if_any", []) or []:
        value = str(item).strip()
        if value and value.lower() in low:
            return True
    for pattern in hint.get("if_regex", []) or []:
        try:
            if re.search(str(pattern), task, re.I):
                return True
        except re.error:
            continue
    return False


def _safe_priority(value, default=60):
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _is_good_grep_term(value):
    value = str(value).strip()
    if ERROR_RE.fullmatch(value):
        return True
    if _is_symbol(value):
        return True
    if not value.isascii() and len(value) >= 3:
        return True
    return False


def _is_precise_grep_term(value):
    value = str(value).strip()
    return bool(ERROR_RE.fullmatch(value) or (value.isascii() and _is_symbol(value)))


def _is_symbol(word):
    if len(word) < 3:
        return False
    low = word.lower()
    if low in LOW_VALUE_WORDS:
        return False
    if "_" in word:
        return any(ch.isalpha() for ch in word)
    if any(ch.isdigit() for ch in word) and any(ch.isalpha() for ch in word):
        return True
    has_upper = any(ch.isupper() for ch in word)
    has_lower = any(ch.islower() for ch in word)
    if has_upper and has_lower:
        return True
    return word.isupper() and len(word) > 4


def _index_terms(doc):
    primary = [doc.get("title", ""), doc.get("base", "")]
    primary.extend(doc.get("aliases", []) or [])
    metadata = []
    metadata.extend(doc.get("tags", []) or [])
    metadata.extend(doc.get("paradigms", []) or [])
    metadata.extend([doc.get("kind", ""), doc.get("category", ""), doc.get("section", "")])
    return [(value, False) for value in primary] + [(value, True) for value in metadata]


def _index_term_matches(task_lower, raw_value, is_metadata):
    value = str(raw_value).strip()
    value_lower = value.lower()
    ignored_values = LOW_VALUE_METADATA if is_metadata else LOW_VALUE_WORDS
    if len(value) < 3 or value_lower in ignored_values:
        return False
    if not value.isascii():
        return value_lower in task_lower
    pattern = r"(?<![0-9a-z_])%s(?![0-9a-z_])" % re.escape(value_lower)
    return re.search(pattern, task_lower) is not None


def _index_exact_terms(task, idx):
    lower = task.lower()
    terms = []
    for doc in idx.get("docs", []):
        for value, is_metadata in _index_terms(doc):
            if _index_term_matches(lower, value, is_metadata):
                terms.append(str(value).strip())
    return _dedupe(terms)


def _hint_phrases(task):
    cfg = _load_hint_config()
    phrases = []
    grep_terms = []
    signals = []
    for hint in cfg["triggers"]:
        if not isinstance(hint, dict) or not _matches_hint(task, hint):
            continue
        name = str(hint.get("name") or "domain_hint")
        priority = _safe_priority(hint.get("priority", 60), 60)
        signals.append({"name": name, "priority": priority})
        for query in hint.get("queries", []) or []:
            phrases.append({"query": str(query), "reason": "hint:%s" % name, "priority": priority})
        for pattern in hint.get("grep", []) or []:
            grep_terms.append(str(pattern))
    return {
        "phrases": phrases,
        "grep": _dedupe(grep_terms),
        "signals": signals,
        "source": cfg["source"],
        "warnings": cfg["warnings"],
    }


def _detect_intent(task, signals):
    labels = []
    reasons = []
    for label, pattern in INTENT_PATTERNS:
        if re.search(pattern, task, re.I):
            labels.append(label)
            reasons.append("keyword:%s" % label)
    for signal in signals or []:
        label = SIGNAL_INTENTS.get(signal.get("name"))
        if label:
            labels.append(label)
            reasons.append("hint:%s" % signal.get("name"))
    labels = _dedupe(labels)
    # Usage questions are commonly implementation questions too; keep both so
    # examples and guides can outrank glossary cards for "how to" tasks.
    if ("api_usage" in labels or "similar_example" in labels) and "implementation" not in labels:
        labels.append("implementation")
    return {"labels": labels, "reasons": _dedupe(reasons)}


def _intent_signal_boost(labels, item):
    category = item.get("category", "")
    path = item.get("path", "")
    tags = set(item.get("tags", []) or [])
    boost = 0.0
    implementation_scope = "examples/" in path or category.endswith("_api")
    if any(label in labels for label in ("api_usage", "implementation")) and implementation_scope:
        boost += 0.4
    if "implementation" in labels and category in ("operator_practice", "programming_guide"):
        boost += 0.4
    if "performance" in labels and ({"performance", "optimization", "profiling"} & tags):
        boost += 0.5
    if "debug" in labels and ({"debug", "trap", "known_issue"} & tags):
        boost += 0.4
    if "similar_example" in labels and ("examples/" in path or category == "simd_cpp_api"):
        boost += 1.2
    return boost


def _overview_intent_boost(labels, item):
    if "overview" in labels:
        path = item.get("path", "")
        haystack = " ".join([item.get("title", ""), item.get("description", ""), path]).lower()
        if "overview" in haystack or "\u603b\u89c8" in haystack or "\u6c47\u603b" in haystack:
            return 3.2
    return 0.0


def _item_intent_boost(intent, item):
    labels = intent.get("labels", []) if intent else []
    if not labels:
        return 0.0
    kind = item.get("kind", "")
    bundle = item.get("bundle", "")
    base_boost = sum(
        INTENT_KIND_BOOSTS.get(label, {}).get(kind, 0.0)
        + INTENT_BUNDLE_BOOSTS.get(label, {}).get(bundle, 0.0)
        for label in labels
    )
    return base_boost + _intent_signal_boost(labels, item) + _overview_intent_boost(labels, item)


def _rerank_for_intent(results, intent, queries=None):
    if not results:
        return results
    reranked = []
    for rank, item in enumerate(results):
        copy = dict(item)
        boost = _item_intent_boost(intent, copy)
        score = float(copy.get("score") or 0.0)
        annotate_relevance([copy], intent=intent, queries=queries)
        rel_boost = relevance_bonus(copy)
        copy["intent_boost"] = round(boost, 3)
        copy["relevance_boost"] = round(rel_boost, 3)
        copy["rank_score"] = round(score + boost + rel_boost, 4)
        copy["_original_rank"] = rank
        reranked.append(copy)
    reranked.sort(key=lambda item: (-item["rank_score"], item["_original_rank"], item["path"]))
    for item in reranked:
        item.pop("_original_rank", None)
    return reranked


def _intent_primary_queries(task, intent):
    labels = set(intent.get("labels", []) if intent else [])
    queries = []
    low = task.lower()
    if "overview" in labels:
        if "overview" in low or "summary" in low:
            queries.append("overview")
        if "\u603b\u89c8" in task or "\u6c47\u603b" in task:
            queries.append("\u603b\u89c8")
        if "\u4f18\u5316" in task or "optimization" in low or "performance" in low:
            queries.append("\u4f18\u5316\u5efa\u8bae")
    if "similar_example" in labels:
        if "tiling" in low or "\u5207\u5206" in task or "\u591a\u6838" in task:
            queries.append("tiling strategy")
        queries.append("example")
    return _dedupe(queries)


def _route_for_plan(task, intent, exact_queries):
    labels = set(intent.get("labels", []) if intent else [])
    if "overview" in labels or QUERY_ROUTE_TRIGGERS.search(task):
        return "browse_first"
    if exact_queries:
        return "search_first"
    if labels & {"similar_example", "implementation", "debug", "performance", "api_usage"}:
        return "search_first"
    return "browse_first"


def _missing_signals(task, intent, exact_queries, route=None):
    labels = set(intent.get("labels", []) if intent else [])
    missing = []
    open_browse = route == "browse_first" and ("overview" in labels or QUERY_ROUTE_TRIGGERS.search(task))
    if not exact_queries:
        missing.append("未识别到精确 API、宏、错误码、文件名或知识卡别名")
    if open_browse:
        return _dedupe(missing)
    if labels & {"implementation", "similar_example", "performance", "debug"}:
        if not OP_RE.search(task) and not exact_queries:
            missing.append("未识别到具体算子、kernel 或实现对象")
    if labels & {"implementation", "api_usage", "similar_example", "performance", "debug", "migration"}:
        if not SOC_RE.search(task):
            missing.append("未识别到芯片/SoC/硬件型号")
    if labels & {"implementation", "api_usage", "similar_example", "performance", "validation"}:
        if not DTYPE_RE.search(task):
            missing.append("未识别到 dtype/数据类型")
        if not SHAPE_RE.search(task):
            missing.append("未识别到 shape/维度/输入规模")
    if "debug" in labels and not (ERROR_RE.search(task) or BACKTICK_RE.search(task)):
        missing.append("未识别到可复核的错误码、原始报错片段或日志关键短语")
    performance_signal = re.search(
        r"(msprof|profiling|profile|trace|timeline|\u8017\u65f6|\u5e26\u5bbd)", task, re.I
    )
    if "performance" in labels and not performance_signal:
        missing.append("未识别到 profiling/msprof 指标或性能瓶颈现象")
    return _dedupe(missing)


def _sufficiency_rule(route, intent):
    labels = set(intent.get("labels", []) if intent else [])
    checks = [
        "回答前必须至少 get/read `read_first` 中 1-3 张卡片，不能只依据 search/preflight 摘要回答",
        "读取后至少确认 1 张 strong 证据卡，或 2 张 medium 证据卡互相支撑；weak 证据不能单独支撑结论",
        "读取后由 Agent 判断证据是否覆盖用户问题中的 API 语义、实现约束、样例或诊断链路",
    ]
    if route == "browse_first":
        checks.append("开放式问题先用 overview/browse 建立目录视图，再选择具体卡片读取")
    if labels & {"implementation", "similar_example", "api_usage"}:
        checks.append("实现类问题至少确认 API/接口约束和一个实现或样例依据")
    if "debug" in labels:
        checks.append("诊断类问题至少复核错误关键字命中的卡片；证据不足时继续 grep 或二次 preflight")
    if "performance" in labels:
        checks.append("性能类问题至少追踪 runbooks/优化经验或 profiling 相关卡片")
    checks.append("若读取后仍缺少关键约束，继续 neighbors、overview 或二次 preflight，而不是直接补全猜测")
    return checks


def _candidate_queries(task, idx):
    exact = []
    exact.extend(BACKTICK_RE.findall(task))
    exact.extend(ERROR_RE.findall(task))
    exact.extend(
        w for w in ACL_MACRO_RE.findall(task)
        if (w.startswith("ACL_") or "_" in w) and w.lower() not in LOW_VALUE_WORDS
    )
    exact.extend(w for w in IDENT_RE.findall(task) if _is_symbol(w))
    exact.extend(_index_exact_terms(task, idx))

    candidates = []
    for q in _dedupe(exact):
        candidates.append({"query": q, "reason": "exact", "priority": 100})
    hint_info = _hint_phrases(task)
    candidates.extend(hint_info["phrases"])
    return candidates, hint_info


def _command(root_arg, verb, args):
    parts = ["knowledge_query.py"]
    if root_arg:
        parts.extend(["--knowledge-root", root_arg])
    parts.append(verb)
    parts.extend(args)
    return " ".join(shlex.quote(str(p)) for p in parts)


def _queries_by_reason(candidates, reason_prefix, max_queries):
    values = []
    for cand in candidates:
        if str(cand.get("reason", "")).startswith(reason_prefix):
            values.append(cand["query"])
    return _dedupe(values)[:max_queries]


def plan_task(task, idx, max_queries=6, knowledge_root=None, platform=None):
    candidates, hint_info = _candidate_queries(task, idx)
    candidates = sorted(candidates, key=lambda c: -c["priority"])
    intent = _detect_intent(task, hint_info["signals"])

    warnings = []
    warnings.extend(hint_info["warnings"])
    exact_queries = _queries_by_reason(candidates, "exact", max_queries)
    exact_queries = _dedupe(_intent_primary_queries(task, intent) + exact_queries)[:max_queries]
    hint_queries = _queries_by_reason(candidates, "hint:", max_queries)

    if not exact_queries and not hint_queries:
        words = [w for w in re.findall(r"[0-9A-Za-z_]+", task) if w.lower() not in LOW_VALUE_WORDS]
        exact_queries = _dedupe(words[:max_queries])
        warnings.append("no exact symbols detected; using filtered task words")
    queries = _dedupe(exact_queries + hint_queries)[:max_queries]
    warnings.extend(query_warnings(queries))

    grep_terms = []
    grep_terms.extend(q for q in exact_queries[:8] if _is_good_grep_term(q))
    grep_terms.extend(hint_info["grep"])
    grep_terms = _dedupe(grep_terms)
    route = _route_for_plan(task, intent, exact_queries)
    platform_policy = platform_context(task, explicit=platform)
    platform_args = (["--platform", platform_policy["target"]]
                     if platform_policy["enabled"] else [])

    commands = []
    if route == "browse_first":
        commands.append(_command(knowledge_root, "overview", ["--task", task]))
    if exact_queries:
        for q in exact_queries[:max_queries]:
            commands.append(_command(knowledge_root, "search", ["--query", q] + platform_args))
    if hint_queries:
        for q in hint_queries[:max_queries]:
            commands.append(_command(knowledge_root, "search", ["--query", q] + platform_args))
    commands.extend(_command(knowledge_root, "grep", [p] + platform_args) for p in grep_terms[:3])

    return {
        "task": task,
        "queries": queries,
        "exact_queries": exact_queries[:max_queries],
        "hint_queries": hint_queries[:max_queries],
        "grep": grep_terms,
        "commands": commands,
        "knowledge_signals": hint_info["signals"],
        "intent": intent,
        "route": route,
        "platform_filter": platform_filter_output(platform_policy),
        "missing_signals": _missing_signals(task, intent, exact_queries, route=route),
        "sufficiency_rule": _sufficiency_rule(route, intent),
        "hint_source": hint_info["source"],
        "warnings": _dedupe(warnings),
    }


def _paths_from_results(results):
    return [item["path"] for item in results]


def _paths_from_grep(grep_matches, precise=None):
    paths = []
    for group in grep_matches:
        if precise is True and not _is_precise_grep_term(group["pattern"]):
            continue
        if precise is False and _is_precise_grep_term(group["pattern"]):
            continue
        for match in group["matches"]:
            paths.append(match["path"])
    return paths


def _drop_paths(results, paths):
    banned = set(paths)
    return [item for item in results if item.get("path") not in banned]


def _append_suggestions(state, paths, source, quota):
    out, seen, limit = state
    added = 0
    for path in paths:
        if path in seen:
            continue
        seen.add(path)
        out.append({"path": path, "source": source})
        added += 1
        if limit and len(out) >= limit:
            return True
        if quota and added >= quota:
            return False
    return False


def _suggested_details(path_groups, k, limit):
    if not limit:
        return []
    exact_quota = max(k, 1)
    hint_quota = max(1, min(2, k))
    precise_quota = max(1, min(2, k))
    broad_quota = max(1, min(2, k // 2 or 1))
    groups = [
        ("precise_grep", _dedupe(path_groups["precise_grep"]), precise_quota),
        ("exact", path_groups["exact"], exact_quota),
        ("broad_grep", _dedupe(path_groups["broad_grep"]), broad_quota),
        ("hint", path_groups["hint"], hint_quota),
    ]
    seen = set()
    out = []
    state = (out, seen, limit)
    for source, paths, quota in groups:
        if _append_suggestions(state, paths, source, quota):
            return out
    # Fill any remaining budget in the same source order after source quotas
    # have protected exact/hint coverage.
    for source, paths, _ in groups:
        if _append_suggestions(state, paths, source, 0):
            return out
    return out


def _doc_brief(idx, path):
    for doc in idx.get("docs", []):
        if doc.get("path") == path:
            return {
                "path": path,
                "local_path": local_path_for_doc(path),
                "title": doc.get("title", ""),
                "kind": doc.get("kind", ""),
                "category": doc.get("category", ""),
                "bundle": doc.get("bundle", ""),
                "description": doc.get("description", ""),
                "tags": doc.get("tags", []),
                "platforms": doc.get("platforms", []),
                "resource": doc.get("resource", ""),
            }
    return attach_local_paths({"path": path})


def _annotate_suggestions(idx, suggested_get_details, intent, queries):
    source_bonus = {
        "precise_grep": 0.10,
        "exact": 0.06,
        "broad_grep": 0.03,
        "hint": 0.01,
        "result": 0.0,
    }
    level_bonus = {"strong": 0.10, "medium": 0.03, "weak": 0.0}
    out = []
    for rank, item in enumerate(suggested_get_details):
        path = item.get("path")
        if not path:
            continue
        brief = _doc_brief(idx, path)
        merged = dict(item)
        for key, value in brief.items():
            merged.setdefault(key, value)
        merged["relevance"] = evaluate_relevance(merged, intent=intent, queries=queries)
        merged["read_purpose"] = read_purpose(merged, intent=intent)
        level = (merged.get("relevance") or {}).get("level")
        score = (
            float((merged.get("relevance") or {}).get("score") or 0.0)
            + source_bonus.get(merged.get("source"), 0.0)
            + level_bonus.get(level, 0.0)
        )
        merged["selection_score"] = round(score, 3)
        merged["_rank"] = rank
        out.append(merged)
    out.sort(key=lambda item: (
        -float(item.get("selection_score") or 0.0),
        item["_rank"],
        item.get("path", ""),
    ))
    for item in out:
        item.pop("_rank", None)
    return out


def _read_first(idx, suggested_get_details, fallback_results, plan, limit=3):
    source_reason = {
        "precise_grep": "精确 grep 命中 API/宏/错误码，优先读取整卡复核上下文",
        "exact": "精确 query 主召回命中，适合作为第一批证据",
        "broad_grep": "宽泛 grep 命中，可用于补充错误短语或中文概念上下文",
        "hint": "领域 hint 补充召回，适合在精确结果不足时读取",
    }
    paths = []
    details_by_path = {}
    for item in suggested_get_details:
        path = item.get("path")
        if not path:
            continue
        paths.append(path)
        details_by_path[path] = item
    if not paths:
        paths = _paths_from_results(fallback_results)
    out = []
    for path in _dedupe(paths)[:limit]:
        brief = _doc_brief(idx, path)
        detail = details_by_path.get(path, {})
        source = detail.get("source", "result")
        if detail.get("relevance"):
            brief["relevance"] = detail["relevance"]
        else:
            brief["relevance"] = evaluate_relevance(
                brief,
                intent=plan.get("intent", {}),
                queries=plan.get("exact_queries", []) or plan.get("queries", []),
            )
        brief["source"] = source
        brief["reason"] = source_reason.get(source, "检索结果靠前，建议先读取后再判断是否继续检索")
        brief["read_purpose"] = detail.get("read_purpose") or read_purpose(brief, intent=plan.get("intent", {}))
        brief["verify_after_read"] = verify_after_read(
            brief,
            intent=plan.get("intent", {}),
            missing_signals=plan.get("missing_signals", []),
        )
        out.append(brief)
    return out


def _query_command(knowledge_root, verb, path, extra=None):
    args = [path]
    if extra:
        args.extend(extra)
    return _command(knowledge_root, verb, args)


def _neighbor_follow_candidates(read_first, knowledge_root, platform_args):
    out = []
    for card in read_first[:2]:
        path = card.get("path")
        if not path:
            continue
        out.append({
            "action": "neighbors",
            "path": path,
            "when": "读取该卡片后仍需要相似样例、相关 API、约束或经验上下文",
            "command": _query_command(
                knowledge_root, "neighbors", path, ["--hops", "1", "-k", "8"] + platform_args
            ),
        })
    return out


def _pipeline_follow_candidate(knowledge_root, when, rerank, scope, command_args):
    return {
        "action": "pipeline",
        "when": when,
        "command": _command(
            knowledge_root,
            "pipeline",
            ["--recall", "bm25,tagtype", "--rerank", rerank, "--scope", scope]
            + command_args,
        ),
    }


def _follow_candidates(plan, read_first, knowledge_root):
    labels = set(plan.get("intent", {}).get("labels", []) or [])
    query_args = []
    platform_args = []
    platform_policy = plan.get("platform_filter", {})
    if platform_policy.get("enabled") and platform_policy.get("target"):
        platform_args = ["--platform", platform_policy["target"]]
    for query in (plan.get("exact_queries") or plan.get("queries") or [])[:2]:
        query_args.extend(["--query", query])
    out = _neighbor_follow_candidates(read_first, knowledge_root, platform_args)
    command_args = query_args + platform_args
    if "api_usage" in labels:
        out.append(_pipeline_follow_candidate(
            knowledge_root, "需要复核 API/接口/参数约束时", "bm25f", "reference/", command_args
        ))
    if labels & {"implementation", "similar_example"}:
        out.append(_pipeline_follow_candidate(
            knowledge_root, "需要找相似算子实现、kernel 写法或 Tiling 样例时",
            "bm25f", "ops/", command_args
        ))
    if "performance" in labels:
        out.append(_pipeline_follow_candidate(
            knowledge_root, "需要性能优化、profiling 或 runbook 经验时",
            "quality", "runbooks/", command_args
        ))
    if plan.get("route") == "browse_first":
        out.append({
            "action": "overview",
            "when": "问题是开放式、枚举式或用户想先了解知识范围时",
            "command": _command(knowledge_root, "overview", ["--task", plan.get("task", "")]),
        })
    return out


def _next_actions(plan, read_first, follow_candidates, knowledge_root):
    actions = []
    platform_args = []
    platform_policy = plan.get("platform_filter", {})
    if platform_policy.get("enabled") and platform_policy.get("target"):
        platform_args = ["--platform", platform_policy["target"]]
    if plan.get("route") == "browse_first":
        actions.append({
            "step": "browse",
            "instruction": "先查看 overview/browse 输出，选择最相关的目录或类别，再进入具体卡片",
            "command": _command(knowledge_root, "overview", ["--task", plan.get("task", "")]),
        })
    if read_first:
        paths = [card["path"] for card in read_first if card.get("path")]
        actions.append({
            "step": "read",
            "instruction": "读取 read_first 中 1-3 张卡片；优先 Read local_path，并按 relevance、"
                           "read_purpose、verify_after_read 判断证据是否覆盖问题",
            "command": _command(knowledge_root, "get", paths),
        })
    else:
        actions.append({
            "step": "revise_query",
            "instruction": "没有可直接读取的卡片；请 Agent 根据 missing_signals 改写短 query 或补充约束后再次 preflight",
            "command": _command(
                knowledge_root, "preflight", ["--task", plan.get("task", ""), "--brief"] + platform_args
            ),
        })
    if follow_candidates:
        actions.append({
            "step": "follow",
            "instruction": "读取后如果证据不足，从 follow_candidates 里选择 neighbors、pipeline 或 overview 继续检索",
        })
    actions.append({
        "step": "answer",
        "instruction": "只有在 sufficiency_rule 满足后再回答；回答中保留 doc-id/resource 作为依据",
    })
    return actions


def _brief_result(item):
    result = {}
    keys = (
        "path", "title", "kind", "category", "score", "rank_score",
        "intent_boost", "relevance_boost", "relevance", "snippet",
    )
    for key in keys:
        if key in item:
            result[key] = item[key]
    return result


def brief_preflight_output(out):
    plan = out["plan"]
    grep_matches = out.get("grep_matches", [])
    return attach_local_paths({
        "knowledge_root": out.get("knowledge_root"),
        "task": plan.get("task"),
        "intent": plan.get("intent", {}),
        "route": plan.get("route"),
        "platform_filter": out.get("platform_filter", plan.get("platform_filter", {})),
        "exact_queries": plan.get("exact_queries", []),
        "hint_queries": plan.get("hint_queries", []),
        "grep": plan.get("grep", []),
        "read_first": out.get("read_first", []),
        "next_actions": out.get("next_actions", []),
        "follow_candidates": out.get("follow_candidates", []),
        "missing_signals": plan.get("missing_signals", []),
        "sufficiency_rule": plan.get("sufficiency_rule", []),
        "results": [_brief_result(item) for item in out.get("results", [])],
        "supplemental_results": [_brief_result(item) for item in out.get("supplemental_results", [])],
        "suggested_get": out.get("suggested_get", []),
        "suggested_get_local_paths": out.get("suggested_get_local_paths", []),
        "suggested_get_details": out.get("suggested_get_details", []),
        "grep_match_count": sum(len(group.get("matches", [])) for group in grep_matches),
        "counts": {
            "results": len(out.get("results", [])),
            "supplemental_results": len(out.get("supplemental_results", [])),
            "grep_groups": len(grep_matches),
        },
        "warnings": plan.get("warnings", []),
    })


def _preflight_retrieval(plan, idx, k, grep_k, target_platform):
    exact_results = (
        search_results(plan["exact_queries"], k=k, idx=idx, platform=target_platform)
        if plan["exact_queries"] else []
    )
    hint_results = (
        search_results(plan["hint_queries"], k=k, idx=idx, platform=target_platform)
        if plan["hint_queries"] else []
    )
    exact_results = _rerank_for_intent(exact_results, plan["intent"], plan["exact_queries"])
    hint_results = _rerank_for_intent(hint_results, plan["intent"], plan["hint_queries"])
    grep_matches = []
    for pattern in plan["grep"][:4]:
        matches = grep_results(pattern, k=grep_k, platform=target_platform)
        if not matches:
            continue
        annotate_relevance(matches, intent=plan["intent"], queries=[pattern])
        grep_matches.append({"pattern": pattern, "matches": matches})
    return exact_results, hint_results, grep_matches


def _suggestion_path_groups(exact_results, hint_results, grep_matches):
    precise = _paths_from_grep(grep_matches, precise=True)
    broad = _paths_from_grep(grep_matches, precise=False)
    return {
        "grep": _dedupe(precise + broad),
        "precise_grep": _dedupe(precise),
        "broad_grep": _dedupe(broad),
        "exact": _paths_from_results(exact_results),
        "hint": _paths_from_results(hint_results),
    }


def _annotated_suggestion_details(idx, plan, path_groups, k):
    grep_paths = path_groups["grep"]
    limit = max(k, min(k + len(grep_paths), k * 2 if k else 0))
    details = _suggested_details(path_groups, k, limit)
    return _annotate_suggestions(
        idx, details, plan["intent"], plan["exact_queries"] or plan["queries"]
    )


def preflight_task(task, idx, *, max_queries=6, k=8, grep_k=8, knowledge_root=None, platform=None):
    plan = plan_task(task, idx, max_queries=max_queries, knowledge_root=knowledge_root, platform=platform)
    target_platform = (plan["platform_filter"].get("target")
                       if plan["platform_filter"].get("enabled") else None)
    exact_results, hint_results, grep_matches = _preflight_retrieval(
        plan, idx, k, grep_k, target_platform
    )
    primary_results = exact_results if exact_results else hint_results
    supplemental_results = _drop_paths(hint_results, _paths_from_results(primary_results))
    path_groups = _suggestion_path_groups(exact_results, hint_results, grep_matches)
    suggested_get_by_source = {
        key: path_groups[key]
        for key in ("grep", "precise_grep", "broad_grep", "exact", "hint")
    }
    suggested_get_details = _annotated_suggestion_details(idx, plan, path_groups, k)
    suggested_get = [item["path"] for item in suggested_get_details]
    read_first = _read_first(idx, suggested_get_details, primary_results, plan, limit=3)
    follow_candidates = _follow_candidates(plan, read_first, knowledge_root)
    next_actions = _next_actions(plan, read_first, follow_candidates, knowledge_root)
    retained_count = len(primary_results) + sum(len(group["matches"]) for group in grep_matches)
    platform_filter = platform_filter_output(plan["platform_filter"], retained_count)

    return attach_local_paths({
        "plan": plan,
        "route": plan.get("route"),
        "platform_filter": platform_filter,
        "read_first": read_first,
        "next_actions": next_actions,
        "follow_candidates": follow_candidates,
        "missing_signals": plan.get("missing_signals", []),
        "sufficiency_rule": plan.get("sufficiency_rule", []),
        "results": primary_results,
        "exact_results": exact_results,
        "supplemental_results": supplemental_results,
        "grep_matches": grep_matches,
        "suggested_get_by_source": suggested_get_by_source,
        "suggested_get_details": suggested_get_details,
        "suggested_get": suggested_get,
    })
