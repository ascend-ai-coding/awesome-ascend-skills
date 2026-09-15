# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

"""retrieval.overview - browse-first view over the knowledge-card index."""
from __future__ import annotations

import json
import math
import shlex

from retrieval.aliases import get_alias_map
from retrieval.output import emit_stdout
from retrieval.paths import attach_local_paths
from retrieval.relevance import annotate_relevance
from retrieval.scope import display_scope, facet_filter, normalize_scope
from retrieval.tokenizer import tokenize


KIND_PRIOR = {
    "index": 4.0,
    "guide": 2.2,
    "operator_optimization": 1.8,
    "api": 1.4,
    "example": 1.4,
    "operator": 1.2,
    "implementation_trap": 1.2,
    "debugging_journey": 1.2,
    "glossary": 1.0,
}


def _doc_text(doc):
    parts = [
        doc.get("path", ""),
        doc.get("title", ""),
        doc.get("description", ""),
        doc.get("kind", ""),
        doc.get("category", ""),
        doc.get("bundle", ""),
        " ".join(doc.get("tags", []) or []),
        " ".join(doc.get("aliases", []) or []),
    ]
    return " ".join(str(p) for p in parts if p)


def _score_doc(doc, query_tokens):
    score = KIND_PRIOR.get(doc.get("kind", ""), 0.5)
    if doc.get("path", "").endswith("/index.md") or doc.get("path") == "index.md":
        score += 2.0
    if not query_tokens:
        return score

    text = _doc_text(doc).lower()
    title = str(doc.get("title", "")).lower()
    tags = " ".join(doc.get("tags", []) or []).lower()
    for tok in query_tokens:
        if tok in title:
            score += 3.0
        elif tok in tags:
            score += 2.0
        elif tok in text:
            score += 1.0
    return score


def _brief_doc(doc, score):
    return {
        "path": doc.get("path", ""),
        "title": doc.get("title", ""),
        "kind": doc.get("kind", ""),
        "category": doc.get("category", ""),
        "bundle": doc.get("bundle", ""),
        "description": doc.get("description", ""),
        "score": round(score, 4),
    }


def _scored_docs(idx, keep, query_tokens):
    docs = []
    for doc_id, doc in enumerate(idx.get("docs", [])):
        if keep is not None and doc_id not in keep:
            continue
        docs.append((_score_doc(doc, query_tokens), doc))
    return docs


def _grouped_docs(scored_docs):
    grouped = {}
    for score, doc in scored_docs:
        key = (doc.get("bundle", ""), doc.get("category", ""))
        group = grouped.setdefault(
            key,
            {
                "bundle": key[0],
                "category": key[1],
                "count": 0,
                "kind_counts": {},
                "cards": [],
                "_score": 0.0,
            },
        )
        group["count"] += 1
        kind = doc.get("kind", "")
        group["kind_counts"][kind] = group["kind_counts"].get(kind, 0) + 1
        group["_score"] = max(group["_score"], score)
        group["cards"].append((score, doc))
    return grouped


def _group_rows(grouped, query_text, per_group, limit):
    rows = []
    for group in grouped.values():
        group["cards"].sort(key=lambda item: (-item[0], item[1].get("path", "")))
        top_cards = [_brief_doc(doc, score) for score, doc in group["cards"][:per_group]]
        annotate_relevance(top_cards, queries=[query_text] if query_text else None)
        score = group["_score"] + math.log(group["count"] + 1.0)
        rows.append({
            "bundle": group["bundle"],
            "category": group["category"],
            "count": group["count"],
            "kind_counts": dict(sorted(group["kind_counts"].items())),
            "score": round(score, 4),
            "cards": top_cards,
        })
    rows.sort(key=lambda item: (-item["score"], item["bundle"], item["category"]))
    return rows[:limit]


def _first_cards(rows, limit=3):
    read_first = []
    seen = set()
    for group in rows:
        for card in group["cards"]:
            path = card.get("path")
            if not path or path in seen:
                continue
            seen.add(path)
            read_first.append(card)
            if len(read_first) >= limit:
                return read_first
    return read_first


def _next_actions(knowledge_root, read_first):
    return [
        {
            "step": "choose_scope",
            "instruction": "Agent 先根据 groups 判断最相关的 bundle/category，而不是直接扩大 top-k",
        },
        {
            "step": "read",
            "instruction": "读取 read_first 中 1-3 张目录卡或概览卡；若输出含 local_path，"
                           "优先 Read local_path，再决定进入 API、样例、runbook 或二次 preflight",
            "command": _get_command(knowledge_root, read_first),
        },
        {
            "step": "drill_down",
            "instruction": "如果目录卡不足以回答，围绕选定类别执行 preflight/search/pipeline，并对核心卡调用 neighbors",
        },
    ]


def _overview_filters(scope, bundle, kind, category, status):
    return {
        "scope": display_scope(normalize_scope(scope)) if scope else "",
        "bundle": bundle,
        "kind": kind,
        "category": category,
        "status": status,
    }


def _overview_summary(docs, grouped, rows):
    return {
        "card_count": len(docs),
        "group_count": len(grouped),
        "shown_groups": len(rows),
    }


def overview_result(
    idx,
    *,
    task=None,
    query=None,
    scope=None,
    bundle=None,
    kind=None,
    category=None,
    status="active",
    groups=8,
    per_group=4,
    knowledge_root=None,
):
    query_text = query or task or ""
    qtokens = set(tokenize(query_text))
    alias_map = get_alias_map(idx)
    keep = facet_filter(
        idx,
        scope=scope,
        bundle=bundle,
        kind=kind,
        category=category,
        status=status,
        alias_map=alias_map,
    )
    docs = _scored_docs(idx, keep, qtokens)
    grouped = _grouped_docs(docs)
    rows = _group_rows(grouped, query_text, per_group, groups)
    read_first = _first_cards(rows)
    return {
        "route": "browse_first",
        "task": task,
        "query": query,
        "knowledge_root": knowledge_root,
        "filters": _overview_filters(scope, bundle, kind, category, status),
        "summary": _overview_summary(docs, grouped, rows),
        "groups": rows,
        "read_first": read_first,
        "next_actions": _next_actions(knowledge_root, read_first),
    }


def _get_command(knowledge_root, cards):
    paths = [c.get("path") for c in cards if c.get("path")]
    if not paths:
        return None
    parts = ["knowledge_query.py"]
    if knowledge_root:
        parts.extend(["--knowledge-root", knowledge_root])
    parts.append("get")
    parts.extend(paths)
    return " ".join(shlex.quote(str(part)) for part in parts)


def cmd_overview(args, idx, knowledge_root=None):
    out = overview_result(
        idx,
        task=args.task,
        query=args.query,
        scope=args.scope or args.dir,
        bundle=args.bundle,
        kind=args.kind,
        category=args.category,
        status=args.status,
        groups=args.groups,
        per_group=args.per_group,
        knowledge_root=knowledge_root,
    )
    annotate_relevance(out.get("read_first", []), queries=[out.get("query") or out.get("task") or ""])
    emit_stdout(json.dumps(attach_local_paths(out), ensure_ascii=False, indent=2))
