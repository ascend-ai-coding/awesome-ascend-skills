# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

"""retrieval.search — BM25F recall + deterministic rerank.

Multiple queries are still supported internally for explicit legacy calls, but
the CLI rejects them by default to avoid mixing unrelated retrieval intents.
"""
import json
import math
import re
from dataclasses import dataclass

from retrieval.config import (B, EXACT_NAME_BONUS, FIELD_WEIGHTS, K1, SCORE_PREC,
                              TAG_EXACT_BONUS)
from retrieval.index import load_index
from retrieval.output import emit_stderr, emit_stdout
from retrieval.paths import attach_local_paths
from retrieval.platforms import (filter_candidate_ids, platform_context,
                                 platform_filter_output)
from retrieval.relevance import annotate_relevance
from retrieval.scope import _filter_candidates, facet_filter
from retrieval.snippet import snippet_for_doc
from retrieval.tokenizer import tokenize

CJK_RE = re.compile(r"[\u3400-\u9fff]")


@dataclass
class _ScoreAccumulator:
    total: dict
    fields: dict
    terms: dict
    queries: dict


def query_warnings(queries):
    warnings = []
    for q in queries or []:
        text = str(q).strip()
        if not text:
            continue
        toks = tokenize(text)
        cjk_len = len(CJK_RE.findall(text))
        if len(toks) > 8 or len(text) > 80 or cjk_len > 24:
            warnings.append(
                "broad query detected; use a shorter single --query or run plan/preflight: %s" % text
            )
    return warnings


def _idf(term, idx):
    n = idx["meta"]["card_count"]
    df = idx["df"].get(term, 0)
    return math.log((n - df + 0.5) / (df + 0.5) + 1.0)


def _wtf(term, doc_id, idx):
    pt = idx["_pt"].get(term, {})
    fields = pt.get(doc_id)
    if not fields:
        return 0.0
    doc = idx["docs"][doc_id]
    avg = idx["avg_field_len"]
    total = 0.0
    for field, tf in fields.items():
        a = avg.get(field, 0.0)
        if a == 0:                       # empty-field protection
            continue
        w = FIELD_WEIGHTS[field]
        ln = doc["field_lengths"].get(field, 0)
        denom = 1 - B + B * ln / a
        total += w * tf / denom
    return total


def _tag_subseq_hit(tag, qtokens):
    tt = tokenize(tag)
    if not tt:
        return False
    for i in range(len(qtokens) - len(tt) + 1):
        if qtokens[i:i + len(tt)] == tt:
            return True
    return False


def _score_one_query(query, idx, candidates):
    qtokens = tokenize(query)
    qset = set(qtokens)
    global_tags = set(idx["meta"]["global_tags"])
    scores, matched_fields, matched_terms = {}, {}, {}
    for term in qset:
        idf = _idf(term, idx)
        for doc_id in idx["_docs_with"].get(term, ()):
            if candidates is not None and doc_id not in candidates:
                continue
            wtf = _wtf(term, doc_id, idx)
            if wtf <= 0:
                continue
            s = idf * wtf / (K1 + wtf)
            scores[doc_id] = scores.get(doc_id, 0.0) + s
            matched_terms.setdefault(doc_id, set()).add(term)
            matched_fields.setdefault(doc_id, set()).update(idx["_pt"][term][doc_id].keys())
    # deterministic bonuses
    for doc_id in list(scores):
        doc = idx["docs"][doc_id]
        if qset & set(doc.get("names", [])):
            scores[doc_id] += EXACT_NAME_BONUS
        for g in doc["tags"]:
            if g in global_tags:
                continue
            if _tag_subseq_hit(g, qtokens):
                scores[doc_id] += TAG_EXACT_BONUS
    return scores, matched_fields, matched_terms


def _accumulate_scores(queries, idx, candidates):
    scores = _ScoreAccumulator({}, {}, {}, {})
    for query_index, query in enumerate(queries):
        query_scores, matched_fields, matched_terms = _score_one_query(query, idx, candidates)
        for doc_id, score in query_scores.items():
            scores.total[doc_id] = scores.total.get(doc_id, 0.0) + score
            scores.fields.setdefault(doc_id, set()).update(matched_fields.get(doc_id, ()))
            scores.terms.setdefault(doc_id, set()).update(matched_terms.get(doc_id, ()))
            scores.queries.setdefault(doc_id, set()).add(query_index)
    return scores


def _result_payload(idx, doc_id, members, scores):
    doc = idx["docs"][doc_id]
    variants = sorted(idx["docs"][member]["path"] for member in members if member != doc_id)
    return {
        "path": doc["path"], "bundle": doc["bundle"], "title": doc["title"],
        "description": doc["description"], "kind": doc["kind"], "category": doc["category"],
        "tags": doc["tags"], "platforms": doc.get("platforms", []),
        "score": round(scores.total.get(doc_id, 0.0), SCORE_PREC),
        "snippet": snippet_for_doc(doc, scores.terms.get(doc_id, ())),
        "matched": {
            "fields": sorted(scores.fields.get(doc_id, ())),
            "terms": sorted(scores.terms.get(doc_id, ())),
            "queries": sorted(scores.queries.get(doc_id, ())),
        },
        "variants": variants,
    }


def _grouped_results(idx, ranked, scores, k):
    groups = {}
    for doc_id in ranked:
        doc = idx["docs"][doc_id]
        key = (doc["bundle"], doc["reldir"], doc["base"])
        groups.setdefault(key, []).append(doc_id)

    results, emitted = [], set()
    for doc_id in ranked:
        doc = idx["docs"][doc_id]
        key = (doc["bundle"], doc["reldir"], doc["base"])
        if key in emitted:
            continue
        emitted.add(key)
        results.append(_result_payload(idx, doc_id, groups.get(key, []), scores))
        if len(results) >= k:
            break
    return results


def search_results(queries, bundle=None, kind=None, category=None, section=None, *, k=10,
                   idx=None, platform=None, scope=None):
    idx = idx or load_index()
    if scope:
        candidates = facet_filter(
            idx, scope=scope, bundle=bundle, kind=kind, category=category,
            section=section, status="all",
        )
    else:
        candidates = _filter_candidates(idx, bundle, kind, category, section)
    context = platform_context(explicit=platform) if platform else platform_context()
    candidates = filter_candidate_ids(idx, candidates, context)
    scores = _accumulate_scores(queries, idx, candidates)
    ranked = sorted(
        scores.total,
        key=lambda doc_id: (
            -round(scores.total.get(doc_id, 0.0), SCORE_PREC), idx["docs"][doc_id]["path"],
        ),
    )
    results = _grouped_results(idx, ranked, scores, k)
    return annotate_relevance(results, queries=queries)


def cmd_search(queries, bundle, kind, category, section, *, k, platform=None, scope=None):
    warnings = query_warnings(queries)
    for warning in warnings:
        emit_stderr("WARN: %s" % warning)
    results = search_results(
        queries, bundle, kind, category, section, k=k, platform=platform, scope=scope
    )
    context = platform_context(explicit=platform) if platform else platform_context()
    emit_stdout(json.dumps(attach_local_paths({
        "results": results,
        "warnings": warnings,
        "platform_filter": platform_filter_output(context, len(results)),
    }), ensure_ascii=False, indent=2))
