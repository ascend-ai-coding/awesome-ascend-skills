# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

"""retrieval.judge — shared LLM listwise relevance scorer for the opt-in model rerank
routes (llm-judge auto path + LLM reranker). One LLM call scores the whole candidate set
(listwise), NOT one call per card (pointwise would blow up cost/latency linearly).

Determinism guards (SPEC-Retrieve §6/§9):
  · Candidates are presented in a CANONICAL order (sorted by path), so the same candidate
    set yields the same prompt regardless of incoming hit order.
  · The cache key binds model + prompt/schema version + policy + material + query + the
    ORDERED candidate payload (path + title + kind + content), so order/content changes
    bust the cache while a true repeat hits it.
"""
from retrieval.cache import fingerprint
from retrieval.cards import _parse_front, id_to_path, strip_related
from retrieval.config import LLM_PROMPT_VERSION, LLM_SCHEMA_VERSION, SCORE_PREC

JUDGE_SYSTEM = (
    "You are a retrieval relevance judge for an Ascend C / NPU operator knowledge base. "
    "Given a query and candidate knowledge cards, score each candidate's relevance. "
    "Be strict and consistent. Respond with JSON only — no prose.")


def _candidate_item(idx, by_path, path, material):
    d = idx["docs"][by_path[path]] if path in by_path else {}
    item = {"path": path, "title": d.get("title", ""), "kind": d.get("kind", "")}
    if material != "card":
        item["content"] = d.get("description", "")
        return item
    try:
        with open(id_to_path(path), encoding="utf-8") as card_file:
            raw = card_file.read()
        _, body = _parse_front(raw)
        item["content"] = strip_related(body).strip()
    except OSError:
        item["content"] = d.get("description", "")
    return item


def _candidate_payload(idx, hits, material):
    """Canonical (path-sorted) candidate list; `content` = description (header, default,
    cheap) or the related-stripped card body (material='card')."""
    by_path = {d["path"]: i for i, d in enumerate(idx["docs"])}
    paths = sorted({h["path"] for h in hits})
    cands = []
    for p in paths:
        cands.append(_candidate_item(idx, by_path, p, material))
    return paths, cands


def _build_prompt(query, cands, policy):
    import json
    return json.dumps({
        "task": ("Score EACH candidate 0..10 by relevance to the query (10 = directly "
                 "answers it, 0 = irrelevant). Return ONLY JSON: "
                 "{\"scores\": {\"<path>\": <number>, ...}, \"why\": {\"<path>\": \"<short reason>\"}}. "
                 "Include every candidate path exactly once."),
        "policy": policy, "query": query or "", "candidates": cands,
    }, ensure_ascii=False, indent=2)


def cache_key(model, query, cands, policy, material):
    return fingerprint({
        "model": model, "prompt_version": LLM_PROMPT_VERSION,
        "schema_version": LLM_SCHEMA_VERSION, "policy": policy, "material": material,
        "query": query or "", "candidates": cands,          # ordered payload (path-sorted)
    })


def judge_listwise(client, idx, query, hits, policy="relevance", *, material="header"):
    """-> ({path: float score}, {path: why}). Uses client.complete_json with the bound
    cache key; raises NotConfigured / ModelRuntimeError from the client on failure.
    """
    paths, cands = _candidate_payload(idx, hits, material)
    key = cache_key(client.model, query, cands, policy, material)
    prompt = _build_prompt(query, cands, policy)
    obj = client.complete_json(prompt, system=JUDGE_SYSTEM, cache_key=key)
    raw = obj.get("scores", obj) if isinstance(obj, dict) else {}
    whys = obj.get("why", {}) if isinstance(obj, dict) else {}
    scores = {}
    for p in paths:
        try:
            scores[p] = float(raw.get(p, 0.0))
        except (TypeError, ValueError):
            scores[p] = 0.0
    return scores, (whys if isinstance(whys, dict) else {})


def apply_scores(hits, scores, suffix, whys=None):
    """Replace each hit's score with the judged score, relabel method, deterministic sort.
    Carries an optional `why` from the judge.
    """
    whys = whys or {}
    out = []
    for h in hits:
        nh = dict(h)
        nh["score"] = round(float(scores.get(h["path"], 0.0)), SCORE_PREC)
        nh["method"] = (h.get("method", "") + ">" + suffix).lstrip(">")
        if whys.get(h["path"]):
            nh["why"] = whys[h["path"]]
        out.append(nh)
    out.sort(key=lambda h: (-h["score"], h["path"]))
    return out
