# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

"""rerank.reranker_model — model reranker (opt-in, non-deterministic).

Default backend = `LLMReranker`: one listwise Claude Code SDK call scores the candidate
set (shared retrieval.judge), then a deterministic reorder + fingerprint cache. The slot
is the extension point for a real cross-encoder (`CrossEncoderReranker`, not yet wired —
raises `NotConfigured` until a backend is configured).

Clients are constructed lazily per model name — instantiation is side-effect-free so the
REGISTRY stays stdlib-only; claude_agent_sdk is imported only when a rerank actually runs.
"""
from retrieval.errors import NotConfigured
from retrieval.rerank.base import Rerank

JUDGE_POLICY = "relevance"


class ModelReranker(Rerank):
    """Base for model-backed rerankers. Subclasses implement `rerank`."""


class LLMReranker(ModelReranker):
    def __init__(self):
        self._clients = {}                       # model name -> LLMClient (lazy)

    def rerank(self, idx, query, hits, spec):
        if not hits:
            return []
        from retrieval.judge import apply_scores, judge_listwise
        client = self._llm(spec.get("llm_model"))
        scores, whys = judge_listwise(client, idx, query, hits, policy=JUDGE_POLICY,
                                      material=spec.get("material", "header"))
        return apply_scores(hits, scores, "reranker", whys)

    def _llm(self, model):
        if model not in self._clients:
            from retrieval.llm import LLMClient
            self._clients[model] = LLMClient(model=model)
        return self._clients[model]


class CrossEncoderReranker(ModelReranker):
    """Extension point: a real cross-encoder (e.g. sentence-transformers CrossEncoder).
    Not configured here -> NotConfigured (exit 3).
    """

    def rerank(self, idx, query, hits, spec):
        raise NotConfigured(
            "reranker",
            "cross-encoder backend not configured (needs sentence-transformers + a "
            "cross-encoder model). Use the default LLM reranker, or deterministic "
            "bm25f/tagidf/quality.")


rerank = LLMReranker().rerank
