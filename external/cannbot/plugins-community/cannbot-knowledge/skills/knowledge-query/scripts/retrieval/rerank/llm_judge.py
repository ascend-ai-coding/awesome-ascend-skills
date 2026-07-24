# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

"""rerank.llm-judge — LLM-as-judge rerank, three paths (priority order):

  1. --verdicts <file>   : deterministic reorder by precomputed {path: score}. Unchanged.
  2. auto (Claude Code SDK): no verdicts -> one listwise judge call (shared retrieval.judge),
     fingerprint-cached. This is the "implemented" path.
  3. prepare-judge       : emit material for an agent to judge in-context, write verdicts,
     then re-run with (1). Zero extra model cost. `prepare()` is UNCHANGED (its displayed
     `key` keeps the legacy formula — it's just the manual-flow id, separate from the auto
     path's internal cache key in retrieval.judge).
"""
import hashlib
import json

from retrieval.errors import NotConfigured
from retrieval.judge import _candidate_item
from retrieval.output import emit_stdout
from retrieval.rerank.base import Rerank

PROMPT_VERSION = "1"
JUDGE_POLICY = "relevance"


def _key(query, hits):
    h = hashlib.sha1()
    h.update(("v=%s;policy=%s;q=%s;" % (PROMPT_VERSION, JUDGE_POLICY, query or "")).encode("utf-8"))
    for p in sorted(x["path"] for x in hits):
        h.update((p + "\n").encode("utf-8"))
    return h.hexdigest()


def prepare(idx, query, hits, material="header"):
    by_path = {d["path"]: i for i, d in enumerate(idx["docs"])}
    cands = []
    for hh in hits:
        p = hh["path"]
        cands.append(_candidate_item(idx, by_path, p, material))
    payload = {
        "task": "Rank candidates by relevance to the query; output a verdicts JSON "
                "{path: score} (higher = more relevant). Judge from the contents below.",
        "query": query, "prompt_version": PROMPT_VERSION, "judge_policy": JUDGE_POLICY,
        "material": material, "key": _key(query, hits), "candidates": cands,
    }
    emit_stdout(json.dumps(payload, ensure_ascii=False, indent=2))


class LLMJudge(Rerank):
    def __init__(self):
        self._clients = {}

    def rerank(self, idx, query, hits, spec):
        from retrieval.judge import apply_scores
        vfile = spec.get("verdicts")
        if vfile:                                # (1) precomputed, deterministic
            with open(vfile, encoding="utf-8") as verdict_file:
                v = json.load(verdict_file)
            scores = v.get("scores", v) if isinstance(v, dict) else {}
            return apply_scores(hits, scores, "llm-judge")
        if not hits:
            return []
        from retrieval.judge import judge_listwise   # (2) auto Claude Code SDK
        client = self._llm(spec.get("llm_model"))
        scores, whys = judge_listwise(client, idx, query, hits, policy=JUDGE_POLICY,
                                      material=spec.get("material", "header"))
        return apply_scores(hits, scores, "llm-judge", whys)

    def _llm(self, model):
        if model not in self._clients:
            from retrieval.llm import LLMClient
            self._clients[model] = LLMClient(model=model)
        return self._clients[model]


rerank = LLMJudge().rerank
