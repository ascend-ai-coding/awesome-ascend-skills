# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

"""retrieval.recall.base — shared base for recall modules.

Every recall is a `Recall` subclass exposing `recall(idx, spec) -> [Hit]`; modules
export a side-effect-free singleton's bound method (`recall = Foo().recall`) so the
REGISTRY and instantiation stay model-dependency-free (heavy libs lazy-import inside
the model backends only).

The base offers *helpers*, not policy: candidate access, alias-expanded queries,
hit construction, and a DEFAULT `_reorder` (-score, path). Subclasses with a different
ordering (graph: hop-first) override `_reorder` / sort inline — the base never forces
one sort, so deterministic golden outputs are preserved.
"""
from abc import ABC, abstractmethod

from retrieval.aliases import expand_query
from retrieval.hit import make_hit


class Recall(ABC):
    method = ""

    @abstractmethod
    def recall(self, idx, spec):
        ...

    # --- shared helpers (opt-in; subclasses use what they need) ---
    def _candidates(self, spec):
        return spec.get("candidates")

    def _queries(self, spec):
        """Alias-expanded query strings (per-query word-level expansion), default on."""
        amap = spec.get("alias_map") or {}
        qs = spec.get("queries") or []
        if spec.get("alias", True):
            qs = [expand_query(q, amap) for q in qs]
        return qs

    def _hit(self, d, score, why=""):
        return make_hit(d, score, self.method, why)

    def _topk(self, scored, idx, k):
        """doc-id->score map -> top-k doc-ids by (-score, path)."""
        return sorted(scored, key=lambda d: (-scored[d], idx["docs"][d]["path"]))[:k]

    def _reorder(self, hits):
        hits.sort(key=lambda h: (-h["score"], h["path"]))
        return hits
