# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

"""recall.bm25 — BM25F sparse recall (reuses the legacy scoring primitives),
alias-aware query expansion by default.
"""
from retrieval.recall.base import Recall
from retrieval.search import _score_one_query


class Bm25(Recall):
    method = "bm25"

    def recall(self, idx, spec):
        cands = self._candidates(spec)
        k = spec.get("k", 50)
        total = {}
        for q in self._queries(spec):
            sc, _, _ = _score_one_query(q, idx, cands)
            for did, s in sc.items():
                total[did] = total.get(did, 0.0) + s
        return [self._hit(idx["docs"][d], total.get(d, 0.0)) for d in self._topk(total, idx, k)]


recall = Bm25().recall
