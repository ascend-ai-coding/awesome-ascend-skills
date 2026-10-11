# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

"""rerank.bm25f — deterministic re-score of candidate hits by BM25F vs the query."""
from retrieval.config import SCORE_PREC
from retrieval.rerank.base import Rerank
from retrieval.search import _score_one_query


class Bm25fRerank(Rerank):

    def rerank(self, idx, query, hits, spec):
        by_path = self._by_path(idx)
        cand = {by_path[h["path"]] for h in hits if h["path"] in by_path}
        sc, _, _ = _score_one_query(query or "", idx, cand)
        out = []
        for h in hits:
            did = by_path.get(h["path"])
            nh = dict(h)
            nh["score"] = round(sc.get(did, 0.0) if did is not None else 0.0, SCORE_PREC)
            out.append(nh)
        return self._reorder(out)


rerank = Bm25fRerank().rerank
