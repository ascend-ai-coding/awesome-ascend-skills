# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

"""rerank.tagidf — re-score hits by tag-IDF similarity to a seed card's tags
(spec['seed']) or, absent a seed, to the query terms treated as tags.
"""
import math

from retrieval.config import SCORE_PREC
from retrieval.rerank.base import Rerank
from retrieval.tokenizer import tokenize


class TagidfRerank(Rerank):

    def rerank(self, idx, query, hits, spec):
        by_path = self._by_path(idx)
        global_tags = set(idx["meta"]["global_tags"])
        n = idx["meta"]["card_count"]
        tag_df = idx["tag_df"]

        def tidf(g):
            dfc = tag_df.get(g, 0)
            return math.log((n - dfc + 0.5) / (dfc + 0.5) + 1.0)

        seed = spec.get("seed")
        if seed and seed in by_path:
            ref = {g for g in idx["docs"][by_path[seed]]["tags"] if g not in global_tags}
        else:
            ref = set(tokenize(query or ""))
        out = []
        for h in hits:
            did = by_path.get(h["path"])
            dtags = set(idx["docs"][did]["tags"]) if did is not None else set(h.get("tags", []))
            shared = (ref & dtags) - global_tags
            score = round(sum(tidf(g) for g in shared), SCORE_PREC)
            nh = dict(h)
            nh["score"] = score
            out.append(nh)
        return self._reorder(out)


rerank = TagidfRerank().rerank
