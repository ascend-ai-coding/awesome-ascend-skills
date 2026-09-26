# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

"""recall.tagtype — match by tags/type/paradigm (structure), score by tag-IDF of
shared tags. Candidates are usually pre-narrowed by facet_filter; this ranks them.
Used by similar-example / optimization-practice patterns.
"""
import math

from retrieval.aliases import expand_values
from retrieval.recall.base import Recall


class Tagtype(Recall):
    method = "tagtype"

    def recall(self, idx, spec):
        cands = self._candidates(spec)
        k = spec.get("k", 50)
        amap = spec.get("alias_map") or {}
        n = idx["meta"]["card_count"]
        tag_df = idx["tag_df"]
        req = set()
        for t in (spec.get("tags") or []):
            req |= expand_values([t], amap)

        def tidf(g):
            dfc = tag_df.get(g, 0)
            return math.log((n - dfc + 0.5) / (dfc + 0.5) + 1.0)

        docids = cands if cands is not None else range(len(idx["docs"]))
        hits = []
        for did in docids:
            d = idx["docs"][did]
            if req:
                shared = {g for g in d.get("tags", []) if g.lower() in req}
                if not shared:
                    continue
                score = sum(tidf(g) for g in shared)
            else:
                score = 1.0
            hits.append(self._hit(d, score))
        self._reorder(hits)
        return hits[:k]


recall = Tagtype().recall
