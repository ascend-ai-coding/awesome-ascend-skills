# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

"""rerank.quality — sort hits by runbook evidence quality: quality_score desc,
then severity (high>medium>low), then incoming score. For optimization-practice.
Custom ordering: sorts inline, does NOT use the base default _reorder.
"""
from retrieval.config import SCORE_PREC
from retrieval.rerank.base import Rerank

_SEV = {"high": 3, "medium": 2, "low": 1, "": 0}


class QualityRerank(Rerank):

    def rerank(self, idx, query, hits, spec):
        by_path = self._by_path(idx)
        out = []
        for h in hits:
            d = idx["docs"][by_path[h["path"]]] if h["path"] in by_path else {}
            qs = d.get("quality_score")
            nh = dict(h)
            nh["quality_score"] = qs
            nh["severity"] = d.get("severity", "")
            nh["_sort"] = (qs if qs is not None else -1.0, _SEV.get(d.get("severity", ""), 0),
                           round(h.get("score", 0.0), SCORE_PREC))
            out.append(nh)
        out.sort(key=lambda h: (-h["_sort"][0], -h["_sort"][1], -h["_sort"][2], h["path"]))
        for h in out:
            h.pop("_sort", None)
        return out


rerank = QualityRerank().rerank
