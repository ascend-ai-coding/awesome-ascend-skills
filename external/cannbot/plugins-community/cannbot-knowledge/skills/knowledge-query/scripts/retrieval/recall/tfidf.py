# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

"""recall.tfidf — lightweight TF-IDF sparse recall (no field weighting / no BM
saturation); alias-aware. A simpler, faster alternative to bm25.
"""
from retrieval.recall.base import Recall
from retrieval.search import _idf
from retrieval.tokenizer import tokenize


def _term_scores(idx, term, candidates):
    idf = _idf(term, idx)
    scores = {}
    for doc_id in idx["_docs_with"].get(term, ()):
        if candidates is not None and doc_id not in candidates:
            continue
        term_fields = idx.get("_pt", {}).get(term, {}).get(doc_id, {})
        scores[doc_id] = idf * sum(term_fields.values())
    return scores


class Tfidf(Recall):
    method = "tfidf"

    def recall(self, idx, spec):
        cands = self._candidates(spec)
        k = spec.get("k", 50)
        total = {}
        for qq in self._queries(spec):
            for term in set(tokenize(qq)):
                for did, score in _term_scores(idx, term, cands).items():
                    total[did] = total.get(did, 0.0) + score
        return [self._hit(idx["docs"][d], total.get(d, 0.0)) for d in self._topk(total, idx, k)]


recall = Tfidf().recall
