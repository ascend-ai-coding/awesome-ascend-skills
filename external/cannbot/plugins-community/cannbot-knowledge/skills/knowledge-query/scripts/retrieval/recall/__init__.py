# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

"""retrieval.recall — pluggable recall modules. Each exposes
`recall(idx, spec) -> list[Hit]`. Registered methods: bm25, tfidf, tagtype,
graph, dense(Embedding API/hashing). spec is a dict (queries/candidates/alias_map/k/tags/...).
"""
from retrieval.recall import bm25, tfidf, tagtype, graph_hop, dense

REGISTRY = {
    "bm25": bm25.recall,
    "tfidf": tfidf.recall,
    "tagtype": tagtype.recall,
    "graph": graph_hop.recall,
    "dense": dense.recall,
}
