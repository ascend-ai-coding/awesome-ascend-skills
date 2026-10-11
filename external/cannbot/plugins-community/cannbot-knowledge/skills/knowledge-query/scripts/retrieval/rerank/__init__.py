# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

"""retrieval.rerank — pluggable rerank modules. Each exposes
`rerank(idx, query, hits, spec) -> list[Hit]`. Registered: bm25f, tagidf,
quality, reranker(LLMReranker via Claude Code SDK), llm-judge (prepare/apply).
"""
from retrieval.rerank import bm25f, tagidf, quality, reranker_model, llm_judge

REGISTRY = {
    "bm25f": bm25f.rerank,
    "tagidf": tagidf.rerank,
    "quality": quality.rerank,
    "reranker": reranker_model.rerank,
    "llm-judge": llm_judge.rerank,
}
