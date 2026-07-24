# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

"""retrieval.rerank.base — shared base for rerank modules.

Every rerank is a `Rerank` subclass exposing `rerank(idx, query, hits, spec) -> [Hit]`;
modules export a side-effect-free singleton's bound method (`rerank = Foo().rerank`).

Helpers only, never policy: `_by_path` index, a DEFAULT `_reorder` (-score, path), and
`_relabel` to append a method suffix. Subclasses with a custom ordering (quality:
quality_score/severity-first) sort inline and do NOT call `_reorder`, so golden outputs
are preserved.
"""
from abc import ABC, abstractmethod


class Rerank(ABC):

    @staticmethod
    def _relabel(method, suffix):
        return (method + ">" + suffix).lstrip(">")

    @abstractmethod
    def rerank(self, idx, query, hits, spec):
        ...

    # --- shared helpers ---
    def _by_path(self, idx):
        return {d["path"]: i for i, d in enumerate(idx["docs"])}

    def _reorder(self, hits):
        hits.sort(key=lambda h: (-h["score"], h["path"]))
        return hits
