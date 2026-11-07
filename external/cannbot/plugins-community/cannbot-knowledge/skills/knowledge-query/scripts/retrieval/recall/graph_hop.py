# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

"""recall.graph — multi-hop graph recall from a seed card (v3 LLM-judged edges via
retrieval.graph._load_graph). Returns hits with hop/edge_type/why; score = edge weight
/ hop. Empty if no graph. Sorts hop-first (overrides the default -score ordering).
"""
from retrieval.graph import _load_graph, _reachable_nodes
from retrieval.recall.base import Recall


class GraphHop(Recall):
    method = "graph"

    def recall(self, idx, spec):
        seed = spec.get("seed")
        hops = spec.get("hops", 1)
        k = spec.get("k", 50)
        g = _load_graph()
        if g is None or not seed:
            return []
        if seed not in g["adj"] and seed not in g["nodes"]:
            return []
        reach = _reachable_nodes(g, seed, hops, None)
        by_path = {d["path"]: d for d in idx["docs"]}
        doc_ids = {d["path"]: did for did, d in enumerate(idx["docs"])}
        candidates = self._candidates(spec)
        hits = []
        for o, (hop, e) in reach.items():
            if candidates is not None and o in doc_ids and doc_ids[o] not in candidates:
                continue
            d = by_path.get(o) or {"path": o, **{kk: g["nodes"].get(o, {}).get(kk, "")
                                                 for kk in ("title", "kind", "bundle")}, "tags": []}
            h = self._hit(d, e.get("weight", 0.0) / hop, e.get("why", ""))
            h["hop"], h["edge_type"] = hop, e["type"]
            hits.append(h)
        self._reorder(hits)
        return hits[:k]

    def _reorder(self, hits):                 # hop-first, then -score, then path
        hits.sort(key=lambda h: (h["hop"], -h["score"], h["path"]))
        return hits


recall = GraphHop().recall
