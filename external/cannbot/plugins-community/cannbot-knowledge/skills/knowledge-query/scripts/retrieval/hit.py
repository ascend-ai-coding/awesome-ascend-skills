# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

"""retrieval.hit — the uniform Hit contract + multi-route merge.

A Hit is a plain dict so it serializes to JSON directly:
  {path, score, method, title, kind, bundle, tags, why}
Recall produces [Hit]; rerank consumes/returns [Hit]; pipeline merges across routes.
"""
from retrieval.config import SCORE_PREC


def make_hit(d, score, method, why=""):
    return {
        "path": d["path"], "score": round(score, SCORE_PREC), "method": method,
        "title": d.get("title", ""), "kind": d.get("kind", ""),
        "bundle": d.get("bundle", ""), "tags": d.get("tags", []),
        "platforms": d.get("platforms", []), "why": why,
    }


def _merge_hit(aggregate, hit, mode):
    path = hit["path"]
    if path not in aggregate:
        merged = dict(hit)
        merged["_methods"] = {hit["method"]}
        aggregate[path] = merged
        return
    merged = aggregate[path]
    if mode == "sum":
        merged["score"] += hit["score"]
    else:
        merged["score"] = max(merged["score"], hit["score"])
    merged["_methods"].add(hit["method"])
    if hit.get("why") and not merged.get("why"):
        merged["why"] = hit["why"]


# Merge several Hit lists by path; scores use sum/max and methods form a union.
def merge_hits(hitlists, mode="sum"):
    agg = {}
    for hits in hitlists:
        for h in hits:
            _merge_hit(agg, h, mode)
    out = []
    for a in agg.values():
        a["score"] = round(a["score"], SCORE_PREC)
        a["method"] = ",".join(sorted(a.pop("_methods")))
        out.append(a)
    out.sort(key=lambda h: (-h["score"], h["path"]))
    return out
