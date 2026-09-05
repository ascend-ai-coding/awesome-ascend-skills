# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

"""retrieval.aliases — deterministic query expansion via the index's alias_map.

Single source of truth = card frontmatter `aliases` (same field okf_graph reads);
build derives a multi-valued map (alias -> sorted variant set). Expansion is
word-level (plus whole-query), idempotent. No external file, no YAML.
"""


def get_alias_map(idx):
    return idx.get("meta", {}).get("alias_map", {})


def expand_token(token, alias_map):
    """A token -> sorted set including its variants (idempotent)."""
    t = (token or "").strip().lower()
    if not t:
        return []
    return sorted({t} | set(alias_map.get(t, [])))


def expand_query(query, alias_map):
    """Augment a query string with alias variants of its words (+ whole string).
    Returns the augmented query string (original words preserved, variants appended).
    """
    words = query.split()
    extra = []
    keys = [w.lower() for w in words] + [query.strip().lower()]
    for kk in keys:
        for v in alias_map.get(kk, []):
            if v not in extra:
                extra.append(v)
    return query + ((" " + " ".join(extra)) if extra else "")


def expand_values(values, alias_map):
    """Expand a list of filter values into the union set of their variants."""
    out = set()
    for v in values or []:
        out.update(expand_token(v, alias_map))
    return out
