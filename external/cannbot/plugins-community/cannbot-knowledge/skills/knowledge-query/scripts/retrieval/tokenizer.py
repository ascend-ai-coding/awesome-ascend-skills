# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

"""retrieval.tokenizer — deterministic tokenizer (build & query share it).

ASCII: lowercase + CamelCase/snake split, keep merged form (exact-name queries).
CJK: character bigrams. Zero-dependency, reproducible.
"""
from retrieval.config import _TOKEN_RE, _CAMEL_RE


def _ascii_tokens(s):
    low = s.lower()
    out = [low]                       # merged form (exact-name queries)
    for p in _CAMEL_RE.findall(s):    # CamelCase split; snake already run-split
        pl = p.lower()
        if pl and pl != low:
            out.append(pl)
    return out


def _cjk_bigrams(s):
    if len(s) == 1:
        return [s]
    return [s[i:i + 2] for i in range(len(s) - 1)]


def tokenize(text):
    out = []
    for m in _TOKEN_RE.finditer(text or ""):
        s = m.group(0)
        if s[0].isascii() and (s[0].isalpha() or s[0].isdigit()):
            out.extend(_ascii_tokens(s))
        else:
            out.extend(_cjk_bigrams(s))
    return out
