# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

"""retrieval.facets — path-derived facets + okf.v1/legacy type->kind."""
import re

from retrieval.config import KNOWN_KINDS, TYPE_KIND


def derive_facets(path):
    segs = path[:-3].split("/")            # drop .md
    bundle = segs[0]
    mid = segs[1:-1]
    stem = segs[-1]
    reldir = "/".join(mid)
    section = mid[0] if mid else ""
    category = mid[-1] if mid else ""
    base = re.sub(r"_\d+$", "", stem.lower())
    base = re.sub(r"(接口|_tiling|（[^）]*）|\([^)]*\))$", "", base)
    return bundle, reldir, section, category, (base or stem.lower())


def kind_of(type_raw, kind_raw=""):
    if kind_raw in KNOWN_KINDS:
        return kind_raw
    return TYPE_KIND.get(type_raw, "unknown")
