# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

"""Path helpers for doc-id output fields.

The retrieval protocol keeps doc-id as the stable identifier. `local_path` is
only a convenience for Agents that need to read the card file directly after a
query result has selected the evidence.
"""
from __future__ import annotations

import os

from retrieval.cards import id_to_path


def is_doc_id(value):
    return isinstance(value, str) and value.endswith(".md") and not os.path.isabs(value)


def local_path_for_doc(doc_id):
    if not is_doc_id(doc_id):
        return ""
    return os.path.abspath(id_to_path(doc_id))


def attach_local_paths(obj):
    if isinstance(obj, list):
        for item in obj:
            attach_local_paths(item)
        return obj
    if not isinstance(obj, dict):
        return obj

    path = obj.get("path")
    if is_doc_id(path):
        obj.setdefault("local_path", local_path_for_doc(path))

    source = obj.get("source")
    if is_doc_id(source):
        obj.setdefault("source_local_path", local_path_for_doc(source))

    variants = obj.get("variants")
    if isinstance(variants, list):
        variant_paths = [local_path_for_doc(v) for v in variants if is_doc_id(v)]
        if variant_paths:
            obj.setdefault("variant_local_paths", variant_paths)

    suggested_get = obj.get("suggested_get")
    if isinstance(suggested_get, list):
        suggested_paths = [local_path_for_doc(v) for v in suggested_get if is_doc_id(v)]
        if suggested_paths:
            obj.setdefault("suggested_get_local_paths", suggested_paths)

    for value in obj.values():
        attach_local_paths(value)
    return obj
