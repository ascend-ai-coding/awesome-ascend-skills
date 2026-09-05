# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

"""retrieval.scope — candidate filtering by facets + scope-prefix normalization.

`_filter_candidates` is the legacy (frozen) filter. `facet_filter` is the Stage-2
filter: scope-prefix + bundle/kind/category/section/type/tags/paradigm/severity/
confidence/status, with alias-set expansion on values and default stub exclusion.
"""
from retrieval.aliases import expand_values
from retrieval.platforms import is_950_only, platform_context

REFERENCE_SCOPE = "__reference_root__"


def _filter_candidates(idx, bundle, kind, category, section):
    if not any([bundle, kind, category, section]):
        return None
    keep = set()
    for doc_id, d in enumerate(idx["docs"]):
        if bundle and d["bundle"] != bundle:
            continue
        if kind and d["kind"] != kind:
            continue
        if category and d["category"] != category:
            continue
        if section and d["section"] != section:
            continue
        keep.add(doc_id)
    return keep


def normalize_scope(s):
    """Accept `reference/asc-devkit/...`, `asc-devkit/...`, `ops/...`,
    `runbooks/...` -> doc-id prefix. reference/ prefix is stripped (reference
    doc-ids carry no root prefix); ops/runbooks keep their prefix.
    """
    if not s:
        return ""
    s = s.strip().lstrip("/")
    if s.lower() in ("all", "*"):
        return ""
    if s in ("reference", "reference/"):
        return REFERENCE_SCOPE
    if s.startswith("reference/"):
        s = s[len("reference/"):]
    return s


def display_scope(scope):
    return "reference/" if scope == REFERENCE_SCOPE else scope


def _status_matches(doc, status):
    value = doc.get("status")
    if status == "active":
        return value != "stub"
    if status in ("verified", "stub"):
        return value == status
    return True


def _scope_matches(path, scope):
    if scope == REFERENCE_SCOPE:
        return not path.startswith(("ops/", "runbooks/"))
    return not scope or path.startswith(scope)


def _metadata_matches(doc, filters):
    checks = (
        (filters["bundle"], doc["bundle"]),
        (filters["kind"], doc["kind"]),
        (filters["category"], doc["category"]),
        (filters["section"], doc["section"]),
        (filters["severity"], doc.get("severity", "")),
        (filters["confidence"], doc.get("confidence", "")),
    )
    return all(not expected or actual == expected for expected, actual in checks)


def _facet_matches(doc, filters, tag_sets, paradigm_set, platform_policy):
    if platform_policy and platform_policy["enabled"] and is_950_only(doc):
        return False
    if not _status_matches(doc, filters["status"]):
        return False
    if not _scope_matches(doc["path"], filters["scope"]):
        return False
    if not _metadata_matches(doc, filters):
        return False
    ctype = filters["ctype"]
    if ctype and doc.get("type_raw", "").lower() != ctype.lower():
        return False
    doc_paradigms = {value.lower() for value in doc.get("paradigms", [])}
    if paradigm_set is not None and not paradigm_set & doc_paradigms:
        return False
    doc_tags = {value.lower() for value in doc.get("tags", [])}
    return not tag_sets or all(tag_set & doc_tags for tag_set in tag_sets)


def facet_filter(idx, *, scope=None, bundle=None, kind=None, category=None,
                 section=None, ctype=None, tags=None, paradigm=None,
                 severity=None, confidence=None, status="active", alias_map=None,
                 platform=None):
    """Return a candidate doc_id set (AND across facets), or None if nothing active.
    status: 'active' (default, excludes stub) | 'verified' | 'stub' | 'all'.
    """
    alias_map = alias_map or {}
    scope = normalize_scope(scope) if scope else ""
    tag_sets = [expand_values([t], alias_map) for t in (tags or [])]   # AND across, OR within
    paradigm_set = expand_values([paradigm], alias_map) if paradigm else None
    active = any([scope, bundle, kind, category, section, ctype, tags, paradigm,
                  severity, confidence, platform]) or status != "all"
    if not active:
        return None
    filters = {
        "scope": scope,
        "bundle": bundle,
        "kind": kind,
        "category": category,
        "section": section,
        "ctype": ctype,
        "severity": severity,
        "confidence": confidence,
        "status": status,
    }
    platform_policy = platform_context(explicit=platform) if platform else None
    return {
        doc_id
        for doc_id, doc in enumerate(idx["docs"])
        if _facet_matches(doc, filters, tag_sets, paradigm_set, platform_policy)
    }
