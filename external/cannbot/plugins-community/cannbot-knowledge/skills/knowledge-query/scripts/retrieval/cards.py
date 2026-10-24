# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

"""retrieval.cards — card reading, frontmatter parse, field extraction, paths.

Frontmatter parsed with a narrow stdlib regex parser (no PyYAML); inline-list
fields (tags) handled like okf_graph. okf:related managed block stripped before
indexing so graph injection never feeds back into recall.
"""
import os
import re

from retrieval.config import _REL_RE, _SIG_HEADINGS, CONTENT_ROOTS, REFERENCE, ROOT

# frontmatter keys parsed as lists (inline `[a, b]` or block `- a`). Legacy only
# read `tags`; aliases/paradigms are added for Stage-2 facets (legacy ignores them).
_LIST_KEYS = ("tags", "aliases", "paradigms", "platforms")


def _parse_front(text):
    lines = text.split("\n")
    if not lines or lines[0].strip() != "---":
        return {}, text
    end = None
    for i in range(1, len(lines)):
        if lines[i].strip() == "---":
            end = i
            break
    if end is None:
        return {}, text
    body = "\n".join(lines[end + 1:])
    if body.startswith("\n"):
        body = body[1:]
    return _parse_fm(lines[1:end]), body


def _list_value(fm_lines, index, value):
    if value.startswith("["):
        inner = value[1:value.rfind("]")] if "]" in value else value[1:]
        return [item.strip().strip("'\"") for item in inner.split(",") if item.strip()], index + 1
    items = []
    index += 1
    while index < len(fm_lines) and fm_lines[index].lstrip().startswith("- "):
        items.append(fm_lines[index].lstrip()[2:].strip().strip("'\""))
        index += 1
    return items, index


def _parse_fm(fm_lines):
    fm = {}
    index = 0
    while index < len(fm_lines):
        match = re.match(r"^([A-Za-z_][\w-]*):\s*(.*)$", fm_lines[index])
        if not match:
            index += 1
            continue
        key, value = match.group(1), match.group(2).strip()
        if key in _LIST_KEYS:
            fm[key], index = _list_value(fm_lines, index, value)
            continue
        fm[key] = value.strip("'\"")
        index += 1
    return fm


def strip_related(body):
    return _REL_RE.sub("", body)


def _headings(body):
    return " ".join(ln.lstrip("#").strip()
                    for ln in body.split("\n") if ln.lstrip().startswith("#"))


def _signatures(body):
    """Identifiers inside fenced code blocks under a signature heading."""
    out, in_sig, in_code = [], False, False
    for ln in body.split("\n"):
        st = ln.strip()
        if st.startswith("#"):
            in_sig = any(h in st for h in _SIG_HEADINGS)
            continue
        if st.startswith("```") or st.startswith("~~~"):
            in_code = not in_code
            continue
        if in_sig and in_code:
            out.append(ln)
    return " ".join(out)


def _front_block(raw):
    """Return the raw frontmatter block (between the first two --- lines)."""
    lines = raw.split("\n")
    if not lines or lines[0].strip() != "---":
        return ""
    for i in range(1, len(lines)):
        if lines[i].strip() == "---":
            return "\n".join(lines[:i + 1])
    return ""


def _concept_paths_under(name, root):
    paths = []
    for current_root, _, files in os.walk(root):
        for filename in files:
            if not filename.endswith(".md") or filename == "index.md":
                continue
            rel = os.path.relpath(os.path.join(current_root, filename), root).replace(os.sep, "/")
            paths.append(rel if name == "reference" else "%s/%s" % (name, rel))
    return paths


def concept_paths():
    """All concept cards across content roots, excluding every index.md. Ids match
    okf_graph node ids: reference cards keep the bundle prefix (relative to
    reference/), ops/runbooks carry their root prefix (relative to repo root).
    """
    out = []
    for name, root in CONTENT_ROOTS:
        if not os.path.isdir(root):
            continue
        out.extend(_concept_paths_under(name, root))
    return sorted(out)


def id_to_path(doc_id):
    """Resolve a doc-id (graph-node scheme) to its file path on disk."""
    if doc_id.startswith("ops/") or doc_id.startswith("runbooks/"):
        return os.path.join(ROOT, doc_id)
    return os.path.join(REFERENCE, doc_id)
