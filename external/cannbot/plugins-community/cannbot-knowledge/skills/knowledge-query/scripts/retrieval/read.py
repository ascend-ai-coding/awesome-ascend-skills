# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

"""retrieval.read — card injection (`get`) + tag-IDF nearest neighbors (`related`)."""
import json
import math
import os

from retrieval.config import SCORE_PREC
from retrieval.cards import _front_block, _parse_front, id_to_path, strip_related
from retrieval.errors import CliError
from retrieval.index import load_index
from retrieval.output import emit_stdout
from retrieval.paths import attach_local_paths, local_path_for_doc
from retrieval.platforms import is_950_only, platform_context, platform_filter_output


def _limit_text(text, max_chars):
    if not max_chars or max_chars <= 0 or len(text) <= max_chars:
        return text
    return text[: max_chars - 3].rstrip() + "..."


def _section_text(body, heading):
    if not heading:
        return body
    target = heading.strip().lower().lstrip("#").strip()
    lines = body.splitlines()
    out = []
    capture = False
    level = None
    for line in lines:
        stripped = line.lstrip()
        if stripped.startswith("#"):
            cur_level = len(stripped) - len(stripped.lstrip("#"))
            title = stripped.lstrip("#").strip().lower()
            if capture and level is not None and cur_level <= level:
                break
            if title == target or target in title:
                capture = True
                level = cur_level
        if capture:
            out.append(line)
    return "\n".join(out) if out else ""


def _neighbor_metadata(base, filename):
    with open(os.path.join(base, filename), encoding="utf-8") as card_file:
        frontmatter, _ = _parse_front(card_file.read())
    return frontmatter.get("title", filename[:-3]), frontmatter.get("description", "")


def _neighbors(path):
    d = os.path.dirname(path)
    base = os.path.dirname(id_to_path(path))
    out = []
    if os.path.isdir(base):
        for f in sorted(os.listdir(base)):
            if f.endswith(".md") and f != "index.md" and f != os.path.basename(path):
                title, description = _neighbor_metadata(base, f)
                out.append((d + "/" + f, title, description))
    return out


def cmd_get(paths, with_related, max_chars=None, section=None, neighbor_limit=20):
    chunks = []
    for path in paths:
        full = id_to_path(path)
        if not os.path.exists(full):
            chunks.append("## NOT FOUND: %s" % path)
            continue
        with open(full, encoding="utf-8") as card_file:
            raw = card_file.read()
        fm, body = _parse_front(raw)
        if not with_related:
            body = strip_related(body).rstrip() + "\n"
        if section:
            narrowed = _section_text(body, section)
            body = narrowed if narrowed else "## SECTION NOT FOUND: %s\n" % section
        body = _limit_text(body.rstrip(), max_chars)
        block = [
            "===== %s =====" % path,
            "local_path: %s" % local_path_for_doc(path),
            _front_block(raw),
            "",
            body,
            "",
        ]
        block.append("resource: %s" % fm.get("resource", "(none)"))
        nb = _neighbors(path) if neighbor_limit != 0 else []
        if nb:
            block.append("\nsame-dir neighbors:")
            shown = nb if neighbor_limit is None or neighbor_limit < 0 else nb[:neighbor_limit]
            for p, t, _ in shown:
                block.append("- %s — %s" % (p, t))
            if len(shown) < len(nb):
                block.append("- ... %d more omitted; use --neighbor-limit %d or neighbors for graph context"
                             % (len(nb) - len(shown), len(nb)))
        chunks.append("\n".join(block))
    emit_stdout("\n\n".join(chunks))


def cmd_related(path, bundle, same_category, platform=None):
    idx = load_index()
    platform_policy = platform_context(explicit=platform) if platform else platform_context()
    by_path = {d["path"]: i for i, d in enumerate(idx["docs"])}
    if path not in by_path:
        raise CliError("unknown concept: %s" % path)
    src = idx["docs"][by_path[path]]
    global_tags = set(idx["meta"]["global_tags"])
    src_tags = [g for g in src["tags"] if g not in global_tags]
    n = idx["meta"]["card_count"]
    tag_df = idx["tag_df"]

    def tag_idf(g):
        dfc = tag_df.get(g, 0)
        return math.log((n - dfc + 0.5) / (dfc + 0.5) + 1.0)

    sims = []
    for i, d in enumerate(idx["docs"]):
        if i == by_path[path]:
            continue
        if bundle and d["bundle"] != bundle:
            continue
        if same_category and d["category"] != src["category"]:
            continue
        if platform_policy["enabled"] and is_950_only(d):
            continue
        shared = (set(src_tags) & set(d["tags"])) - global_tags
        if not shared:
            continue
        sim = round(sum(tag_idf(g) for g in shared), SCORE_PREC)
        sims.append((sim, d["path"], d["title"], d["description"], sorted(shared)))
    sims.sort(key=lambda x: (-x[0], x[1]))
    out = [{"path": p, "title": t, "description": desc, "score": s, "shared_tags": sh}
           for s, p, t, desc, sh in sims[:20]]
    emit_stdout(json.dumps(attach_local_paths({
        "source": path,
        "related": out,
        "platform_filter": platform_filter_output(platform_policy, len(out)),
    }), ensure_ascii=False, indent=2))
