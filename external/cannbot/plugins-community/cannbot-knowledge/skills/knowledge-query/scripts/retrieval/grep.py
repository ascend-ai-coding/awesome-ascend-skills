# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

"""retrieval.grep — regex over card text, scoped. Defaults to body-only with the
okf:related block stripped (so it never matches graph-injected related titles).
Output is {path,title,line,snippet} (not bare lines), for easy follow-up `get`.
"""
import json
import os
import re

from retrieval.cards import (_front_block, _parse_front, concept_paths,
                             id_to_path, strip_related)
from retrieval.errors import CliError
from retrieval.output import emit_stdout
from retrieval.paths import attach_local_paths
from retrieval.platforms import is_950_only, platform_context, platform_filter_output
from retrieval.relevance import annotate_relevance
from retrieval.scope import normalize_scope


def _compile_pattern(pattern):
    try:
        return re.compile(pattern)
    except re.error as error:
        raise CliError("bad regex: %s" % error) from error


def _search_text(raw, body, only, with_related):
    if only == "frontmatter":
        return _front_block(raw)
    if only == "all":
        return raw if with_related else _front_block(raw) + "\n" + strip_related(body)
    return body if with_related else strip_related(body)


def _matches_in_card(path, regex, only, with_related, limit, *, platform_policy):
    with open(id_to_path(path), encoding="utf-8") as card_file:
        raw = card_file.read()
    frontmatter, body = _parse_front(raw)
    if platform_policy["enabled"] and is_950_only(frontmatter):
        return []
    text = _search_text(raw, body, only, with_related)
    title = frontmatter.get("title", os.path.basename(path)[:-3])
    matches = []
    for line_number, line in enumerate(text.split("\n"), 1):
        if not regex.search(line):
            continue
        matches.append({
            "path": path, "title": title, "line": line_number, "snippet": line.strip()[:200]
        })
        if len(matches) >= limit:
            break
    return matches


def grep_results(pattern, scope=None, only="body", with_related=False, k=50, *, platform=None):
    regex = _compile_pattern(pattern)
    pref = normalize_scope(scope) if scope else ""
    platform_policy = platform_context(explicit=platform) if platform else platform_context()
    matches = []
    for path in concept_paths():
        if pref and not path.startswith(pref):
            continue
        matches.extend(_matches_in_card(
            path, regex, only, with_related, k - len(matches), platform_policy=platform_policy
        ))
        if len(matches) >= k:
            break
    return matches


def cmd_grep(pattern, scope, only, with_related, k, *, platform=None):
    matches = grep_results(pattern, scope, only, with_related, k, platform=platform)
    annotate_relevance(matches, queries=[pattern])
    context = platform_context(explicit=platform) if platform else platform_context()
    emit_stdout(json.dumps(attach_local_paths({
        "matches": matches,
        "platform_filter": platform_filter_output(context, len(matches)),
    }), ensure_ascii=False, indent=2))
