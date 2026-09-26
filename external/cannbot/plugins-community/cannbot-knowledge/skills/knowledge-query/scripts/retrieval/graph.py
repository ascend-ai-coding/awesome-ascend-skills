# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

"""retrieval.graph — multi-hop neighbors over the knowledge graph (node-ids ==
retrieval doc-ids). Uses the **v3 LLM-judged edges** (the same edges injected into
cards' `# 相关` block), computed FRESH from the committed graph/edge_judgments.json
(zero new LLM, never stale). Falls back to the prebuilt .build/okf.graph.json
artifact, then to tag-IDF `related`.
"""
import json
import logging
import os
import sys

from retrieval.config import GRAPH_JSON, ROOT, SCORE_PREC, _PART_OF
from retrieval.errors import CliError
from retrieval.index import load_index
from retrieval.output import emit_stderr, emit_stdout
from retrieval.paths import attach_local_paths
from retrieval.platforms import is_950_only, platform_context, platform_filter_output
from retrieval.read import cmd_related

LOGGER = logging.getLogger(__name__)


def _adj(edges):
    adj = {}
    for e in edges:
        adj.setdefault(e["source"], []).append(e)
        adj.setdefault(e["target"], []).append(e)
    return adj


def _load_graph():
    # Prefer fresh v3 (LLM-judged) edges from the committed judgments cache —
    # always in sync with the current cards, zero LLM. (ops-knowledge-ingest owns okf_graph;
    # we import it read-only and reuse load_nodes + v3_edges.)
    try:
        # Import the sibling skill bundled in this plugin. ROOT is the target
        # knowledge-base root and must not be assumed to contain skill code.
        ing = os.path.abspath(os.path.join(
            os.path.dirname(__file__), "..", "..", "..", "ops-knowledge-ingest", "scripts"
        ))
        if ing not in sys.path:
            sys.path.insert(0, ing)
        import okf_graph as G
        nodes = G.load_nodes()
        return {"nodes": nodes, "adj": _adj(G.v3_edges(nodes))}
    except Exception:
        LOGGER.debug("failed to load fresh graph judgments; trying the artifact", exc_info=True)
    # Fallback: prebuilt artifact (may be stale w.r.t. current cards).
    if os.path.exists(GRAPH_JSON):
        with open(GRAPH_JSON, encoding="utf-8") as graph_file:
            g = json.load(graph_file)
        return {"nodes": {n["id"]: n for n in g.get("nodes", [])}, "adj": _adj(g.get("edges", []))}
    return None


def _other_endpoint(edge, node, type_filter, seen):
    if edge["type"] == _PART_OF:
        return None
    if type_filter and edge["type"] not in type_filter:
        return None
    other = edge["target"] if edge["source"] == node else edge["source"]
    return None if other in seen else other


def _advance_frontier(graph, frontier, type_filter, seen, reach, *, hop):
    next_frontier = []
    for node in frontier:
        for edge in graph["adj"].get(node, []):
            other = _other_endpoint(edge, node, type_filter, seen)
            if other is None:
                continue
            seen.add(other)
            reach[other] = (hop, edge)
            next_frontier.append(other)
    return next_frontier


def _reachable_nodes(graph, path, hops, type_filter):
    seen = {path}
    reach = {}
    frontier = [path]
    for hop in range(1, hops + 1):
        frontier = _advance_frontier(graph, frontier, type_filter, seen, reach, hop=hop)
    return reach


def _neighbor_rows(graph, reach, bundle, platform_policy, by_path):
    rows = []
    for other, (hop, edge) in reach.items():
        node = graph["nodes"].get(other, {})
        if bundle and node.get("bundle") != bundle:
            continue
        if platform_policy["enabled"] and is_950_only(by_path.get(other, {})):
            continue
        rows.append({
            "path": other, "title": node.get("title", ""), "kind": node.get("kind", ""),
            "description": node.get("description", ""), "bundle": node.get("bundle", ""),
            "edge_type": edge["type"], "why": edge.get("why", ""), "hop": hop,
            "weight": round(edge.get("weight", 0.0), SCORE_PREC),
        })
    rows.sort(key=lambda row: (row["hop"], -row["weight"], row["path"]))
    return rows


def cmd_neighbors(path, hops, types, bundle, k, *, platform=None):
    platform_policy = platform_context(explicit=platform) if platform else platform_context()
    g = _load_graph()
    if g is None:
        emit_stderr("no graph judgments/artifact — degrading to tag-IDF related; "
                    "run okf_graph (candidates→judge→inject) first")
        cmd_related(path, bundle, False, platform=platform)  # graceful fallback
        return None
    if path not in g["adj"] and path not in g["nodes"]:
        raise CliError("unknown graph node: %s (build/inject the graph first)" % path)
    type_filter = set(types.split(",")) if types else None
    reach = _reachable_nodes(g, path, hops, type_filter)
    by_path = {doc["path"]: doc for doc in load_index()["docs"]} if platform_policy["enabled"] else {}
    rows = _neighbor_rows(g, reach, bundle, platform_policy, by_path)
    selected = rows[:k]
    emit_stdout(json.dumps(attach_local_paths({
        "source": path,
        "neighbors": selected,
        "platform_filter": platform_filter_output(platform_policy, len(selected)),
    }), ensure_ascii=False, indent=2))
    return None
