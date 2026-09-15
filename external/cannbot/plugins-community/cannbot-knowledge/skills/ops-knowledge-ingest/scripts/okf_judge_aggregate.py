#!/usr/bin/env python3
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

"""Aggregate the okf-judge-edges workflow output into graph/edge_judgments.json.

Usage: okf_judge_aggregate.py <workflow_output.json> [model_id]

Reads the workflow result (results:[{idx,kept:[{target_id,type,reason}]}]), maps idx→focal
id via .build/cand/manifest.json, and writes the committed judgment cache keyed by the
unordered pair of card content fingerprints (for fingerprint-based incremental reuse).
"""
import argparse
import importlib.util
import json
import logging
import os
import sys


def _load_script_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError("cannot load module %s from %s" % (name, path))
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
G = _load_script_module("cannbot_knowledge_okf_graph", os.path.join(SCRIPT_DIR, "okf_graph.py"))

VALID = {"exemplifies", "explains", "declares", "demonstrates", "same_topic", "related"}


def _emit(message, *args):
    """Write one unprefixed log record to stdout, preserving the CLI contract."""
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(logging.Formatter("%(message)s"))
    record = logging.LogRecord(__name__, logging.INFO, "", 0, message, args, None)
    handler.handle(record)
    handler.close()


def _aggregate_judgments(results, index_to_id, nodes, fingerprints):
    judgments, seen, dropped, bad = {}, set(), 0, 0
    kept_total = 0
    for result in results:
        focal = index_to_id.get(result.get("idx"))
        if focal is None or focal not in nodes:
            continue
        for kept in result.get("kept", []):
            target = kept.get("target_id")
            judgment_type = kept.get("type")
            reason = (kept.get("reason") or "").strip()
            if target not in nodes or target == focal or judgment_type not in VALID:
                bad += 1
                continue
            pair = tuple(sorted([focal, target]))
            if pair in seen:
                dropped += 1
                continue
            seen.add(pair)
            first_fp, second_fp = sorted([fingerprints[pair[0]], fingerprints[pair[1]]])
            judgments["%s|%s" % (first_fp, second_fp)] = {
                "a": focal, "b": target, "related": True,
                "type": judgment_type, "reason": reason,
            }
            kept_total += 1
    return judgments, kept_total, dropped, bad


def main():
    if G.report_root_argument_error():
        return 1
    ap = argparse.ArgumentParser(
        description="Aggregate edge-judge workflow output into the OKF graph cache."
    )
    ap.add_argument("--knowledge-root", help="OKF knowledge-base root")
    ap.add_argument("workflow_output")
    ap.add_argument("model_id", nargs="?", default="claude-opus-4-8")
    args = ap.parse_args()
    out_path = args.workflow_output
    model = args.model_id
    with open(out_path, encoding="utf-8") as workflow_file:
        raw = json.load(workflow_file)
    payload = raw.get("result", raw)
    results = payload.get("results", [])
    here = G.ROOT
    manifest_path = os.path.join(here, ".build/cand/manifest.json")
    with open(manifest_path, encoding="utf-8") as manifest_file:
        manifest = json.load(manifest_file)
    index_to_id = {item["idx"]: item["id"] for item in manifest}

    nodes = G.load_nodes()
    fingerprints = G.card_fp_map(nodes)
    judgments, kept_total, dropped, bad = _aggregate_judgments(
        results, index_to_id, nodes, fingerprints
    )

    cache = {"version": 3, "model": model,
             "card_fp": {node_id: fingerprints[node_id] for node_id in G.concept_ids(nodes)},
             "judgments": judgments}
    with open(G.JUDGMENTS_PATH, "w", encoding="utf-8") as judgments_file:
        json.dump(cache, judgments_file, ensure_ascii=False, indent=1, sort_keys=True)
    _emit("aggregate: %d kept edges from %d focal cards", kept_total, len(results))
    _emit("  dedup-dropped(both-sides): %d  invalid(target/type): %d", dropped, bad)
    _emit("  -> %s", os.path.relpath(G.JUDGMENTS_PATH, here))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
