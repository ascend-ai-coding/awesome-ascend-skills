#!/usr/bin/env python3
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

"""knowledge_query.py — composable retrieval CLI over the OKF corpus (knowledge-query skill).

Legacy verbs (stable compatibility path, no alias/status facets):
  build / search / get / related / neighbors / verify
Composable verbs (Stage-2, alias-aware, scope-normalized, default --status active):
  recall --method {bm25,tfidf,tagtype,graph,dense}
  rerank --method {bm25f,tagidf,quality,reranker,llm-judge}
  pipeline --recall a,b --rerank x        (multi-route recall + merge + rerank)
  overview / browse                       (browse-first category/index view)
  grep <regex>                            (body-only + related-stripped)
  prepare-judge                           (material for llm-judge)
See SPEC-Retrieve.md. Pure Python stdlib default (zero model deps, byte-reproducible);
model routes (dense/reranker/llm-judge) are opt-in real implementations
(Embedding API / Claude Code SDK), lazy-loaded + fingerprint-cached.
"""
from __future__ import annotations

import argparse
import importlib
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))   # make `retrieval` importable

_root_module = importlib.import_module("retrieval.root")
configure_environment_from_argv = _root_module.configure_environment_from_argv
resolution_to_dict = _root_module.resolution_to_dict
_errors_module = importlib.import_module("retrieval.errors")
CliError = _errors_module.CliError
NotConfigured = _errors_module.NotConfigured
ModelRuntimeError = _errors_module.ModelRuntimeError
_output_module = importlib.import_module("retrieval.output")
_log_output = _output_module.emit_stdout
_log_error = _output_module.emit_stderr

ROOT_ARGUMENT_ERROR = None
try:
    sys.argv, ROOT_RESOLUTION = configure_environment_from_argv(sys.argv)
except CliError as root_argument_error:
    ROOT_ARGUMENT_ERROR = root_argument_error
    ROOT_RESOLUTION = {}

_index_module = importlib.import_module("retrieval.index")
cmd_build = _index_module.cmd_build
cmd_verify = _index_module.cmd_verify
load_index = _index_module.load_index
cmd_search = importlib.import_module("retrieval.search").cmd_search
_read_module = importlib.import_module("retrieval.read")
cmd_get = _read_module.cmd_get
cmd_related = _read_module.cmd_related
cmd_neighbors = importlib.import_module("retrieval.graph").cmd_neighbors
cmd_grep = importlib.import_module("retrieval.grep").cmd_grep
cmd_overview = importlib.import_module("retrieval.overview").cmd_overview
_plan_module = importlib.import_module("retrieval.plan")
brief_preflight_output = _plan_module.brief_preflight_output
plan_task = _plan_module.plan_task
preflight_task = _plan_module.preflight_task
attach_local_paths = importlib.import_module("retrieval.paths").attach_local_paths
_platforms_module = importlib.import_module("retrieval.platforms")
platform_context = _platforms_module.platform_context
platform_filter_output = _platforms_module.platform_filter_output
annotate_relevance = importlib.import_module("retrieval.relevance").annotate_relevance
cmd_eval = importlib.import_module("retrieval.eval").cmd_eval
facet_filter = importlib.import_module("retrieval.scope").facet_filter
attach_snippets = importlib.import_module("retrieval.snippet").attach_snippets
get_alias_map = importlib.import_module("retrieval.aliases").get_alias_map
recall_pkg = importlib.import_module("retrieval.recall")
rerank_pkg = importlib.import_module("retrieval.rerank")
llm_judge = rerank_pkg.llm_judge
merge_hits = importlib.import_module("retrieval.hit").merge_hits


def _model_err(error):
    """Structured emit for opt-in model routes: NotConfigured -> 3, ModelRuntimeError -> 2."""
    _log_output("%s", json.dumps(error.as_json(), ensure_ascii=False))
    code = 3 if isinstance(error, NotConfigured) else 2
    raise CliError(code=code)


# --- composable handlers ----------------------------------------------------
def _spec_from(a, idx, alias_map):
    cands = facet_filter(
        idx, scope=a.scope or a.dir, bundle=a.bundle, kind=a.kind, category=a.category,
        section=a.section, ctype=a.type, tags=a.tags, paradigm=a.paradigm,
        severity=a.severity, confidence=a.confidence, status=a.status, alias_map=alias_map,
        platform=a.platform)
    return {
        "queries": a.query or [], "tags": a.tags or [], "paradigm": a.paradigm,
        "seed": a.seed, "hops": a.hops, "candidates": cands, "alias_map": alias_map,
        "alias": not a.no_alias, "k": a.k,
        # opt-in model-route params (ignored by deterministic methods)
        "embed_backend": getattr(a, "backend", None),
        "embed_model": getattr(a, "embedding_model", None),
        "llm_model": getattr(a, "llm_model", None),
        "material": getattr(a, "material", "header"),
    }


def _rerank_spec(a):
    return {"seed": getattr(a, "seed", None), "verdicts": getattr(a, "verdicts", None),
            "llm_model": getattr(a, "llm_model", None),
            "material": getattr(a, "material", "header")}


def _emit(hits, idx=None, queries=None, platform=None):
    if idx is not None:
        attach_snippets(hits, queries, idx)
    annotate_relevance(hits, queries=queries)
    context = platform_context(explicit=platform) if platform else platform_context()
    output = attach_local_paths({
        "hits": hits,
        "platform_filter": platform_filter_output(context, len(hits)),
    })
    _log_output("%s", json.dumps(output, ensure_ascii=False, indent=2))


def cmd_recall(a):
    idx = load_index()
    amap = get_alias_map(idx)
    try:
        hits = recall_pkg.REGISTRY[a.method](idx, _spec_from(a, idx, amap))
    except (NotConfigured, ModelRuntimeError) as error:
        _model_err(error)
    _emit(hits, idx, a.query, a.platform)


def _load_hits(src):
    if src in ("-", None):
        data = sys.stdin.read()
    else:
        with open(src, encoding="utf-8") as hits_file:
            data = hits_file.read()
    obj = json.loads(data)
    return obj["hits"] if isinstance(obj, dict) and "hits" in obj else obj


def cmd_rerank(a):
    idx = load_index()
    hits = _load_hits(a.hits)
    q = " ".join(a.query) if a.query else ""
    try:
        out = rerank_pkg.REGISTRY[a.method](idx, q, hits, _rerank_spec(a))
    except (NotConfigured, ModelRuntimeError) as error:
        _model_err(error)
    _emit(out[:a.k] if a.k else out, idx, a.query)


def cmd_pipeline(a):
    idx = load_index()
    amap = get_alias_map(idx)
    spec = _spec_from(a, idx, amap)
    lists = []
    for m in a.recall.split(","):
        m = m.strip()
        try:
            lists.append(recall_pkg.REGISTRY[m](idx, spec))
        except (NotConfigured, ModelRuntimeError) as error:
            _model_err(error)
    merged = merge_hits(lists, mode="sum")
    q = " ".join(a.query) if a.query else ""
    try:
        out = rerank_pkg.REGISTRY[a.rerank](idx, q, merged, _rerank_spec(a))
    except (NotConfigured, ModelRuntimeError) as error:
        _model_err(error)
    _emit(out[:a.k], idx, a.query, a.platform)


def cmd_prepare_judge(a):
    idx = load_index()
    hits = _load_hits(a.hits)
    q = " ".join(a.query) if a.query else ""
    llm_judge.prepare(idx, q, hits, material=a.material)


def cmd_discover():
    _log_output("%s", json.dumps(resolution_to_dict(ROOT_RESOLUTION), ensure_ascii=False, indent=2))


def cmd_plan(a):
    idx = load_index()
    selected = resolution_to_dict(ROOT_RESOLUTION).get("selected")
    knowledge_root = selected.get("path") if isinstance(selected, dict) else None
    out = plan_task(a.task, idx, max_queries=a.max_queries, knowledge_root=knowledge_root, platform=a.platform)
    _log_output("%s", json.dumps(attach_local_paths(out), ensure_ascii=False, indent=2))


def cmd_preflight(a):
    idx = load_index()
    selected = resolution_to_dict(ROOT_RESOLUTION).get("selected")
    knowledge_root = selected.get("path") if isinstance(selected, dict) else None
    out = preflight_task(
        a.task, idx, max_queries=a.max_queries, k=a.k, grep_k=a.grep_k,
        knowledge_root=knowledge_root, platform=a.platform,
    )
    out["knowledge_root"] = knowledge_root
    if a.brief:
        out = brief_preflight_output(out)
    _log_output("%s", json.dumps(attach_local_paths(out), ensure_ascii=False, indent=2))


def cmd_browse(a):
    idx = load_index()
    selected = resolution_to_dict(ROOT_RESOLUTION).get("selected")
    knowledge_root = selected.get("path") if isinstance(selected, dict) else None
    cmd_overview(a, idx, knowledge_root=knowledge_root)


# --- argparse ---------------------------------------------------------------
def _add_facets(p):
    p.add_argument("--bundle")
    p.add_argument("--kind")
    p.add_argument("--category")
    p.add_argument("--section")
    p.add_argument("--type")
    p.add_argument("--paradigm")
    p.add_argument("--severity")
    p.add_argument("--confidence")
    p.add_argument("--tags", action="append")
    p.add_argument("--scope")
    p.add_argument("--dir")
    p.add_argument("--platform", choices=["a3"])
    p.add_argument("--status", default="active", choices=["active", "verified", "stub", "all"])
    p.add_argument("--seed")
    p.add_argument("--hops", type=int, default=1)
    p.add_argument("--no-alias", action="store_true")
    p.add_argument("-k", type=int, default=20)


def _add_model_args(p):
    # opt-in model routes (dense recall / LLM rerank). Deterministic methods ignore these.
    p.add_argument("--backend", help="dense embedding backend: api(default)|hashing")
    p.add_argument("--embedding-model", help="override embedding model (api backend)")
    p.add_argument("--llm-model", help="override Claude Code SDK model (llm-judge/reranker)")


def _add_core_commands(subparsers):
    subparsers.add_parser("discover")
    build = subparsers.add_parser("build")
    build.add_argument("--allow-empty", action="store_true")
    plan = subparsers.add_parser("plan")
    plan.add_argument("--task", required=True)
    plan.add_argument("--platform", choices=["a3"])
    plan.add_argument("--max-queries", type=int, default=6)
    preflight = subparsers.add_parser("preflight")
    preflight.add_argument("--task", required=True)
    preflight.add_argument("--platform", choices=["a3"])
    preflight.add_argument("--max-queries", type=int, default=6)
    preflight.add_argument("-k", type=int, default=8)
    preflight.add_argument("--grep-k", type=int, default=8)
    preflight.add_argument("--brief", action="store_true")


def _add_overview_commands(subparsers):
    for name in ("overview", "browse"):
        overview = subparsers.add_parser(name)
        overview.add_argument("--task", help="open-ended task/question to browse for")
        overview.add_argument("--query", help="short focus term for the overview")
        overview.add_argument("--scope")
        overview.add_argument("--dir")
        overview.add_argument("--bundle")
        overview.add_argument("--kind")
        overview.add_argument("--category")
        overview.add_argument("--status", default="active", choices=["active", "verified", "stub", "all"])
        overview.add_argument("--groups", type=int, default=8)
        overview.add_argument("--per-group", type=int, default=4)


def _add_search_command(subparsers):
    search = subparsers.add_parser("search")
    search.add_argument("--query", action="append", required=True)
    search.add_argument(
        "--allow-multi-query", action="store_true",
        help="legacy score-summed retrieval across multiple --query values",
    )
    search.add_argument("--bundle")
    search.add_argument("--kind")
    search.add_argument("--category")
    search.add_argument("--section")
    search.add_argument("--scope")
    search.add_argument("--dir")
    search.add_argument("--platform", choices=["a3"])
    search.add_argument("-k", type=int, default=10)


def _add_read_commands(subparsers):
    get_parser = subparsers.add_parser("get")
    get_parser.add_argument("paths", nargs="+")
    get_parser.add_argument("--with-related", action="store_true")
    get_parser.add_argument("--section")
    get_parser.add_argument("--max-chars", type=int)
    get_parser.add_argument("--neighbor-limit", type=int, default=20)
    related = subparsers.add_parser("related")
    related.add_argument("path")
    related.add_argument("--bundle")
    related.add_argument("--same-category", action="store_true")
    neighbors = subparsers.add_parser("neighbors")
    neighbors.add_argument("path")
    neighbors.add_argument("--hops", type=int, default=1)
    neighbors.add_argument("--types")
    neighbors.add_argument("--bundle")
    neighbors.add_argument("--platform", choices=["a3"])
    neighbors.add_argument("-k", type=int, default=10)


def _add_verification_commands(subparsers):
    verify = subparsers.add_parser("verify")
    verify.add_argument("--level", default="index", choices=["index", "schema", "strict"])
    verify.add_argument("--limit", type=int, default=50)
    verify.add_argument("--json", action="store_true")
    evaluate = subparsers.add_parser("eval")
    evaluate.add_argument("--cases")
    evaluate.add_argument("-k", type=int, default=5)
    evaluate.add_argument("--grep-k", type=int, default=3)
    evaluate.add_argument("--max-queries", type=int, default=6)
    evaluate.add_argument("--fail-under", type=float)


def _add_composable_commands(subparsers):
    recall = subparsers.add_parser("recall")
    recall.add_argument("--method", required=True, choices=list(recall_pkg.REGISTRY))
    recall.add_argument("--query", action="append")
    _add_facets(recall)
    _add_model_args(recall)
    rerank = subparsers.add_parser("rerank")
    rerank.add_argument("--method", required=True, choices=list(rerank_pkg.REGISTRY))
    rerank.add_argument("--query", action="append")
    rerank.add_argument("--hits", default="-")
    rerank.add_argument("--seed")
    rerank.add_argument("--verdicts")
    rerank.add_argument("-k", type=int, default=0)
    rerank.add_argument("--llm-model")
    rerank.add_argument("--material", default="header", choices=["header", "card"])
    pipeline = subparsers.add_parser("pipeline")
    pipeline.add_argument("--recall", required=True, help="comma-separated recall methods")
    pipeline.add_argument("--rerank", default="bm25f", choices=list(rerank_pkg.REGISTRY))
    pipeline.add_argument("--query", action="append")
    pipeline.add_argument("--verdicts")
    _add_facets(pipeline)
    _add_model_args(pipeline)
    pipeline.add_argument("--material", default="header", choices=["header", "card"])
    grep_parser = subparsers.add_parser("grep")
    grep_parser.add_argument("pattern")
    grep_parser.add_argument("--scope")
    grep_parser.add_argument("--dir")
    grep_parser.add_argument("--platform", choices=["a3"])
    grep_parser.add_argument("--only", default="body", choices=["body", "frontmatter", "all"])
    grep_parser.add_argument("--with-related", action="store_true")
    grep_parser.add_argument("-k", type=int, default=50)
    prepare = subparsers.add_parser("prepare-judge")
    prepare.add_argument("--query", action="append")
    prepare.add_argument("--hits", default="-")
    prepare.add_argument("--material", default="header", choices=["header", "card"])


def _build_parser():
    parser = argparse.ArgumentParser(
        prog="knowledge_query",
        epilog="Global root options accepted anywhere: --knowledge-root PATH, --knowledge-roots PATHS. "
               "Without them, knowledge_query resolves CANNBOT_KNOWLEDGE_ROOT/CANNBOT_KNOWLEDGE_ROOTS, "
               "~/.config/cannbot/knowledge.env, then bounded structural discovery.",
    )
    subparsers = parser.add_subparsers(dest="cmd", required=True)
    _add_core_commands(subparsers)
    _add_overview_commands(subparsers)
    _add_search_command(subparsers)
    _add_read_commands(subparsers)
    _add_verification_commands(subparsers)
    _add_composable_commands(subparsers)
    return parser


def _dispatch_legacy(args):
    if args.cmd == "discover":
        return cmd_discover()
    if args.cmd == "build":
        return cmd_build(allow_empty=args.allow_empty)
    if args.cmd == "plan":
        return cmd_plan(args)
    if args.cmd == "preflight":
        return cmd_preflight(args)
    if args.cmd in ("overview", "browse"):
        return cmd_browse(args)
    if args.cmd == "search":
        return cmd_search(
            args.query, args.bundle, args.kind, args.category, args.section,
            k=args.k, platform=args.platform, scope=args.scope or args.dir,
        )
    if args.cmd == "get":
        return cmd_get(args.paths, args.with_related, args.max_chars, args.section, args.neighbor_limit)
    if args.cmd == "related":
        return cmd_related(args.path, args.bundle, args.same_category)
    if args.cmd == "neighbors":
        return cmd_neighbors(
            args.path, args.hops, args.types, args.bundle, args.k, platform=args.platform,
        )
    if args.cmd == "verify":
        return cmd_verify(level=args.level, limit=args.limit, emit_json=args.json)
    return cmd_eval(
        args.cases, k=args.k, grep_k=args.grep_k,
        max_queries=args.max_queries, fail_under=args.fail_under,
    )


def _dispatch_composable(args):
    if args.cmd == "recall":
        return cmd_recall(args)
    if args.cmd == "rerank":
        return cmd_rerank(args)
    if args.cmd == "pipeline":
        return cmd_pipeline(args)
    if args.cmd == "grep":
        return cmd_grep(
            args.pattern, args.scope or args.dir, args.only, args.with_related, args.k,
            platform=args.platform,
        )
    return cmd_prepare_judge(args)


def main():
    if ROOT_ARGUMENT_ERROR is not None:
        if ROOT_ARGUMENT_ERROR.message is not None:
            _log_error("%s", ROOT_ARGUMENT_ERROR.message)
        return ROOT_ARGUMENT_ERROR.code
    parser = _build_parser()
    args = parser.parse_args()
    if args.cmd == "search" and len(args.query) > 1 and not args.allow_multi_query:
        parser.error(
            "search accepts one --query by default; use preflight/pipeline or run separate searches. "
            "Use --allow-multi-query only for legacy score-summed retrieval."
        )
    legacy_commands = {
        "discover", "build", "plan", "preflight", "overview", "browse",
        "search", "get", "related", "neighbors", "verify", "eval",
    }
    try:
        if args.cmd in legacy_commands:
            return _dispatch_legacy(args)
        return _dispatch_composable(args)
    except CliError as error:
        if error.message is not None:
            _log_error("%s", error.message)
        return error.code


if __name__ == "__main__":
    raise SystemExit(main())
