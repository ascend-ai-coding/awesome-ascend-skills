# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

"""retrieval.config — paths, constants, BM25F params, kind registry, regexes.

Single home for ROOT and all tunables (SPEC-Retrieve.md). ROOT is the resolved
knowledge-base root. knowledge_query.py resolves --knowledge-root, CANNBOT_KNOWLEDGE_ROOT(S), the
persisted cannbot env file, or structural discovery before importing this
module. Direct imports still fall back to environment variables or cwd.
"""
import os
import re

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
# target knowledge-base root; scripts stay plugin-local.
ROOT = os.path.abspath(
    os.environ.get("CANNBOT_KNOWLEDGE_ROOT")
    or os.environ.get("OKF_KNOWLEDGE_ROOT")
    or os.environ.get("KNOWLEDGE_ROOT")
    or os.getcwd()
)
REFERENCE = os.path.join(ROOT, "reference")
# Content roots. reference cards are id'd WITHOUT a root prefix (bundle is the first
# segment, e.g. asc-devkit/...); ops/runbooks carry their root prefix. This matches
# okf_graph node ids exactly, so a retrieval doc-id == a graph node-id (multi-hop).
CONTENT_ROOTS = [
    ("reference", REFERENCE),
    ("ops", os.path.join(ROOT, "ops")),
    ("runbooks", os.path.join(ROOT, "runbooks")),
]
INDEX_DIR = os.path.join(ROOT, "search")
INDEX_PATH = os.path.join(INDEX_DIR, "okf.index.json")
GRAPH_JSON = os.path.join(ROOT, ".build", "okf.graph.json")   # compiled graph (gitignored)

REL_START = "<!-- okf:related:start -->"
REL_END = "<!-- okf:related:end -->"
_REL_RE = re.compile(re.escape(REL_START) + r".*?" + re.escape(REL_END), re.S)

# --- BM25F parameters (defaults fixed for reproducibility; written to meta) ----
FIELD_WEIGHTS = {
    "title": 3.0, "tags": 2.5, "description": 2.0,
    "signatures": 1.5, "headings": 1.2, "path": 1.2, "body": 1.0,
}
K1 = 1.2
B = 0.75
EXACT_NAME_BONUS = 5.0
TAG_EXACT_BONUS = 2.0
STAT_PREC = 6
SCORE_PREC = 4
INDEX_VERSION = 4          # adds structured platforms metadata for retrieval filtering

# --- type -> kind registry (cross-bundle; SPEC §3.2.1). Extend per new bundle. --
KNOWN_KINDS = {
    "api", "guide", "example", "operator", "glossary",
    "operator_optimization", "implementation_trap", "debugging_journey",
    "cross_skill_gap", "index",
    # legacy/runtime grouping kinds kept for compatibility with older graph/index data
    "header", "field_note", "runbook", "other",
}
TYPE_KIND = {
    # okf.v1 canonical subtypes
    "api_reference": "api",
    "code_example": "example",
    "devkit_guide": "guide",
    "programming_guide": "guide",
    "profiling_guide": "guide",
    "migration_guide": "guide",
    "term": "glossary",
    "paradigm": "glossary",
    "operator_spec": "operator",
    "optimization_runbook": "operator_optimization",
    "implementation_trap": "implementation_trap",
    "debugging_journey": "debugging_journey",
    "cross_skill_gap": "cross_skill_gap",
    "root_index": "index",
    "bundle_index": "index",
    "section_index": "index",
    # legacy display/source types
    "ASC-DevKit API Reference": "api",
    "ASC-DevKit Example": "example",
    "ASC-DevKit Header": "header",
    "ASC-DevKit Guide": "guide",
    "Ascend C Dev Guide": "guide",
    "Ascend C Profiling Guide": "guide",
    "ASC-DevKit API Guide": "api",
    "Guide": "guide",
    "Glossary": "glossary",
    # ops/ + runbooks/ content roots
    "Operator": "operator",
    "Implementation Trap": "implementation_trap",
    "Debugging Journey": "debugging_journey",
    "Cross-Skill Gap": "cross_skill_gap",
    "Runbook": "operator_optimization",
    "OKF Bundle Index": "index",
    "Glossary Index": "index",
}

_CJK = r"一-鿿㐀-䶿"
_TOKEN_RE = re.compile(r"[0-9A-Za-z]+|[%s]+" % _CJK)
_CAMEL_RE = re.compile(r"[A-Z]+(?=[A-Z][a-z])|[A-Z]?[a-z]+|[A-Z]+|[0-9]+")
_SIG_HEADINGS = ("函数原型", "主要接口", "接口原型", "function prototype")

_PART_OF = "part_of"

# --- opt-in model routes (non-default, non-deterministic) ----------------------
# Everything below configures the OPT-IN model backends (dense recall, LLM rerank,
# llm-judge). The deterministic default chain never touches these. All heavy client
# libs (claude_agent_sdk / openai) are lazy-imported inside the backends, so importing
# this module (and the recall/rerank registries) stays stdlib-only.
CACHE_DIR = os.path.join(ROOT, ".build")               # gitignored
LLM_CACHE_DIR = os.path.join(CACHE_DIR, "llm_cache")   # judge/reranker verdict cache
EMBED_CACHE_DIR = os.path.join(CACHE_DIR, "embed_cache")  # card-vector cache (api backend)

# LLM (Claude Code SDK) — judge + reranker share these.
LLM_MODEL = os.environ.get("CANNBOT_KNOWLEDGE_QUERY_LLM_MODEL", "claude-haiku-4-5")
LLM_TIMEOUT_S = float(os.environ.get("CANNBOT_KNOWLEDGE_QUERY_LLM_TIMEOUT", "180"))
LLM_MAX_BUDGET_USD = float(os.environ.get("CANNBOT_KNOWLEDGE_QUERY_LLM_BUDGET", "0.5"))
LLM_PROMPT_VERSION = "1"      # bump when the judge prompt template changes (cache key)
LLM_SCHEMA_VERSION = "1"      # bump when the verdicts output schema changes (cache key)
