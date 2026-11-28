#!/usr/bin/env python3
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# Licensed under the CANN Open Software License Agreement Version 2.0.
"""Small, dependency-free retriever for the handler knowledge base."""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import os
import re
import sys
from pathlib import Path

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE))
from runtime_paths import KNOWLEDGE_CORPUS, KNOWLEDGE_STATE  # noqa: E402
from cli_output import write_stdout  # noqa: E402

CONTENT_ROOTS = ("reference", "runbooks")
REQUIRED_FIELDS = (
    "schema_version",
    "kind",
    "type",
    "source_family",
    "title",
    "description",
    "tags",
    "created_at",
    "updated_at",
)
LOGGER = logging.getLogger(__name__)
RUNTIME_CORPUS_SCHEMA = "issue-history.v2"
RUNTIME_STATE_SCHEMA = "issue-knowledge-refresh.v1"
RUNTIME_WARNING = (
    "自动采集的历史证据未经人工审阅，只能用于提出调查方向，不能证明当前 Issue 根因。"
)


def _write_stdout(text):
    """Write the JSON or document output protocol to stdout."""
    write_stdout(text)


def default_root():
    return Path(__file__).resolve().parents[1] / "knowledge"


def resolve_root(cli_root=None):
    root = Path(
        cli_root
        or os.environ.get("ISSUE_HANDLER_KNOWLEDGE_ROOT")
        or default_root()
    )
    root = root.expanduser().resolve()
    if not (root / "index.md").is_file():
        raise ValueError(f"knowledge root is invalid: {root}")
    return root


def parse_scalar(value):
    value = value.strip()
    if value in ("[]", ""):
        return [] if value == "[]" else ""
    if value.startswith("[") and value.endswith("]"):
        return [
            part.strip().strip("'\"") for part in value[1:-1].split(",") if part.strip()
        ]
    return value.strip("'\"")


def parse_card(path, root):
    text = path.read_text(encoding="utf-8")
    if not text.startswith("---\n"):
        return None
    try:
        header, body = text[4:].split("\n---\n", 1)
    except ValueError:
        return None
    meta = {}
    current_list = None
    current_item = None
    for raw in header.splitlines():
        if not raw.strip() or raw.lstrip().startswith("#"):
            continue
        indent = len(raw) - len(raw.lstrip())
        line = raw.strip()
        if indent == 0 and ":" in line:
            key, value = line.split(":", 1)
            meta[key] = parse_scalar(value)
            current_list = key if value.strip() == "" else None
            current_item = None
        elif current_list and line.startswith("- "):
            if not isinstance(meta.get(current_list), list):
                meta[current_list] = []
            item_text = line[2:]
            if ":" in item_text:
                key, value = item_text.split(":", 1)
                current_item = {key: parse_scalar(value)}
                meta[current_list].append(current_item)
            else:
                meta[current_list].append(parse_scalar(item_text))
                current_item = None
        elif current_item is not None and ":" in line:
            key, value = line.split(":", 1)
            current_item[key] = parse_scalar(value)
    return {
        "id": path.relative_to(root).as_posix(),
        "local_path": str(path),
        "meta": meta,
        "body": body.strip(),
    }


def load_cards(root):
    cards = []
    for content_root in CONTENT_ROOTS:
        base = root / content_root
        if not base.exists():
            continue
        for path in sorted(base.rglob("*.md")):
            if path.name == "index.md":
                continue
            card = parse_card(path, root)
            if card:
                cards.append(card)
    return cards


def terms(text):
    lowered = text.lower()
    ascii_terms = re.findall(r"[a-z0-9_./:+-]{2,}", lowered)
    cjk_runs = re.findall(r"[\u4e00-\u9fff]+", lowered)
    cjk_terms = []
    for run in cjk_runs:
        cjk_terms.extend(run[i: i + 2] for i in range(max(1, len(run) - 1)))
    return set(ascii_terms + cjk_terms)


def score_card(card, query):
    query_terms = terms(query)
    meta = card["meta"]
    title = str(meta.get("title", ""))
    raw_tags = meta.get("tags", [])
    tags = " ".join(raw_tags if isinstance(raw_tags, list) else [str(raw_tags)])
    description = str(meta.get("description", ""))
    title_terms, tag_terms = terms(title), terms(tags)
    desc_terms, body_terms = terms(description), terms(card["body"])
    score = 5 * len(query_terms & title_terms)
    score += 4 * len(query_terms & tag_terms)
    score += 2 * len(query_terms & desc_terms)
    score += len(query_terms & body_terms)
    normalized = query.strip().lower()
    if (
        normalized
        and normalized in f"{title}\n{tags}\n{description}\n{card['body']}".lower()
    ):
        score += 8
    return score


def search(root, query, limit=8):
    ranked = []
    for card in load_cards(root):
        score = score_card(card, query)
        if score:
            ranked.append(
                {
                    "id": card["id"],
                    "title": card["meta"].get("title", ""),
                    "description": card["meta"].get("description", ""),
                    "resource": card["meta"].get("resource", ""),
                    "status": card["meta"].get("status", ""),
                    "confidence": card["meta"].get("confidence", ""),
                    "local_path": card["local_path"],
                    "score": score,
                }
            )
    return sorted(ranked, key=lambda item: (-item["score"], item["id"]))[:limit]


def load_runtime_corpus(corpus_path, state_path):
    """Load a runtime corpus only when its commit metadata verifies it."""
    corpus_path = Path(corpus_path).expanduser().resolve()
    state_path = Path(state_path).expanduser().resolve()
    if not corpus_path.is_file() or not state_path.is_file():
        return None, {
            "status": "missing",
            "corpus": str(corpus_path),
            "state": str(state_path),
        }
    try:
        corpus_data = corpus_path.read_bytes()
        corpus = json.loads(corpus_data)
        state = json.loads(state_path.read_text(encoding="utf-8"))
        if not isinstance(corpus, dict) or not isinstance(state, dict):
            raise ValueError("runtime artifacts must contain JSON objects")
        if (corpus.get("metadata") or {}).get("schema_version") != RUNTIME_CORPUS_SCHEMA:
            raise ValueError("unsupported runtime corpus schema")
        if state.get("schema_version") != RUNTIME_STATE_SCHEMA:
            raise ValueError("unsupported runtime state schema")
        digest = hashlib.sha256(corpus_data).hexdigest()
        if digest != state.get("corpus_sha256"):
            raise ValueError("runtime corpus digest mismatch")
        if state.get("repository") != (corpus.get("metadata") or {}).get("repository"):
            raise ValueError("runtime repository mismatch")
        if not isinstance(corpus.get("issues"), list):
            raise ValueError("runtime corpus issues must be a list")
    except (OSError, ValueError) as exc:
        return None, {
            "status": "invalid",
            "corpus": str(corpus_path),
            "reason": str(exc),
        }
    return corpus, {
        "status": "usable",
        "corpus": str(corpus_path),
        "repository": (corpus.get("metadata") or {}).get("repository"),
        "generated_at": (corpus.get("metadata") or {}).get("generated_at"),
        "last_success_at": state.get("last_success_at"),
        "last_refresh_status": state.get("last_refresh_status"),
        "trust": "runtime_evidence_only",
    }


def score_runtime_issue(issue, query):
    query_terms = terms(query)
    title = str(issue.get("title") or "")
    description = str(issue.get("description") or "")
    comments = "\n".join(
        str(comment.get("body") or "") for comment in issue.get("comments") or []
    )
    linked_prs = "\n".join(
        str(pr.get("title") or "") for pr in issue.get("linked_prs") or []
    )
    score = 5 * len(query_terms & terms(title))
    score += 2 * len(query_terms & terms(description))
    score += len(query_terms & terms(comments))
    score += len(query_terms & terms(linked_prs))
    normalized = query.strip().lower()
    searchable = f"{title}\n{description}\n{comments}\n{linked_prs}".lower()
    if normalized and normalized in searchable:
        score += 8
    return score


def search_runtime(corpus, query, limit=5):
    """Return compact, explicitly provisional historical evidence candidates."""
    ranked = []
    repository = (corpus.get("metadata") or {}).get("repository") or "unknown"
    for issue in corpus.get("issues") or []:
        score = score_runtime_issue(issue, query)
        if not score:
            continue
        number = issue.get("number")
        ranked.append(
            {
                "id": f"runtime:{repository}#{number}",
                "title": issue.get("title") or "",
                "description": issue.get("description") or "",
                "resource": issue.get("url") or "",
                "status": "provisional",
                "confidence": "low",
                "score": score,
                "evidence_signals": issue.get("evidence_signals") or [],
                "handling_outcome": issue.get("handling_outcome") or "",
                "warning": RUNTIME_WARNING,
            }
        )
    return sorted(ranked, key=lambda item: (-item["score"], item["id"]))[:limit]


def _runtime_inputs(args):
    if args.no_runtime_corpus:
        return None, {"status": "disabled"}
    repository_root = Path(
        args.repository_root
        or os.environ.get("ISSUE_HANDLER_REPOSITORY_ROOT")
        or Path.cwd()
    ).expanduser().resolve()
    corpus_path = (
        Path(args.runtime_corpus)
        if args.runtime_corpus
        else repository_root / KNOWLEDGE_CORPUS
    )
    state_path = (
        Path(args.runtime_state)
        if args.runtime_state
        else (
            corpus_path.parent / KNOWLEDGE_STATE.name
            if args.runtime_corpus
            else repository_root / KNOWLEDGE_STATE
        )
    )
    return load_runtime_corpus(corpus_path, state_path)


def verify(root):
    findings = []
    for card in load_cards(root):
        meta = card["meta"]
        for field in REQUIRED_FIELDS:
            if meta.get(field) in (None, "", []):
                findings.append({"card": card["id"], "error": f"missing {field}"})
        sources = meta.get("sources")
        if not isinstance(sources, list) or not sources:
            findings.append({"card": card["id"], "error": "missing sources"})
            continue
        primaries = [
            source
            for source in sources
            if isinstance(source, dict) and source.get("role") == "primary"
        ]
        if len(primaries) != 1:
            findings.append(
                {"card": card["id"], "error": "sources must contain one primary"}
            )
        elif meta.get("resource") != primaries[0].get("url"):
            findings.append(
                {
                    "card": card["id"],
                    "error": "resource must equal primary source url",
                }
            )
    return findings


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--knowledge-root", default=None)
    parser.add_argument(
        "--repository-root",
        default=None,
        help=(
            "Target repository root containing .cannbot; defaults to "
            "ISSUE_HANDLER_REPOSITORY_ROOT or the current directory"
        ),
    )
    parser.add_argument("--runtime-corpus", default=None)
    parser.add_argument("--runtime-state", default=None)
    parser.add_argument("--no-runtime-corpus", action="store_true")
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("discover")
    search_parser = sub.add_parser("search")
    search_parser.add_argument("--query", required=True)
    search_parser.add_argument("-k", "--limit", type=int, default=8)
    preflight = sub.add_parser("preflight")
    preflight.add_argument("--task", required=True)
    preflight.add_argument("-k", "--limit", type=int, default=5)
    get_parser = sub.add_parser("get")
    get_parser.add_argument("doc_id")
    sub.add_parser("verify")
    return parser.parse_args(argv)


def _discover_command(root, runtime_status) -> int:
    payload = {
        "knowledge_root": str(root),
        "cards": len(load_cards(root)),
        "runtime_corpus": runtime_status,
    }
    _write_stdout(json.dumps(payload, ensure_ascii=False, indent=2))
    return 0


def _search_command(args, root, runtime_corpus, runtime_status) -> int:
    query = args.query if args.command == "search" else args.task
    results = search(root, query, args.limit)
    runtime_candidates = (
        search_runtime(runtime_corpus, query, min(args.limit, 5))
        if runtime_corpus
        else []
    )
    payload = {
        "query": query,
        "results": results,
        "runtime_candidates": runtime_candidates,
        "runtime_corpus": runtime_status,
    }
    if args.command == "preflight":
        route = "read_first" if results else "runtime_evidence_only"
        if not results and not runtime_candidates:
            route = "no_hit"
        payload.update(
            {
                "route": route,
                "read_first": results[:3],
                "sufficiency_rule": (
                    "受审规则卡优先；运行时历史证据仅用于提出调查方向，"
                    "必须用当前 Issue、代码或复现独立验证，不能单独证明根因。"
                ),
            }
        )
    _write_stdout(json.dumps(payload, ensure_ascii=False, indent=2))
    return 0


def _get_command(args, root) -> int:
    target = (root / args.doc_id).resolve()
    if root not in target.parents or not target.is_file():
        LOGGER.error("unknown doc id: %s", args.doc_id)
        return 2
    _write_stdout(target.read_text(encoding="utf-8"))
    return 0


def _verify_command(root, runtime_status) -> int:
    findings = verify(root)
    payload = {
        "cards": len(load_cards(root)),
        "findings": findings,
        "ok": not findings,
        "runtime_corpus": runtime_status,
    }
    _write_stdout(json.dumps(payload, ensure_ascii=False, indent=2))
    return 1 if findings else 0


def main(argv=None):
    args = parse_args(argv)
    try:
        root = resolve_root(args.knowledge_root)
    except ValueError as exc:
        LOGGER.error("%s", exc)
        return 2

    runtime_corpus, runtime_status = _runtime_inputs(args)
    if args.command == "discover":
        return _discover_command(root, runtime_status)
    if args.command in ("search", "preflight"):
        return _search_command(args, root, runtime_corpus, runtime_status)
    if args.command == "get":
        return _get_command(args, root)
    return _verify_command(root, runtime_status)


if __name__ == "__main__":
    raise SystemExit(main())
