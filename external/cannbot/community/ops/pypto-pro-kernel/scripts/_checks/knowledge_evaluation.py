# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

"""Grade frozen retrieval questions against the public CLI without running kernels."""

from __future__ import annotations

import json
from pathlib import Path
import statistics
import subprocess
import sys
import time

from . import feedback, knowledge, query_metadata
from .paths import EXAMPLES_ROOT, PYPTO_ROOT, SKILL_ROOT

SCHEMA = "pypto-knowledge-evaluation/1"
CASE_SCHEMA = "pypto-knowledge-benchmark/1"
_FIELDS = {
    "id",
    "split",
    "category",
    "query",
    "relevant_ids",
    "expected_lookup",
    "forbidden_ids",
    "unknown_filters",
    "no_recipe",
    "resolve",
    "required_ids",
    "required_roles",
    "rationale",
}
_LOOKUPS = {
    "matched",
    "retrieval_miss",
    "not_integrated",
    "source_unavailable",
    "evidence_insufficient",
}


def load(path):
    data = Path(path).read_bytes()
    if len(data) > 4 * 1024 * 1024:
        raise ValueError("Knowledge benchmark exceeds 4 MiB")
    value = query_metadata.loads(data.decode("utf-8"))
    if (
        not isinstance(value, dict)
        or set(value) != {"schema", "cases"}
        or value["schema"] != CASE_SCHEMA
    ):
        raise ValueError("Expected pypto-knowledge-benchmark/1 with cases")
    cases, seen = value["cases"], set()
    if not isinstance(cases, list) or not cases:
        raise ValueError("Knowledge benchmark needs nonempty cases")
    for case in cases:
        if not isinstance(case, dict) or set(case) - _FIELDS:
            raise ValueError("Unknown knowledge benchmark fields")
        for field in ("id", "category"):
            if not isinstance(case.get(field), str) or not case[field].strip():
                raise ValueError("Knowledge cases need nonempty id/category")
        if case["id"] in seen:
            raise ValueError("Duplicate benchmark case: " + case["id"])
        seen.add(case["id"])
        if case.get("split") not in {"development", "heldout"} or not isinstance(
            case.get("query"), dict
        ):
            raise ValueError("Knowledge cases need split and query")
        if set(case["query"]) - {
            "text",
            "kinds",
            "filters",
            "limit",
            "reference_sha256",
        }:
            raise ValueError(
                "Benchmark queries must be stateless discovery/resolution inputs"
            )
        knowledge.request(**{"text": None, **case["query"]})
        if case.get("expected_lookup", "matched") not in _LOOKUPS:
            raise ValueError("Unknown expected knowledge classification")
        for field in (
            "relevant_ids",
            "forbidden_ids",
            "required_ids",
            "unknown_filters",
            "required_roles",
        ):
            values = case.get(field, [])
            if not isinstance(values, list) or any(
                not isinstance(item, str) or not item.strip() for item in values
            ):
                raise ValueError(field + " requires strings")
        if (
            not case.get("relevant_ids")
            and case.get("expected_lookup", "matched") == "matched"
        ):
            raise ValueError("Matched cases need independently selected relevant IDs")
        for field in ("resolve", "no_recipe"):
            if field in case and not isinstance(case[field], bool):
                raise ValueError(field + " requires a boolean")
    return cases, {"path": str(Path(path).resolve()), "sha256": feedback.digest(data)}


def arguments(query):
    result = ["query", "--format", "json"]
    for kind in query.get("kinds", []):
        result += ["--kind", kind]
    for name in ("filters", "limit", "reference_sha256"):
        if name in query:
            value = (
                json.dumps(query[name], ensure_ascii=False)
                if name == "filters"
                else str(query[name])
            )
            result += ["--" + name.replace("_", "-"), value]
    if query.get("text") is not None:
        result += ["--", query["text"]]
    return result


def invoke(command, argv):
    started = time.monotonic()
    child = subprocess.run([*command, *argv], capture_output=True, timeout=45)
    observation = {
        "exit_code": child.returncode,
        "wall_seconds": time.monotonic() - started,
        "stdout_bytes": len(child.stdout),
        "stdout_sha256": feedback.digest(child.stdout),
        "stderr_bytes": len(child.stderr),
        "stderr_tail": child.stderr.decode("utf-8", errors="replace")[-1000:],
    }
    try:
        report = json.loads(child.stdout)
    except ValueError as error:
        raise ValueError("Checker did not return JSON: " + str(error)) from error
    if not isinstance(report, dict):
        raise ValueError("Checker report must be a JSON object")
    return report, observation


def drain(content, command):
    pages, count, seen = 0, 0, set()
    while True:
        pages += 1
        count += sum(
            len(item.get("text", "").encode("utf-8"))
            for item in content.get("items", [])
        )
        if content.get("status") == "complete" and content.get("coverage", {}).get(
            "complete"
        ):
            return {"status": "complete", "pages": pages, "text_bytes": count}
        following = content.get("next")
        if not following or pages >= 64:
            return {
                "status": "partial",
                "pages": pages,
                "text_bytes": count,
                "reason": content.get("reason", "continuation_unavailable"),
            }
        request = {key: value for key, value in following.items() if key != "request_index"}
        identity = json.dumps(request, sort_keys=True)
        if identity in seen:
            return {
                "status": "partial",
                "pages": pages,
                "text_bytes": count,
                "reason": "continuation_did_not_advance",
            }
        seen.add(identity)
        content, observed = invoke(
            command, ["read", "--request", json.dumps(request, ensure_ascii=False)]
        )
        if observed["exit_code"] != 0:
            return {
                "status": "unverified",
                "pages": pages,
                "text_bytes": count,
                "reason": "read_failed",
            }


def resolve(row, command, roles):
    target = row.get("resolve")
    if not target or row["kind"] not in {"document", "example"}:
        return {"status": "not_applicable"}
    report, observed = invoke(command, arguments(target))
    view = report.get("knowledge", {})
    if (
        observed["exit_code"] != 0
        or view.get("lookup", {}).get("status") != "matched"
        or not view.get("content")
    ):
        return {
            "status": "unverified",
            "reason": "resolution_failed",
            "observation": observed,
        }
    result = drain(view["content"], command)
    result["roles"] = {}
    items = view.get("items", [])
    program = items[0].get("program", {}) if items else {}
    for role in roles:
        request = program.get("read_requests", {}).get(role)
        if not request:
            result["roles"][role] = {
                "status": "unverified",
                "reason": "missing_role_read",
            }
            continue
        content, observed = invoke(
            command, ["read", "--request", json.dumps(request, ensure_ascii=False)]
        )
        result["roles"][role] = (
            drain(content, command)
            if observed["exit_code"] == 0
            else {"status": "unverified", "reason": "role_read_failed"}
        )
    if any(value["status"] != "complete" for value in result["roles"].values()):
        result["status"] = "unverified"
    return result


def grade(case, report, observed):
    view = report.get("knowledge", {})
    items = view.get("items", [])
    expected = case.get("expected_lookup", "matched")
    actual = view.get("lookup", {}).get("status")
    ids = [row["id"] for row in items]
    relevant = set(case.get("relevant_ids", []))
    first = next(
        (index for index, identity in enumerate(ids[:5], 1) if identity in relevant),
        None,
    )
    failures = []
    if observed["exit_code"] != (0 if actual == "matched" else 2):
        failures.append("unexpected_process_status")
    if actual != expected:
        failures.append("lookup_classification")
    if relevant and first is None:
        failures.append("relevant_source_not_in_top5")
    if forbidden := set(case.get("forbidden_ids", [])) & set(ids):
        failures.append("forbidden_condition_route")
    if not set(case.get("required_ids", [])) <= set(ids):
        failures.append("required_source_missing")
    if case.get("no_recipe") and (
        report.get("context") or any("text" in row for row in items)
    ):
        failures.append("weak_discovery_promoted_to_recipe")
    if "unknown_filters" in case and any(
        row.get("applicability", {}).get("unknown_filters") != case["unknown_filters"]
        for row in items
        if row["id"] in relevant
    ):
        failures.append("unknown_conditions_lost")
    partial = view.get("coverage", {}).get("complete") is False and expected not in {
        "source_unavailable",
        "not_integrated",
    }
    if partial:
        failures.append("source_coverage_incomplete")
    return {
        "id": case["id"],
        "category": case["category"],
        "split": case["split"],
        "query": case["query"],
        "status": "unverified" if partial else "passed" if not failures else "failed",
        "failures": failures,
        "lookup": actual,
        "positive": bool(relevant),
        "hit_at_5": None if partial else first is not None if relevant else None,
        "first_relevant_rank": first,
        "reciprocal_rank_at_5": None
        if partial
        else 1 / first
        if first
        else 0
        if relevant
        else None,
        "returned_ids": ids,
        "forbidden_hits": sorted(forbidden),
        "coverage": view.get("lookup", {}).get("coverage"),
        "corpus_sha256": view.get("corpus_sha256"),
        "checker_identity": report.get("identity"),
        "observation": observed,
    }


def run(
    path,
    *,
    checker=None,
    split="all",
    pypto_root=PYPTO_ROOT,
    examples_root=EXAMPLES_ROOT,
):
    evaluator_sources = [
        feedback.file_identity(Path(module.__file__))
        for module in (sys.modules[__name__], knowledge, query_metadata, feedback)
    ]
    cases, identity = load(path)
    chosen = [case for case in cases if split == "all" or case["split"] == split]
    if not chosen:
        raise ValueError("The selected benchmark split is empty")
    checker = Path(checker or SKILL_ROOT / "scripts/check.py").resolve()
    command = [
        sys.executable,
        str(checker),
        "--pypto-root",
        str(Path(pypto_root).resolve()),
        "--examples-root",
        str(Path(examples_root).resolve()),
    ]
    results = []
    for case in chosen:
        try:
            requested = {"limit": 5, **case["query"]}
            report, observed = invoke(command, arguments(requested))
            result = grade(case, report, observed)
            if case.get("resolve"):
                row = next(
                    (
                        row
                        for row in report.get("knowledge", {}).get("items", [])
                        if row["id"] in case.get("relevant_ids", [])
                    ),
                    None,
                )
                result["read"] = (
                    resolve(row, command, case.get("required_roles", []))
                    if row
                    else {"status": "unverified", "reason": "relevant_source_missing"}
                )
                if result["read"]["status"] != "complete":
                    if result["status"] != "unverified":
                        result["status"] = "failed"
                    result["failures"].append("source_read_incomplete")
            results.append(result)
        except (OSError, ValueError, KeyError, subprocess.TimeoutExpired) as error:
            results.append(
                {
                    "id": case["id"],
                    "category": case["category"],
                    "split": case["split"],
                    "status": "unverified",
                    "error": str(error),
                    "positive": bool(case.get("relevant_ids")),
                    "hit_at_5": None,
                    "reciprocal_rank_at_5": None,
                    **(
                        {
                            "read": {
                                "status": "unverified",
                                "reason": "query_did_not_complete",
                            }
                        }
                        if case.get("resolve")
                        else {}
                    ),
                }
            )
    if feedback.file_identity(path) != identity:
        raise ValueError("Benchmark changed during evaluation")
    if any(feedback.file_identity(row["path"]) != row for row in evaluator_sources):
        raise ValueError("Evaluator changed during evaluation; freeze the grader")
    checker_ids = {
        row["checker_identity"]["checker_sha256"]
        for row in results
        if row.get("checker_identity")
    }
    if len(checker_ids) > 1:
        raise ValueError("Checker changed during evaluation; freeze the implementation")
    positive = [row for row in results if row.get("positive")]
    scored = [row for row in positive if row.get("hit_at_5") is not None]
    reads = [row["read"] for row in results if "read" in row]
    durations = [row["observation"]["wall_seconds"] for row in results if "observation" in row]
    return {
        "schema": SCHEMA,
        "status": "failed"
        if any(row["status"] == "failed" for row in results)
        else "unverified"
        if any(row["status"] == "unverified" for row in results)
        else "complete",
        "benchmark": identity,
        "split": split,
        "checker": feedback.file_identity(checker),
        "evaluator": {
            "sources": evaluator_sources,
            "python": sys.version,
            "executable": sys.executable,
        },
        "metrics": {
            "cases": len(results),
            "passed": sum(row["status"] == "passed" for row in results),
            "failed": sum(row["status"] == "failed" for row in results),
            "unverified": sum(row["status"] == "unverified" for row in results),
            "positive_cases": len(positive),
            "scored_positive_cases": len(scored),
            "hit_at_5": sum(row["hit_at_5"] for row in scored) / len(positive)
            if positive and len(scored) == len(positive)
            else None,
            "hit_at_5_on_scored": sum(row["hit_at_5"] for row in scored) / len(scored)
            if scored
            else None,
            "mrr_at_5": statistics.mean(row["reciprocal_rank_at_5"] for row in scored)
            if positive and len(scored) == len(positive)
            else None,
            "source_reads": len(reads),
            "source_read_completion": sum(row["status"] == "complete" for row in reads)
            / len(reads)
            if reads
            else None,
            "query_wall_seconds": {
                "median": statistics.median(durations),
                "max": max(durations),
            }
            if durations
            else None,
            "context_tokens": None,
        },
        "results": results,
        "scope": (
            "Deterministic retrieval and original-source reads only; source identities,"
            " conditions and source roles are graded separately. No model, installed"
            " support, numerical or performance acceptance."
        ),
    }
