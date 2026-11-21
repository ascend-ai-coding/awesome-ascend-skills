# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

"""Small, source-bound feedback shared by query, checks and execution."""

from __future__ import annotations

import hashlib
import json
import platform
import shlex
from pathlib import Path

from .paths import SKILL_ROOT

SCHEMA = "pypto-kernel-feedback/1"
FRAMEWORK_FILES = (
    "python/pypto_pro/language/_api.py",
    "python/pypto_pro/language/_vf_api.py",
    "python/pypto_pro/ir/op/block_ops.py",
    "python/pypto_pro/runtime/jit.py",
    "python/pypto_pro/language/parser/_buffer_parser.py",
)


def digest(data):
    return hashlib.sha256(data).hexdigest()


def file_identity(path):
    path = Path(path).resolve()
    return {"path": str(path), "sha256": digest(path.read_bytes())}


def identity(pypto_root, sources=()):
    files = {
        name: digest((pypto_root / name).read_bytes())
        for name in FRAMEWORK_FILES
        if (pypto_root / name).is_file()
    }
    implementation = {
        str(p.relative_to(SKILL_ROOT)): digest(p.read_bytes())
        for p in sorted((SKILL_ROOT / "scripts/_checks").glob("*.py"))
    }
    return {
        "python": platform.python_version(),
        "sources": [file_identity(p) for p in sources],
        "checker_sha256": digest(json.dumps(implementation, sort_keys=True).encode()),
        "selected_source": {
            "root": str(pypto_root.resolve()),
            "files": files,
            "complete": len(files) == len(FRAMEWORK_FILES),
        },
        "installed_runtime": None,
    }


def diagnostic(
    rule, severity, message, *, location=None, evidence=None, next_action=None
):
    if severity not in {"error", "advisory", "unverified"}:
        raise ValueError(f"Invalid severity: {severity}")
    return {
        "rule": rule,
        "severity": severity,
        "message": message,
        "location": location,
        "evidence": evidence or {},
        "next_action": next_action,
    }


def envelope(stage, pypto_root, sources=()):
    return {
        "schema": SCHEMA,
        "stage": stage,
        "status": "passed",
        "identity": identity(pypto_root, sources),
        "diagnostics": [],
        "context": [],
        "coverage": {},
        "validation": {
            "native_parse": "not_run",
            "compilation": "not_established",
            "numerical": "not_established",
            "performance": "not_established",
        },
    }


def finding_dict(finding):
    result = diagnostic(
        finding.rule,
        "advisory" if finding.advisory else "error",
        finding.message,
        location={"file": str(finding.path.resolve()), "line": finding.lineno},
    )
    result["waived"] = finding.waived
    return result


def finish(result):
    if any(
        d["severity"] == "error" and not d.get("waived") for d in result["diagnostics"]
    ):
        result["status"] = "failed"
    return result


def exit_status(result):
    return {"passed": 0, "failed": 1, "unverified": 2}[result["status"]]


def presentation(result, *, details=False):
    """Keep identities and actionable evidence; expand descriptor inventories on request."""
    if details:
        return result
    result = json.loads(json.dumps(result))
    selected = result["identity"]["selected_source"]
    files = selected.pop("files", {})
    selected["files_sha256"] = digest(json.dumps(files, sort_keys=True).encode())
    runtime = result["identity"].get("installed_runtime")
    if runtime:
        runtime.pop("python_sources", None)
    if result["stage"] != "query":
        result.pop("transfers", None)
        result.pop("omitted_transfer_sites", None)
    scopes = result["coverage"].get("lexical_names")
    if isinstance(scopes, list):
        result["coverage"]["lexical_names"] = {
            "checked": sum(s["checked"] for s in scopes),
            "unverified": sum(s["unverified"] for s in scopes),
            "reasons": sorted({s["reason"] for s in scopes if s["unverified"]}),
        }
    if result.get("native_preflight"):
        result["native_preflight"].pop("coverage", None)
    return result


def _print_program_reads(item):
    if item.get("facets"):
        print(
            "    Declared routing conditions: "
            + json.dumps(item["facets"], ensure_ascii=False)
        )
    for role, request in item["program"].get("read_requests", {}).items():
        print(
            "    Read "
            + role
            + ": read --request "
            + shlex.quote(json.dumps(request, ensure_ascii=False))
        )
    for source in item.get("condition_sources", []):
        print(
            "    Check "
            + source["role"]
            + " statements: read --request "
            + shlex.quote(json.dumps(source["read"], ensure_ascii=False))
        )


def print_result(result, *, json_output=True, details=False):
    result = presentation(result, details=details)
    if json_output:
        print(json.dumps(result, ensure_ascii=False, separators=(",", ":")))
        return
    print(f"{result['stage']}: {result['status']}")
    print("validation " + json.dumps(result["validation"], ensure_ascii=False))
    for item in result["diagnostics"]:
        loc = item.get("location") or {}
        where = f"{loc['file']}:{loc.get('line', 1)}: " if loc.get("file") else ""
        print(f"{where}{item['severity']} {item['rule']}: {item['message']}")
        if item.get("next_action"):
            print(f"  next: {item['next_action']}")
    for item in result.get("signatures", []):
        print(item["signature"])
    if result.get("admission"):
        admission = result["admission"]
        print(f"Knowledge admission: {admission['objects']} objects")
        for gate in admission["maintenance"]:
            print(f"  {gate['name']}: {gate['status']}")
        print(admission["evidence_limit"])
    knowledge = result.get("knowledge")
    if knowledge:
        print(
            f"Knowledge: {knowledge['lookup']['status']} ({knowledge['page']['returned']}/{knowledge['page']['total']})"
        )
        for item in knowledge["items"]:
            reference = item["reference"]
            print(
                f"{item['id']} [{item['kind']}; {item['match']['reason']};"
                f" applicability={item['applicability']['status']}]"
            )
            print(
                f"  {reference['path']}:{reference.get('line', 1)} sha256={reference['sha256']}"
            )
            if item.get("declaration") and item["kind"] != "api":
                print("  " + item["declaration"])
            if item.get("text"):
                print(item["text"])
            if item["kind"] == "example":
                print("  Program roles: " + ", ".join(item["example_roles"]))
                for role, entry in item["program"]["entries"].items():
                    print(
                        f"  {role}: {entry['path']}:{entry.get('line', 1)} sha256={entry['sha256']}"
                    )
                print(
                    "  Static imports: "
                    + ", ".join(item["dependencies"]["external_modules"])
                )
                _print_program_reads(item)
            if item["applicability"]["unknown_filters"]:
                print(
                    "  Unknown conditions: "
                    + ", ".join(item["applicability"]["unknown_filters"])
                )
        print(knowledge["evidence_limit"])
        if recovery := knowledge.get("recovery"):
            print("Recovery program sources: " + recovery["evidence_limit"])
            for item in recovery["items"]:
                reference = item["reference"]
                print(f"  {item['id']} [{', '.join(item['example_roles'])}]")
                print(
                    f"    {reference['path']}:{reference.get('line', 1)} sha256={reference['sha256']}"
                )
                print("    Matched terms: " + ", ".join(item["matched_terms"]))
                print("    Unmatched terms: " + ", ".join(item["unmatched_terms"]))
                _print_program_reads(item)
                print(
                    "    Resolve with query "
                    + shlex.quote(item["id"])
                    + " --reference-sha256 "
                    + reference["sha256"]
                )
        if knowledge["lookup"].get("reason"):
            print("Lookup detail: " + knowledge["lookup"]["reason"])
        if knowledge.get("content"):
            content = knowledge["content"]
            for item in content["items"]:
                print(f"\n{item['path']}:{item['start_line']}\n{item['text']}")
            if content.get("next"):
                following = {
                    key: value
                    for key, value in content["next"].items()
                    if key != "request_index"
                }
                print(
                    "Content is partial; continue with read --request "
                    + shlex.quote(json.dumps(following, ensure_ascii=False))
                )
        if knowledge["page"]["next_cursor"]:
            print("Continue with --cursor " + knowledge["page"]["next_cursor"])
    for item in result.get("related_contracts", []):
        print(
            f"Related contract: {item['title']} — {item['when']} Query: {item['query']}"
        )
    if result.get("omitted_related_contracts"):
        print(
            f"Additional contract routes omitted: {result['omitted_related_contracts']}"
        )
    if result.get("selected_frontend_store"):
        print(
            "selected_frontend_store "
            + json.dumps(result["selected_frontend_store"], ensure_ascii=False)
        )
    for item in result.get("transfers", []):
        print("observed_transfer " + json.dumps(item, ensure_ascii=False))
    if result.get("omitted_transfer_sites"):
        print(f"Additional transfer sites omitted: {result['omitted_transfer_sites']}")
    for item in result["context"]:
        print(
            f"\n{item['reference']['path']}:{item['reference']['line']}\n{item['text']}"
        )
        for example in item.get("examples", []):
            print(f"\n{example['path']}\n{example.get('text', '')}")
    if result.get("missing_facts"):
        print("Unresolved: " + "; ".join(result["missing_facts"]))
    if result.get("action_guidance"):
        print(
            "action_guidance "
            + json.dumps(result["action_guidance"], ensure_ascii=False)
        )
