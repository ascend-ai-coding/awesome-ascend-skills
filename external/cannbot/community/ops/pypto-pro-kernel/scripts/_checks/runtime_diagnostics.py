# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

"""Associate recognized diagnostic bodies with maintained context, never source excerpts."""

from __future__ import annotations

import re
from pathlib import Path

from . import feedback, query, ub_l1
from .paths import EXAMPLES_ROOT

_HEAD = re.compile(
    r"\[[^\]\n]+\]\[(PRO_\w+)\]:ErrCode: (F[0-9A-F]{5})! (?:Enum: [\w:]+\. )?(.*)"
)
_LOCATION = re.compile(r"^\s*--> (.+):(\d+):(-?\d+)\s*$", re.MULTILINE)
_EXCEPTION = re.compile(
    r"^(?:[\w]+\.)*(?:[A-Z]\w*(?:Error|Exception)|NotSupported|Invalid\w+|RuntimeFailure):\s*(.*)"
)
_RANK = re.compile(
    r"Tensor parameter ['\"]([^'\"]+)['\"] rank mismatch: expected (\d+), got (\d+)"
)
_TRANSPORT = re.compile(
    r"^(?:(?:ssh:|scp:|unix_listener:|mux_client_\w+:).*"
    r"(?:Permission denied|Operation not permitted|Connection refused|Connection timed out|"
    r"Connection closed|Could not resolve hostname|No route to host|Broken pipe|Connection reset|cannot bind)"
    r"|Host key verification failed\.?$|(?:\S+@\S+: )?Permission denied \([^)]+\)\.?(?:\s*)$)",
    re.I,
)
_IMPORT = re.compile(r"^(?:No module named ['\"]|cannot import name ['\"])")


def _records(text):
    for index, line in enumerate(text.splitlines()):
        line = re.sub(r"^E\s+", "", line)
        if line.startswith((" ", "\t")) or line.startswith(
            ("assert ", "raise ", "print(", "'", '"')
        ):
            continue
        error = _EXCEPTION.match(line)
        head = _HEAD.match(error.group(1) if error else line)
        if (
            head
            or error
            or _RANK.fullmatch(line)
            or _TRANSPORT.search(line)
            or re.match(r"^(?:move|insert|vf\.store_align): ", line)
        ):
            yield index, head.group(0) if head else error.group(1) if error else line


def bodies(text):
    """Select diagnostic lines; quoted Python/pytest source is not evidence."""
    return list(dict.fromkeys(message for _, message in _records(text)))[-8:]


def explain_message(
    message,
    *,
    location=None,
    module=None,
    code=None,
    origin="output_tail",
    stage="unknown",
):
    head = _HEAD.fullmatch(message)
    body = head[3] if head else message
    evidence = {
        "module": module or (head[1] if head else None),
        "error_code": code or (head[2] if head else None),
        "origin": origin,
        "stage": stage,
    }
    if _TRANSPORT.search(body):
        rule, action = (
            "execution-transport-failed",
            "Resolve the connection/authentication/control-socket boundary before"
            " another execution request; do not infer a kernel cause. Confirmed"
            " non-dispatch permits independently justified source fixes, kept"
            " unverified. Reconcile unknown dispatch and preserve its source before any replay.",
        )
    elif _IMPORT.search(body):
        rule, action = (
            "execution-import-failed",
            "Check the selected interpreter, environment and named import against"
            " the installed package; no kernel arithmetic or API-absence conclusion follows from an import failure.",
        )
    elif stage in {"argument_binding", "entry_import"}:
        rule, action = (
            "native-stage-failed",
            "Resolve the recorded argument-binding or entry-import failure and"
            " inspect the selected callable before inferring a kernel-body parser or layout defect.",
        )
    elif body == "Expected a native @pl.jit Pro kernel":
        rule, action = (
            "native-entry-kind",
            "Inspect the selected callable, JIT decoration and actual annotation"
            " values before layout changes. Keep wrappers separate from their"
            " wrapped entry; string annotations require a current-version dispatch"
            " probe, not a blanket future-import ban.",
        )
    elif re.match(r"^(?:Unsupported|Incomplete) type annotation:", body):
        rule, action = (
            "jit-annotation-type",
            "Inspect the named parameter declaration and the selected parser's"
            " annotation syntax; scalar annotation aliases and dtype subexpressions follow different paths.",
        )
    elif _RANK.fullmatch(body):
        rule, action = (
            "tensor-rank-mismatch",
            "Compare the selected callable's Tensor annotation with the actual"
            " launch argument; verify the executed entry before a cache hypothesis.",
        )
    elif (
        body
        == 'make_tile_group() depth is required when mutex_ids is "auto", None, or empty'
    ):
        rule, action = (
            "tile-group-depth-required",
            "Set the intended positive compile-time slot depth and verify each"
            " slot's address and lifetime; keep cross-core synchronization separate.",
        )
    elif re.fullmatch(
        r"vf\.add requires dst, src0, src1 to have the same type, got dst=DT_\w+ src0=DT_\w+ src1=DT_\w+",
        body,
    ):
        rule, action = (
            "vf-add-operand-dtype",
            "Trace all operand definitions and the destination's first binding."
            " Check scalar broadcasts for an explicit dtype; preserve the required arithmetic precision.",
        )
    elif (
        re.match(r"(?:move|insert):", body)
        and re.search(
            r"offset|shape|dtype|layout|data path|scale|phase|Tile|NZ|align16",
            body,
            re.I,
        )
        or evidence["module"] == "PRO_IR"
        and body == "offset must contain exactly 2 elements"
    ):
        rule, action = (
            "native-transfer-contract",
            "Resolve the named operands, physical/valid shapes, offset and selected"
            " API signature; use native parsing after the repair.",
        )
        evidence["condition_status"] = "unverified"
    elif body.startswith(
        "vf.store_align (DATA_BLOCK_COPY) does not take an explicit offset:"
    ):
        rule, action = (
            "vf-nz-store-contract",
            "Use the cursor-based store form without an explicit offset; match block_stride to the NZ physical pitch.",
        )
    else:
        return None
    item = feedback.diagnostic(
        rule, "error", message, location=location, evidence=evidence, next_action=action
    )
    item["context_ids"] = query.ids_for_rules([rule])
    return item


def _bind_transfer_context(item, transfers):
    location = item.get("location") or {}
    if (
        item["rule"] != "native-transfer-contract"
        or not location.get("file")
        or not location.get("line")
    ):
        return
    head = _HEAD.fullmatch(item["message"])
    body = head[3] if head else item["message"]
    api = re.match(r"(move|insert):", body)
    sites = [
        site
        for site in transfers
        if Path(site["file"]).resolve() == Path(location["file"]).resolve()
        and site["line"] <= location["line"] <= (site.get("end_line") or site["line"])
        and (not api or not site.get("api") or site["api"] == api[1])
    ]
    if len(sites) != 1:
        return  # A line shared by several calls does not identify an operand pair.
    site = sites[0]
    item["evidence"]["transfer"] = site
    item["evidence"]["condition_status"] = "source_site_matched"
    if (
        site["source"].get("space") == "Vec"
        and site["destination"].get("space") == "Mat"
    ):
        item["context_ids"] = ["ub-l1"]
    if (
        api
        and api[1] == "move"
        and "NZ" in body
        and ("align16" in body or "not a multiple of 16" in body)
        and any(
            rule == ub_l1.PITCH
            for rule, _, _ in ub_l1.transfer_issues(
                "move", site["source"], site["destination"], False
            )
        )
    ):
        item["rule"] = ub_l1.PITCH
        item["context_ids"] = query.ids_for_rules([ub_l1.PITCH])
        item["next_action"] = (
            "Check the matched Vec-to-Vec NZ move and producer's physical pitch. Use"
            " the aligned conversion control or complete direct VF-to-NZ recipe; this"
            " condition does not reject Vec-to-Mat insert or prove producer bytes."
        )


def from_exception(error, *, stage, examples_root=EXAMPLES_ROOT, transfers=()):
    raw = getattr(error, "message", str(error))
    span = getattr(error, "span", None) or {}
    location = (
        {
            "file": span.get("filename") or span.get("file"),
            "line": span.get("line") or span.get("begin_line"),
            "column": span.get("column") or span.get("begin_column"),
        }
        if span
        else None
    )
    code = getattr(error, "error_code", None)
    code = f"{int(code):06X}" if isinstance(code, int) else None
    item = explain_message(
        raw,
        location=location,
        module=getattr(error, "module", None),
        code=code,
        origin="exception",
        stage=stage,
    )
    if item is None:
        item = feedback.diagnostic(
            "native-rejection",
            "error",
            raw,
            location=location,
            evidence={
                "exception": type(error).__name__,
                "stage": stage,
                "error_code": code,
                "module": getattr(error, "module", None),
                "origin": "exception",
            },
            next_action="Use the original diagnostic and source location; no specific root cause was inferred.",
        )
    item["evidence"]["hint"] = getattr(error, "hint", None)
    item["evidence"]["exception"] = type(error).__name__
    _bind_transfer_context(item, transfers)
    return [item], query.context(
        item.get("context_ids", []), examples_root=examples_root
    )


def from_output(text, *, examples_root=EXAMPLES_ROOT, transfers=()):
    found = []
    lines = text.splitlines()
    for index, message in list(_records(text))[-8:]:
        location = None
        for line in lines[index + 1: index + 14]:
            line = re.sub(r"^E\s+", "", line)
            if bodies(line):
                break
            location_match = _LOCATION.match(line)
            if location_match:
                location = {
                    "file": location_match[1],
                    "line": int(location_match[2]),
                    "column": int(location_match[3]),
                }
                break
        head = _HEAD.search(message)
        item = explain_message(
            message,
            location=location,
            module=head[1] if head else None,
            code=head[2] if head else None,
        )
        if item is None and head:
            item = feedback.diagnostic(
                "framework-rejection",
                "error",
                message,
                location=location,
                evidence={
                    "module": head[1],
                    "error_code": head[2],
                    "origin": "output_tail",
                    "stage": "unknown",
                },
                next_action="Inspect the original operation/type contract; no specific root cause was inferred.",
            )
        if item:
            _bind_transfer_context(item, transfers)
            found.append(item)
    ids = {ident for item in found for ident in item.get("context_ids", [])}
    return found, query.context(ids, examples_root=examples_root)
