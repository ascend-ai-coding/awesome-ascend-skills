# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

"""Validate owner metadata and retain partial reference coverage without importing kernels."""

from __future__ import annotations

import json
import os
import re
from pathlib import Path, PurePosixPath

from . import catalog, feedback, host_io, knowledge, links

_MARKER = re.compile(r"^<!--\s*kernel-query:\s*(.*?)\s*-->$")
_FENCE = re.compile(r"^ {0,3}(`{3,}|~{3,})")
_ID = re.compile(r"[a-z0-9][a-z0-9_-]*")
_CONTRACT_FIELDS = {
    "id",
    "aliases",
    "term_groups",
    "symbols",
    "rules",
    "example",
    "candidate",
    "when",
    "missing_facts",
    "review_symbols",
    "discovery_terms",
    "discovery_aliases",
    "facets",
}
_EXAMPLE_FIELDS = {"aliases", "term_groups", "facets", "exclude"}
_CANDIDATE_FIELDS = {
    "id",
    "title",
    "conditions",
    "decrease",
    "increase",
    "reject_if",
    "comparison",
}


def loads(text):
    def pairs(items):
        result = {}
        for key, value in items:
            if key in result:
                raise ValueError(f"Duplicate metadata key: {key}")
            result[key] = value
        return result

    def constant(value):
        raise ValueError(f"Non-finite metadata value: {value}")

    return json.loads(text, object_pairs_hook=pairs, parse_constant=constant)


def _string(value, field):
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{field} requires a nonempty string")


def _strings(value, field):
    if not isinstance(value, list):
        raise ValueError(f"{field} requires a string list")
    for item in value:
        _string(item, field)


def _groups(value, field):
    if not isinstance(value, list):
        raise ValueError(f"{field} requires a list of term groups")
    for group in value:
        _strings(group, field)
        if not group:
            raise ValueError(f"{field} requires nonempty term groups")


def validate(meta, *, kind):
    fields = _CONTRACT_FIELDS if kind == "contract" else _EXAMPLE_FIELDS
    if not isinstance(meta, dict) or set(meta) - fields:
        raise ValueError(f"Invalid {kind} metadata fields")
    meta = dict(meta)
    for key in ("aliases", "symbols", "rules", "missing_facts", "discovery_aliases"):
        if key in meta:
            _strings(meta[key], key)
            if key in {"aliases", "discovery_aliases"} and any(
                value != value.strip() for value in meta[key]
            ):
                raise ValueError(f"{key} cannot contain leading or trailing whitespace")
    for key in ("term_groups", "discovery_terms", "review_symbols"):
        if key in meta:
            _groups(meta[key], key)
    for key in ("symbols", "review_symbols"):
        groups = [meta[key]] if key == "symbols" and key in meta else meta.get(key, [])
        for group in groups:
            if any(knowledge.symbol(name) is None for name in group):
                raise ValueError(f"{key} requires pl/vf public-symbol spellings")
        if key == "symbols" and key in meta:
            meta[key] = list(
                dict.fromkeys(knowledge.symbol(name) for name in meta[key])
            )
        elif key in meta:
            meta[key] = [
                [knowledge.symbol(name) for name in group] for group in meta[key]
            ]
    if kind == "contract":
        if not isinstance(meta.get("id"), str) or not _ID.fullmatch(meta["id"]):
            raise ValueError("Contract id requires a lowercase owner token")
        if "when" in meta:
            _string(meta["when"], "when")
        if meta.get("review_symbols") and not meta.get("when"):
            raise ValueError("review_symbols requires an explicit when condition")
        if "example" in meta:
            _string(meta["example"], "example")
            path = PurePosixPath(meta["example"])
            if (
                path.is_absolute()
                or ".." in path.parts
                or "\\" in meta["example"]
                or "\x00" in meta["example"]
                or path.name != "kernel.py"
            ):
                raise ValueError(
                    "example requires a relative kernel.py path inside the example library"
                )
            if path.parts[0] not in ("basic_api", *catalog.GROUPS) or any(
                part.startswith(("_", ".")) for part in path.parts
            ):
                raise ValueError("example must select a public example program")
        if "candidate" in meta:
            candidate = meta["candidate"]
            if not isinstance(candidate, dict) or set(candidate) != _CANDIDATE_FIELDS:
                raise ValueError("Invalid maintained candidate metadata")
            if not isinstance(candidate["id"], str) or not _ID.fullmatch(
                candidate["id"]
            ):
                raise ValueError("Candidate id requires a lowercase owner token")
            for key in ("title", "comparison"):
                _string(candidate[key], "candidate." + key)
            for key in ("conditions", "decrease", "increase", "reject_if"):
                _strings(candidate[key], "candidate." + key)
                if not candidate[key]:
                    raise ValueError(f"candidate.{key} requires at least one statement")
    if "exclude" in meta:
        if not isinstance(meta["exclude"], list):
            raise ValueError("exclude requires a list")
        for exclusion in meta["exclude"]:
            if not isinstance(exclusion, dict) or set(exclusion) != {
                "term_groups",
                "reason",
            }:
                raise ValueError("Exclusions require term_groups and reason")
            _groups(exclusion["term_groups"], "exclude.term_groups")
            if not exclusion["term_groups"]:
                raise ValueError("Exclusions require nonempty term_groups")
            _string(exclusion["reason"], "exclude.reason")
    knowledge.facets({**meta, "id": meta.get("id", "example")})
    return meta


def collect(root):
    """Read current bytes, quarantine invalid/duplicate owners, and retain located errors."""
    root = Path(root).resolve()
    entries, files, errors = [], {}, []
    limits = {
        key: host_io.CEILINGS[key]
        for key in ("max_files", "max_file_bytes", "max_scan_bytes")
    }
    seen, scanned, markers = 0, 0, 0

    def error(path, reason, detail="", *, line=1, digest=None, identity=None):
        errors.append(
            {
                "path": str(path),
                "line": line,
                "reason": reason,
                "detail": detail[:400],
                **({"sha256": digest} if digest else {}),
                **({"id": identity} if identity else {}),
            }
        )

    if not root.is_dir():
        error(root, "source_unavailable", "Reference root is missing or unreadable")
    for parent, directories, names in os.walk(
        root,
        followlinks=False,
        onerror=lambda exc: error(
            exc.filename, "source_unavailable", type(exc).__name__
        ),
    ):
        if Path(parent) == root and "knowledge-cards" in directories:
            directories.remove("knowledge-cards")  # Card examples cannot declare kernel contracts.
        directories.sort()
        for name in sorted(names):
            if not name.lower().endswith(".md"):
                continue
            path = Path(parent) / name
            seen += 1
            digest = None
            try:
                if not path.resolve().is_relative_to(root):
                    raise ValueError("Reference path escapes its selected root")
                if seen > limits["max_files"]:
                    raise ValueError("reference_file_scan_limit")
                data = host_io._bytes(
                    path,
                    min(limits["max_file_bytes"], limits["max_scan_bytes"] - scanned),
                )
                scanned += len(data)
                digest = feedback.digest(data)
                files[str(path.resolve())] = digest
                text = data.decode("utf-8")
                if "\x00" in text:
                    raise ValueError("Reference contains binary data")
                lines = text.splitlines()
                headings = links.heading_locations(text)
                fence, marker_lines = None, []
                for line, value in enumerate(lines, 1):
                    opening = _FENCE.match(value)
                    if opening:
                        if fence is None:
                            fence = opening[1]
                        elif (
                            opening[1].startswith(fence)
                            and not value[opening.end():].strip()
                        ):
                            fence = None
                        continue
                    if fence is not None or value.startswith(("    ", "\t")):
                        continue
                    stripped = value.strip()
                    if re.match(r"^<!--\s*kernel-query:", stripped):
                        marker_lines.append((line, stripped))
                markers += len(marker_lines)
                marked_lines = {line for line, _ in marker_lines}
                for line, value in marker_lines:
                    try:
                        marker = _MARKER.fullmatch(value)
                        if marker is None:
                            raise ValueError(
                                "kernel-query requires a single-line JSON comment"
                            )
                        meta = validate(loads(marker[1]), kind="contract")
                        following = next(
                            (
                                number
                                for number in range(line + 1, len(lines) + 1)
                                if lines[number - 1].strip()
                            ),
                            None,
                        )
                        heading = next(
                            (item for item in headings if item["line"] == following),
                            None,
                        )
                        if heading is None:
                            raise ValueError("Query marker must precede an ATX heading")
                        end = next(
                            (
                                item["line"]
                                for item in headings
                                if item["line"] > following
                                and item["level"] <= heading["level"]
                            ),
                            len(lines) + 1,
                        )
                        body = "\n".join(
                            lines[number - 1]
                            for number in range(following, end)
                            if number not in marked_lines
                        ).strip()
                        meta.update(
                            line=following,
                            title=heading["title"],
                            _reference={"path": str(path.resolve()), "sha256": digest},
                        )
                        entries.append((meta, path, body))
                    except (ValueError, TypeError) as exc:
                        error(
                            path, "metadata_invalid", str(exc), line=line, digest=digest
                        )
            except (OSError, ValueError) as exc:
                error(path, "source_unavailable", str(exc), digest=digest)
                if seen >= limits["max_files"] or scanned >= limits["max_scan_bytes"]:
                    break
        if seen >= limits["max_files"] or scanned >= limits["max_scan_bytes"]:
            error(root, "source_scan_limit", "Reference scan is partial")
            break
    for field in ("id", "candidate"):
        owners = {}
        for meta, path, _ in entries:
            value = meta["id"] if field == "id" else meta.get("candidate", {}).get("id")
            if value:
                owners.setdefault(value, []).append((meta, path))
        duplicates = {value for value, rows in owners.items() if len(rows) > 1}
        for value in sorted(duplicates):
            for meta, path in owners[value]:
                error(
                    path,
                    "duplicate_id" if field == "id" else "duplicate_candidate_id",
                    value,
                    line=meta["line"],
                    digest=meta["_reference"]["sha256"],
                    identity="contract:" + meta["id"],
                )
        entries = [
            entry
            for entry in entries
            if (
                entry[0]["id"]
                if field == "id"
                else entry[0].get("candidate", {}).get("id")
            )
            not in duplicates
        ]
    manifest = feedback.digest(
        json.dumps(
            {"root": str(root), "files": files, "errors": errors}, sort_keys=True
        ).encode()
    )
    return entries, {
        "complete": not errors,
        "files_read": len(files),
        "files_seen": seen,
        "scan_bytes": scanned,
        "markers": markers,
        "objects": len(entries),
        "limits": limits,
        "manifest_sha256": manifest,
        "error_count": len(errors),
        "errors": errors[:8],
        "omitted_errors": max(0, len(errors) - 8),
        "_files": files,
        "_errors": errors,
    }
