# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

"""Bounded, explicit-scope source reads shared by CLI and a host hook.

These operations constrain their own reads. They do not sandbox arbitrary Python,
native imports or shell commands offered by the calling host.
"""

from __future__ import annotations

import hashlib
import json
import shutil
import subprocess
from pathlib import Path

from .paths import EXAMPLES_ROOT, PYPTO_ROOT, SKILL_ROOT

SCHEMA = "pypto-context-io/1"
DEFAULTS = {
    "max_output_bytes": 12288,
    "max_files": 128,
    "max_scan_bytes": 2097152,
    "max_file_bytes": 524288,
    "max_lines": 120,
    "max_matches": 24,
}
CEILINGS = {
    "max_output_bytes": 65536,
    "max_files": 1024,
    "max_scan_bytes": 16777216,
    "max_file_bytes": 4194304,
    "max_lines": 500,
    "max_matches": 100,
}


def policy(
    path=None,
    *,
    workspace=None,
    pypto_root=PYPTO_ROOT,
    examples_root=EXAMPLES_ROOT,
    cards_root=None,
):
    workspace = Path(workspace or Path.cwd()).resolve()
    data = {}
    base = workspace
    if path is not None:
        path = Path(path).resolve()
        if path.stat().st_size > 65536:
            raise ValueError("Context policy exceeds 64 KiB")
        data = json.loads(path.read_text())
        if not isinstance(data, dict) or data.get("schema") != "pypto-context-policy/1":
            raise ValueError("Expected pypto-context-policy/1")
        base = path.parent
    defaults = [workspace, SKILL_ROOT, pypto_root, examples_root]
    if cards_root is not None:
        defaults.append(Path(cards_root).expanduser().resolve())
    roots = data.get("roots", defaults)
    if not isinstance(roots, list) or not roots or len(roots) > 16:
        raise ValueError("Context roots must be an explicit list of 1..16 paths")
    roots = tuple((base / p).resolve() for p in roots)
    limits = {**DEFAULTS, **data.get("limits", {})}
    if set(limits) != set(DEFAULTS):
        raise ValueError("Unknown context limit")
    for name, value in limits.items():
        low = 2048 if name == "max_output_bytes" else 1
        if type(value) is not int or not low <= value <= CEILINGS[name]:
            raise ValueError(f"Invalid context limit {name}")
    receipts = {}
    for item in data.get("receipts", []):
        if (
            len(receipts) >= 16
            or not isinstance(item, dict)
            or not isinstance(item.get("request_id"), str)
        ):
            raise ValueError("Invalid pinned receipt")
        target = str((base / item["path"]).resolve())
        receipts[target] = {
            "request_id": item["request_id"],
            "source": str((base / item["source"]).resolve()),
        }
    return {
        "roots": roots,
        "limits": limits,
        "receipts": receipts,
        "sha256": hashlib.sha256(
            json.dumps(
                {
                    "roots": list(map(str, roots)),
                    "limits": limits,
                    "receipts": receipts,
                },
                sort_keys=True,
            ).encode()
        ).hexdigest(),
    }


def checked_path(path, config, *, must_exist=True):
    if len(str(path)) > 2048:
        raise ValueError("Path exceeds context input limit")
    path = Path(path).resolve()
    if not any(path == root or root in path.parents for root in config["roots"]):
        raise ValueError("Path is outside the configured context roots")
    if must_exist and not path.exists():
        raise ValueError("Context path does not exist")
    return path


def _json(value):
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"), allow_nan=False)


def _result(operation, config):
    return {
        "schema": SCHEMA,
        "operation": operation,
        "status": "complete",
        "policy_sha256": config["sha256"],
        "items": [],
        "coverage": {"files_read": 0, "scan_bytes": 0, "complete": True},
        "budget": {
            "limit_bytes": config["limits"]["max_output_bytes"],
            "result_bytes": 0,
            "context_tokens": None,
        },
    }


def encode(result):
    """Count the actual serialized reply, including budget metadata and Unicode."""
    for _ in range(12):
        size = len(_json(result).encode()) + 1
        if size == result["budget"]["result_bytes"]:
            if size > result["budget"]["limit_bytes"]:
                raise ValueError("Context metadata exceeds output budget; narrow paths")
            return _json(result)
        result["budget"]["result_bytes"] = size
    raise ValueError("Could not stabilize context reply size")


def _fits(result, reserve=1024):
    return len(_json(result).encode()) + reserve <= result["budget"]["limit_bytes"]


def _partial(result, reason):
    result["status"] = "partial"
    result["coverage"]["complete"] = False
    result["reason"] = reason


def _bytes(path, limit):
    size = path.stat().st_size
    if not path.is_file() or size > limit:
        raise ValueError(
            "File is not regular or exceeds the remaining source-read budget"
        )
    with path.open("rb") as stream:
        data = stream.read(size)
    if len(data) != size or path.stat().st_size != size:
        raise ValueError("Source changed during bounded read")
    return data


def failure(operation, error, *, limit=12288):
    result = {
        "schema": SCHEMA,
        "operation": operation,
        "status": "unverified",
        "items": [],
        "error": str(error).encode()[:400].decode(errors="ignore"),
        "coverage": {"complete": False},
        "budget": {"limit_bytes": limit, "result_bytes": 0, "context_tokens": None},
    }
    encode(result)
    return result


def _read_text(item, data):
    item["text"] = data.decode("utf-8", errors="ignore")
    # Split only on LF: CRLF bytes and empty source lines retain their positions.
    parts = item["text"].split("\n")
    item["numbered_text"] = "".join(
        f"{item['start_line'] + i}|{part}" + ("\n" if i < len(parts) - 1 else "")
        for i, part in enumerate(parts)
        if part or i < len(parts) - 1
    )


def read(
    paths,
    *,
    config,
    start_line=1,
    max_lines=None,
    offset_bytes=None,
    source_sha256=None,
    fragment=None,
):
    if not paths or len(paths) > 16 or type(start_line) is not int or start_line < 1:
        raise ValueError("Read requires 1..16 paths and a positive start_line")
    if offset_bytes is not None and (
        len(paths) != 1
        or type(offset_bytes) is not int
        or offset_bytes < 0
        or not source_sha256
    ):
        raise ValueError(
            "Byte continuation requires one path, nonnegative offset_bytes and source_sha256"
        )
    if max_lines is not None and (type(max_lines) is not int or max_lines < 1):
        raise ValueError("max_lines must be positive")
    if fragment is not None and (len(paths) != 1 or start_line != 1):
        raise ValueError(
            "Markdown fragment reads require one path and no start_line override"
        )
    count = min(
        max_lines or config["limits"]["max_lines"], config["limits"]["max_lines"]
    )
    result = _result("read", config)
    for index, target in enumerate(paths):
        if result["coverage"]["files_read"] >= config["limits"]["max_files"]:
            _partial(result, "file_scan_budget")
            result["remaining_path_index"] = index
            break
        path = checked_path(target, config)
        if (
            not path.is_file()
            or path.stat().st_size > config["limits"]["max_file_bytes"]
        ):
            raise ValueError("Read requires a regular file within max_file_bytes")
        data = _bytes(
            path,
            min(
                config["limits"]["max_file_bytes"],
                config["limits"]["max_scan_bytes"] - result["coverage"]["scan_bytes"],
            ),
        )
        if len(data) > config["limits"]["max_file_bytes"] or b"\0" in data:
            raise ValueError("File exceeds read limit or contains binary data")
        identity = hashlib.sha256(data).hexdigest()
        data.decode("utf-8")
        if source_sha256 and identity != source_sha256:
            raise ValueError("Source changed; byte continuation is invalidated")
        result["coverage"]["files_read"] += 1
        result["coverage"]["scan_bytes"] += len(data)
        if result["coverage"]["scan_bytes"] > config["limits"]["max_scan_bytes"]:
            raise ValueError("Read exceeds max_scan_bytes; select fewer files")
        lines = data.splitlines(keepends=True)
        section = None
        boundary = len(data)
        if fragment is not None:
            if path.suffix.lower() != ".md":
                raise ValueError("Fragments require a Markdown file")
            from .links import fragment_range

            section = fragment_range(data.decode("utf-8"), fragment)
            boundary = sum(map(len, lines[: section["end_line"]]))
            start_line = section["start_line"]
        begin = (
            offset_bytes
            if offset_bytes is not None
            else sum(map(len, lines[: start_line - 1]))
        )
        if section and not sum(map(len, lines[: start_line - 1])) <= begin <= boundary:
            raise ValueError(
                "Byte continuation is outside the selected Markdown fragment"
            )
        if begin > len(data):
            raise ValueError("offset_bytes is beyond the source")
        if begin < len(data) and data[begin] & 0xC0 == 0x80:
            raise ValueError("offset_bytes must be a UTF-8 character boundary")
        remaining_lines = data[begin:].splitlines(keepends=True)
        requested_end = (
            min(boundary, begin + sum(map(len, remaining_lines[:max_lines])))
            if max_lines is not None
            else boundary
        )
        end = min(requested_end, begin + sum(map(len, remaining_lines[:count])))
        item = {
            "path": str(path),
            "sha256": identity,
            "offset_bytes": begin,
            "start_line": data[:begin].count(b"\n") + 1,
            "start_column_bytes": begin - (data.rfind(b"\n", 0, begin) + 1),
        }
        if section:
            item["section"] = section
        result["items"].append(item)
        lo, hi = 0, end - begin
        while lo < hi:
            mid = (lo + hi + 1) // 2
            _read_text(item, data[begin: begin + mid])
            if _fits(result):
                lo = mid
            else:
                hi = mid - 1
        _read_text(item, data[begin: begin + lo])
        consumed = len(item["text"].encode())
        item["end_offset_bytes"] = begin + consumed
        item["request_complete"] = begin + consumed >= requested_end
        item["file_complete"] = begin + consumed >= len(data)
        if not item["request_complete"]:
            _partial(result, "read_page_limit")
            result["next"] = {
                "path": str(path),
                "offset_bytes": begin + consumed,
                "source_sha256": identity,
            }
            if section:
                result["next"]["fragment"] = section["fragment"]
            if max_lines is not None:
                result["next"]["max_lines"] = max(
                    1, max_lines - data[begin: begin + consumed].count(b"\n")
                )
            result["remaining_path_index"] = index + 1
            break
        if index + 1 < len(paths) and not _fits(result, reserve=1000):
            _partial(result, "output_budget")
            result["remaining_path_index"] = index + 1
            break
    encode(result)
    return result


def read_requests(requests, *, config):
    """Combine explicit per-file ranges under one shared scan and reply budget."""
    if not isinstance(requests, list) or not 1 <= len(requests) <= 16:
        raise ValueError("Batch read needs 1..16 explicit requests")
    result = _result("read", config)
    for index, request in enumerate(requests):
        if result["coverage"]["files_read"] >= config["limits"]["max_files"]:
            _partial(result, "file_scan_budget")
            result["remaining_request_index"] = index
            break
        if (
            not isinstance(request, dict)
            or "path" not in request
            or not isinstance(request["path"], str)
            or set(request)
            - {
                "path",
                "start_line",
                "max_lines",
                "offset_bytes",
                "source_sha256",
                "fragment",
            }
        ):
            raise ValueError("Invalid batch read request")
        used = len(_json(result).encode())
        remaining = config["limits"]["max_output_bytes"] - used - 256
        if remaining < 2048:
            _partial(result, "output_budget")
            result["remaining_request_index"] = index
            break
        child = {
            **config,
            "limits": {
                **config["limits"],
                "max_output_bytes": remaining,
                "max_scan_bytes": config["limits"]["max_scan_bytes"]
                - result["coverage"]["scan_bytes"],
            },
        }
        if child["limits"]["max_scan_bytes"] < 1:
            _partial(result, "byte_scan_budget")
            result["remaining_request_index"] = index
            break
        target = checked_path(request["path"], config)
        if (
            target.is_file()
            and target.stat().st_size > child["limits"]["max_scan_bytes"]
        ):
            _partial(result, "byte_scan_budget")
            result["remaining_request_index"] = index
            break
        kwargs = {key: value for key, value in request.items() if key != "path"}
        page = read([Path(request["path"])], config=child, **kwargs)
        for item in page["items"]:
            result["items"].append({**item, "request_index": index})
        for key in ("files_read", "scan_bytes"):
            result["coverage"][key] += page["coverage"][key]
        if page["status"] == "partial":
            _partial(result, page["reason"])
            result["next"] = {**page["next"], "request_index": index}
            result["remaining_request_index"] = index + 1
            break
    encode(result)
    return result


def search(
    paths,
    *,
    config,
    pattern=None,
    glob=None,
    files=False,
    ignore_case=False,
    fixed=False,
):
    if (
        not paths
        or len(paths) > 16
        or (
            not files
            and (not isinstance(pattern, str) or not pattern or len(pattern) > 1024)
        )
    ):
        raise ValueError(
            "Search needs explicit paths and a bounded pattern, or files=True"
        )
    roots = [checked_path(path, config) for path in paths]
    engine = shutil.which("rg")
    if engine is None:
        raise ValueError("Bounded search requires ripgrep; read/query remain available")
    result = _result("search", config)
    result["paths"] = list(map(str, roots))
    command = [engine, "--files", "--null"]
    if glob:
        command += ["--glob", glob]
    command += ["--", *map(str, roots)]
    proc = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)
    seen = 0
    try:
        buffer = b""
        while chunk := proc.stdout.read1(4096):
            buffer += chunk
            while b"\0" in buffer:
                raw, buffer = buffer.split(b"\0", 1)
                seen += 1
                if seen > config["limits"]["max_files"]:
                    _partial(result, "file_scan_budget")
                    return result
                path = checked_path(raw.decode(), config)
                if files:
                    result["items"].append({"path": str(path)})
                else:
                    size = path.stat().st_size
                    if size > config["limits"]["max_file_bytes"]:
                        _partial(result, "file_size_limit")
                        continue
                    if (
                        result["coverage"]["scan_bytes"] + size
                        > config["limits"]["max_scan_bytes"]
                    ):
                        _partial(result, "byte_scan_budget")
                        return result
                    data = _bytes(
                        path,
                        min(
                            config["limits"]["max_file_bytes"],
                            config["limits"]["max_scan_bytes"]
                            - result["coverage"]["scan_bytes"],
                        ),
                    )
                    if len(data) > config["limits"]["max_file_bytes"]:
                        _partial(result, "file_size_limit")
                        continue
                    size = len(data)
                    result["coverage"]["files_read"] += 1
                    try:
                        data.decode("utf-8")
                        readable = b"\0" not in data
                    except UnicodeDecodeError:
                        readable = False
                    if not readable:
                        _partial(result, "binary_source_omitted")
                        result["coverage"]["scan_bytes"] += size
                        continue
                    source_digest = hashlib.sha256(data).hexdigest()
                    args = [
                        engine,
                        "--json",
                        "--max-count",
                        str(config["limits"]["max_matches"] + 1),
                    ]
                    if ignore_case:
                        args += ["-i"]
                    if fixed:
                        args += ["-F"]
                    found = subprocess.run(
                        [*args, "--", pattern],
                        input=data,
                        capture_output=True,
                        timeout=3,
                    )
                    if found.returncode not in (0, 1):
                        detail = found.stderr.decode("utf-8", errors="replace")[
                            :300
                        ].strip()
                        raise ValueError(
                            "ripgrep rejected the search pattern or source; for literal code, use search --fixed. "
                            + detail
                        )
                    result["coverage"]["scan_bytes"] += size
                    for line in found.stdout.splitlines():
                        entry = json.loads(line)
                        if entry.get("type") == "match":
                            body = entry["data"]
                            text = body["lines"].get("text", "")
                            result["items"].append(
                                {
                                    "path": str(path),
                                    "sha256": source_digest,
                                    "line": body["line_number"],
                                    "text": text.encode()[:512].decode(errors="ignore"),
                                    "line_truncated": len(text.encode()) > 512,
                                }
                            )
                if len(result["items"]) >= config["limits"]["max_matches"] or not _fits(
                    result
                ):
                    result["items"] = result["items"][: config["limits"]["max_matches"]]
                    while result["items"] and not _fits(result):
                        result["items"].pop()
                    _partial(result, "match_or_output_budget")
                    return result
        if proc.wait(timeout=3) not in (0, 1):
            raise ValueError("ripgrep could not enumerate the explicit paths")
        return result
    finally:
        proc.stdout.close()
        if proc.poll() is None:
            proc.terminate()
            try:
                proc.wait(timeout=1)
            except subprocess.TimeoutExpired:
                proc.kill()
                proc.wait()
        result["coverage"]["paths_seen"] = seen
        result["coverage"]["complete"] &= result["status"] == "complete"
        if result["status"] == "partial":
            result["next_action"] = (
                "Narrow paths/glob or read the relevant source lines; a partial search cannot establish absence."
            )
        encode(result)
