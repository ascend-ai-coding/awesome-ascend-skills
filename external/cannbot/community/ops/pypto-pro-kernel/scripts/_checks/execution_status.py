# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

"""Read one existing execution receipt with bounded waiting; never dispatch work."""

from __future__ import annotations

import hashlib
import json
import math
import time
from pathlib import Path

SCHEMA = "pypto-execution-status/1"
MAX_RECEIPT_BYTES = 65536
TERMINAL = {"completed", "failed", "cancelled"}


def inspect(receipt, *, request_id, source, wait_seconds=20, poll_seconds=1):
    """Bind every observed state to the requested identity, not a PID or exit code alone."""
    if not isinstance(request_id, str) or not 1 <= len(request_id) <= 256:
        raise ValueError("request_id must contain 1..256 characters")
    if (
        isinstance(wait_seconds, bool)
        or not math.isfinite(wait_seconds)
        or not 0 <= wait_seconds <= 30
    ):
        raise ValueError("wait_seconds must be between 0 and 30")
    if (
        isinstance(poll_seconds, bool)
        or not math.isfinite(poll_seconds)
        or not 0.05 <= poll_seconds <= 5
    ):
        raise ValueError("poll_seconds must be between 0.05 and 5")
    receipt, source = Path(receipt).resolve(), Path(source).resolve()
    if not source.is_file():
        raise ValueError("Status source must be a regular file")
    original = hashlib.sha256(source.read_bytes()).hexdigest()
    result = {
        "schema": SCHEMA,
        "status": "unverified",
        "request_id": request_id,
        "source": {"path": str(source), "sha256": original},
        "receipt": str(receipt),
        "state": None,
        "exit_code": None,
        "outcome": "unknown",
        "terminal": False,
        "receipt_reads": 0,
        "elapsed_seconds": 0,
        "validation": {
            "native": "not_established",
            "numerical": "not_established",
            "performance": "not_established",
        },
    }
    start = time.monotonic()
    deadline = start + wait_seconds
    last_error = None
    while True:
        if hashlib.sha256(source.read_bytes()).hexdigest() != original:
            result["reason"] = "source_changed"
            break
        result["receipt_reads"] += 1
        try:
            if receipt.exists() and not receipt.is_file():
                result["reason"] = "receipt_not_regular_file"
                break
            with receipt.open("rb") as stream:
                payload = stream.read(MAX_RECEIPT_BYTES + 1)
            if len(payload) > MAX_RECEIPT_BYTES:
                result["reason"] = "receipt_size_limit"
                break
            data = json.loads(payload)
            if not isinstance(data, dict):
                result["reason"] = "invalid_receipt"
                break
        except (FileNotFoundError, json.JSONDecodeError, UnicodeDecodeError) as exc:
            # Missing/partially replaced receipts remain unresolved, never completed failures.
            last_error = type(exc).__name__
        else:
            if (
                data.get("request_id") != request_id
                or data.get("source_sha256") != original
            ):
                result["reason"] = "receipt_identity_mismatch"
                break
            state = data.get("state")
            if state not in {"queued", "running", *TERMINAL}:
                result["reason"] = "unknown_receipt_state"
                break
            result["state"] = state
            result["receipt_sha256"] = hashlib.sha256(payload).hexdigest()
            if state in TERMINAL:
                code = data.get("exit_code")
                if not isinstance(code, int) or isinstance(code, bool):
                    result["reason"] = "terminal_exit_missing"
                    break
                if state in {"failed", "cancelled"} and code == 0:
                    result["reason"] = "inconsistent_terminal_receipt"
                    break
                result.update(
                    status="observed",
                    terminal=True,
                    exit_code=code,
                    outcome="passed" if code == 0 else "failed",
                )
                result["next_action"] = (
                    "Inspect this request's verifier evidence; process completion alone"
                    " does not establish kernel correctness or performance."
                )
                break
            result["outcome"] = "pending"
            last_error = None
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            result["reason"] = "wait_budget_exhausted"
            if last_error:
                result["receipt_error"] = last_error
            break
        time.sleep(min(poll_seconds, remaining))
    # Do not accept a receipt for a source edited during the final read.
    if hashlib.sha256(source.read_bytes()).hexdigest() != original:
        result.update(
            status="unverified",
            terminal=False,
            outcome="unknown",
            exit_code=None,
            reason="source_changed",
        )
    result["elapsed_seconds"] = round(time.monotonic() - start, 3)
    if result["status"] == "unverified":
        result["next_action"] = (
            "Keep this request and its checked source. Reconcile the existing receipt/controller; "
            "a missing, changed or still-running receipt is not permission to launch another verifier."
        )
    return result
