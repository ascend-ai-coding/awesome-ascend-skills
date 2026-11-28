# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

"""Identity, bounded waits and honest process outcomes without relaunching work."""

import hashlib
import json
import threading
import time

import pytest

from _checks import execution_status


def fixture(tmp_path, **changes):
    source = tmp_path / "candidate.py"
    source.write_text("VALUE = 41\n")
    receipt = tmp_path / "receipt.json"
    data = {
        "request_id": "job-one",
        "source_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
        "state": "running",
        **changes,
    }
    receipt.write_text(json.dumps(data))
    return source, receipt, data


@pytest.mark.parametrize("code", [0, 7, -15])
def test_terminal_outcome_preserves_code_without_kernel_acceptance(tmp_path, code):
    source, receipt, _ = fixture(tmp_path, state="completed", exit_code=code)
    result = execution_status.inspect(
        receipt, request_id="job-one", source=source, wait_seconds=0
    )
    assert result["terminal"] and result["exit_code"] == code
    assert result["outcome"] == ("passed" if code == 0 else "failed")
    assert set(result["validation"].values()) == {"not_established"}
    assert result["receipt_reads"] == 1


def test_wait_reads_the_same_request_until_terminal(tmp_path):
    source, receipt, data = fixture(tmp_path)

    def complete():
        time.sleep(0.08)
        replacement = receipt.with_suffix(".tmp")
        replacement.write_text(
            json.dumps({**data, "state": "completed", "exit_code": 0})
        )
        replacement.replace(receipt)

    worker = threading.Thread(target=complete)
    worker.start()
    result = execution_status.inspect(
        receipt,
        request_id="job-one",
        source=source,
        wait_seconds=0.5,
        poll_seconds=0.05,
    )
    worker.join()
    assert result["terminal"] and 2 <= result["receipt_reads"] <= 5
    assert result["request_id"] == "job-one"


@pytest.mark.parametrize(
    "changes", [{"request_id": "another"}, {"source_sha256": "0" * 64}]
)
def test_wrong_request_or_source_does_not_become_a_pass(tmp_path, changes):
    source, receipt, _ = fixture(tmp_path, state="completed", exit_code=0, **changes)
    result = execution_status.inspect(
        receipt, request_id="job-one", source=source, wait_seconds=0
    )
    assert not result["terminal"] and result["outcome"] == "unknown"
    assert result["reason"] == "receipt_identity_mismatch"


@pytest.mark.parametrize("content", [None, "{", "[]", '{"state":"running"}'])
def test_missing_or_invalid_receipt_is_unverified(tmp_path, content):
    source, receipt, _ = fixture(tmp_path)
    if content is None:
        receipt.unlink()
    else:
        receipt.write_text(content)
    result = execution_status.inspect(
        receipt, request_id="job-one", source=source, wait_seconds=0
    )
    assert not result["terminal"] and result["outcome"] == "unknown"


def test_running_wait_budget_is_pending_not_failed(tmp_path):
    source, receipt, _ = fixture(tmp_path)
    result = execution_status.inspect(
        receipt,
        request_id="job-one",
        source=source,
        wait_seconds=0.11,
        poll_seconds=0.05,
    )
    assert result["outcome"] == "pending" and not result["terminal"]
    assert result["elapsed_seconds"] < 0.5 and result["receipt_reads"] <= 4


def test_source_change_during_wait_invalidates_receipt(tmp_path):
    source, receipt, _ = fixture(tmp_path)
    worker = threading.Thread(
        target=lambda: (time.sleep(0.05), source.write_text("VALUE = 42\n"))
    )
    worker.start()
    result = execution_status.inspect(
        receipt,
        request_id="job-one",
        source=source,
        wait_seconds=0.4,
        poll_seconds=0.05,
    )
    worker.join()
    assert result["reason"] == "source_changed" and result["outcome"] == "unknown"


@pytest.mark.parametrize("seconds", [-1, 31, float("nan"), float("inf")])
def test_wait_limits_are_enforced(tmp_path, seconds):
    source, receipt, _ = fixture(tmp_path)
    with pytest.raises(ValueError):
        execution_status.inspect(
            receipt, request_id="job-one", source=source, wait_seconds=seconds
        )
