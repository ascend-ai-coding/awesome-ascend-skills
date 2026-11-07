# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest

SPEC = importlib.util.spec_from_file_location(
    "pipeline_tasks_under_test", Path(__file__).resolve().parents[1] / "scripts/pipeline_tasks.py"
)
TASKS = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(TASKS)


def add(state, iid="1", digest="a", **kwargs):
    return TASKS.ensure_task(state, iid, "analyze", digest, {"issue": iid}, 100, **kwargs)


def finish(state, worker="w", now=101):
    task = TASKS.claim(state, worker, now)
    return TASKS.submit(state, task["task_id"], task["attempt_id"], {"ok": True}, now + 1)


def test_ensure_is_idempotent_and_snapshots_are_isolated():
    state = {}
    first = add(state)
    first["payload"]["issue"] = "changed"
    again = add(state)
    assert again["payload"]["issue"] == "1"
    assert len(state.get("tasks", {})) == 1
    assert json.loads(json.dumps(state)) == state


def test_priority_long_jobs_first_and_worker_and_global_limits():
    state = {}
    add(state, "1", priority=2, effort=20)
    add(state, "2", priority=1, effort=1)
    add(state, "3", priority=2, effort=100)
    add(state, "4")
    assert TASKS.claim(state, "w1", 101)["iid"] == "2"
    assert TASKS.claim(state, "w1", 101) is None
    assert TASKS.claim(state, "w2", 101)["iid"] == "3"
    assert TASKS.claim(state, "w3", 101)["iid"] == "1"
    assert TASKS.claim(state, "w4", 101) is None


def test_dependency_wait_does_not_consume_slot_and_accept_unblocks():
    state = {}
    parent = add(state)
    child = add(state, "2", dependencies=[parent["task_id"]], priority=0)
    assert child["status"] == "waiting"
    done = finish(state)
    assert TASKS.claim(state, "w2", 103) is None
    TASKS.accept(state, done["task_id"], 104)
    assert TASKS.claim(state, "w2", 105, max_running=1)["task_id"] == child["task_id"]


def test_backpressure_releases_immediately_after_one_review():
    state = {}
    for iid in range(4):
        add(state, str(iid))
    submitted = [finish(state, f"w{i}") for i in range(3)]
    assert TASKS.claim(state, "w4", 104) is None
    TASKS.accept(state, submitted[0]["task_id"], 104)
    assert TASKS.claim(state, "w4", 104) is not None


def test_expired_attempt_cannot_overwrite_replacement():
    state = {}
    add(state)
    old = TASKS.claim(state, "old", 101, lease_seconds=1)
    with pytest.raises(ValueError, match="stale_lease"):
        TASKS.submit(state, old["task_id"], old["attempt_id"], {}, 102)
    new = TASKS.claim(state, "new", 103)
    assert new["attempt_id"] != old["attempt_id"]
    with pytest.raises(ValueError, match="stale_attempt"):
        TASKS.submit(state, old["task_id"], old["attempt_id"], {}, 104)
    TASKS.submit(state, new["task_id"], new["attempt_id"], {}, 104)


def test_submit_retry_is_idempotent_before_and_after_acceptance():
    state = {}
    add(state)
    task = finish(state)
    assert TASKS.submit(state, task["task_id"], task["attempt_id"], {"ok": True}, 200) == task
    with pytest.raises(ValueError, match="result_conflict"):
        TASKS.submit(state, task["task_id"], task["attempt_id"], {}, 200)
    accepted = TASKS.accept(state, task["task_id"], 201)
    assert TASKS.submit(state, task["task_id"], task["attempt_id"], {"ok": True}, 202) == accepted


def test_input_refresh_invalidates_transitive_consumers_and_running_attempt():
    state = {}
    parent = add(state)
    child = add(state, "2", dependencies=[parent["task_id"]])
    grandchild = add(state, "3", dependencies=[child["task_id"]])
    running = TASKS.claim(state, "w", 101)
    replacement = add(state, digest="b")
    assert replacement["status"] == "ready"
    for task in (parent, child, grandchild):
        assert state.get("tasks", {}).get(task["task_id"], {}).get("status") == "superseded"
    with pytest.raises(ValueError, match="stale_input"):
        TASKS.submit(state, running["task_id"], running["attempt_id"], {}, 102)
    assert TASKS.claim(state, "w", 102)["task_id"] == replacement["task_id"]


def test_digest_revisit_never_reuses_attempt_id():
    state = {}
    add(state)
    old = TASKS.claim(state, "old", 101)
    add(state, digest="b")
    add(state, digest="a")
    new = TASKS.claim(state, "new", 102)
    assert new["attempt_id"] != old["attempt_id"]
    with pytest.raises(ValueError, match="stale_attempt"):
        TASKS.submit(state, old["task_id"], old["attempt_id"], {}, 103)


def test_review_requeue_rejects_late_delivery_and_accept_requires_submission():
    state = {}
    add(state)
    done = finish(state)
    queued = TASKS.requeue(state, done["task_id"], "needs evidence", 104)
    assert queued["status"] == "ready"
    assert queued["result"] is None
    with pytest.raises(ValueError, match="stale_attempt"):
        TASKS.submit(state, done["task_id"], done["attempt_id"], {"ok": True}, 105)
    with pytest.raises(ValueError, match="not_submitted"):
        TASKS.accept(state, done["task_id"], 105)


def test_restart_and_status_recover_expired_leases():
    state = {}
    add(state)
    TASKS.claim(state, "w", 101, lease_seconds=2)
    restored = json.loads(json.dumps(state))
    summary = TASKS.status(restored, 103)
    assert summary["counts"]["running"] == 0
    assert summary["counts"]["ready"] == 1
    assert TASKS.expire(restored, 104) == []
    assert TASKS.claim(restored, "w", 104)["attempts"] == 2


def test_dependency_validation_and_identity_encoding():
    state = {}
    with pytest.raises(ValueError, match="unknown_dependency"):
        add(state, dependencies=["missing"])
    first = TASKS.ensure_task(state, "1/2", "a", "b", {}, 100)
    second = TASKS.ensure_task(state, "1", "2/a", "b", {}, 100)
    assert first["task_id"] != second["task_id"]
    child = add(state, dependencies=[first["task_id"]])
    TASKS.ensure_task(state, "1/2", "a", "c", {}, 101)
    with pytest.raises(ValueError, match="dependency_cycle"):
        TASKS.ensure_task(state, "1/2", "a", "b", {}, 102, dependencies=[child["task_id"]])


@pytest.mark.parametrize(
    "kwargs,code",
    [
        ({"lease_seconds": 0}, "invalid_lease"),
        ({"max_running": 0}, "invalid_max_running"),
        ({"review_limit": -1}, "invalid_review_limit"),
    ],
)
def test_invalid_claim_settings(kwargs, code):
    with pytest.raises(ValueError, match=code):
        TASKS.claim({}, "w", 100, **kwargs)


def test_hold_survives_expire_and_restart_until_explicit_requeue():
    state = {}
    add(state)
    done = finish(state)
    held = TASKS.hold(state, done["task_id"], "needs_evidence", 103)
    assert held["held"] and held["status"] == "waiting"
    assert held["worker"] is None and held["lease_expires_at"] is None
    restored = json.loads(json.dumps(state))
    assert TASKS.status(restored, 500)["counts"]["waiting"] == 1
    assert TASKS.claim(restored, "w", 501) is None
    with pytest.raises(ValueError, match="stale_lease"):
        TASKS.submit(restored, done["task_id"], done["attempt_id"], {"ok": True}, 501)
    queued = TASKS.requeue(restored, done["task_id"], "evidence_received", 502)
    assert queued["held"] is False
    assert TASKS.claim(restored, "w", 503)["attempts"] == 2


def test_holding_running_task_frees_slot():
    state = {}
    add(state)
    add(state, "2")
    task = TASKS.claim(state, "w", 101, max_running=1)
    TASKS.hold(state, task["task_id"], "needs_escalation", 102)
    assert TASKS.claim(state, "w", 103, max_running=1)["iid"] == "2"


def test_renew_extends_live_lease_and_refuses_old_attempts():
    state = {}
    add(state)
    task = TASKS.claim(state, "w", 101, lease_seconds=5)
    with pytest.raises(ValueError, match="stale_attempt"):
        TASKS.renew(state, task["task_id"], "wrong", 103)
    renewed = TASKS.renew(state, task["task_id"], task["attempt_id"], 104, lease_seconds=10)
    assert renewed["lease_expires_at"] == 114
    assert TASKS.renew(state, task["task_id"], task["attempt_id"], 105, lease_seconds=1)["lease_expires_at"] == 114
    with pytest.raises(ValueError, match="stale_lease"):
        TASKS.renew(state, task["task_id"], task["attempt_id"], 114)
    assert state.get("tasks", {}).get(task["task_id"], {}).get("status") == "ready"


def test_timing_totals_include_current_phase_and_do_not_double_count():
    state = {}
    add(state)
    task = TASKS.claim(state, "w", 110, lease_seconds=5)
    timings = TASKS.status(state, 112)["timings"]
    assert timings == {"queue_seconds": 10, "execution_seconds": 2, "review_seconds": 0}
    assert TASKS.status(state, 112)["timings"] == timings
    # Execution stops at the deadline; remaining time counts as queued.
    assert TASKS.status(state, 120)["timings"] == {"queue_seconds": 15, "execution_seconds": 5, "review_seconds": 0}
    task = TASKS.claim(state, "w", 120)
    TASKS.submit(state, task["task_id"], task["attempt_id"], {}, 123)
    TASKS.hold(state, task["task_id"], "needs_evidence", 125)
    assert TASKS.status(state, 200)["timings"] == {"queue_seconds": 15, "execution_seconds": 8, "review_seconds": 2}
