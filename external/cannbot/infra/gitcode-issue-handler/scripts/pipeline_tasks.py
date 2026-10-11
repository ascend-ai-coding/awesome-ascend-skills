# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Pure task scheduling transitions. Callers own persistence and exclusive locking.

All times are epoch seconds. Returned tasks are snapshots, not mutable state handles.
Dependencies are task IDs; only accepted dependencies unblock work. Hosts claim a
lease before starting an agent and review submitted results before accepting them.
"""

from __future__ import annotations

from copy import deepcopy
import math
from urllib.parse import quote
from call_options import MISSING, bind_extra

STATUSES = ("ready", "running", "submitted", "accepted", "waiting", "superseded")


def _time(value):
    try:
        value = float(value)
    except (TypeError, ValueError):
        raise ValueError("invalid_time") from None
    if not math.isfinite(value):
        raise ValueError("invalid_time")
    return value


def _tasks(state):
    return state.setdefault("tasks", {})


def _task(state, task_id):
    try:
        return _tasks(state)[task_id]
    except KeyError:
        raise ValueError("unknown_task") from None


def _dependencies_ready(tasks, task):
    return all(tasks.get(dep, {}).get("status") == "accepted" for dep in task["dependencies"])


def _release(task, now):
    task.update(worker=None, lease_expires_at=None, updated_at=now)


def _transition(task, next_status, now):
    timings = task.setdefault("timings", {})
    phase = {"ready": "queue_seconds", "running": "execution_seconds", "submitted": "review_seconds"}.get(
        task["status"]
    )
    if phase:
        timings[phase] = timings.get(phase, 0) + max(0, now - task.get("status_since", task["updated_at"]))
    task.update(status=next_status, status_since=now, updated_at=now)


def _supersede(tasks, ids, now):
    """Invalidate transitive consumers, including already reviewed results."""
    pending = set(ids)
    while pending:
        task_id = pending.pop()
        task = tasks[task_id]
        if task["status"] == "superseded":
            continue
        _transition(task, "superseded", now)
        _release(task, now)
        pending.update(
            t["task_id"] for t in tasks.values() if task_id in t["dependencies"] and t["status"] != "superseded"
        )


def _validated_request(iid, stage, input_digest, priority, effort):
    iid, stage, input_digest = str(iid), str(stage), str(input_digest)
    if not iid or not stage or not input_digest:
        raise ValueError("invalid_identity")
    if not isinstance(priority, (int, float)) or not math.isfinite(priority):
        raise ValueError("invalid_priority")
    if not isinstance(effort, (int, float)) or not math.isfinite(effort) or effort < 0:
        raise ValueError("invalid_effort")
    return iid, stage, input_digest


def _check_dependency_cycle(task_id, deps, tasks):
    # Reusing an old digest must not introduce a cycle through old consumers.
    ancestors = list(deps)
    visited = set()
    while ancestors:
        dep = ancestors.pop()
        if dep == task_id:
            raise ValueError("dependency_cycle")
        if dep not in visited:
            visited.add(dep)
            ancestors.extend(tasks[dep]["dependencies"])


def ensure_task(state, iid, stage, input_digest, payload, *args, **kwargs):
    """Ensure one current task per issue/stage; duplicate inputs are immutable."""
    now, priority, effort, dependencies = bind_extra(
        args, kwargs, ("now", "priority", "effort", "dependencies"), (MISSING, 2, 1, None)
    )
    now = _time(now)
    iid, stage, input_digest = _validated_request(iid, stage, input_digest, priority, effort)
    task_id = "/".join(quote(part, safe="") for part in (iid, stage, input_digest))
    deps = list(dict.fromkeys(dependencies or []))
    tasks = _tasks(state)
    if task_id in deps:
        raise ValueError("dependency_cycle")
    if any(dep not in tasks for dep in deps):
        raise ValueError("unknown_dependency")
    existing = tasks.get(task_id)
    if existing and existing["status"] != "superseded":
        return deepcopy(existing)
    _check_dependency_cycle(task_id, deps, tasks)
    _supersede(tasks, [t["task_id"] for t in tasks.values() if t["iid"] == iid and t["stage"] == stage], now)
    task = dict(
        task_id=task_id,
        iid=iid,
        stage=stage,
        input_digest=input_digest,
        payload=deepcopy(payload),
        priority=priority,
        effort=effort,
        dependencies=deps,
        status="waiting",
        attempt_id=None,
        attempts=existing["attempts"] if existing else 0,
        worker=None,
        lease_expires_at=None,
        created_at=now,
        updated_at=now,
        claimed_at=None,
        submitted_at=None,
        accepted_at=None,
        result=None,
        held=False,
        status_since=now,
        timings={},
    )
    if _dependencies_ready(tasks, task):
        task["status"] = "ready"
    tasks[task_id] = task
    return deepcopy(task)


def expire(state, now):
    """Requeue expired leases and refresh dependency waits; return expired IDs."""
    now = _time(now)
    tasks = _tasks(state)
    expired = []
    for task in tasks.values():
        if task["status"] == "running" and task["lease_expires_at"] <= now:
            expired.append(task["task_id"])
            _transition(task, "ready", task["lease_expires_at"])
            task["reason"] = "lease_expired"
            _release(task, now)
        if task["status"] in ("ready", "waiting") and not task.get("held"):
            next_status = "ready" if _dependencies_ready(tasks, task) else "waiting"
            if task["status"] != next_status:
                _transition(task, next_status, now)
    return expired


def claim(state, worker, now, *args, **kwargs):
    lease_seconds, max_running, review_limit = bind_extra(
        args, kwargs, ("lease_seconds", "max_running", "review_limit"), (300, 3, 3)
    )
    now = _time(now)
    if not worker:
        raise ValueError("invalid_worker")
    lease_seconds = _time(lease_seconds)
    if lease_seconds <= 0:
        raise ValueError("invalid_lease")
    if not isinstance(max_running, int) or max_running < 1:
        raise ValueError("invalid_max_running")
    if not isinstance(review_limit, int) or review_limit < 1:
        raise ValueError("invalid_review_limit")
    expire(state, now)
    tasks = _tasks(state)
    running = [t for t in tasks.values() if t["status"] == "running"]
    if len(running) >= max_running or any(t["worker"] == worker for t in running):
        return None
    if sum(t["status"] == "submitted" for t in tasks.values()) >= review_limit:
        return None
    ready = [t for t in tasks.values() if t["status"] == "ready"]
    if not ready:
        return None
    task = min(ready, key=lambda t: (t["priority"], -t["effort"], t["created_at"], t["task_id"]))
    task["attempts"] += 1
    _transition(task, "running", now)
    task.update(
        worker=worker,
        claimed_at=now,
        updated_at=now,
        attempt_id=f"{task['task_id']}@{task['attempts']}",
        lease_expires_at=now + lease_seconds,
    )
    return deepcopy(task)


def renew(state, task_id, attempt_id, now, lease_seconds=300):
    """Extend a live attempt's lease; expired attempts cannot be resurrected."""
    now = _time(now)
    lease_seconds = _time(lease_seconds)
    if lease_seconds <= 0:
        raise ValueError("invalid_lease")
    expire(state, now)
    task = _task(state, task_id)
    if task["status"] == "superseded":
        raise ValueError("stale_input")
    if not attempt_id or task["attempt_id"] != attempt_id:
        raise ValueError("stale_attempt")
    if task["status"] != "running":
        raise ValueError("stale_lease")
    task.update(lease_expires_at=max(task["lease_expires_at"], now + lease_seconds), updated_at=now)
    return deepcopy(task)


def submit(state, task_id, attempt_id, result, now):
    now = _time(now)
    expire(state, now)
    task = _task(state, task_id)
    if task["status"] == "superseded":
        raise ValueError("stale_input")
    if not attempt_id or task["attempt_id"] != attempt_id:
        raise ValueError("stale_attempt")
    if task["status"] in ("submitted", "accepted"):
        if task["result"] != result:
            raise ValueError("result_conflict")
        return deepcopy(task)
    if task["status"] != "running":
        raise ValueError("stale_lease")
    _transition(task, "submitted", now)
    task.update(result=deepcopy(result), submitted_at=now)
    _release(task, now)
    return deepcopy(task)


def accept(state, task_id, now):
    now = _time(now)
    task = _task(state, task_id)
    if task["status"] == "accepted":
        return deepcopy(task)
    if task["status"] != "submitted":
        raise ValueError("not_submitted")
    _transition(task, "accepted", now)
    task.update(accepted_at=now)
    expire(state, now)
    return deepcopy(task)


def requeue(state, task_id, reason, now):
    now = _time(now)
    task = _task(state, task_id)
    if task["status"] in ("accepted", "superseded"):
        raise ValueError("terminal_task")
    _transition(task, "ready", now)
    task.update(held=False, reason=str(reason), attempt_id=None, submitted_at=None, result=None)
    _release(task, now)
    expire(state, now)
    return deepcopy(task)


def hold(state, task_id, reason, now):
    """Wait for an external event until explicit requeue, without using a slot."""
    now = _time(now)
    task = _task(state, task_id)
    if task["status"] in ("accepted", "superseded"):
        raise ValueError("terminal_task")
    _transition(task, "waiting", now)
    task.update(held=True, reason=str(reason))
    _release(task, now)
    return deepcopy(task)


def status(state, now):
    """Return grouped snapshots and cumulative queue/execution/review seconds.

    Timings include the current phase, sum retries, and exclude external and
    dependency waits. Expired execution ends at its lease deadline.
    """
    now = _time(now)
    expire(state, now)
    groups = {name: [] for name in STATUSES}
    totals = dict(queue_seconds=0, execution_seconds=0, review_seconds=0)
    for task in _tasks(state).values():
        snapshot = deepcopy(task)
        phase = {"ready": "queue_seconds", "running": "execution_seconds", "submitted": "review_seconds"}.get(
            task["status"]
        )
        timings = snapshot.setdefault("timings", {})
        if phase:
            timings[phase] = timings.get(phase, 0) + max(0, now - task.get("status_since", task["updated_at"]))
        for key in totals:
            totals[key] += snapshot["timings"].get(key, 0)
        groups[task["status"]].append(snapshot)
    return {"counts": {name: len(tasks) for name, tasks in groups.items()}, "timings": totals, **groups}
