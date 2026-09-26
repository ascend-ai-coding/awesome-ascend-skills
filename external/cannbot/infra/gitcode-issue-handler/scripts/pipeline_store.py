#!/usr/bin/env python3
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# Licensed under the CANN Open Software License Agreement Version 2.0.
"""A repository lock and atomic durable queue, with conservative recovery."""

from __future__ import annotations

import copy
import fcntl
import json
import shutil
import os
import tempfile
import time
import uuid
from pathlib import Path

from fetch_cache import _atomic_write_json


def new_state(repo):
    return {
        "version": 1,
        "repository": repo,
        "generation": 0,
        "issues": {},
        "tasks": {},
        "operations": {},
        "scan": {},
        "metrics": [],
        "migration_done": False,
        "needs_full_scan": True,
    }


def _valid_issue(iid, entry):
    if not isinstance(entry, dict) or not isinstance(entry.get("issue"), dict):
        return False
    valid_phases = {"refresh", "scope", "classify", "attention", "waiting", "observe", "closed", "ignored"}
    return str(entry["issue"].get("iid")) == iid and entry.get("phase") in valid_phases


def _valid_task(key, task, issues):
    if not isinstance(task, dict):
        return False
    valid_statuses = {"ready", "running", "submitted", "accepted", "waiting", "superseded"}
    identity_valid = task.get("task_id") == key and task.get("iid") in issues and task.get("status") in valid_statuses
    shape_valid = (
        isinstance(task.get("dependencies"), list)
        and isinstance(task.get("attempts"), int)
        and isinstance(task.get("payload"), dict)
        and isinstance(task.get("created_at"), (int, float))
    )
    lease_valid = task.get("status") != "running" or isinstance(task.get("lease_expires_at"), (int, float))
    return identity_valid and shape_valid and lease_valid


class PipelineStore:
    def __init__(self, root, repo):
        self.root = Path(root)
        self.repo = repo
        self.path = self.root / "data" / "pipeline-state.json"
        self.backup = self.path.with_suffix(".previous.json")
        self.lock = None

    def __enter__(self):
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.lock = self.path.with_suffix(".lock").open("a+")
        try:
            fcntl.flock(self.lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            self.lock.close()
            raise ValueError("pipeline_busy") from None
        return self

    def __exit__(self, *_):
        fcntl.flock(self.lock.fileno(), fcntl.LOCK_UN)
        self.lock.close()

    def load(self):
        if not self.path.exists():
            return new_state(self.repo)
        try:
            return self._read(self.path)
        except (ValueError, OSError):
            preserved = self.path.with_suffix(f".invalid-{uuid.uuid4().hex}.json")
            shutil.copy2(self.path, preserved)
            try:
                state = self._read(self.backup)
            except (ValueError, OSError):
                state = new_state(self.repo)
            state["needs_full_scan"] = True
            state["migration_done"] = False
            state["recovery"] = {"preserved": str(preserved), "at": time.time(), "requires_operation_audit": True}
            return state

    def save(self, state):
        if state.get("repository") != self.repo:
            raise ValueError("repository_mismatch")
        # Preserve only a valid previous generation. A damaged file must not
        # replace the last usable checkpoint.
        if self.path.exists():
            try:
                old = self._read(self.path)
            except (ValueError, OSError):
                old = None
            if old is not None:
                _atomic_write_json(str(self.backup), old)
        state["generation"] += 1
        _atomic_write_json(str(self.path), state)

    def artifact(self, relative, value):
        path = self.root / relative
        _atomic_write_json(str(path), copy.deepcopy(value))
        return str(path)

    def text_artifact(self, relative, text):
        path = self.root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = None
        try:
            with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", dir=path.parent, delete=False) as stream:
                temporary = stream.name
                stream.write(text)
                stream.flush()
                os.fsync(stream.fileno())
            os.replace(temporary, path)
        finally:
            if temporary and os.path.exists(temporary):
                os.unlink(temporary)
        return str(path)

    def _read(self, path):
        value = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(value, dict):
            raise ValueError("invalid_state")
        if value.get("repository") != self.repo:
            raise RuntimeError("repository_mismatch")
        if value.get("version") != 1 or any(
            not isinstance(value.get(key), dict) for key in ("issues", "tasks", "scan", "operations")
        ):
            raise ValueError("unsupported_or_invalid_state")
        metadata_types = (
            ("generation", int),
            ("migration_done", bool),
            ("needs_full_scan", bool),
            ("metrics", list),
        )
        if not all(isinstance(value.get(key), expected) for key, expected in metadata_types):
            raise ValueError("invalid_state_metadata")
        for iid, entry in value["issues"].items():
            if not _valid_issue(iid, entry):
                raise ValueError("invalid_issue_state")
        for key, task in value["tasks"].items():
            if not _valid_task(key, task, value["issues"]):
                raise ValueError("invalid_task_state")
        if any(not isinstance(op, dict) for op in value["operations"].values()):
            raise ValueError("invalid_operation_state")
        return value
