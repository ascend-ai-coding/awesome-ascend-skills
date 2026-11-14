# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
"""Private resident worker transport and lifecycle shared by session clients."""

import argparse
from dataclasses import dataclass
import fcntl
import hashlib
import importlib
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import shutil
import signal
import subprocess
import sys
import tempfile
import time
import traceback
from typing import Optional

_CHILDREN = {}


@dataclass(frozen=True)
class DependencyContext:
    """Client-selected files, generation and exclusions for worker reuse."""

    folders: tuple[Path, ...]
    epoch: Path
    excluded_file: Optional[str] = None
    ignored_names: tuple[str, ...] = ()
    ignored_dirs: tuple[str, ...] = ()


def write_json(path, value):
    with tempfile.NamedTemporaryFile(
        mode="w",
        encoding="utf-8",
        dir=path.parent,
        prefix=path.name,
        suffix=".tmp",
        delete=False,
    ) as file:
        temporary = Path(file.name)
        json.dump(value, file)
    try:
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def worker_alive(session):
    """Whether this session still owns its recorded process."""
    if not (session / "pid").exists():
        return False
    pid = int((session / "pid").read_text())
    result = subprocess.run(
        [shutil.which("ps"), "-p", str(pid), "-o", "args="],
        capture_output=True,
        text=True,
        check=False,
    )
    return "--serve" in result.stdout and str(session) in result.stdout


def _stop(session):
    if worker_alive(session):
        pid = int((session / "pid").read_text())
        os.killpg(pid, signal.SIGTERM)
        deadline = time.monotonic() + 2
        while worker_alive(session) and time.monotonic() < deadline:
            time.sleep(0.05)
        if worker_alive(session):
            os.killpg(pid, signal.SIGKILL)
    if (session / "pid").exists():
        process = _CHILDREN.pop(int((session / "pid").read_text()), None)
        if process is not None:
            process.wait(timeout=5)


def close_worker(root):
    """Close a private queue worker using the same ownership and lock checks."""
    if root.exists():
        with (root / "lock").open("a") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX)
            if (root / "active.json").exists():
                _stop(Path(json.loads((root / "active.json").read_text())["session"]))


def _file_digest(path):
    digest = hashlib.sha256()
    with path.open("rb") as file:
        while chunk := file.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def context_fingerprint(config, context):
    """Hash the client-selected dependencies, generation and runtime context."""
    dependencies = {}
    for folder in context.folders:
        for path in folder.rglob("*"):
            if not path.is_file():
                continue
            if path.name in context.ignored_names or any(
                part in context.ignored_dirs for part in path.parts
            ):
                continue
            if str(path.resolve()) != context.excluded_file:
                dependencies[str(path.resolve())] = _file_digest(path)
    versions = {}
    for package in ("pypto", "torch", "torch-npu"):
        try:
            versions[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            versions[package] = None
    generation = (
        hashlib.sha256(context.epoch.read_bytes()).hexdigest()
        if context.epoch.exists()
        else None
    )
    identity = dict(
        config,
        dependencies=dependencies,
        epoch=generation,
        interpreter=sys.executable,
        python=sys.version,
        runtime=versions,
        host=platform.uname(),
        environment=dict(os.environ),
    )
    return hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()


def dispatch(root, config, fingerprint, payload, timeout):
    """Submit a serialized request to a private, ownership-checked worker."""
    root.mkdir(parents=True, exist_ok=True, mode=0o700)
    with (root / "lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        active = (
            json.loads((root / "active.json").read_text())
            if (root / "active.json").exists()
            else {}
        )
        session = Path(active.get("session", root / "absent"))
        if active.get("fingerprint") != fingerprint or not worker_alive(session):
            _stop(session)
            session = Path(tempfile.mkdtemp(prefix="worker-", dir=root))
            write_json(session / "config.json", config)
            with (session / "worker.log").open("w") as log:
                process = subprocess.Popen(
                    [sys.executable, __file__, "--serve", str(session)],
                    stdout=log,
                    stderr=log,
                    start_new_session=True,
                )
            _CHILDREN[process.pid] = process
            (session / "pid").write_text(str(process.pid))
            write_json(
                root / "active.json",
                dict(session=str(session), fingerprint=fingerprint),
            )
        response = session / "response.json"
        response.unlink(missing_ok=True)
        write_json(session / "request.json", payload)
        deadline = time.monotonic() + timeout
        while (
            not response.exists()
            and worker_alive(session)
            and time.monotonic() < deadline
        ):
            time.sleep(0.05)
        if not response.exists():
            _stop(session)
            return {
                "test_rc": 124,
                "test_output": (session / "worker.log").read_text()[-9000:],
            }
        return json.loads(response.read_text())


def serve(session):
    """Serve requests through a handler initialized once by the selected client."""
    config = json.loads((session / "config.json").read_text())
    os.chdir(config["cwd"])
    handler = importlib.import_module(config["worker_module"]).create_handler(config)
    request_file = session / "request.json"
    idle_deadline = time.monotonic() + 600
    while time.monotonic() < idle_deadline:
        if not request_file.exists():
            time.sleep(0.05)
            continue
        payload = json.loads(request_file.read_text())
        request_file.unlink()
        try:
            result = handler(payload)
        except Exception:
            result = {"test_rc": 1, "test_output": traceback.format_exc()}
        write_json(session / "response.json", result)
        idle_deadline = time.monotonic() + 600


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--serve", required=True)
    args = parser.parse_args()
    serve(Path(args.serve))


if __name__ == "__main__":
    main()
