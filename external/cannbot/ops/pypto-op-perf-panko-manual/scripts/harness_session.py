# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
"""Reuse the CPU harness and metadata; retain its commands and stop/report gates."""

import argparse
import builtins
import contextlib
import importlib
import io
import json
import os
from pathlib import Path
import shutil
import sys
import traceback

import session_worker


def run_main(function):
    """Retain the original harness exit code without stopping the worker."""
    try:
        function()
    except builtins.SystemExit as error:
        if error.code is None or isinstance(error.code, int):
            return error.code or 0
        traceback.print_exception(error, limit=0)
        return 1
    return 0


def context_fingerprint(config):
    """Restart the harness when scripts, runtime or search generation change."""
    epoch = Path(config["op_dir"]) / "optimization" / ".evaluation" / "epoch.json"
    context = session_worker.DependencyContext(
        (Path(__file__).parent,),
        epoch,
        ignored_names=("tuning_notes.md",),
        ignored_dirs=("optimization", "output", "prof", "__pycache__"),
    )
    return session_worker.context_fingerprint(config, context)


def create_handler(config):
    """Initialize harness-local state and retain its imported command module."""
    os.environ["TILE_FWK_DEVICE_ID"] = str(config["device"])
    return lambda argv: execute(config, argv)


def execute(config, argv):
    """Run the original main through its persistent module and original exit handling."""
    stdout, stderr = io.StringIO(), io.StringIO()
    previous = os.getcwd(), sys.argv, sys.path[:], os.environ.copy()
    with contextlib.redirect_stdout(stdout), contextlib.redirect_stderr(stderr):
        try:
            harness = importlib.import_module("panko_harness")
            sys.argv = [harness.__file__, *argv]
            code = run_main(harness.main)
        except Exception:
            traceback.print_exc()
            code = 1
        finally:
            os.chdir(previous[0])
            sys.argv, sys.path = previous[1:3]
            os.environ.clear()
            os.environ.update(previous[3])
    result = dict(returncode=code, stdout=stdout.getvalue(), stderr=stderr.getvalue())
    if code == 0 and argv and argv[0] == "record":
        result["terminal"] = complete_stop(config)
    elif code == 0 and argv and argv[0] == "stop":
        verdict = json.loads(result["stdout"])
        result["finished"] = verdict["stop"]
        result["terminal"] = complete_stop(config, verdict)
    return result


def complete_stop(config, verdict=None):
    """Create a terminal marker only after the original restoration and report finish."""
    harness = importlib.import_module("panko_harness")
    stdout = io.StringIO()
    arguments = argparse.Namespace(op_dir=config["op_dir"], force="")
    with contextlib.redirect_stdout(stdout):
        if verdict is None:
            harness.cmd_stop(arguments)
            verdict = json.loads(stdout.getvalue())
        if (
            not verdict["stop"]
            or verdict.get("forced")
            or not verdict.get("delivered_is_best")
        ):
            return None
        harness.cmd_report(arguments)
    state = harness.load(config["op_dir"])
    terminal = dict(
        directive="STOP",
        reason=state["progress"]["stop_reason"],
        run_start=state["progress"].get("_start_ts"),
        delivered_code_hash=verdict.get("delivered_code_hash"),
    )
    path = Path(config["op_dir"]) / "optimization" / "controller-terminal.json"
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(terminal))
    temporary.replace(path)
    return terminal


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--op-dir", required=True)
    parser.add_argument("--close", action="store_true")
    parser.add_argument("--timeout", type=float, default=3600)
    parser.add_argument("arguments", nargs=argparse.REMAINDER)
    args = parser.parse_args()
    root = Path(args.op_dir).resolve() / "optimization" / ".harness"
    if args.close:
        session_worker.close_worker(root)
        session_worker.close_worker(
            Path(args.op_dir).resolve() / "optimization" / ".evaluation"
        )
        return 0
    argv = args.arguments[1:] if args.arguments[:1] == ["--"] else args.arguments
    if not argv:
        parser.error("an original harness command is required after --")
    if "--op-dir" in argv:
        index = argv.index("--op-dir")
        if (
            index + 1 == len(argv)
            or Path(argv[index + 1]).resolve() != Path(args.op_dir).resolve()
        ):
            parser.error("command and session must address the same operator directory")
    config = dict(
        worker_module="harness_session",
        op_dir=str(Path(args.op_dir).resolve()),
        cwd=os.getcwd(),
        device=os.environ.get("TILE_FWK_DEVICE_ID", "0"),
    )
    fingerprint = context_fingerprint(config)
    result = session_worker.dispatch(root, config, fingerprint, argv, args.timeout)
    if "returncode" not in result:
        session_worker.close_worker(
            Path(args.op_dir).resolve() / "optimization" / ".evaluation"
        )
        result = dict(
            returncode=result["test_rc"],
            stdout="",
            stderr=result.get("test_output", ""),
        )
    shutil.copyfileobj(io.StringIO(result["stdout"]), sys.stdout)
    shutil.copyfileobj(io.StringIO(result["stderr"]), sys.stderr)
    if result.get("terminal") or result.get("finished"):
        session_worker.close_worker(root)
    return result["returncode"]


if __name__ == "__main__":
    sys.exit(main())
