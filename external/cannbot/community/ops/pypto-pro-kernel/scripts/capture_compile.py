#!/usr/bin/env python3
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

"""Retain successful Bisheng diagnostics without modifying the installed JIT."""

import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import shlex
import shutil
import subprocess
import sys
import time
import uuid


def _sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _write(path, value):
    path.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")


def _inputs(arguments):
    # This is a bounded source snapshot, not a compiler dependency scanner.
    files = set()
    for value in arguments:
        path = Path(value)
        if path.is_file() and path.suffix in {".cpp", ".cce", ".cc", ".c"}:
            files.add(path.resolve())
            for suffix in ("*.h", "*.hpp", "*.cpp", "*.cce"):
                files.update(
                    p.resolve() for p in path.parent.glob(suffix) if p.is_file()
                )
    return {str(p): _sha(p) for p in sorted(files)}


def _output(arguments):
    values = [arguments[i + 1] for i, value in enumerate(arguments[:-1]) if value == "-o"]
    return Path(values[0]).resolve() if len(values) == 1 else None


def _resources(text):
    pattern = (
        r"\[BISHENG\] SIMT VF Function properties for ([^\r\n]+?):\s*"
        r"Stack size:\s*(\d+) bytes,\s*Used register number:\s*(\d+)"
    )
    return [
        {"symbol": symbol, "stack_bytes": int(stack), "used_registers": int(registers)}
        for symbol, stack, registers in re.findall(pattern, text)
    ]


def _record(config_path, arguments):
    config = json.loads(Path(config_path).read_text(encoding="utf-8"))
    compiler = Path(config["compiler"])
    compiler_before = _sha(compiler)
    inputs = _inputs(arguments)
    output = _output(arguments)
    output_was_absent = not output.exists() if output else None
    effective = list(arguments)
    if "--cce-res-usage" not in effective:
        effective.append("--cce-res-usage")
    start = time.time_ns()
    result = subprocess.run([str(compiler), *effective], capture_output=True)
    unchanged = all(Path(p).is_file() and _sha(p) == sha for p, sha in inputs.items())
    compiler_after = _sha(compiler)
    record = {
        "schema": "pypto-compiler-capture/1",
        "started_ns": start,
        "elapsed_seconds": (time.time_ns() - start) / 1e9,
        "cwd": os.getcwd(),
        "compiler": str(compiler),
        "compiler_sha256": compiler_before,
        "compiler_unchanged": compiler_before
        == compiler_after
        == config["compiler_sha256"],
        "original_arguments": arguments,
        "effective_arguments": effective,
        "inputs": inputs,
        "input_scope": "translation_units_and_sibling_headers",
        "inputs_unchanged": unchanged,
        "returncode": result.returncode,
        "stdout": result.stdout.decode("utf-8", errors="replace"),
        "stderr": result.stderr.decode("utf-8", errors="replace"),
        "output": str(output) if output else None,
        "output_was_absent": output_was_absent,
        "output_sha256": _sha(output)
        if result.returncode == 0 and output_was_absent and output.is_file()
        else None,
    }
    record["resources"] = _resources(record["stdout"] + "\n" + record["stderr"])
    _write(Path(config["records"]) / (uuid.uuid4().hex + ".json"), record)
    sys.stdout.buffer.write(result.stdout)
    sys.stderr.buffer.write(result.stderr)
    return result.returncode


def main(argv=None):
    argv = sys.argv[1:] if argv is None else argv
    if argv and argv[0] == "_record":
        return _record(argv[1], argv[2:])
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--compiler",
        required=True,
        help="Real Bisheng executable, resolved before PATH interception",
    )
    parser.add_argument(
        "--output",
        required=True,
        type=Path,
        help="New evidence directory; existing directories are refused",
    )
    parser.add_argument(
        "command",
        nargs=argparse.REMAINDER,
        help="-- followed by the native runner command",
    )
    args = parser.parse_args(argv)
    command = args.command[1:] if args.command[:1] == ["--"] else args.command
    if not command:
        parser.error("a runner command is required after --")
    found = shutil.which(args.compiler)
    if not found:
        parser.error("compiler executable was not found")
    compiler = Path(found).resolve()
    capability = subprocess.run([str(compiler), "--help"], capture_output=True)
    if (
        capability.returncode != 0
        or b"--cce-res-usage" not in capability.stdout + capability.stderr
    ):
        parser.error("installed compiler did not advertise --cce-res-usage")
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    (output / "bin").mkdir()
    (output / "records").mkdir()
    config = {
        "compiler": str(compiler),
        "compiler_sha256": _sha(compiler),
        "records": str(output / "records"),
    }
    config_path = output / "config.json"
    _write(config_path, config)
    shim = output / "bin" / "bisheng"
    shim.write_text(
        "#!/bin/sh\nexec "
        + " ".join(
            shlex.quote(str(p))
            for p in (sys.executable, Path(__file__).resolve(), "_record", config_path)
        )
        + ' "$@"\n',
        encoding="utf-8",
    )
    shim.chmod(0o755)
    environment = dict(
        os.environ, PATH=str(output / "bin") + os.pathsep + os.environ.get("PATH", "")
    )
    result = subprocess.run(command, env=environment)
    records = sorted((output / "records").glob("*.json"))
    _write(
        output / "manifest.json",
        {
            "schema": "pypto-compiler-session/1",
            "command": command,
            "returncode": result.returncode,
            "script_sha256": _sha(__file__),
            "compiler": config,
            "captures": [
                {"path": str(p.relative_to(output)), "sha256": _sha(p)} for p in records
            ],
            "status": "captured" if records else "unavailable",
            "execution_binding": "not_established",
        },
    )
    print(
        json.dumps(
            {"manifest": str(output / "manifest.json"), "captures": len(records)}
        ),
        file=sys.stderr,
    )
    return result.returncode


if __name__ == "__main__":
    raise SystemExit(main())
