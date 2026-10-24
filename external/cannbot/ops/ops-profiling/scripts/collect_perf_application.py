#!/usr/bin/env python3
# ----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ----------------------------------------------------------------------------------------------------------
"""Collect a complete case set with one ``msprof application`` session series.

The application runner must execute exactly ten adjacent pairs per case::

    warmup -> synchronize -> target operator -> synchronize

Only the sum of target-kernel task durations in each pair is a sample.  The
reported ``total_us`` is the median of the ten samples.  The runner manifest is
the source of truth for launch topology; unrelated framework kernels are
ignored, while missing, duplicated, or reordered relevant launches fail the
collection.

Before profiling, ``--ops-warmup-probe-cmd`` verifies that the runner can use
the preferred framework operator on the target device.  ArgMax is preferred;
Max is used for an ArgMax-family target so warmup and target remain distinct.
Only a failed probe triggers ``--warmup-build-cmd`` and the bundled compiled
warmup.  Failure of both paths fails the collection.

CANN environment adaptation (measured on Ascend950PR, CANN 9.0.0 build
20260422 and an incomplete CANN 9.1.0 install, build 20260610):

* ``--aic-mode=task-based`` is version related: CANN 9.0.0 only exports
  ``op_summary`` when it is passed explicitly, while 9.1.0 exports it either
  way (verified harmless with the flag).  The collector always passes it and
  records the resolved CANN identity under ``cann_environment``.
* Incomplete CANN installations (ops packages missing) may lack the opapi
  library family (``libopapi*.so``) and the system aclnn warmup symbols, in
  which case framework-ops warmup is impossible and only the compiled warmup
  can serve as fallback.  The collector inventories the installation before
  profiling and records it, so probe failures become diagnosable instead of
  surfacing as cryptic runtime errors (e.g. ACLNN 561103 / torch_npu
  std::bad_alloc on a torch_npu built against a different CANN).
* Each msprof session is capped at ``--max-launches-per-session`` relevant
  launches (default 720).  Above roughly 720-840 relevant op records per
  session, the CANN 9.0.0 analysis pipeline silently exports an *empty*
  ``op_summary`` while msprof still exits 0.  Larger case sets are
  automatically split into consecutive sessions whose results are merged and
  validated as one collection.  The rendered ``--application`` value is
  additionally kept below msprof's 2048-character hard limit.
"""

from __future__ import annotations

import argparse
import csv
import datetime as dt
import hashlib
import json
import os
import shlex
import shutil
import statistics
import subprocess
import sys
import tempfile
import time
from pathlib import Path

SAMPLES_PER_CASE = 10
FALLBACK_TENSOR_BYTES = 256 * 1024 * 1024
EXECUTION_MANIFEST_SCHEMA_VERSION = 5
MANIFEST_SCHEMA_VERSION = 5
VALIDATION_SCHEMA_VERSION = 3
COMPILED_WARMUP_IMPLEMENTATION = "ops_profiling_warmup_v1"
# CANN 9.0.0 (build 20260422, Ascend950PR) analysis-pipeline flakiness, measured
# on identical msprof invocations: 540 relevant op records per session have never
# failed (frozen-protocol scale, 10+ runs); 720 passed twice but failed once with
# a silently empty op_summary (aging/unaging export path jitter); 840 failed every
# time.  The default keeps the field-proven bound with a single automatic retry
# of a session whose op_summary comes back empty.
DEFAULT_MAX_LAUNCHES_PER_SESSION = 540
DEFAULT_EMPTY_SUMMARY_RETRIES = 1
# msprof rejects ``--application`` values of 2048+ characters; keep headroom.
MSPROF_APPLICATION_ARG_LIMIT = 2048
MSPROF_APPLICATION_ARG_HEADROOM = 150
OPAPI_FAMILY = (
    "libopapi.so",
    "libopapi_math.so",
    "libopapi_nn.so",
    "libopapi_transformer.so",
    "libopapi_cv.so",
    "libopapi_ras.so",
    "libopapi_oam.so",
)
WARMUP_SYMBOLS = ("aclnnArgMaxGetWorkspaceSize", "aclnnMaxGetWorkspaceSize")
ALLOWED_AIC_METRICS = (
    "ArithmeticUtilization",
    "PipeUtilization",
    "Memory",
    "MemoryL0",
    "ResourceConflictRatio",
    "MemoryUB",
    "L2Cache",
)
REQUIRED_IDENTITY_FIELDS = (
    "source_sha256",
    "binary_sha256",
    "device_id",
    "cann_version",
    "frequency",
)
FIELDS = (
    "case_id",
    "total_us",
    "aggregation",
    "sample_count",
    "min_us",
    "max_us",
    "spread_over_median",
    "block_num",
    "target_kernel_sequence",
    "warmup_kernel_sequence",
    "sample_total_us",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="single-session msprof application collector: 10 warmup-target pairs + median"
    )
    parser.add_argument("--cases", required=True)
    parser.add_argument(
        "--runner-file",
        required=True,
        help="persistent runner source used by --application-cmd; recorded and hashed for reuse gates",
    )
    parser.add_argument(
        "--identity-manifest",
        required=True,
        help="JSON containing source/binary/device/CANN/frequency identity for comparison gates",
    )
    parser.add_argument(
        "--application-cmd",
        required=True,
        help=(
            "persistent runner template; must contain {case_ids}, {samples}, "
            "{execution_manifest}, {warmup_mode}, and {warmup_op}"
        ),
    )
    parser.add_argument("--target-op", required=True, help="logical operator name, e.g. Exp or ArgMax")
    parser.add_argument(
        "--ops-warmup-probe-cmd",
        required=True,
        help="command template that verifies the preferred op warmup; must contain {warmup_op}",
    )
    parser.add_argument(
        "--warmup-build-cmd",
        required=True,
        help="command that builds the local compiled warmup only when the ops probe fails",
    )
    parser.add_argument(
        "--compiled-warmup-binary",
        required=True,
        help="expected project-local _C.so; hashed after a successful warmup build",
    )
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--msprof-bin", default="msprof")
    parser.add_argument("--aic-metrics", choices=ALLOWED_AIC_METRICS, default="PipeUtilization")
    parser.add_argument("--timeout", type=int, default=1800, help="application profiling timeout in seconds")
    parser.add_argument(
        "--warmup-build-timeout", type=int, default=900, help="compiled warmup build timeout in seconds"
    )
    parser.add_argument(
        "--ops-warmup-probe-timeout",
        type=int,
        default=120,
        help="ops warmup availability probe timeout in seconds",
    )
    parser.add_argument(
        "--max-launches-per-session",
        type=int,
        default=DEFAULT_MAX_LAUNCHES_PER_SESSION,
        help=(
            "cap on relevant op launches per msprof session; larger case sets are "
            "split into consecutive sessions (CANN 9.0.0 exports an empty op_summary "
            "above roughly 720-840 records while still exiting 0)"
        ),
    )
    parser.add_argument(
        "--parse-only",
        action="store_true",
        help="parse an existing output directory; do not build or invoke msprof",
    )
    return parser.parse_args()


def now_utc() -> str:
    return dt.datetime.now(dt.timezone.utc).replace(microsecond=0).isoformat()


def atomic_write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            json.dump(value, stream, ensure_ascii=False, indent=2, sort_keys=True)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def read_case_ids(path: Path) -> list[str]:
    with path.open(newline="", encoding="utf-8-sig") as stream:
        rows = list(csv.DictReader(stream))
    if not rows or "case_id" not in rows[0]:
        raise ValueError("cases CSV must contain at least one row and a case_id column")
    case_ids = [str(row["case_id"]).strip() for row in rows]
    if any(not case_id for case_id in case_ids):
        raise ValueError("case_id must be non-empty")
    if len(case_ids) != len(set(case_ids)):
        raise ValueError("case_id must be unique")
    if any("," in case_id for case_id in case_ids):
        raise ValueError("case_id must not contain a comma")
    return case_ids


def normalized_op_name(name: str) -> str:
    leaf = name.rsplit("::", 1)[-1].rsplit(".", 1)[-1]
    return "".join(character.lower() for character in leaf if character.isalnum())


def is_argmax_target(name: str) -> bool:
    """Return true only for the ArgMax logical family, including its V2 spelling."""
    return normalized_op_name(name) in {"argmax", "argmaxv2"}


def ops_warmup_op(target_op: str) -> str:
    return "Max" if is_argmax_target(target_op) else "ArgMax"


def validate_args(args: argparse.Namespace) -> None:
    if args.timeout < 1 or args.warmup_build_timeout < 1 or args.ops_warmup_probe_timeout < 1:
        raise ValueError("timeouts must be positive")
    if args.max_launches_per_session < 1:
        raise ValueError("--max-launches-per-session must be positive")
    if "{warmup_op}" not in args.ops_warmup_probe_cmd:
        raise ValueError("--ops-warmup-probe-cmd missing placeholder: {warmup_op}")
    placeholders = (
        "{case_ids}",
        "{samples}",
        "{execution_manifest}",
        "{warmup_mode}",
        "{warmup_op}",
    )
    missing = [placeholder for placeholder in placeholders if placeholder not in args.application_cmd]
    if missing:
        raise ValueError(f"--application-cmd missing placeholders: {missing}")
    if not Path(args.runner_file).is_file():
        raise ValueError(f"--runner-file does not exist: {args.runner_file}")
    if not Path(args.identity_manifest).is_file():
        raise ValueError(f"--identity-manifest does not exist: {args.identity_manifest}")


def load_identity_manifest(path: Path) -> dict:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError("identity manifest must be a JSON object")
    missing = [field for field in REQUIRED_IDENTITY_FIELDS if value.get(field) in (None, "", [], {})]
    if missing:
        raise ValueError(f"identity manifest missing non-empty fields: {missing}")
    source_sha = str(value["source_sha256"])
    if len(source_sha) != 64 or any(character not in "0123456789abcdefABCDEF" for character in source_sha):
        raise ValueError("identity manifest source_sha256 must be a 64-digit hexadecimal value")
    binary_sha = value["binary_sha256"]
    binary_values = binary_sha if isinstance(binary_sha, list) else [binary_sha]
    if any(
        len(str(item)) != 64 or any(character not in "0123456789abcdefABCDEF" for character in str(item))
        for item in binary_values
    ):
        raise ValueError("identity manifest binary_sha256 must contain 64-digit hexadecimal values")
    return value


def _cann_lib_dirs(ascend_home: Path) -> list[Path]:
    candidates = [
        ascend_home / "x86_64-linux" / "lib64",
        ascend_home / "lib64",
    ]
    return [directory for directory in candidates if directory.is_dir()]


def _nm_defines_symbol(library: Path, symbol: str) -> bool | None:
    """Return whether ``library`` defines ``symbol``; None when nm is unusable."""
    try:
        result = subprocess.run(
            ["nm", "-D", str(library)],
            capture_output=True,
            text=True,
            timeout=120,
        )
    except (OSError, subprocess.TimeoutExpired):
        return None
    if result.returncode != 0:
        return False
    for line in result.stdout.splitlines():
        fields = line.split()
        if len(fields) >= 3 and fields[1] in ("T", "W", "i") and fields[2] == symbol:
            return True
    return False


def inspect_cann_environment() -> dict:
    """Inventory the actually-resolved CANN installation.

    Guards against two measured failure modes: (a) machines with several CANN
    installs where the shell silently resolves a different toolkit than the one
    the caller believes was sourced, and (b) incomplete installations whose
    ops packages are missing, which remove the system aclnn warmup ops (and
    with them the framework-ops warmup path) without any upfront error.
    """
    home_value = os.environ.get("ASCEND_HOME_PATH", "").strip()
    environment: dict = {
        "ascend_home_path": home_value,
        "version": None,
        "timestamp": None,
        "opapi_libraries": {},
        "warmup_symbol_probe": {},
        "framework_ops_warmup_available": None,
        "notes": [],
    }
    if not home_value:
        environment["notes"].append("ASCEND_HOME_PATH is not set; the shell resolved msprof from PATH")
        return environment
    ascend_home = Path(home_value)
    version_candidates = (
        ascend_home / "version.info",
        ascend_home / "x86_64-linux" / "version.info",
        ascend_home / "compiler" / "version.info",
        ascend_home / "opp" / "version.info",
        ascend_home / "runtime" / "version.info",
    )
    for version_file in version_candidates:
        if not version_file.is_file():
            continue
        for line in version_file.read_text(encoding="utf-8", errors="replace").splitlines():
            key, _, value = line.partition("=")
            key = key.strip()
            if key == "Version" and not environment["version"]:
                environment["version"] = value.strip()
            elif key == "timestamp" and not environment["timestamp"]:
                environment["timestamp"] = value.strip()
        if environment["version"]:
            break
    lib_dirs = _cann_lib_dirs(ascend_home)
    probe_targets: list[Path] = []
    for name in OPAPI_FAMILY:
        found = ""
        for directory in lib_dirs:
            candidate = directory / name
            if candidate.exists():
                found = str(candidate)
                probe_targets.append(candidate)
                break
        environment["opapi_libraries"][name] = found
    nnopbase = ""
    for directory in lib_dirs:
        candidate = directory / "libnnopbase.so"
        if candidate.exists():
            nnopbase = str(candidate)
            probe_targets.append(candidate)
            break
    environment["opapi_libraries"]["libnnopbase.so"] = nnopbase
    if not probe_targets:
        environment["framework_ops_warmup_available"] = False
        environment["notes"].append(
            "no opapi libraries found under ASCEND_HOME_PATH; the installation looks incomplete (ops packages missing)"
        )
        return environment
    findings: list[bool] = []
    for symbol in WARMUP_SYMBOLS:
        located = ""
        unknown = False
        for library in probe_targets:
            defined = _nm_defines_symbol(library, symbol)
            if defined is None:
                unknown = True
                continue
            if defined:
                located = str(library)
                break
        if located:
            environment["warmup_symbol_probe"][symbol] = located
            findings.append(True)
        elif unknown:
            environment["warmup_symbol_probe"][symbol] = "unknown (nm unavailable)"
            findings.append(True)  # cannot rule out; let the runtime probe decide
        else:
            environment["warmup_symbol_probe"][symbol] = ""
            findings.append(False)
    if not any(findings):
        environment["framework_ops_warmup_available"] = False
        environment["notes"].append(
            "neither aclnnArgMax nor aclnnMax host symbols exist in the opapi "
            "libraries; framework-ops warmup is unavailable on this installation "
            "(incomplete CANN/ops install) and only the compiled warmup can be used"
        )
    else:
        environment["framework_ops_warmup_available"] = True
    return environment


def warmup_kernel_estimate(warmup: dict) -> int:
    """Conservative per-group warmup kernel count used for session planning."""
    if warmup["mode"] == "compiled":
        return 1
    return 2 if normalized_op_name(str(warmup["logical_op"])) == "argmax" else 1


def launches_per_case_estimate(warmup: dict, target_kernels: int = 1) -> int:
    return SAMPLES_PER_CASE * (warmup_kernel_estimate(warmup) + max(1, target_kernels))


def rendered_application_length(template: str, chunk: list[str], execution_manifest: Path, warmup: dict) -> int:
    return len(render_application_command(template, chunk, execution_manifest, warmup))


def plan_session_chunks(
    case_ids: list[str],
    warmup: dict,
    template: str,
    execution_manifest_paths: list[Path],
    max_launches_per_session: int,
    target_kernels: int = 1,
) -> list[list[str]]:
    """Greedy chunking under the per-session launch cap and msprof arg limit.

    ``execution_manifest_paths`` provides one provisional manifest path per
    allowed session (used only for realistic command-length measurement).
    """
    per_case = launches_per_case_estimate(warmup, target_kernels)
    cases_per_session = max(1, max_launches_per_session // per_case)
    chunks: list[list[str]] = []
    for start in range(0, len(case_ids), cases_per_session):
        chunks.append(case_ids[start:start + cases_per_session])
    arg_budget = MSPROF_APPLICATION_ARG_LIMIT - MSPROF_APPLICATION_ARG_HEADROOM
    bounded: list[list[str]] = []
    for position, chunk in enumerate(chunks):
        manifest_path = execution_manifest_paths[min(position, len(execution_manifest_paths) - 1)]
        while len(chunk) > 1 and rendered_application_length(template, chunk, manifest_path, warmup) >= arg_budget:
            middle = (len(chunk) + 1) // 2
            chunks.append(chunk[middle:])
            chunk = chunk[:middle]
        bounded.append(chunk)
    bounded.sort(key=len, reverse=True)
    ordered: list[list[str]] = []
    remaining = list(case_ids)
    for chunk in bounded:
        if not remaining:
            break
        head: list[str] = []
        for case_id in chunk:
            if case_id in remaining:
                head.append(case_id)
                remaining.remove(case_id)
        if head:
            ordered.append(head)
    return ordered


def actual_max_launches_per_case(manifest: dict) -> int:
    """Observed per-case relevant launches from a completed session manifest."""
    worst = 0
    for run in manifest.get("case_runs", []):
        warmup = validate_kernel_sequence(
            run.get("warmup_kernel_sequence"), "warmup_kernel_sequence", str(run.get("case_id", ""))
        )
        target = validate_kernel_sequence(
            run.get("target_kernel_sequence"), "target_kernel_sequence", str(run.get("case_id", ""))
        )
        worst = max(worst, SAMPLES_PER_CASE * (len(warmup) + len(target)))
    return worst


def run_warmup_build(command_template: str, log_path: Path, timeout: int, action: str = "warmup build") -> dict:
    command = shlex.split(command_template)
    if not command:
        raise ValueError("--warmup-build-cmd is empty")
    started_at = now_utc()
    started = time.monotonic()
    status = "failed"
    returncode: int | None = None
    error = ""
    with log_path.open("w", encoding="utf-8") as log:
        try:
            process = subprocess.run(
                command,
                stdout=log,
                stderr=subprocess.STDOUT,
                text=True,
                timeout=timeout,
            )
            returncode = process.returncode
            status = "success" if process.returncode == 0 else "failed"
        except subprocess.TimeoutExpired:
            error = f"{action} timed out after {timeout}s"
        except OSError as build_error:
            error = str(build_error)
    return {
        "status": status,
        "returncode": returncode,
        "error": error,
        "started_at": started_at,
        "completed_at": now_utc(),
        "duration_seconds": round(time.monotonic() - started, 3),
        "command_sha256": hashlib.sha256(command_template.encode("utf-8")).hexdigest(),
        "log": str(log_path),
    }


def run_ops_warmup_probe(command_template: str, target_op: str, log_path: Path, timeout: int) -> dict:
    warmup_op = ops_warmup_op(target_op)
    rendered = command_template.format(warmup_op=warmup_op)
    result = run_warmup_build(rendered, log_path, timeout, action="ops warmup probe")
    result["warmup_op"] = warmup_op
    if result["status"] == "success":
        declaration = parse_probe_declaration(log_path)
        if declaration:
            result["warmup_implementation"] = declaration.get("implementation")
            result["warmup_declaration"] = declaration
    return result


def parse_probe_declaration(log_path: Path) -> dict | None:
    """Optional probe protocol: a successful probe may print

    ``OPS_WARMUP_PROBE_RESULT: {"implementation": "..."}``

    as a stdout line.  Torch runners do not print it; runners that warm up
    through another stack (e.g. the pure-aclnn runner using aclnnArgMax /
    aclnnMax directly) declare their true implementation so the recorded
    warmup identity stays honest.
    """
    try:
        lines = log_path.read_text(encoding="utf-8", errors="replace").splitlines()
    except OSError:
        return None
    for line in reversed(lines):
        marker = "OPS_WARMUP_PROBE_RESULT:"
        index = line.find(marker)
        if index < 0:
            continue
        try:
            value = json.loads(line[index + len(marker):].strip())
        except json.JSONDecodeError:
            return None
        return value if isinstance(value, dict) else None
    return None


def select_warmup(target_op: str, probe: dict, build: dict) -> dict:
    if probe["status"] == "success":
        preferred = ops_warmup_op(target_op)
        declared = probe.get("warmup_implementation")
        implementation = (
            declared if isinstance(declared, str) and declared else f"torch_{preferred.lower()}_256mib_fp16_v1"
        )
        return {
            "mode": "framework_ops",
            "logical_op": preferred,
            "implementation": implementation,
            "fallback_reason": "",
            "tensor_bytes": FALLBACK_TENSOR_BYTES,
            "tensor_allocation": {
                "shape": [128, 1024, 1024],
                "dtype": "float16",
                "elements": FALLBACK_TENSOR_BYTES // 2,
            },
        }
    if build["status"] == "success":
        return {
            "mode": "compiled",
            "logical_op": "OpsProfilingWarmup",
            "implementation": COMPILED_WARMUP_IMPLEMENTATION,
            "fallback_reason": "ops_warmup_probe_failed",
            "tensor_bytes": FALLBACK_TENSOR_BYTES,
            "tensor_allocation": {
                "shape": [128, 1024, 1024],
                "dtype": "float16",
                "elements": FALLBACK_TENSOR_BYTES // 2,
            },
        }
    raise ValueError("ops warmup unavailable and compiled warmup build failed")


def attach_compiled_warmup_binary(warmup: dict, binary_path: Path) -> dict:
    if warmup["mode"] != "compiled":
        return warmup
    binary_path = binary_path.resolve()
    if not binary_path.is_file():
        raise ValueError(f"compiled warmup build did not produce binary: {binary_path}")
    selected = dict(warmup)
    selected["binary"] = str(binary_path)
    selected["binary_sha256"] = sha256_file(binary_path)
    return selected


def prepare_warmup(
    args: argparse.Namespace, probe_log: Path, build_log: Path, compiled_binary: Path
) -> tuple[dict, dict, dict]:
    probe = run_ops_warmup_probe(
        args.ops_warmup_probe_cmd,
        args.target_op,
        probe_log,
        args.ops_warmup_probe_timeout,
    )
    if probe["status"] == "success":
        build = {"status": "not_needed", "reason": "ops_warmup_probe_succeeded"}
    else:
        build = run_warmup_build(args.warmup_build_cmd, build_log, args.warmup_build_timeout)
    selected = select_warmup(args.target_op, probe, build)
    selected = attach_compiled_warmup_binary(selected, compiled_binary)
    return probe, build, selected


def normalized_key(name: str) -> str:
    return "".join(character.lower() for character in name if character.isalnum())


def field(row: dict[str, str], expected: str) -> str:
    key = normalized_key(expected)
    for name, value in row.items():
        if normalized_key(name or "") == key:
            return (value or "").strip().replace("\t", "")
    raise ValueError(f"op_summary missing field {expected!r}")


def find_op_summary(out_dir: Path) -> Path:
    candidates = sorted(out_dir.rglob("op_summary*.csv"))
    if len(candidates) != 1:
        raise ValueError(f"expected exactly one op_summary CSV, found {len(candidates)}")
    return candidates[0]


def wait_for_op_summary(session_dir: Path, timeout_seconds: float = 120.0) -> Path:
    """msprof writes op_summary during its own post-processing; poll bounded."""
    deadline = time.monotonic() + timeout_seconds
    while time.monotonic() < deadline:
        candidates = sorted(session_dir.rglob("op_summary*.csv"))
        if candidates:
            return candidates[0]
        time.sleep(2)
    candidates = sorted(session_dir.rglob("op_summary*.csv"))
    if len(candidates) != 1:
        raise ValueError(
            f"expected exactly one op_summary CSV under {session_dir}, found {len(candidates)}; "
            "msprof may have exported an empty analysis (per-session op-record cliff or an "
            "incomplete CANN install — see cann_environment in collection_manifest.json)"
        )
    return candidates[0]


def validate_kernel_sequence(value: object, label: str, case_id: str) -> list[str]:
    if not isinstance(value, list) or not value:
        raise ValueError(f"case {case_id}: {label} must be a non-empty list")
    if any(not isinstance(name, str) or not name for name in value):
        raise ValueError(f"case {case_id}: {label} contains an invalid kernel name")
    return value


def load_execution_manifest(
    path: Path,
    expected_case_ids: list[str],
    target_op: str,
    selected_warmup: dict,
) -> dict:
    manifest = json.loads(path.read_text(encoding="utf-8"))
    if manifest.get("schema_version") != EXECUTION_MANIFEST_SCHEMA_VERSION:
        raise ValueError(f"execution manifest schema_version must be {EXECUTION_MANIFEST_SCHEMA_VERSION}")
    if manifest.get("samples_per_case") != SAMPLES_PER_CASE:
        raise ValueError(f"execution manifest samples_per_case must be {SAMPLES_PER_CASE}")
    if normalized_op_name(str(manifest.get("target_operator", ""))) != normalized_op_name(target_op):
        raise ValueError("execution manifest target_operator mismatch")
    warmup = manifest.get("warmup")
    if not isinstance(warmup, dict):
        raise ValueError("execution manifest missing warmup object")
    if warmup.get("mode") != selected_warmup["mode"]:
        raise ValueError("execution manifest warmup mode mismatch")
    if normalized_op_name(str(warmup.get("logical_op", ""))) != normalized_op_name(str(selected_warmup["logical_op"])):
        raise ValueError("execution manifest warmup logical_op mismatch")
    if warmup.get("implementation") != selected_warmup["implementation"]:
        raise ValueError("execution manifest warmup implementation mismatch")
    if warmup.get("tensor_bytes") != FALLBACK_TENSOR_BYTES:
        raise ValueError(f"warmup tensor_bytes must be exactly {FALLBACK_TENSOR_BYTES}")
    if warmup.get("tensor_allocation") != selected_warmup["tensor_allocation"]:
        raise ValueError("execution manifest warmup tensor_allocation mismatch")
    if selected_warmup["mode"] == "framework_ops":
        expected_op = ops_warmup_op(target_op)
        if normalized_op_name(str(warmup.get("logical_op", ""))) != normalized_op_name(expected_op):
            raise ValueError(f"framework ops warmup for {target_op} must be {expected_op}")
    runs = manifest.get("case_runs")
    if not isinstance(runs, list):
        raise ValueError("execution manifest missing case_runs")
    actual_case_ids = [str(run.get("case_id", "")) for run in runs]
    if actual_case_ids != expected_case_ids:
        raise ValueError(f"runner case order mismatch: {actual_case_ids}")
    for run in runs:
        case_id = str(run["case_id"])
        warmup_sequence = validate_kernel_sequence(run.get("warmup_kernel_sequence"), "warmup_kernel_sequence", case_id)
        target_sequence = validate_kernel_sequence(run.get("target_kernel_sequence"), "target_kernel_sequence", case_id)
        if set(warmup_sequence) & set(target_sequence):
            raise ValueError(f"case {case_id}: warmup and target kernel names must be disjoint")
        if selected_warmup["mode"] == "compiled" and warmup_sequence != ["OpsProfilingWarmup"]:
            raise ValueError(f"case {case_id}: compiled warmup sequence must be ['OpsProfilingWarmup']")
        expected_iteration = warmup_sequence + target_sequence
        if run.get("iteration_kernel_sequence") != expected_iteration:
            raise ValueError(f"case {case_id}: iteration sequence must be warmup + target")
    return manifest


def parse_sessions(sessions: list[tuple[Path, dict]]) -> tuple[list[dict], dict]:
    """Parse consecutive msprof sessions as one logical collection.

    ``sessions`` is an ordered list of ``(op_summary_csv, execution_manifest)``
    pairs; together their case runs must cover the full expected launch
    topology without gaps, duplicates, or reordering.
    """
    if not sessions:
        raise ValueError("no sessions to parse")

    expected_flat: list[tuple[str, int, str, str, int]] = []
    relevant_names: set[str] = set()
    for _, manifest in sessions:
        for run in manifest["case_runs"]:
            warmup = run["warmup_kernel_sequence"]
            target = run["target_kernel_sequence"]
            relevant_names.update(warmup + target)
            for sample_index in range(SAMPLES_PER_CASE):
                for sequence_index, name in enumerate(warmup + target):
                    role = "warmup" if sequence_index < len(warmup) else "target"
                    expected_flat.append((str(run["case_id"]), sample_index, name, role, sequence_index))

    profiled: list[dict] = []
    for op_summary, _manifest in sessions:
        session_rows = []
        with op_summary.open(newline="", encoding="utf-8-sig") as stream:
            for source_index, row in enumerate(csv.DictReader(stream)):
                name = field(row, "Op Name")
                if name not in relevant_names:
                    continue
                try:
                    duration = float(field(row, "Task Duration(us)"))
                    start = float(field(row, "Task Start Time(us)"))
                except ValueError as error:
                    raise ValueError(f"invalid duration/start in op_summary row {source_index + 2}") from error
                session_rows.append(
                    {
                        "source_row": source_index + 2,
                        "kernel": name,
                        "duration_us": duration,
                        "task_start_us": start,
                        "block_num": field(row, "Block Num"),
                    }
                )
        session_rows.sort(key=lambda item: (float(item["task_start_us"]), int(item["source_row"])))
        profiled.extend(session_rows)

    actual_names = [str(row["kernel"]) for row in profiled]
    expected_names = [entry[2] for entry in expected_flat]
    if actual_names != expected_names:
        mismatch = next(
            (index for index, pair in enumerate(zip(actual_names, expected_names)) if pair[0] != pair[1]),
            min(len(actual_names), len(expected_names)),
        )
        hint = ""
        if not actual_names:
            hint = (
                "; msprof exported zero relevant op records — likely causes: the session "
                "exceeded the per-session op-record cliff (see --max-launches-per-session) "
                "or the CANN analysis pipeline failed on an incomplete installation"
            )
        raise ValueError(
            "profile launch topology mismatch at relevant launch "
            f"{mismatch}: actual_count={len(actual_names)}, expected_count={len(expected_names)}{hint}"
        )

    target_totals: dict[str, list[float]] = {}
    target_blocks: dict[str, list[str]] = {}
    for _, manifest in sessions:
        for run in manifest["case_runs"]:
            case_id = str(run["case_id"])
            target_totals.setdefault(case_id, [0.0] * SAMPLES_PER_CASE)
            target_blocks.setdefault(case_id, [])
    mapped: list[dict] = []
    for profile_row, (case_id, sample_index, kernel, role, sequence_index) in zip(profiled, expected_flat):
        mapped.append(
            {
                "case_id": case_id,
                "sample_index": sample_index,
                "sequence_index": sequence_index,
                "role": role,
                **profile_row,
            }
        )
        if role == "target":
            target_totals[case_id][sample_index] += float(profile_row["duration_us"])
            target_blocks[case_id].append(str(profile_row["block_num"]))

    rows: list[dict] = []
    for _, manifest in sessions:
        for run in manifest["case_runs"]:
            case_id = str(run["case_id"])
            values = target_totals[case_id]
            median = statistics.median(values)
            spread = (max(values) - min(values)) / median if median else 0.0
            block_values = list(dict.fromkeys(target_blocks[case_id]))
            rows.append(
                {
                    "case_id": case_id,
                    "total_us": round(median, 6),
                    "aggregation": "median",
                    "sample_count": SAMPLES_PER_CASE,
                    "min_us": round(min(values), 6),
                    "max_us": round(max(values), 6),
                    "spread_over_median": round(spread, 6),
                    "block_num": " + ".join(block_values),
                    "target_kernel_sequence": " + ".join(run["target_kernel_sequence"]),
                    "warmup_kernel_sequence": " + ".join(run["warmup_kernel_sequence"]),
                    "sample_total_us": json.dumps([round(value, 6) for value in values]),
                }
            )

    validation = {
        "schema_version": VALIDATION_SCHEMA_VERSION,
        "status": "pass",
        "samples_per_case": SAMPLES_PER_CASE,
        "pair_order": "warmup_then_target",
        "aggregation": "median of 10 per-pair target kernel duration sums",
        "warmup_time_included": False,
        "session_count": len(sessions),
        "expected_relevant_launches": len(expected_flat),
        "parsed_relevant_launches": len(profiled),
        "mapped_relevant_launches": len(mapped),
        "mapped_launches": mapped,
    }
    return rows, validation


def parse_profile(op_summary: Path, manifest: dict) -> tuple[list[dict], dict]:
    """Backward-compatible single-session entry point."""
    return parse_sessions([(op_summary, manifest)])


def write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)


def render_application_command(template: str, case_ids: list[str], execution_manifest: Path, warmup: dict) -> str:
    return template.format(
        case_ids=",".join(case_ids),
        samples=SAMPLES_PER_CASE,
        execution_manifest=str(execution_manifest),
        warmup_mode=warmup["mode"],
        warmup_op=warmup["logical_op"],
    )


def _run_msprof_session(
    args: argparse.Namespace,
    application: str,
    session: dict,
) -> None:
    command = [
        args.msprof_bin,
        f"--application={application}",
        f"--output={session['dir']}",
        # Version-related: CANN 9.0.0 only exports op_summary with task-based
        # aic-mode; CANN 9.1.0 exports it either way (verified harmless).
        "--aic-mode=task-based",
        f"--aic-metrics={args.aic_metrics}",
    ]
    session["dir"].mkdir(parents=True, exist_ok=True)
    with session["msprof_log"].open("w", encoding="utf-8") as log:
        process = subprocess.run(
            command,
            stdout=log,
            stderr=subprocess.STDOUT,
            text=True,
            timeout=args.timeout,
        )
    if process.returncode:
        raise RuntimeError(f"msprof exited {process.returncode}; see {session['msprof_log']}")


def _collect_sessions(
    args: argparse.Namespace,
    case_ids: list[str],
    out_dir: Path,
    selected_warmup: dict,
) -> list[dict]:
    """Run the case set as one or more msprof sessions under the launch cap."""
    sessions: list[dict] = []
    remaining = list(case_ids)
    target_kernels_estimate = 1
    session_number = 0
    while remaining:
        provisional_paths: list[Path] = []
        for index in range(1, len(remaining) + 1):
            if len(remaining) == 1 or index == 1:
                # provisional single-session + first-session manifest paths are
                # enough for realistic command-length measurement
                provisional_paths.append(out_dir / "execution-manifest.json")
            else:
                provisional_paths.append(out_dir / f"session-{index:03d}" / "execution-manifest.json")
        chunks = plan_session_chunks(
            remaining,
            selected_warmup,
            args.application_cmd,
            provisional_paths,
            args.max_launches_per_session,
            target_kernels_estimate,
        )
        next_chunk = chunks[0]
        session_number += 1
        if len(chunks) == 1 and session_number == 1:
            session = {
                "index": 1,
                "dir": out_dir,
                "execution_manifest": out_dir / "execution-manifest.json",
                "msprof_log": out_dir / "msprof.log",
            }
        else:
            directory = out_dir / f"session-{session_number:03d}"
            session = {
                "index": session_number,
                "dir": directory,
                "execution_manifest": directory / "execution-manifest.json",
                "msprof_log": directory / "msprof.log",
            }
        application = render_application_command(
            args.application_cmd, next_chunk, session["execution_manifest"], selected_warmup
        )
        if len(application) >= MSPROF_APPLICATION_ARG_LIMIT - MSPROF_APPLICATION_ARG_HEADROOM:
            raise ValueError(
                "rendered --application value exceeds the msprof argument-length budget; "
                "shorten case ids or the runner command"
            )
        retries = 0
        while True:
            _run_msprof_session(args, application, session)
            if not session["execution_manifest"].is_file():
                raise ValueError(f"runner did not create {session['execution_manifest']}")
            manifest = load_execution_manifest(
                session["execution_manifest"], next_chunk, args.target_op, selected_warmup
            )
            try:
                session["op_summary"] = wait_for_op_summary(session["dir"], timeout_seconds=60)
                break
            except ValueError:
                if retries >= DEFAULT_EMPTY_SUMMARY_RETRIES:
                    raise
                retries += 1
                # msprof exits only after its analysis pass, so a missing
                # op_summary will not appear later; drop the failed PROF data
                # so the retry cannot collide with it.
                for prof_dir in session["dir"].glob("PROF_*"):
                    shutil.rmtree(prof_dir, ignore_errors=True)
        session["retries"] = retries
        session["manifest"] = manifest
        sessions.append(session)
        for case_id in next_chunk:
            remaining.remove(case_id)
        # Adaptive refinement: the observed per-case launch count replaces the
        # conservative estimate for planning the remaining sessions.
        observed = actual_max_launches_per_case(manifest)
        if observed:
            target_kernels_estimate = max(1, observed // SAMPLES_PER_CASE - warmup_kernel_estimate(selected_warmup))
    return sessions


def main() -> int:
    args = parse_args()
    started_at = now_utc()
    cases = Path(args.cases).resolve()
    runner_file = Path(args.runner_file).resolve()
    identity_manifest_path = Path(args.identity_manifest).resolve()
    compiled_warmup_binary = Path(args.compiled_warmup_binary)
    out_dir = Path(args.out_dir).resolve()
    msprof_log = out_dir / "msprof.log"
    build_log = out_dir / "warmup-build.log"
    probe_log = out_dir / "warmup-ops-probe.log"
    probe: dict = {"status": "parse_only"}
    build: dict = {"status": "parse_only"}
    selected_warmup: dict = {}
    identity: dict = {}
    cann_environment: dict = {}
    sessions: list[dict] = []
    try:
        validate_args(args)
        case_ids = read_case_ids(cases)
        identity = load_identity_manifest(identity_manifest_path)
        cann_environment = inspect_cann_environment()
        if args.parse_only:
            existing_collection = json.loads((out_dir / "collection_manifest.json").read_text(encoding="utf-8"))
            selected_warmup = dict(existing_collection.get("warmup", {}))
            if not selected_warmup:
                raise ValueError("parse-only collection manifest missing warmup selection")
            for entry in existing_collection.get("sessions", []):
                directory = Path(entry["dir"])
                sessions.append(
                    {
                        "index": entry["index"],
                        "dir": directory,
                        "execution_manifest": Path(entry["execution_manifest"]),
                        "msprof_log": Path(entry["msprof_log"]),
                        "op_summary": Path(entry["op_summary"]),
                        "manifest": load_execution_manifest(
                            Path(entry["execution_manifest"]),
                            entry["case_ids"],
                            args.target_op,
                            selected_warmup,
                        ),
                    }
                )
            if not sessions:
                sessions = [
                    {
                        "index": 1,
                        "dir": out_dir,
                        "execution_manifest": out_dir / "execution-manifest.json",
                        "msprof_log": msprof_log,
                        "op_summary": find_op_summary(out_dir),
                        "manifest": load_execution_manifest(
                            out_dir / "execution-manifest.json",
                            case_ids,
                            args.target_op,
                            selected_warmup,
                        ),
                    }
                ]
        else:
            if out_dir.exists() and any(out_dir.iterdir()):
                raise ValueError(f"output directory is not empty: {out_dir}")
            out_dir.mkdir(parents=True, exist_ok=True)
            try:
                probe, build, selected_warmup = prepare_warmup(args, probe_log, build_log, compiled_warmup_binary)
            except ValueError as warmup_error:
                if cann_environment.get("framework_ops_warmup_available") is False:
                    raise ValueError(
                        "framework-ops warmup is unavailable: the CANN installation at "
                        f"'{cann_environment.get('ascend_home_path')}' does not provide the "
                        "aclnnArgMax/aclnnMax host symbols (incomplete ops install); the "
                        "compiled warmup is the only remaining fallback and it did not "
                        f"succeed either (original error: {warmup_error})"
                    ) from warmup_error
                raise
            sessions = _collect_sessions(args, case_ids, out_dir, selected_warmup)

        parsed_sessions = [(session["op_summary"], session["manifest"]) for session in sessions]
        rows, validation = parse_sessions(parsed_sessions)
        performance_csv = out_dir / "performance.csv"
        validation_json = out_dir / "validation.json"
        write_csv(performance_csv, rows)
        validation.update(
            {
                "case_order": case_ids,
                "op_summary": str(sessions[0]["op_summary"]),
                "warmup": selected_warmup,
            }
        )
        atomic_write_json(validation_json, validation)
        first_session = sessions[0]
        session_records = []
        for session in sessions:
            session_records.append(
                {
                    "index": session["index"],
                    "dir": str(session["dir"]),
                    "case_ids": [str(run["case_id"]) for run in session["manifest"]["case_runs"]],
                    "execution_manifest": str(session["execution_manifest"]),
                    "op_summary": str(session["op_summary"]),
                    "msprof_log": str(session["msprof_log"]),
                    "empty_summary_retries": session.get("retries", 0),
                }
            )
        atomic_write_json(
            out_dir / "collection_manifest.json",
            {
                "schema_version": MANIFEST_SCHEMA_VERSION,
                "status": "complete",
                "mode": (
                    "msprof_application_single_session" if len(sessions) == 1 else "msprof_application_session_series"
                ),
                "started_at": started_at,
                "completed_at": now_utc(),
                "target_operator": args.target_op,
                "cases_csv": str(cases),
                "cases_csv_sha256": sha256_file(cases),
                "case_count_expected": len(case_ids),
                "case_count_collected": len(rows),
                "case_ids": case_ids,
                "samples_per_case": SAMPLES_PER_CASE,
                "pair_order": "warmup_then_target",
                "aggregation": "median_target_total_us",
                "warmup_time_included": False,
                "warmup_ops_probe": probe,
                "warmup_build": build,
                "warmup": selected_warmup,
                "cann_environment": cann_environment,
                "session_count": len(sessions),
                "max_launches_per_session": args.max_launches_per_session,
                "sessions": session_records,
                "aic_mode": "task-based",
                "aic_mode_rationale": ("required for op_summary export on CANN 9.0.0; verified harmless on 9.1.0"),
                "aic_metrics": args.aic_metrics,
                "runner_file": str(runner_file),
                "runner_file_sha256": sha256_file(runner_file),
                "identity_manifest": str(identity_manifest_path),
                "identity_manifest_sha256": sha256_file(identity_manifest_path),
                "identity": identity,
                "application_command_sha256": hashlib.sha256(args.application_cmd.encode("utf-8")).hexdigest(),
                "execution_manifest": str(first_session["execution_manifest"]),
                "performance_csv": str(performance_csv),
                "validation_json": str(validation_json),
                "op_summary": str(first_session["op_summary"]),
                "msprof_log": str(first_session["msprof_log"]),
            },
        )
        print(performance_csv)
        return 0
    except (OSError, ValueError, RuntimeError, subprocess.TimeoutExpired) as error:
        out_dir.mkdir(parents=True, exist_ok=True)
        atomic_write_json(
            out_dir / "collection_manifest.json",
            {
                "schema_version": MANIFEST_SCHEMA_VERSION,
                "status": "failed",
                "started_at": started_at,
                "completed_at": now_utc(),
                "target_operator": args.target_op,
                "samples_per_case": SAMPLES_PER_CASE,
                "warmup_ops_probe": probe,
                "warmup_build": build,
                "warmup": selected_warmup,
                "cann_environment": cann_environment,
                "identity_manifest": str(identity_manifest_path),
                "identity": identity,
                "error": str(error),
            },
        )
        print(f"[ERROR] {error}", file=sys.stderr)
        return 1 if isinstance(error, (RuntimeError, subprocess.TimeoutExpired)) else 2


if __name__ == "__main__":
    raise SystemExit(main())
