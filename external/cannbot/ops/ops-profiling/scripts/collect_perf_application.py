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
  launches (default 540).  Above roughly 720-840 relevant op records per
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
        "--empty-summary-retries", type=int, default=DEFAULT_EMPTY_SUMMARY_RETRIES,
        help="additional attempts for missing/empty summary only; 0 disables retry; all raw attempts retained",
    )
    parser.add_argument(
        "--planned-topology", default=None,
        help="schema-v1 JSON binding exact per-case preparation/warmup/target sequences to identity evidence",
    )
    parser.add_argument(
        "--host-metadata-capture", choices=("off", "on"), default="off",
        help="explicit task-time=on/runtime-api=on/ge-api=l1, only after current msprof help/version validation",
    )
    parser.add_argument(
        "--host-metadata-capability-evidence", default=None,
        help="SHA-bound zero-target current-binary runtime-api parser probes when help omits its on/off enum",
    )
    parser.add_argument(
        "--parse-only",
        action="store_true",
        help="parse an existing output directory; do not build or invoke msprof",
    )
    parser.add_argument(
        "--parse-out-dir", default=None,
        help="new derived output for --parse-only; default is an independent parse-only-* directory",
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
    if getattr(args, "empty_summary_retries", DEFAULT_EMPTY_SUMMARY_RETRIES) < 0:
        raise ValueError("--empty-summary-retries must be non-negative")
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


def launches_per_case_estimate(
    warmup: dict, target_kernels: int = 1, preparation_kernels: int = 0
) -> int:
    return preparation_kernels + SAMPLES_PER_CASE * (
        warmup_kernel_estimate(warmup) + max(1, target_kernels)
    )


def exact_case_launches(run: dict) -> int:
    return len(run.get("preparation_kernel_sequence", [])) + SAMPLES_PER_CASE * (
        len(run["warmup_kernel_sequence"]) + len(run["target_kernel_sequence"])
    )


def rendered_application_length(template: str, chunk: list[str], execution_manifest: Path, warmup: dict) -> int:
    return len(render_application_command(template, chunk, execution_manifest, warmup))


def plan_session_chunks(
    case_ids: list[str],
    warmup: dict,
    template: str,
    execution_manifest_paths: list[Path],
    max_launches_per_session: int,
    target_kernels: int = 1,
    planned_topology: dict | None = None,
    preparation_kernels: int = 0,
    warmup_kernels: int | None = None,
) -> list[list[str]]:
    """Greedy chunking under the per-session launch cap and msprof arg limit.

    ``execution_manifest_paths`` provides one provisional manifest path per
    allowed session (used only for realistic command-length measurement).
    """
    if not execution_manifest_paths:
        raise ValueError("session planning requires at least one manifest path")
    per_case = launches_per_case_estimate(warmup, target_kernels, preparation_kernels)
    if warmup_kernels is not None:
        per_case = preparation_kernels + SAMPLES_PER_CASE * (warmup_kernels + target_kernels)
    declared = {
        str(run["case_id"]): exact_case_launches(run)
        for run in (planned_topology or {}).get("case_runs", [])
    }
    if planned_topology and any(case_id not in declared for case_id in case_ids):
        raise ValueError("planned topology does not cover every requested case")
    arg_budget = MSPROF_APPLICATION_ARG_LIMIT - MSPROF_APPLICATION_ARG_HEADROOM
    chunks: list[list[str]] = []
    chunk: list[str] = []
    launches = 0
    for case_id in case_ids:
        cost = declared.get(case_id, per_case)
        if cost > max_launches_per_session:
            raise ValueError(f"case {case_id}: {cost} launches exceed session cap {max_launches_per_session}")
        path = execution_manifest_paths[min(len(chunks), len(execution_manifest_paths) - 1)]
        if chunk and (launches + cost > max_launches_per_session or
                      rendered_application_length(template, chunk + [case_id], path, warmup) >= arg_budget):
            chunks.append(chunk)
            chunk, launches = [], 0
            path = execution_manifest_paths[min(len(chunks), len(execution_manifest_paths) - 1)]
        if rendered_application_length(template, chunk + [case_id], path, warmup) >= arg_budget:
            raise ValueError(f"case {case_id}: rendered --application exceeds argument-length budget")
        chunk.append(case_id)
        launches += cost
    if chunk:
        chunks.append(chunk)
    return chunks


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
        worst = max(worst, len(run.get("preparation_kernel_sequence", []))
                    + SAMPLES_PER_CASE * (len(warmup) + len(target)))
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
    return validate_execution_manifest(manifest, expected_case_ids, target_op, selected_warmup)


def validate_execution_manifest(
    manifest: dict, expected_case_ids: list[str], target_op: str, selected_warmup: dict
) -> dict:
    if not isinstance(manifest, dict):
        raise ValueError("execution manifest must be a JSON object")
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
    if not isinstance(runs, list) or any(not isinstance(run, dict) for run in runs):
        raise ValueError("execution manifest missing case_runs")
    actual_case_ids = [str(run.get("case_id", "")) for run in runs]
    if actual_case_ids != expected_case_ids:
        raise ValueError(f"runner case order mismatch: {actual_case_ids}")
    for run in runs:
        case_id = str(run["case_id"])
        preparation = run.get("preparation_kernel_sequence", [])
        if preparation:
            validate_kernel_sequence(preparation, "preparation_kernel_sequence", case_id)
        elif preparation != []:
            raise ValueError(f"case {case_id}: preparation_kernel_sequence must be a list")
        warmup_sequence = validate_kernel_sequence(
            run.get("warmup_kernel_sequence"), "warmup_kernel_sequence", case_id
        )
        target_sequence = validate_kernel_sequence(
            run.get("target_kernel_sequence"), "target_kernel_sequence", case_id
        )
        if set(warmup_sequence) & set(target_sequence):
            raise ValueError(f"case {case_id}: warmup and target kernel names must be disjoint")
        if selected_warmup["mode"] == "compiled" and warmup_sequence != ["OpsProfilingWarmup"]:
            raise ValueError(f"case {case_id}: compiled warmup sequence must be ['OpsProfilingWarmup']")
        expected_iteration = warmup_sequence + target_sequence
        if run.get("iteration_kernel_sequence") != expected_iteration:
            raise ValueError(f"case {case_id}: iteration sequence must be warmup + target")
    return manifest


def verify_file_receipt(receipt: dict) -> None:
    if not isinstance(receipt, dict):
        raise ValueError("evidence receipt must be a JSON object")
    path = Path(receipt.get("path", ""))
    if not path.is_file() or sha256_file(path) != receipt.get("sha256"):
        raise ValueError(f"evidence file missing or SHA mismatch: {path}")


def load_planned_topology(
    path: Path, args: argparse.Namespace, case_ids: list[str], selected_warmup: dict
) -> dict:
    """An exact budget is a predeclared, identity-bound topology, never a lower estimate."""
    plan = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(plan, dict):
        raise ValueError("planned topology must be a JSON object")
    if plan.get("schema_version") != 1:
        raise ValueError("planned topology schema_version must be 1")
    bindings = plan.get("bindings", {})
    for name, actual in (
        ("cases_csv_sha256", sha256_file(Path(args.cases))),
        ("runner_file_sha256", sha256_file(Path(args.runner_file))),
        ("identity_manifest_sha256", sha256_file(Path(args.identity_manifest))),
    ):
        if bindings.get(name) != actual:
            raise ValueError(f"planned topology binding mismatch: {name}")
    evidence = plan.get("evidence")
    if not isinstance(evidence, list) or not evidence:
        raise ValueError("planned topology requires SHA-bound declaration/probe/normal-collection evidence")
    for receipt in evidence:
        if not isinstance(receipt, dict):
            raise ValueError("planned topology evidence must contain JSON objects")
        if receipt.get("kind") not in {"runner_declaration", "warmup_probe", "normal_collection"}:
            raise ValueError("unsupported planned topology evidence kind")
        verify_file_receipt(receipt)
    # Share the same manifest validator; topology planning cannot relax mapping.
    declared = {**plan, "schema_version": EXECUTION_MANIFEST_SCHEMA_VERSION}
    validate_execution_manifest(declared, case_ids, args.target_op, selected_warmup)
    if plan.get("paired_confirmation") is not None:
        validate_confirmation_labels(plan, case_ids)
        effect_plan = plan["paired_confirmation"].get("effect_plan")
        if effect_plan is not None:
            verify_file_receipt(effect_plan)
        if getattr(args, "empty_summary_retries", DEFAULT_EMPTY_SUMMARY_RETRIES) != 0:
            raise ValueError("paired confirmation requires --empty-summary-retries 0")
    return plan


def validate_confirmation_labels(plan: dict, case_ids: list[str]) -> None:
    """Validate frozen AB/BA metadata, leaving statistical rules to the policy module."""
    declaration = plan["paired_confirmation"]
    rounds = declaration.get("round_count")
    if not isinstance(rounds, int) or isinstance(rounds, bool) or rounds < 1:
        raise ValueError("paired confirmation round_count must be positive")
    if declaration.get("kind") == "shared_candidate_paired_v1":
        from collections import Counter
        from itertools import permutations
        variants = ["original", "current", "candidate"]
        if declaration.get("baseline_variants") != variants[:2] or declaration.get("candidate_variant") != variants[2]:
            raise ValueError("shared candidate requires explicit original/current/candidate roles")
        if rounds < 6 or rounds % 6 or len(case_ids) != 3 * rounds:
            raise ValueError("shared candidate requires complete six-permutation rounds")
        if len(plan.get("case_runs", [])) != len(case_ids) or len(set(case_ids)) != len(case_ids):
            raise ValueError("shared candidate execution coverage mismatch")
        seen, orders, input_sha = set(), [], None
        for index in range(rounds):
            group = plan["case_runs"][3 * index:3 * index + 3]
            labels = [run.get("confirmation", {}) for run in group]
            round_id = labels[0].get("round_id")
            if (not isinstance(round_id, (str, int)) or isinstance(round_id, bool) or round_id == ""
                    or round_id in seen or any(label.get("round_id") != round_id for label in labels)):
                raise ValueError("shared candidate missing/duplicate round_id")
            seen.add(round_id)
            order = [label.get("variant") for label in labels]
            if set(order) != set(variants) or any(label.get("order") != order for label in labels):
                raise ValueError("shared candidate order/variant mismatch")
            if any(type(label.get("position")) is not int or label["position"] != j for j, label in enumerate(labels)):
                raise ValueError("shared candidate positions must be 0,1,2")
            orders.append(tuple(order))
            for run, case_id, label in zip(group, case_ids[3 * index:3 * index + 3], labels):
                if str(run.get("case_id")) != case_id:
                    raise ValueError("shared candidate case order mismatch")
                value = label.get("input_sha256", "")
                if not isinstance(value, str) or len(value) != 64 or any(c not in "0123456789abcdef" for c in value):
                    raise ValueError("shared candidate requires actual shared input SHA256")
                input_sha = input_sha or value
                if value != input_sha:
                    raise ValueError("shared candidate cases must share one immutable snapshot")
        if Counter(orders) != Counter({order: rounds // 6 for order in permutations(variants)}):
            raise ValueError("shared candidate requires balanced complete permutations")
        return
    if len(case_ids) != rounds * 2:
        raise ValueError("paired confirmation requires exactly two cases per round")
    seen = set()
    input_sha = None
    for index in range(rounds):
        pair = plan["case_runs"][index * 2:index * 2 + 2]
        labels = [run.get("confirmation", {}) for run in pair]
        round_id = labels[0].get("round_id")
        if (not isinstance(round_id, (str, int)) or isinstance(round_id, bool) or round_id == ""
                or round_id in seen or labels[1].get("round_id") != round_id):
            raise ValueError("paired confirmation missing/duplicate round_id")
        seen.add(round_id)
        order = labels[0].get("order")
        variants = [label.get("variant") for label in labels]
        expected = ["baseline", "candidate"] if order == "AB" else ["candidate", "baseline"]
        if order not in {"AB", "BA"} or labels[1].get("order") != order or variants != expected:
            raise ValueError("paired confirmation order/variant mismatch")
        if [label.get("position") for label in labels] != [0, 1]:
            raise ValueError("paired confirmation positions must be 0,1")
        for label in labels:
            value = label.get("input_sha256", "")
            if not isinstance(value, str) or len(value) != 64 or any(c not in "0123456789abcdef" for c in value):
                raise ValueError("paired confirmation requires actual shared input SHA256")
            input_sha = input_sha or value
            if value != input_sha:
                raise ValueError("paired confirmation cases must share one immutable input snapshot")


def validate_planned_execution(plan: dict | None, manifest: dict) -> None:
    if plan is None:
        return
    declared = {str(run["case_id"]): run for run in plan["case_runs"]}
    for run in manifest["case_runs"]:
        wanted = declared[str(run["case_id"])]
        for key in ("preparation_kernel_sequence", "warmup_kernel_sequence", "target_kernel_sequence",
                    "iteration_kernel_sequence"):
            if run.get(key, []) != wanted.get(key, []):
                raise ValueError(f"case {run['case_id']}: planned topology mismatch: {key}")
        if wanted.get("confirmation") != run.get("confirmation"):
            raise ValueError(f"case {run['case_id']}: paired confirmation execution label mismatch")


def verify_planned_inputs(args: argparse.Namespace, receipt: dict) -> None:
    verify_file_receipt(receipt)
    plan = receipt["plan"]
    for evidence in plan["evidence"]:
        verify_file_receipt(evidence)
    effect_plan = plan.get("paired_confirmation", {}).get("effect_plan")
    if effect_plan is not None:
        verify_file_receipt(effect_plan)
    for key, path in (("cases_csv_sha256", args.cases), ("runner_file_sha256", args.runner_file),
                      ("identity_manifest_sha256", args.identity_manifest)):
        if sha256_file(Path(path)) != plan["bindings"][key]:
            raise ValueError(f"planned topology binding changed during collection: {key}")


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
            preparation = run.get("preparation_kernel_sequence", [])
            relevant_names.update(preparation + warmup + target)
            for sequence_index, name in enumerate(preparation):
                expected_flat.append((str(run["case_id"]), -1, name, "preparation", sequence_index))
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
        "preparation_time_included": False,
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


class CollectionDataError(ValueError):
    def __init__(self, category: str, message: str):
        super().__init__(message)
        self.category = category


def host_metadata_options(args: argparse.Namespace, out_dir: Path) -> dict:
    """Opt in only; inspect the executable that will actually be invoked."""
    result = {"requested": getattr(args, "host_metadata_capture", "off"), "flags": []}
    if result["requested"] == "off":
        return result
    evidence, dependencies = {}, {}
    executable = shutil.which(args.msprof_bin) or args.msprof_bin
    binary = Path(executable).resolve()

    def checked(item):
        path = Path(item.get("path", ""))
        if not path.is_absolute() or not path.is_file() or sha256_file(path) != item.get("sha256"):
            raise CollectionDataError("unsupported_host_metadata_flags", "host metadata capability receipt changed")
        dependencies[str(path)] = item["sha256"]
        return path
    for option, name in (("--help", "help"), ("--version", "version")):
        command = [args.msprof_bin, option]
        process = subprocess.run(command, capture_output=True, text=True, timeout=30)
        output = process.stdout + process.stderr
        path = out_dir / f"msprof-{name}.log"
        with path.open("x", encoding="utf-8") as stream:
            stream.write(output)
        evidence[name] = {"path": str(path), "sha256": sha256_file(path),
                          "command": command, "returncode": process.returncode}
        if name == "version" and (process.returncode != 0 or not output.strip()):
            package_root = binary.parent.parent / "profiler_tool"
            packages = []
            for metadata in package_root.glob("msprof-*.dist-info/METADATA"):
                fields = dict(line.split(":", 1) for line in metadata.read_text(errors="replace").splitlines()
                              if line.startswith(("Name:", "Version:")))
                if fields.get("Name", "").strip() == "msprof" and fields.get("Version", "").strip():
                    packages.append((metadata.resolve(), fields["Version"].strip()))
            if binary.is_file() and packages and all(p.is_relative_to(package_root.resolve()) for p, _ in packages):
                result["installed_package_versions"] = sorted(version for _, version in packages)
                result["CLI_version_available"] = False
                result["installed_package_selection"] = ("unknown_multiple_metadata_entries"
                                                         if len(packages) > 1 else "one_metadata_entry")
                for index, (metadata, package_version) in enumerate(
                        sorted(packages, key=lambda item: (item[1], str(item[0])))):
                    key = "installed_package_version" if index == 0 else f"installed_package_version_{index + 1}"
                    evidence[key] = {
                        "path": str(metadata), "sha256": sha256_file(metadata), "version": package_version,
                        "kind": "installed_msprof_distribution_metadata_not_CLI_version",
                    }
                    checked(evidence[key])
                evidence["binary"] = {"path": str(binary), "sha256": sha256_file(binary)}
                checked(evidence["binary"])
                continue
        if process.returncode != 0 or not output.strip():
            atomic_write_json(out_dir / "host-metadata-capabilities.json", {**result, "evidence": evidence})
            raise CollectionDataError("unsupported_host_metadata_flags",
                                      f"current msprof {option} has no usable evidence")
    help_text = Path(evidence["help"]["path"]).read_text(encoding="utf-8")
    # Require option and advertised value together in that option's help block.
    for option, value in (("--task-time", "on"), ("--runtime-api", "on"), ("--ge-api", "l1")):
        position = help_text.find(option)
        block = help_text[position + len(option):] if position >= 0 else ""
        block = block.split("--", 1)[0]
        tokens = block.replace(",", " ").replace("/", " ").replace("=", " ")
        tokens = tokens.translate(str.maketrans({c: " " for c in "[](){}:;|\"'"})).split()
        if position >= 0 and option == "--runtime-api" and value not in tokens:
            capability_path = getattr(args, "host_metadata_capability_evidence", None)
            if capability_path:
                capability_file = Path(capability_path).resolve()
                evidence["runtime_parser_capability"] = {"path": str(capability_file),
                                                         "sha256": sha256_file(capability_file)}
                checked(evidence["runtime_parser_capability"])
                capability = json.loads(capability_file.read_text())
                rows = capability.get("results", [])
                valid = capability.get("schema_version") == 1 and isinstance(rows, list)
                by_label = {row.get("label"): row for row in rows} if valid else {}
                positive, negative = by_label.get("runtime_on", {}), by_label.get("runtime_invalid", {})
                logs = {}
                for label, row in (("runtime_on", positive), ("runtime_invalid", negative)):
                    command = row.get("command", [])
                    tool = row.get("msprof_binary", {})
                    if (not isinstance(command, list) or not command
                            or Path(command[0]).resolve() != binary or tool.get("path") != str(binary)
                            or not binary.is_file() or tool.get("sha256") != sha256_file(binary)
                            or command.count("--application=/bin/true") != 1
                            or any(arg.startswith("--application")
                                   and arg != "--application=/bin/true" for arg in command)
                            or row.get("target_launches") != 0):
                        raise CollectionDataError(
                            "unsupported_host_metadata_flags",
                            "runtime-api probe is not from the current binary and zero-target application")
                    checked(tool)
                    log_path = checked(row.get("log", {}))
                    evidence[label + "_parser_log"] = row["log"]
                    logs[label] = log_path.read_text(errors="replace").lower()
                positive_flags = [arg for arg in positive["command"] if arg.startswith("--runtime-api=")]
                negative_flags = [arg for arg in negative["command"] if arg.startswith("--runtime-api=")]
                negative_text = logs["runtime_invalid"].translate(str.maketrans({c: " " for c in "\"':;,.()[]{}"}))
                if (positive_flags != ["--runtime-api=on"] or len(negative_flags) != 1
                        or negative_flags[0] in ("--runtime-api=on", "--runtime-api=off")
                        or negative.get("returncode") in (0, None)
                        or "start profiling" in logs["runtime_invalid"]
                        or "runtime-api" not in logs["runtime_invalid"]
                        or not {"on", "off"}.issubset(set(negative_text.split()))
                        or "start profiling" not in logs["runtime_on"]
                        or (positive.get("returncode") != 0
                            and "ai-related business" not in logs["runtime_on"])):
                    raise CollectionDataError(
                        "unsupported_host_metadata_flags",
                        "runtime-api on acceptance/invalid-value rejection is not proven")
                result["runtime_api_support_basis"] = ("actual_zero_target_option_parser_contrast_"
                                                       "not_successful_AI_collection")
                tokens.append("on")
        if position < 0 or value not in tokens:
            atomic_write_json(out_dir / "host-metadata-capabilities.json", {**result, "evidence": evidence})
            raise CollectionDataError("unsupported_host_metadata_flags",
                                      f"current msprof help does not advertise {option}={value}")
    result.update({"flags": ["--task-time=on", "--runtime-api=on", "--ge-api=l1"], "evidence": evidence})
    for path, digest in dependencies.items():
        if sha256_file(Path(path)) != digest:
            raise CollectionDataError("unsupported_host_metadata_flags",
                                      "host metadata capability changed during validation")
    atomic_write_json(out_dir / "host-metadata-capabilities.json", result)
    return result


def classify_collection_failure(error: Exception, log_path: Path | None = None) -> str:
    if isinstance(error, CollectionDataError):
        category = error.category
    else:
        category = "launch_mapping_error" if isinstance(error, ValueError) else "profiler_execution_failed"
    text = str(error)
    if log_path is not None and log_path.is_file():
        text += "\n" + log_path.read_text(encoding="utf-8", errors="replace")
        for path in sorted(log_path.parent.rglob("*.log")):
            if path != log_path:
                text += "\n" + path.read_text(encoding="utf-8", errors="replace")
    lowered = text.lower()
    if any(marker in lowered for marker in (
        "no runtimehosttask", "no ge_info", "missing runtimehosttask", "missing ge_info",
        "runtimehosttask/ge_info", "unable to create ge summary", "unable create ge summary",
    )):
        return "missing_host_metadata"
    if any(("runtimehosttask" in line or "ge_info" in line) and
           any(marker in line for marker in ("can't find", "not found", "missing", "no "))
           for line in lowered.splitlines()):
        return "missing_host_metadata"
    if any(marker in lowered for marker in (
        "unrecognized argument", "unrecognised argument", "unknown option", "invalid option",
        "argument length", "application length", "2048", "invalid parameter",
    )) and not isinstance(error, CollectionDataError):
        return "cli_rejected"
    if "op_summary" in lowered and ("found 0" in lowered or "missing" in lowered):
        return "missing_summary" if category != "missing_host_metadata" else category
    return category


def audit_attempt(attempt: dict) -> None:
    """Persist immutable raw/log receipts before another attempt is allowed."""
    directory = attempt["dir"]
    files = []
    for path in sorted(directory.rglob("*")):
        if path.is_file() and path.name != "attempt.json":
            files.append({"path": str(path.resolve()), "bytes": path.stat().st_size,
                          "sha256": sha256_file(path)})
    attempt["files"] = files
    attempt["completed_at"] = now_utc()
    atomic_write_json(directory / "attempt.json", serializable_attempt(attempt))


def serializable_attempt(attempt: dict) -> dict:
    return {key: str(value) if isinstance(value, Path) else value
            for key, value in attempt.items() if key != "manifest"}


def session_record(session: dict) -> dict:
    keys = ("index", "dir", "case_ids", "execution_manifest", "op_summary", "msprof_log",
            "planned_relevant_launches", "actual_relevant_launches")
    result = {key: str(session[key]) if isinstance(session[key], Path) else session[key]
              for key in keys
              if key in session}
    result["empty_summary_retries"] = session.get("retries", 0)
    result["attempts"] = [serializable_attempt(attempt) for attempt in session.get("attempts", [])]
    return result


def link_success_artifacts(session: dict, attempt: dict) -> None:
    # Compatibility links are explicit references to the successful originals.
    # Never replace a foreign path, and never copy data to suggest a false origin.
    links = [(session["dir"] / "execution-manifest.json", attempt["execution_manifest"]),
             (session["dir"] / "msprof.log", attempt["msprof_log"])]
    links.extend((session["dir"] / raw.name, raw) for raw in attempt["dir"].glob("PROF_*"))
    for link, target in links:
        if link.exists() or link.is_symlink():
            raise ValueError(f"refusing to overwrite existing compatibility path: {link}")
        link.symlink_to(target.resolve(), target_is_directory=target.is_dir())


def _run_msprof_session(args: argparse.Namespace, application: str, session: dict) -> None:
    if len(application) >= MSPROF_APPLICATION_ARG_LIMIT:
        raise CollectionDataError("cli_rejected", "rendered --application must be shorter than 2048 characters")
    command = [args.msprof_bin, f"--application={application}", f"--output={session['dir']}",
               "--aic-mode=task-based", f"--aic-metrics={args.aic_metrics}"]
    command.extend(getattr(args, "_host_metadata_options", {}).get("flags", []))
    session.update({"command": command, "application_length": len(application),
                    "started_at": now_utc(), "returncode": None, "data_status": "pending"})
    atomic_write_json(session["dir"] / "command.json", {"command": command})
    with session["msprof_log"].open("x", encoding="utf-8") as log:
        process = subprocess.run(command, stdout=log, stderr=subprocess.STDOUT, text=True, timeout=args.timeout)
    session["returncode"] = process.returncode
    if process.returncode:
        raise RuntimeError(f"msprof exited {process.returncode}; see {session['msprof_log']}")


def _collect_sessions(args: argparse.Namespace, case_ids: list[str], out_dir: Path,
                      selected_warmup: dict) -> list[dict]:
    """Exact planned budgets, or conservative defaults; every attempt isolated."""
    sessions: list[dict] = []
    args._collection_sessions = sessions  # main can retain partial audit on failure
    plan_path = getattr(args, "planned_topology", None)
    plan = load_planned_topology(Path(plan_path), args, case_ids, selected_warmup) if plan_path else None
    args._planned_topology_receipt = ({"path": str(Path(plan_path).resolve()),
                                      "sha256": sha256_file(Path(plan_path)), "plan": plan} if plan else None)
    remaining = list(case_ids)
    target_estimate, preparation_estimate, observed_warmup = 1, 0, None
    retries_allowed = getattr(args, "empty_summary_retries", DEFAULT_EMPTY_SUMMARY_RETRIES)
    while remaining:
        number = len(sessions) + 1
        # The planning path is at least as long as the real single-session path;
        # max-attempt index is included in the rendered length budget.
        paths = [out_dir / f"session-{index:03d}" / f"attempt-{retries_allowed + 1:03d}" / "execution-manifest.json"
                 for index in range(number, number + len(remaining))]
        chunks = plan_session_chunks(remaining, selected_warmup, args.application_cmd, paths,
                                     args.max_launches_per_session, target_estimate, plan,
                                     preparation_estimate, observed_warmup)
        chunk = chunks[0]
        directory = out_dir if number == 1 and len(chunks) == 1 else out_dir / f"session-{number:03d}"
        directory.mkdir(parents=True, exist_ok=True)
        declared = {str(run["case_id"]): exact_case_launches(run) for run in (plan or {}).get("case_runs", [])}
        estimated = preparation_estimate + SAMPLES_PER_CASE * (
            (observed_warmup if observed_warmup is not None
             else warmup_kernel_estimate(selected_warmup)) + target_estimate)
        session = {"index": number, "dir": directory, "case_ids": chunk, "attempts": [], "retries": 0,
                   "planned_relevant_launches": sum(declared.get(case_id, estimated) for case_id in chunk)}
        sessions.append(session)
        for attempt_number in range(1, retries_allowed + 2):
            attempt_dir = directory / f"attempt-{attempt_number:03d}"
            attempt_dir.mkdir(exist_ok=False)
            attempt = {"index": attempt_number, "dir": attempt_dir,
                       "execution_manifest": attempt_dir / "execution-manifest.json",
                       "msprof_log": attempt_dir / "msprof.log"}
            session["attempts"].append(attempt)
            application = render_application_command(args.application_cmd, chunk,
                                                     attempt["execution_manifest"], selected_warmup)
            try:
                if args._planned_topology_receipt:
                    verify_planned_inputs(args, args._planned_topology_receipt)
                _run_msprof_session(args, application, attempt)
                if args._planned_topology_receipt:
                    verify_planned_inputs(args, args._planned_topology_receipt)
                if not attempt["execution_manifest"].is_file():
                    raise ValueError(f"runner did not create {attempt['execution_manifest']}")
                manifest = load_execution_manifest(attempt["execution_manifest"], chunk,
                                                   args.target_op, selected_warmup)
                validate_planned_execution(plan, manifest)
                actual = sum(exact_case_launches(run) for run in manifest["case_runs"])
                session["actual_relevant_launches"] = actual
                if actual > args.max_launches_per_session:
                    raise ValueError(
                        f"actual relevant launches {actual} exceed session cap {args.max_launches_per_session}")
                try:
                    summary = wait_for_op_summary(attempt_dir, timeout_seconds=0)
                except ValueError as error:
                    raise CollectionDataError("missing_summary", str(error)) from error
                # Header-only and zero-byte summaries are data failures even at RC 0.
                with summary.open(newline="", encoding="utf-8-sig") as stream:
                    if next(csv.DictReader(stream), None) is None:
                        raise CollectionDataError("empty_summary", f"empty op_summary: {summary}")
                parse_profile(summary, manifest)  # per-attempt expected == parsed == mapped
                attempt.update({"op_summary": summary, "data_status": "valid", "failure_category": None})
                audit_attempt(attempt)
                session.update({"execution_manifest": attempt["execution_manifest"],
                                "msprof_log": attempt["msprof_log"],
                                "op_summary": summary, "manifest": manifest})
                link_success_artifacts(session, attempt)
                break
            except (OSError, ValueError, RuntimeError, subprocess.TimeoutExpired) as error:
                category = classify_collection_failure(error, attempt["msprof_log"])
                attempt.update({"data_status": "invalid", "failure_category": category, "error": str(error)})
                audit_attempt(attempt)
                if category not in {"missing_summary", "empty_summary"} or attempt_number > retries_allowed:
                    raise
                session["retries"] += 1
        remaining = remaining[len(chunk):]
        if plan is None:
            # The next budget counts preparation independently instead of rounding
            # primes into the target-kernel count. Defaults remain conservative.
            target_estimate = max(
                target_estimate,
                max(len(run["target_kernel_sequence"]) for run in manifest["case_runs"]))
            preparation_estimate = max(
                preparation_estimate,
                max(len(run.get("preparation_kernel_sequence", [])) for run in manifest["case_runs"]))
            actual_warmup = max(len(run["warmup_kernel_sequence"]) for run in manifest["case_runs"])
            observed_warmup = max(observed_warmup or 0, actual_warmup)
    return sessions


def main() -> int:
    args = parse_args()
    started_at = now_utc()
    cases = Path(args.cases).resolve()
    runner_file = Path(args.runner_file).resolve()
    identity_manifest_path = Path(args.identity_manifest).resolve()
    compiled_warmup_binary = Path(args.compiled_warmup_binary)
    out_dir = Path(args.out_dir).resolve()
    result_dir = out_dir
    derived_from = None
    msprof_log = out_dir / "msprof.log"
    build_log = out_dir / "warmup-build.log"
    probe_log = out_dir / "warmup-ops-probe.log"
    probe: dict = {"status": "parse_only"}
    build: dict = {"status": "parse_only"}
    selected_warmup: dict = {}
    identity: dict = {}
    cann_environment: dict = {}
    sessions: list[dict] = []
    output_owned = False
    host_metadata = {"requested": getattr(args, "host_metadata_capture", "off"), "flags": []}
    planned_receipt = None
    captured_aic_metrics = args.aic_metrics
    captured_launch_cap = args.max_launches_per_session
    captured_retries = getattr(args, "empty_summary_retries", DEFAULT_EMPTY_SUMMARY_RETRIES)
    captured_application_sha = hashlib.sha256(args.application_cmd.encode("utf-8")).hexdigest()
    try:
        validate_args(args)
        case_ids = read_case_ids(cases)
        identity = load_identity_manifest(identity_manifest_path)
        cann_environment = inspect_cann_environment()
        if args.parse_only:
            original_collection_path = out_dir / "collection_manifest.json"
            existing_collection = json.loads(
                original_collection_path.read_text(encoding="utf-8")
            )
            derived_from = {"path": str(original_collection_path), "sha256": sha256_file(original_collection_path)}
            if existing_collection.get("status") != "complete":
                raise ValueError("parse-only requires a complete original collection")
            for key, path in (("cases_csv_sha256", cases), ("runner_file_sha256", runner_file),
                              ("identity_manifest_sha256", identity_manifest_path)):
                if existing_collection.get(key) and sha256_file(path) != existing_collection[key]:
                    raise ValueError(f"parse-only original collection identity drift: {key}")
            selected_warmup = dict(existing_collection.get("warmup", {}))
            host_metadata = existing_collection.get("host_metadata_capture", host_metadata)
            # A reparse does not recollect with the new CLI/environment. Preserve
            # the original acquisition protocol in the derived receipt.
            probe = existing_collection.get("warmup_ops_probe", probe)
            build = existing_collection.get("warmup_build", build)
            cann_environment = existing_collection.get("cann_environment", cann_environment)
            captured_aic_metrics = existing_collection.get("aic_metrics", captured_aic_metrics)
            captured_launch_cap = existing_collection.get("max_launches_per_session", captured_launch_cap)
            captured_retries = existing_collection.get("empty_summary_retries_allowed", captured_retries)
            captured_application_sha = existing_collection.get("application_command_sha256", captured_application_sha)
            planned_receipt = existing_collection.get("planned_topology")
            if planned_receipt:
                verify_planned_inputs(args, planned_receipt)
            if not selected_warmup:
                raise ValueError("parse-only collection manifest missing warmup selection")
            for entry in existing_collection.get("sessions", []):
                directory = Path(entry["dir"])
                sessions.append({
                    "index": entry["index"],
                    "dir": directory,
                    "execution_manifest": Path(entry["execution_manifest"]),
                    "msprof_log": Path(entry["msprof_log"]),
                    "op_summary": Path(entry["op_summary"]),
                    "case_ids": entry["case_ids"],
                    "attempts": entry.get("attempts", []),
                    "retries": entry.get("empty_summary_retries", 0),
                    "manifest": load_execution_manifest(
                        Path(entry["execution_manifest"]),
                        entry["case_ids"],
                        args.target_op,
                        selected_warmup,
                    ),
                })
            if not sessions:
                sessions = [{
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
                }]
            for session in sessions:
                validate_planned_execution(planned_receipt["plan"] if planned_receipt else None, session["manifest"])
                for attempt in session.get("attempts", []):
                    for receipt in attempt.get("files", []):
                        verify_file_receipt(receipt)
            result_dir = (Path(args.parse_out_dir).resolve() if getattr(args, "parse_out_dir", None)
                          else out_dir / f"parse-only-{dt.datetime.now(dt.timezone.utc):%Y%m%dT%H%M%S}-{os.getpid()}")
            if result_dir.exists():
                raise ValueError(f"parse-only derived output already exists: {result_dir}")
            result_dir.mkdir(parents=True, exist_ok=False)
            output_owned = True
        else:
            if out_dir.exists() and any(out_dir.iterdir()):
                raise ValueError(f"output directory is not empty: {out_dir}")
            out_dir.mkdir(parents=True, exist_ok=True)
            output_owned = True
            host_metadata = host_metadata_options(args, out_dir)
            args._host_metadata_options = host_metadata
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
            planned_receipt = args._planned_topology_receipt

        parsed_sessions = [(session["op_summary"], session["manifest"]) for session in sessions]
        rows, validation = parse_sessions(parsed_sessions)
        if [row["case_id"] for row in rows] != case_ids:
            raise ValueError("merged session case order does not match complete cases CSV")
        performance_csv = result_dir / "performance.csv"
        validation_json = result_dir / "validation.json"
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
        session_records = [session_record({**session, "case_ids": [
            str(run["case_id"]) for run in session["manifest"]["case_runs"]
        ]}) for session in sessions]
        atomic_write_json(result_dir / "collection_manifest.json", {
            "schema_version": MANIFEST_SCHEMA_VERSION,
            "status": "complete",
            "mode": (
                "msprof_application_single_session"
                if len(sessions) == 1 else "msprof_application_session_series"
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
            "max_launches_per_session": captured_launch_cap,
            "empty_summary_retries_allowed": captured_retries,
            "sessions": session_records,
            "host_metadata_capture": host_metadata,
            "planned_topology": planned_receipt,
            "derived_from_collection_manifest": derived_from,
            "paired_confirmation": (
                {"plan": {key: planned_receipt[key] for key in ("path", "sha256")},
                 "declaration": planned_receipt["plan"]["paired_confirmation"],
                 "case_labels": [{"case_id": run["case_id"], **run["confirmation"]}
                                 for run in planned_receipt["plan"]["case_runs"]]}
                if planned_receipt and planned_receipt["plan"].get("paired_confirmation") else None
            ),
            "aic_mode": "task-based",
            "aic_mode_rationale": (
                "required for op_summary export on CANN 9.0.0; verified harmless on 9.1.0"
            ),
            "aic_metrics": captured_aic_metrics,
            "runner_file": str(runner_file),
            "runner_file_sha256": sha256_file(runner_file),
            "identity_manifest": str(identity_manifest_path),
            "identity_manifest_sha256": sha256_file(identity_manifest_path),
            "identity": identity,
            "application_command_sha256": captured_application_sha,
            "execution_manifest": str(first_session["execution_manifest"]),
            "performance_csv": str(performance_csv),
            "validation_json": str(validation_json),
            "op_summary": str(first_session["op_summary"]),
            "msprof_log": str(first_session["msprof_log"]),
        })
        print(performance_csv)
        return 0
    except (OSError, ValueError, RuntimeError, subprocess.TimeoutExpired) as error:
        # A rejected nonempty destination belongs to its prior owner. Preserve it.
        if not output_owned:
            print(f"[ERROR] {error}", file=sys.stderr)
            return 1 if isinstance(error, (RuntimeError, subprocess.TimeoutExpired)) else 2
        sessions = getattr(args, "_collection_sessions", sessions)
        attempts = [attempt
                    for session in sessions
                    for attempt in session.get("attempts", [])]
        failure_category = (attempts[-1].get("failure_category") if attempts else None)
        result_dir.mkdir(parents=True, exist_ok=True)
        atomic_write_json(result_dir / "collection_manifest.json", {
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
            "host_metadata_capture": host_metadata,
            "planned_topology": getattr(args, "_planned_topology_receipt", planned_receipt),
            "sessions": [session_record(session) for session in sessions],
            "failure_category": failure_category or classify_collection_failure(error),
            "error": str(error),
        })
        print(f"[ERROR] {error}", file=sys.stderr)
        return 1 if isinstance(error, (RuntimeError, subprocess.TimeoutExpired)) else 2


if __name__ == "__main__":
    raise SystemExit(main())
