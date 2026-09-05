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
from __future__ import annotations

import argparse
import hashlib
import json
import logging
import os
import platform
import re
import shlex
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, NamedTuple, Optional, Sequence, Tuple

logger = logging.getLogger(__name__)

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))
import runtime_guard as rtguard

# Single source of truth for the CLI capability parsers lives in
# collect_discovery; collect.py re-exports the same definitions. Aliased so the
# prepare_environment keyword parameters do not shadow module-level names.
from collect_discovery import parse_supported_metrics as _parse_supported_metrics
from collect_discovery import parse_supported_options as _parse_supported_options

cli_logger = logging.getLogger(__name__ + ".cli")
cli_logger.propagate = False
if not cli_logger.handlers:
    _cli_handler = logging.StreamHandler(sys.stdout)
    _cli_handler.setFormatter(logging.Formatter("%(message)s"))
    cli_logger.addHandler(_cli_handler)

PROFILE_SCHEMA = "msopprof-environment/v1"

# Keep the profile useful for CANN/msOpProf without silently persisting unrelated shell state.
REUSABLE_ENV_KEYS = {
    "PATH", "LD_LIBRARY_PATH", "LIBRARY_PATH", "PYTHONPATH", "CPATH",
    "CMAKE_PREFIX_PATH", "PKG_CONFIG_PATH", "ASCEND_HOME_PATH",
    "ASCEND_TOOLKIT_HOME", "ASCEND_OPP_PATH", "ASCEND_AICPU_PATH",
    "CANN_INSTALL_PATH", "TOOLCHAIN_HOME", "TBE_IMPL_PATH",
    "TVM_AICPU_LIBRARY_PATH", "ASCENDCL_HOME",
}
REUSABLE_ENV_PREFIXES = (
    "ASCEND_", "CANN_", "NPU_", "HCCL_", "TBE_", "TVM_",
    "TOOLCHAIN_", "ASCENDCL_", "ATB_",
)
SENSITIVE_NAME_PATTERN = re.compile(
    r"(?:TOKEN|SECRET|PASSWORD|PASSWD|CREDENTIAL|PRIVATE|COOKIE|AUTH|API[_-]?KEY|ACCESS[_-]?KEY)",
    re.IGNORECASE,
)


def now_iso() -> str:
    return datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds")


def write_json(path: Path, obj: Any, mode: int = 0o600) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding="utf-8")
    try:
        os.chmod(tmp, mode)
    except OSError:
        logger.debug("failed to chmod temp profile %s", tmp, exc_info=True)
    tmp.replace(path)
    try:
        os.chmod(path, mode)
    except OSError:
        logger.debug("failed to chmod profile %s", path, exc_info=True)


def read_json(path: Path, default: Any = None) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return default


def file_fingerprint(path: Path) -> Dict[str, Any]:
    st = path.stat()
    return {"size": st.st_size, "mtime_ns": st.st_mtime_ns}


def text_sha256(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8", errors="replace")).hexdigest()


def resolve_profile_path(raw: Optional[str], base_dir: Path) -> Optional[Path]:
    if not raw:
        return None
    path = Path(raw).expanduser()
    if not path.is_absolute():
        path = base_dir / path
    return path.resolve()


def parse_key_value(items: Sequence[str], option_name: str) -> Dict[str, str]:
    out: Dict[str, str] = {}
    for item in items:
        if "=" not in item:
            raise rtguard.UsageError(f"ERROR: {option_name} expects KEY=VALUE, got: {item}")
        key, value = item.split("=", 1)
        key = key.strip()
        if not key:
            raise rtguard.UsageError(f"ERROR: {option_name} contains an empty key: {item}")
        if SENSITIVE_NAME_PATTERN.search(key):
            raise rtguard.UsageError(
                f"ERROR: refusing to persist sensitive-looking environment variable {key}. "
                "Export secrets outside the reusable msOpProf profile."
            )
        out[key] = value
    return out


def is_reusable_environment_key(name: str, explicit_keys: Iterable[str] = ()) -> bool:
    if SENSITIVE_NAME_PATTERN.search(name):
        return False
    if name in set(explicit_keys):
        return True
    return name in REUSABLE_ENV_KEYS or name.startswith(REUSABLE_ENV_PREFIXES)


def snapshot_reusable_environment(
    environment: Mapping[str, str], explicit_keys: Iterable[str] = ()
) -> Dict[str, str]:
    explicit = set(explicit_keys)
    return {
        key: str(value)
        for key, value in sorted(environment.items())
        if is_reusable_environment_key(key, explicit)
    }


def apply_environment(values: Mapping[str, str]) -> None:
    for key, value in values.items():
        os.environ[str(key)] = str(value)


def capture_sourced_environment(
    commands: Sequence[str], cwd: Path, timeout: int
) -> Dict[str, Any]:
    """Execute source/init commands once and return the resulting complete environment.

    A NUL marker separates command output from `env -0`, so source scripts may print banners
    without corrupting parsing. The complete environment is used only in-memory for this run;
    the saved profile is filtered to CANN/msOpProf-relevant variables.
    """
    if not commands:
        return {
            "commands": [],
            "return_code": 0,
            "elapsed_seconds": 0.0,
            "environment": dict(os.environ),
            "stderr": "",
        }
    marker = "\0__MSOPPROF_ENV_BEGIN__\0"
    script = "set -e\n" + "\n".join(commands) + "\nprintf '\\0__MSOPPROF_ENV_BEGIN__\\0'\nenv -0\n"
    # The managed runner terminates the whole process group on timeout so
    # grandchildren spawned by source/init scripts cannot leak.
    result = rtguard.run_managed_process(rtguard.ManagedProcessSpec(
        ["bash", "-lc", script],
        timeout=timeout,
        cwd=cwd,
        heartbeat_seconds=0,
    ))
    stdout = result.get("stdout") or ""
    payload = stdout.split(marker, 1)[1] if marker in stdout else ""
    environment: Dict[str, str] = {}
    for item in payload.split("\0"):
        if "=" not in item:
            continue
        key, value = item.split("=", 1)
        environment[key] = value
    return {
        "commands": list(commands),
        "return_code": result.get("return_code"),
        "elapsed_seconds": result.get("elapsed_seconds"),
        "environment": environment,
        "stderr": result.get("stderr") or "",
    }


def run_help(msprof_path: str, timeout: int = 20) -> Dict[str, Any]:
    result = rtguard.run_managed_process(rtguard.ManagedProcessSpec(
        [msprof_path, "op", "--help"],
        timeout=timeout,
        heartbeat_seconds=0,
    ))
    text = (result.get("stdout") or "") + "\n" + (result.get("stderr") or "")
    return {
        "return_code": result.get("return_code"),
        "help_text": text,
        "elapsed_seconds": result.get("elapsed_seconds"),
    }


def find_executable(requested: str) -> str:
    resolved = shutil.which(requested)
    if resolved:
        return str(Path(resolved).resolve())
    candidate = Path(requested).expanduser()
    if candidate.is_file():
        return str(candidate.resolve())
    raise rtguard.UsageError(
        ("ERROR: msprof not found. Initialize/reuse an environment profile, source the CANN environment, or "
            "pass --msprof.")
    )


def _source_path_candidates(commands: Sequence[str], cwd: Path) -> List[Path]:
    candidates: List[Path] = []
    for command in commands:
        try:
            tokens = shlex.split(command, posix=True)
        except ValueError:
            continue
        for index, token in enumerate(tokens[:-1]):
            if token not in {"source", "."}:
                continue
            raw = os.path.expandvars(tokens[index + 1])
            path = Path(raw).expanduser()
            if not path.is_absolute():
                path = cwd / path
            path = path.resolve()
            if path.is_file() and path not in candidates:
                candidates.append(path)
    return candidates


def source_file_records(commands: Sequence[str], cwd: Path) -> List[Dict[str, Any]]:
    return [
        {"path": str(path), "fingerprint": file_fingerprint(path)}
        for path in _source_path_candidates(commands, cwd)
    ]


def host_signature() -> Dict[str, str]:
    return {
        "system": platform.system(),
        "machine": platform.machine(),
        "python": platform.python_version(),
    }


class ProfileInputs(NamedTuple):
    """Bundled inputs for ``build_profile`` (G.FNM.03)."""

    profile_path: Path
    source_commands: Sequence[str]
    source_cwd: Path
    environment: Mapping[str, str]
    explicit_env_keys: Iterable[str]
    requested_msprof: str
    msprof_path: str
    help_text: str
    supported_metrics: Sequence[str]
    supported_options: Sequence[str]


def _profile_reuse_contract() -> Dict[str, List[str]]:
    return {
        "skip_when_valid": [
            "CANN source/init commands",
            "msprof executable discovery",
            "msprof op --help capability probe",
        ],
        "always_recheck": [
            "operator executable resolution",
            "operator BasicInfo/kernel discovery",
            "debug-symbol and instrumentation evidence",
            "requested feature payload validity",
        ],
        "lightweight_validation": [
            "profile schema",
            "host system/machine",
            "source-file fingerprints",
            "msprof path and fingerprint",
        ],
    }


def build_profile(inputs: ProfileInputs) -> Dict[str, Any]:
    executable = Path(inputs.msprof_path).resolve()
    return {
        "schema": PROFILE_SCHEMA,
        "created_at": now_iso(),
        "profile_path": str(inputs.profile_path),
        "path_base": str(inputs.profile_path.parent),
        "scope": ("Reusable host/CANN environment context; operator-specific binaries and kernels are "
            "intentionally excluded."),
        "host": host_signature(),
        "source": {
            "commands": list(inputs.source_commands),
            "cwd": str(inputs.source_cwd.resolve()),
            "files": source_file_records(inputs.source_commands, inputs.source_cwd),
        },
        "environment": snapshot_reusable_environment(inputs.environment, inputs.explicit_env_keys),
        "tools": {
            "msprof": {
                "requested": inputs.requested_msprof,
                "path": str(executable),
                "fingerprint": file_fingerprint(executable),
            }
        },
        "cli": {
            "help_sha256": text_sha256(inputs.help_text),
            "help_text": inputs.help_text,
            "supported_metrics": list(inputs.supported_metrics),
            "supported_options": list(inputs.supported_options),
        },
        "reuse_contract": _profile_reuse_contract(),
    }


def _validate_host(profile: Mapping[str, Any], reasons: List[str]) -> None:
    saved_host = profile.get("host") or {}
    current_host = host_signature()
    for key in ("system", "machine"):
        if saved_host.get(key) and saved_host.get(key) != current_host.get(key):
            reasons.append(f"host {key} changed")


def _validate_source_files(
    profile: Mapping[str, Any],
    expected_source_commands: Optional[Sequence[str]],
    reasons: List[str],
) -> None:
    saved_commands = list((profile.get("source") or {}).get("commands") or [])
    if expected_source_commands is not None and list(expected_source_commands) != saved_commands:
        reasons.append("source command list changed")
    for record in (profile.get("source") or {}).get("files") or []:
        path = Path(str(record.get("path", ""))).expanduser()
        if not path.is_file():
            reasons.append(f"source file missing: {path}")
            continue
        if record.get("fingerprint") != file_fingerprint(path):
            reasons.append(f"source file changed: {path}")


def _validate_msprof_record(
    profile: Mapping[str, Any],
    requested_msprof: Optional[str],
    reasons: List[str],
) -> None:
    msprof_record = ((profile.get("tools") or {}).get("msprof") or {})
    msprof_path = Path(str(msprof_record.get("path", ""))).expanduser()
    if not msprof_path.is_file():
        reasons.append(f"msprof path missing: {msprof_path}")
    elif msprof_record.get("fingerprint") != file_fingerprint(msprof_path):
        reasons.append(f"msprof binary changed: {msprof_path}")
    if requested_msprof and requested_msprof != "msprof":
        requested_path = shutil.which(requested_msprof) or requested_msprof
        try:
            if Path(requested_path).expanduser().resolve() != msprof_path.resolve():
                reasons.append("explicit --msprof differs from profile")
        except OSError:
            reasons.append("explicit --msprof cannot be resolved")


def _validate_cli_inventory(profile: Mapping[str, Any], reasons: List[str]) -> None:
    cli = profile.get("cli") or {}
    if not isinstance(cli.get("supported_metrics"), list):
        reasons.append("supported metric inventory missing")
    if not isinstance(cli.get("supported_options"), list):
        reasons.append("supported option inventory missing")
    if not isinstance(cli.get("help_text"), str):
        reasons.append("saved msprof help text missing")
    if not isinstance(profile.get("environment"), dict):
        reasons.append("saved environment map missing")


def validate_profile(
    profile: Mapping[str, Any],
    *,
    expected_source_commands: Optional[Sequence[str]] = None,
    requested_msprof: Optional[str] = None,
) -> Dict[str, Any]:
    reasons: List[str] = []
    if profile.get("schema") != PROFILE_SCHEMA:
        reasons.append("schema mismatch")
    _validate_host(profile, reasons)
    _validate_source_files(profile, expected_source_commands, reasons)
    _validate_msprof_record(profile, requested_msprof, reasons)
    _validate_cli_inventory(profile, reasons)
    return {
        "valid": not reasons,
        "reasons": reasons,
        "validated_at": now_iso(),
        "method": "metadata_only",
    }


class _ProbeResult(NamedTuple):
    msprof_path: str
    help_result: Dict[str, Any]
    help_text: str
    supported_metrics: List[str]
    supported_options: List[str]


def _initial_validation(
    existing: Mapping[str, Any],
    source_commands: Sequence[str],
    requested_msprof: str,
) -> Dict[str, Any]:
    if not existing:
        return {
            "valid": False,
            "reasons": ["profile not found"],
            "validated_at": now_iso(),
            "method": "metadata_only",
        }
    return validate_profile(
        existing,
        expected_source_commands=list(source_commands) if source_commands else None,
        requested_msprof=requested_msprof,
    )


def _reuse_profile_result(
    existing: Mapping[str, Any],
    request: "EnvironmentRequest",
    validation: Mapping[str, Any],
    saved_commands: Sequence[str],
) -> Dict[str, Any]:
    apply_environment(existing.get("environment") or {})
    apply_environment(dict(request.explicit_environment or {}))
    cli = existing.get("cli") or {}
    tools = existing.get("tools") or {}
    msprof_record = tools.get("msprof") or {}
    msprof_path = str(msprof_record.get("path"))
    return {
        "status": "reused",
        "profile_path": str(request.profile_path) if request.profile_path else None,
        "profile_mode": request.mode,
        "profile_validation": validation,
        "source_commands_executed": False,
        "cli_probe_executed": False,
        "msprof": msprof_path,
        "help_text": cli.get("help_text", ""),
        "supported_metrics": list(cli.get("supported_metrics") or []),
        "supported_options": list(cli.get("supported_options") or []),
        "saved_environment_keys": sorted((existing.get("environment") or {}).keys()),
        "source_commands": saved_commands,
    }


def _source_or_raise(commands: Sequence[str], cwd: Path, timeout: int) -> Dict[str, Any]:
    source_result = capture_sourced_environment(commands, cwd, timeout)
    if source_result.get("return_code") != 0:
        raise rtguard.UsageError(
            "ERROR: environment source/init command failed: "
            + (source_result.get("stderr") or f"return code {source_result.get('return_code')}")
        )
    return source_result


def _probe_msprof(requested_msprof: str, parse_metrics: Any, parse_options: Any) -> _ProbeResult:
    msprof_path = find_executable(requested_msprof)
    help_result = run_help(msprof_path)
    if help_result.get("return_code") not in {0, 1} and not help_result.get("help_text"):
        raise rtguard.UsageError("ERROR: unable to inspect `msprof op --help`.")
    help_text = str(help_result.get("help_text") or "")
    return _ProbeResult(
        msprof_path=msprof_path,
        help_result=help_result,
        help_text=help_text,
        supported_metrics=list(parse_metrics(help_text)),
        supported_options=list(parse_options(help_text)),
    )


def _persist_profile(
    profile_path: Path,
    existing: Mapping[str, Any],
    inputs: ProfileInputs,
) -> Tuple[str, Dict[str, Any], List[str]]:
    """Write the refreshed profile; return (status, validation, saved_keys)."""
    profile = build_profile(inputs)
    write_json(profile_path, profile)
    status = "refreshed" if existing else "created"
    validation = validate_profile(profile, requested_msprof=inputs.requested_msprof)
    saved_keys = sorted((profile.get("environment") or {}).keys())
    return status, validation, saved_keys


def _probed_result(
    request: "EnvironmentRequest",
    source_result: Mapping[str, Any],
    probed: _ProbeResult,
    outcome: Tuple[str, Dict[str, Any], List[str]],
    effective_commands: Sequence[str],
) -> Dict[str, Any]:
    status, validation, saved_keys = outcome
    source_summary = {
        "return_code": source_result.get("return_code"),
        "elapsed_seconds": source_result.get("elapsed_seconds"),
        "stderr": source_result.get("stderr"),
    }
    return {
        "status": status,
        "profile_path": str(request.profile_path) if request.profile_path else None,
        "profile_mode": request.mode,
        "profile_validation": validation,
        "source_commands_executed": bool(effective_commands),
        "source_result": source_summary,
        "cli_probe_executed": True,
        "cli_probe_elapsed_seconds": probed.help_result.get("elapsed_seconds"),
        "msprof": probed.msprof_path,
        "help_text": probed.help_text,
        "supported_metrics": probed.supported_metrics,
        "supported_options": probed.supported_options,
        "saved_environment_keys": saved_keys,
        "source_commands": effective_commands,
    }


def _check_mode(profile_path: Optional[Path], mode: str) -> None:
    if mode not in {"auto", "readonly", "refresh", "off"}:
        raise rtguard.UsageError(f"ERROR: unsupported environment profile mode: {mode}")
    if mode in {"readonly", "refresh"} and profile_path is None:
        raise rtguard.UsageError(f"ERROR: --env-profile-mode {mode} requires --env-profile.")


def _probed_only_keys(explicit_environment: Mapping[str, str]) -> List[str]:
    return sorted(snapshot_reusable_environment(os.environ, explicit_environment.keys()).keys())


def _existing_profile_state(
    profile_path: Optional[Path],
    source_commands: Sequence[str],
) -> Tuple[Dict[str, Any], List[str], List[str]]:
    """Return (existing profile, saved commands, effective commands)."""
    existing = read_json(profile_path, {}) if profile_path and profile_path.is_file() else {}
    saved_commands = list((existing.get("source") or {}).get("commands") or []) if existing else []
    effective_commands = list(source_commands) if source_commands else saved_commands
    return existing, saved_commands, effective_commands


class EnvironmentRequest(NamedTuple):
    """Bundled inputs for ``prepare_environment`` (G.FNM.03)."""

    profile_path: Optional[Path]
    mode: str
    source_commands: Sequence[str]
    source_cwd: Path
    requested_msprof: str
    source_timeout: int
    parse_supported_metrics: Any
    parse_supported_options: Any
    explicit_environment: Optional[Mapping[str, str]] = None


def prepare_environment(request: EnvironmentRequest) -> Dict[str, Any]:
    """Prepare the current process environment and return a capability snapshot.

    `auto` reuses a valid profile and otherwise refreshes it. `readonly` requires a valid
    profile. `refresh` always executes source/probe steps and rewrites the profile. `off`
    ignores the profile but still honors source commands for the current invocation.
    """
    _check_mode(request.profile_path, request.mode)

    explicit_environment = dict(request.explicit_environment or {})
    existing, saved_commands, effective_commands = _existing_profile_state(
        request.profile_path, request.source_commands
    )
    validation = _initial_validation(existing, request.source_commands, request.requested_msprof)

    if request.mode in {"auto", "readonly"} and validation.get("valid"):
        return _reuse_profile_result(
            existing, request, validation, saved_commands
        )

    if request.mode == "readonly":
        raise rtguard.UsageError(
            "ERROR: reusable environment profile is unavailable or invalid: "
            + "; ".join(validation.get("reasons") or ["unknown reason"])
        )

    source_result = _source_or_raise(effective_commands, request.source_cwd, request.source_timeout)
    apply_environment(source_result.get("environment") or {})
    apply_environment(explicit_environment)
    probed = _probe_msprof(
        request.requested_msprof, request.parse_supported_metrics, request.parse_supported_options
    )

    if request.profile_path is not None and request.mode != "off":
        inputs = ProfileInputs(
            profile_path=request.profile_path, source_commands=effective_commands,
            source_cwd=request.source_cwd,
            environment=os.environ, explicit_env_keys=explicit_environment.keys(),
            requested_msprof=request.requested_msprof, msprof_path=probed.msprof_path,
            help_text=probed.help_text,
            supported_metrics=probed.supported_metrics, supported_options=probed.supported_options,
        )
        outcome = _persist_profile(request.profile_path, existing, inputs)
    else:
        outcome = ("probed", validation, _probed_only_keys(explicit_environment))

    return _probed_result(
        request, source_result, probed, outcome, effective_commands
    )


def public_context(result: Mapping[str, Any]) -> Dict[str, Any]:
    """Return manifest-safe environment context without persisting variable values."""
    return {
        "status": result.get("status"),
        "profile_path": result.get("profile_path"),
        "profile_mode": result.get("profile_mode"),
        "profile_validation": result.get("profile_validation"),
        "source_commands_executed": bool(result.get("source_commands_executed")),
        "cli_probe_executed": bool(result.get("cli_probe_executed")),
        "msprof": result.get("msprof"),
        "saved_environment_keys": list(result.get("saved_environment_keys") or []),
        "source_commands": list(result.get("source_commands") or []),
        "reuse_note": (
            "A valid profile skips environment sourcing, msprof path discovery, and msprof op --help. "
            "Operator-specific discovery and payload validation still run."
        ),
    }


def _source_summary(profile: Mapping[str, Any]) -> Dict[str, Any]:
    return {
        "source_commands": list((profile.get("source") or {}).get("commands") or []),
        "source_files": [x.get("path") for x in ((profile.get("source") or {}).get("files") or [])],
        "environment_keys": sorted((profile.get("environment") or {}).keys()),
    }


def _summary(profile: Mapping[str, Any], validation: Mapping[str, Any]) -> Dict[str, Any]:
    return {
        "schema": profile.get("schema"),
        "created_at": profile.get("created_at"),
        "profile_path": profile.get("profile_path"),
        "valid": validation.get("valid"),
        "validation_reasons": validation.get("reasons"),
        **_source_summary(profile),
        "msprof": (((profile.get("tools") or {}).get("msprof") or {}).get("path")),
        "supported_metrics": list((profile.get("cli") or {}).get("supported_metrics") or []),
        "supported_options": list((profile.get("cli") or {}).get("supported_options") or []),
    }


def _run_init_command(args: argparse.Namespace, base_dir: Path, profile_path: Path) -> int:
    explicit = parse_key_value(args.env_var, "--env-var")
    result = prepare_environment(EnvironmentRequest(
        profile_path=profile_path,
        mode="refresh",
        source_commands=args.source,
        source_cwd=base_dir,
        requested_msprof=args.msprof,
        source_timeout=args.timeout,
        parse_supported_metrics=_parse_supported_metrics,
        parse_supported_options=_parse_supported_options,
        explicit_environment=explicit,
    ))
    profile = read_json(profile_path, {})
    cli_logger.info(json.dumps({"result": public_context(result), "profile": _summary(profile,
        validate_profile(profile))}, ensure_ascii=False, indent=2))
    return 0


def main(argv: Optional[Sequence[str]] = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    parser = argparse.ArgumentParser(
        description="Create, validate, or inspect a reusable msOpProf/CANN environment profile."
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    init = subparsers.add_parser("init", help="Source the environment, probe msprof once, and save a reusable profile.")
    init.add_argument("--profile", required=True)
    init.add_argument("--base-dir", default=".")
    init.add_argument("--source", action="append", default=[], help="Repeatable shell source/init command.")
    init.add_argument("--env-var", action="append", default=[], metavar="KEY=VALUE")
    init.add_argument("--msprof", default="msprof")
    init.add_argument("--timeout", type=int, default=120)

    validate = subparsers.add_parser("validate", help=("Perform metadata-only validation without sourcing or "
        "running msprof."))
    validate.add_argument("--profile", required=True)
    validate.add_argument("--base-dir", default=".")

    show = subparsers.add_parser("show", help=("Show reusable paths, commands, keys, and capabilities without "
        "environment values."))
    show.add_argument("--profile", required=True)
    show.add_argument("--base-dir", default=".")

    args = parser.parse_args(argv)
    base_dir = Path(args.base_dir).expanduser().resolve()
    profile_path = resolve_profile_path(args.profile, base_dir)
    if profile_path is None:
        raise ValueError("--profile is required")

    try:
        if args.command == "init":
            return _run_init_command(args, base_dir, profile_path)
        profile = read_json(profile_path, {})
        validation = validate_profile(profile)
        cli_logger.info(json.dumps(_summary(profile, validation), ensure_ascii=False, indent=2))
        return 0 if validation.get("valid") else 2
    except rtguard.UsageError as exc:
        rtguard.log_usage_error(exc)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
