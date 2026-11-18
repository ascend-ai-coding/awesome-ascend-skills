#!/usr/bin/env python3
"""Verify a minimized ATK package and run one ATK invocation."""

import argparse
import hashlib
import json
import re
import shutil
import subprocess
import sys
from pathlib import Path


DATA_OPTIONS = {
    "input_directory": "--input_data",
    "benchmark_outputs_directory": "--bm_output_path",
}
REPEAT_OPTIONS = {
    "repeat_count": "--loop_nums",
    "determinism_repeat_count": "--dc_loop_nums",
}
BENCHMARK_OPTION = "--bm_device"


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def validate_digest(digest):
    if not isinstance(digest, str) or not re.fullmatch(r"[0-9a-f]{64}", digest):
        raise ValueError("Expected a lowercase SHA-256 digest")


def package_parts(name):
    if not isinstance(name, str) or name.startswith("-") or re.search(r"[\\:\x00-\x1f\x7f]", name):
        raise ValueError(f"Invalid package path: {name!r}")
    parts = name.split("/")
    if any(part in {"", ".", ".."} for part in parts):
        raise ValueError(f"Expected a relative, canonical package path: {name!r}")
    return parts


def package_file(root, name):
    path = root
    for part in package_parts(name):
        path = path / part
        if path.is_symlink():
            raise ValueError(f"Symlink in package path: {name!r}")
    if not path.is_file():
        raise ValueError(f"Missing package file: {name!r}")
    return path


def unique_keys(pairs):
    obj = {}
    for key, value in pairs:
        if key in obj:
            raise ValueError(f"Duplicate JSON key: {key!r}")
        obj[key] = value
    return obj


def load_manifest(root, expected_digest):
    raw = package_file(root, "repro.json").read_bytes()
    digest = hashlib.sha256(raw).hexdigest()
    if expected_digest is not None:
        validate_digest(expected_digest)
        if digest != expected_digest:
            raise ValueError("SHA-256 mismatch: repro.json")
    manifest = json.loads(raw, object_pairs_hook=unique_keys)
    if not isinstance(manifest, dict) or type(manifest.get("schema_version")) is not int or manifest["schema_version"] != 1:
        raise ValueError("Expected replay manifest schema_version 1")
    if type(manifest.get("case_id")) is not int:
        raise ValueError("case_id must be an integer")
    timeout = manifest.get("timeout_seconds", 1800)
    if type(timeout) is not int or timeout <= 0:
        raise ValueError("timeout_seconds must be a positive integer")
    benchmark = manifest.get("benchmark_device")
    if benchmark is not None and benchmark not in ("npu", "gpu", "cpu"):
        raise ValueError("benchmark_device must be npu, gpu, or cpu")
    for field in REPEAT_OPTIONS:
        value = manifest.get(field)
        if value is not None and (type(value) is not int or value <= 0):
            raise ValueError(f"{field} must be a positive integer")
    files = manifest.get("files")
    if not isinstance(files, dict) or not files or "repro.json" in files:
        raise ValueError("files must map payload paths to digests and exclude repro.json")
    entries = [("case_file", {".json"}), ("node_file", {".yaml", ".yml"})]
    if manifest.get("plugin_file") is not None:
        entries.append(("plugin_file", {".py"}))
    for key, suffixes in entries:
        name = manifest.get(key)
        if not isinstance(name, str) or name not in files or Path(name).suffix not in suffixes:
            raise ValueError(f"{key} must identify a listed file of the expected type")
    for name, file_digest in files.items():
        validate_digest(file_digest)
        package_file(root, name)
    for field in DATA_OPTIONS:
        directory = manifest.get(field)
        if directory is not None:
            package_parts(directory)
            if not any(name.startswith(directory + "/") for name in files):
                raise ValueError(f"{field} must contain files listed in the manifest")
    return manifest, raw, digest


def stage_package(source, destination, manifest, raw):
    destination.mkdir()
    for name, digest in manifest["files"].items():
        target = destination / name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(package_file(source, name), target)
        if sha256(target) != digest:
            raise ValueError(f"SHA-256 mismatch: {name!r}")
    (destination / "repro.json").write_bytes(raw)
    cases = json.loads((destination / manifest["case_file"]).read_text(encoding="utf-8"), object_pairs_hook=unique_keys)
    if not isinstance(cases, list) or len(cases) != 1 or not isinstance(cases[0], dict):
        raise ValueError("The native ATK case file must contain a JSON list with one case")
    case_id = cases[0].get("id")
    if type(case_id) is not int or case_id != manifest["case_id"]:
        raise ValueError("ATK id does not match manifest case_id")


def available_options(executable, work, log_path, requested):
    if not requested:
        return
    with log_path.open("x", encoding="utf-8") as log:
        result = subprocess.run(
            [executable, "task", "--help"],
            cwd=work,
            stdin=subprocess.DEVNULL,
            stdout=log,
            stderr=subprocess.STDOUT,
            check=False,
            timeout=30,
        )
    if result.returncode != 0:
        raise ValueError("ATK task --help failed; see task-help.log")
    help_text = log_path.read_text(encoding="utf-8", errors="replace")
    for option in requested:
        if not re.search(rf"(?<![\w-]){re.escape(option)}(?=[\s,=]|$)", help_text):
            raise ValueError(f"Installed ATK does not advertise {option}; see task-help.log")


def replay(package, output, atk, expected_digest=None):
    root = package.resolve(strict=True)
    if not root.is_dir():
        raise ValueError("Package must be a directory")
    output = output.resolve()
    if output == root or root in output.parents:
        raise ValueError("Use an output directory outside the source package")
    manifest, raw, manifest_digest = load_manifest(root, expected_digest)
    executable = shutil.which(atk)
    if executable is None:
        raise ValueError(f"ATK executable unavailable: {atk!r}")
    executable = str(Path(executable).absolute())

    output.mkdir()
    work = output / "work"
    argv = [executable, "task", "-c", manifest["case_file"], "-n", manifest["node_file"]]
    if manifest.get("plugin_file") is not None:
        argv.extend(["-p", manifest["plugin_file"]])
    argv.extend(["-to", str(manifest.get("timeout_seconds", 1800))])
    options = []
    option_fields = {**DATA_OPTIONS, "benchmark_device": BENCHMARK_OPTION, **REPEAT_OPTIONS}
    for field, option in option_fields.items():
        value = manifest.get(field)
        if value is not None:
            options.append(option)
            argv.extend([option, str(value)])
    receipt = {
        "case_id": manifest["case_id"],
        "manifest_sha256": manifest_digest,
        "argv": argv,
        "cwd": "work",
        "console_log": "console.log",
        "returncode": None,
        "execution_status": "PREPARING",
        "reproduction_verdict": "NOT_EVALUATED",
    }
    receipt_path = output / "execution.json"
    receipt_path.write_text(json.dumps(receipt, indent=2) + "\n", encoding="utf-8")
    try:
        stage_package(root, work, manifest, raw)
        available_options(executable, work, output / "task-help.log", options)
    except (OSError, ValueError, subprocess.TimeoutExpired) as error:
        receipt.update(execution_status="SETUP_ERROR", error=str(error))
        receipt_path.write_text(json.dumps(receipt, indent=2) + "\n", encoding="utf-8")
        print(f"Replay setup failed: {error}", file=sys.stderr)
        return 2
    receipt.update(execution_status="RUNNING")
    receipt_path.write_text(json.dumps(receipt, indent=2) + "\n", encoding="utf-8")
    with (output / "console.log").open("x", encoding="utf-8") as log:
        try:
            result = subprocess.run(argv, cwd=work, stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT, check=False)
        except OSError as error:
            receipt.update(execution_status="LAUNCH_ERROR", error=str(error))
            log.write(f"ATK launch error: {error}\n")
        else:
            receipt.update(execution_status="EXITED", returncode=result.returncode)
    receipt_path.write_text(json.dumps(receipt, indent=2) + "\n", encoding="utf-8")
    code = receipt["returncode"]
    if code is None:
        return 2
    return code if code >= 0 else 128 - code


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("package", type=Path, help="Directory containing repro.json and its payload")
    parser.add_argument("--output", type=Path, required=True, help="New directory outside the package")
    parser.add_argument("--atk", default="atk", help="ATK executable name or path")
    parser.add_argument("--manifest-sha256", help="Trusted SHA-256 for repro.json")
    args = parser.parse_args()
    try:
        return replay(args.package, args.output, args.atk, args.manifest_sha256)
    except (ValueError, OSError) as error:
        print(f"Replay setup failed: {error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
