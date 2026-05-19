# ----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ----------------------------------------------------------------------------------------------------------
"""Collect baseline performance metadata for a model.

Parses prefill/decode timing from ModelRunner inference logs, collects
environment information (NPU model, CANN/torch versions), and writes
baseline_metadata.json.

Usage:
    python collect_baseline.py --model-name <name> --log <log_file> [--output <path>]
"""

import argparse
import json
import logging
import os
import re
import subprocess
import sys
from datetime import datetime, timezone

logger = logging.getLogger(__name__)


#
# Argument parsing
#

def parse_args():
    parser = argparse.ArgumentParser(
        description="Collect NPU inference baseline metadata."
    )
    parser.add_argument(
        "--model-name", default="", help="Model name used in log lines."
    )
    parser.add_argument(
        "--log", dest="log_file", required=True, help="Path to the inference log file."
    )
    parser.add_argument(
        "--output", default="./baseline/baseline_metadata.json",
        help="Output path for the JSON metadata file.",
    )
    parser.add_argument(
        "--model-source", default="", help="Model source (e.g. HF link)."
    )
    parser.add_argument(
        "--yaml", dest="yaml_file", default="",
        help="YAML config file to read model_name from.",
    )
    args = parser.parse_args()

    if args.yaml_file and not args.model_name:
        try:
            with open(args.yaml_file, encoding="utf-8") as fh:
                import yaml  # optional dependency
                cfg = yaml.safe_load(fh)
                args.model_name = cfg.get("model_name", "")
        except ImportError:
            logger.warning("PyYAML not installed; cannot read --yaml-file")
        except (OSError, ValueError) as exc:
            logger.warning("Failed to read %s: %s", args.yaml_file, exc)

    if not args.model_name:
        parser.error("--model-name is required (or use --yaml-file)")

    return args


#
# Log parsing
#

def parse_log(log_path, model_name):
    """Parse inference log and extract timing metrics."""
    with open(log_path, "r", encoding="utf-8") as fh:
        lines = fh.readlines()

    prefill_re = re.compile(
        re.escape(model_name) + r".*prefill.*?(\d+(?:\.\d+)?)\s*ms", re.I
    )
    decode_re = re.compile(
        re.escape(model_name) + r".*decode.*?(\d+(?:\.\d+)?)\s*ms", re.I
    )

    prefill_times = []
    decode_times = []
    output_text = ""

    pat_result = re.compile(
        r"Inference decode result(?:\s+for\s+.*)?\s*:\s*$"
    )

    for idx, line in enumerate(lines):
        m = prefill_re.search(line)
        if m:
            prefill_times.append((idx, float(m.group(1))))
            continue
        m = decode_re.search(line)
        if m:
            decode_times.append((idx, float(m.group(1))))
            continue
        if pat_result.search(line):
            if idx + 1 < len(lines):
                output_text = lines[idx + 1].strip()

    prefill_ms = prefill_times[-1][1] if prefill_times else None
    formal_line = prefill_times[-1][0] if prefill_times else 0
    formal_decodes = [ms for ln, ms in decode_times if ln > formal_line]
    decode_avg = (
        sum(formal_decodes) / len(formal_decodes)
        if formal_decodes
        else None
    )

    return prefill_ms, decode_avg, output_text


#
# Environment collection
#

def _run_cmd(cmd_list):
    """Run a command and return stdout, or empty string on failure."""
    try:
        result = subprocess.run(
            cmd_list, capture_output=True, text=True, timeout=10,
        )
        return result.stdout.strip()
    except (OSError, subprocess.TimeoutExpired):
        return ""


def get_npu_info():
    """Get NPU model and card count via npu-smi."""
    raw = _run_cmd(["npu-smi", "info"])
    if not raw:
        return "unknown", 0
    model_pat = re.compile(r"((?:Ascend|Atlas)\s*\S+)")
    models = model_pat.findall(raw)
    npu_model = models[0] if models else "unknown"
    id_pat = re.compile(r"^\s*(\d+)\s", re.MULTILINE)
    card_count = len(set(id_pat.findall(raw)))
    return npu_model, card_count


def get_cann_version():
    """Get CANN toolkit version from known paths."""
    version_paths = [
        "/usr/local/Ascend/ascend-toolkit/latest/version.cfg",
        "/usr/local/Ascend/ascend-toolkit/latest/version.info",
    ]
    for vp in version_paths:
        try:
            with open(vp, "r") as fh:
                content = fh.read()
            m = re.search(r"(\d+\.\d+\.\d+(?:\.\w+)?)", content)
            if m:
                return m.group(1)
        except OSError:
            continue
    return "unknown"


def get_python_package_version(pkg_name):
    """Get installed Python package version."""
    try:
        from importlib.metadata import version
        return version(pkg_name)
    except Exception:
        return "unknown"


def collect_environment():
    """Collect environment information."""
    return {
        "npu_model": _run_cmd(["npu-smi", "info"]),
        "cann_version": get_cann_version(),
        "torch_version": get_python_package_version("torch"),
        "torch_npu_version": get_python_package_version("torch_npu"),
    }


#
# Main
#

def main():
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    args = parse_args()

    if not os.path.isfile(args.log_file):
        logger.error("File not found: %s", args.log_file)
        sys.exit(1)

    prefill_ms, decode_ms, output_text = parse_log(args.log_file, args.model_name)

    if prefill_ms is None:
        logger.warning("No prefill timing found in log")
    if decode_ms is None:
        logger.warning("No decode timing found in log")

    metadata = {
        "model_name": args.model_name,
        "timestamp": datetime.now(tz=timezone.utc).isoformat(),
        "performance": {
            "prefill_ms": prefill_ms,
            "decode_avg_ms": decode_ms,
        },
        "output_sample": output_text[:200] if output_text else "",
    }

    output_dir = os.path.dirname(args.output) or "."
    os.makedirs(output_dir, exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False)

    logger.info("Baseline written to %s", args.output)


if __name__ == "__main__":
    main()
