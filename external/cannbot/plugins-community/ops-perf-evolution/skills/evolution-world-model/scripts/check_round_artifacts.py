#!/usr/bin/env python3
#
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
#
# ----------------------------------------------------------------------------------------------------------
"""
check_round_artifacts.py — Pre-refine artifact validation for LINGXI evolution.

Checks each parallel variant's artifacts before refine, producing a structured
JSON report. Designed to run between "collect results" (4.3.3) and "refine" (4.4.1).

Checks performed per variant:
  1. Kernel file existence: at least one .cpp or .h file in the kernel directory
  2. Kernel file change detection: diff against shared baseline (detects no-op copies)
  3. evaluation_results.json completeness: file exists + required fields present
  4. Build artifact check: .so or custom_opp directory exists (confirms compilation ran)

Usage:
  python3 plugins-community/ops-perf-evolution/skills/evolution-world-model/scripts/check_round_artifacts.py \
    --results-dir output/{op_name}_evo_{ts}/round_{r} \
    --shared-dir output/{op_name}_evo_{ts}/shared \
    --parallel-map '{"0":"n1","1":"n2","2":"x0"}' \
    --op-name {op_name} \
    [--mode lingxi|ops]

Output: JSON to stdout with per-variant check results + overall summary.
Exit code: 0 always (non-blocking), warnings/failures reported in JSON.
"""

import argparse
import filecmp
import glob
import json
import os
import sys
import logging

LOGGER = logging.getLogger(__name__)

# 数据输出专用 logger：CLI 结果走 stdout（agent 调用协议通道），
# 与 LOGGER（stderr 进度/警告）分离，避免 lint G.LOG.02 误报 print。
DATA_LOGGER = logging.getLogger(f"{__name__}.data")
_DATA_HANDLER = logging.StreamHandler(sys.stdout)
_DATA_HANDLER.setFormatter(logging.Formatter("%(message)s"))
DATA_LOGGER.addHandler(_DATA_HANDLER)
DATA_LOGGER.propagate = False
DATA_LOGGER.setLevel(logging.INFO)


_KERNEL_EXTS = ("cpp", "h")


def _find_kernel_files(variant_dir, op_name, mode):
    """Find kernel source files (.cpp/.h) in a variant directory.

    Many AscendC kernels are header-only templates living entirely in .h files,
    so scanning only .cpp produced false "kernel_unchanged" issues on operators
    like rms_norm (main body in rms_norm.h / rms_norm_base.h).
    """
    if mode == "ops":
        base_patterns = [
            os.path.join(variant_dir, "modified_files", "op_kernel", "*.{ext}"),
            os.path.join(variant_dir, "modified_files", "op_kernel", "**", "*.{ext}"),
        ]
    else:
        # TileLang flow: kernel files in kernel/ directory
        # Also check legacy DSL path for backward compatibility
        base_patterns = [
            os.path.join(variant_dir, "kernel", "*.{ext}"),
            os.path.join(variant_dir, "kernel", "**", "*.{ext}"),
            os.path.join(variant_dir, f"{op_name}Custom", "op_kernel", "*.{ext}"),
            os.path.join(variant_dir, f"{op_name}Custom", "op_kernel", "**", "*.{ext}"),
        ]
    files = []
    for pat in base_patterns:
        for ext in _KERNEL_EXTS:
            files.extend(glob.glob(pat.format(ext=ext), recursive=True))
    return sorted(set(files))


def _find_shared_kernels(shared_dir, op_name, mode):
    """Find baseline kernel source files (.cpp/.h) in shared directory."""
    if mode == "ops":
        base_patterns = [
            os.path.join(shared_dir, "original", "op_kernel", "*.{ext}"),
            os.path.join(shared_dir, "original", "op_kernel", "**", "*.{ext}"),
        ]
    else:
        # TileLang flow: kernel files in kernel/ directory
        # Also check legacy DSL path for backward compatibility
        base_patterns = [
            os.path.join(shared_dir, "kernel", "*.{ext}"),
            os.path.join(shared_dir, "kernel", "**", "*.{ext}"),
            os.path.join(shared_dir, f"{op_name}Custom", "op_kernel", "*.{ext}"),
            os.path.join(shared_dir, f"{op_name}Custom", "op_kernel", "**", "*.{ext}"),
        ]
    files = []
    for pat in base_patterns:
        for ext in _KERNEL_EXTS:
            files.extend(glob.glob(pat.format(ext=ext), recursive=True))
    return sorted(set(files))


def _check_kernel_changed(kernel_files, shared_dir, op_name, mode):
    """Compare kernel files against shared baseline to detect no-op copies."""
    shared_kernels = _find_shared_kernels(shared_dir, op_name, mode)
    if not shared_kernels:
        return {"changed": True, "reason": "no_shared_baseline"}

    shared_by_name = {os.path.basename(sp): sp for sp in shared_kernels}

    compared = 0
    for kf in kernel_files:
        shared_path = shared_by_name.get(os.path.basename(kf))
        if shared_path and os.path.isfile(shared_path):
            compared += 1
            if not filecmp.cmp(kf, shared_path, shallow=False):
                return {"changed": True, "reason": "files_differ"}

    if compared == 0:
        return {"changed": True, "reason": "no_matching_baseline"}
    return {"changed": False, "reason": "all_identical_to_baseline"}


def _check_eval_json(variant_dir):
    """Check evaluation_results.json existence and field completeness."""
    eval_path = os.path.join(variant_dir, "evaluation_results.json")
    if not os.path.isfile(eval_path):
        return {"exists": False, "valid": False, "reason": "file_not_found"}

    try:
        with open(eval_path, "r", encoding="utf-8") as f:
            data = json.load(f)
    except (json.JSONDecodeError, OSError) as e:
        return {"exists": True, "valid": False, "reason": f"parse_error: {e}"}

    comp = data.get("comparison", )
    if not isinstance(comp, dict):
        return {"exists": True, "valid": False, "reason": "missing_comparison_dict"}

    missing = []
    for field in ("compilation_success", "precision_passed", "speedup"):
        if field not in comp:
            missing.append(field)

    if missing:
        return {"exists": True, "valid": False,
                "reason": f"missing_fields: {missing}"}

    # Pipeline presence (profiling evidence for wm_ops.refine). Not fatal —
    # msprof can legitimately fail or be skipped when compilation/precision
    # fail. Only report a warning signal so downstream diagnose can tell
    # "no evidence" apart from "evidence says nothing interesting".
    evolved = data.get("evolved") if isinstance(data.get("evolved"), dict) else {}
    pipeline = evolved.get("pipeline") if isinstance(evolved, dict) else None
    pipeline_missing = not (isinstance(pipeline, dict) and pipeline)
    # Only flag missing when compilation+precision both passed — otherwise
    # skipping msprof is expected behavior.
    compile_ok = bool(comp.get("compilation_success"))
    precision_ok = bool(comp.get("precision_passed"))
    pipeline_should_exist = compile_ok and precision_ok

    return {"exists": True, "valid": True, "reason": "ok",
            "compilation_success": compile_ok,
            "precision_passed": precision_ok,
            "speedup": comp.get("speedup"),
            "pipeline_present": not pipeline_missing,
            "pipeline_expected": pipeline_should_exist}


def _find_build_artifacts(variant_dir, op_name, mode):
    """Find compiled .so files or custom_opp directories."""
    artifacts = []
    if mode == "ops":
        for so in glob.glob(os.path.join(variant_dir, "**", "*.so"), recursive=True):
            artifacts.append(so)
    else:
        # TileLang flow: .so can be anywhere in the variant dir
        for so in glob.glob(os.path.join(variant_dir, "**", "*.so"), recursive=True):
            artifacts.append(so)
        # Legacy DSL path
        build_out = os.path.join(variant_dir, f"{op_name}Custom", "build_out")
        if os.path.isdir(build_out):
            for so in glob.glob(os.path.join(build_out, "**", "*.so"), recursive=True):
                artifacts.append(so)
        for d in glob.glob(os.path.join(variant_dir, "custom_opp_*")):
            if os.path.isdir(d):
                artifacts.append(d)
    return sorted(set(artifacts))


def check_variant(variant_dir, shared_dir, op_name, mode):
    """Run all checks on a single variant directory."""
    result = {"variant_dir": variant_dir, "checks": {}, "status": "ok", "issues": []}

    # 1. Kernel file existence
    kernel_files = _find_kernel_files(variant_dir, op_name, mode)
    result["checks"]["kernel_exists"] = {
        "found": len(kernel_files) > 0,
        "count": len(kernel_files),
        "files": [os.path.basename(f) for f in kernel_files[:5]],
    }

    # 2. Kernel file change detection
    if kernel_files:
        result["checks"]["kernel_changed"] = _check_kernel_changed(
            kernel_files, shared_dir, op_name, mode)
    else:
        result["checks"]["kernel_changed"] = {
            "changed": False, "reason": "no_kernel_files"}

    # 3. evaluation_results.json completeness
    result["checks"]["eval_json"] = _check_eval_json(variant_dir)

    # 4. Build artifact check
    build_artifacts = _find_build_artifacts(variant_dir, op_name, mode)
    result["checks"]["build_artifacts"] = {
        "found": len(build_artifacts) > 0,
        "count": len(build_artifacts),
    }

    # Determine issues
    if not result["checks"]["kernel_exists"]["found"]:
        result["issues"].append("no_kernel_files")
    if not result["checks"]["kernel_changed"].get("changed", False):
        result["issues"].append("kernel_unchanged")
    if not result["checks"]["eval_json"].get("valid", False):
        result["issues"].append("eval_invalid")
    # Pipeline was expected (compile+precision both ok) but not written —
    # profiling_insight will be empty this round, decision tree degrades.
    if (result["checks"]["eval_json"].get("pipeline_expected") and
            not result["checks"]["eval_json"].get("pipeline_present")):
        result["issues"].append("pipeline_missing")
    if not result["checks"]["build_artifacts"]["found"]:
        result["issues"].append("no_build_artifacts")

    if result["issues"]:
        result["status"] = "fail" if "no_kernel_files" in result["issues"] else "warn"

    return result


def main():
    parser = argparse.ArgumentParser(
        description="Pre-refine artifact validation for LINGXI evolution rounds."
    )
    parser.add_argument("--results-dir", required=True)
    parser.add_argument("--shared-dir", required=True)
    parser.add_argument("--parallel-map", required=True,
                        help='JSON: {"0":"n1","1":"n2","2":"x0"}')
    parser.add_argument("--op-name", required=True)
    parser.add_argument("--mode", default="lingxi", choices=["lingxi", "ops"])
    args = parser.parse_args()

    parallel_map = json.loads(args.parallel_map)
    results = []
    fail_count = 0
    warn_count = 0

    for p_idx_str, node_id in sorted(parallel_map.items()):
        p_idx = int(p_idx_str)
        variant_dir = os.path.join(args.results_dir, f"parallel_{p_idx}")

        if not os.path.isdir(variant_dir):
            results.append({
                "parallel_index": p_idx, "node_id": node_id,
                "status": "fail", "issues": ["variant_dir_missing"], "checks": {},
            })
            fail_count += 1
            continue

        check = check_variant(variant_dir, args.shared_dir, args.op_name, args.mode)
        check["parallel_index"] = p_idx
        check["node_id"] = node_id
        results.append(check)

        if check["status"] == "fail":
            fail_count += 1
        elif check["status"] == "warn":
            warn_count += 1

    report = {
        "total": len(parallel_map),
        "ok": len(parallel_map) - fail_count - warn_count,
        "warn": warn_count,
        "fail": fail_count,
        "variants": results,
    }

    DATA_LOGGER.info("%s", json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
