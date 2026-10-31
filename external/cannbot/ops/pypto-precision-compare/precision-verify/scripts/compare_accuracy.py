#!/usr/bin/env python3
# coding: utf-8
# Copyright (c) 2025-2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""
PyPTO 精度检查点文件对比工具
自动检测检查点并对比 jit 和 golden 的中间结果
"""

import argparse
import glob
import logging
import math
import os
import re
import sys
from dataclasses import dataclass, replace

import torch

# PyPTO 数据类型映射
# 参考：https://raw.gitcode.com/cann/pypto/raw/master/docs/zh/api/datatype/DataType.md
DTYPE_MAP = {
    0: ("int4", torch.int8, 1),  # DT_INT4: 4位有符号整数，2个元素打包成1个int8存储
    1: ("int8", torch.int8, 1),  # DT_INT8: 8位有符号整数
    2: ("int16", torch.int16, 2),  # DT_INT16: 16位有符号整数
    3: ("int32", torch.int32, 4),  # DT_INT32: 32位有符号整数
    4: ("int64", torch.int64, 8),  # DT_INT64: 64位有符号整数
    5: ("fp8", torch.float8_e4m3fn, 1),  # DT_FP8: 8位浮点数（通用）
    6: ("fp16", torch.float16, 2),  # DT_FP16: 16位半精度浮点数
    7: ("fp32", torch.float32, 4),  # DT_FP32: 32位单精度浮点数
    8: ("bf16", torch.bfloat16, 2),  # DT_BF16: 16位Brain Float格式
    9: ("hf4", torch.uint8, 1),  # DT_HF4: 4位Half Float格式，2个元素打包成1个uint8存储
    10: ("hf8", torch.uint8, 1),  # DT_HF8: 8位Half Float格式，原始字节存储
    11: ("uint8", torch.uint8, 1),  # DT_UINT8: 8位无符号整数
    12: ("uint16", torch.uint16, 2),  # DT_UINT16: 16位无符号整数
    13: ("uint32", torch.uint32, 4),  # DT_UINT32: 32位无符号整数
    14: ("uint64", torch.uint64, 8),  # DT_UINT64: 64位无符号整数
    15: ("bool", torch.bool, 1),  # DT_BOOL: 布尔类型
    16: ("double", torch.float64, 8),  # DT_DOUBLE: 64位双精度浮点数
    17: ("fp8_e4m3", torch.float8_e4m3fn, 1),  # DT_FP8E4M3: 8位浮点数，4位指数，3位尾数
    18: ("fp8_e5m2", torch.float8_e5m2, 1),  # DT_FP8E5M2: 8位浮点数，5位指数，2位尾数
    19: (
        "fp8_e8m0",
        torch.uint8,
        1,
    ),  # DT_FP8E8M0: 8位浮点数，8位指数，0位尾数（scale专用）
    20: (
        "fp4_e2m1x2",
        torch.uint8,
        1,
    ),  # DT_FP4_E2M1X2: 4位浮点数，2位指数，1位尾数，x2打包
    21: (
        "fp4_e1m2x2",
        torch.uint8,
        1,
    ),  # DT_FP4_E1M2X2: 4位浮点数，1位指数，2位尾数，x2打包
}

UNSUPPORTED_PACKED_DTYPES = {0, 9, 20, 21}

# 容差标准映射表（基于 PyPTO 代码库实践）
# 参考：verify.md、examples、models 中的实际使用标准
TOLERANCE_MAP = {
    0: (0, 1e-4),  # INT4:  整数类型，允许极少误差点
    1: (0, 1e-4),  # INT8:  整数类型，允许极少误差点
    2: (0, 1e-4),  # INT16: 整数类型，允许极少误差点
    3: (0, 1e-4),  # INT32: 整数类型，允许极少误差点
    4: (0, 1e-4),  # INT64: 整数类型，允许极少误差点
    5: (1e-1, 1e-2),  # FP8:   8位浮点，量化场景精度较低
    6: (1e-3, 1e-3),  # FP16: 半精度浮点，mantissa=10bits
    7: (1e-3, 1e-4),  # FP32: 单精度浮点，高精度基准
    8: (5e-3, 5e-2),  # BF16: BFloat16，mantissa=7bits，广泛使用于attention
    9: (1e-1, 1e-2),  # HF4:  4位Half Float，低精度量化
    10: (1e-1, 1e-2),  # HF8:  8位Half Float，低精度量化
    11: (0, 1e-4),  # UINT8:  无符号整数，允许极少误差点
    12: (0, 1e-4),  # UINT16: 无符号整数，允许极少误差点
    13: (0, 1e-4),  # UINT32: 无符号整数，允许极少误差点
    14: (0, 1e-4),  # UINT64: 无符号整数，允许极少误差点
    15: (0, 1e-4),  # BOOL:   布尔类型，允许极少误差点
    16: (1e-6, 1e-6),  # DOUBLE: 64位双精度浮点，高精度
    17: (1e-1, 1e-2),  # FP8E4M3: 8位浮点(4位指数,3位尾数)，量化场景
    18: (1e-1, 1e-2),  # FP8E5M2: 8位浮点(5位指数,2位尾数)，量化场景
    19: (1e-1, 1e-2),  # FP8E8M0: 8位浮点(8位指数,0位尾数)，MXFP8 scale
    20: (1e-1, 1e-2),  # FP4_E2M1X2: 4位浮点(2位指数,1位尾数)，量化场景
    21: (1e-1, 1e-2),  # FP4_E1M2X2: 4位浮点(1位指数,2位尾数)，量化场景
}

# 初始化日志
logging.basicConfig(
    level=logging.INFO,
    format="%(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class ComparisonOptions:
    """Tolerance and output controls for one tensor comparison."""

    dtype: int | None = None
    verbose: bool = True
    custom_rtol: float | None = None
    custom_atol: float | None = None


def find_latest_output_dir(work_dir="."):
    """找到最新的 output 目录"""
    output_base = os.path.join(work_dir, "output")
    if not os.path.exists(output_base):
        return None

    output_dirs = [d for d in os.listdir(output_base) if d.startswith("output_")]
    if not output_dirs:
        return None

    output_dirs.sort(
        key=lambda x: os.path.getmtime(os.path.join(output_base, x)), reverse=True
    )
    return output_dirs[0]


def scan_checkpoints_from_dir(tensor_dir):
    """从指定目录扫描所有检查点文件"""
    if not os.path.exists(tensor_dir):
        return []

    data_files = glob.glob(os.path.join(tensor_dir, "*.data"))
    checkpoints = set()

    for data_file in data_files:
        basename = os.path.basename(data_file)
        parts = basename.rsplit("_", 1)
        if len(parts) == 2 and parts[1].replace(".data", "").isdigit():
            checkpoint_name = parts[0]
            checkpoints.add(checkpoint_name)

    def get_sort_key(name):
        match = re.match(r"^(\d+)", name)
        return int(match.group(1)) if match else float("inf")

    return sorted(list(checkpoints), key=get_sort_key)


def get_tolerance_by_dtype(dtype, custom_rtol=None, custom_atol=None):
    """根据数据类型返回对应的容差标准"""
    if custom_rtol is not None and custom_atol is not None:
        return custom_rtol, custom_atol
    return TOLERANCE_MAP.get(dtype, (1e-3, 1e-3))


def _read_jit_metadata(csv_file):
    dtype = None
    shape = None
    if not os.path.exists(csv_file):
        return dtype, shape
    with open(csv_file, "r", encoding="utf-8") as metadata:
        for line in metadata:
            if line.startswith("dtype,"):
                dtype = int(line.split(",")[1].strip())
            elif line.startswith("shape,"):
                shape = tuple(map(int, line.split(",")[1].strip().split("x")))
    return dtype, shape


def read_jit_data(filename):
    """读取 jit 生成的数据文件，自动检测数据类型"""
    dtype, shape = _read_jit_metadata(filename.replace(".data", ".csv"))

    if dtype is None:
        logger.warning("  Warning: CSV file not found, cannot determine data type")
        return None, None

    if dtype not in DTYPE_MAP:
        logger.warning(f"  Warning: unknown data type {dtype}")
        return None, None
    if dtype in UNSUPPORTED_PACKED_DTYPES:
        type_name = DTYPE_MAP[dtype][0]
        logger.warning(
            f"  Warning: unpacking data type {type_name} not supported yet (dtype={dtype})"
        )
        return None, dtype

    _type_name, torch_dtype, _bytes_per_element = DTYPE_MAP[dtype]

    with open(filename, "rb") as f:
        raw_bytes = f.read()
    data_tensor = torch.frombuffer(bytearray(raw_bytes), dtype=torch_dtype)
    if shape is not None:
        expected_elements = math.prod(shape)
        if data_tensor.numel() != expected_elements:
            logger.warning(
                "  Warning: data element count does not match shape: file=%s, shape=%s",
                data_tensor.numel(),
                shape,
            )
            return None, dtype
        data_tensor = data_tensor.reshape(shape)

    return data_tensor, dtype


def read_golden_data(filename, dtype):
    """读取 golden 数据文件（仅支持 .pt 格式）"""
    golden_tensor = torch.load(filename, map_location="cpu", weights_only=True)
    return golden_tensor


_LEGACY_OPTION_NAMES = ("dtype", "verbose", "custom_rtol", "custom_atol")


def _resolve_comparison_options(options, legacy_values, legacy_options):
    if len(legacy_values) > len(_LEGACY_OPTION_NAMES):
        raise TypeError("too many positional arguments")
    merged_options = dict(legacy_options)
    for name, value in zip(_LEGACY_OPTION_NAMES, legacy_values):
        if name in merged_options:
            raise TypeError(f"multiple values for argument {name!r}")
        merged_options[name] = value
    if options is None:
        options = ComparisonOptions()
    if merged_options:
        options = replace(options, **merged_options)
    return options


def _check_tensor_layout(jit_data, golden_data, name):
    if jit_data.dtype != golden_data.dtype:
        logger.warning(f"\n{name}: ✗ FAIL")
        logger.warning(
            f"  Data type mismatch: jit={jit_data.dtype}, golden={golden_data.dtype}"
        )
        return False
    if jit_data.shape != golden_data.shape:
        logger.warning(f"\n{name}: ✗ FAIL")
        logger.warning(
            f"  Data shape mismatch: jit={jit_data.shape}, golden={golden_data.shape}"
        )
        return False
    return True


def _empty_comparison():
    return {
        "total_count": 0,
        "warn_count": 0,
        "fail_count": 0,
        "max_diff": 0.0,
        "max_val": 0.0,
        "actual_rtol": 0.0,
        "actual_atol": 0.0,
        "error_threshold": 0,
        "match": True,
    }


def _calculate_comparison(jit_data, golden_data, rtol, atol):
    total_count = jit_data.numel()
    if total_count == 0:
        return _empty_comparison()
    jit_float = jit_data.to(torch.float64)
    golden_float = golden_data.to(torch.float64)
    finite_pair = torch.isfinite(jit_float) & torch.isfinite(golden_float)
    same_inf = (
        torch.isinf(jit_float)
        & torch.isinf(golden_float)
        & (torch.signbit(jit_float) == torch.signbit(golden_float))
    )
    invalid_mask = ~(finite_pair | same_inf)
    safe_jit = torch.where(finite_pair, jit_float, 0.0)
    safe_golden = torch.where(finite_pair, golden_float, 0.0)
    diff = torch.abs(safe_jit - safe_golden)
    abs_sum = torch.abs(safe_jit) + torch.abs(safe_golden)
    tol_attn = abs_sum * rtol / 2 + atol
    tol_fail = tol_attn * 128
    zero_mask = finite_pair & (abs_sum <= 0)
    eligible_mask = finite_pair & (~zero_mask)
    zero_count = zero_mask.sum().item()
    invalid_count = invalid_mask.sum().item()
    warn_count = ((diff > tol_attn) & eligible_mask).sum().item() + invalid_count
    fail_count = ((diff > tol_fail) & eligible_mask).sum().item() + invalid_count
    max_diff = float("inf") if invalid_count else torch.max(diff).item()
    if torch.isinf(golden_float).any().item():
        max_val = float("inf")
    else:
        finite_golden = torch.where(torch.isfinite(golden_float), golden_float, 0.0)
        max_val = torch.max(torch.abs(finite_golden)).item()
    rel_diff = torch.where(eligible_mask, 2 * diff / abs_sum, 0.0)
    relative_error = float("inf") if invalid_count else torch.max(rel_diff).item()
    error_threshold = max(16, int((total_count - zero_count) ** 0.5) // 2)
    error_threshold = min(
        error_threshold, int((total_count - zero_count) * min(rtol, atol))
    )
    return {
        "total_count": total_count,
        "warn_count": warn_count,
        "fail_count": fail_count,
        "max_diff": max_diff,
        "max_val": max_val,
        "actual_rtol": relative_error,
        "actual_atol": max_diff,
        "error_threshold": error_threshold,
        "match": warn_count <= error_threshold and fail_count == 0,
    }


def _log_comparison(name, stats, options, rtol, atol):
    status = "✓ PASS" if stats["match"] else "✗ FAIL"
    logger.info(f"\n{name}: {status}")
    logger.info(f"  Max diff: {stats['max_diff']:.6f}")
    logger.info(f"  Max val: {stats['max_val']:.6f}")
    # Warn告警点(允许有error_threshold个离群值)； Fail失败点(差异过大直接失败)
    logger.info(
        "  Warn: %s/%s, Fail: %s/%s",
        stats["warn_count"],
        stats["total_count"],
        stats["fail_count"],
        stats["total_count"],
    )
    logger.info(f"  Threshold: warn<={stats['error_threshold']}, fail=0")
    logger.info(f"  Tolerance: rtol={rtol}, atol={atol} (dtype={options.dtype})")
    logger.info(
        f"  Actual: rtol={stats['actual_rtol']:.6f}, atol={stats['actual_atol']:.6f}"
    )


def _log_mismatched_elements(jit_data, golden_data, total_count):
    logger.info("  First 10 elements comparison:")
    jit_flat = jit_data.flatten()
    golden_flat = golden_data.flatten()
    for index in range(min(10, total_count)):
        jit_val = jit_flat[index].item()
        golden_val = golden_flat[index].item()
        logger.info(
            "    [%d] jit=%.6f, golden=%.6f, diff=%.6f",
            index,
            jit_val,
            golden_val,
            abs(jit_val - golden_val),
        )


def compare_with_golden(
    jit_data, golden_data, name, *legacy_values, options=None, **legacy_options
):
    """对比 jit 结果与 golden 结果；旧位置参数和关键字参数均保持兼容。"""
    if jit_data is None or golden_data is None:
        logger.warning("\n%s: ✗ FAIL", name)
        logger.warning("  Failed to read the data to be compared")
        return False
    if not _check_tensor_layout(jit_data, golden_data, name):
        return False
    options = _resolve_comparison_options(options, legacy_values, legacy_options)
    if options.dtype is None:
        rtol, atol = 1e-3, 1e-3
    else:
        rtol, atol = get_tolerance_by_dtype(
            options.dtype, options.custom_rtol, options.custom_atol
        )
    if options.verbose:
        logger.info(f"  Data type: {jit_data.dtype}")
        logger.info(f"  Data shape: {jit_data.shape}")
    stats = _calculate_comparison(jit_data, golden_data, rtol, atol)
    _log_comparison(name, stats, options, rtol, atol)
    if options.verbose and not stats["match"]:
        _log_mismatched_elements(jit_data, golden_data, stats["total_count"])
    return stats["match"]


def analyze_results(results):
    """Analyze comparison results and provide bisection suggestions"""
    logger.info("\n" + "=" * 80)
    logger.info("Step 5: continue bisection based on results")
    logger.info("=" * 80)

    first_fail_idx = -1
    for idx, (name, match) in enumerate(results):
        if not match:
            first_fail_idx = idx
            break

    if first_fail_idx == -1:
        logger.info("✓ All checkpoints match")
        logger.info("→ The problem may be in: operations after the checkpoint")
        return

    fail_name = results[first_fail_idx][0]

    if first_fail_idx == 0:
        logger.info(f"✗ The first checkpoint ({fail_name}) already mismatches")
        logger.info("→ The problem may be in: input data or the first computation step")
        logger.info(
            "→ Suggestion: check whether the input data is correct, or insert checkpoints earlier"
        )
    else:
        prev_name = results[first_fail_idx - 1][0]
        logger.info(f"✗ Checkpoint {prev_name} matches, but {fail_name} does not")
        logger.info(
            f"→ Problem location: operations between {prev_name} and {fail_name}"
        )
        logger.info(
            "→ Suggestion: insert new checkpoints between these two checkpoints to further locate the problem"
        )


def setup_logging(work_dir, golden_dir):
    """Configure log output"""
    log_formatter = logging.Formatter("%(message)s")
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(log_formatter)

    operator_dir = (
        os.path.dirname(os.path.abspath(golden_dir))
        if golden_dir
        else os.path.abspath(work_dir)
    )
    operator_name = os.path.basename(operator_dir)
    log_file = os.path.join(operator_dir, f"{operator_name}_verify_result.log")

    file_handler = logging.FileHandler(log_file, mode="w", encoding="utf-8")
    file_handler.setFormatter(log_formatter)

    logger.handlers = [console_handler, file_handler]
    logger.propagate = False
    return operator_name


def _parse_args():
    parser = argparse.ArgumentParser(description="PyPTO 精度检查点文件对比工具")
    parser.add_argument(
        "--work-dir", "-w", default=".", help="工作目录（默认为当前目录）"
    )
    parser.add_argument(
        "--output-dir",
        "-o",
        default=None,
        help="指定 output 目录名（不指定则自动检测最新的）",
    )
    parser.add_argument(
        "--golden-dir",
        "-g",
        default=None,
        help="指定 golden 文件所在目录（默认从工作目录查找）",
    )
    parser.add_argument(
        "--verbose", "-v", action="store_true", help="显示详细的元素级对比"
    )
    parser.add_argument(
        "--list", "-l", action="store_true", help="只列出检查点，不进行对比"
    )
    parser.add_argument(
        "--rtol",
        type=float,
        default=None,
        help="自定义相对容差（用于量化场景，推荐值：0.0078125）",
    )
    parser.add_argument(
        "--atol",
        type=float,
        default=None,
        help="自定义绝对容差（用于量化场景，推荐值：0.0001）",
    )
    return parser.parse_args()


def _find_golden_file(golden_base_dir, checkpoint_name):
    for extension in ("pt", "bin"):
        pattern = os.path.join(golden_base_dir, f"golden_{checkpoint_name}.{extension}")
        matches = glob.glob(pattern)
        if matches:
            return matches[0], pattern
    return None, pattern


def _compare_checkpoint(tensor_dir, golden_base_dir, checkpoint_name, options):
    jit_files = sorted(glob.glob(os.path.join(tensor_dir, f"{checkpoint_name}_*.data")))
    if not jit_files:
        logger.warning(f"\n{checkpoint_name}: ✗ jit file not found")
        return False
    golden_file, golden_pattern = _find_golden_file(golden_base_dir, checkpoint_name)
    if golden_file is None:
        logger.warning(
            f"\n{checkpoint_name}: ✗ golden file not found ({golden_pattern})"
        )
        return False

    jit_file = jit_files[0]
    logger.info("\nFiles used:")
    logger.info(f"  jit: {os.path.basename(jit_file)}")
    logger.info(f"  golden: {os.path.basename(golden_file)}")
    jit_data, dtype = read_jit_data(jit_file)
    golden_data = read_golden_data(golden_file, dtype)
    return compare_with_golden(
        jit_data,
        golden_data,
        checkpoint_name,
        options=replace(options, dtype=dtype),
    )


def _show_checkpoints(checkpoints):
    logger.info("\nCheckpoint list:")
    for index, checkpoint in enumerate(checkpoints, 1):
        logger.info(f"  {index}. {checkpoint}")


def run(args):
    latest_dir = args.output_dir or find_latest_output_dir(args.work_dir)
    if not latest_dir:
        logging.error("✗ output directory not found")
        return 1
    setup_logging(args.work_dir, args.golden_dir)
    logger.info("=" * 80)
    logger.info("PyPTO precision checkpoint file comparison tool")
    logger.info("=" * 80)
    golden_base_dir = args.golden_dir if args.golden_dir else args.work_dir
    tensor_dir = os.path.join(args.work_dir, "output", latest_dir, "tensor")
    checkpoints = scan_checkpoints_from_dir(tensor_dir)
    if not checkpoints:
        logger.error(f"✗ No checkpoint files found in {tensor_dir}")
        return 1
    logger.info(f"✓ Found output directory: {latest_dir}")
    logger.info(f"✓ Found {len(checkpoints)} checkpoints: {checkpoints}")
    if args.list:
        _show_checkpoints(checkpoints)
        return 0
    logger.info("\n" + "=" * 80)
    logger.info("Step 4: compare jit and golden data")
    logger.info("=" * 80)
    options = ComparisonOptions(
        verbose=args.verbose,
        custom_rtol=args.rtol,
        custom_atol=args.atol,
    )
    results = [
        (name, _compare_checkpoint(tensor_dir, golden_base_dir, name, options))
        for name in checkpoints
    ]
    analyze_results(results)
    return 0 if all(match for _, match in results) else 1


def main():
    sys.exit(run(_parse_args()))


if __name__ == "__main__":
    main()
