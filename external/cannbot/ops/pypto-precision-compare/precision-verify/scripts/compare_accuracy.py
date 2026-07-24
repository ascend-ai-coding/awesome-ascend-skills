#!/usr/bin/env python3
# coding: utf-8
# ----------------------------------------------------------------------------------------------------------
# Copyright (c) 2025-2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ----------------------------------------------------------------------------------------------------------
"""
PyPTO 精度检查点文件对比工具
自动检测检查点并对比 jit 和 golden 的中间结果
"""
import os
import sys
import glob
import argparse
import logging
import re
import torch

# PyPTO 数据类型映射
# 参考：tools/verifier/parse_dump_tensors.py 和 docs/zh/api/datatype/DataType.md
DTYPE_MAP = {
    0: ('int4', torch.int8, 1),            # DT_INT4: 4位有符号整数，2个元素打包成1个int8存储
    1: ('int8', torch.int8, 1),            # DT_INT8: 8位有符号整数
    2: ('int16', torch.int16, 2),          # DT_INT16: 16位有符号整数
    3: ('int32', torch.int32, 4),          # DT_INT32: 32位有符号整数
    4: ('int64', torch.int64, 8),          # DT_INT64: 64位有符号整数
    5: ('fp8', torch.float8_e4m3fn, 1),    # DT_FP8: 8位浮点数（通用）
    6: ('fp16', torch.float16, 2),         # DT_FP16: 16位半精度浮点数
    7: ('fp32', torch.float32, 4),         # DT_FP32: 32位单精度浮点数
    8: ('bf16', torch.bfloat16, 2),        # DT_BF16: 16位Brain Float格式
    9: ('hf4', torch.uint8, 1),            # DT_HF4: 4位Half Float格式，2个元素打包成1个uint8存储
    10: ('hf8', torch.uint8, 1),           # DT_HF8: 8位Half Float格式，原始字节存储
    11: ('uint8', torch.uint8, 1),         # DT_UINT8: 8位无符号整数
    12: ('uint16', torch.uint16, 2),       # DT_UINT16: 16位无符号整数
    13: ('uint32', torch.uint32, 4),       # DT_UINT32: 32位无符号整数
    14: ('uint64', torch.uint64, 8),       # DT_UINT64: 64位无符号整数
    15: ('bool', torch.bool, 1),           # DT_BOOL: 布尔类型
    16: ('double', torch.float64, 8),      # DT_DOUBLE: 64位双精度浮点数
    17: ('fp8_e4m3', torch.float8_e4m3fn, 1),  # DT_FP8E4M3: 8位浮点数，4位指数，3位尾数
    18: ('fp8_e5m2', torch.float8_e5m2, 1),    # DT_FP8E5M2: 8位浮点数，5位指数，2位尾数
    19: ('fp8_e8m0', torch.uint8, 1),          # DT_FP8E8M0: 8位浮点数，8位指数，0位尾数（scale专用）
    20: ('fp4_e2m1x2', torch.uint8, 1),        # DT_FP4_E2M1X2: 4位浮点数，2位指数，1位尾数，x2打包
    21: ('fp4_e1m2x2', torch.uint8, 1),        # DT_FP4_E1M2X2: 4位浮点数，1位指数，2位尾数，x2打包
}

# 容差标准映射表（基于 PyPTO 代码库实践）
# 参考：verify.md、examples、models 中的实际使用标准
TOLERANCE_MAP = {
    0: (0, 1e-4),      # INT4:  整数类型，允许极少误差点
    1: (0, 1e-4),      # INT8:  整数类型，允许极少误差点
    2: (0, 1e-4),      # INT16: 整数类型，允许极少误差点
    3: (0, 1e-4),      # INT32: 整数类型，允许极少误差点
    4: (0, 1e-4),      # INT64: 整数类型，允许极少误差点
    5: (1e-1, 1e-2),   # FP8:   8位浮点，量化场景精度较低
    6: (1e-3, 1e-3),   # FP16: 半精度浮点，mantissa=10bits
    7: (1e-3, 1e-4),   # FP32: 单精度浮点，高精度基准
    8: (5e-3, 5e-2),   # BF16: BFloat16，mantissa=7bits，广泛使用于attention
    9: (1e-1, 1e-2),   # HF4:  4位Half Float，低精度量化
    10: (1e-1, 1e-2),  # HF8:  8位Half Float，低精度量化
    11: (0, 1e-4),     # UINT8:  无符号整数，允许极少误差点
    12: (0, 1e-4),     # UINT16: 无符号整数，允许极少误差点
    13: (0, 1e-4),     # UINT32: 无符号整数，允许极少误差点
    14: (0, 1e-4),     # UINT64: 无符号整数，允许极少误差点
    15: (0, 1e-4),     # BOOL:   布尔类型，允许极少误差点
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
    format='%(message)s',
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger(__name__)


def find_latest_output_dir(work_dir="."):
    """找到最新的 output 目录"""
    output_base = os.path.join(work_dir, "output")
    if not os.path.exists(output_base):
        return None

    output_dirs = [d for d in os.listdir(output_base) if d.startswith("output_")]
    if not output_dirs:
        return None

    output_dirs.sort(key=lambda x: os.path.getmtime(os.path.join(output_base, x)), reverse=True)
    return output_dirs[0]


def scan_checkpoints_from_dir(tensor_dir):
    """从指定目录扫描所有检查点文件"""
    if not os.path.exists(tensor_dir):
        return []

    data_files = glob.glob(os.path.join(tensor_dir, "*.data"))
    checkpoints = set()

    for data_file in data_files:
        basename = os.path.basename(data_file)
        parts = basename.rsplit('_', 1)
        if len(parts) == 2 and parts[1].replace('.data', '').isdigit():
            checkpoint_name = parts[0]
            checkpoints.add(checkpoint_name)

    def get_sort_key(name):
        match = re.match(r'^(\d+)', name)
        return int(match.group(1)) if match else float('inf')

    return sorted(list(checkpoints), key=get_sort_key)


def get_tolerance_by_dtype(dtype, custom_rtol=None, custom_atol=None):
    """根据数据类型返回对应的容差标准"""
    if custom_rtol is not None and custom_atol is not None:
        return custom_rtol, custom_atol
    return TOLERANCE_MAP.get(dtype, (1e-3, 1e-3))


def read_jit_data(filename):
    """读取 jit 生成的数据文件，自动检测数据类型"""
    csv_file = filename.replace('.data', '.csv')

    dtype = None
    shape = None
    if os.path.exists(csv_file):
        with open(csv_file, 'r') as f:
            for line in f:
                if line.startswith('dtype,'):
                    dtype = int(line.split(',')[1].strip())
                elif line.startswith('shape,'):
                    shape_str = line.split(',')[1].strip()
                    shape = tuple(map(int, shape_str.split('x')))

    if dtype is None:
        logger.warning(f"  警告: 未找到 CSV 文件，无法确定数据类型")
        return None, None

    if dtype not in DTYPE_MAP:
        logger.warning(f"  警告: 未知的数据类型 {dtype}")
        return None, None

    type_name, torch_dtype, bytes_per_element = DTYPE_MAP[dtype]

    with open(filename, "rb") as f:
        raw_bytes = f.read()
    data_tensor = torch.frombuffer(raw_bytes, dtype=torch_dtype)
    
    if shape is not None:
        data_tensor = data_tensor.reshape(shape)
    
    return data_tensor, dtype


def read_golden_data(filename, dtype):
    """读取 golden 数据文件（仅支持 .pt 格式）"""
    golden_tensor = torch.load(filename, map_location='cpu')
    return golden_tensor


def compare_with_golden(jit_data, golden_data, name, dtype=None, verbose=True, custom_rtol=None, custom_atol=None):
    """对比 jit 结果与 golden 结果"""
    if jit_data.dtype != golden_data.dtype:
        logger.info(f"\n{name}: ✗ FAIL")
        logger.info(f"  数据类型不一致: jit={jit_data.dtype}, golden={golden_data.dtype}")
        return False

    if jit_data.shape != golden_data.shape:
        logger.info(f"\n{name}: ✗ FAIL")
        logger.info(f"  数据形状不一致: jit={jit_data.shape}, golden={golden_data.shape}")
        return False

    if dtype is not None:
        rtol, atol = get_tolerance_by_dtype(dtype, custom_rtol, custom_atol)
    else:
        rtol, atol = 1e-3, 1e-3

    if verbose:
        logger.info(f"  数据类型: {jit_data.dtype}")
        logger.info(f"  数据形状: {jit_data.shape}")

    total_count = jit_data.numel()

    diff = torch.abs(jit_data - golden_data)
    abs_sum = torch.abs(jit_data) + torch.abs(golden_data)

    tol_attn = abs_sum * rtol / 2 + atol
    tol_fail = tol_attn * 128

    zero_mask = abs_sum <= 0
    zero_count = zero_mask.sum().item()

    warn_mask = (diff > tol_attn) & (~zero_mask)
    warn_count = warn_mask.sum().item()

    fail_mask = (diff > tol_fail) & (~zero_mask)
    fail_count = fail_mask.sum().item()

    max_diff = torch.max(diff).item()
    max_val = torch.max(torch.abs(golden_data)).item()
    rel_diff = torch.where(abs_sum == 0, 0.0, 2 * diff / abs_sum)
    relative_error = torch.max(rel_diff).item()

    actual_rtol = relative_error
    actual_atol = max_diff

    error_threshold = max(16, int((total_count - zero_count) ** 0.5) // 2)
    error_threshold = min(error_threshold, int((total_count - zero_count) * min(rtol, atol)))
    match = warn_count <= error_threshold and fail_count == 0

    status = "✓ PASS" if match else "✗ FAIL"
    logger.info(f"\n{name}: {status}")
    logger.info(f"  Max diff: {max_diff:.6f}")
    logger.info(f"  Max val: {max_val:.6f}")
    # Warn告警点(允许有error_threshold个离群值)； Fail失败点(差异过大直接失败)
    logger.info(f"  Warn: {warn_count}/{total_count}, Fail: {fail_count}/{total_count}")
    logger.info(f"  Threshold: warn<={error_threshold}, fail=0")
    logger.info(f"  Tolerance: rtol={rtol}, atol={atol} (dtype={dtype})")
    logger.info(f"  Actual: rtol={actual_rtol:.6f}, atol={actual_atol:.6f}")

    if verbose and not match:
        logger.info(f"  前10个元素对比:")
        jit_flat = jit_data.flatten()
        golden_flat = golden_data.flatten()
        for i in range(min(10, total_count)):
            jit_val = jit_flat[i].item()
            golden_val = golden_flat[i].item()
            diff_val = abs(jit_val - golden_val)
            logger.info(f"    [{i}] jit={jit_val:.6f}, golden={golden_val:.6f}, diff={diff_val:.6f}")

    return match


def analyze_results(results):
    """分析对比结果，给出二分建议"""
    logger.info("\n" + "=" * 80)
    logger.info("步骤 5：根据结果继续二分")
    logger.info("=" * 80)

    first_fail_idx = -1
    for idx, (name, match) in enumerate(results):
        if not match:
            first_fail_idx = idx
            break

    if first_fail_idx == -1:
        logger.info("✓ 所有检查点都匹配")
        logger.info("→ 问题可能在：检查点之后的操作")
        return

    fail_name = results[first_fail_idx][0]

    if first_fail_idx == 0:
        logger.info(f"✗ 第一个检查点 ({fail_name}) 就不匹配")
        logger.info(f"→ 问题可能在：输入数据或第一个计算步骤")
        logger.info(f"→ 建议：检查输入数据是否正确，或在更早的位置插入检查点")
    else:
        prev_name = results[first_fail_idx - 1][0]
        logger.info(f"✗ 检查点 {prev_name} 匹配，但 {fail_name} 不匹配")
        logger.info(f"→ 问题位置：{prev_name} 和 {fail_name} 之间的操作")
        logger.info(f"→ 建议：在这两个检查点之间插入新的检查点，进一步定位问题")


def setup_logging(work_dir, golden_dir):
    """配置日志输出"""
    log_formatter = logging.Formatter('%(message)s')
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(log_formatter)

    operator_dir = os.path.dirname(os.path.abspath(golden_dir)) if golden_dir else os.path.abspath(work_dir)
    operator_name = os.path.basename(operator_dir)
    log_file = os.path.join(operator_dir, f'{operator_name}_verify_result.log')

    file_handler = logging.FileHandler(log_file, mode='w', encoding='utf-8')
    file_handler.setFormatter(log_formatter)

    logger.handlers = [console_handler, file_handler]
    logger.propagate = False
    return operator_name


def main():
    parser = argparse.ArgumentParser(description='PyPTO 精度检查点文件对比工具')
    parser.add_argument('--work-dir', '-w', default='.', help='工作目录（默认为当前目录）')
    parser.add_argument('--output-dir', '-o', default=None, help='指定 output 目录名（不指定则自动检测最新的）')
    parser.add_argument('--golden-dir', '-g', default=None, help='指定 golden 文件所在目录（默认从工作目录查找）')
    parser.add_argument('--verbose', '-v', action='store_true', help='显示详细的元素级对比')
    parser.add_argument('--list', '-l', action='store_true', help='只列出检查点，不进行对比')
    parser.add_argument('--rtol', type=float, default=None, help='自定义相对容差（用于量化场景，推荐值：0.0078125）')
    parser.add_argument('--atol', type=float, default=None, help='自定义绝对容差（用于量化场景，推荐值：0.0001）')

    args = parser.parse_args()

    latest_dir = args.output_dir if args.output_dir else find_latest_output_dir(args.work_dir)

    if not latest_dir:
        logging.error("✗ 未找到 output 目录")
        sys.exit(1)

    operator_name = setup_logging(args.work_dir, args.golden_dir)

    logger.info("=" * 80)
    logger.info("PyPTO 精度检查点文件对比工具")
    logger.info("=" * 80)

    golden_base_dir = args.golden_dir if args.golden_dir else args.work_dir
    tensor_dir = os.path.join(args.work_dir, "output", latest_dir, "tensor")
    checkpoints = scan_checkpoints_from_dir(tensor_dir)

    if not checkpoints:
        logger.error(f"✗ 未在 {tensor_dir} 找到检查点文件")
        sys.exit(1)

    logger.info(f"✓ 找到 output 目录: {latest_dir}")
    logger.info(f"✓ 找到 {len(checkpoints)} 个检查点: {checkpoints}")

    if args.list:
        logger.info("\n检查点列表:")
        for idx, ckpt in enumerate(checkpoints, 1):
            logger.info(f"  {idx}. {ckpt}")
        sys.exit(0)

    logger.info("\n" + "=" * 80)
    logger.info("步骤 4：对比 jit 和 golden 数据")
    logger.info("=" * 80)

    results = []

    for checkpoint_name in checkpoints:
        jit_pattern = os.path.join(tensor_dir, f"{checkpoint_name}_*.data")
        jit_files = sorted(glob.glob(jit_pattern))

        if not jit_files:
            logger.warning(f"\n{checkpoint_name}: ✗ 未找到 jit 文件")
            results.append((checkpoint_name, False))
            continue

        golden_pattern = os.path.join(golden_base_dir, f"golden_{checkpoint_name}.pt")
        golden_files = glob.glob(golden_pattern)
        
        if not golden_files:
            golden_pattern = os.path.join(golden_base_dir, f"golden_{checkpoint_name}.bin")
            golden_files = glob.glob(golden_pattern)

        if not golden_files:
            logger.warning(f"\n{checkpoint_name}: ✗ 未找到 golden 文件 ({golden_pattern})")
            results.append((checkpoint_name, False))
            continue

        jit_file = jit_files[0]
        golden_file = golden_files[0]

        logger.info(f"\n使用文件:")
        logger.info(f"  jit: {os.path.basename(jit_file)}")
        logger.info(f"  golden: {os.path.basename(golden_file)}")

        jit_data, dtype = read_jit_data(jit_file)
        golden_data = read_golden_data(golden_file, dtype)

        match = compare_with_golden(jit_data, golden_data, checkpoint_name,
                                   dtype=dtype, verbose=args.verbose,
                                   custom_rtol=args.rtol, custom_atol=args.atol)
        results.append((checkpoint_name, match))

    analyze_results(results)

    all_match = all(match for _, match in results)
    sys.exit(0 if all_match else 1)


if __name__ == "__main__":
    main()
