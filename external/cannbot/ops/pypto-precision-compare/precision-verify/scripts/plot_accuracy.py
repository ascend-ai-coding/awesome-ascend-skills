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
PyPTO 精度变化折线图绘制工具
从验证结果日志文件中提取精度数据并绘制变化趋势图
"""
import re
import sys
import os
import logging
import matplotlib.pyplot as plt
import numpy as np


logging.basicConfig(
    level=logging.INFO,
    format='%(message)s',
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger(__name__)


def extract_operator_name(log_file_path):
    """从日志文件路径中提取算子名称"""
    log_file_name = os.path.basename(log_file_path)
    if log_file_name.endswith('_verify_result.log'):
        return log_file_name[:-len('_verify_result.log')]
    return 'operator'


def parse_checkpoints(lines):
    """从日志行中提取检查点名称和对应的精度数据"""
    checkpoint_pattern = r'^(\d+_[^:\s]+):'
    checkpoints = []
    checkpoint_line_indices = []

    for i, line in enumerate(lines):
        match = re.match(checkpoint_pattern, line)
        if match:
            checkpoint_name = match.group(1)
            checkpoints.append(checkpoint_name)
            checkpoint_line_indices.append(i)

    results = []
    for i, checkpoint in enumerate(checkpoints):
        start_line = checkpoint_line_indices[i]
        tolerance_rtol = None
        tolerance_atol = None
        actual_rtol = None
        actual_atol = None

        for j in range(start_line, min(start_line + 20, len(lines))):
            if 'Tolerance: rtol=' in lines[j]:
                num_pattern = r'([-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)'
                tolerance_pattern = f'Tolerance: rtol={num_pattern}, atol={num_pattern}'
                tolerance_match = re.search(tolerance_pattern, lines[j])
                if tolerance_match:
                    tolerance_rtol = float(tolerance_match.group(1))
                    tolerance_atol = float(tolerance_match.group(2))
            elif 'Actual: rtol=' in lines[j]:
                num_pattern = r'([-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)'
                actual_pattern = f'Actual: rtol={num_pattern}, atol={num_pattern}'
                actual_match = re.search(actual_pattern, lines[j])
                if actual_match:
                    actual_rtol = float(actual_match.group(1))
                    actual_atol = float(actual_match.group(2))
                    break

        if actual_rtol is not None and actual_atol is not None:
            results.append((checkpoint, tolerance_rtol, tolerance_atol, actual_rtol, actual_atol))

    return results


def plot_accuracy(results, output_path):
    """绘制精度变化折线图"""
    checkpoints_list = [r[0] for r in results]
    tol_rtol_values = [r[1] for r in results]
    tol_atol_values = [r[2] for r in results]
    act_rtol_values = [r[3] for r in results]
    act_atol_values = [r[4] for r in results]

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10))

    ax1.plot(range(len(checkpoints_list)), act_rtol_values, marker='o', linewidth=2, markersize=8,
             color='blue', label='Actual rtol')
    ax1.plot(range(len(checkpoints_list)), tol_rtol_values, marker='s', linewidth=2, markersize=6,
             color='green', linestyle='--', label='Tolerance rtol')
    ax1.set_xlabel('Checkpoint', fontsize=14)
    ax1.set_ylabel('Relative Tolerance (rtol)', fontsize=14)
    ax1.set_title('Relative Tolerance (rtol) Change Across Checkpoints', fontsize=16, fontweight='bold', y=1.02)
    ax1.grid(True, alpha=0.3)
    ax1.set_xticks(range(len(checkpoints_list)))
    ax1.set_xticklabels(checkpoints_list, rotation=45, ha='right', fontsize=11)
    ax1.legend(loc='upper right', fontsize=11)

    for i, (ckpt, rtol) in enumerate(zip(checkpoints_list, act_rtol_values)):
        if rtol > 0:
            ax1.annotate(f'{rtol:.4f}', (i, rtol), textcoords="offset points",
                       xytext=(0, 10), ha='center', fontsize=8, color='blue', fontweight='bold')

    ax2.plot(range(len(checkpoints_list)), act_atol_values, marker='o', linewidth=2, markersize=8,
             color='red', label='Actual atol')
    ax2.plot(range(len(checkpoints_list)), tol_atol_values, marker='s', linewidth=2, markersize=6,
             color='green', linestyle='--', label='Tolerance atol')
    ax2.set_xlabel('Checkpoint', fontsize=14)
    ax2.set_ylabel('Absolute Tolerance (atol)', fontsize=14)
    ax2.set_title('Absolute Tolerance (atol) Change Across Checkpoints', fontsize=16, fontweight='bold', y=1.02)
    ax2.grid(True, alpha=0.3)
    ax2.set_xticks(range(len(checkpoints_list)))
    ax2.set_xticklabels(checkpoints_list, rotation=45, ha='right', fontsize=11)
    ax2.legend(loc='upper right', fontsize=11)

    for i, (ckpt, atol) in enumerate(zip(checkpoints_list, act_atol_values)):
        if atol > 0:
            ax2.annotate(f'{atol:.2f}', (i, atol), textcoords="offset points",
                       xytext=(0, 10), ha='center', fontsize=8, color='red', fontweight='bold')

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    return output_path


def print_summary(results):
    """打印精度数据汇总"""
    logger.info("\n精度数据汇总:")
    logger.info("-" * 100)
    logger.info(f"{'Checkpoint':<30} {'Tolerance':<25} {'Actual':<25} {'Status':<10}")
    logger.info("-" * 100)
    for ckpt, tol_rtol, tol_atol, act_rtol, act_atol in results:
        status = "FAIL" if (act_rtol > tol_rtol or act_atol > tol_atol) else "PASS"
        tol_str = f"rt={tol_rtol:.6f}, at={tol_atol:.6f}"
        act_str = f"rt={act_rtol:.6f}, at={act_atol:.6f}"
        logger.info(f"{ckpt:<30} {tol_str:<25} {act_str:<25} {status:<10}")
    logger.info("-" * 100)


def main():
    """主函数"""
    if len(sys.argv) < 2:
        logger.error("错误: 必须传入 log 文件路径作为参数")
        logger.error("用法: python3 plot_accuracy.py <verify_result.log>")
        sys.exit(1)

    log_file = sys.argv[1]
    log_file_path = os.path.abspath(log_file)
    log_file_dir = os.path.dirname(log_file_path)

    operator_name = extract_operator_name(log_file_path)

    with open(log_file, 'r') as f:
        lines = f.readlines()

    results = parse_checkpoints(lines)

    logger.info(f'提取到 {len(results)} 个检查点的精度数据')
    for i, (ckpt, tol_rtol, tol_atol, act_rtol, act_atol) in enumerate(results):
        tol_str = f'Tolerance(rt={tol_rtol:.6f}, at={tol_atol:.6f})'
        act_str = f'Actual(rt={act_rtol:.6f}, at={act_atol:.6f})'
        logger.info(f'{i+1}. {ckpt}: {tol_str} | {act_str}')

    output_image = os.path.join(log_file_dir, f'{operator_name}_accuracy_change.png')
    plot_accuracy(results, output_image)
    logger.info(f"\n精度变化折线图已保存至: {output_image}")

    print_summary(results)


if __name__ == "__main__":
    main()
