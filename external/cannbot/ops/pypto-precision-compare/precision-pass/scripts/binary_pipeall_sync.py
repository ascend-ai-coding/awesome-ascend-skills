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
CCE 文件二分定位工具（按文件顺序 + 代码行级二分）。

核心逻辑：
1. 文件级二分：定位问题 CCE 文件
2. 行级二分：在问题文件中定位具体缺少同步的代码行

用法：
  # 查看所有 CCE 文件
  python3 binary_pipeall_sync.py --kernel-dir kernel_aicore --info
  
  # 文件级二分 + 行级二分（完整流程）
  python3 binary_pipeall_sync.py \
      --kernel-dir kernel_aicore \
      --test-cmd "python3 test.py" \
      --run-dir .
  
  # 仅行级二分（已知问题文件）
  python3 binary_pipeall_sync.py \
      --kernel-dir kernel_aicore \
      --cce-file kernel_aicore/problem.cpp \
      --test-cmd "python3 test.py" \
      --run-dir .
"""

import os
import re
import sys
import shlex
import shutil
import subprocess
import logging
import argparse
from pathlib import Path
from typing import List, Tuple, Optional

logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger(__name__)


def get_all_cce_files(kernel_dir: str) -> List[str]:
    """
    获取所有 CCE 文件（_aiv.cpp 和 _aic.cpp），按文件名排序。
    """
    cce_files = []
    kernel_path = Path(kernel_dir)
    
    for pattern in ["*_aiv.cpp", "*_aic.cpp"]:
        for f in kernel_path.glob(pattern):
            cce_files.append(str(f))
    
    cce_files.sort()
    return cce_files


def read_file(path: str) -> List[str]:
    with open(path, 'r', encoding='utf-8') as f:
        return f.readlines()


def write_file(path: str, lines: List[str]):
    with open(path, 'w', encoding='utf-8') as f:
        f.writelines(lines)


def insert_pipe_all_for_file(cce_path: str) -> Tuple[List[str], int]:
    """
    在单个 CCE 文件中插入 pipe_barrier(PIPE_ALL)。
    
    参考 enableDebug_{true} 的逻辑：在每个重要操作后插入。
    
    返回：(修改后的内容, 插入数量)
    """
    lines = read_file(cce_path)
    result = lines.copy()
    inserted_count = 0
    
    patterns_insert = [
        r'SUBKERNEL_PHASE',
        r'^\s*T[A-Z]',
    ]
    
    patterns_exclude = [
        r'^\s*//',
        r'^\s*$',
        r'^\s*extern',
        r'^\s*UBTileTensor',
        r'^\s*GMTileTensor',
        r'^\s*}',
    ]
    
    insert_positions = []
    for i, line in enumerate(lines):
        if any(re.search(p, line) for p in patterns_exclude):
            continue
        
        should_insert = any(re.search(p, line) for p in patterns_insert)
        if should_insert:
            if i + 1 < len(result) and 'pipe_barrier(PIPE_ALL)' in result[i + 1]:
                continue
            insert_positions.append(i)
    
    for pos in reversed(insert_positions):
        ref_line = lines[pos]
        indent_match = re.match(r'^\s*', ref_line)
        indent = indent_match.group() if indent_match else ''
        result.insert(pos + 1, f'{indent}pipe_barrier(PIPE_ALL);\n')
        inserted_count += 1
    
    return result, inserted_count


def insert_pipe_all_to_files(cce_files: List[str]) -> dict:
    """
    在多个 CCE 文件中插入 PIPE_ALL。
    
    返回：{cce_path: backup_path}
    """
    backups = {}
    
    for cce_path in cce_files:
        bak = cce_path + ".bak"
        shutil.copy(cce_path, bak)
        backups[cce_path] = bak
        
        modified, count = insert_pipe_all_for_file(cce_path)
        write_file(cce_path, modified)
        
        logger.info(f"  {os.path.basename(cce_path)}: {count} PIPE_ALL inserted")
    
    return backups


def restore_cce_files(backups: dict):
    """
    恢复所有 CCE 文件到原始状态。
    """
    for cce_path, bak in backups.items():
        shutil.copy(bak, cce_path)
        os.remove(bak)


def get_insertable_positions(cce_path: str) -> List[int]:
    """
    获取 CCE 文件中所有可插入 PIPE_ALL 的行索引。
    
    Returns:
        可插入位置的行索引列表（0-based）
    """
    lines = read_file(cce_path)
    positions = []
    
    patterns_insert = [
        r'SUBKERNEL_PHASE',
        r'^\s*T[A-Z]',
    ]
    
    patterns_exclude = [
        r'^\s*//',
        r'^\s*$',
        r'^\s*extern',
        r'^\s*UBTileTensor',
        r'^\s*GMTileTensor',
        r'^\s*}',
    ]
    
    for i, line in enumerate(lines):
        if any(re.search(p, line) for p in patterns_exclude):
            continue
        
        if not any(re.search(p, line) for p in patterns_insert):
            continue
        
        # 检查下一行是否已有 PIPE_ALL
        if i + 1 < len(lines) and 'pipe_barrier(PIPE_ALL)' in lines[i + 1]:
            continue
        
        positions.append(i)
    
    return positions


def insert_pipe_all_to_positions(cce_path: str, positions: List[int]) -> int:
    """
    在指定位置插入 PIPE_ALL。
    
    Args:
        cce_path: CCE 文件路径
        positions: 要插入的行索引列表（0-based，在这些行之后插入）
    
    Returns:
        实际插入数量
    """
    lines = read_file(cce_path)
    result = lines.copy()
    
    # 从后向前插入，避免索引变化
    for pos in reversed(positions):
        if pos >= len(lines):
            continue
        ref_line = lines[pos]
        indent_match = re.match(r'^\s*', ref_line)
        indent = indent_match.group() if indent_match else ''
        result.insert(pos + 1, f'{indent}pipe_barrier(PIPE_ALL);\n')
    
    write_file(cce_path, result)
    return len(positions)


def run_line_bisection(cce_path: str, kernel_dir: str, test_cmd: str, run_dir: str) -> Optional[int]:
    """
    在单个 CCE 文件内执行行级二分，定位缺少同步的具体代码行。
    
    Args:
        cce_path: 问题 CCE 文件路径
        kernel_dir: kernel_aicore 目录
        test_cmd: 测试命令
        run_dir: 测试运行目录
    
    Returns:
        问题代码行号（1-based），如果未定位则返回 None
    """
    all_positions = get_insertable_positions(cce_path)
    total_ops = len(all_positions)
    
    if total_ops == 0:
        logger.error(f"文件中无可插入操作: {cce_path}")
        return None
    
    lines = read_file(cce_path)
    
    logger.info(f"\n{'='*60}")
    logger.info(f"CCE Line Binary Search")
    logger.info(f"{'='*60}")
    logger.info(f"File: {os.path.basename(cce_path)}")
    logger.info(f"Total operations: {total_ops}")
    logger.info("")
    
    left = 0
    right = total_ops - 1
    round_num = 1
    
    bak = cce_path + ".linebak"
    shutil.copy(cce_path, bak)
    
    try:
        while left < right:
            mid = (left + right) // 2
            
            logger.info(f"\n{'='*60}")
            logger.info(f"Round {round_num}: left={left}, right={right}, mid={mid}")
            logger.info(f"{'='*60}")
            logger.info(f"测试操作 [{left}, {mid}] ({mid - left + 1} ops)")
            logger.info("")
            
            # 恢复原始文件
            shutil.copy(bak, cce_path)
            
            # 只在前半段位置插入
            test_positions = all_positions[left:mid + 1]
            insert_pipe_all_to_positions(cce_path, test_positions)
            
            if not compile_kernel(kernel_dir):
                logger.error("编译失败")
                return None
            
            passed = test_precision(test_cmd, run_dir)
            
            logger.info("")
            if passed:
                logger.info(f"结果: PASS → 问题在已插入部分 [{left}, {mid}]")
                logger.info(f"      继续在已插入部分二分")
                right = mid
            else:
                logger.info(f"结果: FAIL → 问题在未插入部分 [{mid+1}, {right}]")
                logger.info(f"      继续在未插入部分二分")
                left = mid + 1
            
            round_num += 1
        
        # 定位到具体行
        problem_pos = all_positions[left]
        problem_line = problem_pos + 1  # 转为 1-based
        problem_code = lines[problem_pos].strip()
        
        logger.info(f"\n{'='*60}")
        logger.info(f"LINE BINARY SEARCH COMPLETED!")
        logger.info(f"{'='*60}")
        logger.info(f"Found: operation [{left}]")
        logger.info(f"  Line number: {problem_line}")
        logger.info(f"  Code: {problem_code}")
        logger.info("")
        
        return problem_line
    
    finally:
        # 恢复原始文件
        shutil.copy(bak, cce_path)
        os.remove(bak)


def compile_kernel(kernel_dir: str) -> bool:
    """
    编译 kernel_aicore 目录下的所有 CCE 文件。
    """
    makefiles = []
    kernel_path = Path(kernel_dir)
    
    for f in kernel_path.glob("Makefile_*.compile"):
        makefiles.append(str(f))
    
    if not makefiles:
        logger.warning("未找到任何 Makefile")
        return True
    
    # Makefile 期望从 kernel_aicore 的父目录执行
    parent_dir = os.path.abspath(os.path.join(kernel_dir, ".."))
    kernel_dir_name = os.path.basename(kernel_dir)
    
    success = True
    for mf in makefiles:
        makefile_rel_path = os.path.join(kernel_dir_name, os.path.basename(mf))
        logger.info(f"编译: make -f {makefile_rel_path}")
        result = subprocess.run(
            ['make', '-f', makefile_rel_path],
            cwd=parent_dir,
            capture_output=True,
            text=True,
            timeout=300
        )
        
        if result.returncode != 0:
            logger.error(f"  ERROR: {result.stderr}")
            success = False
        else:
            logger.info(f"  OK")
    
    return success


def test_precision(test_cmd: str, run_dir: str) -> bool:
    """
    执行精度测试。
    
    返回：True if precision pass, False if fail
    """
    logger.info("运行测试...")
    result = subprocess.run(
        shlex.split(test_cmd),
        cwd=run_dir,
        shell=False,
        capture_output=True,
        text=True,
        timeout=600
    )
    
    output = result.stdout + result.stderr
    
    fail_patterns = [
        r'FAIL',
        r'ERROR',
        r'Exception',
        r'Traceback',
        r'precision\s+fail',
        r'PRECISION\s+FAIL',
    ]
    
    for pattern in fail_patterns:
        if re.search(pattern, output, re.IGNORECASE):
            return False
    
    pass_patterns = [
        r'PASSED',
        r'\bpassed\b',
        r'precision\s+(test\s+)?pass',
        r'PRECISION\s+(TEST\s+)?PASS',
        r'✓',
        r'Max\s+diff:\s+0\.0+\b',
    ]
    
    for pattern in pass_patterns:
        if re.search(pattern, output, re.IGNORECASE):
            return True
    
    if result.returncode == 0 and not output.strip():
        return True
    
    return False


def show_info(kernel_dir: str):
    """
    显示所有 CCE 文件信息。
    """
    cce_files = get_all_cce_files(kernel_dir)
    
    logger.info(f"\n{'='*60}")
    logger.info(f"All CCE Files ({len(cce_files)} total)")
    logger.info(f"{'='*60}\n")
    
    for i, f in enumerate(cce_files):
        size = os.path.getsize(f)
        logger.info(f"[{i:3d}] {os.path.basename(f)} ({size} bytes)")


def run_bisection(kernel_dir, test_cmd, run_dir, enable_line_search=True):
    """
    执行文件级二分搜索，定位问题 CCE 文件。
    
    Args:
        kernel_dir: kernel_aicore 目录
        test_cmd: 测试命令
        run_dir: 测试运行目录
        enable_line_search: 是否在定位文件后继续行级二分
    
    Returns:
        (problem_file, problem_line) 或 (problem_file, None)
    """
    all_cce_files = get_all_cce_files(kernel_dir)
    total_files = len(all_cce_files)
    
    if total_files == 0:
        logger.error("没有找到 CCE 文件")
        return None, None
    
    logger.info(f"\n{'='*60}")
    logger.info(f"CCE File Binary Search")
    logger.info(f"{'='*60}")
    logger.info(f"Total CCE files: {total_files}")
    logger.info("")
    
    left = 0
    right = total_files - 1
    round_num = 1
    
    while left < right:
        mid = (left + right) // 2
        
        logger.info(f"\n{'='*60}")
        logger.info(f"Round {round_num}: left={left}, right={right}, mid={mid}")
        logger.info(f"{'='*60}")
        logger.info(f"测试文件 [{left}, {mid}] ({mid - left + 1} files)")
        logger.info("")
        
        # 只测试当前区间的前半部分（不是从 0 开始）
        test_files = all_cce_files[left:mid + 1]
        
        backups = insert_pipe_all_to_files(test_files)
        
        try:
            if not compile_kernel(kernel_dir):
                logger.error("编译失败")
                restore_cce_files(backups)
                return None, None
            
            passed = test_precision(test_cmd, run_dir)
            
            logger.info("")
            if passed:
                logger.info(f"结果: PASS → 问题在文件 [{left}, {mid}]")
                logger.info(f"      继续在前半段二分")
                right = mid
            else:
                logger.info(f"结果: FAIL → 问题在文件 [{mid+1}, {right}]")
                logger.info(f"      去后半段二分")
                left = mid + 1
            
            round_num += 1
        
        finally:
            restore_cce_files(backups)
    
    problem_file = all_cce_files[left]
    
    logger.info(f"\n{'='*60}")
    logger.info(f"FILE BINARY SEARCH COMPLETED!")
    logger.info(f"{'='*60}")
    logger.info(f"Found: file [{left}]")
    logger.info(f"  Path: {problem_file}")
    logger.info(f"  Name: {os.path.basename(problem_file)}")
    logger.info("")
    
    # 继续行级二分
    if enable_line_search:
        problem_line = run_line_bisection(problem_file, kernel_dir, test_cmd, run_dir)
        return problem_file, problem_line
    
    return problem_file, None


def main():
    parser = argparse.ArgumentParser(
        description='CCE File Binary Search Tool (file + line level)'
    )
    parser.add_argument('--kernel-dir', required=True,
                        help='kernel_aicore directory')
    parser.add_argument('--info', action='store_true',
                        help='Show all CCE files')
    parser.add_argument('--test-cmd',
                        help='Test command (required unless --info)')
    parser.add_argument('--run-dir', default='.',
                        help='Test run directory')
    parser.add_argument('--cce-file',
                        help='Specify CCE file for line-level search only')
    parser.add_argument('--no-line-search', action='store_true',
                        help='Skip line-level search after file search')
    
    args = parser.parse_args()
    
    if not os.path.isdir(args.kernel_dir):
        parser.error(f"Directory not found: {args.kernel_dir}")
    
    if args.info:
        show_info(args.kernel_dir)
        return
    
    if not args.test_cmd:
        parser.error("--test-cmd is required for binary search")
    
    # 仅行级二分模式
    if args.cce_file:
        if not os.path.isfile(args.cce_file):
            parser.error(f"File not found: {args.cce_file}")
        
        logger.info(f"\n仅执行行级二分: {args.cce_file}")
        problem_line = run_line_bisection(
            args.cce_file,
            args.kernel_dir,
            args.test_cmd,
            args.run_dir
        )
        
        if problem_line:
            logger.info(f"\n最终结果:")
            logger.info(f"  文件: {args.cce_file}")
            logger.info(f"  行号: {problem_line}")
        return
    
    # 文件级二分 + 行级二分模式
    problem_file, problem_line = run_bisection(
        args.kernel_dir,
        args.test_cmd,
        args.run_dir,
        enable_line_search=not args.no_line_search
    )
    
    if problem_file:
        logger.info(f"\n最终结果:")
        logger.info(f"  文件: {problem_file}")
        if problem_line:
            logger.info(f"  行号: {problem_line}")


if __name__ == '__main__':
    main()