#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

import os
import shutil
import subprocess
import logging
from pathlib import Path


def setup_logging():
    logging.basicConfig(level=logging.INFO, format='%(message)s')


def validate_path(path, path_type="路径"):
    if not os.path.exists(path):
        return False, f"错误：{path_type}不存在: {path}"
    return True, None


def read_file(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        return f.readlines()


def write_file(file_path, lines):
    with open(file_path, 'w', encoding='utf-8') as f:
        f.writelines(lines)


def print_error_info(output, logger, max_lines=10):
    error_lines = [line for line in output.split('\n') if 'error' in line.lower()]
    if error_lines:
        logger.info("Error 信息:")
        for line in error_lines[:max_lines]:
            logger.info(f"  {line}")


def get_commentable_lines(lines, error_in_t=False):
    commentable_lines = []
    fast_commentable_lines = []
    skip_keywords = ['set_flag', 'wait_flag', 'pipe_barrier']

    # Find the last } in the file ([aicore] main function's closing brace) — never comment it
    tensor_close = None
    for i in range(len(lines) - 1, -1, -1):
        stripped = lines[i].strip()
        if '}' in stripped and not stripped.startswith('//'):
            tensor_close = i + 1
            break

    for i, line in enumerate(lines, 1):
        stripped = line.strip()
        if i == tensor_close:
            continue
        should_skip = (
            not stripped or
            stripped.startswith('//') or
            '#' in stripped or
            any(keyword in stripped for keyword in skip_keywords)
        )
        if '{' in stripped and '[aicore]' in stripped:
            should_skip = True

        if should_skip:
            continue
        else:
            commentable_lines.append(i)
            is_t_operation = stripped.startswith('T') and '<' in stripped and '>' in stripped
            if is_t_operation:
                fast_commentable_lines.append(i)

    if error_in_t:
        return fast_commentable_lines
    else:
        return commentable_lines


def _build_brace_pairs(lines):
    pairs = {}
    stack = []
    for i, line in enumerate(lines):
        for ch in line:
            if ch == '{':
                stack.append(i)
            elif ch == '}' and stack:
                pairs[stack.pop()] = i
    return pairs


def comment_lines_by_indices(lines, line_indices):
    lines = lines[:]
    brace_pairs = _build_brace_pairs(lines)
    extra = set()
    for ln in set(line_indices):
        close = brace_pairs.get(ln - 1)
        if close is not None:
            extra.add(close + 1)
    for line_num in sorted(set(line_indices) | extra, reverse=True):
        lines[line_num - 1] = '// ' + lines[line_num - 1]
    return lines


def uncomment_lines_by_indices(lines, line_indices):
    lines = lines[:]
    brace_pairs = _build_brace_pairs(lines)
    extra = set()
    for ln in set(line_indices):
        close = brace_pairs.get(ln - 1)
        if close is not None:
            extra.add(close + 1)
    for line_num in sorted(set(line_indices) | extra):
        line_idx = line_num - 1
        if lines[line_idx].strip().startswith('//'):
            lines[line_idx] = lines[line_idx][3:]
    return lines


def comment_lines_by_range(lines, start_idx, end_idx):
    for i in range(start_idx, end_idx + 1):
        if not lines[i].strip().startswith('//'):
            lines[i] = '// ' + lines[i]


def has_error(returncode, output, use_pypto_test_framework=False):
    if not use_pypto_test_framework and returncode == 0:
        return False
    return "aicore error" in output.lower()


def run_test(test_cmd, run_dir):
    import shlex
    if isinstance(test_cmd, str):
        test_cmd = shlex.split(test_cmd)
    result = subprocess.run(
        test_cmd,
        cwd=run_dir,
        capture_output=True,
        text=True,
        errors='ignore',
        timeout=1800,
        check=False
    )
    return result.returncode, result.stdout + result.stderr


def comment_special_lines(lines):
    for i, line in enumerate(lines):
        if 'set_flag' in line or 'wait_flag' in line or 'pipe_barrier' in line:
            if not line.strip().startswith('//'):
                lines[i] = '// ' + line
    return lines


def backup_and_test(cce_file, test_cmd, run_dir, modify_func, use_pypto_test_framework=False):
    backup_file = cce_file + ".bak"
    shutil.copy(cce_file, backup_file)

    cce_lines = read_file(cce_file)
    original_lines = cce_lines.copy()

    cce_lines = comment_special_lines(cce_lines)
    modified_lines = modify_func(cce_lines)

    try:
        write_file(cce_file, modified_lines)
        returncode, output = run_test(test_cmd, run_dir)
        error_exists = has_error(returncode, output, use_pypto_test_framework)
    finally:
        write_file(cce_file, original_lines)
        if os.path.exists(backup_file):
            os.remove(backup_file)

    return error_exists, output, original_lines


_PIP_CMD = shutil.which("pip3") or shutil.which("pip") or "/usr/bin/pip3"
DEFAULT_TIMEOUT = 1800

KNOWN_LOCATIONS = {
    'device_switch.h': "framework/src/machine/utils/device_switch.h",
}


def locate_file(base_dir, known_rel_path, filename):
    primary = os.path.join(base_dir, known_rel_path)
    if os.path.exists(primary):
        return primary
    for root, _, files in os.walk(base_dir):
        if filename in files:
            return os.path.join(root, filename)
    return None


def find_installed_tile_fwk_config():
    result = subprocess.run([_PIP_CMD, "show", "pypto"], capture_output=True, text=True)
    if result.returncode != 0:
        return None
    location = None
    for line in result.stdout.split('\n'):
        if line.startswith('Location:'):
            location = line.split(':', 1)[1].strip()
            break
    if not location:
        return None
    for root, _, files in os.walk(location):
        if 'tile_fwk_config.json' in files:
            return os.path.join(root, 'tile_fwk_config.json')
    return None


def _get_pypto_target_location():
    result = subprocess.run([_PIP_CMD, "show", "pypto"], capture_output=True, text=True)
    if result.returncode != 0:
        return None
    for line in result.stdout.split('\n'):
        if line.startswith('Location:'):
            return line.split(':', 1)[1].strip()
    return None


def _remove_existing_pypto(target):
    for name in os.listdir(target):
        if name == 'pypto' or (name.startswith('pypto-') and name.endswith('.dist-info')):
            path = os.path.join(target, name)
            if os.path.isdir(path):
                shutil.rmtree(path)


def build_and_install(pypto_path):
    result = subprocess.run(
        ["python3", "build_ci.py", "-f", "python3", "--disable_auto_execute"],
        shell=False, capture_output=True, text=True, cwd=pypto_path,
        timeout=DEFAULT_TIMEOUT)
    if result.returncode != 0:
        return False

    whl_files = list(Path(pypto_path).glob("build_out/pypto*.whl"))
    if not whl_files:
        return False

    target = _get_pypto_target_location()
    if target:
        _remove_existing_pypto(target)
    install_cmd = [_PIP_CMD, "install", str(whl_files[0]), "--force", "--no-deps"]
    if target:
        install_cmd += ["--target", target]
    result = subprocess.run(
        install_cmd, shell=False, capture_output=True, text=True, timeout=300)
    if result.returncode != 0:
        return False
    return True
