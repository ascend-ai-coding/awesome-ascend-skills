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
"""ops-evaluation 脚本共享工具：评估排队锁 + vendors 子目录检测。

evaluate_ops.py / evaluate_ops_direct.py / build_ops.py 原本各自重复实现，
统一到本模块维护（行为与原版一致）。
"""

import fcntl
import os
import time


def acquire_eval_lock(lock_path: str, timeout: float = 300) -> int:
    """阻塞获取评估排队锁。返回 fd。"""
    fd = os.open(lock_path, os.O_RDWR | os.O_CREAT, 0o644)
    deadline = time.monotonic() + timeout
    while True:
        try:
            fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
            return fd
        except OSError as e:
            if time.monotonic() >= deadline:
                os.close(fd)
                raise TimeoutError(
                    f"Failed to acquire eval lock {lock_path} within {timeout}s"
                ) from e
            time.sleep(1)


def release_eval_lock(fd: int):
    """释放评估排队锁。"""
    try:
        fcntl.flock(fd, fcntl.LOCK_UN)
        os.close(fd)
    except OSError:
        pass


def detect_vendor_subdir(install_path: str) -> str:
    """检测安装后的 vendors 子目录名（如 "custom_nn"）。"""
    vendors_dir = os.path.join(install_path, "vendors")
    if os.path.isdir(vendors_dir):
        subdirs = []
        for d in os.listdir(vendors_dir):
            if not os.path.isdir(os.path.join(vendors_dir, d)):
                continue
            if d.startswith("custom") or d.startswith("omni_custom"):
                subdirs.append(d)
        if subdirs:
            return subdirs[0]
    if os.path.isdir(os.path.join(vendors_dir, "customize")):
        return "customize"
    return "custom_nn"
