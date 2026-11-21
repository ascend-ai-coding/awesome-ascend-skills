#
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
#
"""超时控制模块 — 固定超时 + 空闲超时双重保障"""
import os
import signal
import time
import threading
from pathlib import Path
from typing import Optional


def is_alive(pid: int) -> bool:
    """检查进程是否存活"""
    try:
        os.kill(pid, 0)
        return True
    except (OSError, ProcessLookupError):
        return False


def terminate_process_group(pid: int, grace_sec: int = 10) -> bool:
    """终止进程组（SIGTERM → grace → SIGKILL）"""
    if not is_alive(pid):
        return True
    
    # 发送 SIGTERM 给进程组
    try:
        os.killpg(pid, signal.SIGTERM)
    except ProcessLookupError:
        return True
    except PermissionError:
        return False
    
    # 等待 grace period
    start = time.time()
    while time.time() - start < grace_sec:
        if not is_alive(pid):
            return True
        time.sleep(0.1)
    
    # 仍然存活，发送 SIGKILL
    if is_alive(pid):
        try:
            os.killpg(pid, signal.SIGKILL)
        except ProcessLookupError:
            return True
        except PermissionError:
            return False
        time.sleep(0.5)
    
    return not is_alive(pid)


class FixedTimeout:
    """固定超时计时器"""
    
    def __init__(self, pid: int, timeout_sec: int, callback):
        self.pid = pid
        self.timeout_sec = timeout_sec
        self.callback = callback
        self.timer: Optional[threading.Timer] = None
        self.cancelled = False
    
    def start(self):
        """启动固定超时计时器"""
        self.timer = threading.Timer(self.timeout_sec, self._on_timeout)
        self.timer.daemon = True
        self.timer.start()
    
    def cancel(self):
        """取消计时器"""
        self.cancelled = True
        if self.timer:
            self.timer.cancel()
    
    def _on_timeout(self):
        if not self.cancelled:
            self.callback("fixed_timeout")


def idle_timeout_monitor(
    pid: int,
    log_path: Path,
    idle_seconds: int,
    callback,
    poll_interval: int = 5
) -> threading.Thread:
    """启动空闲超时监控线程"""
    
    def monitor():
        last_size = 0
        last_change = time.time()
        
        while is_alive(pid):
            current_size = log_path.stat().st_size if log_path.exists() else 0
            
            if current_size > last_size:
                last_size = current_size
                last_change = time.time()
            elif time.time() - last_change > idle_seconds:
                # 空闲超时
                callback("idle_timeout")
                return
            
            time.sleep(poll_interval)
    
    thread = threading.Thread(target=monitor, daemon=True)
    thread.start()
    return thread
