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
Session 级 NPU 设备租约系统。

核心设计：一个算子的整个进化生命周期（baseline + 所有轮次 evolved）绑定同一张卡，
确保性能数据可比。多个算子可并行进化，各自绑定不同的卡。

两层锁机制：
1. session lease (flock): 主 agent 持有，整个进化 session 独占一张卡
2. eval lock (flock): 子 agent 竞争，同一 session 内评估串行排队

用法:
    # 主 agent: 获取 session 级设备租约
    lease = SessionDeviceLease(pool_dir="output/session/device_pool")
    device_id = lease.acquire_session(op_name="FastGELU", timeout=60)

    # 子 agent: 评估前获取排队锁
    with lease.eval_lock(timeout=300):
        run_evaluation(device_id=device_id)

    # 主 agent: 进化结束释放
    lease.release_session()

CLI:
    python device_lease.py list-available --pool-dir <dir>
    python device_lease.py acquire-session --pool-dir <dir> --op-name <name> --session-id <id>
    python device_lease.py release-session --pool-dir <dir> --session-id <id>
    python device_lease.py acquire-eval --pool-dir <dir> --timeout 300
    python device_lease.py release-eval --pool-dir <dir>
"""

import argparse
import fcntl
import json
import logging
import os
import shutil
import signal
import subprocess
import sys
import time
from contextlib import contextmanager
from pathlib import Path
from typing import List, Optional

logger = logging.getLogger(__name__)

# 数据输出专用 logger：CLI 结果走 stdout（agent 调用协议通道），
# 与 LOGGER（stderr 进度/警告）分离，避免 lint G.LOG.02 误报 print。
DATA_LOGGER = logging.getLogger(f"{__name__}.data")
_DATA_HANDLER = logging.StreamHandler(sys.stdout)
_DATA_HANDLER.setFormatter(logging.Formatter("%(message)s"))
DATA_LOGGER.addHandler(_DATA_HANDLER)
DATA_LOGGER.propagate = False
DATA_LOGGER.setLevel(logging.INFO)

# npu-smi 可执行文件绝对路径（避免依赖 PATH 查找）
NPU_SMI_BIN = shutil.which("npu-smi") or "/usr/local/bin/npu-smi"


def _is_pid_alive(pid: int) -> bool:
    """检查指定 pid 的进程是否存活。"""
    try:
        os.kill(pid, 0)
        return True
    except (ProcessLookupError, PermissionError):
        return False
    except OSError:
        return False


def _parse_npu_smi_line(line: str) -> Optional[int]:
    """从 npu-smi info 的表格行解析设备 ID，非数据行返回 None。"""
    line = line.strip()
    if not (line.startswith("|") and len(line) > 2):
        return None
    for part in line.split("|"):
        part = part.strip()
        if part.isdigit():
            dev_id = int(part)
            return dev_id if dev_id < 100 else None
    return None


def _detect_npu_devices() -> List[int]:
    """检测系统中所有可用的 NPU 设备 ID。"""
    try:
        result = subprocess.run(
            [NPU_SMI_BIN, "info"],
            capture_output=True, text=True, timeout=10,
        )
        if result.returncode != 0:
            return []
        device_ids = []
        for line in result.stdout.splitlines():
            dev_id = _parse_npu_smi_line(line)
            if dev_id is not None:
                device_ids.append(dev_id)
        return sorted(set(device_ids))
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return []


def _try_flock_once(fd: int, deadline: float) -> tuple:
    """尝试非阻塞 flock 一次。

    Returns (acquired, timed_out, last_error)。
    """
    try:
        fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        return True, False, None
    except OSError as e:
        return False, time.monotonic() >= deadline, e


def _release_lock_quietly(fd: int):
    """释放 flock，失败仅记录。"""
    try:
        fcntl.flock(fd, fcntl.LOCK_UN)
        logger.debug("Eval lock released")
    except OSError:
        pass


def _parse_memory_pct(line: str) -> float:

    """从 npu-smi usages 行提取显存占用百分比，无则返回 -1。"""
    for part in line.split():
        if "%" in part:
            try:
                return float(part.replace("%", "")) / 100.0
            except ValueError:
                continue
    return -1.0


def _get_device_memory_usage(device_id: int) -> float:
    """获取指定设备的显存占用率 (0.0 ~ 1.0)。返回 -1 表示查询失败。"""
    try:
        result = subprocess.run(
            [NPU_SMI_BIN, "info", "-t", "usages", "-i", str(device_id)],
            capture_output=True, text=True, timeout=10,
        )
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return -1.0
    if result.returncode != 0:
        return -1.0
    return _parse_memory_usage_from_smi(result.stdout)


def _parse_memory_usage_from_smi(stdout: str) -> float:
    """从 npu-smi usages 输出解析显存占用率，无匹配行返回 -1。"""
    for line in stdout.splitlines():
        if "HBM Capacity" in line or "Memory Usage" in line or "HBM Usage" in line:
            pct = _parse_memory_pct(line)
            if pct >= 0:
                return pct
    return -1.0


class SessionDeviceLease:
    """Session 级设备租约管理器。

    文件结构:
        pool_dir/
        ├── session_device_{id}.lock    # session 级 flock
        ├── session_{id}_lease.json     # 租约元数据
        └── eval.lock                   # 评估排队锁
    """

    def __init__(self, pool_dir: str):
        self.pool_dir = Path(pool_dir)
        self.pool_dir.mkdir(parents=True, exist_ok=True)
        self._session_lock_fd: Optional[int] = None
        self._session_device_id: Optional[int] = None
        self._session_id: Optional[str] = None
        self._eval_lock_fd: Optional[int] = None

    @property
    def device_id(self) -> Optional[int]:
        return self._session_device_id

    # CLI 模式下 acquire-session 进程写完 lease 即退出，PID 必然失效。
    # 因此不能仅靠 PID 判断租约有效性，需结合租约年龄：
    #   PID 存活 → 有效
    #   PID 已死 + 租约年龄 < MAX_LEASE_AGE_HOURS → 有效（session 仍在运行）
    #   PID 已死 + 租约年龄 ≥ MAX_LEASE_AGE_HOURS → 过期，清理
    MAX_LEASE_AGE_HOURS = 12

    def list_available(self, device_ids: Optional[List[int]] = None) -> List[int]:
        """列出当前未被任何 session 锁定的设备。"""
        if device_ids is None:
            device_ids = _detect_npu_devices()
        if not device_ids:
            return []

        available = []
        for dev_id in device_ids:
            if not self._is_lease_alive(dev_id):
                available.append(dev_id)
        return available

    def acquire_session(
        self,
        op_name: str,
        session_id: str,
        device_ids: Optional[List[int]] = None,
        timeout: float = 60,
        preferred_device: Optional[int] = None,
    ) -> int:
        """获取 session 级设备租约。

        使用 lease.json + pid 存活检查判断设备是否被占用。
        不依赖 flock 持有，acquire 完成后即可退出进程。

        选择策略：
        1. 若指定 preferred_device 且空闲，直接使用
        2. 否则选择显存占用最低的空闲设备
        3. 所有设备被占用时阻塞等待，超时抛异常

        Returns:
            绑定的 device_id

        Raises:
            TimeoutError: 超时未获取到设备
            RuntimeError: 无可用设备
        """
        if self._session_device_id is not None:
            raise RuntimeError("Session lease already held")

        if device_ids is None:
            device_ids = _detect_npu_devices()
        if not device_ids:
            raise RuntimeError("No NPU devices detected")

        # 优先尝试 preferred_device
        if preferred_device is not None and preferred_device in device_ids:
            try_order = [preferred_device] + [d for d in device_ids if d != preferred_device]
        else:
            # 按显存占用排序（低占用优先）
            usage = []
            for dev_id in device_ids:
                mem = _get_device_memory_usage(dev_id)
                usage.append((dev_id, mem if mem >= 0 else 999.0))
            usage.sort(key=lambda x: x[1])
            try_order = [d for d, _ in usage]

        deadline = time.monotonic() + timeout

        while True:
            for dev_id in try_order:
                if not self._is_lease_alive(dev_id):
                    # 设备空闲，写入 lease 占位
                    self._session_device_id = dev_id
                    self._session_id = session_id
                    self._write_lease_metadata(dev_id, op_name, session_id)
                    logger.info(
                        f"Session lease acquired: device {dev_id} "
                        f"for {op_name} (session: {session_id})"
                    )
                    return dev_id

            # 所有设备被占用，等待重试
            if time.monotonic() >= deadline:
                raise TimeoutError(
                    f"Failed to acquire session device within {timeout}s. "
                    f"All devices {device_ids} are occupied by other sessions."
                )
            time.sleep(2)

    def adopt_session(self, session_id: str, device_id: Optional[int] = None):
        """从已有 lease 文件恢复 session 状态（CLI release-session 场景）。

        CLI 下 release-session 是新进程，需要把 lease 元数据中的
        session_id / device_id 回填到本实例，release_session 才能正确清理。
        """
        self._session_id = session_id
        self._session_device_id = device_id

    def release_session(self):
        """释放 session 级设备租约（删除 lease 文件）。"""
        if self._session_device_id is None:
            return

        dev_id = self._session_device_id
        session_id = self._session_id

        # 清理 lease 元数据文件
        if session_id:
            lease_path = self.pool_dir / f"session_{session_id}_lease.json"
            try:
                lease_path.unlink(missing_ok=True)
            except OSError:
                pass

        self._session_device_id = None
        self._session_id = None
        logger.info(f"Session lease released: device {dev_id} (session: {session_id})")

    @contextmanager
    def eval_lock(self, timeout: float = 300):
        """评估排队锁 — 子 agent 评估前获取，评估后释放。

        同一 session 内多个子 agent 通过此锁串行排队使用绑定的卡。

        Args:
            timeout: 最大等待时间（秒）

        Yields:
            device_id

        Raises:
            TimeoutError: 超时未获取到锁
        """
        lock_path = self.pool_dir / "eval.lock"
        fd = os.open(str(lock_path), os.O_RDWR | os.O_CREAT, 0o644)

        acquired = False
        deadline = time.monotonic() + timeout

        try:
            while not acquired:
                acquired, timed_out, last_err = _try_flock_once(fd, deadline)
                if timed_out:
                    os.close(fd)
                    raise TimeoutError(
                        f"Failed to acquire eval lock within {timeout}s"
                    ) from last_err
                if not acquired:
                    time.sleep(1)

            logger.debug("Eval lock acquired (device %s)", self._session_device_id)
            yield self._session_device_id
        finally:
            if acquired:
                _release_lock_quietly(fd)
            try:
                os.close(fd)
            except OSError:
                pass

    def _write_lease_metadata(self, device_id: int, op_name: str, session_id: str):
        """写入租约元数据文件。"""
        lease_path = self.pool_dir / f"session_{session_id}_lease.json"
        metadata = {
            "device_id": device_id,
            "op_name": op_name,
            "session_id": session_id,
            "pid": os.getpid(),
            "start_time": time.strftime("%Y-%m-%dT%H:%M:%S"),
        }
        with open(lease_path, "w", encoding="utf-8") as f:
            json.dump(metadata, f, indent=2)

    def _clean_stale_lease(self, lease_file: Path, device_id: int, pid):
        """清理过期租约的元数据文件和设备锁文件。"""
        logger.info(
            "Cleaning stale lease: device %d (pid %s dead, lease expired)",
            device_id, pid,
        )
        try:
            lease_file.unlink(missing_ok=True)
            lock_path = self.pool_dir / f"session_device_{device_id}.lock"
            lock_path.unlink(missing_ok=True)
        except OSError as e:
            # 清理失败不影响判定结果（租约已视为失效），仅记录
            logger.debug("stale lease cleanup failed: %s", e)

    def _is_lease_alive(self, device_id: int) -> bool:
        """检查指定设备的 lease 是否仍然有效。"""
        for f in self.pool_dir.iterdir():
            if not f.name.endswith("_lease.json"):
                continue
            try:
                with open(f, "r", encoding="utf-8") as fh:
                    meta = json.load(fh)
            except (json.JSONDecodeError, OSError):
                continue
            if meta.get("device_id") != device_id:
                continue
            pid = meta.get("pid")
            if pid and _is_pid_alive(pid):
                return True
            # PID 已死，检查租约年龄
            if self._is_lease_fresh(meta.get("start_time", "")):
                return True  # 租约较新，session 可能仍在运行
            # 租约过期，清理
            self._clean_stale_lease(f, device_id, pid)
            return False
        return False

    def _is_lease_fresh(self, start_str: str) -> bool:
        """按租约开始时间判断是否仍在 MAX_LEASE_AGE_HOURS 有效期内。"""
        if not start_str:
            return False
        try:
            start_ts = time.mktime(
                time.strptime(start_str, "%Y-%m-%dT%H:%M:%S")
            )
        except (ValueError, OverflowError):
            return False
        age_hours = (time.time() - start_ts) / 3600
        return age_hours < self.MAX_LEASE_AGE_HOURS


# ---------------------------------------------------------------------------
# Standalone eval lock functions (for subprocess / CLI usage by sub-agents)
# ---------------------------------------------------------------------------

def acquire_eval_lock_blocking(pool_dir: str, timeout: float = 300) -> int:
    """阻塞获取评估锁，返回 fd。供子 agent 在 shell 中使用。"""
    pool_path = Path(pool_dir)
    lock_path = pool_path / "eval.lock"
    fd = os.open(str(lock_path), os.O_RDWR | os.O_CREAT, 0o644)

    deadline = time.monotonic() + timeout
    while True:
        try:
            fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
            return fd
        except OSError as e:
            if time.monotonic() >= deadline:
                os.close(fd)
                raise TimeoutError(f"Eval lock timeout ({timeout}s)") from e
            time.sleep(1)


def release_eval_lock(fd: int):
    """释放评估锁。"""
    try:
        fcntl.flock(fd, fcntl.LOCK_UN)
        os.close(fd)
    except OSError:
        pass


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="NPU Device Lease Manager")
    subparsers = parser.add_subparsers(dest="command", required=True)

    p_list = subparsers.add_parser("list-available", help="List available NPU devices")
    p_list.add_argument("--pool-dir", required=True)
    p_list.add_argument("--device-ids", type=str, default=None,
                        help="Comma-separated device IDs to check")

    p_acq = subparsers.add_parser("acquire-session", help="Acquire session device lease")
    p_acq.add_argument("--pool-dir", required=True)
    p_acq.add_argument("--op-name", required=True)
    p_acq.add_argument("--session-id", required=True)
    p_acq.add_argument("--timeout", type=float, default=60)
    p_acq.add_argument("--device-ids", type=str, default=None)
    p_acq.add_argument("--preferred-device", type=int, default=None)

    p_rel = subparsers.add_parser("release-session", help="Release session device lease")
    p_rel.add_argument("--pool-dir", required=True)
    p_rel.add_argument("--session-id", required=True)

    p_eval = subparsers.add_parser("acquire-eval", help="Acquire eval lock (blocking)")
    p_eval.add_argument("--pool-dir", required=True)
    p_eval.add_argument("--timeout", type=float, default=300)

    p_rel_eval = subparsers.add_parser("release-eval", help="Release eval lock")
    p_rel_eval.add_argument("--pool-dir", required=True)

    return parser


def _parse_device_ids(raw: str) -> Optional[List[int]]:
    if not raw:
        return None
    return [int(d.strip()) for d in raw.split(",")]


def _cmd_acquire_session(args) -> int:
    lease = SessionDeviceLease(args.pool_dir)
    try:
        dev = lease.acquire_session(
            op_name=args.op_name,
            session_id=args.session_id,
            device_ids=_parse_device_ids(args.device_ids),
            timeout=args.timeout,
            preferred_device=args.preferred_device,
        )
        # 同步输出 device_id 后退出
        # 设备占用状态通过 lease.json + pid 存活检查判断
        DATA_LOGGER.info("%s", dev)
        return 0
    except (TimeoutError, RuntimeError) as e:
        logger.error("ERROR: %s", e)
        return 1


def _cmd_release_session(args) -> int:
    # 删除 lease 文件来释放设备
    lease = SessionDeviceLease(args.pool_dir)
    # 读取 lease 文件获取 device_id 并回填到实例
    lease_path = Path(args.pool_dir) / f"session_{args.session_id}_lease.json"
    device_id = None
    if lease_path.exists():
        try:
            with open(lease_path, "r") as f:
                meta = json.load(f)
            device_id = meta.get("device_id")
        except (json.JSONDecodeError, OSError):
            pass
    lease.adopt_session(args.session_id, device_id)
    lease.release_session()
    logger.info("Session %s released", args.session_id)
    return 0


class _LeaseExitSignal(Exception):
    """SIGTERM/SIGINT 触发的优雅退出信号（替代 lambda 中的 sys.exit）。"""


def _cmd_acquire_eval(args) -> int:
    try:
        fd = acquire_eval_lock_blocking(args.pool_dir, args.timeout)
    except TimeoutError as e:
        logger.error("ERROR: %s", e)
        return 1
    DATA_LOGGER.info("ACQUIRED (fd=%d)", fd)
    # 保持锁直到被终止
    signal.signal(signal.SIGTERM, lambda *_: (_ for _ in ()).throw(_LeaseExitSignal()))
    signal.signal(signal.SIGINT, lambda *_: (_ for _ in ()).throw(_LeaseExitSignal()))
    try:
        while True:
            time.sleep(60)
    except _LeaseExitSignal:
        release_eval_lock(fd)
    return 0


def main():
    logging.basicConfig(level=logging.INFO, format="%(levelname)s - %(message)s")
    args = _build_parser().parse_args()

    if args.command == "list-available":
        lease = SessionDeviceLease(args.pool_dir)
        available = lease.list_available(_parse_device_ids(args.device_ids))
        DATA_LOGGER.info("%s", ",".join(str(d) for d in available))
        return

    if args.command == "acquire-session":
        sys.exit(_cmd_acquire_session(args))

    if args.command == "release-session":
        sys.exit(_cmd_release_session(args))

    if args.command == "acquire-eval":
        sys.exit(_cmd_acquire_eval(args))

    if args.command == "release-eval":
        # eval lock 通过进程退出自动释放
        logger.info("Eval lock is released when the acquire-eval process exits")


if __name__ == "__main__":
    main()
