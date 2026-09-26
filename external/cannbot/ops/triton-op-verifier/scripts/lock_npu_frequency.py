#!/usr/bin/env python3
# ----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ----------------------------------------------------------------------------------------------------------

"""A5 NPU 频率锁定与监控辅助模块。

本模块不引入 PyTorch 依赖，所有外部工具通过子进程调用；
当 `drv_hlt_dsmi_test` 不可用或与当前驱动版本不匹配时，会 fallback 到
直接通过 `ctypes` 调用系统 `libdrvdsmi_host.so` 的 `dsmi_set_device_info`
实现等价的 A5 锁频语义。
"""

import ctypes
import logging
import os
import re
import shutil
import subprocess
import threading
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

logger = logging.getLogger("triton_op_verifier.benchmark.frequency")


# 默认搜索路径：可执行文件若不在 PATH 中，则尝试这些目录
_DRV_HLT_DSMI_TEST_CANDIDATES = [
    "/usr/local/Ascend/driver/tools/drv_hlt_dsmi_test",
]
_NPU_SMI_CANDIDATES = [
    "/usr/local/bin/npu-smi",
    "/usr/bin/npu-smi",
]
_DRVDSMI_HOST_CANDIDATES = [
    "/usr/local/Ascend/driver/lib64/driver/libdrvdsmi_host.so",
    "/usr/lib64/libdrvdsmi_host.so",
    "/usr/lib/libdrvdsmi_host.so",
]

# DSMI set_device_info 的调用参数，等价于 drv_hlt_dsmi_test 工具关闭低功耗 idle 的动作。
# 取值来自对 libdrvhltdsmi.so 中 set_lp_idle 实现的反汇编：主命令字 8、子命令字 11，
# 值为 0、长度 1 字节。
_DSMI_LOCK_MAIN_CMD = 8
_DSMI_LOCK_SUB_CMD = 11


def _find_executable(name: str, candidates: List[str]) -> Optional[str]:
    """查找可执行文件；优先 PATH，再遍历候选路径。"""
    found = shutil.which(name)
    if found:
        return found
    for path in candidates:
        if os.path.isfile(path) and os.access(path, os.X_OK):
            return path
    return None


def _find_shared_object(candidates: List[str]) -> Optional[str]:
    """查找共享库文件；优先候选路径，再尝试系统 ldconfig。"""
    for path in candidates:
        if os.path.isfile(path):
            return path
    found = ctypes.util.find_library("drvdsmi_host")
    if found:
        return found
    return None


def _run(cmd: List[str], timeout: int = 30, cwd: Optional[str] = None) -> Tuple[int, str, str]:
    """执行命令并返回 (returncode, stdout, stderr)。"""
    try:
        proc = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=timeout,
            cwd=cwd,
        )
        return proc.returncode, proc.stdout, proc.stderr
    except FileNotFoundError:
        return 127, "", f"command not found: {cmd[0]}"
    except subprocess.TimeoutExpired:
        return 124, "", f"command timed out after {timeout}s: {cmd[0]}"
    except Exception as e:
        return 1, "", f"{type(e).__name__}: {e}"


def _strip_ansi(text: str) -> str:
    """去掉 ANSI 转义字符。"""
    ansi_escape = re.compile(r"\x1B(?:[@-Z\\-_]|\[[0-?]*[ -/]*[@-~])")
    return ansi_escape.sub("", text)


def parse_npu_smi_common(output: str) -> Dict[str, int]:
    """从 `npu-smi info -t common -i <dev>` 的输出解析频率字段。

    Returns:
        {"rated_freq_mhz": int | None, "cur_freq_mhz": int | None}
    """
    text = _strip_ansi(output)
    result: Dict[str, int] = {}
    rated_match = re.search(r"Aicore\s+Freq\(MHZ\)\s*:\s*(\d+)", text)
    cur_match = re.search(r"Aicore\s+curFreq\(MHZ\)\s*:\s*(\d+)", text)
    if rated_match:
        result["rated_freq_mhz"] = int(rated_match.group(1))
    if cur_match:
        result["cur_freq_mhz"] = int(cur_match.group(1))
    return result


def find_drv_hlt_dsmi_test() -> Optional[str]:
    """定位 drv_hlt_dsmi_test 可执行文件。

    该二进制依赖同一目录下的 libdrvhltdsmi.so，因此返回路径后调用方需要在该目录下执行。
    """
    return _find_executable("drv_hlt_dsmi_test", _DRV_HLT_DSMI_TEST_CANDIDATES)


def find_npu_smi() -> Optional[str]:
    """定位 npu-smi 可执行文件。"""
    return _find_executable("npu-smi", _NPU_SMI_CANDIDATES)


def find_drvdsmi_host() -> Optional[str]:
    """定位 libdrvdsmi_host.so 共享库（直接锁频 fallback 需要）。"""
    return _find_shared_object(_DRVDSMI_HOST_CANDIDATES)


def detect_npu_devices(max_devices: int = 128) -> List[int]:
    """探测可用的 NPU 设备 ID 列表。

    优先使用 `npu-smi info -t common -i <id>` 逐设备探测；若 npu-smi 不可用，
    则 fallback 到 torch.npu.device_count()。
    """
    npu_smi = find_npu_smi()
    devices: List[int] = []
    if npu_smi:
        for dev_id in range(max_devices):
            rc, stdout, _ = _run([npu_smi, "info", "-t", "common", "-i", str(dev_id)], timeout=10)
            if rc != 0:
                break
            if "NPU ID" in stdout:
                devices.append(dev_id)
            else:
                break
        if devices:
            return devices

    try:
        import torch
        import torch_npu  # noqa: F401
        count = torch.npu.device_count()
        if count and count > 0:
            return list(range(count))
    except Exception as e:
        logger.debug("torch.npu.device_count() 探测失败: %s", e)

    return devices


def get_device_frequency(dev_id: int) -> Optional[int]:
    """读取单个设备的当前工作频率（MHz）。

    优先读取 `Aicore curFreq(MHZ)`，不存在时 fallback 到 `Aicore Freq(MHZ)`。
    """
    npu_smi = find_npu_smi()
    if not npu_smi:
        return None
    rc, stdout, _ = _run([npu_smi, "info", "-t", "common", "-i", str(dev_id)], timeout=10)
    if rc != 0:
        return None
    parsed = parse_npu_smi_common(stdout)
    return parsed.get("cur_freq_mhz") or parsed.get("rated_freq_mhz")


def get_all_frequencies() -> Dict[int, Optional[int]]:
    """读取所有设备的当前工作频率。"""
    return {dev_id: get_device_frequency(dev_id) for dev_id in detect_npu_devices()}


_dsmi_init_done = False


def _ensure_dsmi_init() -> None:
    """保证全局只调用一次 dsmi_init（ctypes fallback 路径需要）。"""
    global _dsmi_init_done
    if _dsmi_init_done:
        return
    so_path = find_drvdsmi_host()
    if not so_path:
        return
    try:
        lib = ctypes.CDLL(so_path, mode=ctypes.RTLD_GLOBAL)
        lib.dsmi_init.restype = ctypes.c_int
        lib.dsmi_init()
        _dsmi_init_done = True
    except Exception as e:
        logger.debug("[锁频 fallback] dsmi_init 预初始化失败: %s", e)


def lock_device_frequency(dev_id: int) -> Tuple[bool, str]:
    """对单个设备执行 A5 锁频。

    优先使用 `drv_hlt_dsmi_test set_lp_idle <dev> 0`；
    如果该二进制不存在或执行失败（常见原因是与当前驱动版本不匹配导致 so 加载失败），
    则 fallback 到直接通过 ctypes 调用系统 `libdrvdsmi_host.so` 的
    `dsmi_set_device_info(dev_id, main_cmd=8, sub_cmd=11, &value=0, size=1)`，
    该调用与 HLT 工具内部实现等价。

    Returns:
        (success, message)
    """
    binary = find_drv_hlt_dsmi_test()
    if binary:
        binary_dir = os.path.dirname(os.path.abspath(binary))
        rc, stdout, stderr = _run(
            [binary, "set_lp_idle", str(dev_id), "0"],
            timeout=30,
            cwd=binary_dir,
        )
        combined = (stdout + "\n" + stderr).strip()
        if rc == 0:
            return True, f"set_lp_idle {dev_id} 0 成功"
        # HLT 工具存在但失败：记录下来，继续尝试 ctypes fallback
        hlt_error = f"drv_hlt_dsmi_test set_lp_idle {dev_id} 0 失败 (rc={rc}): {combined or 'empty output'}"
        logger.debug("[锁频 HLT] %s", hlt_error)
    else:
        hlt_error = "未找到 drv_hlt_dsmi_test 可执行文件"
        logger.debug("[锁频 HLT] %s", hlt_error)

    # Fallback：直接调用 libdrvdsmi_host.so
    so_path = find_drvdsmi_host()
    if not so_path:
        return False, f"{hlt_error}；且未找到 libdrvdsmi_host.so，无法 fallback"

    try:
        lib = ctypes.CDLL(so_path, mode=ctypes.RTLD_GLOBAL)
        lib.dsmi_init.restype = ctypes.c_int
        init_ret = lib.dsmi_init()
        if init_ret != 0:
            return False, f"{hlt_error}；dsmi_init 失败 (ret={init_ret})"

        lib.dsmi_set_device_info.argtypes = [
            ctypes.c_uint, ctypes.c_uint, ctypes.c_uint, ctypes.c_void_p, ctypes.c_uint
        ]
        lib.dsmi_set_device_info.restype = ctypes.c_int
        value = ctypes.c_ubyte(0)
        ret = lib.dsmi_set_device_info(
            dev_id, _DSMI_LOCK_MAIN_CMD, _DSMI_LOCK_SUB_CMD,
            ctypes.byref(value), ctypes.sizeof(value),
        )
        if ret != 0:
            return False, f"{hlt_error}；dsmi_set_device_info 锁频失败 (ret={ret})"
        return True, f"dsmi_set_device_info 锁频设备 {dev_id} 成功"
    except Exception as e:
        return False, f"{hlt_error}；ctypes 调用 libdrvdsmi_host.so 异常: {type(e).__name__}: {e}"


def lock_npu_frequency(
    devices: Optional[List[int]] = None,
    verify: bool = True,
) -> Tuple[bool, List[int], Dict[int, Optional[int]], Dict[int, str]]:
    """对所有指定 NPU 设备锁频，并可选验证锁频后频率是否稳定。

    Args:
        devices: 要锁频的设备列表；None 表示自动探测。
        verify: 锁频后是否再次采样频率。

    Returns:
        (
            all_locked: bool,
            locked_devices: 成功执行 set_lp_idle 的设备 ID 列表,
            baseline_freqs: 锁频后的频率快照 {dev_id: mhz | None},
            messages: 每个设备的操作信息 {dev_id: message},
        )
    """
    if devices is None:
        devices = detect_npu_devices()

    locked_devices: List[int] = []
    messages: Dict[int, str] = {}
    baseline_freqs: Dict[int, Optional[int]] = {}

    if not devices:
        logger.warning("未探测到 NPU 设备，跳过锁频")
        return False, locked_devices, baseline_freqs, messages

    # dsmi_init 全局只需要调用一次；在第一个设备尝试前完成
    _ensure_dsmi_init()

    for dev_id in devices:
        ok, msg = lock_device_frequency(dev_id)
        messages[dev_id] = msg
        if ok:
            locked_devices.append(dev_id)
            logger.info("[锁频] %s", msg)
        else:
            logger.warning("[锁频] %s", msg)

    if verify and locked_devices:
        time.sleep(0.5)  # 给驱动留一点稳定时间
        baseline_freqs = {dev_id: get_device_frequency(dev_id) for dev_id in locked_devices}
        for dev_id, freq in baseline_freqs.items():
            if freq is None:
                messages[dev_id] = messages.get(dev_id, "") + "；锁频后无法读取频率"
                logger.warning("[锁频] 设备 %d 锁频后无法读取频率", dev_id)
            else:
                logger.info("[锁频] 设备 %d 当前频率: %d MHz", dev_id, freq)

    return len(locked_devices) == len(devices), locked_devices, baseline_freqs, messages


@dataclass
class FrequencyMonitorReport:
    """频率监控报告。

    violations 的每一项依次为：采样序号、设备号、基线频率（MHz）、实际频率（MHz）。
    """
    baseline_freqs: Dict[int, Optional[int]] = field(default_factory=dict)
    samples: List[Dict[int, Optional[int]]] = field(default_factory=list)
    violations: List[Tuple[int, int, Optional[int], Optional[int]]] = field(default_factory=list)

    def has_drift(self) -> bool:
        return bool(self.violations)


class FrequencyMonitor:
    """在 benchmark 期间后台采样 NPU 频率，检测是否发生漂移。

    Usage:
        monitor = FrequencyMonitor(devices=[0,1], interval=1.0)
        with monitor:
            run_benchmark()
        report = monitor.report
        if report.has_drift():
            ...
    """

    def __init__(
        self,
        devices: Optional[List[int]] = None,
        interval: float = 1.0,
        baseline_freqs: Optional[Dict[int, Optional[int]]] = None,
    ):
        self.devices = devices if devices is not None else detect_npu_devices()
        self.interval = max(0.1, interval)
        self._stop_event = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self.report = FrequencyMonitorReport()
        if baseline_freqs is not None:
            self.report.baseline_freqs = dict(baseline_freqs)
        else:
            self.report.baseline_freqs = {dev_id: get_device_frequency(dev_id) for dev_id in self.devices}

    def __enter__(self) -> "FrequencyMonitor":
        if not self.devices:
            logger.warning("[频率监控] 无可用 NPU 设备，监控未启动")
            return self
        logger.info(
            "[频率监控] 启动对设备 %s 的监控，基线频率: %s，采样间隔 %.1fs",
            self.devices,
            {k: f"{v} MHz" for k, v in self.report.baseline_freqs.items() if v is not None},
            self.interval,
        )
        self._stop_event.clear()
        self._thread = threading.Thread(target=self._sample_loop, daemon=True)
        self._thread.start()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        if self._thread is None:
            return
        self._stop_event.set()
        self._thread.join(timeout=self.interval + 1.0)
        if self.report.has_drift():
            logger.warning(
                "[频率监控] 共发现 %d 次频率漂移: %s",
                len(self.report.violations),
                self.report.violations,
            )
        else:
            logger.info("[频率监控] 测试期间未发现频率漂移")

    def _sample_loop(self) -> None:
        """后台采样线程：记录每个设备当前频率，发现漂移时记入 violations。"""
        sample_index = 0
        while not self._stop_event.is_set():
            sample_index += 1
            snapshot: Dict[int, Optional[int]] = {}
            for dev_id in self.devices:
                freq = get_device_frequency(dev_id)
                snapshot[dev_id] = freq
                baseline = self.report.baseline_freqs.get(dev_id)
                if freq is not None and baseline is not None and freq != baseline:
                    self.report.violations.append((sample_index, dev_id, baseline, freq))
                    logger.warning(
                        "[频率漂移] 设备 %d 频率从 %d MHz 变为 %d MHz (sample #%d)",
                        dev_id, baseline, freq, sample_index,
                    )
            self.report.samples.append(snapshot)
            self._stop_event.wait(timeout=self.interval)


def format_frequency_report(report: FrequencyMonitorReport) -> str:
    """把监控报告格式化为人类可读的字符串。"""
    lines = ["NPU 频率监控报告:"]
    lines.append("  基线频率:")
    for dev_id, freq in sorted(report.baseline_freqs.items()):
        lines.append(f"    设备 {dev_id}: {freq if freq is not None else '未知'} MHz")
    lines.append(f"  采样次数: {len(report.samples)}")
    if report.has_drift():
        lines.append(f"  漂移次数: {len(report.violations)}")
        for sample_idx, dev_id, baseline, actual in report.violations[:20]:
            lines.append(
                f"    sample #{sample_idx}, 设备 {dev_id}: {baseline} MHz -> {actual} MHz"
            )
        if len(report.violations) > 20:
            lines.append(f"    ... 还有 {len(report.violations) - 20} 条 ...")
    else:
        lines.append("  漂移次数: 0")
    return "\n".join(lines)
