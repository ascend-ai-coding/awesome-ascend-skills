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

"""verify.py / benchmark.py 共用的工具函数，避免重复实现。"""
import datetime
import hashlib
import os
from typing import Any, Dict, List

# CANN 安装信息文件的候选文件名。顺序与 triton ascend backend 求哈希时的顺序一致，
# 保证这里记录的路径与真正进 cache key 的那个文件是同一个。
CANN_INFO_FILE_NAMES = ("ascend_toolkit_install.info", "ascend_all_cann_install.info")

# 这两个环境变量一旦设置，triton 会改用自定义 / 远端缓存后端，
# TRITON_CACHE_DIR 随之失效，缓存隔离不再成立。
CACHE_OVERRIDE_ENV_VARS = ("TRITON_CACHE_MANAGER", "TRITON_REMOTE_CACHE_BACKEND")

TRITON_CACHE_DIR_ENV = "TRITON_CACHE_DIR"
DEFAULT_TRITON_CACHE_DIR = "~/.triton/cache"


def move_to_device(x: Any, device: Any) -> Any:
    """递归把 x 中的所有 torch.Tensor（含嵌套 list/tuple 内的）迁移到 device。

    - torch.Tensor: 直接 .to(device)
    - list: 递归迁移每个元素，保留 list 类型
    - tuple: 递归迁移每个元素，保留 tuple 类型
    - 其他（标量 / None）: 原样返回

    verify.py / benchmark.py 共用，避免在两个脚本中重复实现递归迁移逻辑。
    """
    import torch
    if isinstance(x, torch.Tensor):
        return x.to(device)
    if isinstance(x, list):
        return [move_to_device(e, device) for e in x]
    if isinstance(x, tuple):
        return tuple(move_to_device(e, device) for e in x)
    return x


def describe_input(inputs: List[Any]) -> List[Dict[str, Any]]:
    """将输入列表描述为结构化字段，便于写入 JSON。

    - torch.Tensor → {"type": "tensor", "shape": [...], "dtype": "..."}
    - 其他标量/对象 → {"type": "scalar", "value": repr(x)}
    """
    try:
        import torch
    except Exception:
        torch = None

    descs: List[Dict[str, Any]] = []
    for x in inputs:
        if torch is not None and isinstance(x, torch.Tensor):
            descs.append({
                "type": "tensor",
                "shape": list(x.shape),
                "dtype": str(x.dtype),
            })
        else:
            try:
                val = x if isinstance(x, (int, float, bool, str)) else repr(x)
            except Exception:
                val = "<unrepr>"
            descs.append({"type": "scalar", "value": val})
    return descs


# ---------------------------------------------------------------------------
# 环境指纹采集
#
# 目的：让 verify_result.json 自带"本次跑在什么环境上"的证据，使精度结论可以
# 跨机器 / 跨时间 / 跨 agent 复核。所有采集项都必须只读、且失败时降级为
# {"error": ...}，绝不能把验证本身带挂。
# ---------------------------------------------------------------------------


def _describe_error(exc: BaseException) -> str:
    """把异常压成一行可读文本，便于直接写进 JSON。"""
    return "{}: {}".format(type(exc).__name__, exc)


def _safe_section(collector) -> Any:
    """执行单个采集函数；成功返回其结果，任何异常都收敛成 {"error": ...}，绝不向外抛。"""
    try:
        return collector()
    except Exception as e:  # 采集失败不得影响验证主流程
        return {"error": _describe_error(e)}


def _local_timestamp() -> str:
    """带本地时区偏移的 ISO-8601 时间戳，精确到秒。

    显式从 UTC 取当前时刻再转本地时区，避免 naive datetime：跨机器比对
    verify 结果时，没有时区的时间戳无法判断先后。
    """
    return datetime.datetime.now(datetime.timezone.utc).astimezone().isoformat(timespec="seconds")


def _collect_triton_info() -> Dict[str, Any]:
    """triton 版本、安装路径，以及 triton_key() 的 sha256 指纹。

    triton_key() 覆盖了全部 triton python 文件 + libtriton.so + backends/ 目录，
    因此单个 sha256 就足以判定"两次运行用的是不是同一个 triton 安装"，
    能抓住小版本漂移（例如编译默认值变化）。原串很长，只存摘要。
    """
    import triton
    from triton.compiler.compiler import triton_key

    info: Dict[str, Any] = {
        "version": getattr(triton, "__version__", None),
        "path": os.path.dirname(os.path.abspath(triton.__file__)),
    }
    try:
        info["key_sha256"] = hashlib.sha256(triton_key().encode("utf-8")).hexdigest()
    except Exception as e:  # 指纹算不出来也要留下其余信息
        info["key_sha256"] = None
        info["key_sha256_error"] = _describe_error(e)
    return info


def default_triton_cache_dir() -> str:
    """triton 未显式指定 TRITON_CACHE_DIR 时的默认缓存目录。"""
    try:
        from triton.runtime.cache import default_cache_dir
        return default_cache_dir()
    except Exception:  # triton 缺失时退回文档中的默认值
        return os.path.expanduser(DEFAULT_TRITON_CACHE_DIR)


def detect_cache_override() -> List[str]:
    """返回已设置的、会让 TRITON_CACHE_DIR 失效的环境变量名列表。"""
    return [name for name in CACHE_OVERRIDE_ENV_VARS if os.environ.get(name)]


def _collect_cache_info(fresh: bool) -> Dict[str, Any]:
    """本次运行实际使用的 Triton 编译缓存状态。

    fresh_effective 记录 TRITON_CACHE_DIR 是否真的决定缓存位置：只要
    TRITON_CACHE_MANAGER / TRITON_REMOTE_CACHE_BACKEND 有一个被设置，
    该变量就不生效，隔离并未真正成立，必须显式记下来而不是假装已隔离。
    """
    return {
        "dir": os.environ.get(TRITON_CACHE_DIR_ENV) or default_triton_cache_dir(),
        "fresh": bool(fresh),
        "fresh_effective": not detect_cache_override(),
        "always_compile": bool(os.environ.get("TRITON_ALWAYS_COMPILE")),
    }


def _find_cann_info_file(ascend_home: str, arch: str) -> Any:
    """按 triton ascend backend 的同一顺序定位 CANN 安装信息文件。"""
    base = os.path.join(ascend_home, arch + "-linux")
    for name in CANN_INFO_FILE_NAMES:
        path = os.path.join(base, name)
        if os.path.exists(path):
            return path
    return None


def _read_cann_version(info_file: Any) -> Any:
    """从 CANN 安装信息文件里读出人类可读的版本号。"""
    if not info_file:
        return None
    with open(info_file, "r", encoding="utf-8", errors="replace") as f:
        for line in f:
            if line.startswith("version="):
                return line.split("=", 1)[1].strip()
    return None


def _collect_cann_info() -> Dict[str, Any]:
    """CANN 安装路径、安装信息文件及其哈希。

    info_sha256 与 Ascend backend options.hash() 使用的是同一个哈希：两次运行
    该值不同则 cache key 必然不同，可直接用来解释缓存命中与否。
    ASCEND_HOME_PATH 未设置时 get_cann_version_file_hash() 会抛 EnvironmentError，
    这条 error 本身就是最有价值的诊断信息，必须记进 JSON。
    """
    info: Dict[str, Any] = {
        "ascend_home_path": os.environ.get("ASCEND_HOME_PATH") or None,
    }
    try:
        from triton.backends.ascend.utils import get_cann_version_file_hash, get_machine_arch
        info["info_sha256"] = get_cann_version_file_hash()
        info_file = _find_cann_info_file(os.environ.get("ASCEND_HOME_PATH", ""), get_machine_arch())
        info["info_file"] = info_file
        info["version"] = _read_cann_version(info_file)
    except Exception as e:  # 环境没 source 是常见故障，要留证据
        info["error"] = _describe_error(e)
    return info


def _collect_torch_info() -> Dict[str, Any]:
    """torch 与 torch_npu 版本痕迹。"""
    import torch

    info: Dict[str, Any] = {
        "version": getattr(torch, "__version__", None),
        "torch_npu_version": None,
    }
    try:
        import torch_npu
        info["torch_npu_version"] = getattr(torch_npu, "__version__", None)
    except Exception as e:  # 非 NPU 环境下缺 torch_npu 属正常
        info["torch_npu_error"] = _describe_error(e)
    return info


def _collect_device_info() -> Dict[str, Any]:
    """本次使用的 NPU 设备信息。

    只有 ASCEND_RT_VISIBLE_DEVICES 对 torch_npu 生效（ASCEND_DEVICE_ID 不生效），
    排查坏卡 / 卡占用时必须知道它的取值。
    """
    info: Dict[str, Any] = {
        "ASCEND_RT_VISIBLE_DEVICES": os.environ.get("ASCEND_RT_VISIBLE_DEVICES"),
        "current_device": None,
        "npu_name": None,
    }
    try:
        import torch
        import torch_npu  # noqa: F401
        if torch.npu.is_available():
            device_idx = torch.npu.current_device()
            info["current_device"] = device_idx
            info["npu_name"] = torch.npu.get_device_name(device_idx)
    except Exception as e:  # CPU-only 环境下不应影响采集
        info["error"] = _describe_error(e)
    return info


def collect_environment(
    fresh: bool = False,
    triton_impl_name: str = "triton_ascend_impl",
    non_compute: bool = False,
) -> Dict[str, Any]:
    """采集本次验证的环境指纹，写入结果 JSON 的 environment 字段。

    必须在真正执行验证的那个进程里调用：主进程未必 import 了 torch_npu，
    且 --fresh-cache 注入的 TRITON_CACHE_DIR 只存在于子进程环境里。

    任何一项采集失败都降级为该分节下的 {"error": ...}，本函数保证只返回 dict、
    永不抛异常，也不修改任何环境变量。
    """
    return {
        "timestamp": _safe_section(_local_timestamp),
        "triton": _safe_section(_collect_triton_info),
        "cache": _safe_section(lambda: _collect_cache_info(fresh)),
        "cann": _safe_section(_collect_cann_info),
        "torch": _safe_section(_collect_torch_info),
        "device": _safe_section(_collect_device_info),
        "cmdline": {
            "triton_impl_name": triton_impl_name,
            "non_compute": bool(non_compute),
        },
    }
