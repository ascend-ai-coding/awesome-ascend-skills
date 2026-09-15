# ----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ----------------------------------------------------------------------------------------------------------

"""_common_utils.collect_environment 的单元测试。

全部用假模块替换 triton / torch / torch_npu，因此只需 CPU，无需 NPU 也无需真装
triton。核心约定：采集函数永远返回 dict、永远不抛异常，失败项降级为 error 字段。

覆盖：全部可用、CANN 缺失（ASCEND_HOME_PATH 未 source）、triton 缺失、
cache manager 被覆盖导致隔离失效。
"""
import json
import sys
import types

import pytest

from _common_utils import CACHE_OVERRIDE_ENV_VARS
from _common_utils import collect_environment

FAKE_TRITON_KEY = "fake-triton-key"
FAKE_CANN_HASH = "0f1e2d3c"
CANN_INFO_TEXT = "package_name=Ascend-cann-toolkit\nversion=8.5.1\narch=x86_64\n"


def _install_module(monkeypatch, name, **attrs):
    """把一个假模块（连同其所有父包）塞进 sys.modules。"""
    module = types.ModuleType(name)
    for key, value in attrs.items():
        setattr(module, key, value)
    monkeypatch.setitem(sys.modules, name, module)
    parent_name, _, child = name.rpartition(".")
    if parent_name:
        parent = sys.modules.get(parent_name)
        if parent is not None:
            monkeypatch.setattr(parent, child, module, raising=False)
    return module


def _install_fake_triton(monkeypatch, cann_home=None, cann_hash=FAKE_CANN_HASH):
    """安装一套最小可用的假 triton 包，含 ascend backend 工具函数。"""
    _install_module(monkeypatch, "triton", __version__="3.2.0", __file__="/fake/triton/__init__.py")
    _install_module(monkeypatch, "triton.compiler")
    _install_module(monkeypatch, "triton.compiler.compiler", triton_key=lambda: FAKE_TRITON_KEY)
    _install_module(monkeypatch, "triton.runtime")
    _install_module(monkeypatch, "triton.runtime.cache", default_cache_dir=lambda: "/fake/home/.triton/cache")
    _install_module(monkeypatch, "triton.backends")
    _install_module(monkeypatch, "triton.backends.ascend")

    def _cann_hash():
        if not cann_home:
            raise EnvironmentError("ASCEND_HOME_PATH is not set, source <ascend-toolkit>/set_env.sh first")
        return cann_hash

    _install_module(
        monkeypatch, "triton.backends.ascend.utils",
        get_cann_version_file_hash=_cann_hash, get_machine_arch=lambda: "x86_64",
    )


def _install_fake_torch(monkeypatch, with_npu=True):
    """安装假 torch / torch_npu；with_npu=False 模拟纯 CPU 机器。"""
    npu = types.SimpleNamespace(
        is_available=lambda: with_npu,
        current_device=lambda: 3,
        get_device_name=lambda idx: "Ascend910B4",
    )
    _install_module(monkeypatch, "torch", __version__="2.9.0", npu=npu)
    if with_npu:
        _install_module(monkeypatch, "torch_npu", __version__="2.9.0.post1")
    else:
        monkeypatch.setitem(sys.modules, "torch_npu", None)


@pytest.fixture(autouse=True)
def clean_cache_env(monkeypatch):
    """每个用例都从"无缓存相关环境变量"的干净状态出发。"""
    for name in CACHE_OVERRIDE_ENV_VARS:
        monkeypatch.delenv(name, raising=False)
    monkeypatch.delenv("TRITON_CACHE_DIR", raising=False)
    monkeypatch.delenv("TRITON_ALWAYS_COMPILE", raising=False)
    monkeypatch.delenv("ASCEND_HOME_PATH", raising=False)
    monkeypatch.delenv("ASCEND_RT_VISIBLE_DEVICES", raising=False)


def _assert_json_serializable(env):
    """结果必须能直接 json.dump 进 verify_result.json。"""
    assert isinstance(env, dict)
    json.dumps(env, ensure_ascii=False)


def test_everything_available(monkeypatch, tmp_path):
    """全部可用：各分节均无 error，指纹字段齐全。"""
    cann_home = tmp_path / "Ascend"
    (cann_home / "x86_64-linux").mkdir(parents=True)
    (cann_home / "x86_64-linux" / "ascend_toolkit_install.info").write_text(
        CANN_INFO_TEXT, encoding="utf-8",
    )
    monkeypatch.setenv("ASCEND_HOME_PATH", str(cann_home))
    monkeypatch.setenv("ASCEND_RT_VISIBLE_DEVICES", "3")
    monkeypatch.setenv("TRITON_CACHE_DIR", "/tmp/triton_verify_cache_ab12cd")
    _install_fake_triton(monkeypatch, cann_home=str(cann_home))
    _install_fake_torch(monkeypatch)

    env = collect_environment(fresh=True, triton_impl_name="triton_ascend_impl")

    _assert_json_serializable(env)
    assert env.get("triton").get("version") == "3.2.0"
    assert len(env.get("triton").get("key_sha256")) == 64
    assert env.get("cache") == {
        "dir": "/tmp/triton_verify_cache_ab12cd",
        "fresh": True,
        "fresh_effective": True,
        "always_compile": False,
    }
    assert env.get("cann").get("info_sha256") == FAKE_CANN_HASH
    assert env.get("cann").get("version") == "8.5.1"
    assert env.get("cann").get("info_file").endswith("ascend_toolkit_install.info")
    assert "error" not in env.get("cann")
    assert env.get("torch").get("torch_npu_version") == "2.9.0.post1"
    assert env.get("device").get("ASCEND_RT_VISIBLE_DEVICES") == "3"
    assert env.get("device").get("npu_name") == "Ascend910B4"
    assert env.get("cmdline").get("non_compute") is False
    assert env.get("timestamp")


def test_cann_missing_records_error_without_raising(monkeypatch):
    """未 source set_env：cann.error 必须有值，其余分节照常采集。"""
    _install_fake_triton(monkeypatch, cann_home=None)
    _install_fake_torch(monkeypatch)

    env = collect_environment()

    _assert_json_serializable(env)
    assert "ASCEND_HOME_PATH" in env.get("cann").get("error")
    assert env.get("cann").get("ascend_home_path") is None
    assert env.get("triton").get("version") == "3.2.0"


def test_triton_missing_records_error_without_raising(monkeypatch):
    """triton 不可导入：triton 分节降级为 error，缓存目录退回默认值。"""
    monkeypatch.setitem(sys.modules, "triton", None)
    _install_fake_torch(monkeypatch, with_npu=False)

    env = collect_environment()

    _assert_json_serializable(env)
    assert env.get("triton").get("error")
    assert env.get("cache").get("dir").endswith(".triton/cache")
    assert env.get("device").get("npu_name") is None


def test_cache_manager_override_marks_isolation_ineffective(monkeypatch):
    """设置了自定义 cache manager 时，fresh_effective 必须记为 false。"""
    monkeypatch.setenv("TRITON_CACHE_MANAGER", "my.pkg:MyCacheManager")
    monkeypatch.setenv("TRITON_ALWAYS_COMPILE", "1")
    _install_fake_triton(monkeypatch, cann_home=None)
    _install_fake_torch(monkeypatch)

    env = collect_environment(fresh=True)

    _assert_json_serializable(env)
    assert env.get("cache").get("fresh") is True
    assert env.get("cache").get("fresh_effective") is False
    assert env.get("cache").get("always_compile") is True


def test_remote_cache_backend_also_marks_isolation_ineffective(monkeypatch):
    """TRITON_REMOTE_CACHE_BACKEND 同样会让 TRITON_CACHE_DIR 失效。"""
    monkeypatch.setenv("TRITON_REMOTE_CACHE_BACKEND", "my.pkg:MyRemoteBackend")
    _install_fake_triton(monkeypatch, cann_home=None)
    _install_fake_torch(monkeypatch)

    assert collect_environment(fresh=True).get("cache").get("fresh_effective") is False


def test_non_compute_flag_is_recorded(monkeypatch):
    """cmdline 分节如实记录本次的判定路径开关。"""
    _install_fake_triton(monkeypatch, cann_home=None)
    _install_fake_torch(monkeypatch, with_npu=False)

    env = collect_environment(triton_impl_name="triton_ascend_impl_v2", non_compute=True)

    assert env.get("cmdline") == {
        "triton_impl_name": "triton_ascend_impl_v2",
        "non_compute": True,
    }
