# ----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ----------------------------------------------------------------------------------------------------------

"""verify.py --fresh-cache / --keep-cache 的单元测试。

用一个只打印自身所见 TRITON_CACHE_DIR 的假子进程脚本替代真正的验证子进程，
因此全部用例只需 CPU，无需 NPU、无需 triton 编译。

覆盖：不传 flag 时环境原样继承、传 flag 时注入全新空目录、CANN 变量不被冲掉、
成功即清理、失败保留并打印路径、--keep-cache 无条件保留、
自定义 cache manager 已设置时告警。
"""
import argparse
import logging
import os
import sys
import tempfile

import pytest

import verify
from _common_utils import CACHE_OVERRIDE_ENV_VARS
from verify import _build_subprocess_cmd as build_subprocess_cmd
from verify import _run_main_process as run_main_process

# 假子进程：把自己看到的关键环境变量写进 argv[1] 指定的文件，然后按 argv[2] 退出。
FAKE_CHILD_SOURCE = '''import os
import sys

report_path = sys.argv[1]
observed = [
    "TRITON_CACHE_DIR=" + os.environ.get("TRITON_CACHE_DIR", ""),
    "ASCEND_HOME_PATH=" + os.environ.get("ASCEND_HOME_PATH", ""),
]
with open(report_path, "w", encoding="utf-8") as handle:
    handle.write("\\n".join(observed))
sys.exit(int(sys.argv[2]))
'''

GLOBAL_CACHE_MARKER = "pretend-this-is-a-cached-kernel"


@pytest.fixture(autouse=True)
def restore_logger():
    """让 caplog 能抓到 verify 的日志：测试期间保持 propagate 打开。"""
    saved = verify.logger.propagate
    verify.logger.propagate = True
    yield
    verify.logger.propagate = saved


@pytest.fixture
def child_script(tmp_path):
    """落盘假子进程脚本，返回其绝对路径。"""
    path = tmp_path / "fake_child.py"
    path.write_text(FAKE_CHILD_SOURCE, encoding="utf-8")
    return str(path)


@pytest.fixture
def global_cache(tmp_path, monkeypatch):
    """伪造"用户全局缓存目录"，用于断言它没有被读写。"""
    cache_dir = tmp_path / "global_triton_cache"
    cache_dir.mkdir()
    (cache_dir / "entry.json").write_text(GLOBAL_CACHE_MARKER, encoding="utf-8")
    monkeypatch.setenv("TRITON_CACHE_DIR", str(cache_dir))
    return cache_dir


@pytest.fixture(autouse=True)
def isolated_tmpdir(tmp_path, monkeypatch):
    """把 mkdtemp 的落点固定到本用例的 tmp_path，避免污染系统 TMPDIR。

    tempfile 在首次调用时就把 TMPDIR 解析并缓存进 tempfile.tempdir，
    因此只改环境变量不生效，必须直接改这个模块级变量。
    """
    workdir = tmp_path / "tmpdir"
    workdir.mkdir()
    monkeypatch.setenv("TMPDIR", str(workdir))
    monkeypatch.setattr(tempfile, "tempdir", str(workdir))
    for name in CACHE_OVERRIDE_ENV_VARS:
        monkeypatch.delenv(name, raising=False)
    return workdir


def _make_args(fresh_cache=False, keep_cache=False, timeout=60):
    """构造 _run_main_process 需要的最小参数集合。"""
    return argparse.Namespace(
        fresh_cache=fresh_cache, keep_cache=keep_cache, timeout=timeout,
    )


def _run(child_script, tmp_path, exit_code=0, fresh_cache=False, keep_cache=False):
    """跑一次主进程流程，返回 (退出码, 子进程观察到的环境变量字典)。"""
    report = tmp_path / "observed.txt"
    cmd = [sys.executable, child_script, str(report), str(exit_code)]
    returncode = run_main_process(_make_args(fresh_cache, keep_cache), cmd)
    observed = {}
    for line in report.read_text(encoding="utf-8").splitlines():
        key, _, value = line.partition("=")
        observed[key] = value
    return returncode, observed


def test_without_fresh_cache_env_is_inherited_as_is(child_script, tmp_path, global_cache,
                                                    isolated_tmpdir):
    """不传 --fresh-cache：子进程沿用外部 TRITON_CACHE_DIR，且不创建任何临时目录。"""
    returncode, observed = _run(child_script, tmp_path)

    assert returncode == 0
    assert observed.get("TRITON_CACHE_DIR") == str(global_cache)
    assert list(isolated_tmpdir.iterdir()) == []


def test_fresh_cache_injects_new_empty_dir(child_script, tmp_path, global_cache,
                                           isolated_tmpdir):
    """传 --fresh-cache：子进程拿到一个新建的空目录，全局缓存原封不动。"""
    before = sorted(p.name for p in global_cache.iterdir())
    mtime_before = global_cache.stat().st_mtime

    returncode, observed = _run(child_script, tmp_path, fresh_cache=True, keep_cache=True)

    injected = observed.get("TRITON_CACHE_DIR")
    assert returncode == 0
    assert injected != str(global_cache)
    assert os.path.basename(injected).startswith("triton_verify_cache_")
    assert os.path.dirname(injected) == str(isolated_tmpdir)
    assert os.listdir(injected) == []
    assert sorted(p.name for p in global_cache.iterdir()) == before
    assert global_cache.stat().st_mtime == mtime_before


def test_fresh_cache_keeps_cann_env_vars(child_script, tmp_path, monkeypatch):
    """注入 TRITON_CACHE_DIR 不能把 ASCEND_HOME_PATH 等 CANN 变量冲掉。"""
    monkeypatch.setenv("ASCEND_HOME_PATH", "/usr/local/Ascend/ascend-toolkit/latest")

    _, observed = _run(child_script, tmp_path, fresh_cache=True, keep_cache=True)

    assert observed.get("ASCEND_HOME_PATH") == "/usr/local/Ascend/ascend-toolkit/latest"


def test_success_removes_temp_dir(child_script, tmp_path):
    """子进程退出码 0 → 临时缓存目录被删除。"""
    _, observed = _run(child_script, tmp_path, exit_code=0, fresh_cache=True)

    assert not os.path.exists(observed.get("TRITON_CACHE_DIR"))


def test_failure_keeps_temp_dir_and_logs_path(child_script, tmp_path, caplog):
    """子进程失败 → 目录保留，且绝对路径出现在 WARNING（走 stderr）里。"""
    with caplog.at_level(logging.WARNING, logger=verify.logger.name):
        returncode, observed = _run(child_script, tmp_path, exit_code=1, fresh_cache=True)

    kept = observed.get("TRITON_CACHE_DIR")
    assert returncode == 1
    assert os.path.isdir(kept)
    assert any(kept in record.getMessage() for record in caplog.records)


def test_keep_cache_keeps_dir_on_success(child_script, tmp_path, caplog):
    """--keep-cache：即便子进程成功也保留目录并打印路径。"""
    with caplog.at_level(logging.WARNING, logger=verify.logger.name):
        returncode, observed = _run(
            child_script, tmp_path, exit_code=0, fresh_cache=True, keep_cache=True,
        )

    kept = observed.get("TRITON_CACHE_DIR")
    assert returncode == 0
    assert os.path.isdir(kept)
    assert any(kept in record.getMessage() for record in caplog.records)


def test_cache_manager_override_triggers_warning(child_script, tmp_path, monkeypatch, caplog):
    """已设置 TRITON_CACHE_MANAGER 时必须告警，不能静默假装已隔离。"""
    monkeypatch.setenv("TRITON_CACHE_MANAGER", "my.pkg:MyCacheManager")

    with caplog.at_level(logging.WARNING, logger=verify.logger.name):
        _run(child_script, tmp_path, fresh_cache=True)

    messages = [record.getMessage() for record in caplog.records]
    assert any("TRITON_CACHE_MANAGER" in message for message in messages)
    assert any("fresh_effective=false" in message for message in messages)


def test_temp_dir_is_removed_even_when_subprocess_raises(tmp_path, monkeypatch, caplog):
    """子进程调用本身抛异常时，临时目录不能无人认领——路径必须打到 stderr。"""
    def _boom(cmd, timeout, env):
        raise OSError("popen exploded")

    monkeypatch.setattr(verify, "_run_verify_subprocess", _boom)

    with caplog.at_level(logging.WARNING, logger=verify.logger.name):
        with pytest.raises(OSError):
            run_main_process(_make_args(fresh_cache=True), ["ignored"])

    kept = [r.getMessage() for r in caplog.records if "已保留" in r.getMessage()]
    assert len(kept) == 1
    assert os.path.isdir(kept[0].split(": ", 1)[1])


def test_fresh_cache_flag_is_passed_through_to_subprocess(tmp_path):
    """--fresh-cache 需透传给子进程用于标记结果 JSON；--keep-cache 不透传。"""
    args = argparse.Namespace(
        op_name="softmax", triton_impl_name="triton_ascend_impl", output=None,
        non_compute=False, fresh_cache=True, keep_cache=True,
    )

    cmd = build_subprocess_cmd(args, str(tmp_path))

    assert "--fresh-cache" in cmd
    assert "--keep-cache" not in cmd
    assert "--subprocess" in cmd


def test_no_fresh_cache_flag_when_disabled(tmp_path):
    """不传 --fresh-cache 时命令行里不应出现该参数。"""
    args = argparse.Namespace(
        op_name="softmax", triton_impl_name="triton_ascend_impl", output=None,
        non_compute=False, fresh_cache=False, keep_cache=False,
    )

    assert "--fresh-cache" not in build_subprocess_cmd(args, str(tmp_path))
