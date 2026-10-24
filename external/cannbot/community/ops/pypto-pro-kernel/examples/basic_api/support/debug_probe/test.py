# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

r"""Debugging API behavior, log capture and subprocess isolation.

printf, dump_data and pto_assert run outside vector functions. A VF register
is exposed by storing it to a tile first; reference.py defines the copy and
multiplication results. A block guard limits dumps to one worker, and expected
log counts use the kernel's reported block count.

Each log-producing case runs in a fresh subprocess so its output can be read
after process exit. Child and parent must import the same pypto_pro. A failing
pto_assert logs and continues; an executed trap raises a device error at host
synchronization and is isolated from the parent context. The parent then runs
a healthy kernel to check recovery. Scalar GM writes by competing workers
are a diagnostic counterexample, with an observed count rather than fixed
worker ownership.
"""

import os
import re
import subprocess
import sys
from pathlib import Path

import pytest
import torch

import pypto_pro.language as pl
from pypto_pro.language import Vf as vf  # noqa: N813

from .kernel import (
    N,
    probe_printf,
    probe_dump_guard,
    probe_register_via_tile,
    probe_assert_passing,
    _assert_failing_kernel,
    _trap_kernel,
    probe_block_shape,
    probe_scalar_gm_writes,
    probe_trap_unreached,
)

from .reference import reference_copy, reference_register

_REPO_ROOT = Path(__file__).resolve().parents[4]
_ENTRY = "examples.basic_api.support.debug_probe.test"


def _run_child(case, expect_zero_exit=True):
    """在子进程里跑一个 case，返回 (stdout+stderr, block_num)。

    为什么是子进程：见模块 docstring §二。进程退出会把设备打印的缓冲刷干净，
    这是能对打印内容下断言的唯一可靠途径。
    """
    import pypto_pro

    # Prepend the package root and retain the selected framework installation.
    env = dict(os.environ)
    existing = env.get("PYTHONPATH", "")
    env["PYTHONPATH"] = (
        f"{_REPO_ROOT}{os.pathsep}{existing}" if existing else str(_REPO_ROOT)
    )
    child = subprocess.run(
        [sys.executable, "-c", f"from {_ENTRY} import run_case; run_case({case!r})"],
        cwd=_REPO_ROOT,
        capture_output=True,
        text=True,
        timeout=900,
        env=env,
    )
    text = child.stdout + child.stderr
    identity = re.search(r"PYPTO_PRO=(\S+)", text)
    assert identity, f"子进程没报工具链身份：{text[-800:]}"
    assert identity.group(1) == pypto_pro.__file__, (
        f"子进程用的是另一个 pypto_pro：\n  子 {identity.group(1)}\n  父 {pypto_pro.__file__}"
    )
    if expect_zero_exit:
        assert child.returncode == 0, (
            f"子进程 {case} 退出码 {child.returncode}：{text[-1200:]}"
        )
    # Host status may share a line with device output; match a substring.
    match = re.search(r"BLOCK_NUM=(\d+)", text)
    assert match, f"没拿到核数：{text[:600]!r}"
    block_num = int(match.group(1))
    assert block_num > 1, f"只启动了 {block_num} 个核，份数对比就没有意义了"
    return text, block_num


# ---------------------------------------------------------------------------
# printf
# ---------------------------------------------------------------------------


def test_printf_output_reaches_the_host_once_per_block(device):
    """框架自己的 ST 只断言 kernel 跑完，打印内容没人看。这里对内容断言。"""
    text, block_num = _run_child("printf")

    assert "HOST_CHECK_OK printf" in text, "kernel 没跑完，后面的断言不算"
    assert "PROBE_PRINTF int=-4242" in text, f"没看到 printf 的输出：{text[:600]!r}"
    assert "flt=2.500000" in text, "%f 不是六位小数？"
    assert text.count("PROBE_PRINTF") == block_num, (
        f"printf 应该每个核各打一份 = {block_num} 份，实际 {text.count('PROBE_PRINTF')} 份"
    )


# ---------------------------------------------------------------------------
# dump_data
# ---------------------------------------------------------------------------


def test_a_block_guard_collapses_the_dump_to_one_copy(device):
    """两个 dump 点在同一次运行里：守卫的 1 份，没守卫的 block_num 份。

    放同一次运行才排除得掉「这次只起了一个核」这个平凡解释。
    """
    text, block_num = _run_child("guard")

    assert "HOST_CHECK_OK guard" in text, "搬运结果和输入不一致，dump 的份数就不用看了"
    assert text.count("[flag] GUARDED") == 1, (
        f"守卫过的 dump 应该只有 1 份，实际 {text.count('[flag] GUARDED')} 份"
    )
    assert text.count("[flag] UNGUARDED") == block_num, (
        f"没守卫的 dump 应该有 {block_num} 份，实际 {text.count('[flag] UNGUARDED')} 份"
    )


def test_a_register_is_seen_by_storing_it_to_a_tile_first(device):
    """vf 体里打印不了，那就 `store_align` 落 Tile 再 dump。

    断言 dump 出来的是**寄存器里算过的值**（×100），不只是「有输出」。
    """
    text, _ = _run_child("register")

    assert "HOST_CHECK_OK register" in text, "×100 的结果本身就不对，dump 内容不用看了"
    assert "[flag] REGISTER" in text, f"没看到 REGISTER 这个 dump 点：{text[:600]!r}"
    # 输入 0,1,2,3 → ×100 → 0,100,200,300。dump 按六位小数打。
    for expected in ("100.000000", "200.000000", "300.000000"):
        assert expected in text, f"dump 里没有 {expected}"


# ---------------------------------------------------------------------------
# 两条负面用例：打印进不了 vf 体
# ---------------------------------------------------------------------------


def _tile_type():
    return pl.TileType(shape=[1, N], dtype=pl.DT_FP32, target_memory=pl.MemorySpace.Vec)


def _kernel_calling(body):
    @pl.jit(auto_mutex=True)
    def k(a: pl.Tensor[[1, N], pl.DT_FP32], out: pl.Tensor[[1, N], pl.DT_FP32]):
        g = pl.make_tile_group(type=_tile_type(), addrs=0x0, mutex_ids=[0])
        h = pl.make_tile_group(type=_tile_type(), addrs=0x400, mutex_ids=[1])
        with pl.section_vector():
            t, out_tile = g.current(), h.current()
            pl.load(t, a, [0, 0])
            body(t, out_tile)
            pl.store(out, out_tile, [0, 0])

    return k


def test_printf_inside_a_vector_function_is_refused(device):
    """Reject printf in VF when the enclosing kernel is first parsed."""

    @pl.vector_function
    def body(src, dst):
        preg = vf.create_mask(pattern=pl.MaskPattern.ALL, dtype=pl.DT_FP32)
        r = vf.load_align(src, 0)
        pl.printf("never\n")
        vf.store_align(dst, r, preg, 0)

    a = torch.zeros(1, N, device=device, dtype=torch.float32)
    out = torch.zeros_like(a)
    with pytest.raises(
        Exception, match=r"pl\.printf.*not supported inside @pl\.vector_function"
    ):
        _kernel_calling(body)(a, out)


def test_dump_data_inside_a_vector_function_is_refused(device):
    """Reject dump_data inside VF."""

    @pl.vector_function
    def body(src, dst):
        preg = vf.create_mask(pattern=pl.MaskPattern.ALL, dtype=pl.DT_FP32)
        r = vf.load_align(src, 0)
        vf.store_align(dst, r, preg, 0)
        pl.dump_data(dst)

    a = torch.zeros(1, N, device=device, dtype=torch.float32)
    out = torch.zeros_like(a)
    with pytest.raises(
        Exception, match=r"pl\.dump_data.*not supported inside @pl\.vector_function"
    ):
        _kernel_calling(body)(a, out)


# ---------------------------------------------------------------------------
# 别用标量写 GM 探每核状态
# ---------------------------------------------------------------------------


def test_scalar_writes_to_gm_lose_most_blocks(device):
    """钉住「会丢」这个事实，不钉丢多少——实测 5~7 / 56，每次不同。

    同一批测试里 `test_a_block_guard_collapses_the_dump_to_one_copy` 的 UNGUARDED
    打满了 56 份，证明核确实都在跑，所以这里少掉的是写，不是核。
    """
    shape = torch.zeros(2, device=device, dtype=torch.int32)
    probe_block_shape(shape)
    torch.npu.synchronize()
    block_num = shape.tolist()[0]

    out = torch.zeros(128, device=device, dtype=torch.int32)
    probe_scalar_gm_writes(out)
    torch.npu.synchronize()

    values = out.tolist()
    landed = [i for i, v in enumerate(values) if v != 0]
    assert landed, "一个都没落地，那是另一个问题"
    assert len(landed) < block_num, (
        f"{len(landed)} 个槽位落地、共 {block_num} 个核——这条过了说明写不再丢了，"
        "本例这条结论要重测"
    )
    for i in landed:
        assert values[i] == i + 1000, f"槽位 {i} 落地的值不对：{values[i]}"


# ---------------------------------------------------------------------------
# pto_assert
# ---------------------------------------------------------------------------


def test_a_passing_assert_does_nothing(device):
    """三种写法（裸条件 / 带消息 / 带格式化消息）条件成立时都静默通过。"""
    text, _ = _run_child("assert_pass")
    assert "HOST_CHECK_OK assert_pass" in text, (
        f"条件成立的 assert 不该有任何影响：{text[-800:]!r}"
    )


def test_a_failing_assert_only_logs_and_does_not_abort(device):
    """A false pto_assert logs its condition and message, then continues."""
    text, block_num = _run_child("assert_fail")

    assert "ASSERT_DID_NOT_RAISE out=[9]" in text, (
        f"assert 失败之后 `out[0] = 9` 应该照样执行、host 侧应该无异常：{text[-1000:]!r}"
    )
    assert "PROBE_ASSERT_MSG offset=3" in text, (
        f"assert 的格式化消息没打出来（`%d` 应该填 3）：{text[-1000:]!r}"
    )
    # 设备打的是**条件的源码文本**，不是它的值
    assert text.count("Assertion failed: offset == 12345") == block_num, (
        f"失败的 assert 应该每个核各打一份 = {block_num} 份，"
        f"实际 {text.count('Assertion failed: offset == 12345')} 份"
    )


# ---------------------------------------------------------------------------
# trap
# ---------------------------------------------------------------------------


def test_an_unreached_trap_leaves_the_kernel_healthy(device):
    """`trap` 编在永不成立的分支里：kernel 正常结束。这条可以留在本进程。"""
    out = torch.zeros(1, device=device, dtype=torch.int32)
    probe_trap_unreached(out)
    torch.npu.synchronize()
    assert out.tolist()[0] == 7


def test_trap_aborts_the_kernel_and_the_card_recovers(device):
    """真触发 `trap`，然后验本进程的卡还好着。

    这也是 `pl.trap` 在整个 `pypto` 仓 0 调用的原因之一：它没法放进普通测试套。
    """
    text, _ = _run_child("trap")

    assert "TRAP_RAISED" in text, f"trap 没有让 host 侧报错：{text[-1000:]!r}"
    assert "device error" in text, f"报错不是设备错误？{text[-1000:]!r}"

    # 验卡：子进程里 trap 过之后，本进程仍然能正常跑 kernel
    out = torch.zeros(1, device=device, dtype=torch.int32)
    probe_trap_unreached(out)
    torch.npu.synchronize()
    assert out.tolist()[0] == 7, "trap 之后本进程的卡状态异常"


def run_case(case: str) -> None:
    """跑一个 probe kernel，把设备打印和 host 侧校验结果一起打到 stdout。

    由 test.py 以 `python -c "...run_case('printf')"` 的形式在子进程里调。
    """
    import numpy as np
    import torch
    import torch_npu  # noqa: F401

    # 报回工具链身份：子进程如果 import 到另一个 pypto_pro，这一格的结论就不是
    # 这一版的结论。父进程会和自己的比（见 test.py 的 `_run_child`）。
    import pypto_pro

    print(f"PYPTO_PRO={pypto_pro.__file__}")

    device = f"npu:{int(os.environ.get('TILE_FWK_DEVICE_ID', 0))}"
    torch.npu.set_device(device)

    i32 = torch.int32
    shape = torch.zeros(2, device=device, dtype=i32)
    probe_block_shape(shape)
    torch.npu.synchronize()
    print(f"BLOCK_NUM={shape.tolist()[0]} SUBBLOCK_NUM={shape.tolist()[1]}")

    if case == "printf":
        out = torch.zeros(1, device=device, dtype=i32)
        probe_printf(out, -4242, 2.5)
        torch.npu.synchronize()
        assert out.tolist()[0] == 1
        print("HOST_CHECK_OK printf")

    elif case == "guard":
        a = torch.arange(N, device=device, dtype=torch.float32).reshape(1, N)
        out = torch.zeros_like(a)
        probe_dump_guard(a, out)
        torch.npu.synchronize()
        np.testing.assert_allclose(
            out.cpu().numpy(), reference_copy(a.cpu()).numpy(), rtol=0, atol=0
        )
        print("HOST_CHECK_OK guard")

    elif case == "register":
        a = torch.arange(N, device=device, dtype=torch.float32).reshape(1, N)
        out = torch.zeros_like(a)
        probe_register_via_tile(a, out)
        torch.npu.synchronize()
        np.testing.assert_allclose(
            out.cpu().numpy(), reference_register(a.cpu()).numpy(), rtol=0, atol=0
        )
        print("HOST_CHECK_OK register")

    elif case == "assert_pass":
        out = torch.zeros(1, device=device, dtype=i32)
        probe_assert_passing(out, True, 7)
        torch.npu.synchronize()
        assert out.tolist()[0] == 1
        print("HOST_CHECK_OK assert_pass")

    elif case == "assert_fail":
        # The subprocess captures the device log after exit.
        out = torch.zeros(1, device=device, dtype=i32)
        try:
            _assert_failing_kernel(out, 3)
            torch.npu.synchronize()
        except RuntimeError as exc:
            print(
                f"ASSERT_RAISED {type(exc).__name__}: {str(exc).strip().splitlines()[0]}"
            )
            return
        print(f"ASSERT_DID_NOT_RAISE out={out.tolist()}")

    elif case == "trap":
        # **这一支会弄坏本进程的设备上下文**，所以只能在子进程里跑：测试套是一个
        # pytest 进程跑完全部样例的，在里面 trap 会让后面所有测试失败。
        # 实测卡在新进程里恢复正常，不需要复位。
        out = torch.zeros(1, device=device, dtype=i32)
        try:
            _trap_kernel(out)
            torch.npu.synchronize()  # ← 错误在这里抛，不在上一行
        except RuntimeError as exc:
            print(
                f"TRAP_RAISED {type(exc).__name__}: {str(exc).strip().splitlines()[0]}"
            )
            return
        print(f"TRAP_DID_NOT_RAISE out={out.tolist()}")

    else:
        raise ValueError(f"未知的 case：{case}")
