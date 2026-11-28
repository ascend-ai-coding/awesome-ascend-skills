# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

"""kernel 入口与流水的独立数值和接口验证。

逐块缩放使用独立 NumPy 参考；带步长的循环必须保留未访问块的哨兵值。
四种 preload 分别通过同一参考，并要求输出逐位相同。可解析的 MemorySpace
闭包别名须保持同样的数值结果。非法 range、preload 与弃用装饰器按当前接口契约验证。
"""

import warnings

import numpy as np
import pytest
import torch

import pypto_pro as pp
import pypto_pro.language as pl
from pypto_pro._errors import InvalidArgument, NotSupported, OutOfRange

from .kernel import (
    FULL_M,
    NT,
    PRELOADS,
    STEP,
    TM,
    TN,
    cube_section_moves_through_l1,
    loop_over_blocks,
    loop_with_a_step,
    loop_with_python_range,
    make_pipelined,
    make_pipelined_with_an_aliased_memory_space,
)
from .reference import (
    SENTINEL,
    all_equal,
    make_input,
    make_input_fp16,
    scaled_by_block_index,
)

PIPELINE_BASE = 2
"""流水 stage 里 `scale = ki + 2`，照 ST。参考用同一个 base。"""


def _run_loop(kernel, device, sentinel=False):
    a = make_input(FULL_M, TN)
    if sentinel:
        out = torch.full(
            (FULL_M, TN), float(SENTINEL), dtype=torch.float32, device=device
        )
    else:
        out = torch.zeros((FULL_M, TN), dtype=torch.float32, device=device)
    kernel[None, 1](torch.from_numpy(a).to(device), out)
    torch.npu.synchronize()
    return a, out.cpu().numpy()


def _run_pipelined(preload, device):
    a = make_input_fp16(FULL_M, TN)
    out = torch.zeros((FULL_M, TN), dtype=torch.float16, device=device)
    make_pipelined(preload)[None, 1](torch.from_numpy(a).to(device), out)
    torch.npu.synchronize()
    return a, out.cpu().numpy()


def _parse(kernel):
    kernel.to_kernel_def().parse_target_program(pp.ir.SectionKind.Vector)


# ---------------------------------------------------------------------------
# 一、pl.range
# ---------------------------------------------------------------------------


def test_the_loop_visits_every_block_in_order(device):
    """验证 1。第 k 块乘 `k + 1`。

    **乘轮次号而不是常数**：常数的话跑错次数、跑错顺序都看不出来。
    """
    a, got = _run_loop(loop_over_blocks, device)
    np.testing.assert_allclose(got, scaled_by_block_index(a, TM, NT), rtol=1e-6, atol=0)


def test_a_step_skips_the_odd_blocks(device):
    """验证 2。`pl.range(0, 4, 2)` 只碰 0、2 号块，1、3 号仍是哨兵。

    **哨兵是这条判据的全部**：不填哨兵只能证「碰到的对」，证不到「没碰的没动」。
    """
    a = make_input(FULL_M, TN)
    out = torch.full((FULL_M, TN), float(SENTINEL), dtype=torch.float32, device=device)
    loop_with_a_step[None, 1](torch.from_numpy(a).to(device), out)
    torch.npu.synchronize()
    got = out.cpu().numpy()

    untouched = np.full_like(a, SENTINEL)
    expected = scaled_by_block_index(a, TM, NT, step=STEP, untouched=untouched)
    np.testing.assert_allclose(got, expected, rtol=1e-6, atol=0)
    for block in range(1, NT, STEP):
        segment = got[slice(block * TM, (block + 1) * TM)]
        assert (segment == SENTINEL).all(), (
            f"第 {block} 块被动了：{segment[0][:4].tolist()}"
        )


def test_a_python_range_is_refused():
    """验证 3。内置 `range` 被拒，消息直接点出该用什么。

    这个 kernel 和 ``loop_over_blocks`` **只差一个词**，所以它验的确实是 `range` 本身。
    """
    with pytest.raises(NotSupported) as raised:
        _parse(loop_with_python_range)
    assert "must use pl.range()" in str(raised.value), str(raised.value)


# ---------------------------------------------------------------------------
# 二、section_cube
# ---------------------------------------------------------------------------


def test_a_mixed_kernel_runs(device):
    """验证 5。cube 段搬一趟 L1，vector 段把原值送出去。

    cube 段那一趟的结果不出去——**它的作用是让这个 kernel 成为混合 kernel**,
    而那件事的可观测后果在 ``support/block_grid`` 验证 4（`subblock_num` 变成 2）。
    """
    a = make_input_fp16(TM, TN)
    out = torch.zeros((TM, TN), dtype=torch.float16, device=device)
    cube_section_moves_through_l1[None, 1](torch.from_numpy(a).to(device), out)
    torch.npu.synchronize()
    np.testing.assert_array_equal(out.cpu().numpy(), a)


# ---------------------------------------------------------------------------
# 三、pl.pipeline
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("preload", PRELOADS)
def test_each_preload_gives_the_right_answer(preload, device):
    """验证 4 前半。每个 `preload` 各自对上参考。

    参考里没有任何流水的概念——**一份参考对四个 `preload`**，这正是要验的点。
    """
    a, got = _run_pipelined(preload, device)
    expected = scaled_by_block_index(a, TM, NT, base=PIPELINE_BASE)
    np.testing.assert_allclose(
        got.astype(np.float32), expected.astype(np.float32), rtol=1e-3, atol=1e-3
    )


def test_every_preload_gives_the_same_bits(device):
    """验证 4 后半，**这一例最有分量的一条**。四个 `preload` 的结果两两逐位相同。

    它验的不是某个 API 的返回值，而是**流水变换不改变语义**——那是整个 `pl.pipeline`
    存在的前提。逐位而不是 `allclose`：`preload` 改的是指令排布，不是算术。

    `preload=0` 在里面是因为 `config.py` 的 docstring 说它是「先确认串行版对」的起点;
    `preload=3` 在里面是因为它大于 `NT/2`，填充和排空占掉大半个循环。
    """
    results = [_run_pipelined(preload, device)[1] for preload in PRELOADS]
    assert all_equal(results), (
        "不同 preload 给出不同的结果——流水变换改变了语义。"
        + "；".join(
            f"preload={p} 首元素 {r.ravel()[0]}" for p, r in zip(PRELOADS, results)
        )
    )


def test_an_aliased_memory_space_keeps_the_pipeline_result(device):
    """可解析的 MemorySpace 闭包别名与字面量声明使用相同的流水契约。"""
    a = make_input_fp16(FULL_M, TN)
    out = torch.zeros((FULL_M, TN), dtype=torch.float16, device=device)
    make_pipelined_with_an_aliased_memory_space()[None, 1](
        torch.from_numpy(a).to(device), out
    )
    torch.npu.synchronize()
    expected = scaled_by_block_index(a, TM, NT, base=PIPELINE_BASE)
    np.testing.assert_allclose(
        out.cpu().numpy().astype(np.float32),
        expected.astype(np.float32),
        rtol=1e-3,
        atol=1e-3,
    )


@pytest.mark.parametrize(
    ("bad", "error"),
    [(-1, OutOfRange), (1.5, ValueError), (True, ValueError), ((), InvalidArgument)],
)
def test_an_invalid_preload_is_refused(bad, error):
    """验证 7。四种非法 `preload`，每种的异常类型都不同。

    `True` 单独一条：它 `isinstance(True, int)` 为真，**所以不特判就会被当成 1 放过**。
    框架特判了（消息里点名 bool），这条钉住它。
    """
    with pytest.raises(error) as raised:
        pl.pipeline.PipelineConfig(preload=bad)
    assert "preload" in str(raised.value), str(raised.value)


# ---------------------------------------------------------------------------
# 四、@pl.inline
# ---------------------------------------------------------------------------


def test_inline_is_deprecated_and_its_replacement_does_not_exist():
    """验证 8。`@pl.inline` 报 `DeprecationWarning`，而消息指的 `pl.fn` 不存在。

    两半分开断言：**警告在**是框架的现状，**`pl.fn` 不在**是那条消息的缺陷。
    后者按 [skill constraints](../../../../SKILL.md#constraints-and-workspace-policy) 落在 `pypto/`（已改消息并补 UT），所以**这条判据会在那个改动
    落下来之后红**——那时候该改的是这里，不是代码。见 §六。
    """
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")

        @pl.inline
        def helper(x):
            return x

    assert len(caught) == 1, [str(w.message) for w in caught]
    assert caught[0].category is DeprecationWarning
    assert "@pl.inline is deprecated" in str(caught[0].message)
    assert helper(3) == 3, "废弃的装饰器应该原样返回函数，它是个 no-op"

    assert not hasattr(pl, "fn"), (
        "`pl.fn` 现在存在了——那条废弃消息不再是缺陷，§六 和这条判据都该更新"
    )


def test_pl_pipeline_is_a_module_not_a_callable():
    """§五 的判据。`pl.pipeline` 是模块，所以按 `pl.pipeline(` 搜必然是 0 次。

    **这条钉住的是「那个 0 不是证据」**，不是 API 的行为。它不上板,
    留在这儿是因为一条数错的结论比没有结论更危险。
    """
    assert not callable(pl.pipeline), "pl.pipeline 变成可调用的了？那 §五 的数法要重看"
    assert hasattr(pl.pipeline, "stage")
    assert hasattr(pl.pipeline, "PipelineConfig")
