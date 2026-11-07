# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

r"""四个归约：`reduce_sum` `reduce_max` `reduce_min` `pair_reduce_sum`。

**写法：原始。** 前置样例是 [`vf_entry`](../vf_entry/test.py)。

## 一、`reduce_max` / `reduce_min` 的 lane 1 是极值下标，不是 0

**写这一例时是被测试失败逼出来的**：参考按「lane 0 是结果、其余清零」写，结果 `reduce_max`
在 lane 1 上给了 `8.83e-44`。那不是垃圾——`8.83e-44 / 1.4013e-45 = 63`，而最大值 64 在
`1..64` 里的下标正是 63。**lane 1 装的是 argmax，按整数位模式存在 FP32 寄存器里。**

| | lane 0 | lane 1 |
| --- | --- | --- |
| `reduce_sum` | 和 | 0 |
| `reduce_max` | 最大值 | **argmax 下标** |
| `reduce_min` | 最小值 | **argmin 下标** |
| `reduce_*(datablock=True)` | 第 1 组的结果 | 第 2 组的结果，**没有下标** |

要当下标看必须按位重解释：`tensor.view(torch.int32)`。直接当浮点读是个非规格化小数，
既不像下标也不像错误——**假定 lane 1 是 0 的代码会静默拿到脏数据**。

官方文档 `reduction/reduce_max.md` 的「返回值说明」写全了，还多两条：多个极值取**第一个**的
下标；有 NaN 时 `dst[0]` 是 NaN、`dst[1]` 是第一个 NaN 的下标。两条都在下面各有一条测试。

**`_vf_api.py` 的 docstring 说 "The remaining lanes are zeroed"，那句是错的**（`reduce_sum`
的是对的）。这是这一批第三次撞上「源码 docstring 与官方文档相反」——前两次是 `addc`
（反方向）和 `pto_assert`。判据已经写进 [skill constraints](../../../../SKILL.md#constraints-and-workspace-policy)。

## 二、`datablock=True` 就是组内归约——一个已记开放项的答案

设计文档 §八点四 把 ascriptor 的 `cgadd` / `cgmax` / `cgmin` 列成「仍然可能是真差异的只剩
一组」，并写了「做 `vf_reduce` 时查清」。**查清了：不用绕 `squeeze`，它是 `reduce_*` 的一个参数。**

组大小 **8 个 FP32（32 字节）**，实测：输入 1..64 得到
`[36, 100, 164, 228, 292, 356, 420, 484, 0, ...]`，正是 1..8 / 9..16 / … 各自求和。

这是**第三次**踩「把一个接口的一个参数当成缺失的接口」：`load_align` 的 `dist=`、
`store_align` 的模式、现在是 `reduce_*` 的 `datablock=`。

## 三、判据：能逐位比，而且理由不是「碰巧」

官方文档 `reduce_sum.md` 写明：**「指令内累加顺序采用二叉树累加方式，结果具有确定性」**。
所以归约顺序是**定的**，不是未定义行为。含义是：

- 本例操作数 1..64 全整数、和 ≤ 2080 < 2^24，任何顺序都精确 → `rtol=0, atol=0`。
- 换成非整数输入**仍然可以逐位比**，但参考必须也按二叉树两两合并；用 `numpy.sum` 的顺序
  累加会差最后一位。那时该改的是参考的求和方式，**不是悄悄放宽 `rtol`**。

## 四、`pair_reduce_sum` 不带 `datablock`

`dst[i] = src[2i] + src[2i+1]`，64 进 32 出，后半清零。它和 `datablock=True` 是两个不同的
分组概念：一个固定两两相加，一个按 32 字节分组。
"""

import numpy as np
import pytest
import torch

from . import reference as ref
from .kernel import DATABLOCK_LANES, N, reduce_by_block, reduce_whole

_WHOLE_NAMES = ["reduce_sum", "reduce_max", "reduce_min", "pair_reduce_sum"]


def _run_whole(device, a):
    outs = [torch.zeros((1, N), dtype=torch.float32, device=device) for _ in range(4)]
    reduce_whole(torch.from_numpy(a).to(device), *outs)
    torch.npu.synchronize()
    return outs


def _as_float(tensors):
    return [t.cpu().numpy() for t in tensors]


def _lane1_index(tensor):
    """按位把 lane 1 读成 int32——设备写进去的是整数下标。"""
    return int(tensor.cpu().view(torch.int32).numpy().reshape(-1)[1])


def test_the_whole_register_reductions_match_the_reference(device):
    """整寄存器的三个归约 + 成对归约，逐位比。参考含 lane 1 的下标。"""
    a = ref.make_operands(N)
    got = _as_float(_run_whole(device, a))
    want = ref.whole(a)
    for name, g, w in zip(_WHOLE_NAMES, got, want):
        np.testing.assert_allclose(g, w, rtol=0, atol=0, err_msg=f"vf.{name} 不一致")


def test_max_and_min_write_their_index_into_lane_one(device):
    """lane 1 是极值下标。换一组数据让下标变，证明它真的跟着数据走。"""
    a = ref.make_operands(N)  # max 在 63，min 在 0
    outs = _run_whole(device, a)
    assert _lane1_index(outs[1]) == int(a.reshape(-1).argmax()) == N - 1
    assert _lane1_index(outs[2]) == int(a.reshape(-1).argmin()) == 0
    assert _lane1_index(outs[0]) == 0, "reduce_sum 不该写 lane 1"

    moved = a.copy().reshape(-1)
    moved[7] = 1000.0  # max 挪到 7
    moved[40] = -5.0  # min 挪到 40
    moved = moved.reshape(1, N)
    outs = _run_whole(device, moved)
    assert _lane1_index(outs[1]) == 7, "argmax 下标没跟着数据走"
    assert _lane1_index(outs[2]) == 40, "argmin 下标没跟着数据走"


def test_a_tie_reports_the_first_index(device):
    """官方文档：多个极值时取**第一个**的下标。"""
    a = ref.make_tie_operands(N)
    flat = a.reshape(-1)
    outs = _run_whole(device, a)

    max_positions = np.flatnonzero(flat == flat.max())
    min_positions = np.flatnonzero(flat == flat.min())
    assert len(max_positions) >= 2 and len(min_positions) >= 2, (
        "操作数没有并列极值，这条是空的"
    )
    assert _lane1_index(outs[1]) == int(max_positions[0]), (
        f"并列最大值应报第一个下标 {max_positions[0]}，实际 {_lane1_index(outs[1])}"
    )
    assert _lane1_index(outs[2]) == int(min_positions[0]), (
        f"并列最小值应报第一个下标 {min_positions[0]}，实际 {_lane1_index(outs[2])}"
    )


def test_a_nan_input_reports_nan_and_the_first_nan_index(device):
    """官方文档：有 NaN 时 `dst[0]` 是 NaN、`dst[1]` 是第一个 NaN 的下标。"""
    a = ref.make_nan_operands(N)
    flat = a.reshape(-1)
    nan_positions = np.flatnonzero(np.isnan(flat))
    assert len(nan_positions) >= 2, "操作数没有多个 NaN，这条验不到「第一个」"

    got = _as_float(_run_whole(device, a))
    outs = _run_whole(device, a)
    for idx, name in ((1, "reduce_max"), (2, "reduce_min")):
        lane0 = got[idx].reshape(-1)[0]
        assert np.isnan(lane0), f"vf.{name} 的 dst[0] 应该是 NaN，实际 {lane0}"
        assert _lane1_index(outs[idx]) == int(nan_positions[0]), (
            f"vf.{name} 应报第一个 NaN 的下标 {nan_positions[0]}"
        )


def test_the_datablock_group_size_is_eight_fp32_lanes(device):
    """`datablock=True` 的组大小 = 8 个 FP32 = 32 字节，且组内归约**不写下标**。"""
    a = ref.make_operands(N)
    outs = [torch.zeros((1, N), dtype=torch.float32, device=device) for _ in range(3)]
    reduce_by_block(torch.from_numpy(a).to(device), *outs)
    torch.npu.synchronize()
    got = _as_float(outs)

    want = ref.by_block(a)
    for name, g, w in zip(["reduce_sum", "reduce_max", "reduce_min"], got, want):
        np.testing.assert_allclose(
            g, w, rtol=0, atol=0, err_msg=f"vf.{name}(datablock=True) 不一致"
        )

    flat = got[0].reshape(-1)
    expected_groups = N // DATABLOCK_LANES
    nonzero = np.nonzero(flat)[0]
    assert len(nonzero) == expected_groups, (
        f"应该有 {expected_groups} 个非零结果（每 {DATABLOCK_LANES} 个 lane 一组），"
        f"实际 {len(nonzero)} 个：{flat[:12]}"
    )
    assert flat[0] == np.float32(sum(range(1, DATABLOCK_LANES + 1))), (
        f"第一组应为 1..{DATABLOCK_LANES} 之和，实际 {flat[0]}"
    )


def test_datablock_and_whole_reduction_really_differ(device):
    """防空测：两种粒度必须给出不同结果，否则 `datablock=` 这个参数就没验到。"""
    a = ref.make_operands(N)
    whole = _as_float(_run_whole(device, a))
    outs = [torch.zeros((1, N), dtype=torch.float32, device=device) for _ in range(3)]
    reduce_by_block(torch.from_numpy(a).to(device), *outs)
    torch.npu.synchronize()
    block = _as_float(outs)
    assert not np.array_equal(whole[0], block[0]), (
        "datablock=True 和默认给出了相同结果——这个参数没起作用，或者测试写错了"
    )


def test_pair_reduce_sum_halves_the_valid_length(device):
    """`dst[i] = src[2i] + src[2i+1]`：64 进 32 出，后半清零。"""
    a = ref.make_operands(N)
    got = _as_float(_run_whole(device, a))
    pair = got[3].reshape(-1)
    flat = a.reshape(-1)
    assert pair[0] == flat[0] + flat[1], f"第一个应为 {flat[0]} + {flat[1]}"
    assert (pair[slice(N // 2, None)] == 0).all(), "后一半应该全零"


@pytest.mark.parametrize("name", _WHOLE_NAMES)
def test_every_api_in_this_example_is_actually_called(name):
    """不需要卡。防漏：4 个算子每个都得在 kernel 源码里出现。"""
    from pathlib import Path

    source = (Path(__file__).parent / "kernel.py").read_text()
    assert f"vf.{name}(" in source, f"kernel.py 里没有调用 vf.{name}()"
