# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

r"""四个 kernel 作用域 mask 状态 API：`pl.set_vec_mask` `pl.set_mask_norm`
`pl.set_mask_count` `pl.reset_mask`。

**写法：原始。** 前置样例是 [`vf_predicate`](../vf_predicate/test.py)（掩码怎么看出来）。

## 一、这四个为什么归进 vf

它们不是 vf 算子——写在 `pl.section_vector()` 里，不在 `@pl.vector_function` 里。但它们设的
状态**跨 `@vf` 边界可见**：vf 体内 `vf.get_mask_spr()` 读到的就是这里设的值。框架自己的
`mask_state_prefix` 样例测的就是这件事（*"Mask state read across a @vf boundary"*）。

`test_the_mask_state_survives_across_two_vf_calls` 验：设一次、连着两次 vf 调用都看得到。
反过来说——**不复位就会漏给后面的代码**，这是 `pl.reset_mask()` 存在的理由。

## 二、没有官方 API 文档，所以每条都是实测

这四个在 `docs/zh/api/pro_api/` 下**没有自己的页面**，只有 `get_mask_spr.md` 顺带提到
`set_vec_mask`。唯一的第一手来源是
`pypto/python/tests/st/pypto_pro/frontend/element_wise/test_vector_mask.py`。
下面每条行为都是 2026-09-27 在 A5 上量的。

## 三、两种模式，`set_vec_mask` 的含义不同——而 `get_mask_spr` 只懂一种

| 模式 | 切换 | `set_vec_mask(high, low)` 的 `low` |
| --- | --- | --- |
| **norm**（位图，默认） | `pl.set_mask_norm()` | 第 i 位控制第 i 个元素 |
| **count**（计数） | `pl.set_mask_count()` | **有效元素个数** |

**坑：`vf.get_mask_spr()` 只按位图读。** 计数模式下设计数 16，vf 体里读到的是
「只有 lane 4 有效」——因为 `16 = 0b10000`，第 4 位是 1。**不报错，静默拿到错掩码。**

| 设了什么 | tile 级算子理解为 | `get_mask_spr` 读成 |
| --- | --- | --- |
| `set_mask_count(); set_vec_mask(0, 16)` | 前 16 个元素有效 | **只有第 4 个元素有效** |

所以：**计数模式是给 tile 级向量算子用的**（`pl.add` 那类，ST 里就是这么配的）；
vf 路径要按个数掩码，用 **`vf.update_mask(count, dtype=)`**（见
[`vf_entry`](../vf_entry/test.py) 的尾块处理）。

`test_count_mode_is_misread_by_get_mask_spr` 钉住这个错误行为——钉的是「现在是这样」。
如果哪天 `get_mask_spr` 开始理解计数模式，这条会失败，那时该改成断言「前 16 个有效」。

## 四、判据

`rtol=0, atol=0`。掩码经 `select` 变成 1.0 / 0.0，是位不是数，任何一位不同都是错。

**每个 kernel 都在收尾 `pl.reset_mask()`。** 不是仪式——mask 是 kernel 作用域状态，
`state_persists_across_vf_calls` 那条正好证明它会漏出去。
"""

import numpy as np
import pytest
import torch

from . import reference as ref
from .kernel import (
    N,
    count_mode_seen_from_vf,
    norm_bitmap,
    reset_restores_all_lanes,
    state_persists_across_vf_calls,
)

_ALL_APIS = ["set_vec_mask", "set_mask_norm", "set_mask_count", "reset_mask"]


def _out(device, count=1):
    return [
        torch.zeros((1, N), dtype=torch.float32, device=device) for _ in range(count)
    ]


def test_norm_mode_takes_low_as_a_bitmap(device):
    """norm 模式：`set_vec_mask(0, 0x0F0F0F0F)` 的第 i 位控制第 i 个 lane。"""
    outs = _out(device)
    norm_bitmap(*outs)
    torch.npu.synchronize()
    got = outs[0].cpu().numpy()

    want = ref.bits_to_lanes(ref.BITMAP, N)
    np.testing.assert_allclose(
        got, want, rtol=0, atol=0, err_msg="按位图读出来的掩码与设进去的不一致"
    )

    flat = got.reshape(-1)
    assert flat[:4].tolist() == [1, 1, 1, 1], "低 4 位应该有效"
    assert flat[4:8].tolist() == [0, 0, 0, 0], "接着 4 位应该无效"
    assert int(flat.sum()) == 16, (
        f"0x0F0F0F0F 在 64 个 lane 里应有 16 个有效，实际 {int(flat.sum())}"
    )


def test_count_mode_is_misread_by_get_mask_spr(device):
    """**计数模式下 `get_mask_spr` 把计数当位图读。** 钉住现状，不是期望行为。

    设 `set_mask_count(); set_vec_mask(0, 16)`：tile 级算子理解成「前 16 个有效」，
    而 vf 体里读到的是「只有 lane 4 有效」（`16 = 0b10000`）。**不报错。**

    如果这条失败了，说明 `get_mask_spr` 开始理解计数模式——那是行为变化，
    该把断言改成「前 16 个有效」，不要删掉这条测试。
    """
    outs = _out(device)
    count_mode_seen_from_vf(*outs)
    torch.npu.synchronize()
    got = outs[0].cpu().numpy()
    flat = got.reshape(-1)

    np.testing.assert_allclose(
        got,
        ref.count_mode_misread(N),
        rtol=0,
        atol=0,
        err_msg="计数模式下读到的掩码既不是位图也不是计数语义",
    )

    assert int(flat.sum()) == 1, (
        f"应该只有 1 个 lane 有效（把 16 当位图读），实际 {int(flat.sum())} 个"
    )
    assert flat[4] == 1, "有效的那个应该是 lane 4（16 的第 4 位）"
    assert int(flat[:16].sum()) != 16, (
        "前 16 个全有效——说明 get_mask_spr 开始理解计数模式了，改这条测试的断言"
    )


def test_reset_mask_restores_all_lanes(device):
    """先设窄掩码再 `reset_mask()`，读出来应该全有效。"""
    outs = _out(device)
    reset_restores_all_lanes(*outs)
    torch.npu.synchronize()
    got = outs[0].cpu().numpy()

    np.testing.assert_allclose(
        got, ref.all_lanes(N), rtol=0, atol=0, err_msg="reset_mask 之后应该全部有效"
    )
    # 防空测：如果 reset_mask 什么也没做，读到的会是那个 0x000000FF
    assert int(got.sum()) == N, f"应该全 {N} 个有效，实际 {int(got.sum())}"


def test_the_mask_state_survives_across_two_vf_calls(device):
    """设一次，两次 vf 调用都看得到——这就是「跨 @vf 边界」。

    也说明它**会漏给后面的代码**，所以每个 kernel 都要收尾复位。
    """
    outs = _out(device, 2)
    state_persists_across_vf_calls(*outs)
    torch.npu.synchronize()
    first, second = (tensor.cpu().numpy() for tensor in outs)

    want = ref.bits_to_lanes(ref.NARROW, N)
    np.testing.assert_allclose(
        first, want, rtol=0, atol=0, err_msg="第一次调用读到的不对"
    )
    np.testing.assert_allclose(
        second,
        want,
        rtol=0,
        atol=0,
        err_msg="第二次调用没读到同一个状态——那它就不是跨边界持续的",
    )
    assert int(first.sum()) == 16, "0x0000FFFF 应该是低 16 个有效"


@pytest.mark.parametrize("name", _ALL_APIS)
def test_every_api_in_this_example_is_actually_called(name):
    """不需要卡。防漏：四个 `pl.` 侧 API 每个都得在 kernel 源码里出现。"""
    from pathlib import Path

    source = (Path(__file__).parent / "kernel.py").read_text()
    assert f"pl.{name}(" in source, f"kernel.py 里没有调用 pl.{name}()"
