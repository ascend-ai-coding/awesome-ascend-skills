# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

r"""五个位运算：`and_` `or_` `xor` `shift_left` `shift_right`。

**写法：原始。** 前置样例是 [`vf_entry`](../vf_entry/test.py)。

## 一、命名：下划线是躲关键字，不是风格

`and` / `or` / `not` 是 Python 关键字，不能当函数名，所以是 `vf.and_` / `vf.or_` / `vf.not_`。
`xor` 不是关键字，所以**没有**下划线。同一组里命名不统一，按关键字冲突来的。

这正是 [skill constraints](../../../../SKILL.md#constraints-and-workspace-policy) 那条搜索纪律的场景：按 `xor_` 搜不到就试 `xor`,
按 `and` 搜不到就试 `and_`。写 `debug_probe` 时按 `def assert_` 搜漏了 `pto_assert`，同一个坑。

## 二、移位量：两种写法都行

实测（2026-09-27）裸字面量 `2` 和 `pl.const(2, I32)` 都被接受、结果相同。本例两种各用一次。

**移位量是不是能用寄存器（逐元素不同的位移）没测**，本例不声称。

## 三、操作数为什么这样挑

| 操作数 | 取值 | 为什么 |
| --- | --- | --- |
| `a` | -32..31 | **必须含负数**，否则分不清算术右移和逻辑右移 |
| `b` | `0x0F0F0F0F` / `0xF0F0F0F0` / `0x00FF00FF` / `0xFF00FF00` 循环 | 位模式互补，and/or/xor 两两不同；小整数下 or/xor 多数位置相等，测不出区别 |

`test_and_or_xor_give_three_different_results` 直接断言这件事——防止「三个算子都通过了，
但其实结果一样」这种空测。

## 四、实测：`shift_right` 对负数是算术右移

补符号位，不是逻辑右移。`-32 >> 2 == -8`，不是 `0x3FFFFFF8`。
`test_shift_right_is_arithmetic_on_negative_values` 钉住这条——它决定了参考实现能不能
直接用 numpy 的 `>>`（可以，numpy 对 int32 也是算术右移）。

## 五、判据

`rtol=0, atol=0`。整数运算没有浮点误差，任何一位不同都是错。
"""

import numpy as np
import pytest
import torch

from . import reference as ref
from .kernel import N, bitwise_ops, make_mask_logic

_NAMES = ["and_", "or_", "xor", "shift_left", "shift_right"]


def _run(device):
    a, b = ref.make_operands(N)
    outs = [torch.zeros((1, N), dtype=torch.int32, device=device) for _ in range(5)]
    bitwise_ops(torch.from_numpy(a).to(device), torch.from_numpy(b).to(device), *outs)
    torch.npu.synchronize()
    return a, b, [tensor.cpu().numpy() for tensor in outs]


def test_the_five_bitwise_ops_match_the_reference(device):
    """逐位比。"""
    a, b, got = _run(device)
    want = ref.bitwise(a, b)
    for name, g, w in zip(_NAMES, got, want):
        np.testing.assert_array_equal(g, w, err_msg=f"vf.{name} 不一致")


def test_and_or_xor_give_three_different_results(device):
    """防空测：操作数必须让三个算子的结果两两不同，否则「三个都过」不说明什么。"""
    _, _, got = _run(device)
    and_r, or_r, xor_r = got[0], got[1], got[2]
    assert not np.array_equal(and_r, or_r), "and 和 or 结果相同——操作数选得不对"
    assert not np.array_equal(or_r, xor_r), "or 和 xor 结果相同——操作数选得不对"
    assert not np.array_equal(and_r, xor_r), "and 和 xor 结果相同——操作数选得不对"


def test_shift_right_is_arithmetic_on_negative_values(device):
    """负数右移补符号位。`-32 >> 2 == -8`，不是一个大正数。"""
    a, _, got = _run(device)
    shr = got[4]
    negative = a < 0
    assert negative.any(), "输入没有负数，这一条是空的"
    assert (shr[negative] < 0).all(), (
        f"负数右移后应该仍是负数（算术右移）。实际拿到 {shr[negative][:4]}"
    )
    np.testing.assert_array_equal(shr[negative], (a[negative] >> ref.SHIFT))


def test_shift_left_matches_a_multiply_by_four(device):
    """左移 2 位 == 乘 4。在不溢出的范围内两者必须一致——这是对「移位量真的是 2」的交叉验证。"""
    a, _, got = _run(device)
    shl = got[3]
    small = np.abs(a) < (1 << 28)  # 不溢出的范围
    np.testing.assert_array_equal(
        shl[small], (a[small] * (1 << ref.SHIFT)).astype(np.int32)
    )


@pytest.mark.parametrize("name", _NAMES)
def test_every_api_in_this_example_is_actually_called(name):
    """不需要卡。防漏：5 个算子每个都得在 kernel 源码里出现。"""
    from pathlib import Path

    source = (Path(__file__).parent / "kernel.py").read_text()
    assert f"vf.{name}(" in source, f"kernel.py 里没有调用 vf.{name}()"


@pytest.mark.parametrize("active", [53, 64])
def test_mask_logic_truth_tables_and_participation(device, active):
    backing = torch.full((4 * N + 128,), 314159, dtype=torch.int32, device=device)
    out = backing[64:-64].view(4, N)
    kernel = make_mask_logic(active)
    for _ in range(2):
        out.fill_(-99)
        kernel[None, 1](out)
        torch.npu.synchronize()
        np.testing.assert_array_equal(out.cpu().numpy(), ref.mask_logic(N, active))
        np.testing.assert_array_equal(backing[:64].cpu().numpy(), 314159)
        np.testing.assert_array_equal(backing[-64:].cpu().numpy(), 314159)


@pytest.mark.parametrize("active", [0, 53, 64])
def test_mask_pack_unpack_preserves_predicates_across_dtype_keywords(device, active):
    from .kernel import make_mask_pack_roundtrip

    backing = torch.full((N + 128,), 314159, dtype=torch.int32, device=device)
    out = backing[64:-64].view(1, N)
    expected = np.zeros((1, N), dtype=np.int32)
    expected[:, :active] = 7
    kernel = make_mask_pack_roundtrip(active)
    for _ in range(2):
        out.fill_(-99)
        kernel[None, 1](out)
        torch.npu.synchronize()
        np.testing.assert_array_equal(out.cpu().numpy(), expected)
        np.testing.assert_array_equal(backing[:64].cpu().numpy(), 314159)
        np.testing.assert_array_equal(backing[-64:].cpu().numpy(), 314159)
