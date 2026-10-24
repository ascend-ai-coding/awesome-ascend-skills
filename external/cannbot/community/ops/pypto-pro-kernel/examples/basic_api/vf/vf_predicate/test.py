# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

r"""十个谓词 API：`eq` `ne` `lt` `gt` `le` `ge` `select` `not_` `get_mask_spr`
`mask_gen_with_reg_tensor`。

**写法：原始。** 前置样例是 [`vf_entry`](../vf_entry/test.py)。

## 一、掩码不是「每 lane 一位」

官方 `basic_data_structures/mask_reg.md`：**mask_reg 总宽固定 256 bit**，每元素占几位由 dtype 定：

| dtype 位宽 | 元素个数 | 每元素掩码位数 |
| --- | --- | --- |
| 8 bit | 256 | 1 bit |
| 16 bit | 128 | 2 bit |
| **32 bit** | **64** | **4 bit** |
| 64 bit | 32 | 8 bit |

所以 FP32 下一个 lane 对应 4 个掩码位。这决定了 `mask_gen_with_reg_tensor` 的 broadcast 倍数，
也是「掩码 dtype 要和数据 dtype 对上」的原因。

## 二、怎么把掩码看出来——以及一个不能用的办法

掩码寄存器存不进 Tile。本例一律用 **`select(ones, zeros, mask)`** 变成 0/1 向量：
`select` 的语义是文档写明的（真取 `src0`、假取 `src1`），没有歧义。

**`vf.move(x, mask)` 不能用来观察掩码。** 写这一例时我这么试过：非活跃 lane 拿到的是
*另一个输入的值*，不是 0。那取决于编译器把目的寄存器分配到哪儿——**不是可依赖的语义**。
第一版探针就因此得出一串互相矛盾的结论（「`and_` 两个互斥掩码之后还是全真」之类），
全部作废。教训记在 [`verification/evidence.md`](../../../../references/verification/evidence.md)。

## 三、链式比较与掩码逻辑

本例用链式比较表达交集。直接 MaskReg 逻辑运算的契约归 [skill 的掩码组合说明
](../../../../references/api/vf.md#25-combining-masks-with-existing-comparisons-and-selection)，完整真值表见
[vf_bitwise](../vf_bitwise/test.py)。

| 想要 | 怎么写 |
| --- | --- |
| 两个条件取**交集** | 链式——第二个比较拿第一个掩码当 `preg`：`m2 = vf.lt(a, HI, m1)` |
| **反向选择数据** | `select` 换源：`vf.select(zeros, ones, m)` |
| 和**标量**比较 | 直接传标量，`gt.md` 写明接口会自动分发 |

交集能这么写，是因为比较算子**自己带 `preg`**（`gt(src0, src1, preg) -> dst_mask`），
无效位置在结果掩码里置零——那就是 AND。

## 四、RegTensor 的按位取反

本例的 `vf.not_` 处理 RegTensor 数据，参考实现使用 `~a.view(np.int32)`，翻转 FP32 的位型，不做数值取负；MaskReg 重载由另一组真值表测试覆盖。

## 五、两个掩码来源

- **`get_mask_spr(width)`**：读 kernel 体里 `pl.set_vec_mask(high, low)` 设的 SPR。FP32 下取
  MASK0 的 64 bit，每 bit 复制成 4 bit → 第 i 位控制第 i 个 lane。本例设 `0xFFFFFFFF`，
  所以 lane 0..31 有效、32..63 无效。**这是 kernel 体与 vf 体之间的桥**，完整的设/复位面
  在 [`vf_mask_state`](../vf_mask_state/test.py)。
- **`mask_gen_with_reg_tensor(src, offset)`**：从第 `offset` 个 **DataBlock** 的裸比特生成掩码。
  FP32 下 256B 分 32 个 8B 块，所以 `offset=0` 取 **lane 0..1** 的 64 位、`offset=1` 取
  lane 2..3 的——**按字节分块，不是按元素**。这一点最容易搞错，测试里两个 offset 都验。

## 六、判据

`rtol=0, atol=0`。比较结果是 0/1，掩码是位运算，都没有浮点误差。

操作数挑成让六个比较的真值个数**两两不同**（`a=1..64`、`b=32`：1 / 63 / 31 / 32 / 32 / 33
——`gt` 和 `le` 都是 32，但真值**位置**互补）。
`test_the_six_comparisons_are_all_distinct` 直接断言六个输出两两不同，防止「六个都过但其实
算的是同一件事」。
"""

import numpy as np
import pytest
import torch

from . import reference as ref
from .kernel import N, combine, compare_all, mask_from_reg_bits, mask_from_spr

_CMP_NAMES = ["eq", "ne", "lt", "gt", "le", "ge"]
_ALL_APIS = _CMP_NAMES + ["select", "not_", "get_mask_spr", "mask_gen_with_reg_tensor"]


def _dev(array, device):
    return torch.from_numpy(array).to(device)


def _outs(device, count):
    return [
        torch.zeros((1, N), dtype=torch.float32, device=device) for _ in range(count)
    ]


def _run_compare(device):
    a, b = ref.make_operands(N)
    outs = _outs(device, 6)
    compare_all(_dev(a, device), _dev(b, device), *outs)
    torch.npu.synchronize()
    return a, b, [tensor.cpu().numpy() for tensor in outs]


def test_the_six_comparisons_match_the_reference(device):
    """六个比较算子，掩码经 `select` 变成 0/1 后逐位比。"""
    a, b, got = _run_compare(device)
    want = ref.compare(a, b)
    for name, g, w in zip(_CMP_NAMES, got, want):
        np.testing.assert_allclose(g, w, rtol=0, atol=0, err_msg=f"vf.{name} 不一致")


def test_the_six_comparisons_are_all_distinct(device):
    """防空测：六个输出必须两两不同，否则「六个都过」不说明它们算的是不同的事。"""
    _, _, got = _run_compare(device)
    for i in range(len(_CMP_NAMES)):
        for j in range(i + 1, len(_CMP_NAMES)):
            assert not np.array_equal(got[i], got[j]), (
                f"vf.{_CMP_NAMES[i]} 和 vf.{_CMP_NAMES[j]} 的结果完全相同——"
                f"操作数选得不对，或者其中一个没真正被调用"
            )


def test_ne_is_the_complement_of_eq(device):
    """`ne` 在整个 pypto 仓的测试里出现 **0 次**（设计文档 §五）。这条是它的第一次验证。

    `ne` 必须恰好是 `eq` 的补集——一个都不能差。
    """
    _, _, got = _run_compare(device)
    eq, ne = got[0], got[1]
    np.testing.assert_allclose(
        eq + ne, np.ones_like(eq), rtol=0, atol=0, err_msg="eq 与 ne 不是互补的"
    )
    assert eq.sum() > 0, "eq 一个都没真，这条验不到互补"
    assert ne.sum() > 0, "ne 一个都没真"


def test_chaining_a_comparison_through_preg_is_an_and(device):
    """**交集靠链式**：`vf.lt(a, HI, lo_mask)` = `(a > LO) and (a < HI)`。

    这是「掩码代数」那个开放项的答案——不需要 `mask_and`。
    """
    a, _ = ref.make_operands(N)
    outs = _outs(device, 4)
    combine(_dev(a, device), *outs)
    torch.npu.synchronize()
    band, inverted, scalar, not_data = (tensor.cpu().numpy() for tensor in outs)

    want_band, want_inverted, want_scalar, want_not = ref.combine(a)
    np.testing.assert_allclose(
        band, want_band, rtol=0, atol=0, err_msg="链式取交集的结果不对"
    )

    # 防空测：交集必须比任一单独条件都严格
    only_lo = (a > ref.LO).astype(np.float32)
    assert band.sum() < only_lo.sum(), (
        f"交集的真值个数 {band.sum():.0f} 不小于单独 a>LO 的 {only_lo.sum():.0f}——"
        f"说明第二个条件没起作用"
    )


def test_swapping_select_sources_inverts_a_mask(device):
    """**取反靠换源**：`select(zeros, ones, m)` 是 `select(ones, zeros, m)` 的补集。"""
    a, _ = ref.make_operands(N)
    outs = _outs(device, 4)
    combine(_dev(a, device), *outs)
    torch.npu.synchronize()
    band, inverted = (tensor.cpu().numpy() for tensor in outs[:2])

    np.testing.assert_allclose(
        band + inverted, np.ones_like(band), rtol=0, atol=0, err_msg="换源没有得到补集"
    )


def test_a_comparison_takes_a_scalar_second_operand(device):
    """`gt.md`：第二个参数可以是标量或 reg_tensor，接口自动分发。"""
    a, _ = ref.make_operands(N)
    outs = _outs(device, 4)
    combine(_dev(a, device), *outs)
    torch.npu.synchronize()
    scalar = outs[2].cpu().numpy()

    np.testing.assert_allclose(
        scalar, ref.combine(a)[2], rtol=0, atol=0, err_msg="和标量比较的结果不对"
    )


def test_not_is_a_bitwise_not_on_data_not_on_a_mask(device):
    """`vf.not_` 翻的是 FP32 的**位型**，不是掩码，也不是数值取反。

    所以参考要在 int32 视图上算。如果谁把它当成掩码取反用，这条测试的形状能提醒他。
    """
    a, _ = ref.make_operands(N)
    outs = _outs(device, 4)
    combine(_dev(a, device), *outs)
    torch.npu.synchronize()
    not_data = outs[3].cpu().numpy()

    want = ref.combine(a)[3]
    np.testing.assert_array_equal(
        not_data.view(np.int32), want.view(np.int32), err_msg="not_ 的位型不对"
    )
    # 防混淆：它不是 -a
    assert not np.allclose(not_data, -a), "not_ 的结果等于 -a，那它就不是按位取反"


def test_get_mask_spr_reads_the_kernel_scope_mask(device):
    """`pl.set_vec_mask(0, 0xFFFFFFFF)` 在 kernel 体里设，vf 体里 `get_mask_spr()` 读到。

    低 32 位为 1 → lane 0..31 有效、32..63 无效。这是 kernel 体与 vf 体之间的桥。
    """
    a, _ = ref.make_operands(N)
    outs = _outs(device, 1)
    mask_from_spr(_dev(a, device), *outs)
    torch.npu.synchronize()
    got = outs[0].cpu().numpy()

    want = ref.mask_from_spr_expected(N, 0xFFFFFFFF)
    np.testing.assert_allclose(
        got, want, rtol=0, atol=0, err_msg="从 SPR 读到的掩码与设进去的位型不一致"
    )

    flat = got.reshape(-1)
    assert (flat[:32] == 1).all(), "lane 0..31 应该全有效"
    assert (flat[32:] == 0).all(), "lane 32..63 应该全无效"


def test_mask_gen_blocks_are_eight_bytes_not_one_element(device):
    """`mask_gen_with_reg_tensor` 的 `offset` 按 **8 字节 DataBlock** 走，不是按元素。

    FP32 下 `offset=0` 取 lane 0..1 的 64 位，`offset=1` 取 lane 2..3 的。
    两个 offset 都验，并且断言它们不同——否则「按块」这件事就没验到。
    """
    bits, raw = ref.make_bit_pattern(N)
    outs = _outs(device, 2)
    mask_from_reg_bits(_dev(bits, device), *outs)
    torch.npu.synchronize()
    block0, block1 = (tensor.cpu().numpy() for tensor in outs)

    np.testing.assert_allclose(
        block0,
        ref.mask_from_bits_expected(raw, N, 0),
        rtol=0,
        atol=0,
        err_msg="offset=0 的掩码不对",
    )
    np.testing.assert_allclose(
        block1,
        ref.mask_from_bits_expected(raw, N, 1),
        rtol=0,
        atol=0,
        err_msg="offset=1 的掩码不对",
    )
    assert not np.array_equal(block0, block1), (
        "两个 offset 给出相同掩码——位型选得不对，或者 offset 没起作用"
    )


@pytest.mark.parametrize("name", _ALL_APIS)
def test_every_api_in_this_example_is_actually_called(name):
    """不需要卡。防漏：表里列的 10 个 API 每个都得在 kernel 源码里出现。"""
    from pathlib import Path

    source = (Path(__file__).parent / "kernel.py").read_text()
    assert f"vf.{name}(" in source, f"kernel.py 里没有调用 vf.{name}()"
