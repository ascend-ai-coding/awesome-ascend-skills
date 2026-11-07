# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

r"""十个改排布或改位宽的算子：`interleave` `de_interleave` `squeeze` `unsqueeze`
`pack` `unpack` `truncate` `astype` `bit_cast` `muls_cast`。

**写法：原始。** 前置样例是 [`vf_entry`](../vf_entry/test.py) 与
[`vf_predicate`](../vf_predicate/test.py)（掩码怎么看出来）。

## 一、三组，按「改什么」分

| 组 | 算子 | 改什么 |
| --- | --- | --- |
| lane 重排 | `interleave` `de_interleave` `squeeze` `unsqueeze` | 元素**位置**，dtype 不变 |
| 位宽重组 | `pack` `unpack` | 同一批**位**重新分组 |
| 转换 | `astype` `bit_cast` `truncate` `muls_cast` | 元素**值**或它的解释方式 |

十个里有六个会「改 dtype」，但改的是三件不同的事。最容易混的三个：

| 算子 | 对输入 `1.5` | 说明 |
| --- | --- | --- |
| `astype(a, preg, dtype=I32)` | `2` | **真转换**（`vcvt`），默认**就近取偶** |
| `astype(..., round_mode=CAST_TRUNC)` | `1` | 同一个算子，换取整模式 |
| `bit_cast(a, dtype=I32)` | `1069547520` | **只改视图**，`0x3FC00000` 原样读成整数 |
| `truncate(a, preg)` | `1.0` | 向零取整，**结果还是 FP32** |
| `muls_cast(a, 4.0, preg, dtype=FP16)` | `6.0`（FP16） | 乘完转窄，**dtype 只能是 FP16** |

`test_astype_and_bit_cast_differ_by_orders_of_magnitude` 直接断言这两个差着数量级——
否则「两个都过」不说明它们做的是不同的事。

两个签名比实际约束宽的地方：

- **`astype` 的默认 `round_mode` 是 `CAST_RINT`（就近、平局取偶），不是向零取整。**
  `1.5 → 2`、`2.5 → 2`。想要截断得显式传 `round_mode=pl.VFRoundMode.CAST_TRUNC`。
- **`muls_cast` 的 `dtype` 只有一个合法值 `DT_FP16`**，签名看不出来——传别的在 codegen 才拦。

## 二、`interleave` / `de_interleave` 都返回二元组

实测排布：

```
a = 1..64, b = 101..164
interleave  → dst0 = [1, 101, 2, 102, ...]   dst1 = [33, 133, 34, 134, ...]
de_interleave → dst0 = [1, 3, 5, 7, ...]     dst1 = [2, 4, 6, 8, ...]
```

`interleave` 的结果总长 2*VL，**拆成两个寄存器**；`de_interleave` 把 `(src0, src1)` 当成一条
2*VL 的流，`dst0` 取偶位、`dst1` 取奇位。

**与 [`vf_align`](../vf_align/test.py) 的分工**：这里是寄存器↔寄存器，那里的
`LoadDist.DINTLV_B32` / `StoreDist.INTLV_B32` 是 Tile↔寄存器。同一个词，两个层次。

## 三、`squeeze` / `unsqueeze` 是一对，而官方文档的公式与正文矛盾

- `squeeze(src, preg)`：选中的元素**依次搬到低端**，其余置零。
- `unsqueeze(preg)`：**掩码的前缀和**——每个 lane 的值 = 它前面有多少个有效 lane，
  也就是「这个 lane 的元素 squeeze 之后会落到哪」。

`unsqueeze` 的官方页里，**正文**写「dst 首位为 0，mask[i] 为 1 时 dst[i] = dst[i-1] + 1」，
**公式**写 `dst_i = 1 if mask_i = 1`。两者矛盾。**实测是正文那个**：掩码在下标 32..63 为真时，
读到的是 `[0,...,0, 0,1,2,3,...]`，从下标 33 开始递增。

**`unsqueeze` 的 dst 只能是整数类型**（`INT8/UINT8/INT16/UINT16/INT32/UINT32`），
给 FP32 会被 codegen 拒（`backend_cce_vf_ops.cpp:5807`）。所以本例这一段整条链走 I32。

## 四、`pack` / `unpack`：同一个 `part`，一个指目的一个指源

| 算子 | `part` 选的是 | dtype 约束 |
| --- | --- | --- |
| `pack` | **dst 的哪一半** | dst **一律无符号**：`UINT8←INT16/UINT16`、`UINT16←INT32/UINT32`、`UINT32←INT64/UINT64` |
| `unpack` | **src 的哪一半** | **保持符号性**：INT16←INT8、UINT16←UINT8、INT32←INT16、UINT32←UINT16、INT64←INT32、UINT64←UINT32 |

两张表是 codegen 报错原文给出来的（`backend_cce_vf_ops.cpp:3715` / `:3783`）——写这一例时
`pack(..., dtype=DT_INT16)` 和 `unpack(..., dtype=DT_UINT64)` 各被拒一次，报错直接列出了支持对。
**这类报错值得照抄进注释**：它比文档的表格更贴近当前版本。

## 五、判据

`rtol=0, atol=0`。重排是搬位置、位宽重组是搬位、转换用半整数（`1.5, 2.5, ...`）——
整数输入下 `astype` 和 `truncate` 看起来一样，就验不到取整行为。
"""

import numpy as np
import pytest
import torch

from . import reference as ref
from .kernel import N, compact, convert, regroup, shuffle

_ALL_APIS = [
    "interleave",
    "de_interleave",
    "squeeze",
    "unsqueeze",
    "pack",
    "unpack",
    "truncate",
    "astype",
    "bit_cast",
    "muls_cast",
]


def _dev(array, device):
    return torch.from_numpy(array).to(device)


def _outs(device, count, dtype=torch.float32):
    return [torch.zeros((1, N), dtype=dtype, device=device) for _ in range(count)]


# ---------------------------------------------------------------------------
# lane 重排
# ---------------------------------------------------------------------------


def test_interleave_and_de_interleave_return_two_registers(device):
    """两个算子都返回二元组，排布按实测。"""
    a, b = ref.make_shuffle_operands(N)
    outs = _outs(device, 4)
    shuffle(_dev(a, device), _dev(b, device), *outs)
    torch.npu.synchronize()
    i0, i1, d0, d1 = (tensor.cpu().numpy() for tensor in outs)

    want_i0, want_i1 = ref.interleave(a, b)
    want_d0, want_d1 = ref.de_interleave(a, b)
    np.testing.assert_allclose(
        i0, want_i0, rtol=0, atol=0, err_msg="interleave dst0 不一致"
    )
    np.testing.assert_allclose(
        i1, want_i1, rtol=0, atol=0, err_msg="interleave dst1 不一致"
    )
    np.testing.assert_allclose(
        d0, want_d0, rtol=0, atol=0, err_msg="de_interleave dst0 不一致"
    )
    np.testing.assert_allclose(
        d1, want_d1, rtol=0, atol=0, err_msg="de_interleave dst1 不一致"
    )


def test_the_two_returned_registers_are_not_the_same(device):
    """防空测：两个返回值必须不同，否则「返回二元组」这件事没验到。

    操作数的两组值域不重叠（1..64 / 101..164），所以还能断言 `dst0` 真的混了两个源。
    """
    a, b = ref.make_shuffle_operands(N)
    outs = _outs(device, 4)
    shuffle(_dev(a, device), _dev(b, device), *outs)
    torch.npu.synchronize()
    i0, i1, d0, d1 = (tensor.cpu().numpy() for tensor in outs)

    assert not np.array_equal(i0, i1), "interleave 的两个返回值相同"
    assert not np.array_equal(d0, d1), "de_interleave 的两个返回值相同"

    flat = i0.reshape(-1)
    assert flat[0] <= 100 and flat[1] > 100, (
        f"interleave dst0 的头两个应该分别来自 a 和 b，实际 {flat[:2]}"
    )


def test_squeeze_packs_the_selected_elements_to_the_low_end(device):
    """`squeeze`：选中的元素依次搬到低端，其余置零。"""
    a = ref.make_shuffle_operands(N)[0]
    outs = _outs(device, 1) + _outs(device, 1, torch.int32)
    compact(_dev(a, device), *outs)
    torch.npu.synchronize()
    squeezed = outs[0].cpu().numpy()

    np.testing.assert_allclose(
        squeezed, ref.squeeze(a), rtol=0, atol=0, err_msg="squeeze 的结果不一致"
    )
    flat = squeezed.reshape(-1)
    assert flat[0] == 33.0, f"a>32 的第一个是 33，实际 {flat[0]}"
    assert (flat[32:] == 0).all(), "选中 32 个，后 32 个位置应该置零"


def test_unsqueeze_is_the_prefix_sum_of_the_mask(device):
    """`unsqueeze` 给的是**前缀和**（官方页正文），不是 0/1（同一页的公式）。

    掩码在下标 32..63 为真：读到的应该是前 33 个为 0，然后 1,2,3,... 递增。
    """
    a = ref.make_shuffle_operands(N)[0]
    outs = _outs(device, 1) + _outs(device, 1, torch.int32)
    compact(_dev(a, device), *outs)
    torch.npu.synchronize()
    indices = outs[1].cpu().numpy()

    np.testing.assert_array_equal(
        indices, ref.unsqueeze(a), err_msg="unsqueeze 不是掩码的前缀和"
    )
    flat = indices.reshape(-1)
    assert flat[:33].tolist() == [0] * 33, f"前 33 个应该全是 0，实际 {flat[:34]}"
    assert flat[33] == 1 and flat[34] == 2, f"应从下标 33 开始递增，实际 {flat[33:36]}"
    # 防混淆：如果它是 0/1，最大值就是 1
    assert flat.max() > 1, f"最大值 {flat.max()} ≤ 1，那它就是 0/1 而不是前缀和"


# ---------------------------------------------------------------------------
# 位宽重组
# ---------------------------------------------------------------------------


def test_pack_part_selects_the_destination_half(device):
    """`pack` 的 `part` 选 **dst 的哪一半**。LOWER 写低半、UPPER 写高半。"""
    a = ref.make_regroup_operands(N)
    outs = _outs(device, 4, torch.int32)
    regroup(_dev(a, device), *outs)
    torch.npu.synchronize()
    pack_lo, pack_hi = (tensor.cpu().numpy() for tensor in outs[:2])

    np.testing.assert_array_equal(
        pack_lo, ref.pack(a, "LOWER"), err_msg="pack LOWER 不一致"
    )
    np.testing.assert_array_equal(
        pack_hi, ref.pack(a, "UPPER"), err_msg="pack UPPER 不一致"
    )
    # LOWER 的数据在前半、UPPER 的在后半——这才说明 part 选的是目的
    assert (pack_lo.reshape(-1)[slice(N // 2, None)] == 0).all(), (
        "pack LOWER 的后半应该是 0"
    )
    assert (pack_hi.reshape(-1)[: N // 2] == 0).all(), "pack UPPER 的前半应该是 0"


def test_unpack_part_selects_the_source_half(device):
    """`unpack` 的 `part` 选 **src 的哪一半**——和 `pack` 相反。

    两个输出必须不同，否则 `part` 没起作用。
    """
    a = ref.make_regroup_operands(N)
    outs = _outs(device, 4, torch.int32)
    regroup(_dev(a, device), *outs)
    torch.npu.synchronize()
    un_lo, un_hi = (tensor.cpu().numpy() for tensor in outs[2:])

    np.testing.assert_array_equal(
        un_lo, ref.unpack(a, "LOWER"), err_msg="unpack LOWER 不一致"
    )
    np.testing.assert_array_equal(
        un_hi, ref.unpack(a, "UPPER"), err_msg="unpack UPPER 不一致"
    )
    assert not np.array_equal(un_lo, un_hi), (
        "unpack 的 LOWER 与 UPPER 给出相同结果——part 没起作用"
    )
    # 正数扩宽后高位是 0：每隔一个槽位应该是 0
    assert (un_lo.reshape(-1)[1::2] == 0).all(), "int64 的高 32 位对正数应该是 0"


# ---------------------------------------------------------------------------
# 转换
# ---------------------------------------------------------------------------


def _run_convert(device):
    a = ref.make_convert_operands(N)
    outs = (
        _outs(device, 2, torch.int32) + _outs(device, 2) + _outs(device, 1, torch.int32)
    )
    convert(_dev(a, device), *outs)
    torch.npu.synchronize()
    return a, [tensor.cpu().numpy() for tensor in outs]


def test_the_four_conversions_match_the_reference(device):
    """`astype` / `bit_cast` / `truncate` / `muls_cast` 各自比。"""
    a, got = _run_convert(device)
    astype, bitcast, trunc, mulscast, astype_z = got

    np.testing.assert_array_equal(
        astype,
        ref.astype_to_int32(a),
        err_msg="astype 不一致（取整方式和参考不符？看下面的实际值）",
    )
    np.testing.assert_array_equal(
        astype_z,
        ref.astype_toward_zero_to_int32(a),
        err_msg="astype(round_mode=CAST_TRUNC) 不一致",
    )
    np.testing.assert_array_equal(
        bitcast, ref.bit_cast_to_int32(a), err_msg="bit_cast 不一致"
    )
    np.testing.assert_allclose(
        trunc, ref.truncate(a), rtol=0, atol=0, err_msg="truncate 不一致"
    )
    np.testing.assert_allclose(
        mulscast, ref.muls_cast(a), rtol=0, atol=0, err_msg="muls_cast 不一致"
    )


def test_astype_and_bit_cast_differ_by_orders_of_magnitude(device):
    """同一个输入、同一个目标 dtype，两者差着数量级——这就是「转换」与「改视图」的区别。

    防空测：如果两者结果相同，说明其中一个没做它该做的事。
    """
    a, got = _run_convert(device)
    astype, bitcast = got[0], got[1]

    assert not np.array_equal(astype, bitcast), "astype 和 bit_cast 结果相同"
    assert np.abs(astype).max() < 1000, (
        f"astype 的结果应该在输入量级（~{np.abs(a).max():.0f}），实际最大 {np.abs(astype).max()}"
    )
    assert np.abs(bitcast).max() > 10**8, (
        f"bit_cast 的结果应该是位型（~1e9），实际最大 {np.abs(bitcast).max()}"
    )


def test_astype_rounds_to_nearest_even_by_default(device):
    """**默认 `round_mode` 是 `CAST_RINT`（就近、平局取偶），不是向零取整。**

    半整数输入 `1.5 2.5 3.5 4.5` 下：默认给 `2 2 4 4`，`CAST_TRUNC` 给 `1 2 3 4`。
    两份一起断言，「默认不是截断」才立得住——单看一份看不出区别。

    写这一版时我按「默认截断」写参考，就是这条把它推翻的。
    """
    a, got = _run_convert(device)
    astype, astype_z = got[0], got[4]

    assert not np.array_equal(astype, astype_z), (
        "默认 astype 与 round_mode=CAST_TRUNC 结果相同——round_mode 没起作用，"
        "或者这批输入验不到取整差异"
    )
    head = astype.reshape(-1)[:4].tolist()
    assert head == [2, 2, 4, 4], f"1.5/2.5/3.5/4.5 就近取偶应为 [2,2,4,4]，实际 {head}"
    head_z = astype_z.reshape(-1)[:4].tolist()
    assert head_z == [1, 2, 3, 4], f"向零取整应为 [1,2,3,4]，实际 {head_z}"


def test_truncate_stays_in_floating_point(device):
    """`truncate` 的结果仍是 FP32——它取整但不换类型。这是它和 `astype` 的区别。

    取整规则上它等于 `astype(round_mode=CAST_TRUNC)`，差别只在输出类型。
    """
    a, got = _run_convert(device)
    trunc, astype_z = got[2], got[4]

    assert trunc.dtype == np.float32, (
        f"truncate 的输出应该是 float32，实际 {trunc.dtype}"
    )
    flat, src = trunc.reshape(-1), a.reshape(-1)
    assert flat[0] == 1.0, f"1.5 向零取整应为 1.0，实际 {flat[0]}"
    assert (np.abs(flat - src) > 0).all(), "半整数输入取整后必须都变了，否则没验到"
    np.testing.assert_array_equal(
        trunc.astype(np.int32),
        astype_z,
        err_msg="truncate 与 astype(CAST_TRUNC) 的取整规则应当一致",
    )


def test_muls_cast_multiplies_before_converting(device):
    """`muls_cast(a, 4.0, preg, dtype=FP16)` = `fp16(a * 4)`，一条指令。

    **`dtype` 只有一个合法值。** 传 `I32` 不会被解析期拦，在 codegen 才报
    `vf.muls_cast destination only supports FP16`——签名比实际约束宽，见 kernel.py 的说明。

    防空测：如果标量被忽略，结果就等于输入本身（`muls_cast` 不取整，只做 FP32→FP16）。
    """
    a, got = _run_convert(device)
    mulscast = got[3]

    assert not np.allclose(mulscast, a), "muls_cast 的结果等于输入，说明标量没被用上"
    np.testing.assert_allclose(
        mulscast, ref.muls_cast(a), rtol=0, atol=0, err_msg="muls_cast 不一致"
    )


@pytest.mark.parametrize("name", _ALL_APIS)
def test_every_api_in_this_example_is_actually_called(name):
    """不需要卡。防漏：表里列的 10 个算子每个都得在 kernel 源码里出现。"""
    from pathlib import Path

    source = (Path(__file__).parent / "kernel.py").read_text()
    assert f"vf.{name}(" in source, f"kernel.py 里没有调用 vf.{name}()"
