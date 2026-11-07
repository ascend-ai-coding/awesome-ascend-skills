# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

r"""``vf.*`` 的寄存器算术：十三个算子，按**调用形状**分四组。

**写法：原始。** `vf.*` 没有语法糖。

前置样例是 [`vf_entry`](../vf_entry/test.py)——入口、VL、掩码那些机制在那里，这一例只讲算什么。

## 一、四种调用形状

分组的依据是形状不是运算，因为形状才是会写错的地方：

| 形状 | 算子 | 要注意 |
| --- | --- | --- |
| `f(src0, src1, preg)` | add / sub / mul / div / min / max | 最普通的一组 |
| `f(src, scalar, preg)` | adds / muls / mins / maxs | 标量走 `pl.const(v, dtype)` |
| 复合 | abs_sub / axpy | **`axpy` 读 dst 前值**，见下 |
| 返回元组 | mull | `(lo, hi)` 两个寄存器 |

## 一点五、写这一例踩到的三条语法约束

三条都是解析期响亮拒绝、报错直接指到行，但都不在官方文档的接口页里：

1. **`pl.const()` 的第一个参数必须是字面量。** `pl.const(SCALAR, FP32)`（`SCALAR` 是模块级的
   `2.0`）会被拒：*"pl.const() first argument must be a numeric literal"*（`scalar_ops.py:214`）。
   解析期从 AST 上读值，闭包变量读不到。所以 kernel 里是写死的 `2.0`，模块常量 `SCALAR` 只给
   host 侧的参考和测试用。
2. **vf 的结果必须先赋给变量，不能嵌进别的调用的参数里。**
   `vf.store_align(o0, vf.add(a, b, preg), preg, 0)` 会被拒，hint 就是
   *"Use: dst = vf.add(...)"*（`_call_parser.py:1647`）。所以 kernel 里到处是 `r = vf.xxx(...)`
   紧跟一个 `store_align`——不是啰嗦，是必须。解析器要在 IR 里给每个结果一个名字。
3. **kernel 体内不能用列表推导。**
   `gs = [pl.make_tile_group(...) for i in range(6)]` 会被拒（`_expression_parser.py:267`）。
   所以 `arith_binary` 里六个输出组是逐个写开的——那份重复是语言约束，不是没想到。

**这三条的共性：kernel 体是被解析的，不是被执行的。** 能写什么由解析器认得什么决定，不由 Python
决定。凡是「值要在解析期从 AST 上读出来」的地方（`pl.const` 的字面量）、「结果要在 IR 里有名字」
的地方（vf 的返回值）、「控制流要能展开」的地方（列表推导）都会碰到这条线。

三条各花了一轮才发现，写在这里省下下一个人的三轮。

## 二、`axpy` 是这一例最容易写错的

定义是 ``dst_i = scalar * src_i + dst_i``——**它累加进 dst，而 dst 是赋值左边那个变量的前值**：

```python
acc = vf.load_align(src_b, 0)      # ← 累加初值，必须先装
acc = vf.axpy(a, s, preg)          # ← 读的是上一行那个 acc，结果 = s*a + b
```

删掉第一行，`axpy` 就没有 dst 可读。这个写法照的是 `test_vf_basic_ops.py:486-487`，不是我们发明的。
验证 3 用 ``2a + b`` 而不是 ``2a`` 钉住这件事——如果实现其实不读 dst，那条会立刻失败。

## 三、`MergeMode`：默认 ZEROING，而 MERGING 在 A5 上用不了

每个算子都有 `mode=` 参数，管**掩码没选中的车道在目的寄存器里变成什么**：

| 值 | 语义 | A5 上 |
| --- | --- | --- |
| ``ZEROING``（默认） | 置 0 | 可用，验证 5 实测 |
| ``MERGING`` | 保留目的寄存器原值 | **编译期拒绝** |

2026-09-26 实测：`mode=pl.MergeMode.MERGING` 在 codegen 期报
``NOT_IMPLEMENTED_ERROR``（``backend_cce_vf_ops.cpp:162``）。是响亮失败，不是静默。

**官方文档这里有矛盾**：`MergeMode.md` 把两个值并列列出、页头标「Ascend 950PR&950DT：支持」，
读起来像两个都能用；而 `_vf_api.py` 里 `vf.full` 的 docstring 写着 *"MERGING mode is not supported
on current device"*。**实测站 docstring 那边。**

注意区分两件事：这里说的是**算子**对无效车道的处理（置 0），而 `vf_entry` 里「掩码之外的位置保留
目的地原值」说的是 **`store_align` 往 Tile 写**时不碰未选中的位置。两个掩码作用在不同层。

## 四、`addc` / `subc`：两条规则，我第一次两条都踩了

它们是带进位/借位的加减，返回元组 `(carry, dst)` / `(borrow, dst)`。

**规则一：元组的目标变量必须预声明成对的类型。**

```python
carry = vf.create_mask(pattern=pl.MaskPattern.ALLF, dtype=U32)   # ← 先把这个名字绑成 mask
carry, r = vf.addc(a, b, carry_in, preg)
```

`carry` 是 mask_reg 而 `dst` 是 reg_tensor，而**解析器按赋值目标已有的绑定定类型**。这和 `axpy`
读 dst 前值是同一条机制——凡是「目标变量参与类型或取值决定」的地方都要先绑。

少了预声明会怎样：`carry` 被当成 reg_tensor，codegen 出 ``RegTensor<uint32_t>`` 喂给要
``vector_bool&`` 的 intrinsic，编译期报 *"no matching function for call to 'vaddcs'"*。
**是响亮失败**，但错误指向 CANN 头文件而不是用户代码，所以不好定位。

**我第一轮据此判定这是框架 bug，写进了 commit `51f11a9` 的 message。那个结论是错的。**
读完官方 `addc.md` 的调用示例才发现里面有 `carry = vf.create_mask(...)` 那一行。教训是
[skill constraints](../../../../SKILL.md#constraints-and-workspace-policy) 那条：**声称某能力不存在之前，官方文档的调用示例也算一手证据，不能只读功能说明**。

**规则二：mask 里存的是硬件的进位位，不是语义上的「借位」。**

减法在底层是 `a + ~b + carry`，所以同一个 `ALL` 在两边读法相反（实测 2026-09-27）：

| 调用 | 结果 |
| --- | --- |
| ``addc(a, b, ALLF)`` | `a + b`（进位 0） |
| ``addc(a, b, ALL)`` | `a + b + 1`（进位 1） |
| ``subc(a, b, ALL)`` | `a - b`（**不借位**） |
| ``subc(a, b, ALLF)`` | `a - b - 1`（借位 1） |

参数名叫 `borrow_src` 会让人以为「全 0 = 不借位」。实测是反的，验证 6 四个组合全测，就是钉这条。

**为什么它们在框架里一次没被调用过、ascriptor 也没映射**：不是因为坏，是因为宽整数算术在 A5 上
还没有用例。官方文档有调用示例，但 ST 里没有——这一例是第一份可跑的回归。

## 五、如何独立验证

1. 六个 reg ⊕ reg 算子逐个与参考 ``rtol=0, atol=0``。
2. 四个 reg ⊕ 标量算子同上。
3. ``abs_sub`` 与 ``axpy``；`axpy` 的期望是 ``2a + b``，钉住「读 dst 前值」。
4. ``mull`` 的 `(lo, hi)` 拼回 64 位等于真乘积。操作数选得让乘积一定溢出 32 位，否则 `hi` 全 0、
   测不出高位那一半。
5. ``MergeMode.MERGING`` 必须在 codegen 期被拒。
6. ``addc`` / ``subc`` 的四个进位组合，逐个与参考逐位相等。

**能全部 `rtol=0, atol=0` 是因为操作数挑过**：`a` 是小整数，`b` 是 2 的幂。理由写在
[reference.py](reference.py)。除法那条尤其要留意——换成非 2 的幂就得改成带容差，那时这一例的
检验强度就下降了。

## 六、这一例不讲什么

- 超越函数（exp / ln / sqrt）在 ``vf_transcendental``；激活在 ``vf_activation``；位运算在
  ``vf_bitwise``；归约在 ``vf_reduce``。
- 循环与尾块在 ``vf_entry``——这里 `N` 正好一个寄存器，刻意不循环。

设计依据见 [tooling/example-coverage.md](../../../../references/tooling/example-coverage.md) §八。
"""

import numpy as np
import pytest
import torch

import pypto_pro.language as pl
from pypto_pro.language import Vf as vf  # noqa: N813

from . import reference as ref
from .kernel import (
    N,
    arith_binary,
    arith_carry,
    arith_composite,
    arith_scalar,
    arith_wide,
)

FP32 = pl.DT_FP32
I32 = pl.DT_INT32

BINARY_NAMES = ("add", "sub", "mul", "div", "min", "max")
SCALAR_NAMES = ("adds", "muls", "mins", "maxs")
COMPOSITE_NAMES = ("abs_sub", "axpy")


def _dev(array, device):
    return torch.from_numpy(np.ascontiguousarray(array)).to(device)


def _outs(device, count, dtype=torch.float32):
    return [torch.zeros([1, N], device=device, dtype=dtype) for _ in range(count)]


def test_the_six_register_pair_ops_match_the_reference(device):
    """验证 1。六个 `f(src0, src1, preg)`，逐个逐位相等。"""
    a, b = ref.make_operands(N)
    outs = _outs(device, 6)
    arith_binary(_dev(a, device), _dev(b, device), *outs)
    torch.npu.synchronize()

    want = ref.binary(a, b)
    for name, got, expected in zip(BINARY_NAMES, outs, want):
        np.testing.assert_array_equal(got.cpu().numpy(), expected, err_msg=f"vf.{name}")


def test_the_four_scalar_ops_match_the_reference(device):
    """验证 2。四个 `f(src, scalar, preg)`。"""
    a, _ = ref.make_operands(N)
    outs = _outs(device, 4)
    arith_scalar(_dev(a, device), *outs)
    torch.npu.synchronize()

    want = ref.scalar(a)
    for name, got, expected in zip(SCALAR_NAMES, outs, want):
        np.testing.assert_array_equal(got.cpu().numpy(), expected, err_msg=f"vf.{name}")


def test_axpy_accumulates_into_the_previous_value_of_its_destination(device):
    """验证 3。期望是 `2a + b`，不是 `2a`。

    kernel 把 `acc` 先装成 `b` 再调 `axpy`。如果 `axpy` 其实不读 dst 前值，这条会得到 `2a`
    并立刻失败——这就是为什么期望里那个 `+ b` 是检验点，不是装饰。
    """
    a, b = ref.make_operands(N)
    outs = _outs(device, 2)
    arith_composite(_dev(a, device), _dev(b, device), *outs)
    torch.npu.synchronize()

    want = ref.composite(a, b)
    for name, got, expected in zip(COMPOSITE_NAMES, outs, want):
        np.testing.assert_array_equal(got.cpu().numpy(), expected, err_msg=f"vf.{name}")

    # 把「读了前值」单独说一遍：axpy 的结果减掉 b 应当等于 2a
    axpy = outs[1].cpu().numpy()
    np.testing.assert_array_equal(
        axpy - b, (ref.SCALAR * a.astype(np.float64)).astype(np.float32)
    )


def test_mull_splits_the_product_into_low_and_high_halves(device):
    """验证 4。操作数选得让乘积一定溢出 32 位。"""
    a, b = ref.make_wide_operands(N)
    lo_t, hi_t = _outs(device, 2, dtype=torch.int32)
    arith_wide(_dev(a, device), _dev(b, device), lo_t, hi_t)
    torch.npu.synchronize()

    want_lo, want_hi = ref.wide(a, b)
    np.testing.assert_array_equal(lo_t.cpu().numpy(), want_lo, err_msg="mull lo")
    np.testing.assert_array_equal(hi_t.cpu().numpy(), want_hi, err_msg="mull hi")

    # 高位那一半真的非零，否则这条测试是空的
    assert (want_hi != 0).all(), "操作数没让乘积溢出 32 位，hi 全 0，测不出高位"


def test_merging_mode_is_refused_on_this_device(device):
    """验证 5。`MergeMode.MERGING` 在 A5 上编译期就被拒。

    需要卡：这是 codegen 期的拒绝，得走到编译。是响亮失败不是静默——这一点本身值得钉住，
    因为静默接受会给出错的数。
    """

    @pl.vector_function
    def _merging(src_a, src_b, dst):
        preg = vf.create_mask(pattern=pl.MaskPattern.ALL, dtype=FP32)
        a = vf.load_align(src_a, 0)
        b = vf.load_align(src_b, 0)
        out = vf.add(a, b, preg, mode=pl.MergeMode.MERGING)
        vf.store_align(dst, out, preg, 0)

    @pl.jit(auto_mutex=True, name="arith_merging")
    def merging(
        a: pl.Tensor[[1, N], FP32],
        b: pl.Tensor[[1, N], FP32],
        out: pl.Tensor[[1, N], FP32],
    ):
        tt = pl.TileType(shape=[1, N], dtype=FP32, target_memory=pl.MemorySpace.Vec)
        ga = pl.make_tile_group(type=tt, addrs=[0], mutex_ids=[0])
        gb = pl.make_tile_group(type=tt, addrs=[256], mutex_ids=[1])
        go = pl.make_tile_group(type=tt, addrs=[512], mutex_ids=[2])
        with pl.section_vector():
            ta, tb, to = ga.next(), gb.next(), go.next()
            pl.load(ta, a, [0, 0])
            pl.load(tb, b, [0, 0])
            _merging(ta, tb, to)
            pl.store(out, to, [0, 0])

    a, b = ref.make_operands(N)
    out = torch.zeros([1, N], device=device, dtype=torch.float32)
    with pytest.raises(RuntimeError) as raised:
        merging(_dev(a, device), _dev(b, device), out)

    text = str(raised.value)
    assert "NOT_IMPLEMENTED" in text or "not implemented" in text.lower(), text


def test_addc_and_subc_carry_in_is_the_hardware_carry_not_a_borrow(device):
    """验证 6。四个进位组合。

    `subc` 那两行是这条测试的重点：``ALL`` 给出 `a - b`，``ALLF`` 给出 `a - b - 1`——和参数名
    `borrow_src` 的字面读法相反。写错这一条不会报错，只会差 1。
    """
    a, b = ref.make_carry_operands(N)
    outs = _outs(device, 4, dtype=torch.int32)
    arith_carry(_dev(a, device), _dev(b, device), *outs)
    torch.npu.synchronize()

    want = ref.carry(a, b)
    names = ("addc carry=0", "addc carry=1", "subc 不借位", "subc 借位 1")
    for name, got, expected in zip(names, outs, want):
        np.testing.assert_array_equal(got.cpu().numpy(), expected, err_msg=name)

    # 把「两边读法相反」单独说一遍：subc(ALL) 与 subc(ALLF) 必须正好差 1
    sub_no_borrow = outs[2].cpu().numpy().astype(np.int64)
    sub_borrow = outs[3].cpu().numpy().astype(np.int64)
    np.testing.assert_array_equal(
        sub_no_borrow - sub_borrow, np.ones_like(sub_no_borrow)
    )
