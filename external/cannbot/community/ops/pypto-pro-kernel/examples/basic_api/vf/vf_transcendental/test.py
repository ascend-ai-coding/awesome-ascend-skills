# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

r"""八个超越函数：`ln` `log` `log2` `log10` `exp` `exp_sub` `sqrt` `abs`。

**写法：原始。** 前置样例是 [`vf_entry`](../vf_entry/test.py) 与 [`vf_arith`](../vf_arith/test.py)。

## 一、这是第一例不能逐位比的样例

前面四例都能 `rtol=0, atol=0`——操作数挑成 2 的幂或小整数，硬件和 numpy 走同一条精确路径。
**超越函数做不到**：硬件是查表加多项式，numpy 是 libm，不可能逐位相同。

所以判据必须量出来。实测（2026-09-27，FP32，输入 `[0.5, 8.0]`）最大相对误差：

| 函数 | 最大相对误差 | FP32 1 ulp |
| --- | --- | --- |
| `ln` / `log` | 5.4e-08 | ≈ 1.2e-07 |
| `exp` | 7.2e-08 | ≈ 1.2e-07 |

两个都在 1 ulp 之内。阈值取 **`rtol=1e-6`**，是实测值的十几倍——留了换输入区间的余量，
又远小于「算错一个数量级」。**阈值不是拍的**，实测值写在这张表里，下次有人往上调时能看出
他在容忍比当初大多少倍的误差。这条是 [skill constraints](../../../../SKILL.md#constraints-and-workspace-policy)「不静默放宽
比较阈值」的具体做法。

## 二、`log` 与 `ln`：数值相同，但 IR 里是**两个** op

`_vf_api.py` 原话：*"Natural logarithm (alias for :func:`vf.ln`)"*。我据此断言 IR 里只该有
`vf.ln`——**测下来是反的**：`vf.ln` 和 `vf.log` 各一个。

所以「别名」说的是**语义**（同一个数学函数），不是实现。写这条测试的理由是「数值相同不足以
证明是同一个 op」，而验出来的恰好是数值相同、op 不同——理由成立，结论与预期相反。

## 三、`precision=` 进了 IR，但 FP32 上结果逐位相同

`ln` / `log` / `log2` / `log10` / `exp` / `sqrt` 都有 `precision: Optional[bool]` 形参
（`abs` 和 `exp_sub` 没有）。实测：

- **三对（`ln` / `sqrt` / `log2`）的结果逐位相同**，最大相对误差也一样。
- 但它**确实进了 IR**：`@vf.ln(..., precision="precision": True)`。

数值相同只说明当前输入没有区分两种模式。后端确实生成了处理次正规数的补偿路径，
不能由此推断 precision 被忽略。
`test_the_precision_flag_reaches_the_ir` 比 IR，`test_the_precision_flag_changes_nothing_on_fp32`
比数值，分别检查参数传递与这个输入域的结果。

**没测的不声称**：其他 dtype（FP16 / BF16）、其他输入区间、`log10` 与 `exp` 的 `precision`
都没测，所以本例**不说**「这个参数没有用」。

## 四、`exp_sub` 是一条指令，不是两条

`exp_sub(a, b) = exp(a - b)`。ST 里调了 93 次——softmax 的 `exp(x - max)` 就是它。
签名和其余七个不同：**没有 `mode=`，多一个 `layout=`**。

`test_exp_sub_agrees_with_sub_then_exp` 把融合版和手写的 `sub` + `exp` 并排算。两者数学上
相同、数值上**可以有差**（中间精度不同）。实测差多少写在那条测试里——如果将来它们开始超出
`rtol`，说明融合指令的中间精度变了，那是值得知道的事。

## 五、输入区间是有讲究的

| 用途 | 区间 | 为什么 |
| --- | --- | --- |
| 对数 / `sqrt` / `exp` | `[0.5, 8.0]` | 全正（对数的定义域）；`exp(8) ≈ 2981` 不溢出 |
| `abs` | `[-8, 8]` | **必须跨正负**，否则 `abs` 这一条是空测 |
| `exp_sub` | `a - b ∈ [-2, 2]` | 不溢出也不下溢到 0 |

负数进 `ln`（NaN）、`sqrt(-1)`、`exp` 溢出都**没测**，本例不声称它们的行为。

## 六、precision 的目标变量可以复用

临时名按调用分配后，同名连续调用应与独立名字逐位一致；
`test_precision_calls_can_reuse_the_destination_name` 覆盖 FP16/FP32、相同/不同算子。
CPU 参考仍独立校验计算结果，不能用命名对照代替数学正确性。
"""

import re

import numpy as np
import pytest
import torch

import pypto_pro as pp
import pypto_pro.language as pl
from pypto_pro.language import Vf as vf  # noqa: N813

from . import reference as ref
from .kernel import (
    N,
    exp_sub_vs_manual,
    precision_pairs,
    transcendental,
)

_NAMES = ["ln", "log", "log2", "log10", "exp", "sqrt", "abs"]
_ALL_APIS = _NAMES + ["exp_sub"]


def _dev(array, device):
    return torch.from_numpy(array).to(device)


def _outs(device, count):
    return [
        torch.zeros((1, N), dtype=torch.float32, device=device) for _ in range(count)
    ]


def _run_elementwise(device, a):
    outs = _outs(device, 7)
    transcendental(_dev(a, device), *outs)
    torch.npu.synchronize()
    return [tensor.cpu().numpy() for tensor in outs]


def _max_rel(got, want):
    g, w = got.astype(np.float64), want.astype(np.float64)
    return float(np.max(np.abs(g - w) / np.maximum(np.abs(w), 1e-30)))


def test_the_seven_elementwise_functions_match_within_one_ulp(device):
    """七个函数各自比，并且**报出实测的最大相对误差**——阈值的依据就在这些数字里。"""
    a = ref.make_operands(N)
    got = _run_elementwise(device, a)
    want = ref.elementwise(a)

    for name, g, w in zip(_NAMES, got, want):
        if name == "abs":
            continue  # abs 用含负数的输入单独验，见下一条
        rel = _max_rel(g, w)
        assert rel <= ref.RTOL, (
            f"vf.{name} 的最大相对误差 {rel:.3e} 超过阈值 {ref.RTOL:.0e}。"
            f"如果是实现变了，把实测值更新进 reference 的表；不要直接调大阈值"
        )
        np.testing.assert_allclose(
            g, w, rtol=ref.RTOL, atol=0, err_msg=f"vf.{name} 不一致"
        )


def test_abs_needs_negative_input_to_mean_anything(device):
    """`abs` 用跨正负的输入单独验。正数输入下 `abs` 是恒等，测不出东西。"""
    a = ref.make_signed_operands(N)
    got = _run_elementwise(device, a)
    abs_out = got[6]

    assert (a < 0).any(), "输入没有负数，这一条是空的"
    np.testing.assert_allclose(
        abs_out, np.abs(a), rtol=0, atol=0, err_msg="abs 不做近似，应该逐位相同"
    )
    negative = a < 0
    assert (abs_out[negative] == -a[negative]).all(), "负半边应该取反"


def test_log_and_ln_are_separate_ops_with_identical_results():
    """不需要卡。**`log` 在 IR 里是独立的 op，不是 `ln` 的同一个 op。**

    docstring 写 *"alias for :func:`vf.ln`"*，我据此断言 IR 里只该有 `vf.ln`——**测下来是反的**：
    `vf.ln` 和 `vf.log` 各一个。所以「别名」说的是**语义**（同一个数学函数），不是实现。

    这一条的价值正在于此：写它的理由是「数值相同不足以证明是同一个 op」，
    结果验出来的恰恰是数值相同而 op 不同。数值一致由
    `test_the_seven_elementwise_functions_match_within_one_ulp` 覆盖。
    """
    program, _ = transcendental.to_kernel_def().parse_target_program(
        pp.ir.SectionKind.Vector
    )
    text = pp.ir.python_print(program)
    ops = re.findall(r"@(vf\.[\w]+)\(", text)

    assert ops.count("vf.ln") == 1, f"vf.ln 应出现一次，实际 {ops.count('vf.ln')}"
    assert ops.count("vf.log") == 1, (
        f"vf.log 应出现一次。如果变成 0，说明前端开始把 log 折叠成 ln 了——"
        f"那是行为变化，改这条测试。实际 op：{sorted(set(ops))}"
    )


def test_the_precision_flag_reaches_the_ir():
    """不需要卡。`precision=True` 进了 IR——这一条与下一条合起来才说明问题在后端。"""
    program, _ = precision_pairs.to_kernel_def().parse_target_program(
        pp.ir.SectionKind.Vector
    )
    text = pp.ir.python_print(program)
    marker = 'precision="precision": True'
    assert marker in text, (
        "IR 里找不到 precision=True，说明前端把这个参数丢了——那下一条测试的解释就不同了"
    )
    seen = text.count(marker)
    assert seen == 3, (
        f"三个函数各传了一次 precision=True，IR 里应该有三处，实际 {seen} 处"
    )


def test_the_precision_flag_changes_nothing_on_fp32(device):
    """`precision=True` 与默认在 FP32 上**逐位相同**。

    只说明所选输入域下结果相同，不能推出后端忽略了该参数。
    **只针对这三个函数、FP32、`[0.5, 8]`**；其他情况没测。

    若结果变化，应重新核对输入域和数值误差，再决定期望，不能自动放宽判据。
    """
    a = ref.make_operands(N)
    outs = _outs(device, 6)
    precision_pairs(_dev(a, device), *outs)
    torch.npu.synchronize()
    got = [tensor.cpu().numpy() for tensor in outs]

    for name, default, precise in (
        ("ln", got[0], got[1]),
        ("sqrt", got[2], got[3]),
        ("log2", got[4], got[5]),
    ):
        assert np.array_equal(default, precise), (
            f"vf.{name} 的 precision=True 开始产生不同结果了——"
            f"最大相对差 {_max_rel(precise, default):.3e}。这是行为变化，改测试别删测试"
        )


def test_exp_sub_agrees_with_sub_then_exp(device):
    """`exp_sub(a, b)` 与手写的 `sub` + `exp` 一致。

    两条路径数学上相同，数值上可以有差（融合指令的中间精度不同）。这里断言在 `rtol` 内一致,
    并报出实测差值——它变大就说明中间精度变了。
    """
    a, b = ref.make_exp_sub_operands(N)
    outs = _outs(device, 2)
    exp_sub_vs_manual(_dev(a, device), _dev(b, device), *outs)
    torch.npu.synchronize()
    fused, manual = (tensor.cpu().numpy() for tensor in outs)

    want = ref.exp_sub(a, b)
    rel_fused = _max_rel(fused, want)
    assert rel_fused <= ref.RTOL, (
        f"exp_sub 相对 exp(a-b) 的最大相对误差 {rel_fused:.3e} 超过 {ref.RTOL:.0e}"
    )

    rel_between = _max_rel(fused, manual)
    assert rel_between <= ref.RTOL, (
        f"融合版与手写版的最大相对差 {rel_between:.3e} 超过 {ref.RTOL:.0e}——"
        f"融合指令的中间精度可能变了"
    )


def test_exp_sub_is_not_just_exp_of_the_first_operand(device):
    """防空测：如果 `b` 被忽略，结果会是 `exp(a)`。`b` 恒为 2，两者必然不同。"""
    a, b = ref.make_exp_sub_operands(N)
    outs = _outs(device, 2)
    exp_sub_vs_manual(_dev(a, device), _dev(b, device), *outs)
    torch.npu.synchronize()
    fused = outs[0].cpu().numpy()

    exp_a_only = np.exp(a.astype(np.float64)).astype(np.float32)
    assert _max_rel(fused, exp_a_only) > 1e-3, (
        "exp_sub 的结果等于 exp(a)，说明第二个操作数没被用上"
    )


@pytest.mark.parametrize(
    "dtype,torch_dtype,n",
    [(pl.DT_FP32, torch.float32, 64), (pl.DT_FP16, torch.float16, 128)],
)
@pytest.mark.parametrize("repeat", [False, True])
def test_precision_calls_can_reuse_the_destination_name(
    device, dtype, torch_dtype, n, repeat
):
    """Reused and distinct names must produce identical results, including FP16."""

    def build(body):
        @pl.jit(auto_mutex=True)
        def k(
            a: pl.Tensor[[1, n], dtype],
            o1: pl.Tensor[[1, n], dtype],
            o2: pl.Tensor[[1, n], dtype],
        ):
            tt = pl.TileType(
                shape=[1, n], dtype=dtype, target_memory=pl.MemorySpace.Vec
            )
            ga = pl.make_tile_group(type=tt, addrs=0x0000, mutex_ids=[0])
            g1 = pl.make_tile_group(type=tt, addrs=0x0400, mutex_ids=[1])
            g2 = pl.make_tile_group(type=tt, addrs=0x0800, mutex_ids=[2])
            with pl.section_vector():
                ta, t1, t2 = ga.current(), g1.current(), g2.current()
                pl.load(ta, a, [0, 0])
                body(ta, t1, t2)
                pl.store(o1, t1, [0, 0])
                pl.store(o2, t2, [0, 0])

        return k

    @pl.vector_function
    def same_name(src, o1, o2):
        preg = vf.create_mask(pattern=pl.MaskPattern.ALL, dtype=dtype)
        a = vf.load_align(src, 0)
        r = vf.ln(a, preg, precision=True)
        vf.store_align(o1, r, preg, 0)
        r = vf.sqrt(a, preg, precision=True)
        vf.store_align(o2, r, preg, 0)

    @pl.vector_function
    def repeated_name(src, o1, o2):
        preg = vf.create_mask(pattern=pl.MaskPattern.ALL, dtype=dtype)
        a = vf.load_align(src, 0)
        r = vf.ln(a, preg, precision=True)
        vf.store_align(o1, r, preg, 0)
        r = vf.ln(a, preg, precision=True)
        vf.store_align(o2, r, preg, 0)

    @pl.vector_function
    def distinct_names(src, o1, o2):
        preg = vf.create_mask(pattern=pl.MaskPattern.ALL, dtype=dtype)
        a = vf.load_align(src, 0)
        first = vf.ln(a, preg, precision=True)
        vf.store_align(o1, first, preg, 0)
        second = vf.sqrt(a, preg, precision=True)
        vf.store_align(o2, second, preg, 0)

    a = torch.from_numpy(ref.make_operands(n)).to(device=device, dtype=torch_dtype)
    actual = [torch.empty_like(a), torch.empty_like(a)]
    control = [torch.empty_like(a), torch.empty_like(a)]
    build(repeated_name if repeat else same_name)(a, *actual)
    build(distinct_names)(a, *control)
    torch.npu.synchronize()
    assert torch.equal(actual[0], control[0])
    assert torch.equal(actual[1], control[0 if repeat else 1])


@pytest.mark.parametrize("name", _ALL_APIS)
def test_every_api_in_this_example_is_actually_called(name):
    """不需要卡。防漏：表里列的 8 个函数每个都得在 kernel 源码里出现。"""
    from pathlib import Path

    source = (Path(__file__).parent / "kernel.py").read_text()
    assert f"vf.{name}(" in source, f"kernel.py 里没有调用 vf.{name}()"
