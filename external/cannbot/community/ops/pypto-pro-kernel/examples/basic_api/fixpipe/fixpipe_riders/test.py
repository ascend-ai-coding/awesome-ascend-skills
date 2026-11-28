# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

"""fixpipe 的四个 rider：`relu_pre_mode` `atomic` `phase` `order`。

**写法：原始。** 前置样例：[`fixpipe_paths`](../fixpipe_paths/test.py)（三条出路）与
[`fixpipe_quant_matrix`](../fixpipe_quant_matrix/test.py)（`scale` 的 dtype 矩阵）。
四个 rider 各自是什么在 [`kernel.py`](kernel.py)。

## 一、每个 rider 都有一个对照组

这是这一例的形状。rider 的效果只能靠**两份输出对比**来判：

| rider | 对照组 | 差异应当是 |
| --- | --- | --- |
| `relu_pre_mode` | 同一条链不带 relu | 负数变 0，正数不变 |
| `atomic=AtomicAdd` | `AtomicNone` | 跑两次：累加翻倍，覆盖不变 |
| `phase` | 不带 phase | **没有差异**——它是同步优化 |
| `order=[0, 2]` | 第 0 页（没被写的那页） | 数据落在第 `PAGE` 页，另一页仍是 0 |

单看一份输出，「relu 生效」和「输入恰好全正」分不开，「atomic 生效」和
「第二次覆盖了」分不开。**所以每条都要对照组。**

## 二、`ReluPreMode` 只有一个成员不是缺口

ISA 侧的 `pto::ReluPreMode` 也只有 `NoRelu` / `NormalRelu`，Python 侧
「1 个成员 + `None` 表示不开」是完整的。

ISA 另有一个**独立**的 `ClipReluMode_t`（`NOCLIP_RELU` / `CLIP_RELU`），
**Python 侧没有暴露**——那是真缺口，记在设计文档 §十四点二。
两件事名字像，容易当成一个。

## 三、`phase` 的判据是「没有差异」

`phase` 让 fixpipe 用 unit_flag 而不是显式同步等矩阵乘的结果，省一次同步开销。
**它不改数值。** 所以这一条断言的是「带 phase 和不带 phase 的结果一致」——
一个「结果变了」的断言在这里反而说明出了问题。

性能上的收益本例不测（[skill constraints](../../../../SKILL.md#constraints-and-workspace-policy)：性能单独报）。要测得用
[原生任务计时](../../../../references/verification/native-timing.md) 那套差分法，
而且这个规模下一次同步的开销在噪声以下。

## 四、`order` 不是转置

初版把 `order=[1, 0]` 当转置写，上卡撞 `order must be ascending`
（`block_ops.py:347`）。`order` 回答的是**「2 维 Tile 落到高维 Tensor 的哪两根轴」**，
升序是硬约束，所以它关不通转置这条路。改成三条：
一条落位（升序、跳过中间轴），两条拒绝（降序、NZ 目的）。

## 五、`atomic` 把「默认 block_dim 不是 1」暴露了出来

初版 `store_atomic_add(...)` 直接调用，第一次写就拿到 `28 × scale × acc`。
不是 atomic 错了——是**默认 block_dim 是当前 stream 的全量核数**
（`jit.py:1883`，`_AUTO_BLOCK_DIM = 0`），这些 kernel 不按 `get_block_idx()` 分工，
每个 block 都往同一片 GM 累加一份。

**覆盖模式下这件事是隐形的**：每个 block 写的字节完全相同。
所以 `atomic` 这一组顺带成了「有几个 block 在跑」的探针——
`test_atomic_add_reveals_the_block_count` 把它写成测试。

实测这台机器 `get_platform_info()` 报 `core_num=28 / cube=28 / vector=56`，
本例（纯 cube 段）默认启动的倍数是 **28**；
[`support/debug_probe`](../../support/debug_probe/test.py) 里纯 vector 段的
`get_block_num()` 报 **56**。两个数都是实测，机制不在本例的范围内——
要用就用 `get_block_num()` 分工，别在 host 侧写死。

## 六、判据

`rtol=0, atol=0`。目的类型 FP16、`scale=0.5`、输入小整数，`scale*acc` 在 FP16 精确。
**没选 FP32 是为了避开 `FP32→FP32` 丢 scale 那个缺陷**
（[defect](../../../../references/diagnostics/defects/fixpipe-fp32-to-fp32-drops-scale.md)）——
用一个已知有缺陷的载体做 rider 的判据，会把两件事混在一起。
"""

import numpy as np
import pytest
import torch
from pypto_pro._errors import InvalidFormat, InvalidVal

from . import reference as ref
from .kernel import (
    K,
    M,
    N,
    PAGE,
    PAGES,
    SCALE,
    store_atomic_add,
    store_no_phase,
    store_order_descending,
    store_order_nz_out,
    store_order_page,
    store_overwrite,
    store_phase_pair,
    store_with_relu,
    store_without_relu,
)


def _dev(array, device):
    return torch.from_numpy(array).to(device)


def _out(device):
    return torch.zeros((M, N), dtype=torch.float16, device=device)


def _run(kernel, device):
    a, b = ref.make_operands(M, K, N)
    out = _out(device)
    kernel(_dev(a, device), _dev(b, device), out)
    torch.npu.synchronize()
    return a, b, out.cpu().to(torch.float32).numpy().astype(np.float64)


# ---------------------------------------------------------------------------
# relu_pre_mode
# ---------------------------------------------------------------------------


def test_the_accumulator_has_negatives():
    """**relu 那两条的前提。** 输入全正时 relu 是恒等映射，验不到任何东西。

    这条不需要卡：它只算参考。放在最前面，因为 relu 的两条测试都依赖它。
    """
    a, b = ref.make_operands(M, K, N)
    acc = ref.accumulate(a, b)
    n_neg = int((acc < 0).sum())
    assert n_neg > 0, "acc 里没有负数——relu 那两条会验空"
    assert n_neg < acc.size, "acc 全是负数——那 relu 之后全 0，也验不到正数不变"


def test_relu_clamps_the_negatives(device):
    """`relu_pre_mode=NormalRelu`：负数变 0。"""
    a, b, got = _run(store_with_relu, device)
    np.testing.assert_allclose(
        got,
        ref.scaled_relu(a, b, SCALE),
        rtol=0,
        atol=0,
        err_msg="带 relu 的结果不是 max(0, scale*acc)",
    )


def test_relu_is_not_a_no_op(device):
    """对照组：**不带 relu 的同一条链必须有负数**，否则 relu 那条验空了。

    并且两份输出只应在原本为负的位置不同——正数位置必须逐位相同。
    """
    a, b, with_relu = _run(store_with_relu, device)
    _, _, without = _run(store_without_relu, device)

    assert (without < 0).any(), "不带 relu 的输出没有负数——relu 那条验空了"
    assert not np.array_equal(with_relu, without), "带与不带 relu 的输出相同"

    positive = without > 0
    np.testing.assert_allclose(
        with_relu[positive],
        without[positive],
        rtol=0,
        atol=0,
        err_msg="relu 改动了本来是正数的位置",
    )
    assert (with_relu[without < 0] == 0).all(), "原本为负的位置没有全部变成 0"


# ---------------------------------------------------------------------------
# atomic
# ---------------------------------------------------------------------------


#: FP16 在 `[1024, 2048)` 区间的间距是 **1.0**，半整数在那里就不精确了。
#: 1024 以下间距 ≤ 0.5，而这些 kernel 写的都是 0.5 的整数倍，所以精确。
FP16_EXACT_HALVES_BELOW = 1024


def test_atomic_add_accumulates_across_launches(device):
    """`atomic=AtomicAdd`：**同一个 kernel 跑两次，结果翻倍**。

    调用方负责清零——`AtomicAdd` 加在 GM 原有内容上。这里从零张量开始。

    这条验的是「累加」这个语义本身，不是「和参考比一次」。

    **`[None, 1]` 不能省**：默认 block_dim 是全量核数，一次启动就会累加
    `block_dim` 份（见模块 docstring §五）。
    """
    a, b = ref.make_operands(M, K, N)
    out = _out(device)
    once = ref.scaled(a, b, SCALE)

    store_atomic_add[None, 1](_dev(a, device), _dev(b, device), out)
    torch.npu.synchronize()
    after_one = out.cpu().to(torch.float32).numpy().astype(np.float64)
    np.testing.assert_allclose(
        after_one,
        once,
        rtol=0,
        atol=0,
        err_msg="单 block 的第一次 atomic 写（从零开始）应当等于 scale*acc",
    )

    store_atomic_add[None, 1](_dev(a, device), _dev(b, device), out)
    torch.npu.synchronize()
    after_two = out.cpu().to(torch.float32).numpy().astype(np.float64)
    np.testing.assert_allclose(
        after_two,
        once * 2.0,
        rtol=0,
        atol=0,
        err_msg="第二次 atomic 写没有累加到第一次上",
    )


def test_atomic_add_scales_with_block_dim(device):
    """**倍数就是 block_dim**：`[None, 2]` 一次启动拿到 2×。

    kernel 不按 `get_block_idx()` 分工，两个 block 算同一份、往同一片 GM 各加一次。
    这一条把上一条的 `[None, 1]` 从「照抄官方 ST」变成「知道为什么」。
    """
    a, b = ref.make_operands(M, K, N)
    want = ref.scaled(a, b, SCALE)

    out = _out(device)
    store_atomic_add[None, 2](_dev(a, device), _dev(b, device), out)
    torch.npu.synchronize()
    got = out.cpu().to(torch.float32).numpy().astype(np.float64)

    np.testing.assert_allclose(
        got,
        want * 2.0,
        rtol=0,
        atol=0,
        err_msg="block_dim=2 的一次 atomic 启动不是 2 × scale*acc",
    )


def test_atomic_add_reveals_the_block_count(device):
    """不指定 block_dim 时的倍数**大于 1**——默认 block_dim 不是 1。

    这是初版失败的根因，写成测试而不是只写进注释：它是一条关于**启动语义**的事实，
    而且 `AtomicAdd` 是唯一能从输出看出来的地方（覆盖模式下每个 block 写的字节相同）。

    **不写死具体倍数**——那取决于当前 stream 的核数预算。倍数从总量估出来，
    只断言它 ≥ 2。

    **逐位比只在 FP16 精确的那部分做。** `k × scale × acc` 会冲出 FP16
    「半整数精确」的范围（1024 以上间距变成 1.0），所以全量逐位比一定挂——
    初版就是这么挂的，`28 × 19.5 = 546` 对，`28 × 50 = 1400` 附近的中间和不对。
    精确性由 `[None, 1]` / `[None, 2]` 那两条负责，这一条只负责「默认不是 1」。
    """
    a, b = ref.make_operands(M, K, N)
    want = ref.scaled(a, b, SCALE)

    out = _out(device)
    store_atomic_add(_dev(a, device), _dev(b, device), out)  # 故意不带 [None, n]
    torch.npu.synchronize()
    got = out.cpu().to(torch.float32).numpy().astype(np.float64)

    k = int(round(np.abs(got).sum() / np.abs(want).sum()))
    assert k >= 2, (
        f"默认 block_dim 估出来是 {k}——如果真是 1，模块 docstring §五 那段解释就该删掉"
    )

    # 每个 block 加的是同一个值，所以中间和的量级单调增到 |k * want|：
    # 只要终值在 1024 以下，途中每一步都是 1024 以下的 0.5 整数倍，逐位精确。
    exact = np.abs(want) * k < FP16_EXACT_HALVES_BELOW
    assert exact.any(), "没有一个位置落在 FP16 精确区间——这条的逐位比验空了"
    np.testing.assert_allclose(
        got[exact],
        want[exact] * k,
        rtol=0,
        atol=0,
        err_msg=f"FP16 精确区间内的输出不是 {k} × scale*acc",
    )


def test_overwrite_does_not_accumulate(device):
    """对照组：`AtomicNone`（默认）跑两次结果不变。

    有了这一条，上面那几条的「翻倍」才说明是 `atomic` 起作用，
    而不是「写两次总会翻倍」。

    这里**故意不指定 block_dim**：全量 block 都在写同一片 GM，覆盖模式下结果
    仍然正好是 1×——这正是上一条说的「覆盖模式下 block 数是隐形的」。
    """
    a, b = ref.make_operands(M, K, N)
    out = _out(device)
    for _ in range(2):
        store_overwrite(_dev(a, device), _dev(b, device), out)
    torch.npu.synchronize()
    np.testing.assert_allclose(
        out.cpu().to(torch.float32).numpy().astype(np.float64),
        ref.scaled(a, b, SCALE),
        rtol=0,
        atol=0,
        err_msg="覆盖模式下跑两次结果变了",
    )


# ---------------------------------------------------------------------------
# phase
# ---------------------------------------------------------------------------


def test_phase_does_not_change_the_result(device):
    """`AccPhase.Partial/Final` 配 `STPhase.Partial/Final`：**结果和不带 phase 一致**。

    `phase` 是同步优化（fixpipe 用 unit_flag 等矩阵乘），不是数值语义。
    所以判据是「没有差异」——见模块 docstring §三。

    两个 kernel 都累加两次（`matmul` + `matmul_acc`），所以期望是 `scale * 2 * acc`。
    """
    a, b, with_phase = _run(store_phase_pair, device)
    _, _, without = _run(store_no_phase, device)
    want = ref.accumulate_twice(a, b) * SCALE

    np.testing.assert_allclose(
        with_phase, want, rtol=0, atol=0, err_msg="带 phase 的结果不是 scale * 2 * acc"
    )
    np.testing.assert_allclose(
        with_phase,
        without,
        rtol=0,
        atol=0,
        err_msg="带 phase 与不带 phase 的结果不同——phase 不该改数值",
    )


def test_matmul_acc_really_accumulates(device):
    """防空测：`matmul` + `matmul_acc` 的结果必须是单次的两倍。

    如果 `matmul_acc` 覆盖而不是累加，上一条的期望值就错了而它照样会过
    （两个 kernel 都错成同一个值）。**这一条把参考的前提钉住。**
    """
    a, b, twice = _run(store_no_phase, device)
    _, _, once = _run(store_without_relu, device)  # 单次 matmul，同样的 scale

    np.testing.assert_allclose(
        twice, once * 2.0, rtol=0, atol=0, err_msg="matmul_acc 没有累加——它覆盖了"
    )


# ---------------------------------------------------------------------------
# order
# ---------------------------------------------------------------------------


def test_order_selects_which_two_axes_the_tile_lands_on(device):
    """`order=[0, 2]` 把 2 维 Tile 落到 3 维 Tensor 的轴 0 和轴 2，跳过中间那根。

    **别和 `layout=` 混**：`layout=` 是 fixpipe 改排布（NZ→ND/DN/NZ），
    `order=` 是地址计算选轴。

    对照组是**没被写的那一页**：数据落在第 `PAGE` 页，另一页必须还是 0。
    只验「第 PAGE 页对」的话，一个忽略 `order` 而写满整个 Tensor 的实现也能过。
    """
    a, b = ref.make_operands(M, K, N)
    out = torch.zeros((M, PAGES, N), dtype=torch.float16, device=device)
    store_order_page(_dev(a, device), _dev(b, device), out)
    torch.npu.synchronize()
    got = out.cpu().to(torch.float32).numpy().astype(np.float64)

    np.testing.assert_allclose(
        got[:, PAGE, :],
        ref.scaled(a, b, SCALE),
        rtol=0,
        atol=0,
        err_msg=f"第 {PAGE} 页不是 scale*acc",
    )
    other = 1 - PAGE
    assert not got[:, other, :].any(), (
        f"第 {other} 页被写了——order 没有把中间那根轴当成下标"
    )


def test_a_descending_order_is_rejected(device):
    """降序 `order=[2, 0]`：**解析期拒绝**，这是 `order` 关不通转置的原因。

    不需要卡上执行（拒绝发生在 parse），但要求 device 已选好——
    launcher 在 parse 之前会先校验 arch。传 CPU 张量就够。
    """
    a, b = ref.make_operands(M, K, N)
    out = torch.zeros((M, PAGES, N), dtype=torch.float16)
    with pytest.raises(InvalidVal, match="order must be ascending"):
        store_order_descending(torch.from_numpy(a), torch.from_numpy(b), out)


def test_a_nz_destination_only_accepts_the_last_two_axes(device):
    """同一个升序 `order=[0, 2]`，目的 Tensor 换成 NZ——换一条规则拒绝。

    `order` 的自由度只对 ND 目的成立。两条拒绝的**理由不同**，
    所以分成两条测试而不是参数化成一条。
    """
    a, b = ref.make_operands(M, K, N)
    out = torch.zeros((M, PAGES, N), dtype=torch.float16)
    with pytest.raises(
        InvalidFormat, match="NZ transfer only supports the last two tensor axes"
    ):
        store_order_nz_out(torch.from_numpy(a), torch.from_numpy(b), out)


# ---------------------------------------------------------------------------
# 结构检查
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("name", ["relu_pre_mode", "atomic", "phase", "order"])
def test_every_rider_in_this_example_is_actually_passed(name):
    """不需要卡。防漏：四个 rider 每个都得在 kernel 源码里作为关键字传过。"""
    from pathlib import Path

    source = (Path(__file__).parent / "kernel.py").read_text()
    assert f"{name}=" in source, f"kernel.py 里没有传 {name}="
