# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

"""三个接口：`histograms` `clear_spr` `mem_bar`。

**写法：原始。** 前置样例是 [`vf_gather_scatter`](../vf_gather_scatter/test.py)——
`clear_spr` 那一条直接和它的 `addr_reg_walk` 对照，只差一个 `vf.clear_spr()`。

## 一、这三个接口的共同点：都不是纯函数

| 接口 | 状态在哪 | 本例怎么观察 |
| --- | --- | --- |
| `histograms` | **目的寄存器**（既读又写） | 同一份数据统计两次，计数翻倍 |
| `clear_spr` | **AddrReg / AR** | 和不清的版本对照，看偏移是否回到起点 |
| `mem_bar` | 流水线顺序约束 | 只验覆盖和「加了不改数值」 |

## 二、`histograms` 的四条约束，两条会静默出错

1. **src 必须 `DT_UINT8`，`preg` 必须 b8 粒度**——类型不对会被拦，不静默。
2. **dst 必须先 `vf.full(0, ...)` 初始化**：`ACCUMULATE` 是原地累加。
   不初始化就是在垃圾值上加，**没有任何报错**。
3. dst 是 `DT_UINT16`（128 个桶），最大 65535，累加多了溢出。
4. `BIN1` 下值减 128 映射到桶号：值 200 落在 `dst[72]`。

**host 侧没有 UINT16 的 torch 对应**（`_PL_DTYPE_TO_TORCH` 只有 11 项，见
[`dtype_carriers`](../../support/dtype_carriers/test.py)），所以存之前 `bit_cast` 成
`DT_INT16`。计数最多 256，碰不到符号位。

## 三、`mem_bar` 没有数值判据，这是有意的

屏障的作用是排除一类竞态，而**正确的代码不加屏障也可能恰好对**。官方文档还写明：
读写用同一个寄存器时硬件自己保序，根本不需要同步指令。所以「去掉屏障就会错」的反例
既不稳定也不说明问题。

这一例对 `mem_bar` 只断言两件能断言的事：12 种组合都能过编译并跑通，
以及**加了屏障不改变数值**。**说不清的地方要明说说不清**，不给它编一个判据。

（写这一版时我试过把这句话本身写成一条测试——grep 自己的源码看有没有夸大的声称。
那条测试只会匹配到它自己的禁词列表，必然失败。**断言不该建立在自己的散文上。**）

## 三点五、两个假设被实测推翻

| 我的假设 | 实测 |
| --- | --- |
| `clear_spr()` 会把 `create_addr_reg` 的循环偏移清回起点 | **不会。** 加了和不加逐位相同 |
| `(i % 8) * 32` 这组输入能验到 `bin_type` | **不能。** `v` 和 `v+128` 同时存在时两个半区的直方图一模一样，防空测抓住了 |

第二条是防空测的价值：主判据「和参考一致」当时是**过的**——因为参考也对称。
只有「两个 `bin_type` 必须不同」这条能发现输入选错了。

## 四、判据

`rtol=0, atol=0`。直方图是计数；`mem_bar` 那条的算术是 `a+a` 再 `+a`，输入取整数。
"""

from pathlib import Path

import numpy as np
import pytest
import torch

from . import reference as ref
from .kernel import (
    N_BINS,
    N_FP32,
    N_U8,
    addr_reg_cleared_each_round,
    histograms_accumulate_twice,
    histograms_four_ways,
    histograms_masked,
    make_single_histogram,
    mem_bar_all_twelve,
    mem_bar_raw,
)

_ALL_APIS = ["histograms", "clear_spr", "mem_bar"]

_TWELVE_MODES = [
    "VST_VLD",
    "VLD_VST",
    "VST_VST",
    "VST_LD",
    "VST_ST",
    "VLD_ST",
    "ST_VLD",
    "ST_VST",
    "LD_VST",
    "VV_ALL",
    "VS_ALL",
    "SV_ALL",
]


def _dev(array, device):
    return torch.from_numpy(array).to(device)


def _bins(device):
    return torch.zeros((1, N_BINS), dtype=torch.int16, device=device)


def _fp32(device, cols=N_FP32):
    return torch.zeros((1, cols), dtype=torch.float32, device=device)


# ---------------------------------------------------------------------------
# histograms
# ---------------------------------------------------------------------------


_MODES = [
    ("BIN0", "FREQUENCY", 0, ref.frequency),
    ("BIN1", "FREQUENCY", 1, ref.frequency),
    ("BIN0", "ACCUMULATE", 0, ref.accumulate),
    ("BIN1", "ACCUMULATE", 1, ref.accumulate),
]


def _run_single(bin_name, hist_name, device):
    """一个 kernel 只做一次 `histograms`——无混淆基准。"""
    pl = pytest.importorskip("pypto_pro.language")
    kernel = make_single_histogram(
        getattr(pl.BinType, bin_name), getattr(pl.HistType, hist_name)
    )
    a = ref.make_u8_data(N_U8)
    out = _bins(device)
    kernel(_dev(a, device), out)
    torch.npu.synchronize()
    return a, out.cpu().numpy()


@pytest.mark.parametrize(
    ("bin_name", "hist_name", "bin_idx", "expect"),
    _MODES,
    ids=[f"{b}-{h}" for b, h, _, _ in _MODES],
)
def test_each_histogram_mode_alone_matches_the_reference(
    bin_name, hist_name, bin_idx, expect, device
):
    """四种组合**各自一个 kernel**，逐个比。

    一次只做一个 `histograms`，就没有第二个既读又写的 dst 可以污染——这四条是基准，
    下面的合并写法和它们比。
    """
    a, got = _run_single(bin_name, hist_name, device)
    np.testing.assert_array_equal(
        got, expect(a, bin_idx), err_msg=f"{bin_name} + {hist_name} 不一致"
    )


def test_the_combined_kernel_agrees_with_the_single_ones(device):
    """**四个既读又写的 dst 挤在一个 `@pl.vector_function` 里，不互相污染。**

    这条不是形式主义：`precision=True` 那个 codegen 缺陷就是同名临时量撞车
    （见 `references/diagnostics/defects/precision-flag-temp-name-collision.md`），所以
    「四个 histograms 共存」需要先证明，不能假设。

    比的是合并 kernel 的四个输出 vs 四个单模式 kernel 的输出。
    """
    a = ref.make_u8_data(N_U8)
    outs = [_bins(device) for _ in range(4)]
    histograms_four_ways(_dev(a, device), *outs)
    torch.npu.synchronize()
    combined = [tensor.cpu().numpy() for tensor in outs]

    for (bin_name, hist_name, _, _), got in zip(_MODES, combined):
        _, alone = _run_single(bin_name, hist_name, device)
        np.testing.assert_array_equal(
            got,
            alone,
            err_msg=(
                f"合并 kernel 的 {bin_name}+{hist_name} 与单独跑的不一致——"
                "四个 dst 互相污染了"
            ),
        )


def test_bin0_and_bin1_see_different_value_ranges(device):
    """防空测：两个 `bin_type` 必须给出不同的直方图，否则没验到分桶。

    **这条抓到过一次真问题**：第一版输入是 `(i % 8) * 32`，值 `v` 和 `v+128` 同时存在，
    两个半区的桶号分布完全一样，`bin_type` 起不起作用根本看不出来。
    现在的输入是 `{0, 32, 128, 192}` 循环——`BIN0` 命中桶 `{0, 32}`，`BIN1` 命中 `{0, 64}`。

    两半的总计数加起来应该正好是输入的元素个数：所有值都被统计到一次，不重不漏。
    """
    _, f0 = _run_single("BIN0", "FREQUENCY", device)
    a, f1 = _run_single("BIN1", "FREQUENCY", device)

    assert not np.array_equal(f0, f1), "BIN0 与 BIN1 给出相同直方图——bin_type 没起作用"
    total = int(f0.sum()) + int(f1.sum())
    assert total == N_U8, f"两半总计数应为 {N_U8}，实际 {total}"
    # 高低值在寄存器里交错（周期 4），所以「按 lane 分半区」那种读法会被这条排除
    assert int(f0.sum()) == ref.count_below_128(a), (
        "BIN0 的总计数不等于输入里 <128 的元素个数——分桶可能是按 lane 而不是按值"
    )


def test_the_two_accumulate_halves_form_one_continuous_curve(device):
    """**`ACCUMULATE` 的累计是对整个 0..255 值域做的，`bin_type` 只选看哪 128 个桶。**

    `dst[n] = |{v : v <= n + 128*bin}|`。所以 **`BIN1` 的累计从 `BIN0` 结束的地方接着
    往上走**——两半拼成一条 0..255 的连续累计分布。

    这条全部用同一次运行的四个输出之间的**关系**来断言，不依赖参考实现：

    - `a0 == cumsum(f0)`：`BIN0` 里「区间内前缀和」和「全局前缀和」恰好相等；
    - `a1 == a0[-1] + cumsum(f1)`：`BIN1` 接着 `BIN0` 的终点往上走；
    - `a1[-1] == N_U8`：走到顶就是全部元素。

    写这一版时参考按「每个半区各自前缀和」写，**`BIN0` 凑巧过了、`BIN1` 差 128**。
    官方文档那句「从 0 到 n 的所有数值**在对应区间中**出现的总频率」读起来像限制在
    区间内，实测不是。
    """
    outs = [_bins(device) for _ in range(4)]
    histograms_four_ways(_dev(ref.make_u8_data(N_U8), device), *outs)
    torch.npu.synchronize()
    f0, f1, a0, a1 = (
        tensor.cpu().numpy().reshape(-1).astype(np.int64) for tensor in outs
    )

    np.testing.assert_array_equal(
        a0, np.cumsum(f0), err_msg="BIN0 的累计不是自己频率的前缀和"
    )
    np.testing.assert_array_equal(
        a1,
        a0[-1] + np.cumsum(f1),
        err_msg=(
            "BIN1 的累计没有接着 BIN0 的终点——"
            "如果它等于 cumsum(f1)，说明累计被限制在半区内了"
        ),
    )
    assert a1[-1] == N_U8, f"累计走到顶应为 {N_U8}，实际 {a1[-1]}"
    assert a0[-1] != a1[0], (
        "BIN0 的终点等于 BIN1 的起点——两半没有拼成一条连续曲线，这条就验空了"
    )


def test_a_second_call_accumulates_instead_of_overwriting(device):
    """**这一条是 `histograms` 最需要知道的事**：目的寄存器既读又写。

    同一份数据统计两次，计数正好翻倍。如果它是纯函数（覆盖），两个输出会相同。
    """
    a = ref.make_u8_data(N_U8)
    once, twice = _bins(device), _bins(device)
    histograms_accumulate_twice(_dev(a, device), once, twice)
    torch.npu.synchronize()

    once_arr, twice_arr = once.cpu().numpy(), twice.cpu().numpy()
    assert not np.array_equal(once_arr, twice_arr), (
        "统计一次和两次结果相同——说明第二次覆盖了而不是累加，"
        "那 dst 就不需要预初始化，本例的前提要重写"
    )
    np.testing.assert_array_equal(
        twice_arr, once_arr * 2, err_msg="第二次统计应当让计数正好翻倍"
    )


def test_masked_out_lanes_do_not_count(device):
    """`preg` 为 0 的 lane 不参与统计：屏蔽掉所有 ≥ 128 的值，`BIN1` 必须全 0。

    「全 0」比「某个桶少几个」好验——少一点可能有别的解释，全 0 只能是掩码起了作用。
    """
    a = ref.make_u8_data(N_U8)
    bin0, bin1 = _bins(device), _bins(device)
    histograms_masked(_dev(a, device), bin0, bin1)
    torch.npu.synchronize()

    got0, got1 = bin0.cpu().numpy(), bin1.cpu().numpy()
    assert (got1 == 0).all(), (
        f"屏蔽掉 ≥128 的值之后 BIN1 应当全 0，实际非零 {int((got1 != 0).sum())} 个桶"
    )
    np.testing.assert_array_equal(
        got0, ref.frequency_masked_below_128(a, 0), err_msg="BIN0 的计数不应受影响"
    )
    want_total = ref.count_below_128(a)
    assert got0.sum() == want_total, (
        f"BIN0 总计数应为输入里 <128 的元素个数 {want_total}，实际 {int(got0.sum())}"
    )


# ---------------------------------------------------------------------------
# clear_spr
# ---------------------------------------------------------------------------


def test_clear_spr_does_not_reset_the_addr_reg_loop_offset(device):
    """**`clear_spr()` 不清 `create_addr_reg` 的循环偏移。**

    和 [`vf_gather_scatter`](../vf_gather_scatter/kernel.py) 的 `addr_reg_walk` 逐行对照,
    只差循环体开头多一个 `vf.clear_spr()`——**结果逐位相同**：两轮照样搬
    `[0:64]` 和 `[64:128]`，输出等于输入。

    写这一版时我的假设是反的（以为偏移会回到起点，两轮搬同一段），实测推翻了。

    合理的解释：`create_addr_reg` 的偏移**不是一个运行期可清的累加器**，它绑在循环轴上，
    每轮的值在 codegen 期就定了，`clear_spr` 没有可清的东西。

    那它清的是什么？[`vf_unalign`](../vf_unalign/kernel.py) 给了另一半答案：那边
    **漏掉 `clear_spr()` 就会静默出错**，所以它确实清了 squeeze / 非对齐搬运用的 AR 状态。

    **两处在文档里都叫「AddrReg」，实际是两个不同的东西。** 这是文档的命名问题，
    不是本例的实现问题——已报到 pypto 侧。
    """
    a = ref.make_fp32_data(2 * N_FP32)
    out = _fp32(device, cols=2 * N_FP32)
    addr_reg_cleared_each_round(_dev(a, device), out)
    torch.npu.synchronize()

    flat = out.cpu().numpy().reshape(-1)
    np.testing.assert_allclose(
        flat.reshape(1, -1),
        a,
        rtol=0,
        atol=0,
        err_msg=(
            "加了 clear_spr 之后两轮仍应搬完整的两段——"
            "如果后半段变成前半段的副本，说明 clear_spr "
            "开始影响循环偏移了，本例和设计文档都要改"
        ),
    )
    assert not np.array_equal(flat[:N_FP32], flat[N_FP32:]), (
        "输出两半相同——偏移被清了，与实测结论相反"
    )


# ---------------------------------------------------------------------------
# mem_bar
# ---------------------------------------------------------------------------


def test_mem_bar_does_not_change_the_result(device):
    """一个真实 RAW 形状（vf 写 Tile → vf 读同一个 Tile），带屏障，结果是 `3a`。

    断言的是**屏障不改变数值**。它保证的是顺序，不是算术。
    """
    a = ref.make_fp32_data(N_FP32)
    out = _fp32(device)
    mem_bar_raw(_dev(a, device), out)
    torch.npu.synchronize()

    np.testing.assert_allclose(
        out.cpu().numpy(),
        ref.triple(a),
        rtol=0,
        atol=0,
        err_msg="带 mem_bar 的读写往返没给出 3a",
    )


def test_all_twelve_mem_bar_modes_compile_and_run(device):
    """12 种合法组合连着写一遍，能编译、能跑、数据原样透过。

    这是覆盖，不是语义验证——见模块 docstring §三。
    """
    a = ref.make_fp32_data(N_FP32)
    out = _fp32(device)
    mem_bar_all_twelve(_dev(a, device), out)
    torch.npu.synchronize()

    np.testing.assert_allclose(
        out.cpu().numpy(), a, rtol=0, atol=0, err_msg="12 条屏障之间的数据被改动了"
    )


def test_there_are_exactly_twelve_mem_bar_modes():
    """不需要卡。`MemBarMode.__members__` 必须正好是文档那张表的 12 项。

    用 `__members__` 而不是 `dir()`——后者会把 `name` / `value` 描述符算进去，
    这个坑记在 [`vf_align`](../vf_align/test.py)。

    枚举加了成员这条会失败，那时要回头看文档表和本例的 `_mem_bar_all_twelve`。
    """
    pl = pytest.importorskip("pypto_pro.language")
    members = sorted(pl.MemBarMode.__members__)
    assert members == sorted(_TWELVE_MODES), f"MemBarMode 的成员变了：{members}"


# ---------------------------------------------------------------------------
# 覆盖检查
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("name", _ALL_APIS)
def test_every_api_in_this_example_is_actually_called(name):
    """不需要卡。防漏：表里列的 3 个接口每个都得在 kernel 源码里出现。"""
    source = (Path(__file__).parent / "kernel.py").read_text()
    assert f"vf.{name}(" in source, f"kernel.py 里没有调用 vf.{name}()"
