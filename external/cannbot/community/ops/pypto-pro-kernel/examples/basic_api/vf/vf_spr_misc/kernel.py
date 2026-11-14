# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

"""写法：原始。三个接口：`histograms` `clear_spr` `mem_bar`。

三个接口的共同点是**都不是纯函数**：

| 接口 | 状态在哪 | 观察方式 |
| --- | --- | --- |
| `histograms` | **目的寄存器**（既读又写，累加） | 连着调两次，看计数翻倍 |
| `clear_spr` | **AddrReg / AR 特殊寄存器** | 和不清的版本对照，看偏移是否回到起点 |
| `mem_bar` | 流水线的顺序约束 | 只能验「进了 IR、能跑」——语义不是数值 |

`histograms` 和 `clear_spr` 有可观察的数值后果，所以有真判据。`mem_bar` **没有**：
屏障的作用是排除一类竞态，而正确的代码不加屏障也可能恰好对。硬编一个「不加就错」的
反例既不稳定又没意义，所以这一例对 `mem_bar` 只做覆盖，判据写清楚这一点。

第一手来源：官方 `sorting_and_indexing/histograms.md`、`data_movement/clear_spr.md`、
`data_movement/mem_bar.md`；先例 `lightning_indexer` ST 的 `histograms_high_vf`
（UINT8 分桶建索引，这是全仓唯一真用 `histograms` 的地方）。
"""

import pypto_pro.language as pl
from pypto_pro.language import Vf as vf  # noqa: N813

FP32 = pl.DT_FP32
U8 = pl.DT_UINT8
U16 = pl.DT_UINT16
I16 = pl.DT_INT16

N_U8 = 256  # 一个寄存器 = 256 字节 = 256 个 UINT8
N_BINS = 128  # dst 是 UINT16，256 / 2 = 128 个桶
N_FP32 = 64

_A = [0x0000 + 0x400 * i for i in range(2)]
_O = [0x1000 + 0x400 * i for i in range(6)]


def _tile(dtype, cols):
    return pl.TileType(shape=[1, cols], dtype=dtype, target_memory=pl.MemorySpace.Vec)


# ---------------------------------------------------------------------------
# 一、histograms：两个开关，四种组合
# ---------------------------------------------------------------------------


def make_single_histogram(bin_type, hist_type):
    """一个 kernel 只做一次 `histograms`——**用来拿无混淆的基准值**。

    为什么要有这个工厂：`histograms` 的目的寄存器既读又写，同一个
    `@pl.vector_function` 里连着做四次时，四个 dst 会不会被分配到同一个物理寄存器、
    互相污染，**是个需要先排除的问题**（`precision=True` 那个 codegen 缺陷就是同名
    临时量撞车，见 `references/diagnostics/defects/`）。

    一次只做一个，就没有第二个 dst 可以污染。四个单模式的结果是基准，
    `histograms_four_ways`（下面）的结果和它比——相等才说明合并写法是安全的。

    枚举值从闭包里传进 kernel 体是可以的（实测 IR 里 `bin_type="bin_type": 1`），
    这一点和 `pl.const` 的第一个参数必须是字面量不同。
    """

    @pl.vector_function
    def _one(src, dst):
        preg_b8 = vf.create_mask(pattern=pl.MaskPattern.ALL, dtype=U8)
        preg_b16 = vf.create_mask(pattern=pl.MaskPattern.ALL, dtype=U16)
        data = vf.load_align(src, 0)
        hist = vf.full(0, preg_b16, dtype=U16)  # ACCUMULATE 原地累加，必须先清零
        hist = vf.histograms(data, preg_b8, bin_type=bin_type, hist_type=hist_type)
        r = vf.bit_cast(hist, dtype=I16)  # UINT16 在 host 侧没有 torch 对应
        vf.store_align(dst, r, preg_b16, 0)

    @pl.jit(auto_mutex=True)
    def kernel(a: pl.Tensor[[1, N_U8], U8], out: pl.Tensor[[1, N_BINS], I16]):
        ga = pl.make_tile_group(type=_tile(U8, N_U8), addrs=_A[0], mutex_ids=[0])
        g0 = pl.make_tile_group(type=_tile(I16, N_BINS), addrs=_O[0], mutex_ids=[1])
        with pl.section_vector():
            ta, t0 = ga.current(), g0.current()
            pl.load(ta, a, [0, 0])
            _one(ta, t0)
            pl.store(out, t0, [0, 0])

    return kernel


@pl.vector_function
def _histograms_four_ways(src, o_f0, o_f1, o_a0, o_a1):
    """`bin_type` × `hist_type` = 四种组合，一次全存出来。

    | | `BIN0`（看桶 0..127） | `BIN1`（看桶 128..255） |
    | --- | --- | --- |
    | `FREQUENCY` | `dst[n]` = 值 `n` 出现几次 | `dst[n]` = 值 `n+128` 出现几次 |
    | `ACCUMULATE` | `dst[n]` = 值 `≤ n` 共出现几次 | `dst[n]` = 值 **`≤ n+128`** 共出现几次 |

    **`ACCUMULATE` 的累计是对整个 0..255 值域做的**，`bin_type` 只决定看哪 128 个桶，
    不缩小累计范围——所以 `BIN1` 的累计**从 `BIN0` 结束的地方接着往上走**，
    两半拼成一条连续曲线。官方文档那句「从 0 到 n 的所有数值**在对应区间中**出现的
    总频率」读起来像限制在区间内，实测不是（详见 reference.py 的 `accumulate`）。

    四条约束，前两条会静默出错：

    1. **src 必须是 `DT_UINT8`，`preg` 也必须是 b8 粒度**（`create_mask(dtype=U8)`）。
    2. **dst 必须先初始化。** `ACCUMULATE` 是原地累加——目的寄存器既读又写。
       不先 `vf.full(0, ...)` 就是在垃圾值上累加。这也是 dst 为什么要先绑名字。
    3. dst 是 `DT_UINT16`，128 个桶；最大 65535，累加多了会溢出。
    4. `BIN1` 模式下值减 128 映射到桶号：值 `200` 落在 `dst[72]`。

    **UINT16 在 host 侧没有 torch 对应**（`_PL_DTYPE_TO_TORCH` 只有 11 项），
    所以存之前 `bit_cast` 成 `DT_INT16`。计数最多 256，不会碰到符号位。

    **这个合并写法的正确性不是假设的**：`make_single_histogram` 造的四个单模式 kernel
    提供无混淆基准，`test_the_combined_kernel_agrees_with_the_single_ones` 逐个对。
    四个既读又写的 dst 挤在一个 `@pl.vector_function` 里，先证明它们不互相污染。
    """
    preg_b8 = vf.create_mask(pattern=pl.MaskPattern.ALL, dtype=U8)
    preg_b16 = vf.create_mask(pattern=pl.MaskPattern.ALL, dtype=U16)
    data = vf.load_align(src, 0)

    # 四个 dst 各自先清零——ACCUMULATE 原地累加，不初始化就是在垃圾上加
    freq0 = vf.full(0, preg_b16, dtype=U16)
    freq0 = vf.histograms(
        data, preg_b8, bin_type=pl.BinType.BIN0, hist_type=pl.HistType.FREQUENCY
    )
    r = vf.bit_cast(freq0, dtype=I16)
    vf.store_align(o_f0, r, preg_b16, 0)

    freq1 = vf.full(0, preg_b16, dtype=U16)
    freq1 = vf.histograms(
        data, preg_b8, bin_type=pl.BinType.BIN1, hist_type=pl.HistType.FREQUENCY
    )
    r = vf.bit_cast(freq1, dtype=I16)
    vf.store_align(o_f1, r, preg_b16, 0)

    acc0 = vf.full(0, preg_b16, dtype=U16)
    acc0 = vf.histograms(
        data, preg_b8, bin_type=pl.BinType.BIN0, hist_type=pl.HistType.ACCUMULATE
    )
    r = vf.bit_cast(acc0, dtype=I16)
    vf.store_align(o_a0, r, preg_b16, 0)

    acc1 = vf.full(0, preg_b16, dtype=U16)
    acc1 = vf.histograms(
        data, preg_b8, bin_type=pl.BinType.BIN1, hist_type=pl.HistType.ACCUMULATE
    )
    r = vf.bit_cast(acc1, dtype=I16)
    vf.store_align(o_a1, r, preg_b16, 0)


@pl.jit(auto_mutex=True)
def histograms_four_ways(
    a: pl.Tensor[[1, N_U8], U8],
    o_f0: pl.Tensor[[1, N_BINS], I16],
    o_f1: pl.Tensor[[1, N_BINS], I16],
    o_a0: pl.Tensor[[1, N_BINS], I16],
    o_a1: pl.Tensor[[1, N_BINS], I16],
):
    ga = pl.make_tile_group(type=_tile(U8, N_U8), addrs=_A[0], mutex_ids=[0])
    g0 = pl.make_tile_group(type=_tile(I16, N_BINS), addrs=_O[0], mutex_ids=[1])
    g1 = pl.make_tile_group(type=_tile(I16, N_BINS), addrs=_O[1], mutex_ids=[2])
    g2 = pl.make_tile_group(type=_tile(I16, N_BINS), addrs=_O[2], mutex_ids=[3])
    g3 = pl.make_tile_group(type=_tile(I16, N_BINS), addrs=_O[3], mutex_ids=[4])
    with pl.section_vector():
        ta = ga.current()
        t0, t1, t2, t3 = g0.current(), g1.current(), g2.current(), g3.current()
        pl.load(ta, a, [0, 0])
        _histograms_four_ways(ta, t0, t1, t2, t3)
        pl.store(o_f0, t0, [0, 0])
        pl.store(o_f1, t1, [0, 0])
        pl.store(o_a0, t2, [0, 0])
        pl.store(o_a1, t3, [0, 0])


@pl.vector_function
def _histograms_accumulate_twice(src, o_once, o_twice):
    """同一份数据统计两次，**计数翻倍**——这就是「目的寄存器既读又写」的可观察后果。

    `o_once` 和 `o_twice` 用两个独立的 dst，同一份输入：一个统计一次，一个统计两次。
    两者相差正好一倍，才说明第二次是在第一次的结果上累加，而不是覆盖。

    这是 `histograms` 和其他 vf 算子最大的区别：**它不是纯函数。**
    `lightning_indexer` 的 ST 正是靠这个在循环里把多轮数据累进同一个直方图。
    """
    preg_b8 = vf.create_mask(pattern=pl.MaskPattern.ALL, dtype=U8)
    preg_b16 = vf.create_mask(pattern=pl.MaskPattern.ALL, dtype=U16)
    data = vf.load_align(src, 0)

    once = vf.full(0, preg_b16, dtype=U16)
    once = vf.histograms(data, preg_b8, hist_type=pl.HistType.FREQUENCY)
    r = vf.bit_cast(once, dtype=I16)
    vf.store_align(o_once, r, preg_b16, 0)

    twice = vf.full(0, preg_b16, dtype=U16)
    twice = vf.histograms(data, preg_b8, hist_type=pl.HistType.FREQUENCY)
    twice = vf.histograms(data, preg_b8, hist_type=pl.HistType.FREQUENCY)
    r = vf.bit_cast(twice, dtype=I16)
    vf.store_align(o_twice, r, preg_b16, 0)


@pl.jit(auto_mutex=True)
def histograms_accumulate_twice(
    a: pl.Tensor[[1, N_U8], U8],
    o_once: pl.Tensor[[1, N_BINS], I16],
    o_twice: pl.Tensor[[1, N_BINS], I16],
):
    ga = pl.make_tile_group(type=_tile(U8, N_U8), addrs=_A[0], mutex_ids=[0])
    g0 = pl.make_tile_group(type=_tile(I16, N_BINS), addrs=_O[0], mutex_ids=[1])
    g1 = pl.make_tile_group(type=_tile(I16, N_BINS), addrs=_O[1], mutex_ids=[2])
    with pl.section_vector():
        ta, t0, t1 = ga.current(), g0.current(), g1.current()
        pl.load(ta, a, [0, 0])
        _histograms_accumulate_twice(ta, t0, t1)
        pl.store(o_once, t0, [0, 0])
        pl.store(o_twice, t1, [0, 0])


@pl.vector_function
def _histograms_masked(src, o_bin0, o_bin1):
    """`preg` 为 0 的 lane 不参与统计。用「只统计 < 128 的值」把它验到。

    掩码由一次比较得到（`vf.lt`），不是 `MaskPattern`——比较出来的掩码语义明确，
    不依赖对 `MaskPattern.H` 之类「哪一半」的猜测。

    结果的形状很硬：屏蔽掉所有 ≥ 128 的值之后，**`BIN1` 的直方图必须全 0**。
    这比「某个桶少了几个」好验——少一点可能是别的原因，全 0 只能是掩码起了作用。
    """
    preg_b8 = vf.create_mask(pattern=pl.MaskPattern.ALL, dtype=U8)
    preg_b16 = vf.create_mask(pattern=pl.MaskPattern.ALL, dtype=U16)
    data = vf.load_align(src, 0)

    low_only = vf.lt(data, pl.const(128, U8), preg_b8)

    bin0 = vf.full(0, preg_b16, dtype=U16)
    bin0 = vf.histograms(
        data, low_only, bin_type=pl.BinType.BIN0, hist_type=pl.HistType.FREQUENCY
    )
    r = vf.bit_cast(bin0, dtype=I16)
    vf.store_align(o_bin0, r, preg_b16, 0)

    bin1 = vf.full(0, preg_b16, dtype=U16)
    bin1 = vf.histograms(
        data, low_only, bin_type=pl.BinType.BIN1, hist_type=pl.HistType.FREQUENCY
    )
    r = vf.bit_cast(bin1, dtype=I16)
    vf.store_align(o_bin1, r, preg_b16, 0)


@pl.jit(auto_mutex=True)
def histograms_masked(
    a: pl.Tensor[[1, N_U8], U8],
    o_bin0: pl.Tensor[[1, N_BINS], I16],
    o_bin1: pl.Tensor[[1, N_BINS], I16],
):
    ga = pl.make_tile_group(type=_tile(U8, N_U8), addrs=_A[0], mutex_ids=[0])
    g0 = pl.make_tile_group(type=_tile(I16, N_BINS), addrs=_O[0], mutex_ids=[1])
    g1 = pl.make_tile_group(type=_tile(I16, N_BINS), addrs=_O[1], mutex_ids=[2])
    with pl.section_vector():
        ta, t0, t1 = ga.current(), g0.current(), g1.current()
        pl.load(ta, a, [0, 0])
        _histograms_masked(ta, t0, t1)
        pl.store(o_bin0, t0, [0, 0])
        pl.store(o_bin1, t1, [0, 0])


# ---------------------------------------------------------------------------
# 二、clear_spr：清的到底是什么
# ---------------------------------------------------------------------------


@pl.vector_function
def _addr_reg_cleared_each_round(src, dst):
    """循环里每轮清一次 AR，和 [`vf_gather_scatter`](../vf_gather_scatter/kernel.py) 的
    `_addr_reg_walk` 逐行对照——**只差一个 `vf.clear_spr()`**。

    那一例证明了偏移会跨轮累加（两轮搬完 128 个元素，输出等于输入）。
    这一例在每轮开头清一次，看偏移是否回到起点。

    官方 `clear_spr.md` 的说法是「清除特殊寄存器（**AddrReg 寄存器**），重置某些 VF 指令
    使用的累加寄存器」，并建议「在 `@vector_function` 的开头或结尾调用」。
    **放在循环体里是本例故意做的实验**——用来把「它清的是什么」量出来，不是推荐写法。

    **实测结论：它不清这个偏移。** 加了 `clear_spr()` 之后两轮照样搬 `[0:64]` 和
    `[64:128]`，与 `vf_gather_scatter` 那一例逐位相同。

    合理的解释是 `create_addr_reg` 的偏移**不是一个运行期可清的累加器**——它绑在循环轴
    上，每轮的偏移在 codegen 期就定了，`clear_spr` 没有可清的东西。而
    [`vf_unalign`](../vf_unalign/kernel.py) 那边**漏掉 `clear_spr()` 就会出错**，说明它
    确实清了 squeeze / 非对齐搬运用的那个 AR 状态。

    **两处都叫「AddrReg」，是两个不同的东西。** 文档这个命名会误导——已报到 pypto 侧。
    """
    preg = vf.create_mask(pattern=pl.MaskPattern.ALL, dtype=FP32)
    for _ in pl.range(0, 2, 1):
        vf.clear_spr()
        areg = vf.create_addr_reg(N_FP32, dtype=FP32)
        reg = vf.load_align(src, areg)
        vf.store_align(dst, reg, preg, areg)


@pl.jit(auto_mutex=True)
def addr_reg_cleared_each_round(
    a: pl.Tensor[[1, 2 * N_FP32], FP32], out: pl.Tensor[[1, 2 * N_FP32], FP32]
):
    wide = _tile(FP32, 2 * N_FP32)
    ga = pl.make_tile_group(type=wide, addrs=_A[0], mutex_ids=[0])
    g0 = pl.make_tile_group(type=wide, addrs=_O[0], mutex_ids=[1])
    with pl.section_vector():
        ta, t0 = ga.current(), g0.current()
        pl.load(ta, a, [0, 0])
        _addr_reg_cleared_each_round(ta, t0)
        pl.store(out, t0, [0, 0])


# ---------------------------------------------------------------------------
# 三、mem_bar：12 种组合，没有数值判据
# ---------------------------------------------------------------------------


@pl.vector_function
def _mem_bar_raw(src, dst):
    """一个真实的 RAW 场景：vf 写 Tile → vf 从同一个 Tile 读。

    `VST_VLD`（默认）就是为这个形状准备的：矢量 store → 矢量 load。

    **这一条不是「不加屏障就会错」的反例。** 官方 `mem_bar.md` 写明：读写用**同一个
    寄存器**时会触发寄存器保序，硬件自己按代码顺序执行，不需要插同步指令；只有
    寄存器不同才需要。所以「去掉屏障」在这个形状下大概率照样对——
    硬编一个不稳定的反例没有意义。

    这一条验的是：**加了屏障，结果仍然正确**，屏障不改变数值。
    """
    preg = vf.create_mask(pattern=pl.MaskPattern.ALL, dtype=FP32)
    a = vf.load_align(src, 0)
    doubled = vf.add(a, a, preg)
    vf.store_align(dst, doubled, preg, 0)

    vf.mem_bar(mode=pl.MemBarMode.VST_VLD)  # 写 → 读

    back = vf.load_align(dst, 0)
    plus = vf.add(back, a, preg)  # = 3a
    vf.mem_bar(mode=pl.MemBarMode.VST_VST)  # 写 → 写（WAW）
    vf.store_align(dst, plus, preg, 0)


@pl.jit(auto_mutex=True)
def mem_bar_raw(a: pl.Tensor[[1, N_FP32], FP32], out: pl.Tensor[[1, N_FP32], FP32]):
    ga = pl.make_tile_group(type=_tile(FP32, N_FP32), addrs=_A[0], mutex_ids=[0])
    g0 = pl.make_tile_group(type=_tile(FP32, N_FP32), addrs=_O[0], mutex_ids=[1])
    with pl.section_vector():
        ta, t0 = ga.current(), g0.current()
        pl.load(ta, a, [0, 0])
        _mem_bar_raw(ta, t0)
        pl.store(out, t0, [0, 0])


@pl.vector_function
def _mem_bar_all_twelve(src, dst):
    """12 种合法组合各写一次，验它们都能过解析和 codegen。

    枚举一共 **12** 个成员（`MemBarMode.__members__` 实测），与官方文档那张表一一对应：
    3 个纯矢量（`VST_VLD` / `VLD_VST` / `VST_VST`）、6 个矢量↔标量、3 个 `*_ALL`。

    **不能用循环或列表推导展开**——kernel 体是 DSL，枚举值要逐条写。
    这也是这段看起来啰嗦的原因。
    """
    preg = vf.create_mask(pattern=pl.MaskPattern.ALL, dtype=FP32)
    a = vf.load_align(src, 0)

    vf.mem_bar(mode=pl.MemBarMode.VST_VLD)
    vf.mem_bar(mode=pl.MemBarMode.VLD_VST)
    vf.mem_bar(mode=pl.MemBarMode.VST_VST)
    vf.mem_bar(mode=pl.MemBarMode.VST_LD)
    vf.mem_bar(mode=pl.MemBarMode.VST_ST)
    vf.mem_bar(mode=pl.MemBarMode.VLD_ST)
    vf.mem_bar(mode=pl.MemBarMode.ST_VLD)
    vf.mem_bar(mode=pl.MemBarMode.ST_VST)
    vf.mem_bar(mode=pl.MemBarMode.LD_VST)
    vf.mem_bar(mode=pl.MemBarMode.VV_ALL)
    vf.mem_bar(mode=pl.MemBarMode.VS_ALL)
    vf.mem_bar(mode=pl.MemBarMode.SV_ALL)

    vf.store_align(dst, a, preg, 0)


@pl.jit(auto_mutex=True)
def mem_bar_all_twelve(
    a: pl.Tensor[[1, N_FP32], FP32], out: pl.Tensor[[1, N_FP32], FP32]
):
    ga = pl.make_tile_group(type=_tile(FP32, N_FP32), addrs=_A[0], mutex_ids=[0])
    g0 = pl.make_tile_group(type=_tile(FP32, N_FP32), addrs=_O[0], mutex_ids=[1])
    with pl.section_vector():
        ta, t0 = ga.current(), g0.current()
        pl.load(ta, a, [0, 0])
        _mem_bar_all_twelve(ta, t0)
        pl.store(out, t0, [0, 0])
