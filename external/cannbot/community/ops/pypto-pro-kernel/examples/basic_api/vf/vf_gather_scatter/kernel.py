# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

"""写法：原始。四个接口：`gather` `scatter` `arange` `create_addr_reg`。

这四个凑在一起不是因为名字像，是因为它们回答**同一个问题的四个层次**：
「这一批 lane 各自该从哪里取／往哪里放」。

| 接口 | 谁给地址 | 地址的粒度 |
| --- | --- | --- |
| `arange` | 指令自己生成 `start ± i` | 元素，步长固定 ±1 |
| `gather` | 一个**索引寄存器**，每 lane 一个 | 元素（NORM）或 32B 块（DATA_BLOCK_LOAD） |
| `scatter` | 同上，但方向相反（寄存器 → Tile） | 元素 |
| `create_addr_reg` | 绑在**循环轴**上，每轮累加一个 stride | 元素，整条 load/store 共用一个偏移 |

**`create_addr_reg` 和前三个不是一类。** 它不是「每 lane 一个地址」，是「整条搬运一个
偏移，随循环推进」——所以它只能给 `load_align` / `store_align` 当偏移参数用，
不参与 gather/scatter 的索引。表里放在一起是为了讲清这个区别。

第一手来源：官方 `data_movement/{gather,scatter,create_addr_reg}.md` 与
`sorting_and_indexing/arange.md`；先例 `lightning_indexer` ST（`arange` + `gather` 建
直方图索引）、`vf_api` ST kernel 81 / 92（`create_addr_reg`，都是单轮 smoke）。
"""

import pypto_pro.language as pl
from pypto_pro.language import Vf as vf  # noqa: N813

FP32 = pl.DT_FP32
U32 = pl.DT_UINT32
I32 = pl.DT_INT32

N = 64  # 一个 FP32 寄存器 = 256 字节 = 64 个元素
BLOCK_BYTES = 32  # 一个 DataBlock
BLOCKS = N * 4 // BLOCK_BYTES  # = 8

_A = [0x0000 + 0x400 * i for i in range(2)]  # 数据、索引
_O = [0x1000 + 0x400 * i for i in range(4)]


def _tile_type(dtype=FP32, rows=1, cols=N):
    return pl.TileType(
        shape=[rows, cols], dtype=dtype, target_memory=pl.MemorySpace.Vec
    )


# ---------------------------------------------------------------------------
# 一、gather 的两种形式：Tile→reg 与 reg→reg
# ---------------------------------------------------------------------------


@pl.vector_function
def _gather_both_forms(src, idx_tile, o_tile2reg, o_reg2reg):
    """同一个索引，两种形式，结果应当相同——但**签名不同**。

    | | src | `preg` | `data_copy_mode` |
    | --- | --- | --- | --- |
    | Tile→reg | Tile | **必须** | 支持 |
    | reg→reg | 寄存器 | **不能传** | 不支持 |

    后端按 `src` 的类型分发。传错的后果是签名层面的报错，不是静默出错——
    这一点比下面 §二 的 dtype 约束友好。

    **索引的 dtype 被元素宽度锁死**：FP32 源必须配 `DT_UINT32` 索引。
    官方约束表列了 17 行 `(dst, src, index)` 三元组，32 位宽的三行全是 `UINT32`。
    """
    preg = vf.create_mask(pattern=pl.MaskPattern.ALL, dtype=FP32)
    idx = vf.load_align(idx_tile, 0)

    # Tile→reg：直接从 UB 取，要 preg
    from_tile = vf.gather(src, idx, preg)
    vf.store_align(o_tile2reg, from_tile, preg, 0)

    # reg→reg：先把数据读进寄存器，再按索引重排，**不传 preg**
    data = vf.load_align(src, 0)
    from_reg = vf.gather(data, idx)
    vf.store_align(o_reg2reg, from_reg, preg, 0)


@pl.jit(auto_mutex=True)
def gather_both_forms(
    a: pl.Tensor[[1, N], FP32],
    idx: pl.Tensor[[1, N], U32],
    o_tile2reg: pl.Tensor[[1, N], FP32],
    o_reg2reg: pl.Tensor[[1, N], FP32],
):
    ga = pl.make_tile_group(type=_tile_type(), addrs=_A[0], mutex_ids=[0])
    gi = pl.make_tile_group(type=_tile_type(U32), addrs=_A[1], mutex_ids=[1])
    g0 = pl.make_tile_group(type=_tile_type(), addrs=_O[0], mutex_ids=[2])
    g1 = pl.make_tile_group(type=_tile_type(), addrs=_O[1], mutex_ids=[3])
    with pl.section_vector():
        ta, ti = ga.current(), gi.current()
        t0, t1 = g0.current(), g1.current()
        pl.load(ta, a, [0, 0])
        pl.load(ti, idx, [0, 0])
        _gather_both_forms(ta, ti, t0, t1)
        pl.store(o_tile2reg, t0, [0, 0])
        pl.store(o_reg2reg, t1, [0, 0])


# ---------------------------------------------------------------------------
# 二、DATA_BLOCK_LOAD：索引的单位从元素变成字节
# ---------------------------------------------------------------------------


@pl.vector_function
def _gather_datablock(src, idx_tile, dst):
    """`data_copy_mode=DATA_BLOCK_LOAD`：一个索引对应一个 **32 字节块**。

    两处和 NORM 模式不同，都会静默给出错结果而不是报错：

    1. **索引的单位是字节，不是元素**，而且必须 32B 对齐。64 个 FP32 = 256 字节 = 8 块，
       所以有意义的索引只有 8 个，取值必须是 `{0, 32, 64, ..., 224}`。
    2. **只有前 8 个 lane 的索引被读**（VL / 32）。其余 lane 写什么都不影响结果——
       本例把它们置 0，因为 0 号块一定在范围内。

    判据设计：把 8 个块整体倒序。倒序之后每个元素都换了位置，
    「索引没被用上」和「按元素而不是按块」两种错法都会被抓住。
    """
    preg = vf.create_mask(pattern=pl.MaskPattern.ALL, dtype=FP32)
    idx = vf.load_align(idx_tile, 0)
    blocks = vf.gather(src, idx, preg, data_copy_mode=pl.DataCopyMode.DATA_BLOCK_LOAD)
    vf.store_align(dst, blocks, preg, 0)


@pl.jit(auto_mutex=True)
def gather_datablock(
    a: pl.Tensor[[1, N], FP32],
    idx: pl.Tensor[[1, N], U32],
    out: pl.Tensor[[1, N], FP32],
):
    ga = pl.make_tile_group(type=_tile_type(), addrs=_A[0], mutex_ids=[0])
    gi = pl.make_tile_group(type=_tile_type(U32), addrs=_A[1], mutex_ids=[1])
    g0 = pl.make_tile_group(type=_tile_type(), addrs=_O[0], mutex_ids=[2])
    with pl.section_vector():
        ta, ti, t0 = ga.current(), gi.current(), g0.current()
        pl.load(ta, a, [0, 0])
        pl.load(ti, idx, [0, 0])
        _gather_datablock(ta, ti, t0)
        pl.store(out, t0, [0, 0])


# ---------------------------------------------------------------------------
# 三、scatter，以及 gather / scatter 对重复索引的不对称
# ---------------------------------------------------------------------------


@pl.vector_function
def _scatter(src, idx_tile, dst):
    """`scatter(tile, src, index, preg)`：寄存器 → Tile，按索引分散。

    方向和 `gather` 相反，所以**没有返回值**——目的是 Tile，不是寄存器。

    两条约束和 `gather` 不对称：

    - **索引值必须唯一。** 重复了「系统仅保留其中一个」——不报错，保留哪个也没规定。
      `gather` 那边明确允许重复（多个 lane 读同一个源是正常需求）。
    - 目的 Tile 基地址要 32 字节对齐（`gather` 的 Tile→reg 源也一样）。

    这个不对称有道理：读可以广播，写会撞车。但**它只写在文档里，运行期不检查**。
    """
    preg = vf.create_mask(pattern=pl.MaskPattern.ALL, dtype=FP32)
    idx = vf.load_align(idx_tile, 0)
    data = vf.load_align(src, 0)
    vf.scatter(dst, data, idx, preg)


@pl.jit(auto_mutex=True)
def scatter(
    a: pl.Tensor[[1, N], FP32],
    idx: pl.Tensor[[1, N], U32],
    out: pl.Tensor[[1, N], FP32],
):
    ga = pl.make_tile_group(type=_tile_type(), addrs=_A[0], mutex_ids=[0])
    gi = pl.make_tile_group(type=_tile_type(U32), addrs=_A[1], mutex_ids=[1])
    g0 = pl.make_tile_group(type=_tile_type(), addrs=_O[0], mutex_ids=[2])
    with pl.section_vector():
        ta, ti, t0 = ga.current(), gi.current(), g0.current()
        pl.load(ta, a, [0, 0])
        pl.load(ti, idx, [0, 0])
        _scatter(ta, ti, t0)
        pl.store(out, t0, [0, 0])


@pl.vector_function
def _gather_duplicate_index(src, idx_tile, dst):
    """`gather` 吃重复索引：多个 lane 读同一个源元素，是**广播**，不是错误。

    和 `_scatter` 用同一组重复索引对照着看：同一份索引，gather 有定义、scatter 没有。
    """
    preg = vf.create_mask(pattern=pl.MaskPattern.ALL, dtype=FP32)
    idx = vf.load_align(idx_tile, 0)
    data = vf.load_align(src, 0)
    r = vf.gather(data, idx)
    vf.store_align(dst, r, preg, 0)


@pl.jit(auto_mutex=True)
def gather_duplicate_index(
    a: pl.Tensor[[1, N], FP32],
    idx: pl.Tensor[[1, N], U32],
    out: pl.Tensor[[1, N], FP32],
):
    ga = pl.make_tile_group(type=_tile_type(), addrs=_A[0], mutex_ids=[0])
    gi = pl.make_tile_group(type=_tile_type(U32), addrs=_A[1], mutex_ids=[1])
    g0 = pl.make_tile_group(type=_tile_type(), addrs=_O[0], mutex_ids=[2])
    with pl.section_vector():
        ta, ti, t0 = ga.current(), gi.current(), g0.current()
        pl.load(ta, a, [0, 0])
        pl.load(ti, idx, [0, 0])
        _gather_duplicate_index(ta, ti, t0)
        pl.store(out, t0, [0, 0])


@pl.vector_function
def _gather_reg_out_of_range(src, idx_tile, dst):
    """reg→reg 形式下，越界索引**取模回卷**：`i % (VL / sizeof(T))` = `i % 64`。

    这是官方 `gather.md` 里写明的行为，只对 **reg→reg** 有效。Tile→reg 形式那边写的是
    「索引值对应的数据必须在 Tile 有效地址范围内」——**没有回卷承诺**，所以别靠它。

    一句话：同一个接口的两种形式，越界行为一个有定义一个没有。
    """
    preg = vf.create_mask(pattern=pl.MaskPattern.ALL, dtype=FP32)
    idx = vf.load_align(idx_tile, 0)
    data = vf.load_align(src, 0)
    r = vf.gather(data, idx)
    vf.store_align(dst, r, preg, 0)


@pl.jit(auto_mutex=True)
def gather_reg_out_of_range(
    a: pl.Tensor[[1, N], FP32],
    idx: pl.Tensor[[1, N], U32],
    out: pl.Tensor[[1, N], FP32],
):
    ga = pl.make_tile_group(type=_tile_type(), addrs=_A[0], mutex_ids=[0])
    gi = pl.make_tile_group(type=_tile_type(U32), addrs=_A[1], mutex_ids=[1])
    g0 = pl.make_tile_group(type=_tile_type(), addrs=_O[0], mutex_ids=[2])
    with pl.section_vector():
        ta, ti, t0 = ga.current(), gi.current(), g0.current()
        pl.load(ta, a, [0, 0])
        pl.load(ti, idx, [0, 0])
        _gather_reg_out_of_range(ta, ti, t0)
        pl.store(out, t0, [0, 0])


# ---------------------------------------------------------------------------
# 四、arange：步长固定 ±1，要别的步长自己乘
# ---------------------------------------------------------------------------


@pl.vector_function
def _arange_three_ways(o_inc, o_dec, o_step4):
    """`arange(start, dtype)` 只生成 `start ± i`，**步长是硬件特性，改不了**。

    - `INCREASE_ORDER`（默认）：`dst[i] = start + i`
    - `DECREASE_ORDER`：**`start` 是最后一个 lane 的值，不是第 0 个。**
      `dst[i] = start + (VL - 1 - i)`。实测 `start=100`、VL=64 → `163, 162, ..., 100`。
      官方正文写的「以传入的 start 为**最终值**」是对的；`_vf_api.py` 的 docstring
      写成 `dst[i] = start - i` 是错的（已在 pypto 侧改掉）。
      两处对不上时就是这样：**公式看着更精确，反而是错的那一份。**
    - 非 1 步长：`arange` 之后接一条 `vf.muls`。**没有 step 参数**，
      找不到它不是漏读，是它确实不存在。

    `dtype` 是必填的：源是标量，推不出寄存器类型。
    """
    preg = vf.create_mask(pattern=pl.MaskPattern.ALL, dtype=I32)

    inc = vf.arange(0, dtype=I32)
    vf.store_align(o_inc, inc, preg, 0)

    dec = vf.arange(100, dtype=I32, index_order=pl.IndexOrder.DECREASE_ORDER)
    vf.store_align(o_dec, dec, preg, 0)

    step4 = vf.muls(inc, pl.const(4, I32), preg)
    vf.store_align(o_step4, step4, preg, 0)


@pl.jit(auto_mutex=True)
def arange_three_ways(
    o_inc: pl.Tensor[[1, N], I32],
    o_dec: pl.Tensor[[1, N], I32],
    o_step4: pl.Tensor[[1, N], I32],
):
    g0 = pl.make_tile_group(type=_tile_type(I32), addrs=_O[0], mutex_ids=[0])
    g1 = pl.make_tile_group(type=_tile_type(I32), addrs=_O[1], mutex_ids=[1])
    g2 = pl.make_tile_group(type=_tile_type(I32), addrs=_O[2], mutex_ids=[2])
    with pl.section_vector():
        t0, t1, t2 = g0.current(), g1.current(), g2.current()
        _arange_three_ways(t0, t1, t2)
        pl.store(o_inc, t0, [0, 0])
        pl.store(o_dec, t1, [0, 0])
        pl.store(o_step4, t2, [0, 0])


# ---------------------------------------------------------------------------
# 五、create_addr_reg：偏移绑在循环轴上
# ---------------------------------------------------------------------------


@pl.vector_function
def _addr_reg_walk(src, dst):
    """`create_addr_reg(stride0)` + `pl.range`：偏移每轮自动加 `stride0` 个元素。

    两个 ST 先例（`vf_api` kernel 81 / 92）都是 `pl.range(0, 1, 1)`——**单轮，
    验的是「能编译」，不是「会累加」**。本例跑两轮，才看得出偏移动了。

    输入 Tile 有 128 个元素（两个寄存器长），`stride0=64`：
    第 0 轮搬 `[0:64]`，第 1 轮搬 `[64:128]`，输出与输入逐位相同。
    **如果偏移没累加，输出后半段会是前半段的副本**——判据抓的就是这个。

    AddrReg 上限 8 个，最多 4 层循环轴；本例的对齐搬运仍须满足对应地址对齐约束。
    非对齐搬运的官方页面与 create_addr_reg 的范围说明存在冲突，事实与验证边界
    统一见 `references/api/vf.md`「AddrReg：循环寻址与验证边界」。
    """
    preg = vf.create_mask(pattern=pl.MaskPattern.ALL, dtype=FP32)
    for _ in pl.range(0, 2, 1):
        areg = vf.create_addr_reg(N, dtype=FP32)
        reg = vf.load_align(src, areg)
        vf.store_align(dst, reg, preg, areg)


@pl.jit(auto_mutex=True)
def addr_reg_walk(a: pl.Tensor[[1, 2 * N], FP32], out: pl.Tensor[[1, 2 * N], FP32]):
    # `_tile_type(FP32, 1, 2 * N)` —— **kernel 体里调内联辅助函数只能用位置参数**，
    # 写 `_tile_type(cols=2 * N)` 会被解析期拒（`_call_parser.py:1438`）。
    wide = _tile_type(FP32, 1, 2 * N)
    ga = pl.make_tile_group(type=wide, addrs=_A[0], mutex_ids=[0])
    g0 = pl.make_tile_group(type=wide, addrs=_O[0], mutex_ids=[1])
    with pl.section_vector():
        ta, t0 = ga.current(), g0.current()
        pl.load(ta, a, [0, 0])
        _addr_reg_walk(ta, t0)
        pl.store(out, t0, [0, 0])


@pl.vector_function
def _addr_reg_into_store_unalign(src, dst):
    """**只解析，不执行。** 把 AddrReg 当 `store_unalign` 的 `stride` 传进去。

    接口、官方 store_unalign 示例和后端都包含该组合；本探针只保证解析接受，
    不提供实际写入范围或数值的验收。文档冲突与当前证据见 references/api/vf.md。
    """
    reg = vf.load_align(src, 0)
    areg = vf.unalign_reg_for_store()
    addr = vf.create_addr_reg(N, dtype=FP32)
    vf.store_unalign(dst, reg, areg, addr)
    vf.store_unalign_post(dst, areg, addr)


@pl.jit(auto_mutex=True)
def addr_reg_into_store_unalign(
    a: pl.Tensor[[1, N], FP32], out: pl.Tensor[[1, N], FP32]
):
    """**只用来解析，不要调用它。** 见 `_addr_reg_into_store_unalign` 的说明。"""
    ga = pl.make_tile_group(type=_tile_type(), addrs=_A[0], mutex_ids=[0])
    g0 = pl.make_tile_group(type=_tile_type(), addrs=_O[0], mutex_ids=[1])
    with pl.section_vector():
        ta, t0 = ga.current(), g0.current()
        pl.load(ta, a, [0, 0])
        _addr_reg_into_store_unalign(ta, t0)
        pl.store(out, t0, [0, 0])
