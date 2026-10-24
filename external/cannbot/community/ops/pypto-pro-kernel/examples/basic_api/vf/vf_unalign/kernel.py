# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

"""写法：原始。八个非对齐搬运接口，分三条链。

机制的完整解释在 [api/vf.md](../../../../references/api/vf.md) §一——那一节回答「它解决什么
问题、两条 store 链的区别、真实场景、四条约束」。**这里只演示，不重复讲。**

## 三条链，八个接口

| 链 | 接口 | 谁决定「本次推进多少字节」 |
| --- | --- | --- |
| load | `load_unalign_init` → `load_unalign_pre` → `load_unalign(stride)` | 调用方传的 `stride` |
| store，调用方驱动 | `unalign_reg_for_store` → `store_unalign(stride)` → `store_unalign_post(stride)` | 调用方传的 `stride` |
| store，数据驱动 | unalign_reg_for_store→squeeze_store_unalign→squeeze_store_unalign_post | AR 特殊寄存器，由squeeze(STORE_REG)写入 |

`unalign_reg_for_store()` 两条 store 链共用，所以 8 = 3 + 3 + 2。

**`squeeze_store_unalign` 没有 `stride` 参数不是简化**——它的推进量根本不由调用方决定。
这是两条链的分界线：前者用在「每次写多长编译期或循环变量能算出来」，后者用在
「只有运行时才知道」（谓词的 popcount）。

## 尾部缓存与写入位置共同延续

`align_reg` 保留尚未凑满块的字节；squeeze 链的累计目的位移还依赖共享 AR。换一个 align_reg 不会建立独立 AR 位置，详见上述机制文档的多输出流一节。同一条链内：

- 循环里连着写，游标自己推进，不用手算偏移；
- 两个**不同的**循环共用一个 `align_reg`、中间不收尾，第二轮就从第一轮停下的字节位置接着写;
- 最后**必须**调一次 `*_post` 把零头冲出去，否则尾部那不满一块的数据留在寄存器里没落盘。

第一手来源：官方 `data_movement/` 下各接口页；全仓唯一用 squeeze 那条链的 ST 是
`lightning_indexer/test_quant_lightning_indexer_vf.py`（Top-K 索引收集）。
"""

import pypto_pro.language as pl
from pypto_pro.language import Vf as vf  # noqa: N813

FP32 = pl.DT_FP32
I32 = pl.DT_INT32

VL_FP32 = 64  # 一个 FP32 寄存器 = 256 字节 = 64 个元素
N = 64
CHUNK = 16  # 每次搬 16 个元素 = 64 字节，**不是** 256 字节的整数倍
CHUNK_BYTES = CHUNK * 4

_A = 0x0000
_O = [0x1000 + 0x400 * i for i in range(3)]


def _tile_type(dtype=FP32):
    return pl.TileType(shape=[1, N], dtype=dtype, target_memory=pl.MemorySpace.Vec)


# ---------------------------------------------------------------------------
# 一、load 链：从非块对齐的位置读
# ---------------------------------------------------------------------------


@pl.vector_function
def _load_chain(src, dst):
    """`init` → `pre` → `load`。三步缺一不可。

    - `load_unalign_init()` 声明跟踪寄存器；
    - `load_unalign_pre(align_reg, tile)` 预热（`vldas`）——**没有这一步，第一次 load 读到的
      前几个元素是垃圾**，因为跟踪寄存器还不知道起点；
    - `load_unalign(align_reg, tile, stride)` 真正读（`vldus`），游标按 `stride` 推进。

    本例读两次，每次 `CHUNK_BYTES = 64` 字节——不是 256 字节的整数倍，所以第二次的起点
    落在块中间，正是 unalign 存在的理由。
    """
    preg = vf.create_mask(pattern=pl.MaskPattern.ALL, dtype=FP32)
    areg = vf.load_unalign_init()
    vf.load_unalign_pre(areg, src)

    first = vf.load_unalign(areg, src, CHUNK_BYTES)
    vf.store_align(dst, first, preg, 0)


@pl.jit(auto_mutex=True)
def load_chain(a: pl.Tensor[[1, N], FP32], out: pl.Tensor[[1, N], FP32]):
    ga = pl.make_tile_group(type=_tile_type(), addrs=_A, mutex_ids=[0])
    g0 = pl.make_tile_group(type=_tile_type(), addrs=_O[0], mutex_ids=[1])
    with pl.section_vector():
        ta, t0 = ga.current(), g0.current()
        pl.load(ta, a, [0, 0])
        _load_chain(ta, t0)
        pl.store(out, t0, [0, 0])


# ---------------------------------------------------------------------------
# 二、store 链（调用方驱动）：stride 自己传
# ---------------------------------------------------------------------------


@pl.vector_function
def _store_chain(src, dst):
    """两次 `store_unalign` 各写 `CHUNK_BYTES`，最后 `store_unalign_post` 冲零头。

    **`post` 不是可选的。** 前面两次写的是「凑满整块的部分」，不满一块的尾巴留在
    跟踪寄存器里；`post` 把它写出去。少了它，输出尾部会缺数据——而且**不报错**。

    **整型 `stride` 必须配 `post_update=True`。** 文档参数表写的是「仅 post_update = True
    时有效」，但默认值是 `False`——照默认写能过解析，到 bisheng 才炸在
    `static assertion failed ... The last argument can only be 'POST_UPDATE'`。
    传标量 stride 却不让游标推进本来就没有意义，这个组合不该有默认值。
    """
    reg = vf.load_align(src, 0)
    areg = vf.unalign_reg_for_store()

    vf.store_unalign(dst, reg, areg, CHUNK_BYTES, post_update=True)
    vf.store_unalign(dst, reg, areg, CHUNK_BYTES, post_update=True)
    vf.store_unalign_post(dst, areg, CHUNK_BYTES, post_update=True)


@pl.jit(auto_mutex=True)
def store_chain(a: pl.Tensor[[1, N], FP32], out: pl.Tensor[[1, N], FP32]):
    ga = pl.make_tile_group(type=_tile_type(), addrs=_A, mutex_ids=[0])
    g0 = pl.make_tile_group(type=_tile_type(), addrs=_O[0], mutex_ids=[1])
    with pl.section_vector():
        ta, t0 = ga.current(), g0.current()
        pl.load(ta, a, [0, 0])
        _store_chain(ta, t0)
        pl.store(out, t0, [0, 0])


# ---------------------------------------------------------------------------
# 三、store 链（数据驱动）：stream compaction
# ---------------------------------------------------------------------------


@pl.vector_function
def _squeeze_chain(src, dst):
    """变长筛选结果紧密排布，两个循环继续同一条输出流。

    `squeeze(..., gather_mode=STORE_REG)` 做两件事：把命中的元素压到低位，
    **并把有效字节数写进 AR 特殊寄存器**。紧跟的 `squeeze_store_unalign` 从 AR 读那个数
    并使用 AR 的累计目的位移——所以它不需要 `stride`。

    两轮筛选共用同一个 `areg`、中间不收尾：第二轮从第一轮停下的字节位置接着写。
    这里继续的是同一条存储链；独立输出重新初始化的例子见 make_independent_squeeze_streams。

    **`vf.clear_spr()` 不能漏。** AR 是特殊寄存器，进 `@pl.vector_function` 时里面是上一段
    代码留下的值。漏掉它不报错也不挂——`squeeze_store_unalign` 拿一个陈旧的字节数当推进量，
    输出是没被写过的 Tile 残留内容，**看起来像「搬运没生效」而不是「初始化漏了」**。
    """
    preg = vf.create_mask(pattern=pl.MaskPattern.ALL, dtype=FP32)
    a = vf.load_align(src, 0)

    # 先把输出 Tile 清零。**UB 不是零初始化的**，而这条链只写 32 个元素里的前一段，
    # 没写到的位置是上一个 kernel 留下的内容——不清零的话「尾部是什么」不可预测。
    # `vf.full` 产生结果，必须先命名——内联进 `store_align` 会被解析期拦。
    zeros = vf.full(pl.const(0.0, FP32), dtype=FP32)
    vf.store_align(dst, zeros, preg, 0)

    areg = vf.unalign_reg_for_store()
    vf.clear_spr()  # ← 清 AR；漏了就拿到陈旧字节数，静默出错

    # 第一轮：挑出 > 48 的
    hit_hi = vf.gt(a, pl.const(48.0, FP32), preg)
    packed_hi = vf.squeeze(a, hit_hi, gather_mode=pl.SqueezeMode.STORE_REG)
    vf.squeeze_store_unalign(dst, packed_hi, areg)

    # 第二轮：挑出 <= 16 的，**追加**在后面
    hit_lo = vf.le(a, pl.const(16.0, FP32), preg)
    packed_lo = vf.squeeze(a, hit_lo, gather_mode=pl.SqueezeMode.STORE_REG)
    vf.squeeze_store_unalign(dst, packed_lo, areg)

    vf.squeeze_store_unalign_post(dst, areg)  # 一次收尾，冲掉零头


@pl.jit(auto_mutex=True)
def squeeze_chain(a: pl.Tensor[[1, N], FP32], out: pl.Tensor[[1, N], FP32]):
    ga = pl.make_tile_group(type=_tile_type(), addrs=_A, mutex_ids=[0])
    g0 = pl.make_tile_group(type=_tile_type(), addrs=_O[0], mutex_ids=[1])
    with pl.section_vector():
        ta, t0 = ga.current(), g0.current()
        pl.load(ta, a, [0, 0])
        _squeeze_chain(ta, t0)
        pl.store(out, t0, [0, 0])


def make_independent_squeeze_streams(cols):
    """Two independent outputs initialize AR separately, including across VF calls."""
    padded = ((cols + VL_FP32 - 1) // VL_FP32) * VL_FP32
    tile_bytes = padded * 4

    @pl.vector_function
    def pack(src, dst, direction):
        all_lanes = vf.create_mask(pattern=pl.MaskPattern.ALL, dtype=FP32)
        zeros = vf.full(0.0, all_lanes, dtype=FP32)
        for col in pl.range(0, cols, VL_FP32):
            mask = vf.update_mask(pl.min(VL_FP32, cols - col), dtype=FP32)
            vf.store_align(dst, zeros, mask, [0, col])
        align = vf.unalign_reg_for_store()
        vf.clear_spr()
        for col in pl.range(0, cols, VL_FP32):
            mask = vf.update_mask(pl.min(VL_FP32, cols - col), dtype=FP32)
            value = vf.load_align(src, [0, col])
            signed = vf.muls(value, direction, mask)
            selected = vf.gt(signed, 0.0, mask)
            packed = vf.squeeze(value, selected, gather_mode=pl.SqueezeMode.STORE_REG)
            vf.squeeze_store_unalign(dst, packed, align)
        vf.squeeze_store_unalign_post(dst, align)

    @pl.jit(auto_mutex=True)
    def two_streams(
        x: pl.Tensor[[1, cols], FP32],
        positive: pl.Tensor[[1, cols], FP32],
        negative: pl.Tensor[[1, cols], FP32],
    ):
        with pl.section_vector():
            tile_type = pl.TileType(
                shape=[1, padded],
                dtype=FP32,
                target_memory=pl.MemorySpace.Vec,
                valid_shape=[1, cols],
            )
            gx = pl.make_tile_group(type=tile_type, addrs=[0], mutex_ids=[0])
            gp = pl.make_tile_group(type=tile_type, addrs=[tile_bytes], mutex_ids=[1])
            gn = pl.make_tile_group(
                type=tile_type, addrs=[2 * tile_bytes], mutex_ids=[2]
            )
            tx, tp, tn = gx.next(), gp.next(), gn.next()
            pl.load(tx, x, [0, 0])
            pack(tx, tp, 1.0)
            pack(tx, tn, -1.0)
            pl.store(positive, tp, [0, 0])
            pl.store(negative, tn, [0, 0])

    return two_streams


# ---------------------------------------------------------------------------
# 四、负面用例：连着两次 squeeze(STORE_REG) 不夹 store —— **不能上板**
# ---------------------------------------------------------------------------


@pl.vector_function
def _two_squeezes_in_a_row(src, dst):
    """两次 `squeeze(STORE_REG)` 之间没有 `squeeze_store_unalign`。

    反例：squeeze-alternation —— 这一例的内容就是这条违规本身，只解析不上板

    **这段代码不能上板。** 第二次 squeeze 会覆盖 AR 里还没被消费的字节数，
    官方文档说这种用法会导致硬件挂死。本例只验**解析期不拦**这个事实
    （`test_two_squeezes_in_a_row_are_not_rejected_at_parse_time` 只解析，不执行）。

    钉住它是因为：工具链既不报错也不警告，而后果是挂设备——这正是
    [skill constraints](../../../../SKILL.md#constraints-and-workspace-policy)「同一类错误反复出现 → skill 的 `scripts/` 加确定性检查」
    该拦的第一条规则。**那条规则现在有了**：`check.py` 的 `squeeze-alternation`。
    上面那行 `反例：` 声明就是让它放行的东西；把这个函数改回合规写法，声明会立刻报错。
    """
    preg = vf.create_mask(pattern=pl.MaskPattern.ALL, dtype=FP32)
    a = vf.load_align(src, 0)
    areg = vf.unalign_reg_for_store()
    vf.clear_spr()

    hit_hi = vf.gt(a, pl.const(48.0, FP32), preg)
    first = vf.squeeze(a, hit_hi, gather_mode=pl.SqueezeMode.STORE_REG)
    hit_lo = vf.le(a, pl.const(16.0, FP32), preg)
    second = vf.squeeze(a, hit_lo, gather_mode=pl.SqueezeMode.STORE_REG)  # ← AR 被覆盖

    vf.squeeze_store_unalign(dst, first, areg)
    vf.squeeze_store_unalign(dst, second, areg)
    vf.squeeze_store_unalign_post(dst, areg)


@pl.jit(auto_mutex=True)
def two_squeezes_in_a_row(a: pl.Tensor[[1, N], FP32], out: pl.Tensor[[1, N], FP32]):
    """**只用来解析，不要调用它。** 见 `_two_squeezes_in_a_row` 的说明。"""
    ga = pl.make_tile_group(type=_tile_type(), addrs=_A, mutex_ids=[0])
    g0 = pl.make_tile_group(type=_tile_type(), addrs=_O[0], mutex_ids=[1])
    with pl.section_vector():
        ta, t0 = ga.current(), g0.current()
        pl.load(ta, a, [0, 0])
        _two_squeezes_in_a_row(ta, t0)
        pl.store(out, t0, [0, 0])
