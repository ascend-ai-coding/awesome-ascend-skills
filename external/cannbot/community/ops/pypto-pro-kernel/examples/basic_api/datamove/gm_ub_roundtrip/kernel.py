# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

"""写法：原始。GM↔UB 的两套搬运接口：绝对偏移与**块索引**。

`pl.load` / `pl.store` 的偏移是**元素**坐标；`pl.load_tile` / `pl.store_tile` 的偏移是
**tile 块索引**，框架内部乘上 tile 形状。同一块数据两种写法能对上，这一例就是把这件事钉住。

| 接口 | 偏移单位 | 取 `[2, 2]` 号块时怎么写 |
| --- | --- | --- |
| `pl.load(tile, tensor, [r, c])` | 元素 | `[2 * ROWS, 2 * COLS]` |
| `pl.load_tile(tile, tensor, [i, j])` | **tile 块** | `[2, 2]` |

块索引只在「张量正好切成整数个 tile」时省事；形状切不齐的时候仍然要用绝对偏移，因为最后一块
的起点不是块索引能表达的。所以两套都得会，不存在哪一套更"新"。

第一手来源：`pypto/python/tests/st/pypto_pro/frontend/` 下 `pl.load_tile(` 的真实用法
（`datacopy/` 与 `matmul/` 系）。
"""

import pypto_pro.language as pl

FP32 = pl.DT_FP32

ROWS = 8
COLS = 16
TILE_ELEMS = ROWS * COLS
TILE_BYTES = TILE_ELEMS * 4  # 512

# 张量是 2×2 个 tile 块
BLOCKS = 2
FULL_ROWS = ROWS * BLOCKS
FULL_COLS = COLS * BLOCKS

VA_IN = 0x0000
VA_OUT = 0x0400


def _tile_type():
    return pl.TileType(shape=[ROWS, COLS], dtype=FP32, target_memory=pl.MemorySpace.Vec)


@pl.jit(auto_mutex=True, name="roundtrip_absolute")
def roundtrip_absolute(
    x: pl.Tensor[[FULL_ROWS, FULL_COLS], FP32],
    out: pl.Tensor[[FULL_ROWS, FULL_COLS], FP32],
):
    """用**元素坐标**取 `[1, 1]` 号块，原样写回同一个位置。

    这是最小往返：GM → UB → GM，**一个 tile，中间不算任何东西**。不算东西是刻意的——这一例要
    看的是偏移怎么算，一旦中间做了运算或者多用一个 tile，对不上的时候就分不清是偏移错了、
    片上拷贝错了，还是同步错了。
    """
    group = pl.make_tile_group(type=_tile_type(), addrs=[VA_IN], mutex_ids=[0])
    with pl.section_vector():
        tile = group.next()
        pl.load(tile, x, [ROWS, COLS])  # 元素坐标：第 8 行、第 16 列
        pl.store(out, tile, [ROWS, COLS])


@pl.jit(auto_mutex=True, name="roundtrip_block_index")
def roundtrip_block_index(
    x: pl.Tensor[[FULL_ROWS, FULL_COLS], FP32],
    out: pl.Tensor[[FULL_ROWS, FULL_COLS], FP32],
):
    """同一块数据，用**块索引**取。和 `roundtrip_absolute` 的输出必须逐位相同。

    `[1, 1]` 被框架乘成 `[1 * 8, 1 * 16] = [8, 16]`——正好是上面那个 kernel 手写的坐标。
    """
    group = pl.make_tile_group(type=_tile_type(), addrs=[VA_IN], mutex_ids=[0])
    with pl.section_vector():
        tile = group.next()
        pl.load_tile(tile, x, [1, 1])  # 块索引：第 1 行块、第 1 列块
        pl.store_tile(out, tile, [1, 1])


@pl.jit(auto_mutex=True, name="roundtrip_all_blocks")
def roundtrip_all_blocks(
    x: pl.Tensor[[FULL_ROWS, FULL_COLS], FP32],
    out: pl.Tensor[[FULL_ROWS, FULL_COLS], FP32],
):
    """四个块逐个搬一遍，整张张量被覆盖。

    块索引在这种「切得齐」的遍历里最省事：循环变量直接就是索引，不用乘。**注意 `pl.range`**
    ——vf 体外的循环也必须用它，Python 的 `range` 在解析期会被拒。
    """
    group = pl.make_tile_group(
        type=_tile_type(), addrs=[VA_IN, VA_OUT], mutex_ids=[0, 1]
    )
    with pl.section_vector():
        for i in pl.range(0, BLOCKS):
            for j in pl.range(0, BLOCKS):
                tile = group.next()  # 两个槽轮转，相邻两轮不撞同一块 UB
                pl.load_tile(tile, x, [i, j])
                pl.store_tile(out, tile, [i, j])
