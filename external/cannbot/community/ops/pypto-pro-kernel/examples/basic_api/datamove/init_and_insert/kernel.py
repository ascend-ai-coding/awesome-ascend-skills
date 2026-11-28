# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

"""写法：原始。GM 初始化与 UB 子块插入。

init_output 的 offset/size 是按行优先拉平的元素区间，调用必须放在 section_vector 内。
insert 的偏移是二维元素坐标，搬运范围来自源 Tile 的有效形状。

初始化由 V 填充临时 UB，再经 MTE3 写 GM；其 V→MTE3 依赖由该操作生成，跨核消费仍需
调用方建立核间同步。静态 offset+size 越过 GM 张量元素数时，解析期拒绝。

本例显式放置 bare Tile 并手工建立流水依赖。初始化的临时 UB 从地址 0 开始，FP32 路径
覆盖 8192 字节；load→init_output→store 探针用三个地址检查此工具链行为。正常初始化与
insert 用独立数学参考及区间外哨兵验证。
"""

import pypto_pro.language as pl

FP32 = pl.DT_FP32
I32 = pl.DT_INT32

ROWS = 8
COLS = 16
SMALL_ROWS = 4
SMALL_COLS = 8

# 部分初始化的区间是行优先拉平后的 [32,96)，包含四个完整行。
INIT_OFFSET = 32
INIT_SIZE = 64
INIT_VALUE = 7.5

# 小块从 x 的右下角读，插到大块的 [2, 4]。两处都不是 [0, 0]：
# 源和目的都在原点的话，行列写反也会得到正确结果。
SMALL_SRC_ROW = 4
SMALL_SRC_COL = 8
INSERT_ROW = 2
INSERT_COL = 4

VA_BIG = 0x0000  # 8 × 16 FP32 = 0x200
VA_SMALL = 0x0200  # 4 × 8  FP32 = 0x80


def _tile(shape, addr, dtype=FP32):
    return pl.make_tile(
        pl.TileType(shape=shape, dtype=dtype, target_memory=pl.MemorySpace.Vec),
        addr=addr,
    )


# ---------------------------------------------------------------------------
# 一、init_output
# ---------------------------------------------------------------------------


@pl.jit()
def init_whole_tensor(out: pl.Tensor[[ROWS, COLS], FP32]):
    """把整个 GM 张量填成一个常量。**这个 kernel 没有别的动作，也没有输入。**

    单独留一个"只做初始化"的 kernel：和 `pl.store` 混在一起的话，「是 init 写的还是 store 写的」
    就分不清了。

    `init_output` 是 Vector 操作，调用位于 `section_vector` 内。
    """
    with pl.section_vector():
        pl.init_output(out, offset=0, size=ROWS * COLS, value=-1.0)


@pl.jit()
def init_partial_region(out: pl.Tensor[[ROWS, COLS], FP32]):
    """只填 `[32, 96)` 这 64 个元素。**偏移和大小都是元素数，不是行数、不是字节数。**

    主机侧先把整张填成哨兵，所以这一例能同时判两件事：该填的填了，**不该动的没动**。
    """
    with pl.section_vector():
        pl.init_output(out, offset=INIT_OFFSET, size=INIT_SIZE, value=INIT_VALUE)


# `init_output` 覆写的 UB 宽度，从 codegen 算出来并在板上验过边界：
# `(256 * 32) / sizeof(T)` 个元素 × sizeof(T) = 8192 字节（b8/b16/b32 都是这个数；
# b64 被夹到 64 个元素 = 512 字节）。
INIT_OUTPUT_CLOBBER_BYTES = 8192
VA_SAFE = 0x2000
"""模型说的第一个不被覆写的地址，判据在这个边界上验过（`0x1000` 被覆写、`0x2000` 完好）。"""


def make_load_then_init(addr):
    """先把 `x` 载入 `addr` 处的 tile，**再** `init_output`，最后把那个 tile 存出去。

    `init_output` 的目标是另一个张量（`scratch`），所以 `out` 的内容只能来自那个 tile
    ——**覆写和「初始化写对了没有」两件事分得开**。

    `addr` 是唯一的自变量：`0x0` / `0x1000` 在覆写范围里，`0x2000` 在外面。
    """

    @pl.jit()
    def probe(
        x: pl.Tensor[[ROWS, COLS], FP32],
        out: pl.Tensor[[ROWS, COLS], FP32],
        scratch: pl.Tensor[[ROWS, COLS], FP32],
    ):
        """反例：init-output-needs-a-drain —— 这一例量的就是覆写本身，`addr` 是自变量

        `0x0` / `0x1000` 落在覆写范围里（判据要求它们被覆写），`0x2000` 在外面。
        加了屏障这个探针就什么都测不到了。
        """
        src = _tile([ROWS, COLS], addr)
        with pl.section_vector():
            pl.load(src, x, [0, 0])
            pl.system.sync_src(
                set_pipe=pl.PipeType.MTE2, wait_pipe=pl.PipeType.V, event_id=0
            )
            pl.system.sync_dst(
                set_pipe=pl.PipeType.MTE2, wait_pipe=pl.PipeType.V, event_id=0
            )
        with pl.section_vector():
            pl.init_output(scratch, offset=0, size=ROWS * COLS, value=-1.0)
            pl.system.sync_src(
                set_pipe=pl.PipeType.V, wait_pipe=pl.PipeType.MTE3, event_id=1
            )
            pl.system.sync_dst(
                set_pipe=pl.PipeType.V, wait_pipe=pl.PipeType.MTE3, event_id=1
            )
            pl.store(out, src, [0, 0])

    return probe


@pl.jit()
def init_after_tile_declaration(
    x: pl.Tensor[[ROWS, COLS], FP32], out: pl.Tensor[[ROWS, COLS], FP32]
):
    """初始化 GM 后排空流水，再将输入载入 bare Tile，避免临时 UB 的读写重叠。"""
    src = _tile([ROWS, COLS], VA_BIG)
    with pl.section_vector():
        pl.init_output(out, offset=0, size=ROWS * COLS, value=-1.0)
        pl.system.bar_all()
        pl.load(src, x, [0, 0])


@pl.jit()
def init_int_tensor_with_whole_float(out: pl.Tensor[[ROWS, COLS], I32]):
    """INT32 目标按向零截断转换浮点常量；整数值 3.0 转换为 3。

    正负小数常量的转换分别由相邻的 1.5 和 -1.5 用例检查。
    """
    with pl.section_vector():
        pl.init_output(out, offset=0, size=ROWS * COLS, value=3.0)


@pl.jit()
def init_int_tensor_with_fraction(out: pl.Tensor[[ROWS, COLS], I32]):
    """INT32 目标把浮点常量 1.5 按向零截断转换为 1。"""
    with pl.section_vector():
        pl.init_output(out, offset=0, size=ROWS * COLS, value=1.5)


@pl.jit()
def init_int_tensor_with_negative_fraction(out: pl.Tensor[[ROWS, COLS], I32]):
    """负值 -1.5 截断为 -1，区分向零截断与向下取整。"""
    with pl.section_vector():
        pl.init_output(out, offset=0, size=ROWS * COLS, value=-1.5)


# ---------------------------------------------------------------------------
# 二、insert
# ---------------------------------------------------------------------------


@pl.jit()
def insert_small_into_big(
    x: pl.Tensor[[ROWS, COLS], FP32], out: pl.Tensor[[ROWS, COLS], FP32]
):
    """把 `x` 右下角的 `4 × 8` 插到大块的 `[2, 4]`。

    源取右下角、目的取 `[2, 4]`——两处都偏离原点，这样行列写反、单位弄错、差一都能被判出来。
    `insert` 走 V 流水，所以前后各一对 flag。
    """
    big = _tile([ROWS, COLS], VA_BIG)
    small = _tile([SMALL_ROWS, SMALL_COLS], VA_SMALL)
    with pl.section_vector():
        pl.load(big, x, [0, 0])
        pl.load(small, x, [SMALL_SRC_ROW, SMALL_SRC_COL])
        pl.system.sync_src(
            set_pipe=pl.PipeType.MTE2, wait_pipe=pl.PipeType.V, event_id=0
        )
        pl.system.sync_dst(
            set_pipe=pl.PipeType.MTE2, wait_pipe=pl.PipeType.V, event_id=0
        )
        pl.insert(big, small, [INSERT_ROW, INSERT_COL])
        pl.system.sync_src(
            set_pipe=pl.PipeType.V, wait_pipe=pl.PipeType.MTE3, event_id=1
        )
        pl.system.sync_dst(
            set_pipe=pl.PipeType.V, wait_pipe=pl.PipeType.MTE3, event_id=1
        )
        pl.store(out, big, [0, 0])


@pl.jit()
def insert_flush_to_the_corner(
    x: pl.Tensor[[ROWS, COLS], FP32], out: pl.Tensor[[ROWS, COLS], FP32]
):
    """偏移 `[4, 8]`：`4 + 4 = 8`、`8 + 8 = 16`，**恰好贴住右下角**。

    边界上的那一格必须单独测：越界检查如果写成 `>=` 而不是 `>`，这个合法的用法会被误拒。
    """
    big = _tile([ROWS, COLS], VA_BIG)
    small = _tile([SMALL_ROWS, SMALL_COLS], VA_SMALL)
    with pl.section_vector():
        pl.load(big, x, [0, 0])
        pl.load(small, x, [0, 0])
        pl.system.sync_src(
            set_pipe=pl.PipeType.MTE2, wait_pipe=pl.PipeType.V, event_id=0
        )
        pl.system.sync_dst(
            set_pipe=pl.PipeType.MTE2, wait_pipe=pl.PipeType.V, event_id=0
        )
        pl.insert(big, small, [ROWS - SMALL_ROWS, COLS - SMALL_COLS])
        pl.system.sync_src(
            set_pipe=pl.PipeType.V, wait_pipe=pl.PipeType.MTE3, event_id=1
        )
        pl.system.sync_dst(
            set_pipe=pl.PipeType.V, wait_pipe=pl.PipeType.MTE3, event_id=1
        )
        pl.store(out, big, [0, 0])


# ---------------------------------------------------------------------------
# 三、只解析、不上板
# ---------------------------------------------------------------------------


@pl.jit()
def insert_offset_with_one_element(
    x: pl.Tensor[[ROWS, COLS], FP32], out: pl.Tensor[[ROWS, COLS], FP32]
):
    """偏移只给一维。**这是 `insert` 唯一被检查的东西**：元素个数必须恰好 2。"""
    big = _tile([ROWS, COLS], VA_BIG)
    small = _tile([SMALL_ROWS, SMALL_COLS], VA_SMALL)
    with pl.section_vector():
        pl.load(big, x, [0, 0])
        pl.insert(big, small, [INSERT_ROW])
        pl.store(out, big, [0, 0])


@pl.jit()
def insert_past_the_edge(
    x: pl.Tensor[[ROWS, COLS], FP32], out: pl.Tensor[[ROWS, COLS], FP32]
):
    """源区域越过目的 UB Tile 一行；只解析，不执行可能越界的搬运。"""
    big = _tile([ROWS, COLS], VA_BIG)
    small = _tile([SMALL_ROWS, SMALL_COLS], VA_SMALL)
    with pl.section_vector():
        pl.load(big, x, [0, 0])
        pl.insert(big, small, [ROWS - SMALL_ROWS + 1, COLS - SMALL_COLS])
        pl.store(out, big, [0, 0])


@pl.jit()
def init_size_beyond_the_tensor(out: pl.Tensor[[ROWS, COLS], FP32]):
    """静态 `size` 是张量元素数的四倍；只验证解析期的越界拒绝，不执行。"""
    with pl.section_vector():
        pl.init_output(out, offset=0, size=ROWS * COLS * 4, value=-1.0)
