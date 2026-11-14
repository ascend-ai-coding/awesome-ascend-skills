# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

"""写法：原始。GM 就地初始化（`pl.init_output`）与小块插进大块（`pl.insert`）。

**两个 API 的单位不一样，这是最容易错的地方。**

| | 偏移单位 | 形状从哪来 |
| --- | --- | --- |
| `pl.init_output(tensor, offset=, size=, value=)` | **元素**（一维，按行优先拉平） | 没有形状，只有一段 `[offset, offset+size)` |
| `pl.insert(dst_tile, src_tile, [row, col])` | **二维元素坐标** | `src_tile` 自己的形状 |

`init_output` 作用在 **GM 张量**上，`insert` 作用在**片上 tile** 之间（TINSERT）。

**`init_output` 自己插 V→MTE3 的同步**（docstring：「V→MTE3 synchronization is inserted
automatically」），所以它不需要手工配同步；但它**从不发跨核屏障**，数据要给别的核或 cube 子核
用时，两侧都得显式 `pl.system.sync_all(core_type=pl.SyncCoreType.MIX)`。这一例是单核，用不上。

**`init_output` 要求 kernel 里有一个 vector section，空的也行。** 它内部用 V 流水填一块临时
UB 再走 MTE3 搬出去；kernel 里没有 `pl.section_vector()` 时那两条 op 落到一个没有 V 流水的核上，
报的是 C++ 层的 `PIPE_V` 范围错和 `copy_ubuf_to_gm_align_v2 does not support the given target
feature`——**看不出和 section 有关**。加一个 `with pl.section_vector(): pass` 就好。实测见
`test.py` §五。

**文档写了、但解析期不检查的两条**（实测，见 `test.py` §六）：`init_output` 的「必须在 tile
分配之前调用」，以及 `insert` 的偏移是否让 `src` 落在 `dst` 界内。

**那条「必须在 tile 分配之前」的真正后果：`init_output` 覆写 UB `0x0` 起的 8192 字节。**
它的临时 tile 在生成的 CCE 里是 `Tile<Vec, T, 1, chunk>` + `TASSIGN(..., 0x0)`,
`chunk = (256 * 32) / sizeof(T)` 个元素，而 `TEXPANDS` 在 `SetValidShape` **之前**——
所以填的是整个 chunk，不是 `size` 个元素。**和 `size=` 给多少无关。** 实测见 `test.py` §七。

第一手来源：`pypto/python/tests/st/pypto_pro/frontend/fa/`（36 处）与 `datacopy/`（29 处）。
"""

import pypto_pro.language as pl

FP32 = pl.DT_FP32
I32 = pl.DT_INT32

ROWS = 8
COLS = 16
SMALL_ROWS = 4
SMALL_COLS = 8

# 部分初始化的区间：第 32 个元素起 64 个，也就是行优先下的第 2、3、4、5 行前半。
# **刻意不落在整行边界上**：落在整行上的话「按元素」和「按行」两种理解都对，判据就分不出来。
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

    **那个空的 `section_vector` 是必需的，不是装饰。** 去掉它编译就失败（`test.py` §五）。
    """
    pl.init_output(out, offset=0, size=ROWS * COLS, value=-1.0)
    with pl.section_vector():
        pass


@pl.jit()
def init_partial_region(out: pl.Tensor[[ROWS, COLS], FP32]):
    """只填 `[32, 96)` 这 64 个元素。**偏移和大小都是元素数，不是行数、不是字节数。**

    主机侧先把整张填成哨兵，所以这一例能同时判两件事：该填的填了，**不该动的没动**。
    """
    pl.init_output(out, offset=INIT_OFFSET, size=INIT_SIZE, value=INIT_VALUE)
    with pl.section_vector():
        pass


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
        pl.init_output(scratch, offset=0, size=ROWS * COLS, value=-1.0)
        with pl.section_vector():
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
    """`init_output` 放在 `pl.make_tile` **之后**——docstring 说这不行，实际解析期不拦。

    docstring 原文：「Must be called before Tile buffer allocation (``make_tile_group`` etc.)」。

    **`VA_BIG` 是 `0x0000`，而 `init_output` 的临时 tile 也写死在 `0x0`——两者完全重合。**
    生成的 CCE 里是两个 `TASSIGN(..., 0x0)`。这个 kernel 的输出仍然对，只是因为 `src`
    载入之后再没被读过；**不是因为「放后面也行」**。

    而且 `TSTORE`（MTE3 从 UB `0x0` 读）和 `TLOAD`（MTE2 往 UB `0x0` 写）之间**没有同步**,
    两者都正好是 512 字节、同一个地址。这是一个 WAR 竞争，见 `test.py` §五点二。

    **注意它只证到 `make_tile` 这一档**——`pl.mem_space()` / `allocate_tile_group` 那种真正
    的自动分配没在这里试，docstring 点名的正是那一类。

    反例：init-output-needs-a-drain —— 这一例的内容就是那个缺屏障的竞争本身
    """
    src = _tile([ROWS, COLS], VA_BIG)
    pl.init_output(out, offset=0, size=ROWS * COLS, value=-1.0)
    with pl.section_vector():
        pl.load(src, x, [0, 0])


@pl.jit()
def init_int_tensor_with_whole_float(out: pl.Tensor[[ROWS, COLS], I32]):
    """INT32 张量 + `value=3.0`：浮点字面量，但值是整数，**通过**。

    规则不是「value 的类型要和 dtype 一致」，而是**转换不能改变值**——生成的 C++ 里那个字面量
    要过 `-Wliteral-conversion`，它只在值变了的时候报。`value=3.0` 变成 `3` 没丢东西，所以过。
    """
    pl.init_output(out, offset=0, size=ROWS * COLS, value=3.0)
    with pl.section_vector():
        pass


@pl.jit()
def init_int_tensor_with_fraction(out: pl.Tensor[[ROWS, COLS], I32]):
    """INT32 张量 + `value=1.5`：**编译期被拒**，解析期不拦。

    拒的地方在生成的 C++ 上，`-Werror,-Wliteral-conversion`：

    ```
    implicit conversion from 'float' to '...::DType' (aka 'int')
    changes value from 1.5 to 1
    ```

    **消息很准，但拿不到**：框架抛出来的异常只有 `Failed to compile kernel '<name>'`，
    上面那行在 `jit.py:1591` 的日志里。`test.py` 用 `caplog` 去抓它——只断言"编译失败"
    的判据太钝，任何编译错误都会让它绿。
    """
    pl.init_output(out, offset=0, size=ROWS * COLS, value=1.5)
    with pl.section_vector():
        pass


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
    """偏移 `[5, 8]`：`5 + 4 = 9 > 8`，行方向越界一行。**解析期不拦。**

    **这个 kernel 不上板**：越界的 `insert` 会写到目的 tile 之外的 UB，落在哪儿取决于布局，
    结果没有判据可写。它只钉住「解析期不拦」这件事——框架哪天加上界检查，那条测试会红。
    """
    big = _tile([ROWS, COLS], VA_BIG)
    small = _tile([SMALL_ROWS, SMALL_COLS], VA_SMALL)
    with pl.section_vector():
        pl.load(big, x, [0, 0])
        pl.insert(big, small, [ROWS - SMALL_ROWS + 1, COLS - SMALL_COLS])
        pl.store(out, big, [0, 0])


@pl.jit()
def init_size_beyond_the_tensor(out: pl.Tensor[[ROWS, COLS], FP32]):
    """`size` 是张量元素数的 4 倍。**解析期不拦。**

    **这个 kernel 不上板**，而且理由和上面那条不一样：它往 GM 张量之外写，越界的目标不是我们
    自己的 UB 而是设备内存里别的东西。在共享机上试它不值当——**这是明说跳过**。

    因为只解析不编译，它连那个空的 `section_vector` 都不需要（section 的约束在编译期）。
    """
    pl.init_output(out, offset=0, size=ROWS * COLS * 4, value=-1.0)
