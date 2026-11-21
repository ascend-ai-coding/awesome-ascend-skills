# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

"""写法：原始。GM 步长视图（`pl.make_tensor`）与指针算术（`pl.make_ptr` / `pl.addptr`）。

**三个 API 的分工：**

| | 做什么 | 单位 |
| --- | --- | --- |
| `pl.make_ptr(tensor)` | 从张量取出裸指针；带 `dtype=` 时顺便重解释元素类型 | — |
| `pl.addptr(ptr, n)` | 指针前移 `n` 个**元素**（不是字节、不是行） | 元素 |
| `pl.make_tensor(src, shape, stride, dtype=)` | 造一个视图：自己的形状、自己的步长、可选自己的 dtype | 步长按**元素** |

`make_tensor` 的第一个实参既可以是裸指针，也可以直接是一个 `pl.Tensor`。

**视图的步长对通过它的每一次访问都生效，偏移也走步长。** 这是这一例的主题：想表达「从第 i 行
开始、之后每隔 s 个元素取一行」，**不能**用 `pl.load(tile, view, [i, 0])`——那个 `i` 会被乘上
步长。要用 `pl.addptr` 移基址，让视图只负责行距。ST 的注释把这件事写得很清楚：

> The view is positioned by pointer rather than by a load offset: its stride is the *pair*
> spacing, so an offset through the view would step by s, not by one line.

**这套接口替掉了旧的 `pl.set_stride`**（同一份 ST 的模块 docstring）：那个函数就地改写一个张量
的描述符，于是偏移算术仍按**原来的**步长做、效果泄漏到之后对该张量的每一次访问，而且在只需要
一种布局变体时也会写坏张量本身的声明。视图是独立的张量，没有这三个问题。

第一手来源：`pypto/python/tests/st/pypto_pro/frontend/datacopy/test_strided_gather_view.py`。
"""

from functools import lru_cache

import pypto_pro.language as pl

FP32 = pl.DT_FP32
I32 = pl.DT_INT32

ROWS = 16
COLS = 16
HALF_ROWS = ROWS // 2

# 行距取两行：视图的 `[2 * COLS, 1]` 表示「行与行之间隔 2 * COLS 个元素」。
# **取 2 而不是 1**：步长等于行宽时视图和原张量一样，判据什么都证明不了。
ROW_STRIDE = 2 * COLS
CONTIGUOUS_STRIDE = COLS

# 偏移一个步长之后还能读多少行：视图 8 行，跳掉 1 个步长剩 7 行。
# **单独起一个常量而不是在注解里写 `HALF_ROWS - 1`**：`pl.Tensor[[...]]` 的形状注解是按 AST 读的，
# 里面放算术表达式不保险，而形状注解和实际传进来的张量必须逐轴相等（对不上时报
# `Tensor parameter 'out' axis 0 mismatch`）。
OFFSET_ROWS = HALF_ROWS - 1

# `addptr` 的偏移：一整行。用行的倍数而不是任意数，是为了让期望值能写成 `x[1::2]` 这种形式。
PTR_OFFSET_ROWS = 1

# 重叠写：视图 A 覆盖第 0..7 行，视图 B 从第 4 行开始覆盖 8 行（第 4..11 行），重叠 4 行。
OVERLAP_ROWS = 8
OVERLAP_START = 4

VA_FIRST = 0x0000  # 8 × 16 FP32 = 0x200
VA_SECOND = 0x0200


def _tile(rows, dtype, addr):
    """**只能用位置实参调它。** 内联辅助函数不支持关键字实参（[写法约束](../../../../references/api/constraints.md) 有这条）:

    ```
    Call to inline function '_tile' only supports plain positional arguments
    ```

    参数顺序按"最常变的放前面"排，省得在调用处数位置。
    """
    return pl.make_tile(
        pl.TileType(shape=[rows, COLS], dtype=dtype, target_memory=pl.MemorySpace.Vec),
        addr=addr,
    )


@pl.jit()
def view_every_other_row(
    x: pl.Tensor[[ROWS, COLS], FP32], out: pl.Tensor[[HALF_ROWS, COLS], FP32]
):
    """步长 `2 * COLS` 的视图：一次 `pl.load` 取到第 0、2、4 … 14 行。

    视图的形状是 `[8, 16]`，但它铺在一个 `16 × 16` 的张量上——**形状和步长一起决定读哪些元素**，
    形状单独说明不了任何事情。
    """
    tile = _tile(HALF_ROWS, FP32, VA_FIRST)
    with pl.section_vector():
        view = pl.make_tensor(x, [HALF_ROWS, COLS], [ROW_STRIDE, 1])
        pl.load(tile, view, [0, 0])
        pl.system.sync_src(
            set_pipe=pl.PipeType.MTE2, wait_pipe=pl.PipeType.MTE3, event_id=0
        )
        pl.system.sync_dst(
            set_pipe=pl.PipeType.MTE2, wait_pipe=pl.PipeType.MTE3, event_id=0
        )
        pl.store(out, tile, [0, 0])


@pl.jit()
def view_positioned_by_pointer(
    x: pl.Tensor[[ROWS, COLS], FP32], out: pl.Tensor[[HALF_ROWS, COLS], FP32]
):
    """同样的步长，但基址被 `addptr` 前移一行：取到第 1、3、5 … 15 行。

    `pl.addptr(pl.make_ptr(x), PTR_OFFSET_ROWS * COLS)` —— **偏移的单位是元素**，所以"一行"要
    自己写成 `1 * COLS`。这是移基址的正确办法；用 `pl.load(tile, view, [1, 0])` 会走步长，
    落到第 2 行，见 `offset_goes_through_the_stride`。
    """
    tile = _tile(HALF_ROWS, FP32, VA_FIRST)
    with pl.section_vector():
        base = pl.addptr(pl.make_ptr(x), PTR_OFFSET_ROWS * COLS)
        view = pl.make_tensor(base, [HALF_ROWS, COLS], [ROW_STRIDE, 1])
        pl.load(tile, view, [0, 0])
        pl.system.sync_src(
            set_pipe=pl.PipeType.MTE2, wait_pipe=pl.PipeType.MTE3, event_id=0
        )
        pl.system.sync_dst(
            set_pipe=pl.PipeType.MTE2, wait_pipe=pl.PipeType.MTE3, event_id=0
        )
        pl.store(out, tile, [0, 0])


@pl.jit()
def offset_goes_through_the_stride(
    x: pl.Tensor[[ROWS, COLS], FP32], out: pl.Tensor[[OFFSET_ROWS, COLS], FP32]
):
    """**这一例的主题**：`pl.load` 的偏移 `[1, 0]` 不是"往下一行"，是"往下一个步长"。

    视图步长 `2 * COLS`，偏移 `[1, 0]` 于是落到第 **2** 行，读到第 2、4、6 … 行。
    和 `view_positioned_by_pointer`（`addptr` 移一行、读第 1、3、5 …）放在一起看，
    两者的差别就是这一条规则。

    视图只有 8 行而原张量 16 行，偏移一个步长之后还剩 7 行可读；第 8 行会越过张量末尾，
    所以这里的 tile 只取 7 行。**不是凑数**：取 8 行就成了另一个话题（越界读）。
    """
    tile = _tile(OFFSET_ROWS, FP32, VA_FIRST)
    with pl.section_vector():
        view = pl.make_tensor(x, [HALF_ROWS, COLS], [ROW_STRIDE, 1])
        pl.load(tile, view, [1, 0])
        pl.system.sync_src(
            set_pipe=pl.PipeType.MTE2, wait_pipe=pl.PipeType.MTE3, event_id=0
        )
        pl.system.sync_dst(
            set_pipe=pl.PipeType.MTE2, wait_pipe=pl.PipeType.MTE3, event_id=0
        )
        pl.store(out, tile, [0, 0])


@pl.jit()
def view_reinterprets_the_dtype(
    x: pl.Tensor[[ROWS, COLS], FP32], out: pl.Tensor[[HALF_ROWS, COLS], I32]
):
    """`dtype=` 让视图重解释元素类型：同一批字节，按 INT32 读出来。

    FP32 和 INT32 都是 4 字节，所以形状和步长不变，**变的只是怎么解释那些字节**。
    判据是 numpy 的 `x.view(np.int32)`——逐位相等，因为两边做的是同一件事。
    """
    tile = _tile(HALF_ROWS, I32, VA_FIRST)
    with pl.section_vector():
        view = pl.make_tensor(x, [HALF_ROWS, COLS], [CONTIGUOUS_STRIDE, 1], dtype=I32)
        pl.load(tile, view, [0, 0])
        pl.system.sync_src(
            set_pipe=pl.PipeType.MTE2, wait_pipe=pl.PipeType.MTE3, event_id=0
        )
        pl.system.sync_dst(
            set_pipe=pl.PipeType.MTE2, wait_pipe=pl.PipeType.MTE3, event_id=0
        )
        pl.store(out, tile, [0, 0])


@pl.jit()
def overlapping_writes(
    x: pl.Tensor[[ROWS, COLS], FP32], out: pl.Tensor[[ROWS, COLS], FP32]
):
    """两个**重叠**的视图各写一次：后写的赢，框架不拦、不警告。

    视图 A 覆盖第 0..7 行，视图 B 从第 4 行起覆盖 8 行（第 4..11 行），重叠的是第 4..7 行。

    **重叠区的结果不确定。** 实测 20 次（4 个进程各 5 次）：14 次是 B 的数据、6 次是 A 的，
    同一个进程里连着跑也会变。我一开始写的是"两条 store 都在 MTE3 上、同一条流水内按程序序,
    所以结果确定"——**那是错的**，而且它差点变成一条会偶发失败的绿测试。

    确定的部分只有三段（判据只断言这三段）：

    - 第 0..3 行 ← 只有视图 A 覆盖，是 `x` 的第 0..3 行
    - 第 8..11 行 ← 只有视图 B 覆盖，是 `x` 的第 12..15 行
    - 第 12..15 行 ← 没被写过，保持主机侧的哨兵

    **这里没有任何"别名检查"。** 两个视图指向同一块 GM 这件事框架看不见，写重叠了也不报。
    要顺序就自己排，见 `overlapping_writes_ordered`。
    """
    first = _tile(OVERLAP_ROWS, FP32, VA_FIRST)
    second = _tile(OVERLAP_ROWS, FP32, VA_SECOND)
    with pl.section_vector():
        pl.load(first, x, [0, 0])
        pl.load(second, x, [OVERLAP_ROWS, 0])
        pl.system.sync_src(
            set_pipe=pl.PipeType.MTE2, wait_pipe=pl.PipeType.MTE3, event_id=0
        )
        pl.system.sync_dst(
            set_pipe=pl.PipeType.MTE2, wait_pipe=pl.PipeType.MTE3, event_id=0
        )

        view_a = pl.make_tensor(out, [OVERLAP_ROWS, COLS], [CONTIGUOUS_STRIDE, 1])
        pl.store(view_a, first, [0, 0])

        base_b = pl.addptr(pl.make_ptr(out), OVERLAP_START * COLS)
        view_b = pl.make_tensor(base_b, [OVERLAP_ROWS, COLS], [CONTIGUOUS_STRIDE, 1])
        pl.store(view_b, second, [0, 0])


@pl.jit()
def overlapping_writes_ordered(
    x: pl.Tensor[[ROWS, COLS], FP32], out: pl.Tensor[[ROWS, COLS], FP32]
):
    """同样两个重叠视图，中间加一条 `pl.system.bar_mte3()`：结果变确定，后写的赢。

    **这是 `bar_*` 的正当用法**，也是它在本仓唯一出现的地方：两条 `pl.store` 都在 MTE3 上，
    要的是"流水内不并行"，而不是"让 A 等 B"。flag 在这里用不上——`sync_src` / `sync_dst` 要求
    `set_pipe != wait_pipe`，MTE3→MTE3 会被解析期直接拒（见 `same_pipe_flag_is_refused`）。

    实测 20 次（4 个进程各 5 次）全部是后写的赢；不加这条 barrier 的版本 20 次里有 6 次相反。
    """
    first = _tile(OVERLAP_ROWS, FP32, VA_FIRST)
    second = _tile(OVERLAP_ROWS, FP32, VA_SECOND)
    with pl.section_vector():
        pl.load(first, x, [0, 0])
        pl.load(second, x, [OVERLAP_ROWS, 0])
        pl.system.sync_src(
            set_pipe=pl.PipeType.MTE2, wait_pipe=pl.PipeType.MTE3, event_id=0
        )
        pl.system.sync_dst(
            set_pipe=pl.PipeType.MTE2, wait_pipe=pl.PipeType.MTE3, event_id=0
        )

        view_a = pl.make_tensor(out, [OVERLAP_ROWS, COLS], [CONTIGUOUS_STRIDE, 1])
        pl.store(view_a, first, [0, 0])

        pl.system.bar_mte3()

        base_b = pl.addptr(pl.make_ptr(out), OVERLAP_START * COLS)
        view_b = pl.make_tensor(base_b, [OVERLAP_ROWS, COLS], [CONTIGUOUS_STRIDE, 1])
        pl.store(view_b, second, [0, 0])


@pl.jit()
def same_pipe_flag_is_refused(
    x: pl.Tensor[[ROWS, COLS], FP32], out: pl.Tensor[[ROWS, COLS], FP32]
):
    """想用 flag 串两条 MTE3 的 store：`set_pipe == wait_pipe` 被解析期拒。

    留着它是为了让「这里只能用 `bar_mte3()`」有个凭据，而不是一句说法。
    """
    first = _tile(OVERLAP_ROWS, FP32, VA_FIRST)
    with pl.section_vector():
        pl.load(first, x, [0, 0])
        pl.system.sync_src(
            set_pipe=pl.PipeType.MTE3, wait_pipe=pl.PipeType.MTE3, event_id=0
        )
        pl.store(out, first, [0, 0])


@lru_cache(None)
def make_word_gather(
    batch,
    source_rows,
    selected_rows,
    width,
    dtype_name="float16",
    word_bits=64,
    threads=128,
):
    """Copy indexed FP16/BF16 rows without changing their storage bits.

    Source [batch,source_rows,width], indices [batch,selected_rows] and disjoint
    output are contiguous. Runtime indices must lie in [0,source_rows); bases
    and row bytes must align to word_bits/8. The factory checks dimensions,
    not pointer alignment or index values; partial words need a narrower path.
    """
    if min(batch, source_rows, selected_rows, width, threads) < 1:
        raise ValueError("dimensions and threads must be positive")
    if dtype_name not in ("float16", "bfloat16") or word_bits not in (32, 64):
        raise ValueError("FP16/BF16 storage and 32/64-bit integer words are supported")
    elements_per_word = word_bits // 16
    if width % elements_per_word:
        raise ValueError("row byte length must be a whole number of words")
    words = width // elements_per_word
    dtype = pl.DT_FP16 if dtype_name == "float16" else pl.DT_BF16
    carrier = pl.DT_INT32 if word_bits == 32 else pl.DT_INT64

    @pl.vector_function(mode="simt", max_threads=threads)
    def copy_words(src, indices, dst):
        start = pl.simt.block_idx().x * threads + pl.simt.linear_thread_idx()
        step = pl.simt.grid_dim().x * pl.simt.block_dim().x
        for linear in pl.range(start, batch * selected_rows * words, step):
            b = linear // (selected_rows * words)
            row = (linear // words) % selected_rows
            col = linear % words
            selected = indices[b, row]
            dst[b, row, col] = src[b, selected, col]

    @pl.jit()
    def kernel(
        x: pl.Tensor[[batch, source_rows, width], dtype],
        indices: pl.Tensor[[batch, selected_rows], pl.DT_INT32],
        out: pl.Tensor[[batch, selected_rows, width], dtype],
    ):
        # Shape and implicit contiguous strides now count integer words.
        src = pl.make_tensor(x, [batch, source_rows, words], dtype=carrier)
        dst = pl.make_tensor(out, [batch, selected_rows, words], dtype=carrier)
        with pl.section_vector():
            copy_words[threads](src, indices, dst)

    return kernel
