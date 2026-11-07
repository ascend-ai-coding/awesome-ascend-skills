# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

"""写法：原始。填充区：`TilePad` 定填什么，`pl.fillpad` 定什么时候填。

**三件事分在三个地方，这是这一例的主题。**

| | 写在哪 | 管什么 |
| --- | --- | --- |
| `pad=pl.TilePad.zero / max / min / null` | **目的 `TileType` 上**，声明期 | 填充区填成什么值 |
| `pl.set_validshape(src, [r, c])` | 运行期，作用在**源 tile** 上 | 哪一块是有效区 |
| `pl.fillpad(dst, src, mode=...)` | 一条指令 | 把有效区之外真的写掉 |

`TilePad` 只是个声明——**不调 `fillpad` 的话填充区里是什么都没发生过**，也就是上一次运行留在
UB 里的残留。三者缺一个，结果就是「有效区对、外面是脏数据」。

`FillPadMode` 三个值，收窄源的方式是同一个（`set_validshape`），差别在**目的**：

| | `dst` 的形状 | `dst` 的地址 |
| --- | --- | --- |
| `NORMAL`（缺省） | 与 `src` 同形 | 与 `src` **不同** |
| `EXPAND` | **比 `src` 大** | 与 `src` 不同 |
| `INPLACE` | 与 `src` 同形 | 与 `src` **同址** |

**这一例每个 kernel 都手工插同步，一个都没用 `auto_mutex=True`。** 不是风格选择：
`auto_mutex` 靠 `pl.make_tile_group` 的 `mutex_ids` 推依赖，而这些 kernel 用的是裸
`pl.make_tile`，没有 mutex id 可推，于是 MTE2 的 load、V 的 fill、MTE3 的 store 之间
**一条同步都不会插**，结果是时序相关的垃圾而且一个错误都不报。写这一例时踩了这个坑，
代价见 `test.py` §七——那一节是这一例最值得读的部分。

用 INT32 而不是浮点：`max` / `min` 在整型上是明确的 `2147483647` / `-2147483648`，
一眼能认出来是不是填充值；FP32 的极值读出来是 `3.4e38`，和"算错了"长得太像。

第一手来源：`pypto/python/tests/st/pypto_pro/frontend/datacopy/test_pro_fillpad.py`——
那份 ST 的三个 kernel 也都是 `@pl.jit()` 配手工 `sync_src` / `sync_dst`。
"""

import pypto_pro.language as pl

I32 = pl.DT_INT32

ROWS = 8
COLS = 8
VALID_ROWS = 5
VALID_COLS = 7

# `EXPAND` 的目的 tile 比源宽一倍。列数翻倍而不是行数：行数翻倍在 RowMajor 下只是多几行，
# 列数翻倍会改变每行的 stride，能多查出一类错。
WIDE_COLS = 16

VA_SRC = 0x0000  # 8 × 8 INT32 = 0x100 字节
VA_DST = 0x0100  # NORMAL / INPLACE 用；EXPAND 的 8 × 16 占 0x100..0x300

_PADS = {
    "zero": pl.TilePad.zero,
    "max": pl.TilePad.max,
    "min": pl.TilePad.min,
    "null": pl.TilePad.null,
}


def _src_type(pad=pl.TilePad.null):
    """源 tile 的 `valid_shape=[-1, -1]` 是「运行期再定」，`set_validshape` 才填上具体数。

    缺省 `pad=null` 就是「不声明填充语义」——源不需要它，只有 `INPLACE` 那一支例外。
    """
    return pl.TileType(
        shape=[ROWS, COLS],
        dtype=I32,
        target_memory=pl.MemorySpace.Vec,
        valid_shape=[-1, -1],
        pad=pad,
    )


def _dst_type(pad, cols=COLS):
    return pl.TileType(
        shape=[ROWS, cols], dtype=I32, target_memory=pl.MemorySpace.Vec, pad=pad
    )


# ---------------------------------------------------------------------------
# 一、NORMAL：四种 TilePad 各一个 kernel
#
# 类型注解里不能放变量（parser 读的是 AST），所以四个 pad 值各造一个 kernel。
# **工厂里不能给 `@pl.jit` 传 `name=`**：传了之后四个实例互相盖，只有最后一个生效，
# 而且症状是「另外三个报最后那个的解析错误」，很难往这个方向想。
# ---------------------------------------------------------------------------


def _make_normal(pad_name):
    pad = _PADS[pad_name]

    @pl.jit()
    def kernel(x: pl.Tensor[[ROWS, COLS], I32], out: pl.Tensor[[ROWS, COLS], I32]):
        src = pl.make_tile(_src_type(), addr=VA_SRC)
        dst = pl.make_tile(_dst_type(pad), addr=VA_DST)
        with pl.section_vector():
            pl.load(src, x, [0, 0])
            # 一次流水间同步要 sync_src + sync_dst 两条，参数一模一样，成对出现。
            # 只写一条不报错也拦不住任何东西：set 那一半在 sync_src，wait 那一半在 sync_dst。
            pl.system.sync_src(
                set_pipe=pl.PipeType.MTE2, wait_pipe=pl.PipeType.V, event_id=0
            )
            pl.system.sync_dst(
                set_pipe=pl.PipeType.MTE2, wait_pipe=pl.PipeType.V, event_id=0
            )
            pl.set_validshape(src, [VALID_ROWS, VALID_COLS])
            pl.fillpad(dst, src)
            pl.system.sync_src(
                set_pipe=pl.PipeType.V, wait_pipe=pl.PipeType.MTE3, event_id=1
            )
            pl.system.sync_dst(
                set_pipe=pl.PipeType.V, wait_pipe=pl.PipeType.MTE3, event_id=1
            )
            pl.store(out, dst, [0, 0])

    return kernel


fillpad_zero = _make_normal("zero")
fillpad_max = _make_normal("max")
fillpad_min = _make_normal("min")
fillpad_null = _make_normal("null")


# ---------------------------------------------------------------------------
# 二、EXPAND 与 INPLACE
# ---------------------------------------------------------------------------


@pl.jit()
def fillpad_expand(
    x: pl.Tensor[[ROWS, COLS], I32], out: pl.Tensor[[ROWS, WIDE_COLS], I32]
):
    """`EXPAND`：源 `8 × 8`（有效区 `5 × 7`），目的 `8 × 16`，多出来的一半整个是填充值。

    **收窄源的方式和 `NORMAL` 完全一样**，还是 `set_validshape`；`EXPAND` 改的是目的——
    它允许 `dst` 比 `src` 大。我第一版把它理解成「源 tile 的形状更小、不用 set_validshape」，
    结果拿到的是 UB 残留：那样写源根本没被收窄，`fillpad` 没有有效区边界可用。
    """
    src = pl.make_tile(_src_type(), addr=VA_SRC)
    dst = pl.make_tile(_dst_type(pl.TilePad.zero, WIDE_COLS), addr=VA_DST)
    with pl.section_vector():
        pl.load(src, x, [0, 0])
        pl.system.sync_src(
            set_pipe=pl.PipeType.MTE2, wait_pipe=pl.PipeType.V, event_id=0
        )
        pl.system.sync_dst(
            set_pipe=pl.PipeType.MTE2, wait_pipe=pl.PipeType.V, event_id=0
        )
        pl.set_validshape(src, [VALID_ROWS, VALID_COLS])
        pl.fillpad(dst, src, mode=pl.FillPadMode.EXPAND)
        pl.system.sync_src(
            set_pipe=pl.PipeType.V, wait_pipe=pl.PipeType.MTE3, event_id=1
        )
        pl.system.sync_dst(
            set_pipe=pl.PipeType.V, wait_pipe=pl.PipeType.MTE3, event_id=1
        )
        pl.store(out, dst, [0, 0])


@pl.jit()
def fillpad_inplace(x: pl.Tensor[[ROWS, COLS], I32], out: pl.Tensor[[ROWS, COLS], I32]):
    """`INPLACE`：`src` 与 `dst` **同一个地址**，原地把有效区之外填掉。

    省一块 UB，代价是源数据没了。两个 `make_tile` 的 `addr` 都是 `VA_SRC`——**这不是笔误**，
    `INPLACE` 的定义就是同址。写成不同地址不会报错，会安静地得到 `NORMAL` 的行为。

    源类型这里**也带 `pad=`**（照 ST 写）。同址意味着这两个 tile 描述同一块内存，
    让两边的 pad 声明一致，比让它们不一致再去猜哪个生效要省事。
    """
    src = pl.make_tile(_src_type(pl.TilePad.zero), addr=VA_SRC)
    dst = pl.make_tile(_dst_type(pl.TilePad.zero), addr=VA_SRC)
    with pl.section_vector():
        pl.load(src, x, [0, 0])
        pl.system.sync_src(
            set_pipe=pl.PipeType.MTE2, wait_pipe=pl.PipeType.V, event_id=0
        )
        pl.system.sync_dst(
            set_pipe=pl.PipeType.MTE2, wait_pipe=pl.PipeType.V, event_id=0
        )
        pl.set_validshape(src, [VALID_ROWS, VALID_COLS])
        pl.fillpad(dst, src, mode=pl.FillPadMode.INPLACE)
        pl.system.sync_src(
            set_pipe=pl.PipeType.V, wait_pipe=pl.PipeType.MTE3, event_id=1
        )
        pl.system.sync_dst(
            set_pipe=pl.PipeType.V, wait_pipe=pl.PipeType.MTE3, event_id=1
        )
        pl.store(out, dst, [0, 0])


# ---------------------------------------------------------------------------
# 三、对照：声明了 pad 但不调 fillpad
# ---------------------------------------------------------------------------


@pl.jit()
def pad_declared_but_not_filled(
    x: pl.Tensor[[ROWS, COLS], I32], out: pl.Tensor[[ROWS, COLS], I32]
):
    """`pad=zero` 声明了，`fillpad` 没调。**填充区里是 UB 残留，不是 0。**

    这一支是给前面那些判据当对照的：如果没有它，「`fillpad` 真的写了东西」就只是个说法。
    它的输出**没有数值判据**（残留不可预测），测试只断言它和 `fillpad_zero` 的输出**不同**
    ——而且为此要先往那块 UB 里写进已知的非零数据，否则残留碰巧是 0 就什么都证明不了。
    """
    src = pl.make_tile(_src_type(), addr=VA_SRC)
    dst = pl.make_tile(_dst_type(pl.TilePad.zero), addr=VA_DST)
    with pl.section_vector():
        pl.load(dst, x, [0, 0])  # 先把整块写满已知数据
        pl.load(src, x, [0, 0])
        pl.system.sync_src(
            set_pipe=pl.PipeType.MTE2, wait_pipe=pl.PipeType.V, event_id=0
        )
        pl.system.sync_dst(
            set_pipe=pl.PipeType.MTE2, wait_pipe=pl.PipeType.V, event_id=0
        )
        pl.set_validshape(src, [VALID_ROWS, VALID_COLS])
        # 这里刻意不调 pl.fillpad
        pl.system.sync_src(
            set_pipe=pl.PipeType.V, wait_pipe=pl.PipeType.MTE3, event_id=1
        )
        pl.system.sync_dst(
            set_pipe=pl.PipeType.V, wait_pipe=pl.PipeType.MTE3, event_id=1
        )
        pl.store(out, dst, [0, 0])
