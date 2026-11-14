# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

"""写法：原始。tile 的三层形状：物理形状、有效区、重解释后的视图。

| | 谁定 | 什么时候 | 改数据吗 |
| --- | --- | --- | --- |
| **物理形状** | `TileType(shape=...)` | 声明期，编译期常量 | 决定占多少 buffer |
| **有效区** | `pl.set_validshape(tile, [r, c])` | 运行期 | 不改，只限定搬多少 |
| **重解释的视图** | `pl.reinterpret(tile, shape=/dtype=/layout=)` | 声明期 | **不生成任何指令** |

三层是叠加的，不是互斥的。

**`set_validshape` 限定的是搬运量，而且对 load 和 store 两侧都生效**（实测，见 `test.py` §三）：
收窄放在 `pl.load` 前面就少搬进来，放在 `pl.store` 前面就少搬出去。

**`pl.reinterpret` 是编译期的视图声明**，官方 docstring 原话「not a conversion — no
load/move/store/cast instruction is generated」。它在**原地址、原大小**重新声明一个 handle，
继承 memory space / fractal / pad / compact / mutex 映射，**但不继承 valid_shape**——
新 handle 从动态有效区开始，要窄就再调一次 `set_validshape`。原 handle 不变。

`pl.set_validshape` 和 `pl.reinterpret` **都可以作用在整个 `make_tile_group` 上**：前者给组里
每个 tile 都设，后者返回一个新组并保留 mutex 映射。

**这一例只讲 shape 这一面。** `reinterpret` 改 dtype 那一面（含 5 个进不了搬运通道的 dtype
怎么绕）在 `support/dtype_carriers`；收窄与 `compact` 在 matmul 路径上的互动在
`references/diagnostics/validation-traps.md` #4。

第一手来源：`pypto/python/tests/st/pypto_pro/frontend/fa/`（658 处 `make_tile` 系调用）与
官方 `TileType.md`。
"""

import pypto_pro.language as pl
from pypto_pro.language import Vf as vf

FP32 = pl.DT_FP32
I64 = pl.DT_INT64

ROWS = 8
COLS = 16

# 有效区取 4 × 8：两个轴都收窄，而且都不是整半。**只收一个轴的话，行列写反也会"对"。**
VALID_ROWS = 4
VALID_COLS = 8

# 重解释的形状：元素数不变（8×16 = 4×32），所以 footprint 不变，只是逻辑上换了个形状。
WIDE_ROWS = 4
WIDE_COLS = 32

# footprint 超出：16 × 16 = 256 个 FP32 = 1024 字节，而原 buffer 只有 512。
TOO_MANY_ROWS = 16

VA_FIRST = 0x0000  # 8 × 16 FP32 = 0x200
VA_SECOND = 0x0200


def _tile(addr):
    return pl.make_tile(
        pl.TileType(
            shape=[ROWS, COLS],
            dtype=FP32,
            target_memory=pl.MemorySpace.Vec,
            valid_shape=[-1, -1],
        ),
        addr=addr,
    )


@pl.jit()
def whole_tile(x: pl.Tensor[[ROWS, COLS], FP32], out: pl.Tensor[[ROWS, COLS], FP32]):
    """基线：不收窄、不重解释。物理形状就是逻辑形状，整块 `8 × 16` 进出。

    没有这条基线，后面"少搬了"的判据就没有对照——**"128 个元素全被写"这件事本身要先证一遍**。
    """
    tile = _tile(VA_FIRST)
    with pl.section_vector():
        pl.load(tile, x, [0, 0])
        pl.system.sync_src(
            set_pipe=pl.PipeType.MTE2, wait_pipe=pl.PipeType.MTE3, event_id=0
        )
        pl.system.sync_dst(
            set_pipe=pl.PipeType.MTE2, wait_pipe=pl.PipeType.MTE3, event_id=0
        )
        pl.store(out, tile, [0, 0])


@pl.jit()
def narrow_before_store(
    x: pl.Tensor[[ROWS, COLS], FP32], out: pl.Tensor[[ROWS, COLS], FP32]
):
    """整块装进来，收窄到 `4 × 8` 再搬出去：GM 上只有左上角 `4 × 8` 被写。

    **收窄不改 UB 里的数据**，tile 里仍然是完整的 `8 × 16`；它改的是这条 `pl.store` 搬多少。
    """
    tile = _tile(VA_FIRST)
    with pl.section_vector():
        pl.load(tile, x, [0, 0])
        pl.system.sync_src(
            set_pipe=pl.PipeType.MTE2, wait_pipe=pl.PipeType.MTE3, event_id=0
        )
        pl.system.sync_dst(
            set_pipe=pl.PipeType.MTE2, wait_pipe=pl.PipeType.MTE3, event_id=0
        )
        pl.set_validshape(tile, [VALID_ROWS, VALID_COLS])
        pl.store(out, tile, [0, 0])


@pl.jit()
def narrow_before_load(
    x: pl.Tensor[[ROWS, COLS], FP32], out: pl.Tensor[[ROWS, COLS], FP32]
):
    """收窄放在 `pl.load` **前面**：只从 GM 搬 `4 × 8` 进来。

    输出和 `narrow_before_store` 一样（有效区还没恢复，store 也只搬那么多），但**过程不同**：
    这一版 tile 里除了左上角都没被写过。区别在 UB 里，GM 上看不出来——所以这两条判据放在一起
    只能说明"两侧都生效"，说明不了"哪一侧生效了"。要区分得看 IR 或者故意留脏数据。
    """
    tile = _tile(VA_FIRST)
    with pl.section_vector():
        pl.set_validshape(tile, [VALID_ROWS, VALID_COLS])
        pl.load(tile, x, [0, 0])
        pl.system.sync_src(
            set_pipe=pl.PipeType.MTE2, wait_pipe=pl.PipeType.MTE3, event_id=0
        )
        pl.system.sync_dst(
            set_pipe=pl.PipeType.MTE2, wait_pipe=pl.PipeType.MTE3, event_id=0
        )
        pl.store(out, tile, [0, 0])


@pl.jit()
def reinterpret_to_another_shape(
    x: pl.Tensor[[ROWS, COLS], FP32], out: pl.Tensor[[WIDE_ROWS, WIDE_COLS], FP32]
):
    """按 `8 × 16` 装进来，按 `4 × 32` 搬出去。**同一块 buffer，一条搬运指令都没多。**

    元素数不变（128），所以 footprint 不变。判据是 `x.reshape(4, 32)` 逐位相等——
    行优先下这两个形状覆盖同一串字节，**这正是"不生成指令"的可观测后果**。
    """
    tile = _tile(VA_FIRST)
    with pl.section_vector():
        pl.load(tile, x, [0, 0])
        view = pl.reinterpret(tile, shape=[WIDE_ROWS, WIDE_COLS])
        pl.system.sync_src(
            set_pipe=pl.PipeType.MTE2, wait_pipe=pl.PipeType.MTE3, event_id=0
        )
        pl.system.sync_dst(
            set_pipe=pl.PipeType.MTE2, wait_pipe=pl.PipeType.MTE3, event_id=0
        )
        pl.store(out, view, [0, 0])


@pl.jit()
def reinterpret_drops_the_valid_shape(
    x: pl.Tensor[[ROWS, COLS], FP32], out: pl.Tensor[[ROWS, COLS], FP32]
):
    """先收窄到 `4 × 8`，再 `reinterpret` 成同样的形状：**收窄丢了**，整块 `8 × 16` 被搬出去。

    docstring 说得很明确：「valid_shape is NOT inherited (the new handle starts with a dynamic
    valid shape, equivalent to -1)」。判据是「一个哨兵都不剩」——和 `narrow_before_store`
    剩 96 个哨兵对照，差别一眼能看出来。

    **这是个陷阱**：`reinterpret` 读起来像"只改元数据"，容易以为收窄会跟着走。
    """
    tile = _tile(VA_FIRST)
    with pl.section_vector():
        pl.load(tile, x, [0, 0])
        pl.system.sync_src(
            set_pipe=pl.PipeType.MTE2, wait_pipe=pl.PipeType.MTE3, event_id=0
        )
        pl.system.sync_dst(
            set_pipe=pl.PipeType.MTE2, wait_pipe=pl.PipeType.MTE3, event_id=0
        )
        pl.set_validshape(tile, [VALID_ROWS, VALID_COLS])
        view = pl.reinterpret(tile, shape=[ROWS, COLS])
        pl.store(out, view, [0, 0])


@pl.jit(auto_mutex=True)
def narrow_a_whole_group(
    x: pl.Tensor[[ROWS, COLS], FP32], out: pl.Tensor[[ROWS, COLS], FP32]
):
    """`pl.set_validshape(group, ...)` 给组里**每个** tile 都设有效区。

    输出和 `narrow_before_store` 一样是左上角 `4 × 8`。这一条要证的不是结果，是**一次调用作用在
    整组上**——组里两个槽都被收窄了，所以第二个槽搬出去的也只有 `4 × 8`。
    """
    group = pl.make_tile_group(
        type=pl.TileType(
            shape=[ROWS, COLS],
            dtype=FP32,
            target_memory=pl.MemorySpace.Vec,
            valid_shape=[-1, -1],
        ),
        addrs=[VA_FIRST, VA_SECOND],
        mutex_ids="auto",
        depth=2,
    )
    with pl.section_vector():
        pl.set_validshape(group, [VALID_ROWS, VALID_COLS])
        first = group.next()
        pl.load(first, x, [0, 0])
        second = group.next()
        pl.load(second, x, [0, 0])
        pl.store(out, second, [0, 0])


@pl.jit(auto_mutex=True)
def reinterpret_a_whole_group(
    x: pl.Tensor[[ROWS, COLS], FP32], out: pl.Tensor[[WIDE_ROWS, WIDE_COLS], FP32]
):
    """`pl.reinterpret(group, shape=...)` 返回一个新组。**这个 kernel 只用来比 IR，不上板。**

    要钉的是两件事，都在解析后的 IR 里看得见：

    1. 新组的 tile 形状是 `[4, 32]`，地址和大小不变（原地址、原 512 字节）；
    2. **mutex 映射保留**——锁用的还是原来那个 `_tg_group_mutex_ids` tuple。
       这一条要紧：新组丢了 mutex id 的话 `auto_mutex=True` 对它失效，**而那种失效是静默的**
       （`references/diagnostics/validation-traps.md` #7）。

    **不上板的理由是 `.next()` 的游标**：它在原组和新组之间**共享**。下面 `group.next()` 拿到
    槽 0，`wide.next()` 拿到槽 1；槽 1 没有初始化，不能用于数值验收。
    """
    group = pl.make_tile_group(
        type=pl.TileType(
            shape=[ROWS, COLS],
            dtype=FP32,
            target_memory=pl.MemorySpace.Vec,
            valid_shape=[-1, -1],
        ),
        addrs=[VA_FIRST, VA_SECOND],
        mutex_ids="auto",
        depth=2,
    )
    with pl.section_vector():
        tile = group.next()
        pl.load(tile, x, [0, 0])
        wide = pl.reinterpret(group, shape=[WIDE_ROWS, WIDE_COLS])
        pl.store(out, wide.next(), [0, 0])


# ---------------------------------------------------------------------------
# 只解析、不上板：reinterpret 的三条检查
# ---------------------------------------------------------------------------


@pl.jit()
def reinterpret_beyond_the_buffer(
    x: pl.Tensor[[ROWS, COLS], FP32], out: pl.Tensor[[ROWS, COLS], FP32]
):
    """`16 × 16` FP32 = 1024 字节，原 buffer 只有 512。**解析期拒。**"""
    tile = _tile(VA_FIRST)
    with pl.section_vector():
        pl.load(tile, x, [0, 0])
        view = pl.reinterpret(tile, shape=[TOO_MANY_ROWS, COLS])
        pl.store(out, view, [0, 0])


@pl.jit()
def reinterpret_dtype_without_shape(
    x: pl.Tensor[[ROWS, COLS], FP32], out: pl.Tensor[[ROWS, COLS], FP32]
):
    """改 dtype 就必须同时给 shape——**元素数要重新说一遍**，不让框架替你猜。"""
    tile = _tile(VA_FIRST)
    with pl.section_vector():
        pl.load(tile, x, [0, 0])
        view = pl.reinterpret(tile, dtype=I64)
        pl.store(out, view, [0, 0])


@pl.jit()
def reinterpret_with_nothing_to_override(
    x: pl.Tensor[[ROWS, COLS], FP32], out: pl.Tensor[[ROWS, COLS], FP32]
):
    """`dtype` / `shape` / `layout` 一个都不给。**空的重解释被拒**，而不是返回一个同样的 handle。"""
    tile = _tile(VA_FIRST)
    with pl.section_vector():
        pl.load(tile, x, [0, 0])
        view = pl.reinterpret(tile)
        pl.store(out, view, [0, 0])


F16, F32 = pl.DT_FP16, pl.DT_FP32


@pl.vector_function
def _load_view(src, dst):
    mask = vf.create_mask(pattern=pl.MaskPattern.ALL, dtype=F32)
    values = vf.load_align(src, [2, 0])
    vf.store_align(dst, values, mask, 0)


@pl.jit(auto_mutex=True)
def load_reinterpreted_layout(
    x: pl.Tensor[[16, 64], F32], out: pl.Tensor[[1, 64], F32]
):
    buf = pl.make_tile_group(
        type=pl.TileType(shape=[16, 64], dtype=F32, target_memory=pl.MemorySpace.Vec),
        addrs=0x0,
        mutex_ids="auto",
        depth=1,
    )
    dst = pl.make_tile_group(
        type=pl.TileType(shape=[1, 64], dtype=F32, target_memory=pl.MemorySpace.Vec),
        addrs=0x1000,
        mutex_ids="auto",
        depth=1,
    )
    view = pl.reinterpret(buf, shape=[64, 16], layout=pl.ZN)
    with pl.section_vector():
        source = buf.current()
        alias = view.current()
        target = dst.current()
        pl.load(source, x, [0, 0])
        _load_view(alias, target)
        pl.store(out, target, [0, 0])


@pl.pipeline.stage
def _produce(a, b, ag, bg, la, lb, ac, shared, narrow):
    aa = ag.current()
    bb = bg.current()
    cc = ac.current()
    pl.load(aa, a, [0, 0])
    pl.load(bb, b, [0, 0])
    pl.gemm(cc, aa, bb, l0a=la, l0b=lb)
    target = shared.next()
    pl.move(target, cc, acc_to_vec_mode=pl.AccToVecMode.DualModeSplitM)
    pl.gemm(cc, aa, bb, l0a=la, l0b=lb, init=False)
    # 原组与视图共享 cursor；本探针只有一槽，current 取刚才的同一槽。
    target_view = narrow.current()
    pl.move(target_view, cc, acc_to_vec_mode=pl.AccToVecMode.DualModeSplitM)


@pl.pipeline.stage
def _consume(out, shared):
    target = shared.next()
    pl.store(out, target, [pl.get_subblock_idx() * 16, 0])


def make_receive_view(view_rows):
    """目标视图较窄，但真实 backing 恒能容纳每个 AIV 的 16 行。"""
    if not 1 <= view_rows <= 16:
        raise ValueError("view_rows must be in [1, 16]")

    @pl.jit(
        auto_mutex=True,
        pipeline=pl.pipeline.PipelineConfig(preload=0),
        name=f"receive_view_{view_rows}",
    )
    def kernel(
        a: pl.Tensor[[32, 16], F16],
        b: pl.Tensor[[16, 32], F16],
        out: pl.Tensor[[32, 32], F32],
    ):
        ag = pl.make_tile_group(
            type=pl.TileType(
                shape=[32, 16], dtype=F16, target_memory=pl.MemorySpace.Mat
            ),
            addrs=0x0,
            mutex_ids="auto",
            depth=1,
        )
        bg = pl.make_tile_group(
            type=pl.TileType(
                shape=[16, 32], dtype=F16, target_memory=pl.MemorySpace.Mat
            ),
            addrs=0x400,
            mutex_ids="auto",
            depth=1,
        )
        la = pl.make_tile_group(
            type=pl.TileType(
                shape=[32, 16], dtype=F16, target_memory=pl.MemorySpace.Left
            ),
            addrs=0x0,
            mutex_ids="auto",
            depth=1,
        )
        lb = pl.make_tile_group(
            type=pl.TileType(
                shape=[16, 32], dtype=F16, target_memory=pl.MemorySpace.Right
            ),
            addrs=0x0,
            mutex_ids="auto",
            depth=1,
        )
        ac = pl.make_tile_group(
            type=pl.TileType(
                shape=[32, 32], dtype=F32, target_memory=pl.MemorySpace.Acc
            ),
            addrs=0x0,
            mutex_ids="auto",
            depth=1,
        )
        shared = pl.make_tile_group(
            type=pl.TileType(
                shape=[16, 32], dtype=F32, target_memory=pl.MemorySpace.Vec
            ),
            addrs=0x0,
            mutex_ids="auto",
            depth=1,
            fwd_ids=[0],
            bwd_ids=[1],
        )
        narrow = pl.reinterpret(shared, shape=[view_rows, 32])
        for item in pl.range(1):
            with pl.section_cube():
                _produce(a, b, ag, bg, la, lb, ac, shared, narrow)
            with pl.section_vector():
                _consume(out, shared)

    return kernel
