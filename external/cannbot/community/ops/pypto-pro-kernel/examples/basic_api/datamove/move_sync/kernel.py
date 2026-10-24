# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

"""写法：原始。跨流水定序：`sync_src` / `sync_dst` 成对的 flag，以及核间的 `sync_all`。

**跨流水定序只有 flag 这一条路。** `pl.system.sync_src` 设标志、`pl.system.sync_dst` 等标志，
两条参数完全相同，成对出现。

| 想让谁等谁 | 写法 |
| --- | --- |
| V 等 MTE2（算之前数据要到位） | `sync_src/sync_dst(set_pipe=MTE2, wait_pipe=V, event_id=n)` |
| MTE3 等 V（搬出之前要算完） | `sync_src/sync_dst(set_pipe=V, wait_pipe=MTE3, event_id=n)` |

**`bar_*` 不是跨流水屏障，这一例刻意不用它。** 它是流水**内**的屏障（排空那条流水），
用在自动同步分析看不见依赖、需要强制串行的少数场合。拿它做跨流水定序会静默算错——
测量与两个真实用法见 `test.py` §六。

`event_id` 的范围是 `[0, 7]`（`system_ops.py` 的 `MAX_FLAG_EVENT_ID`），和跨核事件的
`[0, 15]` 不是同一个池子。

第一手来源：`pypto/python/tests/st/pypto_pro/frontend/datacopy/test_pro_fillpad.py`
（每个 kernel 都是这个形状），以及 `pypto/python/pypto_pro/ir/op/system_ops.py:44-72`
那两张合法路径表。
"""

import pypto_pro.language as pl
from pypto_pro.language import Vf as vf

FP32 = pl.DT_FP32

ROWS = 8
COLS = 16

VA_SRC = 0x0000  # 8 × 16 FP32 = 0x200 字节
VA_DST = 0x0200

# 两条 flag 用不同的 event_id。**同一时刻用同一个号的两对 flag 会互相认错**，
# 所以"随手写 0"在链一长就出问题；这里按流水段编号。
EVENT_LOAD_DONE = 0
EVENT_COMPUTE_DONE = 1


@pl.vector_function
def _double(dst, src):
    mask = vf.create_mask(pattern=pl.MaskPattern.ALL, dtype=FP32)
    for offset in pl.range(0, ROWS * COLS, 64):
        value = vf.load_align(src, offset)
        doubled = vf.add(value, value, mask)
        vf.store_align(dst, doubled, mask, offset)


def _tile(addr):
    return pl.make_tile(
        pl.TileType(shape=[ROWS, COLS], dtype=FP32, target_memory=pl.MemorySpace.Vec),
        addr=addr,
    )


@pl.jit()
def chain_flag_sync(
    x: pl.Tensor[[ROWS, COLS], FP32], out: pl.Tensor[[ROWS, COLS], FP32]
):
    """MTE2 → V → MTE3 三段，两对 flag 把它们串起来。输出 `x * 2`。

    这是手工同步的标准形状，`pl.fillpad` 的那份 ST 每个 kernel 都长这样。三段各在一条流水上：
    `pl.load` 在 MTE2、VF 加法在 V、`pl.store` 在 MTE3，**默认它们是并行的**。
    """
    src = _tile(VA_SRC)
    dst = _tile(VA_DST)
    with pl.section_vector():
        pl.load(src, x, [0, 0])
        pl.system.sync_src(
            set_pipe=pl.PipeType.MTE2, wait_pipe=pl.PipeType.V, event_id=EVENT_LOAD_DONE
        )
        pl.system.sync_dst(
            set_pipe=pl.PipeType.MTE2, wait_pipe=pl.PipeType.V, event_id=EVENT_LOAD_DONE
        )
        _double(dst, src)
        pl.system.sync_src(
            set_pipe=pl.PipeType.V,
            wait_pipe=pl.PipeType.MTE3,
            event_id=EVENT_COMPUTE_DONE,
        )
        pl.system.sync_dst(
            set_pipe=pl.PipeType.V,
            wait_pipe=pl.PipeType.MTE3,
            event_id=EVENT_COMPUTE_DONE,
        )
        pl.store(out, dst, [0, 0])


@pl.jit(auto_mutex=True)
def chain_auto_mutex(
    x: pl.Tensor[[ROWS, COLS], FP32], out: pl.Tensor[[ROWS, COLS], FP32]
):
    """同一条链交给 `auto_mutex=True`——**前提是 tile 来自 `make_tile_group` 并带 `mutex_ids`**。

    和 `chain_flag_sync` 输出必须逐位相同。两者插的不是同一种 op（一边 `mutex_lock_dyn` /
    `mutex_unlock_dyn`，一边 `sync_src_dyn` / `sync_dst_dyn`），但达到的定序等价。

    **换成裸 `pl.make_tile` 的话这个装饰器什么都不做**，见 `references/diagnostics/validation-traps.md` #7。
    """
    group = pl.make_tile_group(
        type=pl.TileType(
            shape=[ROWS, COLS], dtype=FP32, target_memory=pl.MemorySpace.Vec
        ),
        addrs=[VA_SRC, VA_DST],
        mutex_ids="auto",
        depth=2,
    )
    with pl.section_vector():
        src = group.next()
        dst = group.next()
        pl.load(src, x, [0, 0])
        _double(dst, src)
        pl.store(out, dst, [0, 0])


@pl.jit()
def chain_with_sync_all(
    x: pl.Tensor[[ROWS, COLS], FP32], out: pl.Tensor[[ROWS, COLS], FP32]
):
    """在链尾加一条 `sync_all(AIV_ONLY)`：**核间**同步，不是流水间。

    它管的是「所有 AIV 核都到这儿了」，和上面两对 flag 解决的问题是两回事——把它当流水定序用
    不会报错也不会生效。这里只用 `AIV_ONLY`：本例是纯 vector kernel，`MIX` 按文档要连 AIC 一起
    同步，那既不是这一例能判的，也有挂死的风险（`test.py` §七 明说跳过）。
    """
    src = _tile(VA_SRC)
    dst = _tile(VA_DST)
    with pl.section_vector():
        pl.load(src, x, [0, 0])
        pl.system.sync_src(
            set_pipe=pl.PipeType.MTE2, wait_pipe=pl.PipeType.V, event_id=EVENT_LOAD_DONE
        )
        pl.system.sync_dst(
            set_pipe=pl.PipeType.MTE2, wait_pipe=pl.PipeType.V, event_id=EVENT_LOAD_DONE
        )
        _double(dst, src)
        pl.system.sync_src(
            set_pipe=pl.PipeType.V,
            wait_pipe=pl.PipeType.MTE3,
            event_id=EVENT_COMPUTE_DONE,
        )
        pl.system.sync_dst(
            set_pipe=pl.PipeType.V,
            wait_pipe=pl.PipeType.MTE3,
            event_id=EVENT_COMPUTE_DONE,
        )
        pl.store(out, dst, [0, 0])
        pl.system.sync_all(core_type=pl.SyncCoreType.AIV_ONLY)


# ---------------------------------------------------------------------------
# 只解析、不上板：四类被拒的写法各一个
#
# 每个都只有一行和 `chain_flag_sync` 不同。判据匹配**规则文本**，不匹配错误码。
# ---------------------------------------------------------------------------


@pl.jit()
def same_pipe_both_sides(
    x: pl.Tensor[[ROWS, COLS], FP32], out: pl.Tensor[[ROWS, COLS], FP32]
):
    """`set_pipe == wait_pipe`。自己等自己，`system_ops.py:130` 单独拒一条。"""
    src = _tile(VA_SRC)
    dst = _tile(VA_DST)
    with pl.section_vector():
        pl.load(src, x, [0, 0])
        pl.system.sync_src(set_pipe=pl.PipeType.V, wait_pipe=pl.PipeType.V, event_id=0)
        _double(dst, src)
        pl.store(out, dst, [0, 0])


@pl.jit()
def unsupported_aiv_path(
    x: pl.Tensor[[ROWS, COLS], FP32], out: pl.Tensor[[ROWS, COLS], FP32]
):
    """`V → MTE1`：AIV 侧没有这条路。MTE1 是 cube 侧的 L1→L0 搬运流水。

    拒绝消息把**合法的 wait_pipe 全列出来**（`PipeType.MTE2, PipeType.MTE3, PipeType.S`），
    所以撞上这条不用去翻表。
    """
    src = _tile(VA_SRC)
    dst = _tile(VA_DST)
    with pl.section_vector():
        pl.load(src, x, [0, 0])
        _double(dst, src)
        pl.system.sync_src(
            set_pipe=pl.PipeType.V, wait_pipe=pl.PipeType.MTE1, event_id=0
        )
        pl.store(out, dst, [0, 0])


@pl.jit()
def path_only_legal_on_aiv(
    x: pl.Tensor[[ROWS, COLS], FP32], out: pl.Tensor[[ROWS, COLS], FP32]
):
    """`MTE2 → V` 放在 `pl.section_cube()` 里。**同一行在 vector section 里是合法的。**

    这一条是这一例最值得记的：合法性取决于这段代码落在哪一侧。cube 侧没有 V 流水，
    拒绝消息会说 `on AIC` 并列出 AIC 的表。
    """
    src = _tile(VA_SRC)
    dst = _tile(VA_DST)
    with pl.section_cube():
        pl.load(src, x, [0, 0])
        pl.system.sync_src(
            set_pipe=pl.PipeType.MTE2, wait_pipe=pl.PipeType.V, event_id=0
        )
        pl.store(out, dst, [0, 0])


@pl.jit()
def event_id_out_of_range(
    x: pl.Tensor[[ROWS, COLS], FP32], out: pl.Tensor[[ROWS, COLS], FP32]
):
    """`event_id=8`。flag 的池子是 `[0, 7]`，跨核事件的 `[0, 15]` 是另一个池子。"""
    src = _tile(VA_SRC)
    dst = _tile(VA_DST)
    with pl.section_vector():
        pl.load(src, x, [0, 0])
        pl.system.sync_src(
            set_pipe=pl.PipeType.MTE2, wait_pipe=pl.PipeType.V, event_id=8
        )
        _double(dst, src)
        pl.store(out, dst, [0, 0])
