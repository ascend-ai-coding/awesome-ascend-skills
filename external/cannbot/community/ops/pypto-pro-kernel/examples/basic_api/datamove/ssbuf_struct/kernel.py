# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

"""写法：原始。SSBUF：把结构化元数据从 vector 侧交给 cube 侧。

**SSBUF 是 vector 与 cube 之间传"小而有结构"的东西的通道。** 不是搬数据用的——数据走
GM↔UB↔L1；SSBUF 传的是形状、偏移、块号这类标量元数据。

一次交接固定是四步，缺一步就不成立：

```python
with pl.section_vector():
    msg.batch = 8                                  # 1. 在 vector 侧填字段
    if pl.get_subblock_idx() == 0:                 # 2. 只让一个 subblock 写
        pl.ssbuf_store(msg, 0)
        pl.system.set_cross_core(pipe=pl.PipeType.S, event_id=15)   # 3. 发跨核事件
with pl.section_cube():
    pl.system.wait_cross_core(pipe=pl.PipeType.S, event_id=15,      # 4. cube 侧等
                              sync_mode=pl.CrossCoreSyncMode.UNICAST_BLOCK)
    pl.ssbuf_load(msg, 0)
```

**`offset` 的单位没有文档，实测是字节；而每个字段在 SSBUF 里占 8 字节**，不是它的 dtype 宽度。
所以 N 个 INT32 字段的 struct 占 `8 * N` 字节，第二个 struct 的偏移必须 `>= 8 * N`。
官方 docstring 只写「offset: SSBUF offset」，ST 里也只用过 `0`——这条是量出来的，
完整证据链见 `test.py` §四。

**`pl.struct` 必须带名字。** docstring 里写的「Legacy: `pl.struct(field=val, ...)` — anonymous
struct」这一支解析期会被拒（见 `test.py` §六）。

第一手来源：`pypto/python/tests/st/pypto_pro/frontend/system/test_ssbuf_copy.py`
（整个交接形状照它抄），以及 `system/struct/`、`system/struct_array/` 两组 ST。
"""

import pypto_pro.language as pl

I32 = pl.DT_INT32

# 三个字段，值刻意取成互不相同且都不是 0：串了字段一眼能看出串到了哪个。
BATCH = 8
BLOCK = 1
OFFSET = 32768

# 每个字段在 SSBUF 里占 8 字节（实测）。三个字段 = 24 字节，所以第二个 struct 至少放 24。
SSBUF_BYTES_PER_FIELD = 8
FIELDS = 3
SPACED = SSBUF_BYTES_PER_FIELD * FIELDS  # 24：刚好接在第一个后面
TOO_TIGHT = 16  # 落在第一个的第 3 个槽上

# 跨核事件号。flag 的池子是 [0, 7]，跨核的是 [0, 15]——**这是两个池子**，照 ST 用 15。
CROSS_CORE_EVENT = 15

SECOND = (11, 12, 13)


@pl.jit()
def handoff_one_struct(out: pl.Tensor[[3], I32]):
    """最小交接：vector 填三个字段，cube 读出来写进 GM。**整个形状照 ST 抄。**

    `pl.get_subblock_idx() == 0` 那层判断不是可选的：一个 vector block 有多个 subblock，
    都去写同一个 SSBUF 偏移就是多个写者抢同一块地址。
    """
    message = pl.struct("Message", batch=0, block=0, offset=0)

    with pl.section_vector():
        message.batch = BATCH
        message.block = BLOCK
        message.offset = OFFSET
        if pl.get_subblock_idx() == 0:
            pl.ssbuf_store(message, 0)
            pl.system.set_cross_core(pipe=pl.PipeType.S, event_id=CROSS_CORE_EVENT)

    with pl.section_cube():
        pl.system.wait_cross_core(
            pipe=pl.PipeType.S,
            event_id=CROSS_CORE_EVENT,
            sync_mode=pl.CrossCoreSyncMode.UNICAST_BLOCK,
        )
        pl.ssbuf_load(message, 0)
        out[0] = message.batch
        out[1] = message.block
        out[2] = message.offset


@pl.jit()
def two_structs_spaced_correctly(out: pl.Tensor[[6], I32]):
    """两个三字段 struct，偏移 `0` 和 `24`。**24 = 3 个字段 × 8 字节**，刚好不重叠。"""
    first = pl.struct("SpacedFirst", x=0, y=0, z=0)
    second = pl.struct("SpacedSecond", x=0, y=0, z=0)

    with pl.section_vector():
        first.x = BATCH
        first.y = BLOCK
        first.z = OFFSET
        second.x = SECOND[0]
        second.y = SECOND[1]
        second.z = SECOND[2]
        if pl.get_subblock_idx() == 0:
            pl.ssbuf_store(first, 0)
            pl.ssbuf_store(second, SPACED)
            pl.system.set_cross_core(pipe=pl.PipeType.S, event_id=CROSS_CORE_EVENT)

    with pl.section_cube():
        pl.system.wait_cross_core(
            pipe=pl.PipeType.S,
            event_id=CROSS_CORE_EVENT,
            sync_mode=pl.CrossCoreSyncMode.UNICAST_BLOCK,
        )
        pl.ssbuf_load(first, 0)
        pl.ssbuf_load(second, SPACED)
        out[0] = first.x
        out[1] = first.y
        out[2] = first.z
        out[3] = second.x
        out[4] = second.y
        out[5] = second.z


@pl.jit()
def two_structs_spaced_too_tightly(out: pl.Tensor[[6], I32]):
    """偏移 `0` 和 **16**：第二个 struct 的第一个字段落在第一个 struct 的**第 3 个槽**上。

    按「每字段 8 字节」算，第一个 struct 占槽 `[0, 8, 16]`，所以偏移 16 正好压在它的 `z` 上。
    **预期结果是 `first.z` 读出 `second.x`**，而不是报错——这条是本例的核心判据，
    它把槽宽 8 字节这件事钉住。

    这个"损坏"是确定的：同一进程连跑三轮一致，而且换成 4 字段时模型给出的预测
    （`off=16` → 第 3、4 个字段被压）也逐位命中。**不是偶发**，所以可以进判据。
    """
    first = pl.struct("TightFirst", x=0, y=0, z=0)
    second = pl.struct("TightSecond", x=0, y=0, z=0)

    with pl.section_vector():
        first.x = BATCH
        first.y = BLOCK
        first.z = OFFSET
        second.x = SECOND[0]
        second.y = SECOND[1]
        second.z = SECOND[2]
        if pl.get_subblock_idx() == 0:
            pl.ssbuf_store(first, 0)
            pl.ssbuf_store(second, TOO_TIGHT)
            pl.system.set_cross_core(pipe=pl.PipeType.S, event_id=CROSS_CORE_EVENT)

    with pl.section_cube():
        pl.system.wait_cross_core(
            pipe=pl.PipeType.S,
            event_id=CROSS_CORE_EVENT,
            sync_mode=pl.CrossCoreSyncMode.UNICAST_BLOCK,
        )
        pl.ssbuf_load(first, 0)
        pl.ssbuf_load(second, TOO_TIGHT)
        out[0] = first.x
        out[1] = first.y
        out[2] = first.z
        out[3] = second.x
        out[4] = second.y
        out[5] = second.z


@pl.jit()
def struct_array_two_elements(out: pl.Tensor[[6], I32]):
    """`pl.struct_array(2, "Name", ...)`：两个同构 struct，**偏移规则和两个独立 struct 一样**。

    数组不会帮你算偏移——`ssbuf_store(m[1], SPACED)` 里那个 24 还是得自己写。
    留这一条是为了说明「用数组」不等于「不用管布局」。
    """
    messages = pl.struct_array(2, "ArrayMessage", x=0, y=0, z=0)

    with pl.section_vector():
        messages[0].x = BATCH
        messages[0].y = BLOCK
        messages[0].z = OFFSET
        messages[1].x = SECOND[0]
        messages[1].y = SECOND[1]
        messages[1].z = SECOND[2]
        if pl.get_subblock_idx() == 0:
            pl.ssbuf_store(messages[0], 0)
            pl.ssbuf_store(messages[1], SPACED)
            pl.system.set_cross_core(pipe=pl.PipeType.S, event_id=CROSS_CORE_EVENT)

    with pl.section_cube():
        pl.system.wait_cross_core(
            pipe=pl.PipeType.S,
            event_id=CROSS_CORE_EVENT,
            sync_mode=pl.CrossCoreSyncMode.UNICAST_BLOCK,
        )
        pl.ssbuf_load(messages[0], 0)
        pl.ssbuf_load(messages[1], SPACED)
        out[0] = messages[0].x
        out[1] = messages[0].y
        out[2] = messages[0].z
        out[3] = messages[1].x
        out[4] = messages[1].y
        out[5] = messages[1].z


# ---------------------------------------------------------------------------
# 只解析、不上板：三条被拒的写法
# ---------------------------------------------------------------------------


@pl.jit()
def anonymous_struct(out: pl.Tensor[[3], I32]):
    """`pl.struct(field=...)` 不带名字。**被拒**，尽管 docstring 把它列为 "Legacy" 形式。"""
    message = pl.struct(batch=0, block=0, offset=0)
    with pl.section_vector():
        message.batch = BATCH
        pl.ssbuf_store(message, 0)
    with pl.section_cube():
        pl.ssbuf_load(message, 0)
        out[0] = message.batch


@pl.jit()
def assign_a_make_tuple_field(out: pl.Tensor[[3], I32]):
    """`pl.make_tuple` 的字段**不能赋值**：它聚合已有的 IR 变量，不是可写的 struct。

    这是 `pl.struct` 与 `pl.make_tuple` 的实际分界：前者会生成一个 C++ struct 并且字段可写,
    后者只是给几个已有值起名字（docstring：「without forcing a C++ struct to be generated」）。
    **所以 SSBUF 那条路只能用 `pl.struct`。**
    """
    message = pl.make_tuple(batch=0, block=0, offset=0)
    with pl.section_vector():
        message.batch = BATCH
        pl.ssbuf_store(message, 0)
    with pl.section_cube():
        out[0] = message.batch


@pl.jit()
def unpack_two_fields_at_once(out: pl.Tensor[[3], I32]):
    """`msg.a, msg.b = 1, 2` 这种元组解包**被拒**：解包的目标必须是变量名，不能是字段。

    写这一例时我自己撞了这条，两个探针 kernel 一起挂——而挂出来的消息指向解包，
    很容易误读成"struct 的问题"。
    """
    message = pl.struct("Unpack", batch=0, block=0, offset=0)
    with pl.section_vector():
        message.batch, message.block = BATCH, BLOCK
        pl.ssbuf_store(message, 0)
    with pl.section_cube():
        pl.ssbuf_load(message, 0)
        out[0] = message.batch
