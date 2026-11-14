# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

"""写法：原始。CTRL 特殊寄存器的位域读改复位，以及 `pl.get_spr()`（AR）。

四个 API，**全无 ST**（`pl.get_spr` / `set_ctrl_spr` / `get_ctrl_spr` / `reset_ctrl_spr` 在
`st/` 和 `ut/` 里各 0 次，按 `pl.<name>(` 和裸名两种写法都搜过）。所以这一例的每个数字都是
2026-09-28 在 Ascend950PR 上量出来的。

## CTRL 是什么

一个 64 位标量寄存器，控制矢量计算的若干全局模式。**可写位只有 7 处**：

| 位 | 管什么 | 0 | 1 | 官方说的默认 |
| --- | --- | --- | --- | --- |
| 6-10 | 原子操作配置 | `[8:6]` dtype、`[10:9]` op | | 0 |
| 45 | matmul 寄存器格式 | 按原子操作数类型 | FP8E4M3FN → HF8 | 0 |
| 48 | 浮点数饱和 | **饱和**（inf→MAX、NaN→0） | 非饱和 | 0 |
| 50 | 浮点 NaN 处理 | NaN→0.0 | NaN 保持 | 0 |
| 53 | 整数饱和 | **截断** | 饱和 | 0 |
| 59 | astype 饱和 | **饱和** | 截断 | 0 |
| 60 | 饱和模式全局配置 | 单指令生效 | **全局生效** | 1 |

读不受这个限制：`get_ctrl_spr(0, 63)` 能把整个寄存器读出来。

## 三件与文档不一致或容易误读的事（都实测过）

**1. kernel 入口的 CTRL 是 `0x1001000000000008`，官方写的默认值是 `0x1000000000000008`。**
差的正是 bit 48：入口值是 1（非饱和），文档说默认 0（饱和）。

**2. `reset_ctrl_spr` 恢复的是「硬件默认值」，不是「kernel 进来时的值」。** 对 bit 48 这两者
不同，所以 `set(48) → 干活 → reset(48)` 这套「用完恢复原状」的写法**会把 bit 48 从 1 改成 0**,
静默改变后面所有浮点转换的饱和行为。C++ 里是写死的常量（`backend_cce_ops.cpp` 的
`defaultCtrl = 0x1000000000000008`），不是读回来的。

**3. CTRL 不跨 launch 残留。** 连启动三次「读整寄存器 → 把 bit 48 清零」的 kernel，
三次读到的入口值都是 `0x1001000000000008`。所以 CTRL 的改动是 kernel 局部的,
不会污染下一个 kernel——这一条是好消息，但**必须量过才能依赖**。

## `pl.get_spr()`：文档说的用法拿不到

官方文档与 docstring 都说 AR 存 `vf.squeeze` 的有效元素总字节数。**实测读不到**——读数与
有效元素个数无关，而且**它不是常量**：同一份 kernel 在只跑这一例的进程里读到 128，跑完整章之后
读到 0。判据只断言「同一进程内读数不随 squeeze 变」，不断言那个值。详见 `test.py` §五。

第一手来源：官方 `special_reg_access/{get_spr,set_ctrl_spr,get_ctrl_spr,reset_ctrl_spr}.md`
（可写位表与 CTRL 默认值来自 `set_ctrl_spr.md` 的约束说明）、`backend_cce_ops.cpp` 的
`CheckCtrlBitRange` / `MakeResetCtrlSprCodegenCCE`。**没有 ST 可照。**
"""

import pypto_pro.language as pl
from pypto_pro.language import Vf as vf  # noqa: N813

FP32 = pl.DT_FP32
I64 = pl.DT_INT64

N = 64
_A = 0x0
_B = 0x400

# 官方 `reset_ctrl_spr.md`：CTRL 寄存器的默认值。C++ 侧同一个常量在
# `backend_cce_ops.cpp` 的 `MakeResetCtrlSprCodegenCCE` 里写死。
DOCUMENTED_DEFAULT = 0x1000000000000008

# 读哪些位区间。**`(0, 63)` 放第一个是控制项**：它对不上的话后面每一行都不可信。
RANGES = [
    (0, 63),
    (60, 60),
    (59, 59),
    (53, 53),
    (50, 50),
    (48, 48),
    (45, 45),
    (6, 10),
    (3, 3),
]

PROBE_BIT = 59
"""拿 59 做 set/reset 往返：它是单比特可写位，而且**默认值和文档一致**（不像 48）,
所以往返的判据不会和「入口值与文档不符」那件事缠在一起。"""

NEIGHBOUR_BIT = 60
"""紧邻 59 的可写位，用来验「reset 只碰指定区间」。选邻位而不是远处的位:
掩码算错一格最先撞到的就是邻位。"""


def _tile_type(dtype=FP32, n=N):
    return pl.TileType(shape=[1, n], dtype=dtype, target_memory=pl.MemorySpace.Vec)


# ---------------------------------------------------------------------------
# 一、读：入口状态
# ---------------------------------------------------------------------------


@pl.jit()
def read_entry_state(out: pl.Tensor[[len(RANGES)], I64]):
    """kernel 一进来就读若干位区间，一行一个。

    **读全部写在 `section_vector` 外面**：`get_ctrl_spr` 是标量指令（`PipeType::S`），
    docstring 要求它在 `@pl.jit` 体里、不在 `@pl.vector_function` 里。写出去要一段 section,
    所以形状是「先读一串标量，再进 section 一次性 setval」。
    """
    v0 = pl.get_ctrl_spr(0, 63)
    v1 = pl.get_ctrl_spr(60, 60)
    v2 = pl.get_ctrl_spr(59, 59)
    v3 = pl.get_ctrl_spr(53, 53)
    v4 = pl.get_ctrl_spr(50, 50)
    v5 = pl.get_ctrl_spr(48, 48)
    v6 = pl.get_ctrl_spr(45, 45)
    v7 = pl.get_ctrl_spr(6, 10)
    v8 = pl.get_ctrl_spr(3, 3)
    with pl.section_vector():
        pl.setval(out, 0, v0)
        pl.setval(out, 1, v1)
        pl.setval(out, 2, v2)
        pl.setval(out, 3, v3)
        pl.setval(out, 4, v4)
        pl.setval(out, 5, v5)
        pl.setval(out, 6, v6)
        pl.setval(out, 7, v7)
        pl.setval(out, 8, v8)


# ---------------------------------------------------------------------------
# 二、写与复位
# ---------------------------------------------------------------------------


@pl.jit()
def set_then_reset(out: pl.Tensor[[4], I64]):
    """59 位：进来读一次 → 写 1 → 读 → reset → 读；第 4 个量读邻位 60。

    四个量分四行存，不打包成一个位图：对不上的时候要能分清是哪一步错了。
    """
    before = pl.get_ctrl_spr(PROBE_BIT, PROBE_BIT)
    pl.set_ctrl_spr(PROBE_BIT, PROBE_BIT, 1)
    after_set = pl.get_ctrl_spr(PROBE_BIT, PROBE_BIT)
    pl.reset_ctrl_spr(PROBE_BIT, PROBE_BIT)
    after_reset = pl.get_ctrl_spr(PROBE_BIT, PROBE_BIT)
    neighbour = pl.get_ctrl_spr(NEIGHBOUR_BIT, NEIGHBOUR_BIT)
    with pl.section_vector():
        pl.setval(out, 0, before)
        pl.setval(out, 1, after_set)
        pl.setval(out, 2, after_reset)
        pl.setval(out, 3, neighbour)


@pl.jit()
def reset_bit48_changes_it(out: pl.Tensor[[2], I64]):
    """**这一例最该记住的形状**：什么都没写，只 `reset_ctrl_spr(48, 48)`，bit 48 就变了。

    入口 1（非饱和）→ reset → 0（饱和）。「用完恢复原状」的直觉在这里是错的:
    `reset` 写的是硬件默认值，而入口值不等于硬件默认值。
    """
    before = pl.get_ctrl_spr(48, 48)
    pl.reset_ctrl_spr(48, 48)
    after = pl.get_ctrl_spr(48, 48)
    with pl.section_vector():
        pl.setval(out, 0, before)
        pl.setval(out, 1, after)


@pl.jit()
def clear_bit48_then_read(out: pl.Tensor[[1], I64]):
    """连启动多次：每次先读 bit 48，再把它清零。**下一次仍读到 1 就说明 CTRL 不跨 launch。**

    判据在 `test.py` 里靠**重复启动同一个 kernel**成立，不靠这个 kernel 本身。
    """
    bit48 = pl.get_ctrl_spr(48, 48)
    pl.set_ctrl_spr(48, 48, 0)
    with pl.section_vector():
        pl.setval(out, 0, bit48)


# ---------------------------------------------------------------------------
# 三、AR：`pl.get_spr()`
# ---------------------------------------------------------------------------


@pl.vector_function
def _squeeze_positive(src, dst):
    """反例：squeeze-alternation —— 单次写 AR 的观测实验，用对齐 store 读压紧值。
    反例：clear-spr-before-squeeze —— 观测入口 SPR 状态，不主动清零改变被测状态。
    """
    preg = vf.create_mask(pattern=pl.MaskPattern.ALL, dtype=FP32)
    a = vf.load_align(src, 0)
    cond = vf.gt(a, pl.const(0.0, FP32), preg)
    packed = vf.squeeze(a, cond, gather_mode=pl.SqueezeMode.STORE_REG)
    vf.store_align(dst, packed, preg, 0)


@pl.jit(auto_mutex=True)
def ar_around_squeeze(
    a: pl.Tensor[[1, N], FP32], out: pl.Tensor[[1, N], FP32], ar: pl.Tensor[[2], I64]
):
    """在 squeeze 所在 section 的**前后**各读一次 AR。

    两个数相同就说明 squeeze 没有改变 `get_spr()` 看到的值。
    """
    ga = pl.make_tile_group(type=_tile_type(), addrs=_A, mutex_ids=[0])
    go = pl.make_tile_group(type=_tile_type(), addrs=_B, mutex_ids=[1])
    before = pl.get_spr()
    with pl.section_vector():
        ta, to = ga.current(), go.current()
        pl.load(ta, a, [0, 0])
        _squeeze_positive(ta, to)
        pl.store(out, to, [0, 0])
    after = pl.get_spr()
    with pl.section_vector():
        pl.setval(ar, 0, before)
        pl.setval(ar, 1, after)


@pl.jit(auto_mutex=True)
def ar_without_any_vf(
    a: pl.Tensor[[1, N], FP32], out: pl.Tensor[[1, N], FP32], ar: pl.Tensor[[2], I64]
):
    """**控制项**：一条直通搬运，一个 vf 指令都不带，照样读 AR。

    这是整组 AR 测量的锚。上面那个 kernel 读到 128 有两种解释（squeeze 写的、还是本来就是）,
    这一个把它们分开：**它也读 128，就说明 squeeze 没参与**。
    """
    ga = pl.make_tile_group(type=_tile_type(), addrs=_A, mutex_ids=[0])
    before = pl.get_spr()
    with pl.section_vector():
        ta = ga.current()
        pl.load(ta, a, [0, 0])
        pl.store(out, ta, [0, 0])
    after = pl.get_spr()
    with pl.section_vector():
        pl.setval(ar, 0, before)
        pl.setval(ar, 1, after)


# ---------------------------------------------------------------------------
# 四、被拒的写法
# ---------------------------------------------------------------------------


@pl.jit()
def write_an_unwritable_bit(out: pl.Tensor[[1], I64]):
    """反例：bit 20 不在可写清单里。**编译期**拒（`backend_cce_ops.cpp:1036`），解析期不拦。"""
    pl.set_ctrl_spr(20, 20, 1)
    with pl.section_vector():
        pl.setval(out, 0, pl.get_ctrl_spr(20, 20))


@pl.jit()
def write_undefined_atomic_dtype(out: pl.Tensor[[1], I64]):
    """反例：`CTRL[8:6] = 3'b111` 是未定义的原子操作数类型（合法 0-6）。

    **这条被拒是好事**：C++ 注释说这个编码会让设备 fault，守卫把它挡在编译期。
    """
    pl.set_ctrl_spr(6, 8, 7)
    with pl.section_vector():
        pl.setval(out, 0, pl.get_ctrl_spr(6, 10))


@pl.jit()
def write_undefined_atomic_op(out: pl.Tensor[[1], I64]):
    """反例：`CTRL[10:9] = 2'b11` 是未定义的原子操作类型（合法 0-2 = ADD/MAX/MIN）。"""
    pl.set_ctrl_spr(9, 10, 3)
    with pl.section_vector():
        pl.setval(out, 0, pl.get_ctrl_spr(6, 10))


@pl.vector_function
def _get_spr_in_vf(src, dst):
    """反例：`get_ar()` 是 `__aicore__` 指令，vf 体里用不了。"""
    preg = vf.create_mask(pattern=pl.MaskPattern.ALL, dtype=FP32)
    a = vf.load_align(src, 0)
    value = pl.get_spr()
    b = vf.adds(a, pl.astype(value, FP32), preg)
    vf.store_align(dst, b, preg, 0)


@pl.jit(auto_mutex=True)
def get_spr_inside_a_vector_function(
    a: pl.Tensor[[1, N], FP32], out: pl.Tensor[[1, N], FP32]
):
    """反例：`pl.get_spr()` 放进 `@pl.vector_function`。解析期拒。

    **这条和上面「AR 读不到 squeeze 的计数」是同一件事的两半**：能写 squeeze 的地方读不了 AR,
    能读 AR 的地方看不到 squeeze 写的值。
    """
    ga = pl.make_tile_group(type=_tile_type(), addrs=_A, mutex_ids=[0])
    go = pl.make_tile_group(type=_tile_type(), addrs=_B, mutex_ids=[1])
    with pl.section_vector():
        ta, to = ga.current(), go.current()
        pl.load(ta, a, [0, 0])
        _get_spr_in_vf(ta, to)
        pl.store(out, to, [0, 0])


@pl.vector_function
def _squeeze_result_discarded(src, dst):
    """反例：`vf.squeeze` 的返回值不接。"""
    preg = vf.create_mask(pattern=pl.MaskPattern.ALL, dtype=FP32)
    a = vf.load_align(src, 0)
    cond = vf.gt(a, pl.const(0.0, FP32), preg)
    vf.squeeze(a, cond, gather_mode=pl.SqueezeMode.NO_STORE_REG)
    vf.store_align(dst, a, preg, 0)


@pl.jit(auto_mutex=True)
def squeeze_without_assignment(
    a: pl.Tensor[[1, N], FP32], out: pl.Tensor[[1, N], FP32]
):
    """反例：写 `vf.squeeze(a, cond)` 当语句。解析期拒，要求赋值形式。

    收在这一例是因为**它正是想「只为副作用（写 AR）调 squeeze」时的第一反应**——
    那条路连语法都不成立。
    """
    ga = pl.make_tile_group(type=_tile_type(), addrs=_A, mutex_ids=[0])
    go = pl.make_tile_group(type=_tile_type(), addrs=_B, mutex_ids=[1])
    with pl.section_vector():
        ta, to = ga.current(), go.current()
        pl.load(ta, a, [0, 0])
        _squeeze_result_discarded(ta, to)
        pl.store(out, to, [0, 0])
