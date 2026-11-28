# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

"""写法：原始。饱和标志的读回与恢复，以及「谁说了算」。

两个 API，**全无 ST**（`pl.set_saturation_flag` / `get_saturation_flag` 在 `st/` 和 `ut/` 里
各 0 次）。所以这一例的每个数字都是 2026-09-28 在 Ascend950PR 上量的。

## 一、四个模式各管一个 CTRL 位

| `SaturationFlagMode` | CTRL 位 | 管什么 |
| --- | ---: | --- |
| `FLOAT` | 48 | 浮点计算与浮点精度转换 |
| `FLOAT8` | 50 | float8 的 NaN 处理 |
| `INT` | 53 | 整数指令 |
| `CAST` | 59 | 浮点转整数 / 整数转整数 |

## 二、**极性对三个模式是反的**

`backend_cce_ops.cpp` 的 `IsInvertedPolarity`：

> FLOAT/FLOAT8/CAST: polarity inverted (bit=0 means saturation ON, bit=1 means OFF)
> INT: polarity normal (bit=1 means saturation ON, bit=0 means OFF)

所以 `set_saturation_flag(mode=CAST, enable=True)` 之后 `get_ctrl_spr(59, 59)` 读到的是 **0**。
**两个接口混用时这一点会骗人**：`enable=True` 但那个位是 0，看着像没生效。
`get_saturation_flag` 自己是对称的（它读的时候也反回来），所以单用它不会错。

## 三、CTRL[60] 决定听谁的

官方 `SaturationFlagMode.md` 的约束说明：

> 当 CTRL[60]=0 时为单指令模式（由 saturate 参数控制），当 CTRL[60]=1 时为全局模式
> （由本枚举控制的 CTRL 位生效）。

而 CTRL[60] 的入口值是 **1**（实测，见 ``spr_ctrl``）。**所以默认是全局模式,
`vf.astype(saturate=...)` 默认不起作用。** 六种配置的实测表在 `test.py` §四。

第一手来源：官方 `special_reg_access/{set,get}_saturation_flag.md`、
`basic_data_structures/SaturationFlagMode.md`（位映射与 CTRL[60] 约束）、
`reg_computation/type_conversion/astype.md`（`saturate=` 参数），以及
`backend_cce_ops.cpp` 的 `GetSaturationModeBit` / `IsInvertedPolarity`。**没有 ST 可照。**
"""

import pypto_pro.language as pl
from pypto_pro.language import Vf as vf  # noqa: N813

FP32 = pl.DT_FP32
I16 = pl.DT_INT16
I64 = pl.DT_INT64

N = 64
_A = 0x0
_B = 0x400

INT16_MIN = -32768
INT16_MAX = 32767

MODES = [
    # (名字, 枚举, CTRL 位, 极性是否反转, 入口时饱和是否开着)
    ("FLOAT", pl.SaturationFlagMode.FLOAT, 48, True, False),
    ("FLOAT8", pl.SaturationFlagMode.FLOAT8, 50, True, True),
    ("INT", pl.SaturationFlagMode.INT, 53, False, False),
    ("CAST", pl.SaturationFlagMode.CAST, 59, True, True),
]
"""**最后两列是两件不同的事，别并成一列**（写这一例时并过，判据当场红了）：

- `inverted` 抄的是 C++ 的 `IsInvertedPolarity`——FLOAT / FLOAT8 / CAST 反，INT 正。
  **判据会把它和实测的位值对起来**，抄错了会红。
- `entry_saturating` 是实测的入口状态。它由「入口位值 + 极性」决定，不独立:
  FLOAT 入口关是因为 bit 48 入口为 1（见 ``spr_ctrl``），INT 入口关是因为 bit 53 为 0
  而它极性正常。**两个「关」的来路完全不同**，所以这两列必须分开。
"""

GLOBAL_BIT = 60


def _tile_type(dtype=FP32, n=N):
    return pl.TileType(shape=[1, n], dtype=dtype, target_memory=pl.MemorySpace.Vec)


# ---------------------------------------------------------------------------
# 一、标志的读回与极性
# ---------------------------------------------------------------------------


def make_polarity_probe(mode, bit):
    """一个模式一个 kernel，输出 6 个量：默认 flag/bit、`enable=True` 的 flag/bit、`False` 的。

    **工厂而不是一个 24 槽的大 kernel**：四个模式各自一行输出，对不上时看得出是哪个模式。
    `mode` 和 `bit` 是闭包变量，解析器支持（`ut/.../test_closure_var_resolution.py`）。
    """

    @pl.jit()
    def probe(out: pl.Tensor[[6], I64]):
        default_flag = pl.astype(pl.get_saturation_flag(mode=mode), I64)
        default_bit = pl.get_ctrl_spr(bit, bit)

        pl.set_saturation_flag(mode=mode, enable=True)
        on_flag = pl.astype(pl.get_saturation_flag(mode=mode), I64)
        on_bit = pl.get_ctrl_spr(bit, bit)

        pl.set_saturation_flag(mode=mode, enable=False)
        off_flag = pl.astype(pl.get_saturation_flag(mode=mode), I64)
        off_bit = pl.get_ctrl_spr(bit, bit)

        pl.reset_ctrl_spr(bit, bit)
        with pl.section_vector():
            pl.setval(out, 0, default_flag)
            pl.setval(out, 1, default_bit)
            pl.setval(out, 2, on_flag)
            pl.setval(out, 3, on_bit)
            pl.setval(out, 4, off_flag)
            pl.setval(out, 5, off_bit)

    return probe


# ---------------------------------------------------------------------------
# 二、标志改不改数值结果
# ---------------------------------------------------------------------------


def _make_narrowing_vf(saturate):
    """FP32 → INT16 → FP32。`saturate=` 在 `def` 时固定。

    转回 FP32 再存是为了让参考能用 numpy 算：INT16 的值在 FP32 里是精确的,
    所以读回来的浮点数唯一决定了中间那个 INT16。
    """

    @pl.vector_function
    def narrow(src, dst):
        preg = vf.create_mask(pattern=pl.MaskPattern.ALL, dtype=FP32)
        a = vf.load_align(src, 0)
        narrowed = vf.astype(
            a, preg, dtype=I16, layout=pl.CastLayout.ZERO, saturate=saturate
        )
        back = vf.astype(narrowed, preg, dtype=FP32)
        vf.store_align(dst, back, preg, 0)

    return narrow


def make_narrowing_kernel(*, global_bit, cast_enable, saturate):
    """三个自变量：CTRL[60]、`set_saturation_flag(CAST, enable=)`、`astype(saturate=)`。

    **三个都显式写出来**，一个都不靠默认值：默认值是「全局模式」这件事本身就是要验的,
    靠它会让判据和被验对象缠在一起。
    """
    body = _make_narrowing_vf(saturate)

    @pl.jit(auto_mutex=True)
    def kernel(a: pl.Tensor[[1, N], FP32], out: pl.Tensor[[1, N], FP32]):
        pl.set_ctrl_spr(GLOBAL_BIT, GLOBAL_BIT, global_bit)
        pl.set_saturation_flag(mode=pl.SaturationFlagMode.CAST, enable=cast_enable)
        ga = pl.make_tile_group(type=_tile_type(), addrs=_A, mutex_ids=[0])
        go = pl.make_tile_group(type=_tile_type(), addrs=_B, mutex_ids=[1])
        with pl.section_vector():
            ta, to = ga.current(), go.current()
            pl.load(ta, a, [0, 0])
            body(ta, to)
            pl.store(out, to, [0, 0])
        # 收尾复位。**注意这不是「恢复原状」**：`reset` 写的是硬件默认值，对 bit 48
        # 那和入口值不同（见 ``spr_ctrl`` §三）。这里 reset 的是 59 / 60，两者入口值
        # 与默认值一致，所以是安全的。
        pl.reset_ctrl_spr(59, 59)
        pl.reset_ctrl_spr(GLOBAL_BIT, GLOBAL_BIT)

    return kernel


# `(CTRL[60], enable, saturate, 判别性)`。**后三行是判别行**：它们把「全局说了算」和
# 「单指令说了算」分开。前三行两种模型都能解释，单独看它们证不到任何东西。
CONFIGS = [
    (1, True, pl.SaturateMode.OFF, False),
    (1, False, pl.SaturateMode.OFF, False),
    (1, True, pl.SaturateMode.ON, False),
    (1, False, pl.SaturateMode.ON, True),
    (0, False, pl.SaturateMode.ON, True),
    (0, True, pl.SaturateMode.OFF, True),
]


# ---------------------------------------------------------------------------
# 三、被拒的写法
# ---------------------------------------------------------------------------


@pl.vector_function
def _set_flag_in_vf(src, dst):
    """反例：`set_saturation_flag` 进 vf 体。docstring 说必须在 `@pl.jit` 体里调。"""
    preg = vf.create_mask(pattern=pl.MaskPattern.ALL, dtype=FP32)
    a = vf.load_align(src, 0)
    pl.set_saturation_flag(mode=pl.SaturationFlagMode.CAST, enable=True)
    vf.store_align(dst, a, preg, 0)


@pl.jit(auto_mutex=True)
def set_flag_inside_a_vector_function(
    a: pl.Tensor[[1, N], FP32], out: pl.Tensor[[1, N], FP32]
):
    """反例：在 `@pl.vector_function` 里设饱和标志。

    收这条是因为它是很自然的一步：饱和影响的是 vf 指令，**所以会想在 vf 里设**。
    但标志是标量流水上的全局状态，设它的地方在 kernel 体。
    """
    ga = pl.make_tile_group(type=_tile_type(), addrs=_A, mutex_ids=[0])
    go = pl.make_tile_group(type=_tile_type(), addrs=_B, mutex_ids=[1])
    with pl.section_vector():
        ta, to = ga.current(), go.current()
        pl.load(ta, a, [0, 0])
        _set_flag_in_vf(ta, to)
        pl.store(out, to, [0, 0])


@pl.jit()
def set_flag_without_keywords(out: pl.Tensor[[1], I64]):
    """反例：`set_saturation_flag` 的两个参数都是 keyword-only（`*` 在签名里）。"""
    pl.set_saturation_flag(pl.SaturationFlagMode.CAST, True)
    with pl.section_vector():
        pl.setval(out, 0, pl.get_ctrl_spr(59, 59))
