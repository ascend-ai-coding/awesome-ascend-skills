# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

"""写法：原始。十个「改排布或改位宽」的算子。

按**改什么**分三组——这是这一例的组织方式，因为十个算子里有六个都会「改 dtype」，
但改的是三件不同的事：

| 组 | 算子 | 改什么 |
| --- | --- | --- |
| 一、lane 重排 | `interleave` `de_interleave` `squeeze` `unsqueeze` | 元素**位置**，dtype 不变 |
| 二、位宽重组 | `pack` `unpack` | 同一批**位**重新分组，元素个数变 |
| 三、数值/视图转换 | `astype` `bit_cast` `truncate` `muls_cast` | 元素**值**或它的解释方式 |

三组里最容易混的是「`astype` / `bit_cast` / `pack`」——都改 dtype：

- `astype`：**真转换**（`vcvt`），1.5 → 1（按 round_mode）
- `bit_cast`：**只改视图**，位一个都不动
- `pack`：**截位并重新分组**，两个窄元素挤进一个宽槽位的空间

## 与 `vf_align` 的分工

`interleave` / `de_interleave` 在这里是**寄存器到寄存器**；`vf_align` 里的
`LoadDist.DINTLV_B32` / `StoreDist.INTLV_B32` 是**Tile 与寄存器之间**。同一个词，两个层次。

第一手来源：官方 `reg_computation/` 下各算子页；`astype` ST 调用 122 次、`de_interleave` 35 次。
"""

import pypto_pro.language as pl
from pypto_pro.language import Vf as vf  # noqa: N813

FP32 = pl.DT_FP32
FP16 = pl.DT_FP16
I32 = pl.DT_INT32

VL_FP32 = 64  # 一个 FP32 寄存器 = 256 字节 = 64 个元素
N = 64
HALF = N // 2

_A = [0x0000 + 0x400 * i for i in range(2)]
_O = [0x1000 + 0x400 * i for i in range(6)]


def _tile_type(dtype=FP32):
    return pl.TileType(shape=[1, N], dtype=dtype, target_memory=pl.MemorySpace.Vec)


# ---------------------------------------------------------------------------
# 一、lane 重排：位置变，dtype 不变
# ---------------------------------------------------------------------------


@pl.vector_function
def _shuffle(src_a, src_b, o_i0, o_i1, o_d0, o_d1):
    """`interleave` 与 `de_interleave` 都**返回二元组**。

    实测（2026-09-27）：
    - `interleave(a, b)` → `dst0 = [a0, b0, a1, b1, ...]`（前 VL 个输出），
      `dst1 = [a32, b32, a33, b33, ...]`（后 VL 个）。交织后总长 2*VL，拆成两个寄存器。
    - `de_interleave(a, b)` → `dst0` 取偶位、`dst1` 取奇位；把 (a, b) 当成一条 2*VL 的流。
    """
    preg = vf.create_mask(pattern=pl.MaskPattern.ALL, dtype=FP32)
    a = vf.load_align(src_a, 0)
    b = vf.load_align(src_b, 0)

    i0, i1 = vf.interleave(a, b)
    vf.store_align(o_i0, i0, preg, 0)
    vf.store_align(o_i1, i1, preg, 0)

    d0, d1 = vf.de_interleave(a, b)
    vf.store_align(o_d0, d0, preg, 0)
    vf.store_align(o_d1, d1, preg, 0)


@pl.jit(auto_mutex=True)
def shuffle(
    a: pl.Tensor[[1, N], FP32],
    b: pl.Tensor[[1, N], FP32],
    o_i0: pl.Tensor[[1, N], FP32],
    o_i1: pl.Tensor[[1, N], FP32],
    o_d0: pl.Tensor[[1, N], FP32],
    o_d1: pl.Tensor[[1, N], FP32],
):
    ga = pl.make_tile_group(type=_tile_type(), addrs=_A[0], mutex_ids=[0])
    gb = pl.make_tile_group(type=_tile_type(), addrs=_A[1], mutex_ids=[1])
    g0 = pl.make_tile_group(type=_tile_type(), addrs=_O[0], mutex_ids=[2])
    g1 = pl.make_tile_group(type=_tile_type(), addrs=_O[1], mutex_ids=[3])
    g2 = pl.make_tile_group(type=_tile_type(), addrs=_O[2], mutex_ids=[4])
    g3 = pl.make_tile_group(type=_tile_type(), addrs=_O[3], mutex_ids=[5])
    with pl.section_vector():
        ta, tb = ga.current(), gb.current()
        t0, t1, t2, t3 = g0.current(), g1.current(), g2.current(), g3.current()
        pl.load(ta, a, [0, 0])
        pl.load(tb, b, [0, 0])
        _shuffle(ta, tb, t0, t1, t2, t3)
        pl.store(o_i0, t0, [0, 0])
        pl.store(o_i1, t1, [0, 0])
        pl.store(o_d0, t2, [0, 0])
        pl.store(o_d1, t3, [0, 0])


@pl.vector_function
def _compact(src, o_squeezed, o_indices):
    """`squeeze` 与 `unsqueeze` 是一对：一个搬数据，一个给下标。

    - `squeeze(src, preg)`：把 preg 选中的元素**依次搬到低端**，其余置零。
    - `unsqueeze(preg)`：给出**掩码的前缀和**——每个 lane 的值 = 它前面有多少个有效 lane，
      也就是「这个 lane 的元素 squeeze 之后会落到哪」。

    **`unsqueeze` 的 dst 只能是整数类型**（`INT8/UINT8/INT16/UINT16/INT32/UINT32`）,
    给 FP32 会被 codegen 拒（`backend_cce_vf_ops.cpp:5807`）。所以这里整条链走 I32:
    先 `bit_cast` 成 I32，再按 I32 比较、I32 掩码。
    """
    preg = vf.create_mask(pattern=pl.MaskPattern.ALL, dtype=FP32)
    a = vf.load_align(src, 0)
    cond = vf.gt(a, pl.const(32.0, FP32), preg)
    packed = vf.squeeze(a, cond, gather_mode=pl.SqueezeMode.NO_STORE_REG)
    vf.store_align(o_squeezed, packed, preg, 0)

    preg_i = vf.create_mask(pattern=pl.MaskPattern.ALL, dtype=I32)
    a_bits = vf.bit_cast(a, dtype=I32)
    cond_i = vf.gt(a_bits, pl.const(1107296256, I32), preg_i)  # 32.0f 的位型
    indices = vf.unsqueeze(cond_i)
    vf.store_align(o_indices, indices, preg_i, 0)


@pl.jit(auto_mutex=True)
def compact(
    a: pl.Tensor[[1, N], FP32],
    o_squeezed: pl.Tensor[[1, N], FP32],
    o_indices: pl.Tensor[[1, N], I32],
):
    ga = pl.make_tile_group(type=_tile_type(), addrs=_A[0], mutex_ids=[0])
    g0 = pl.make_tile_group(type=_tile_type(), addrs=_O[0], mutex_ids=[1])
    g1 = pl.make_tile_group(type=_tile_type(I32), addrs=_O[1], mutex_ids=[2])
    with pl.section_vector():
        ta, t0, t1 = ga.current(), g0.current(), g1.current()
        pl.load(ta, a, [0, 0])
        _compact(ta, t0, t1)
        pl.store(o_squeezed, t0, [0, 0])
        pl.store(o_indices, t1, [0, 0])


# ---------------------------------------------------------------------------
# 二、位宽重组：pack / unpack，`part` 的含义两边相反
# ---------------------------------------------------------------------------


@pl.vector_function
def _regroup(src, o_pack_lo, o_pack_hi, o_unpack_lo, o_unpack_hi):
    """**`part` 在两个算子里指的不是同一头**——这是这一对最容易搞错的地方。

    | 算子 | `part` 选的是 | 支持的 dtype 对 |
    | --- | --- | --- |
    | `pack` | **dst 的哪一半**（源全用） | `UINT8←INT16/UINT16`、`UINT16←INT32/UINT32`、`UINT32←INT64/UINT64` |
    | `unpack` | **src 的哪一半**（目的全写） | INT16←INT8、UINT16←UINT8、INT32←INT16、UINT32←UINT16、INT64←INT32、UINT64←UINT32 |

    另一个不对称：**`pack` 的 dst 一律无符号**（给 `DT_INT16` 会被拒），
    **`unpack` 保持符号性**（`INT64←INT32` 可以，`UINT64←INT32` 被拒）。
    两张表都是 codegen 的报错原文给出来的（`backend_cce_vf_ops.cpp:3715` / `:3783`）。

    存出去之前 `bit_cast` 回 I32——窄/宽类型的 Tile 形状不一样，这一例不想把话题扯到那上面。
    """
    preg = vf.create_mask(pattern=pl.MaskPattern.ALL, dtype=I32)
    a = vf.load_align(src, 0)

    p_lo = vf.pack(a, part=pl.PackPart.LOWER, dtype=pl.DT_UINT16)
    view = vf.bit_cast(p_lo, dtype=I32)
    vf.store_align(o_pack_lo, view, preg, 0)

    p_hi = vf.pack(a, part=pl.PackPart.UPPER, dtype=pl.DT_UINT16)
    view = vf.bit_cast(p_hi, dtype=I32)
    vf.store_align(o_pack_hi, view, preg, 0)

    u_lo = vf.unpack(a, part=pl.PackPart.LOWER, dtype=pl.DT_INT64)
    view = vf.bit_cast(u_lo, dtype=I32)
    vf.store_align(o_unpack_lo, view, preg, 0)

    u_hi = vf.unpack(a, part=pl.PackPart.UPPER, dtype=pl.DT_INT64)
    view = vf.bit_cast(u_hi, dtype=I32)
    vf.store_align(o_unpack_hi, view, preg, 0)


@pl.jit(auto_mutex=True)
def regroup(
    a: pl.Tensor[[1, N], I32],
    o_pack_lo: pl.Tensor[[1, N], I32],
    o_pack_hi: pl.Tensor[[1, N], I32],
    o_unpack_lo: pl.Tensor[[1, N], I32],
    o_unpack_hi: pl.Tensor[[1, N], I32],
):
    ga = pl.make_tile_group(type=_tile_type(I32), addrs=_A[0], mutex_ids=[0])
    g0 = pl.make_tile_group(type=_tile_type(I32), addrs=_O[0], mutex_ids=[1])
    g1 = pl.make_tile_group(type=_tile_type(I32), addrs=_O[1], mutex_ids=[2])
    g2 = pl.make_tile_group(type=_tile_type(I32), addrs=_O[2], mutex_ids=[3])
    g3 = pl.make_tile_group(type=_tile_type(I32), addrs=_O[3], mutex_ids=[4])
    with pl.section_vector():
        ta = ga.current()
        t0, t1, t2, t3 = g0.current(), g1.current(), g2.current(), g3.current()
        pl.load(ta, a, [0, 0])
        _regroup(ta, t0, t1, t2, t3)
        pl.store(o_pack_lo, t0, [0, 0])
        pl.store(o_pack_hi, t1, [0, 0])
        pl.store(o_unpack_lo, t2, [0, 0])
        pl.store(o_unpack_hi, t3, [0, 0])


# ---------------------------------------------------------------------------
# 三、数值 / 视图转换：三件不同的事，都「改 dtype」
# ---------------------------------------------------------------------------


@pl.vector_function
def _convert(src, o_astype, o_bitcast, o_trunc, o_mulscast, o_astype_z):
    """四个算子，两两对比才说得清：

    - `astype(src, preg, dtype=I32)`：**真转换**，`1.5 → 2`（默认 round_mode，见下）
    - `bit_cast(src, dtype=I32)`：**只改视图**，`1.5` 的位型 `0x3FC00000` 原样读成整数
    - `truncate(src, preg)`：向零取整，**结果还是 FP32**（`1.5 → 1.0`，不是整数类型）
    - `muls_cast(src, scalar, preg, dtype)`：先乘标量再转，一条指令

    `astype` 和 `bit_cast` 的输出放在一起看最能说明问题：同一个输入、同一个目标 dtype，
    一个给几，一个给几亿。

    **`astype` 的默认取整是 `VFRoundMode.CAST_RINT`——就近、平局取偶，不是向零取整。**
    实测 `1.5 2.5 3.5 4.5 → 2 2 4 4`。这里同时存一份 `round_mode=CAST_TRUNC` 的结果
    （`o_astype_z`），两份放一起才看得出「默认不是截断」：截断那份是 `1 2 3 4`。
    七种模式：`CAST_RINT`（默认）/ `CAST_ROUND` / `CAST_FLOOR` / `CAST_CEIL` /
    `CAST_TRUNC` / `CAST_ODD` / `CAST_HYBRID`。

    **`muls_cast` 的 `dtype` 只有一个合法值：`DT_FP16`。** 签名里它是个自由参数，实际
    约束表只有一行 `FP32 × FP32 → FP16`；传 `I32` 在 codegen 才报
    `muls_cast destination only supports FP16`。所以这里转完再 `astype` 回 FP32 落盘，
    输出张量才是 FP32——这个来回是官方样例的形状，不是本例的绕路。
    """
    preg = vf.create_mask(pattern=pl.MaskPattern.ALL, dtype=FP32)
    a = vf.load_align(src, 0)

    converted = vf.astype(a, preg, dtype=I32)
    vf.store_align(o_astype, converted, preg, 0)

    toward_zero = vf.astype(a, preg, dtype=I32, round_mode=pl.VFRoundMode.CAST_TRUNC)
    vf.store_align(o_astype_z, toward_zero, preg, 0)

    viewed = vf.bit_cast(a, dtype=I32)
    vf.store_align(o_bitcast, viewed, preg, 0)

    truncated = vf.truncate(a, preg)
    vf.store_align(o_trunc, truncated, preg, 0)

    scaled = vf.muls_cast(a, pl.const(4.0, FP32), preg, dtype=FP16)
    scaled_back = vf.astype(scaled, preg, dtype=FP32)
    vf.store_align(o_mulscast, scaled_back, preg, 0)


@pl.jit(auto_mutex=True)
def convert(
    a: pl.Tensor[[1, N], FP32],
    o_astype: pl.Tensor[[1, N], I32],
    o_bitcast: pl.Tensor[[1, N], I32],
    o_trunc: pl.Tensor[[1, N], FP32],
    o_mulscast: pl.Tensor[[1, N], FP32],
    o_astype_z: pl.Tensor[[1, N], I32],
):
    ga = pl.make_tile_group(type=_tile_type(), addrs=_A[0], mutex_ids=[0])
    g0 = pl.make_tile_group(type=_tile_type(I32), addrs=_O[0], mutex_ids=[1])
    g1 = pl.make_tile_group(type=_tile_type(I32), addrs=_O[1], mutex_ids=[2])
    g2 = pl.make_tile_group(type=_tile_type(), addrs=_O[2], mutex_ids=[3])
    g3 = pl.make_tile_group(type=_tile_type(), addrs=_O[3], mutex_ids=[4])
    g4 = pl.make_tile_group(type=_tile_type(I32), addrs=_O[4], mutex_ids=[5])
    with pl.section_vector():
        ta = ga.current()
        t0, t1, t2, t3 = g0.current(), g1.current(), g2.current(), g3.current()
        t4 = g4.current()
        pl.load(ta, a, [0, 0])
        _convert(ta, t0, t1, t2, t3, t4)
        pl.store(o_astype, t0, [0, 0])
        pl.store(o_bitcast, t1, [0, 0])
        pl.store(o_trunc, t2, [0, 0])
        pl.store(o_mulscast, t3, [0, 0])
        pl.store(o_astype_z, t4, [0, 0])
