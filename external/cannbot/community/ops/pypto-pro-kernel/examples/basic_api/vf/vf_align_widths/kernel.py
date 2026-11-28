# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

"""写法：原始（``vf.*`` 没有语法糖）。

``vf.load_align`` / ``vf.store_align`` 的 ``dist=`` 里**位宽绑定**的那一半——展开
（UNPK）、块广播（E2B）、打包（PACK 族）。``vf_align`` 讲同一参数在单一位宽
（FP32）下的分布模式；寄存器到寄存器的同名算子（``vf.pack`` / ``vf.interleave``
本体）在 ``vf_lane_layout``；这里的语义随「源 dtype × dist 粒度」组合变化：

- **UNPK（FP16 源）+ ``astype(FP32, ZERO)``**：保序宽化，FP16 源做 FP32 计算的
  标准入口（普通 load 后直接宽化 astype 是偶/奇源子采样，静默丢一半）。
- **E2B**：element-to-block 广播（官方文档「16→32 扩展」的描述是错的——
  FP32/FP16 两个位宽的实测互证，见 test.py）。
- **``astype(FP16, ZERO)`` + PACK_B32 store**：窄化直出稠密（ND 目标下替代
  de_interleave 拼接）。
- **粗名 ``PACK`` 陷阱**：按源 dtype 自动选粒度，FP16 寄存器选到 PK_B16 =
  取每 b16 元素低 8 bit 的**位截断**——输出垃圾且不报错。

每条 2026-09-29 A5 板证（bit-exact）；机制 owner：references/api/vf.md 二点十三 /
二点十四；跨 dtype 的落地样例：examples/algorithms/rmsnorm_gemm（UNPK 全 FP32
通路）。第一手来源：官方 ``LoadDist.md`` / ``StoreDist.md``（E2B 条描述有误）、
上板 ST
``frontend/lightning_indexer/``（dist 用法最密集的先例）。
"""

import pypto_pro.language as pl
from pypto_pro.language import Vf as vf  # noqa: N813

FP32 = pl.DT_FP32
FP16 = pl.DT_FP16

N = 128  # FP16 一个原生寄存器 = 256B / 2B = 128 元素；FP32 = 64
VL_FP32 = 64

_A16 = 0  # FP16 [1,128] = 256B
_A32 = 512  # FP32 [1,128] = 512B
_OUT = 1536


def _tt(dtype):
    return pl.TileType(shape=[1, N], dtype=dtype, target_memory=pl.MemorySpace.Vec)


# ---------------------------------------------------------------------------
# 一、UNPK 保序宽化：FP16 源 -> FP32 寄存器（跨 dtype 的标准入口）
# ---------------------------------------------------------------------------


@pl.vector_function
def _widen_unpk(src16, dst32):
    """``load_align(dist=UNPK)`` 读连续 64 个 FP16 展开进 64 个 b32 容器低半
    （保源序、高半零填），``astype(FP32, layout=ZERO)`` 从低半做数值转换。
    两条合起来等价于「FP32 源的一次 load」，128 列 = 两个 chunk。

    对照的错误写法（不出现在本例）：普通 ``load_align`` 一次取 128 个原生 FP16，
    直接 ``astype(dtype=FP32)`` 是偶数下标**源子采样**（``layout=ONE`` 给奇数
    下标）——症状是「输出 ≈ 源的偶数列」，不报错。
    """
    preg16 = vf.create_mask(pattern=pl.MaskPattern.ALL, dtype=FP16)
    preg32 = vf.create_mask(pattern=pl.MaskPattern.ALL, dtype=FP32)
    r0 = vf.load_align(src16, 0, dist=pl.LoadDist.UNPK)
    f0 = vf.astype(r0, preg16, dtype=FP32, layout=pl.CastLayout.ZERO)
    vf.store_align(dst32, f0, preg32, 0)
    r1 = vf.load_align(src16, VL_FP32, dist=pl.LoadDist.UNPK)
    f1 = vf.astype(r1, preg16, dtype=FP32, layout=pl.CastLayout.ZERO)
    vf.store_align(dst32, f1, preg32, VL_FP32)


# ---------------------------------------------------------------------------
# 二、E2B：element-to-block 广播（不是文档说的「16->32 扩展」）
# ---------------------------------------------------------------------------


@pl.vector_function
def _load_e2b(src16, dst16):
    """每个源元素复制填满一个 32B DataBlock：FP16 下前 8 个源元素各铺 16 lane。

    与 ``vf_align`` 第六节 FP32 下的既有观察（每元素重复 8 次）互证——E2B 在
    任何位宽下都是块广播，官方文档的「扩展」描述是错的（open-items 上报候选）。
    """
    preg16 = vf.create_mask(pattern=pl.MaskPattern.ALL, dtype=FP16)
    r = vf.load_align(src16, 0, dist=pl.LoadDist.E2B)
    vf.store_align(dst16, r, preg16, 0)


# ---------------------------------------------------------------------------
# 三、PACK_B32 窄化直出：astype(FP16, ZERO) 之后一条 store 抽稠密
# ---------------------------------------------------------------------------


@pl.vector_function
def _narrow_pack_b32(src32, dst16):
    """窄化结果在每个 b32 容器低半；``dist=PACK_B32``（取每 b32 元素低 16 bit
    连续写）一条 store 直出稠密 FP16。粒度**必须显式 B32**——见第四组的陷阱。"""
    preg16 = vf.create_mask(pattern=pl.MaskPattern.ALL, dtype=FP16)
    preg32 = vf.create_mask(pattern=pl.MaskPattern.ALL, dtype=FP32)
    q0 = vf.load_align(src32, 0)
    h0 = vf.astype(q0, preg32, dtype=FP16, layout=pl.CastLayout.ZERO)
    vf.store_align(dst16, h0, preg16, 0, dist=pl.StoreDist.PACK_B32)
    q1 = vf.load_align(src32, VL_FP32)
    h1 = vf.astype(q1, preg32, dtype=FP16, layout=pl.CastLayout.ZERO)
    vf.store_align(dst16, h1, preg16, VL_FP32, dist=pl.StoreDist.PACK_B32)


# ---------------------------------------------------------------------------
# 四、粗名陷阱：PACK on FP16 寄存器 = PK_B16 = 位截断
# ---------------------------------------------------------------------------


@pl.vector_function
def _pack_coarse_trap(src16, dst16):
    """粗名 ``PACK`` 自动按源 dtype 选粒度，FP16 选到 PK_B16——官方语义「取每个
    b16 元素的**低 8 bit**连续打包」：位截断，不是取元素。对 FP16 数据输出垃圾
    且不报错（1.0=0x3C00 低 8 bit 是 0）。这个 kernel 把当前行为钉成一条测试。"""
    preg16 = vf.create_mask(pattern=pl.MaskPattern.ALL, dtype=FP16)
    r = vf.load_align(src16, 0)
    vf.store_align(dst16, r, preg16, 0, dist=pl.StoreDist.PACK)


# ---------------------------------------------------------------------------
# kernels
# ---------------------------------------------------------------------------


@pl.jit(auto_mutex=True, name="alignw_widen_unpk")
def alignw_widen_unpk(a: pl.Tensor[[1, N], FP16], out: pl.Tensor[[1, N], FP32]):
    ga = pl.make_tile_group(type=_tt(FP16), addrs=[_A16], mutex_ids="auto", depth=1)
    go = pl.make_tile_group(type=_tt(FP32), addrs=[_A32], mutex_ids="auto", depth=1)
    with pl.section_vector():
        ta = ga.next()
        to = go.next()
        pl.load(ta, a, [0, 0])
        _widen_unpk(ta, to)
        pl.store(out, to, [0, 0])


@pl.jit(auto_mutex=True, name="alignw_load_e2b")
def alignw_load_e2b(a: pl.Tensor[[1, N], FP16], out: pl.Tensor[[1, N], FP16]):
    ga = pl.make_tile_group(type=_tt(FP16), addrs=[_A16], mutex_ids="auto", depth=1)
    go = pl.make_tile_group(type=_tt(FP16), addrs=[_OUT], mutex_ids="auto", depth=1)
    with pl.section_vector():
        ta = ga.next()
        to = go.next()
        pl.load(ta, a, [0, 0])
        _load_e2b(ta, to)
        pl.store(out, to, [0, 0])


@pl.jit(auto_mutex=True, name="alignw_narrow_pack_b32")
def alignw_narrow_pack_b32(a: pl.Tensor[[1, N], FP32], out: pl.Tensor[[1, N], FP16]):
    ga = pl.make_tile_group(type=_tt(FP32), addrs=[_A32], mutex_ids="auto", depth=1)
    go = pl.make_tile_group(type=_tt(FP16), addrs=[_OUT], mutex_ids="auto", depth=1)
    with pl.section_vector():
        ta = ga.next()
        to = go.next()
        pl.load(ta, a, [0, 0])
        _narrow_pack_b32(ta, to)
        pl.store(out, to, [0, 0])


@pl.jit(auto_mutex=True, name="alignw_pack_coarse_trap")
def alignw_pack_coarse_trap(a: pl.Tensor[[1, N], FP16], out: pl.Tensor[[1, N], FP16]):
    ga = pl.make_tile_group(type=_tt(FP16), addrs=[_A16], mutex_ids="auto", depth=1)
    go = pl.make_tile_group(type=_tt(FP16), addrs=[_OUT], mutex_ids="auto", depth=1)
    with pl.section_vector():
        ta = ga.next()
        to = go.next()
        pl.load(ta, a, [0, 0])
        pl.load(to, out, [0, 0])  # 哨兵先进 UB：后 64 槽「不被写」才可观测
        _pack_coarse_trap(ta, to)
        pl.store(out, to, [0, 0])


def make_pack4_b32(active: int, signed: bool, destination_offset: int = 0):
    """Extract low bytes from b32 lanes; no rounding or saturation is implied.

    Source-mask granularity and destination-element offsets are independent.
    Verified with signed/unsigned output on 872df763f, CANN 9.2, Ascend950PR.
    """
    dtype = pl.DT_INT8 if signed else pl.DT_UINT8

    @pl.vector_function
    def packed(src, dst):
        for offset in pl.range(0, active, VL_FP32):
            mask = vf.update_mask(pl.min(VL_FP32, active - offset), dtype=pl.DT_INT32)
            words = vf.load_align(src, [0, offset])
            vf.store_align(
                dst,
                words,
                mask,
                [0, destination_offset + offset],
                dist=pl.StoreDist.PACK4_B32,
            )

    @pl.jit(auto_mutex=True)
    def kernel(x: pl.Tensor[[1, N], pl.DT_INT32], out: pl.Tensor[[1, N], dtype]):
        inputs = pl.make_tile_group(
            type=pl.TileType(
                shape=[1, N], dtype=pl.DT_INT32, target_memory=pl.MemorySpace.Vec
            ),
            addrs=[0],
            mutex_ids="auto",
            depth=1,
        )
        outputs = pl.make_tile_group(
            type=pl.TileType(
                shape=[1, N], dtype=dtype, target_memory=pl.MemorySpace.Vec
            ),
            addrs=[N * 4],
            mutex_ids="auto",
            depth=1,
        )
        with pl.section_vector():
            source = inputs.current()
            target = outputs.current()
            pl.load(source, x, [0, 0])
            pl.load(target, out, [0, 0])
            packed(source, target)
            pl.store(out, target, [0, 0])

    return kernel


def make_pack4_fragment(active: int, signed: bool, logical_offset: int):
    """Pack at aligned UB offset zero, then drain a byte fragment to logical GM.

    The output's logical offset is not used as a VF destination address.
    This is a data-movement primitive, not a numeric quantizer.
    """
    dtype = pl.DT_INT8 if signed else pl.DT_UINT8

    @pl.vector_function
    def packed(src, dst):
        mask = vf.update_mask(active, dtype=pl.DT_INT32)
        words = vf.load_align(src, [0, 0])
        vf.store_align(dst, words, mask, [0, 0], dist=pl.StoreDist.PACK4_B32)

    @pl.jit(auto_mutex=True)
    def kernel(x: pl.Tensor[[1, 64], pl.DT_INT32], out: pl.Tensor[[1, N], dtype]):
        inputs = pl.make_tile_group(
            type=pl.TileType(
                shape=[1, 64], dtype=pl.DT_INT32, target_memory=pl.MemorySpace.Vec
            ),
            addrs=[0],
            mutex_ids="auto",
            depth=1,
        )
        outputs = pl.make_tile_group(
            type=pl.TileType(
                shape=[1, 64], dtype=dtype, target_memory=pl.MemorySpace.Vec
            ),
            addrs=[256],
            mutex_ids="auto",
            depth=1,
        )
        with pl.section_vector():
            source = inputs.current()
            local_result = outputs.current()
            pl.set_validshape(local_result, [1, active])
            pl.load(source, x, [0, 0])
            packed(source, local_result)
            pl.store(out, local_result, [0, logical_offset])

    return kernel


def make_u8_float_chain(active: int):
    """Convert unsigned bytes through b32 containers and pack an FP16 result.

    UNPK4 zero-extends each byte into a 32-bit container. The INT32 view is
    numerical only because the upper bits are zero and values are 0..255.
    All arithmetic and the PACK_B32 store use one mask bit-group per container.
    Verified on behavior 872df763f, CANN 9.2.0, Ascend950PR.
    """
    if not 1 <= active <= N:
        raise ValueError("active must be in 1..128")

    @pl.vector_function
    def convert(src, dst32, dst16):
        for offset in pl.range(0, active, VL_FP32):
            mask = vf.update_mask(pl.min(VL_FP32, active - offset), dtype=pl.DT_INT32)
            raw = vf.load_align(src, offset, dist=pl.LoadDist.UNPK4)
            integers = vf.bit_cast(raw, dtype=pl.DT_INT32)
            numbers = vf.astype(integers, mask, dtype=FP32)
            halves = vf.astype(numbers, mask, dtype=FP16, layout=pl.CastLayout.ZERO)
            vf.store_align(dst32, numbers, mask, offset)
            vf.store_align(dst16, halves, mask, offset, dist=pl.StoreDist.PACK_B32)

    @pl.jit(auto_mutex=True)
    def kernel(
        x: pl.Tensor[[1, N], pl.DT_UINT8],
        out32: pl.Tensor[[1, N], FP32],
        out16: pl.Tensor[[1, N], FP16],
    ):
        inputs = pl.make_tile_group(
            type=pl.TileType(
                shape=[1, N], dtype=pl.DT_UINT8, target_memory=pl.MemorySpace.Vec
            ),
            addrs=0x0,
            mutex_ids="auto",
            depth=1,
        )
        wide = pl.make_tile_group(
            type=pl.TileType(
                shape=[1, N], dtype=FP32, target_memory=pl.MemorySpace.Vec
            ),
            addrs=0x80,
            mutex_ids="auto",
            depth=1,
        )
        narrow = pl.make_tile_group(
            type=pl.TileType(
                shape=[1, N], dtype=FP16, target_memory=pl.MemorySpace.Vec
            ),
            addrs=0x280,
            mutex_ids="auto",
            depth=1,
        )
        with pl.section_vector():
            source = inputs.next()
            target32 = wide.next()
            target16 = narrow.next()
            pl.load(source, x, [0, 0])
            pl.set_validshape(target32, [1, active])
            pl.set_validshape(target16, [1, active])
            convert(source, target32, target16)
            pl.store(out32, target32, [0, 0])
            pl.store(out16, target16, [0, 0])

    return kernel
