# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

"""写法：原始。Fixpipe 的标量/逐列量化矩阵与 scale 负载探针。

固定 M=N=K=64。FP16×FP16 产生 FP32 Acc，INT8×INT8 产生 INT32 Acc；目的 dtype、layout 与量化方式在工厂调用时确定。Acc→Vec 的 11 个组合中，带 scale 的
FP32→FP32 由解析器拒绝，其他组合分别按精确值、取整/饱和或字节载体检查。Acc→Mat 覆盖 10 个组合，Acc→GM 覆盖 11 个组合的 ND/NZ 两种布局。

组深度均为 1，使用显式地址和 mutex_ids="auto"。L1 的 A/B 从 0、0x2000 开始，预留 FP16 最大尺寸；逐列表或量化后的矩阵从 0x4000 开始。两套 Left/Right 缓冲分别从 0、0x2000
开始，两个 Acc 从 0、0x4000 开始。各内存空间独立。普通 UB 从 0 开始；HF8 解码的 FP32 目标从 Vec 0x1000 开始，避开 4096B 的 HF8 源。固定 FP16→FP32 的生产者复用 riders
样例的原生声明。

Acc→Vec 是单次 FIX→MTE3 交接；HF8 解码首先由 VF 读取，因此等待 FIX→V。逐列表先 GM→L1，再在 MTE2→FIX 同步后搬入 Scaling，随后由同一 FIX 队列消费。核内依赖由自动 mutex 跟踪。

HF8 的 VF 解码只检查前 256 个字节中 stride=4、phase=0 的 64 个值。Acc→Mat 的 HF8 路径仍是运行检查，UINT8 路径只验证读回操作数被拒绝；这些不是独立格式或完整数值验证。参考、判据及这些边界由
reference.py 与 test.py 说明。
"""

import pypto_pro.language as pl
from pypto_pro.language import Vf as vf  # noqa: N813

from ..fixpipe_riders.kernel import _cube_groups

FP16 = pl.DT_FP16
FP32 = pl.DT_FP32
INT8 = pl.DT_INT8
INT32 = pl.DT_INT32
INT64 = pl.DT_INT64
UINT8 = pl.DT_UINT8
HF8 = pl.DT_HF8

M = N = K = 64

ND = pl.TensorLayout.ND
NZ = pl.TensorLayout.NZ

#: 默认量化比例。取 2 的幂，乘法在浮点里精确，判据不用容差。
SCALE = 0.5


def _input_dtype_for(acc_dtype):
    """Select FP16 or INT8 inputs for an FP32 or INT32 accumulator."""
    return FP16 if acc_dtype == FP32 else INT8


def make_quant_move(src_dtype, dst_dtype, *, with_scale=True, scale=SCALE):
    """Create one GEMM→Acc-to-Vec quantization→store chain; with_scale=False supplies the non-quantized control."""
    in_dtype = _input_dtype_for(src_dtype)

    @pl.jit(auto_mutex=True)
    def kernel(
        a: pl.Tensor[[M, K], in_dtype],
        b: pl.Tensor[[K, N], in_dtype],
        out: pl.Tensor[[M, N], dst_dtype],
    ):
        ub = pl.make_tile_group(
            type=pl.TileType(
                shape=[M, N],
                dtype=dst_dtype,
                target_memory=pl.MemorySpace.Vec,
                layout=ND,
            ),
            addrs=0x0,
            mutex_ids="auto",
            depth=1,
        )
        a_l1 = pl.make_tile_group(
            type=pl.TileType(
                shape=[M, K], dtype=in_dtype, target_memory=pl.MemorySpace.Mat
            ),
            addrs=0x0,
            mutex_ids="auto",
            depth=1,
        )
        b_l1 = pl.make_tile_group(
            type=pl.TileType(
                shape=[K, N], dtype=in_dtype, target_memory=pl.MemorySpace.Mat
            ),
            addrs=0x2000,
            mutex_ids="auto",
            depth=1,
        )
        l0a = pl.make_tile_group(
            type=pl.TileType(
                shape=[M, K], dtype=in_dtype, target_memory=pl.MemorySpace.Left
            ),
            addrs=0x0,
            mutex_ids="auto",
            depth=1,
        )
        l0b = pl.make_tile_group(
            type=pl.TileType(
                shape=[K, N], dtype=in_dtype, target_memory=pl.MemorySpace.Right
            ),
            addrs=0x0,
            mutex_ids="auto",
            depth=1,
        )
        acc = pl.make_tile_group(
            type=pl.TileType(
                shape=[M, N], dtype=src_dtype, target_memory=pl.MemorySpace.Acc
            ),
            addrs=0x0,
            mutex_ids="auto",
            depth=1,
        )
        t = ub.current()
        with pl.section_cube():
            cur_a, cur_b, ac = a_l1.current(), b_l1.current(), acc.current()
            pl.load(cur_a, a, [0, 0])
            pl.load(cur_b, b, [0, 0])
            pl.gemm(ac, cur_a, cur_b, l0a=l0a, l0b=l0b)
            if with_scale:
                pl.move(t, ac, scale=scale)
            else:
                pl.move(t, ac)
            pl.system.set_cross_core(pipe=pl.PipeType.FIX, event_id=0)
        with pl.section_vector():
            pl.system.wait_cross_core(pipe=pl.PipeType.MTE3, event_id=0)
            if pl.get_subblock_idx() == 0:
                pl.store(out, t, [0, 0])
            pl.system.bar_all()

    return kernel


def make_move_with_quant_offset(quant_offset):
    """Apply an explicit quant_offset on the Acc-to-Vec INT8 route. Unit inputs produce K*SCALE before offset and
    saturation.
    """

    @pl.jit(auto_mutex=True)
    def kernel(
        a: pl.Tensor[[M, K], FP16],
        b: pl.Tensor[[K, N], FP16],
        out: pl.Tensor[[M, N], INT8],
    ):
        ub = pl.make_tile_group(
            type=pl.TileType(
                shape=[M, N], dtype=INT8, target_memory=pl.MemorySpace.Vec, layout=ND
            ),
            addrs=0x0,
            mutex_ids="auto",
            depth=1,
        )
        a_l1, b_l1, l0a, l0b, acc = _cube_groups()
        t = ub.current()
        with pl.section_cube():
            cur_a, cur_b, ac = a_l1.current(), b_l1.current(), acc.current()
            pl.load(cur_a, a, [0, 0])
            pl.load(cur_b, b, [0, 0])
            pl.gemm(ac, cur_a, cur_b, l0a=l0a, l0b=l0b)
            pl.move(t, ac, scale=0.5, quant_offset=quant_offset)
            pl.system.set_cross_core(pipe=pl.PipeType.FIX, event_id=0)
        with pl.section_vector():
            pl.system.wait_cross_core(pipe=pl.PipeType.MTE3, event_id=0)
            if pl.get_subblock_idx() == 0:
                pl.store(out, t, [0, 0])
            pl.system.bar_all()

    return kernel


def make_per_channel_move(src_dtype, dst_dtype):
    """Consume an INT64 [1,N] Scaling table. Synchronize MTE2→FIX before the L1→Scaling copy; scalar controls use
    make_quant_move.
    """
    in_dtype = _input_dtype_for(src_dtype)

    @pl.jit(auto_mutex=True)
    def kernel(
        a: pl.Tensor[[M, K], in_dtype],
        b: pl.Tensor[[K, N], in_dtype],
        scales: pl.Tensor[[1, N], INT64],
        out: pl.Tensor[[M, N], dst_dtype],
    ):
        ub = pl.make_tile_group(
            type=pl.TileType(
                shape=[M, N],
                dtype=dst_dtype,
                target_memory=pl.MemorySpace.Vec,
                layout=ND,
            ),
            addrs=0x0,
            mutex_ids="auto",
            depth=1,
        )
        a_l1 = pl.make_tile_group(
            type=pl.TileType(
                shape=[M, K], dtype=in_dtype, target_memory=pl.MemorySpace.Mat
            ),
            addrs=0x0,
            mutex_ids="auto",
            depth=1,
        )
        b_l1 = pl.make_tile_group(
            type=pl.TileType(
                shape=[K, N], dtype=in_dtype, target_memory=pl.MemorySpace.Mat
            ),
            addrs=0x2000,
            mutex_ids="auto",
            depth=1,
        )
        l0a = pl.make_tile_group(
            type=pl.TileType(
                shape=[M, K], dtype=in_dtype, target_memory=pl.MemorySpace.Left
            ),
            addrs=0x0,
            mutex_ids="auto",
            depth=1,
        )
        l0b = pl.make_tile_group(
            type=pl.TileType(
                shape=[K, N], dtype=in_dtype, target_memory=pl.MemorySpace.Right
            ),
            addrs=0x0,
            mutex_ids="auto",
            depth=1,
        )
        acc = pl.make_tile_group(
            type=pl.TileType(
                shape=[M, N], dtype=src_dtype, target_memory=pl.MemorySpace.Acc
            ),
            addrs=0x0,
            mutex_ids="auto",
            depth=1,
        )
        scale_mat = pl.make_tile_group(
            type=pl.TileType(
                shape=[1, N], dtype=INT64, target_memory=pl.MemorySpace.Mat, layout=ND
            ),
            addrs=0x4000,
            mutex_ids="auto",
            depth=1,
        )
        scale_fb = pl.make_tile_group(
            type=pl.TileType(
                shape=[1, N], dtype=INT64, target_memory=pl.MemorySpace.Scaling
            ),
            addrs=0x0,
            mutex_ids="auto",
            depth=1,
        )
        t = ub.current()
        with pl.section_cube():
            cur_a, cur_b, ac = a_l1.current(), b_l1.current(), acc.current()
            fp_mat = scale_mat.current()
            fp_tile = scale_fb.current()
            pl.load(cur_a, a, [0, 0])
            pl.load(cur_b, b, [0, 0])
            pl.load(fp_mat, scales, [0, 0])
            pl.gemm(ac, cur_a, cur_b, l0a=l0a, l0b=l0b)
            pl.system.sync_src(
                set_pipe=pl.PipeType.MTE2, wait_pipe=pl.PipeType.FIX, event_id=1
            )
            pl.system.sync_dst(
                set_pipe=pl.PipeType.MTE2, wait_pipe=pl.PipeType.FIX, event_id=1
            )
            pl.move(fp_tile, fp_mat)
            pl.move(t, ac, scale=fp_tile)
            pl.system.set_cross_core(pipe=pl.PipeType.FIX, event_id=0)
        with pl.section_vector():
            pl.system.wait_cross_core(pipe=pl.PipeType.MTE3, event_id=0)
            if pl.get_subblock_idx() == 0:
                pl.store(out, t, [0, 0])
            pl.system.bar_all()

    return kernel


def _second_acc_dtype_for(dst_dtype):
    """Select INT32 for integer operands and FP32 for floating operands in the q@q readback."""
    if dst_dtype == INT8 or dst_dtype == UINT8:
        return INT32
    return FP32


def make_quant_move_to_mat(src_dtype, dst_dtype):
    """Quantize Acc into NZ L1 and read back with q@q using separate L0/Acc buffers. UINT8 readback is rejected; HF8 has
    execution-only checks.
    """
    in_dtype = _input_dtype_for(src_dtype)
    acc2_dtype = _second_acc_dtype_for(dst_dtype)

    @pl.jit(auto_mutex=True)
    def kernel(
        a: pl.Tensor[[M, K], in_dtype],
        b: pl.Tensor[[K, N], in_dtype],
        out: pl.Tensor[[M, N], acc2_dtype],
    ):
        a_l1 = pl.make_tile_group(
            type=pl.TileType(
                shape=[M, K], dtype=in_dtype, target_memory=pl.MemorySpace.Mat
            ),
            addrs=0x0,
            mutex_ids="auto",
            depth=1,
        )
        b_l1 = pl.make_tile_group(
            type=pl.TileType(
                shape=[K, N], dtype=in_dtype, target_memory=pl.MemorySpace.Mat
            ),
            addrs=0x2000,
            mutex_ids="auto",
            depth=1,
        )
        l0a = pl.make_tile_group(
            type=pl.TileType(
                shape=[M, K], dtype=in_dtype, target_memory=pl.MemorySpace.Left
            ),
            addrs=0x0,
            mutex_ids="auto",
            depth=1,
        )
        l0b = pl.make_tile_group(
            type=pl.TileType(
                shape=[K, N], dtype=in_dtype, target_memory=pl.MemorySpace.Right
            ),
            addrs=0x0,
            mutex_ids="auto",
            depth=1,
        )
        acc = pl.make_tile_group(
            type=pl.TileType(
                shape=[M, N], dtype=src_dtype, target_memory=pl.MemorySpace.Acc
            ),
            addrs=0x0,
            mutex_ids="auto",
            depth=1,
        )
        q_l1 = pl.make_tile_group(
            type=pl.TileType(
                shape=[M, N],
                dtype=dst_dtype,
                target_memory=pl.MemorySpace.Mat,
                layout=NZ,
            ),
            addrs=0x4000,
            mutex_ids="auto",
            depth=1,
        )
        q_l0a = pl.make_tile_group(
            type=pl.TileType(
                shape=[M, N], dtype=dst_dtype, target_memory=pl.MemorySpace.Left
            ),
            addrs=0x2000,
            mutex_ids="auto",
            depth=1,
        )
        q_l0b = pl.make_tile_group(
            type=pl.TileType(
                shape=[M, N], dtype=dst_dtype, target_memory=pl.MemorySpace.Right
            ),
            addrs=0x2000,
            mutex_ids="auto",
            depth=1,
        )
        acc2 = pl.make_tile_group(
            type=pl.TileType(
                shape=[M, N], dtype=acc2_dtype, target_memory=pl.MemorySpace.Acc
            ),
            addrs=0x4000,
            mutex_ids="auto",
            depth=1,
        )
        with pl.section_cube():
            cur_a, cur_b, ac = a_l1.current(), b_l1.current(), acc.current()
            pl.load(cur_a, a, [0, 0])
            pl.load(cur_b, b, [0, 0])
            pl.gemm(ac, cur_a, cur_b, l0a=l0a, l0b=l0b)
            q = q_l1.current()
            pl.move(q, ac, scale=SCALE)  # ← Acc → Mat 随路量化
            ac2 = acc2.current()
            pl.gemm(ac2, q, q, l0a=q_l0a, l0b=q_l0b)  # ← 唯一的读回途径
            pl.store(out, ac2, [0, 0])

    return kernel


def make_quant_store(src_dtype, dst_dtype, dst_layout):
    """Store quantized Acc directly to GM in ND or NZ, including scaled FP32 output."""
    in_dtype = _input_dtype_for(src_dtype)

    @pl.jit(auto_mutex=True)
    def kernel(
        a: pl.Tensor[[M, K], in_dtype],
        b: pl.Tensor[[K, N], in_dtype],
        out: pl.Tensor[[M, N], dst_dtype, dst_layout],
    ):
        a_l1 = pl.make_tile_group(
            type=pl.TileType(
                shape=[M, K], dtype=in_dtype, target_memory=pl.MemorySpace.Mat
            ),
            addrs=0x0,
            mutex_ids="auto",
            depth=1,
        )
        b_l1 = pl.make_tile_group(
            type=pl.TileType(
                shape=[K, N], dtype=in_dtype, target_memory=pl.MemorySpace.Mat
            ),
            addrs=0x2000,
            mutex_ids="auto",
            depth=1,
        )
        l0a = pl.make_tile_group(
            type=pl.TileType(
                shape=[M, K], dtype=in_dtype, target_memory=pl.MemorySpace.Left
            ),
            addrs=0x0,
            mutex_ids="auto",
            depth=1,
        )
        l0b = pl.make_tile_group(
            type=pl.TileType(
                shape=[K, N], dtype=in_dtype, target_memory=pl.MemorySpace.Right
            ),
            addrs=0x0,
            mutex_ids="auto",
            depth=1,
        )
        acc = pl.make_tile_group(
            type=pl.TileType(
                shape=[M, N], dtype=src_dtype, target_memory=pl.MemorySpace.Acc
            ),
            addrs=0x0,
            mutex_ids="auto",
            depth=1,
        )
        with pl.section_cube():
            cur_a, cur_b, ac = a_l1.current(), b_l1.current(), acc.current()
            pl.load(cur_a, a, [0, 0])
            pl.load(cur_b, b, [0, 0])
            pl.gemm(ac, cur_a, cur_b, l0a=l0a, l0b=l0b)
            pl.store(out, ac, [0, 0], scale=SCALE)

    return kernel


# ---------------------------------------------------------------------------
# HF8：借向量单元的 astype 在设备上解码
# ---------------------------------------------------------------------------

#: 一个 FP32 寄存器 = 256 字节 = 64 个元素。
VL_FP32 = 64


@pl.vector_function
def _hf8_to_fp32(src, dst):
    """Convert 64 stride-4, phase-0 bytes from the first HF8 register using CastLayout.ZERO."""
    preg = vf.create_mask(pattern=pl.MaskPattern.ALL, dtype=FP32)
    a = vf.load_align(src, 0)
    f = vf.astype(a, preg, dtype=FP32, layout=pl.CastLayout.ZERO)
    vf.store_align(dst, f, preg, 0)


@pl.jit(auto_mutex=True)
def hf8_move_and_decode(
    a: pl.Tensor[[M, K], FP16],
    b: pl.Tensor[[K, N], FP16],
    out: pl.Tensor[[1, VL_FP32], FP32],
):
    """Compare Fixpipe HF8 output with VF interpretation on a 64-value subset; the first consumer waits on V."""
    hf8_ub = pl.make_tile_group(
        type=pl.TileType(
            shape=[M, N], dtype=HF8, target_memory=pl.MemorySpace.Vec, layout=ND
        ),
        addrs=0x0,
        mutex_ids="auto",
        depth=1,
    )
    fp32_ub = pl.make_tile_group(
        type=pl.TileType(
            shape=[1, VL_FP32], dtype=FP32, target_memory=pl.MemorySpace.Vec, layout=ND
        ),
        addrs=0x1000,
        mutex_ids="auto",
        depth=1,
    )
    a_l1, b_l1, l0a, l0b, acc = _cube_groups()
    t = hf8_ub.current()
    tf = fp32_ub.current()
    with pl.section_cube():
        cur_a, cur_b, ac = a_l1.current(), b_l1.current(), acc.current()
        pl.load(cur_a, a, [0, 0])
        pl.load(cur_b, b, [0, 0])
        pl.gemm(ac, cur_a, cur_b, l0a=l0a, l0b=l0b)
        pl.move(t, ac, scale=SCALE)
        pl.system.set_cross_core(pipe=pl.PipeType.FIX, event_id=0)
    with pl.section_vector():
        # **`pipe` 是「等待期间被阻塞的流水」**（官方 `wait_cross_core.md` 参数表原话）,
        # 所以它必须是消费者所在的那条。本章其他 kernel 的消费者是 `pl.store`（MTE3）,
        # 这里是 `vf`（V 流水）：拦 MTE3 的话 V 没被拦，vf 会和 fixpipe 的写竞争——
        # 单独跑能过，跟别的用例一起跑就错（实测最大相对差 7.6）。
        pl.system.wait_cross_core(pipe=pl.PipeType.V, event_id=0)
        if pl.get_subblock_idx() == 0:
            _hf8_to_fp32(t, tf)
            pl.store(out, tf, [0, 0])
        pl.system.bar_all()


# ---------------------------------------------------------------------------
# scale 自己的精度：运行期标量，一个 kernel 扫很多 scale
# ---------------------------------------------------------------------------


@pl.jit(auto_mutex=True)
def store_with_runtime_scale(
    a: pl.Tensor[[M, K], FP16],
    b: pl.Tensor[[K, N], FP16],
    scale_bits: pl.DT_INT64,
    out: pl.Tensor[[M, N], FP32],
):
    """Read a packed INT64 scalar as the scale configuration and store FP32 output; unit inputs expose the effective
    scale after division by K.
    """
    a_l1, b_l1, l0a, l0b, acc = _cube_groups()
    with pl.section_cube():
        cur_a, cur_b, ac = a_l1.current(), b_l1.current(), acc.current()
        pl.load(cur_a, a, [0, 0])
        pl.load(cur_b, b, [0, 0])
        pl.gemm(ac, cur_a, cur_b, l0a=l0a, l0b=l0b)
        pl.store(out, ac, [0, 0], scale=scale_bits)


@pl.jit(auto_mutex=True)
def store_with_runtime_payload_int8(
    a: pl.Tensor[[M, K], FP16],
    b: pl.Tensor[[K, N], FP16],
    payload: pl.DT_INT64,
    out: pl.Tensor[[M, N], INT8],
):
    """Store INT8 using a packed runtime scale/offset payload. Integer output exposes the signed nine-bit offset
    field.
    """
    a_l1, b_l1, l0a, l0b, acc = _cube_groups()
    with pl.section_cube():
        cur_a, cur_b, ac = a_l1.current(), b_l1.current(), acc.current()
        pl.load(cur_a, a, [0, 0])
        pl.load(cur_b, b, [0, 0])
        pl.gemm(ac, cur_a, cur_b, l0a=l0a, l0b=l0b)
        pl.store(out, ac, [0, 0], scale=payload)
