# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

"""写法：原始。残差、RMSNorm 与逐行对称 INT8 动态量化。

契约：连续 FP16 x/residual[M,N]、FP32 gamma[N]，M=1..128、N=1..16385，输入有限且 |x|/|residual|<=8、|gamma|<=2。先按 FP32 相加并以 FP16 RNE 得到
r，再基于该 r 计算 RMSNorm、s=max(abs(u))/127 和 q=clip(RNE(u/s),-127,127)。全零行 s=1、q=0。

每个 worker 按行轮转分工。第一阶段输出 r 与 FP32 inv_rms[M]，第二阶段读入整行 r，常驻 gamma 并在 VF 内归约、量化。列数向上对齐到 128；最大 N 下，两阶段单槽 UB 峰值分别是 66,304 和
115,840 字节。每次调用有两个 native task，额外一次 r 的 GM 读回和调用开销是此分阶段实现的成本；Host 只校验元数据、分配辅助统计量并调度。

FP16 UNPK 后每个逻辑值位于 b32 容器的低半字；这里用 FP32 粒度、计数为 n 的 predicate 覆盖 n 个容器。FP16 RNE 结果经 de_interleave 合并成稠密的 128 元素行块，平方和消费刚舍入
的寄存器。量化先 RNE 转 INT32、裁剪，再用 PACK4_B32 写入 INT8 tile。

内部 gamma 统一乘 2^64，q 使用 scaled_u/scaled_max*127，输出 s 再乘 2^-64。由非零 |r|>=2^-24、|gamma|>=2^-149、inv_rms>2^-5，非零 scaled_u 大于
2^-114；上界小于 16*317*2^65，远低于 FP32 上限。极端幅值比下溢只会影响必然舍入为零的 q；外部 s 舍入到零不用于判定零行或作为内部除数。最大系数乘零而微小系数乘非零值的情况也保留正确的零集合。

各阶段的 tile group 与 auto_mutex 处理搬运/VF 依赖；两次原生调用在同一 stream 上有序执行。验证环境为 Ascend950PR、CANN 9.2.0、torch/torch_npu 2.10.0，PyPTO
行为基线 872df763f。test.py 使用独立 FP64 参考并分别验证各输出，测试范围与固定阈值写在其文件头中。
"""

from functools import lru_cache

import torch
import pypto_pro.language as pl
from pypto_pro.language import Vf as vf  # noqa: N813


FP16 = pl.DT_FP16
FP32 = pl.DT_FP32
INT32 = pl.DT_INT32
INT8 = pl.DT_INT8
LANES = 64
PAIR = 2 * LANES
EPSILON = 1.0e-5


def _align(value, alignment):
    return ((value + alignment - 1) // alignment) * alignment


@lru_cache(maxsize=None)
def _make_residual_kernel(rows, cols):
    padded = _align(cols, PAIR)
    tile_bytes = padded * 2
    x_addr = 0
    residual_addr = _align(x_addr + tile_bytes, 32)
    statistic_addr = _align(residual_addr + tile_bytes, 32)
    reciprocal_cols = 1.0 / cols
    pair_count = padded // PAIR

    @pl.vector_function
    def _produce_r(x_tile, residual_tile, statistic_tile):
        all_f32 = vf.create_mask(pattern=pl.MaskPattern.ALL, dtype=FP32)
        all_f16 = vf.create_mask(pattern=pl.MaskPattern.ALL, dtype=FP16)
        first_f32 = vf.update_mask(1, dtype=FP32)
        total_square = vf.full(0.0, all_f32, dtype=FP32)

        for pair_index in pl.range(pair_count):
            col = pair_index * PAIR
            active0 = pl.min(LANES, pl.max(0, cols - col))
            active1 = pl.min(LANES, pl.max(0, cols - col - LANES))
            mask0 = vf.update_mask(active0, dtype=FP32)
            mask1 = vf.update_mask(active1, dtype=FP32)

            x0h = vf.load_align(x_tile, [0, col], dist=pl.LoadDist.UNPK)
            residual0h = vf.load_align(residual_tile, [0, col], dist=pl.LoadDist.UNPK)
            x0 = vf.astype(x0h, mask0, dtype=FP32, layout=pl.CastLayout.ZERO)
            residual0 = vf.astype(
                residual0h, mask0, dtype=FP32, layout=pl.CastLayout.ZERO
            )
            sum0 = vf.add(x0, residual0, mask0)
            r0h = vf.astype(
                sum0,
                mask0,
                dtype=FP16,
                layout=pl.CastLayout.ZERO,
                round_mode=pl.VFRoundMode.CAST_RINT,
            )
            r0 = vf.astype(r0h, mask0, dtype=FP32, layout=pl.CastLayout.ZERO)
            square0 = vf.mul(r0, r0, mask0)
            part0 = vf.reduce_sum(square0, mask0)
            total_square = vf.add(total_square, part0, all_f32)

            x1h = vf.load_align(x_tile, [0, col + LANES], dist=pl.LoadDist.UNPK)
            residual1h = vf.load_align(
                residual_tile, [0, col + LANES], dist=pl.LoadDist.UNPK
            )
            x1 = vf.astype(x1h, mask1, dtype=FP32, layout=pl.CastLayout.ZERO)
            residual1 = vf.astype(
                residual1h, mask1, dtype=FP32, layout=pl.CastLayout.ZERO
            )
            sum1 = vf.add(x1, residual1, mask1)
            r1h = vf.astype(
                sum1,
                mask1,
                dtype=FP16,
                layout=pl.CastLayout.ZERO,
                round_mode=pl.VFRoundMode.CAST_RINT,
            )
            r1 = vf.astype(r1h, mask1, dtype=FP32, layout=pl.CastLayout.ZERO)
            square1 = vf.mul(r1, r1, mask1)
            part1 = vf.reduce_sum(square1, mask1)
            total_square = vf.add(total_square, part1, all_f32)

            rounded_pair, _ = vf.de_interleave(r0h, r1h)
            vf.store_align(residual_tile, rounded_pair, all_f16, [0, col])

        mean_square = vf.muls(total_square, reciprocal_cols, all_f32)
        mean_square_eps = vf.adds(mean_square, EPSILON, all_f32)
        rms = vf.sqrt(mean_square_eps, all_f32)
        ones = vf.full(1.0, all_f32, dtype=FP32)
        inv_rms = vf.div(ones, rms, all_f32)
        vf.store_align(statistic_tile, inv_rms, first_f32, [0, 0])

    @pl.jit(auto_mutex=True)
    def residual_kernel(
        x: pl.Tensor[[rows, cols], FP16],
        residual: pl.Tensor[[rows, cols], FP16],
        r: pl.Tensor[[rows, cols], FP16],
        inv_rms: pl.Tensor[[rows], FP32],
    ):
        row_type = pl.TileType(
            shape=[1, padded],
            dtype=FP16,
            target_memory=pl.MemorySpace.Vec,
            valid_shape=[-1, -1],
            pad=pl.TilePad.zero,
        )
        statistic_type = pl.TileType(
            shape=[1, LANES],
            dtype=FP32,
            target_memory=pl.MemorySpace.Vec,
            valid_shape=[1, 1],
            pad=pl.TilePad.zero,
        )
        x_group = pl.make_tile_group(type=row_type, addrs=[x_addr], mutex_ids=[0])
        residual_group = pl.make_tile_group(
            type=row_type, addrs=[residual_addr], mutex_ids=[1]
        )
        statistic_group = pl.make_tile_group(
            type=statistic_type, addrs=[statistic_addr], mutex_ids=[2]
        )

        with pl.section_vector():
            for row in pl.range(pl.get_block_idx(), rows, pl.get_block_num()):
                x_tile = x_group.next()
                residual_tile = residual_group.next()
                statistic_tile = statistic_group.next()
                pl.set_validshape(x_tile, [1, cols])
                pl.load(x_tile, x, [row, 0])
                pl.set_validshape(x_tile, [1, padded])
                pl.set_validshape(residual_tile, [1, cols])
                pl.load(residual_tile, residual, [row, 0])
                pl.set_validshape(residual_tile, [1, padded])
                for col in pl.range(cols, padded):
                    pl.setval(x_tile, col, pl.const(0.0, FP16))
                    pl.setval(residual_tile, col, pl.const(0.0, FP16))

                _produce_r(x_tile, residual_tile, statistic_tile)
                pl.set_validshape(residual_tile, [1, cols])
                pl.store(r, residual_tile, [row, 0])
                pl.store(inv_rms, statistic_tile, [row])

    return residual_kernel


@lru_cache(maxsize=None)
def _make_quant_kernel(rows, cols, preload_gamma):
    padded = _align(cols, PAIR)
    row_half_bytes = padded * 2
    row_float_bytes = padded * 4
    q_bytes = padded
    r_addr = 0
    gamma_addr = _align(r_addr + row_half_bytes, 32)
    q_addr = _align(gamma_addr + row_float_bytes, 32)
    scale_addr = _align(q_addr + q_bytes, 32)
    reciprocal_127 = 1.0 / 127.0
    gamma_expansion = 2.0**64
    scale_contraction = 2.0**-64

    @pl.vector_function
    def _normalize_quantize(r_tile, gamma_tile, q_tile, scale_tile, inv_rms_scalar):
        all_f32 = vf.create_mask(pattern=pl.MaskPattern.ALL, dtype=FP32)
        first_f32 = vf.update_mask(1, dtype=FP32)
        inv_rms = vf.full(inv_rms_scalar, all_f32, dtype=FP32)
        max_abs = vf.full(0.0, all_f32, dtype=FP32)

        for block in pl.range(0, padded, LANES):
            active = pl.min(LANES, pl.max(0, cols - block))
            mask = vf.update_mask(active, dtype=FP32)
            r16 = vf.load_align(r_tile, [0, block], dist=pl.LoadDist.UNPK)
            r32 = vf.astype(r16, mask, dtype=FP32, layout=pl.CastLayout.ZERO)
            gamma = vf.load_align(gamma_tile, [0, block])
            gamma_scaled = vf.muls(gamma, gamma_expansion, mask)
            scaled = vf.mul(r32, inv_rms, mask)
            u = vf.mul(scaled, gamma_scaled, mask)
            magnitude = vf.abs(u, mask)
            block_max = vf.reduce_max(magnitude, mask)
            max_abs = vf.max(max_abs, block_max, all_f32)

        zero = vf.full(0.0, all_f32, dtype=FP32)
        is_zero = vf.eq(max_abs, zero, all_f32)
        one = vf.full(1.0, all_f32, dtype=FP32)
        scale_value = vf.muls(max_abs, reciprocal_127, all_f32)
        scale_value = vf.muls(scale_value, scale_contraction, all_f32)
        scale_value = vf.select(one, scale_value, is_zero)
        vf.store_align(scale_tile, scale_value, first_f32, [0, 0])

        # u/max_abs*127 is the stable form of u/s. The zero-row branch uses
        # one as its divisor and therefore writes exact q=0.
        safe_max = vf.select(one, max_abs, is_zero)
        safe_max_broadcast = vf.full(safe_max, all_f32)
        for block in pl.range(0, padded, LANES):
            active = pl.min(LANES, pl.max(0, cols - block))
            mask = vf.update_mask(active, dtype=FP32)
            int_mask = vf.update_mask(active, dtype=INT32)
            r16 = vf.load_align(r_tile, [0, block], dist=pl.LoadDist.UNPK)
            r32 = vf.astype(r16, mask, dtype=FP32, layout=pl.CastLayout.ZERO)
            gamma = vf.load_align(gamma_tile, [0, block])
            gamma_scaled = vf.muls(gamma, gamma_expansion, mask)
            scaled = vf.mul(r32, inv_rms, mask)
            u = vf.mul(scaled, gamma_scaled, mask)
            normalized = vf.div(u, safe_max_broadcast, mask)
            z = vf.muls(normalized, 127.0, mask)
            rounded = vf.astype(
                z,
                mask,
                dtype=INT32,
                round_mode=pl.VFRoundMode.CAST_RINT,
            )
            lower = vf.maxs(rounded, pl.const(-127, INT32), int_mask)
            clipped = vf.mins(lower, pl.const(127, INT32), int_mask)
            vf.store_align(
                q_tile,
                clipped,
                int_mask,
                [0, block],
                dist=pl.StoreDist.PACK4_B32,
            )

    @pl.jit(auto_mutex=True)
    def quant_kernel(
        r: pl.Tensor[[rows, cols], FP16],
        gamma: pl.Tensor[[cols], FP32],
        inv_rms: pl.Tensor[[rows], FP32],
        s: pl.Tensor[[rows], FP32],
        q: pl.Tensor[[rows, cols], INT8],
    ):
        r_type = pl.TileType(
            shape=[1, padded],
            dtype=FP16,
            target_memory=pl.MemorySpace.Vec,
            valid_shape=[-1, -1],
            pad=pl.TilePad.zero,
        )
        gamma_type = pl.TileType(
            shape=[1, padded],
            dtype=FP32,
            target_memory=pl.MemorySpace.Vec,
            valid_shape=[-1, -1],
            pad=pl.TilePad.zero,
        )
        q_type = pl.TileType(
            shape=[1, q_bytes],
            dtype=INT8,
            target_memory=pl.MemorySpace.Vec,
            valid_shape=[-1, -1],
            pad=pl.TilePad.zero,
        )
        scale_type = pl.TileType(
            shape=[1, LANES],
            dtype=FP32,
            target_memory=pl.MemorySpace.Vec,
            valid_shape=[1, 1],
            pad=pl.TilePad.zero,
        )
        r_group = pl.make_tile_group(type=r_type, addrs=[r_addr], mutex_ids=[0])
        gamma_group = pl.make_tile_group(
            type=gamma_type, addrs=[gamma_addr], mutex_ids=[1]
        )
        q_group = pl.make_tile_group(type=q_type, addrs=[q_addr], mutex_ids=[2])
        scale_group = pl.make_tile_group(
            type=scale_type, addrs=[scale_addr], mutex_ids=[3]
        )

        with pl.section_vector():
            if preload_gamma:
                gamma_tile = gamma_group.next()
                pl.set_validshape(gamma_tile, [1, cols])
                pl.load(gamma_tile, gamma, [0])
                pl.set_validshape(gamma_tile, [1, padded])
                for col in pl.range(cols, padded):
                    pl.setval(gamma_tile, col, pl.const(0.0, FP32))

            for row in pl.range(pl.get_block_idx(), rows, pl.get_block_num()):
                r_tile = r_group.next()
                q_tile = q_group.next()
                scale_tile = scale_group.next()
                pl.set_validshape(r_tile, [1, cols])
                pl.load(r_tile, r, [row, 0])
                pl.set_validshape(r_tile, [1, padded])
                for col in pl.range(cols, padded):
                    pl.setval(r_tile, col, pl.const(0.0, FP16))

                if not preload_gamma:
                    gamma_tile = gamma_group.next()
                    pl.set_validshape(gamma_tile, [1, cols])
                    pl.load(gamma_tile, gamma, [0])
                    pl.set_validshape(gamma_tile, [1, padded])
                    for col in pl.range(cols, padded):
                        pl.setval(gamma_tile, col, pl.const(0.0, FP32))

                inv_rms_scalar = pl.getval(inv_rms, row)
                _normalize_quantize(
                    r_tile, gamma_tile, q_tile, scale_tile, inv_rms_scalar
                )
                pl.store(s, scale_tile, [row])
                pl.set_validshape(q_tile, [1, cols])
                pl.store(q, q_tile, [row, 0])

    return quant_kernel


def _validate(x, residual, gamma, r, s, q, workers):
    if type(workers) is not int or workers <= 0:
        raise ValueError("workers must be a positive integer")
    if len(x.shape) != 2:
        raise ValueError("x must have shape [M, N]")
    rows, cols = x.shape
    if not (1 <= rows <= 128 and 1 <= cols <= 16385):
        raise ValueError("expected 1 <= M <= 128 and 1 <= N <= 16385")
    if tuple(residual.shape) != (rows, cols) or tuple(r.shape) != (rows, cols):
        raise ValueError("residual and r must have shape [M, N]")
    if tuple(gamma.shape) != (cols,):
        raise ValueError("gamma must have shape [N]")
    if tuple(s.shape) != (rows,):
        raise ValueError("s must have shape [M]")
    if tuple(q.shape) != (rows, cols):
        raise ValueError("q must have shape [M, N]")
    if any(not tensor.is_contiguous() for tensor in (x, residual, gamma, r, s, q)):
        raise ValueError("all tensors must be contiguous")
    if any(tensor.device != x.device for tensor in (residual, gamma, r, s, q)):
        raise ValueError("all tensors must be on the same device")
    if (x.dtype, residual.dtype, gamma.dtype, r.dtype, s.dtype, q.dtype) != (
        torch.float16,
        torch.float16,
        torch.float32,
        torch.float16,
        torch.float32,
        torch.int8,
    ):
        raise ValueError("expected FP16 x/residual/r, FP32 gamma/s, and INT8 q")
    return rows, cols


def _run_variant(x, residual, gamma, r, s, q, workers, preload_gamma):
    rows, cols = _validate(x, residual, gamma, r, s, q, workers)
    inv_rms = s.new_empty((rows,))
    _make_residual_kernel(rows, cols)[None, workers](x, residual, r, inv_rms)
    return _make_quant_kernel(rows, cols, preload_gamma)[None, workers](
        r, gamma, inv_rms, s, q
    )


def run(x, residual, gamma, r, s, q, workers=1):
    """Run native PyPTO kernels; the Host reads metadata, allocates and launches only."""
    return _run_variant(x, residual, gamma, r, s, q, workers, preload_gamma=True)
