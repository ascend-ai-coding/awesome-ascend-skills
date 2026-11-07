# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

"""写法：原始。流式行均值与总体方差，逐 lane 和块间补偿归约。

契约：连续 FP32 x[M,N]，M=1..64、N=1..262145，有限值在 [-16384,16384]；输出 FP32 mean[M]/var[M]。均值 atol=1e-3/rtol=2e-6，方差
atol=1e-5/rtol=2e-4，常数行与 N=1 方差精确零。
分工与资源：每行由一个 worker 独占；默认搬运 16384 个 FP32，输入 tile 65536 B，加一块 256 B 的归约/输出 tile，峰值 UB 65792 B，单槽且无跨核共享状态。M 是运行期维度，N 和块长决定编
译实例。
数值：第一遍对首元素的偏移求和，第二遍对最终均值的偏差平方求和；中心差绝对值至多 2^15，行绝对差总和至多 2^15*N，平方总和至多 2^30*N，不会 FP32 溢出。每 lane 与块间使用 Kahan 补偿，横向归约先产出 8 个
datablock partial 再逐项合并；溢出界本身不证明舍入误差满足容差，test.py 独立覆盖长行抵消与大偏置。合法极小输入的平方下溢远低于方差绝对误差预算。常数行相对锚点的偏移始终为零，因此恢复的均值等于原值，第二遍偏差精确为
零。
尾部与同步：输入有效掩码作用于减法/乘法，当前 ZEROING 路径把无效贡献置零；累积状态使用全 lane 掩码，保留此前寄存器块的有效和。补偿状态在一个 VF 内保存在独立寄存器，块间状态由外层标量持有；tile-group
auto_mutex 保护搬运、VF 与标量读写。最终紧凑输出经 UB→GM DMA 写回。
代价：一个 native task，两遍 GM 读取；补偿和逐 partial 标量交接有额外成本。默认块长来自完整调用比较，不作为其他归约或设备的默认值。
验证环境：PyPTO 导出行为基线 872df763f，CANN 9.2.0、BiSheng 15.0.5、Ascend950PR；运行具体检查仍需绑定当前源码身份，不能把历史版本名当作新的验收结果。
"""

from functools import lru_cache

import pypto_pro.language as pl
from pypto_pro.language import Vf as vf

FP32 = pl.DT_FP32
CHUNK = 16384
VL = 64
PARTS = 8


@pl.vector_function
def _centered_sum(src, reduced, cols, center):
    """Reduce one bounded chunk of values after subtracting a row anchor."""
    full = vf.create_mask(pattern=pl.MaskPattern.ALL, dtype=FP32)
    center_vec = vf.full(center, full, dtype=FP32)
    accum = vf.full(0.0, full, dtype=FP32)
    compensation = vf.full(0.0, full, dtype=FP32)
    for col in pl.range(0, cols, VL):
        active = pl.min(VL, cols - col)
        mask = vf.update_mask(active, dtype=FP32)
        values = vf.load_align(src, [0, col])
        delta = vf.sub(values, center_vec, mask)
        # delta's tail lanes are zeroed by the masked subtraction. Compensate
        # each lane across register blocks, keeping every state lane live.
        adjusted = vf.sub(delta, compensation, full)
        next_accum = vf.add(accum, adjusted, full)
        rounded = vf.sub(next_accum, accum, full)
        next_compensation = vf.sub(rounded, adjusted, full)
        compensation = vf.move(next_compensation, full)
        accum = vf.move(next_accum, full)
    # Keep each 8-lane datablock sum separate. The JIT caller merges these
    # partials with scalar compensation instead of one long horizontal FP32 sum.
    partials = vf.reduce_sum(accum, full, datablock=True)
    vf.store_align(reduced, partials, full, [0, 0])


@pl.vector_function
def _centered_square_sum(src, reduced, cols, center):
    """Reduce one bounded chunk of squared deviations from the final mean."""
    full = vf.create_mask(pattern=pl.MaskPattern.ALL, dtype=FP32)
    center_vec = vf.full(center, full, dtype=FP32)
    accum = vf.full(0.0, full, dtype=FP32)
    compensation = vf.full(0.0, full, dtype=FP32)
    for col in pl.range(0, cols, VL):
        active = pl.min(VL, cols - col)
        mask = vf.update_mask(active, dtype=FP32)
        values = vf.load_align(src, [0, col])
        delta = vf.sub(values, center_vec, mask)
        square = vf.mul(delta, delta, mask)
        # square's tail lanes are zeroed by the masked multiply; compensate
        # each lane across register blocks.
        adjusted = vf.sub(square, compensation, full)
        next_accum = vf.add(accum, adjusted, full)
        rounded = vf.sub(next_accum, accum, full)
        next_compensation = vf.sub(rounded, adjusted, full)
        compensation = vf.move(next_compensation, full)
        accum = vf.move(next_accum, full)
    partials = vf.reduce_sum(accum, full, datablock=True)
    vf.store_align(reduced, partials, full, [0, 0])


@pl.vector_function
def _write_scalar(dst, value):
    all_lanes = vf.update_mask(VL, dtype=FP32)
    value_vec = vf.full(value, all_lanes, dtype=FP32)
    first = vf.update_mask(1, dtype=FP32)
    vf.store_align(dst, value_vec, first, [0, 0], dist=pl.StoreDist.FIRST_ELEMENT_B32)


def _make_kernel(cols, chunk_size):
    inv_cols = 1.0 / float(cols)

    @pl.jit(auto_mutex=True, name=f"streaming_moments_c{chunk_size}")
    def kernel(
        x: pl.Tensor[[pl.DYNAMIC, cols], FP32],
        mean: pl.Tensor[[pl.DYNAMIC], FP32],
        var: pl.Tensor[[pl.DYNAMIC], FP32],
    ):
        with pl.section_vector():
            # Per worker live UB: one 1xchunk_size FP32 input chunk and one
            # 1x64 FP32 reduction handoff (256 B), in disjoint Vec addresses.
            # One worker owns each row. It makes two streaming passes so every
            # squared deviation uses the finalized mean. Constant rows retain an
            # exact zero deviation from their first sample.
            src_group = pl.make_tile_group(
                type=pl.TileType(
                    shape=[1, chunk_size],
                    dtype=FP32,
                    target_memory=pl.MemorySpace.Vec,
                ),
                addrs=[0],
                mutex_ids=[0],
            )
            sum_group = pl.make_tile_group(
                type=pl.TileType(
                    shape=[1, VL],
                    dtype=FP32,
                    target_memory=pl.MemorySpace.Vec,
                ),
                addrs=[chunk_size * 4],
                mutex_ids=[1],
            )
            for row in pl.range(pl.get_block_idx(), x.shape[0], pl.get_block_num()):
                anchor = pl.getval(x, row * cols)
                delta_total = 0.0
                delta_compensation = 0.0
                for col in pl.range(0, cols, chunk_size):
                    valid = pl.min(chunk_size, cols - col)
                    src = src_group.next()
                    pl.set_validshape(src, [1, valid])
                    pl.load(src, x, [row, col])
                    partial = sum_group.next()
                    _centered_sum(src, partial, valid, anchor)
                    for part in pl.range(PARTS):
                        chunk_sum = pl.getval(partial, part)
                        adjusted = chunk_sum - delta_compensation
                        next_total = delta_total + adjusted
                        delta_compensation = (next_total - delta_total) - adjusted
                        delta_total = next_total

                row_mean = anchor + delta_total * inv_cols
                mean_tile = sum_group.next()
                _write_scalar(mean_tile, row_mean)
                pl.set_validshape(mean_tile, [1, 1])
                pl.store(mean, mean_tile, [row])

                square_total = 0.0
                square_compensation = 0.0
                for col in pl.range(0, cols, chunk_size):
                    valid = pl.min(chunk_size, cols - col)
                    src = src_group.next()
                    pl.set_validshape(src, [1, valid])
                    pl.load(src, x, [row, col])
                    partial = sum_group.next()
                    _centered_square_sum(src, partial, valid, row_mean)
                    for part in pl.range(PARTS):
                        chunk_square_sum = pl.getval(partial, part)
                        adjusted_square = chunk_square_sum - square_compensation
                        next_square_total = square_total + adjusted_square
                        square_compensation = (
                            next_square_total - square_total
                        ) - adjusted_square
                        square_total = next_square_total
                variance = square_total * inv_cols
                var_tile = sum_group.next()
                _write_scalar(var_tile, variance)
                pl.set_validshape(var_tile, [1, 1])
                pl.store(var, var_tile, [row])

    return kernel


@lru_cache(maxsize=None)
def _kernel_for(cols, chunk_size):
    return _make_kernel(int(cols), int(chunk_size))


def _run_with_chunk(x, mean, var, chunk_size, workers=1):
    """Internal, metadata-only launcher used to compare bounded chunk sizes."""
    if x.ndim != 2 or mean.ndim != 1 or var.ndim != 1:
        raise ValueError("expected x[M,N], mean[M], var[M]")
    rows, cols = x.shape
    if rows < 1 or cols < 1 or mean.shape[0] != rows or var.shape[0] != rows:
        raise ValueError("incompatible or empty tensor shapes")
    if workers < 1:
        raise ValueError("workers must be positive")
    if chunk_size < VL or chunk_size % VL:
        raise ValueError("chunk_size must be a positive multiple of 64")
    return _kernel_for(cols, chunk_size)[None, workers](x, mean, var)


def run(x, mean, var, workers=1):
    """Compute row-wise FP32 mean and population variance on an NPU."""
    return _run_with_chunk(x, mean, var, CHUNK, workers=workers)
