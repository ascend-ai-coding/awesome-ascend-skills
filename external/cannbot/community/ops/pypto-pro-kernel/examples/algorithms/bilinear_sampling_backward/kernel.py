# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

"""写法：原始。零扩展双线性采样反向，SIMT 散射与独占部分和。

连续 FP32 X[B,H,W,C]、grid[B,P,2]、grad_out[B,P,C]，grid 的最后轴
为像素坐标 (x,y)。对 floor 选中的四邻点分别零扩展；整数坐标处使用该
单元的右侧导数，不能沿用前向整数点的直接拷贝捷径。输出 dX 与 dGrid
是 sum(sample(X,grid)*grad_out) 的梯度，每次调用完整重新定义输出。

支持 B=1..4、H=1..129、W=1..193、C=1..129、P=1..4097；X 与
上游梯度在[-1,1]，有限坐标 x 在[-2W,2W]、y 在[-2H,2H]。
两个输出对独立 FP64 参考固定 atol=rtol=2e-4；四点全越界的采样点
其 dGrid 精确为零且不贡献 dX。浮点原子结果逐次验数，不要求跨调用位同。

三个同 Stream 原生任务依次清零 dX、按记录/通道组产生散射与部分和、
合并写出 dGrid。dX 保留真正冲突的 FP32 原子更新，dGrid 的每个 lane
只普通写一次 partial[B,P,G,2]，再由唯一写者合并。无有效通道的 lane
也写零；额外工作区占 B*P*G*2*4 字节。Host 只读元数据和分配工作区。

G=32/T=128 是这份机制示例的配置。相同 worker 的完整调用实测中，
大形状从协作归约获益，小中形状可因额外阶段与配置选择而变慢；换形状
应保留直接归约对照，并在新分工下重测线程块/组宽，不能沿用旧分工结论。
验证身份：行为基线 872df763f，CANN 9.2.0，Ascend950PR，torch_npu
2.10.0。独立测试覆盖联合最大尺寸、整数边界、负分数坐标、集中热点、
混合越界点、多个 worker、毒化重跑、输入不变和偏移输出守卫。
"""

from functools import lru_cache

import torch
import pypto_pro.language as pl

F32 = pl.DT_FP32
I32 = pl.DT_INT32
THREADS = 128
GROUP = 32


@lru_cache(maxsize=None)
def _program(bsz, height, width, channels, points):
    dx_count = bsz * height * width * channels
    record_count = bsz * points

    @pl.vector_function(mode="simt", max_threads=THREADS)
    def _initialize(dx):
        tid = pl.simt.linear_thread_idx()
        start = pl.simt.block_idx().x * THREADS + tid
        stride = pl.simt.grid_dim().x * pl.simt.block_dim().x
        zero = pl.simt.cast(0.0, F32)
        for flat in pl.range(start, dx_count, stride):
            batch = flat // (height * width * channels)
            rem = flat % (height * width * channels)
            row = rem // (width * channels)
            rem = rem % (width * channels)
            col = rem // channels
            channel = rem % channels
            dx[batch, row, col, channel] = zero

    @pl.vector_function(mode="simt", max_threads=THREADS)
    def _scatter_and_partials(x, grid, grad_out, dx, partial):
        tid = pl.simt.linear_thread_idx()
        lane = tid % GROUP
        groups_per_block = pl.simt.block_dim().x // GROUP
        group_in_block = tid // GROUP
        record_start = pl.simt.block_idx().x * groups_per_block + group_in_block
        record_stride = pl.simt.grid_dim().x * groups_per_block
        one = pl.simt.cast(1.0, F32)
        zero = pl.simt.cast(0.0, F32)
        one_i32 = pl.simt.cast(1, I32)

        for record in pl.range(record_start, record_count, record_stride):
            batch = record // points
            point = record % points
            gx = grid[batch, point, 0]
            gy = grid[batch, point, 1]
            floor_x = pl.simt.floor(gx)
            floor_y = pl.simt.floor(gy)
            x0 = pl.simt.cast(floor_x, I32)
            y0 = pl.simt.cast(floor_y, I32)
            x1 = x0 + one_i32
            y1 = y0 + one_i32
            fx = gx - floor_x
            fy = gy - floor_y
            wx0 = one - fx
            wy0 = one - fy

            local_gx = zero
            local_gy = zero
            for channel in pl.range(lane, channels, GROUP):
                grad = grad_out[batch, point, channel]
                v00 = zero
                v01 = zero
                v10 = zero
                v11 = zero
                if y0 >= 0:
                    if y0 < height:
                        if x0 >= 0:
                            if x0 < width:
                                v00 = x[batch, y0, x0, channel]
                        if x1 >= 0:
                            if x1 < width:
                                v01 = x[batch, y0, x1, channel]
                if y1 >= 0:
                    if y1 < height:
                        if x0 >= 0:
                            if x0 < width:
                                v10 = x[batch, y1, x0, channel]
                        if x1 >= 0:
                            if x1 < width:
                                v11 = x[batch, y1, x1, channel]

                if y0 >= 0:
                    if y0 < height:
                        if x0 >= 0:
                            if x0 < width:
                                pl.simt.atomic_add(
                                    dx[batch, y0, x0, channel], grad * wx0 * wy0
                                )
                        if x1 >= 0:
                            if x1 < width:
                                pl.simt.atomic_add(
                                    dx[batch, y0, x1, channel], grad * fx * wy0
                                )
                if y1 >= 0:
                    if y1 < height:
                        if x0 >= 0:
                            if x0 < width:
                                pl.simt.atomic_add(
                                    dx[batch, y1, x0, channel], grad * wx0 * fy
                                )
                        if x1 >= 0:
                            if x1 < width:
                                pl.simt.atomic_add(
                                    dx[batch, y1, x1, channel], grad * fx * fy
                                )

                local_gx = local_gx + grad * ((v01 - v00) * wy0 + (v11 - v10) * fy)
                local_gy = local_gy + grad * ((v10 - v00) * wx0 + (v11 - v01) * fx)

            partial[batch, point, lane, 0] = local_gx
            partial[batch, point, lane, 1] = local_gy

    @pl.vector_function(mode="simt", max_threads=THREADS)
    def _merge_partials(partial, dgrid):
        tid = pl.simt.linear_thread_idx()
        start = pl.simt.block_idx().x * THREADS + tid
        stride = pl.simt.grid_dim().x * pl.simt.block_dim().x
        zero = pl.simt.cast(0.0, F32)
        for flat in pl.range(start, record_count * 2, stride):
            record = flat // 2
            axis = flat % 2
            batch = record // points
            point = record % points
            total = zero
            for lane in pl.range(GROUP):
                total = total + partial[batch, point, lane, axis]
            dgrid[batch, point, axis] = total

    @pl.jit()
    def initialize(dx: pl.Tensor[[bsz, height, width, channels], F32]):
        with pl.section_vector():
            _initialize[THREADS](dx)

    @pl.jit()
    def accumulate(
        x: pl.Tensor[[bsz, height, width, channels], F32],
        grid: pl.Tensor[[bsz, points, 2], F32],
        grad_out: pl.Tensor[[bsz, points, channels], F32],
        dx: pl.Tensor[[bsz, height, width, channels], F32],
        partial: pl.Tensor[[bsz, points, GROUP, 2], F32],
    ):
        with pl.section_vector():
            _scatter_and_partials[THREADS](x, grid, grad_out, dx, partial)

    @pl.jit()
    def reduce(
        partial: pl.Tensor[[bsz, points, GROUP, 2], F32],
        dgrid: pl.Tensor[[bsz, points, 2], F32],
    ):
        with pl.section_vector():
            _merge_partials[THREADS](partial, dgrid)

    return initialize, accumulate, reduce


def run(x, grid, grad_out, dx, dgrid, workers=1):
    """检查形状元数据并按顺序 launch 初始化、散射与归约任务。"""
    bsz, height, width, channels = x.shape
    points = grid.shape[1]
    if grid.shape != (bsz, points, 2) or grad_out.shape != (bsz, points, channels):
        raise ValueError("x, grid, and grad_out shapes do not match the contract")
    if dx.shape != (bsz, height, width, channels) or dgrid.shape != (bsz, points, 2):
        raise ValueError("dx or dgrid shape does not match the contract")
    if workers < 1:
        raise ValueError("workers must be positive")
    partial = torch.empty((bsz, points, GROUP, 2), dtype=torch.float32, device=x.device)
    initialize, accumulate, reduce = _program(bsz, height, width, channels, points)
    initialize[None, workers](dx)
    accumulate[None, workers](x, grid, grad_out, dx, partial)
    reduce[None, workers](partial, dgrid)
