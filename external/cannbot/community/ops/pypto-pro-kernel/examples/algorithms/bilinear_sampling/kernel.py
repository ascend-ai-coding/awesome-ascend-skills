# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

"""写法：原始。原生 SIMT 双线性采样，使用带边界判定的四邻点读取。

一组线程共享逻辑采样点，各 lane 在自己的寄存器中复用坐标、权重和有效位。
商余映射给每个输出唯一写者，输入只读，不需要线程间交换或屏障。
整数合法点直接搬运；其他坐标逐邻点零延拓，输出不依赖旧值。
默认线程配置只代表已测候选；小形状、读取有效率和局部性的性能需分别验证。
验证边界：行为版本 872df763f、CANN 9.2.0、Ascend950PR，固定 atol=rtol=2e-5。
"""

from functools import lru_cache

import pypto_pro.language as pl

FP32 = pl.DT_FP32
I32 = pl.DT_INT32

_DEFAULT_THREADS = 256
_DEFAULT_GROUP_WIDTH = 32


@lru_cache(maxsize=None)
def make_kernel(batch, height, width, channels, points, threads, group_width):
    """为一个形状及线程块/组宽组合创建独立原生 SIMT kernel。"""
    if threads < 1 or group_width < 1 or threads % group_width:
        raise ValueError("threads must be a positive multiple of group_width")

    groups_per_block = threads // group_width
    total_records = batch * points

    @pl.vector_function(mode="simt", max_threads=threads)
    def sample_body(x, grid, out):
        tid = pl.simt.linear_thread_idx()
        lane = tid % group_width
        record_start = pl.simt.block_idx().x * groups_per_block + tid // group_width
        record_step = pl.simt.grid_dim().x * groups_per_block

        zero_i32 = pl.simt.cast(0, I32)
        one_i32 = pl.simt.cast(1, I32)
        width_i32 = pl.simt.cast(width, I32)
        height_i32 = pl.simt.cast(height, I32)

        for record in pl.range(record_start, total_records, record_step):
            batch_i = record // points
            point_i = record % points
            coord_x = grid[batch_i, point_i, 0]
            coord_y = grid[batch_i, point_i, 1]
            x0 = pl.simt.cast(pl.simt.floor(coord_x), I32)
            y0 = pl.simt.cast(pl.simt.floor(coord_y), I32)
            x1 = x0 + one_i32
            y1 = y0 + one_i32
            dx = coord_x - pl.simt.cast(x0, FP32)
            dy = coord_y - pl.simt.cast(y0, FP32)

            if dx == 0.0 and dy == 0.0:
                if (
                    x0 >= zero_i32
                    and x0 < width_i32
                    and y0 >= zero_i32
                    and y0 < height_i32
                ):
                    for channel in pl.range(lane, channels, group_width):
                        out[batch_i, point_i, channel] = x[batch_i, y0, x0, channel]
                else:
                    for channel in pl.range(lane, channels, group_width):
                        out[batch_i, point_i, channel] = 0.0
            else:
                one_minus_dx = 1.0 - dx
                one_minus_dy = 1.0 - dy
                weight00 = one_minus_dx * one_minus_dy
                weight10 = dx * one_minus_dy
                weight01 = one_minus_dx * dy
                weight11 = dx * dy

                valid00 = (
                    x0 >= zero_i32
                    and x0 < width_i32
                    and y0 >= zero_i32
                    and y0 < height_i32
                )
                valid10 = (
                    x1 >= zero_i32
                    and x1 < width_i32
                    and y0 >= zero_i32
                    and y0 < height_i32
                )
                valid01 = (
                    x0 >= zero_i32
                    and x0 < width_i32
                    and y1 >= zero_i32
                    and y1 < height_i32
                )
                valid11 = (
                    x1 >= zero_i32
                    and x1 < width_i32
                    and y1 >= zero_i32
                    and y1 < height_i32
                )

                for channel in pl.range(lane, channels, group_width):
                    result = 0.0
                    if valid00:
                        result = result + x[batch_i, y0, x0, channel] * weight00
                    if valid10:
                        result = result + x[batch_i, y0, x1, channel] * weight10
                    if valid01:
                        result = result + x[batch_i, y1, x0, channel] * weight01
                    if valid11:
                        result = result + x[batch_i, y1, x1, channel] * weight11
                    out[batch_i, point_i, channel] = result

    @pl.jit()
    def kernel(
        x: pl.Tensor[[batch, height, width, channels], FP32],
        grid: pl.Tensor[[batch, points, 2], FP32],
        out: pl.Tensor[[batch, points, channels], FP32],
    ):
        with pl.section_vector():
            sample_body[threads](x, grid, out)

    return kernel


def _metadata_shape(tensor):
    return tuple(int(dim) for dim in tensor.shape)


def _validate_call(x, grid, out, workers):
    x_shape = _metadata_shape(x)
    grid_shape = _metadata_shape(grid)
    out_shape = _metadata_shape(out)
    if len(x_shape) != 4 or len(grid_shape) != 3 or len(out_shape) != 3:
        raise ValueError("expected x[B,H,W,C], grid[B,P,2], out[B,P,C]")
    batch, height, width, channels = x_shape
    if grid_shape != (batch, out_shape[1], 2) or out_shape != (
        batch,
        grid_shape[1],
        channels,
    ):
        raise ValueError("x, grid, and out shapes do not match")
    if not (
        1 <= batch <= 4
        and 1 <= height <= 129
        and 1 <= width <= 193
        and 1 <= channels <= 129
        and 1 <= grid_shape[1] <= 4097
    ):
        raise ValueError("shape is outside the supported domain")
    if not (
        str(x.dtype).endswith("float32")
        and str(grid.dtype).endswith("float32")
        and str(out.dtype).endswith("float32")
    ):
        raise TypeError("x, grid, and out must be FP32")
    if not (x.is_contiguous() and grid.is_contiguous() and out.is_contiguous()):
        raise ValueError("x, grid, and out must be contiguous")
    if type(workers) is not int or workers < 1:
        raise ValueError("workers must be a positive integer")
    return batch, height, width, channels, grid_shape[1]


def _run_config(x, grid, out, workers, threads, group_width):
    shape = _validate_call(x, grid, out, workers)
    kernel = make_kernel(*shape, threads, group_width)
    kernel[None, workers](x, grid, out)


def run(x, grid, out, workers=1):
    """使用像素坐标采样，每个越界邻点独立取零。"""
    _run_config(x, grid, out, workers, _DEFAULT_THREADS, _DEFAULT_GROUP_WIDTH)
