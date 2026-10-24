# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

"""写法：原始。VF 逐列均值与总体方差，每个 lane 对应一个保留的列。

输入为紧密 FP32 [M,N]，输出为两个 FP32 [N]；沿 M 归约，方差除以 M。
先相对首行中心化求均值，再累加相对均值的平方偏差。
每个 worker 独占一段列，单槽 tile group 自动同步，尾列使用有效形状和 VF mask。
输入 tile 占 M*64*4 字节，输出缓冲紧接其后；容量由框架检查。
当前验证环境为行为版本 872df763f、CANN 9.2.0；具体数值契约见 test.py。
"""

import pypto_pro.language as pl
from pypto_pro.language import Vf as vf

FP32 = pl.DT_FP32
VF_LANES = 64


@pl.vector_function
def _column_stats(src_tile, mean_tile, variance_tile, rows, valid_cols, inv_rows):
    """对一段列做两遍中心化计算，避免大偏置下 E[x²] - E[x]² 的消减误差。"""
    mask = vf.update_mask(valid_cols, dtype=FP32)
    center = vf.load_align(src_tile, [0, 0])
    sum_delta = vf.sub(center, center, mask)

    for row in pl.range(rows):
        value = vf.load_align(src_tile, [row, 0])
        delta = vf.sub(value, center, mask)
        sum_delta = vf.add(sum_delta, delta, mask)

    mean_delta = vf.muls(sum_delta, inv_rows, mask)
    mean = vf.add(center, mean_delta, mask)
    sum_squares = vf.sub(center, center, mask)

    for row in pl.range(rows):
        value = vf.load_align(src_tile, [row, 0])
        centered = vf.sub(value, mean, mask)
        square = vf.mul(centered, centered, mask)
        sum_squares = vf.add(sum_squares, square, mask)

    variance = vf.muls(sum_squares, inv_rows, mask)
    vf.store_align(mean_tile, mean, mask, [0, 0])
    vf.store_align(variance_tile, variance, mask, [0, 0])


def make_column_stats(rows, cols):
    """Create an FP32 [rows, cols] -> (mean[cols], variance[cols]) kernel."""
    chunks = (cols + VF_LANES - 1) // VF_LANES
    inv_rows = 1.0 / rows
    mean_addr = rows * VF_LANES * 4
    variance_addr = mean_addr + VF_LANES * 4

    @pl.jit(auto_mutex=True)
    def column_stats(
        x: pl.Tensor[[rows, cols], FP32],
        mean: pl.Tensor[[cols], FP32],
        variance: pl.Tensor[[cols], FP32],
    ):
        with pl.section_vector():
            src_group = pl.make_tile_group(
                type=pl.TileType(
                    shape=[rows, VF_LANES],
                    dtype=FP32,
                    target_memory=pl.MemorySpace.Vec,
                    valid_shape=[-1, -1],
                ),
                addrs=[0x0000],
                mutex_ids=[0],
            )
            mean_group = pl.make_tile_group(
                type=pl.TileType(
                    shape=[1, VF_LANES],
                    dtype=FP32,
                    target_memory=pl.MemorySpace.Vec,
                    valid_shape=[-1, -1],
                ),
                addrs=[mean_addr],
                mutex_ids=[1],
            )
            variance_group = pl.make_tile_group(
                type=pl.TileType(
                    shape=[1, VF_LANES],
                    dtype=FP32,
                    target_memory=pl.MemorySpace.Vec,
                    valid_shape=[-1, -1],
                ),
                addrs=[variance_addr],
                mutex_ids=[2],
            )

            for chunk in pl.range(pl.get_block_idx(), chunks, pl.get_block_num()):
                col = chunk * VF_LANES
                active_cols = pl.min(VF_LANES, cols - col)
                src_tile = src_group.next()
                mean_tile = mean_group.next()
                variance_tile = variance_group.next()
                pl.set_validshape(src_tile, [rows, active_cols])
                pl.set_validshape(mean_tile, [1, active_cols])
                pl.set_validshape(variance_tile, [1, active_cols])
                pl.load(src_tile, x, [0, col])
                _column_stats(
                    src_tile, mean_tile, variance_tile, rows, active_cols, inv_rows
                )
                pl.store(mean, mean_tile, [col])
                pl.store(variance, variance_tile, [col])

    return column_stats
