# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

"""写法：原始。原生 VF LayerNorm backward，两个归约方向分别分配输出所有权。

契约：连续 FP32 x/dy[M,N]、gamma[N]、mean/rstd[M]，输出 dx[M,N]、dgamma/dbeta[N]。
严格使用传入的 FP32 统计量，不在 kernel 或 Host 重算前向统计量；Host 仅做形状视图与调度。
dx 按行分工，参数梯度按列块分工。后者一次搬入二维块，在 VF 内逐行累加独立 lane，统计量用 BRC_B32 广播。
这是归约轴驻留版：参数梯度 UB 占 2*M*64*4 + 2*ceil(M/64)*64*4 + 2*64*4 字节；扩大形状前应检查容量或改用行块流式归约。
验证边界：行为版本 872df763f、CANN 9.2.0、Ascend950PR；同目录测试覆盖传入统计量、不同 worker、重复调用及输入/输出守卫。
"""

from functools import lru_cache

import pypto_pro.language as pl
from pypto_pro.language import Vf as vf

FP32 = pl.DT_FP32
VL = 64


@pl.vector_function
def _dx_row(
    x_tile, dy_tile, gamma_tile, dx_tile, cols, inv_cols, mean_value, rstd_value
):
    all_lanes = vf.create_mask(pattern=pl.MaskPattern.ALL, dtype=FP32)
    mean_broadcast = vf.full(mean_value, all_lanes)
    sum_u = vf.full(0.0, all_lanes)
    sum_uxhat = vf.full(0.0, all_lanes)

    for col in pl.range(0, cols, VL):
        active = pl.min(VL, cols - col)
        mask = vf.update_mask(active, dtype=FP32)
        xv = vf.load_align(x_tile, [0, col])
        dyv = vf.load_align(dy_tile, [0, col])
        gamma = vf.load_align(gamma_tile, [0, col])
        centered = vf.sub(xv, mean_broadcast, mask)
        xhat = vf.muls(centered, rstd_value, mask)
        u = vf.mul(dyv, gamma, mask)
        uxhat = vf.mul(u, xhat, mask)
        block_u = vf.reduce_sum(u, mask)
        block_uxhat = vf.reduce_sum(uxhat, mask)
        sum_u = vf.add(sum_u, block_u, all_lanes)
        sum_uxhat = vf.add(sum_uxhat, block_uxhat, all_lanes)

    mean_u = vf.muls(sum_u, inv_cols, all_lanes)
    mean_uxhat = vf.muls(sum_uxhat, inv_cols, all_lanes)
    mean_u_broadcast = vf.full(mean_u, all_lanes)
    mean_uxhat_broadcast = vf.full(mean_uxhat, all_lanes)
    rstd_broadcast = vf.full(rstd_value, all_lanes)

    for col in pl.range(0, cols, VL):
        active = pl.min(VL, cols - col)
        mask = vf.update_mask(active, dtype=FP32)
        xv = vf.load_align(x_tile, [0, col])
        dyv = vf.load_align(dy_tile, [0, col])
        gamma = vf.load_align(gamma_tile, [0, col])
        centered = vf.sub(xv, mean_broadcast, mask)
        xhat = vf.muls(centered, rstd_value, mask)
        u = vf.mul(dyv, gamma, mask)
        correction = vf.mul(xhat, mean_uxhat_broadcast, mask)
        residual = vf.sub(u, mean_u_broadcast, mask)
        residual = vf.sub(residual, correction, mask)
        dxv = vf.mul(rstd_broadcast, residual, mask)
        vf.store_align(dx_tile, dxv, mask, [0, col])


@lru_cache(maxsize=None)
def _make_dx_kernel(rows, cols):
    padded_cols = ((cols + VL - 1) // VL) * VL
    tile_bytes = padded_cols * 4
    inv_cols = 1.0 / cols

    @pl.jit(auto_mutex=True)
    def dx_kernel(
        x: pl.Tensor[[rows, cols], FP32],
        dy: pl.Tensor[[rows, cols], FP32],
        gamma: pl.Tensor[[1, cols], FP32],
        mean: pl.Tensor[[rows], FP32],
        rstd: pl.Tensor[[rows], FP32],
        dx: pl.Tensor[[rows, cols], FP32],
    ):
        with pl.section_vector():
            x_group = pl.make_tile_group(
                type=pl.TileType(
                    shape=[1, padded_cols],
                    dtype=FP32,
                    target_memory=pl.MemorySpace.Vec,
                    valid_shape=[-1, -1],
                ),
                addrs=[0],
                mutex_ids=[0],
            )
            dy_group = pl.make_tile_group(
                type=pl.TileType(
                    shape=[1, padded_cols],
                    dtype=FP32,
                    target_memory=pl.MemorySpace.Vec,
                    valid_shape=[-1, -1],
                ),
                addrs=[tile_bytes],
                mutex_ids=[1],
            )
            gamma_group = pl.make_tile_group(
                type=pl.TileType(
                    shape=[1, padded_cols],
                    dtype=FP32,
                    target_memory=pl.MemorySpace.Vec,
                    valid_shape=[-1, -1],
                ),
                addrs=[2 * tile_bytes],
                mutex_ids=[2],
            )
            dx_group = pl.make_tile_group(
                type=pl.TileType(
                    shape=[1, padded_cols],
                    dtype=FP32,
                    target_memory=pl.MemorySpace.Vec,
                    valid_shape=[-1, -1],
                ),
                addrs=[3 * tile_bytes],
                mutex_ids=[3],
            )

            for row in pl.range(pl.get_block_idx(), rows, pl.get_block_num()):
                x_tile = x_group.next()
                dy_tile = dy_group.next()
                gamma_tile = gamma_group.next()
                dx_tile = dx_group.next()
                pl.set_validshape(x_tile, [1, cols])
                pl.set_validshape(dy_tile, [1, cols])
                pl.set_validshape(gamma_tile, [1, cols])
                pl.set_validshape(dx_tile, [1, cols])
                pl.load(x_tile, x, [row, 0])
                pl.load(dy_tile, dy, [row, 0])
                pl.load(gamma_tile, gamma, [0, 0])
                mean_value = pl.getval(mean, row)
                rstd_value = pl.getval(rstd, row)
                _dx_row(
                    x_tile,
                    dy_tile,
                    gamma_tile,
                    dx_tile,
                    cols,
                    inv_cols,
                    mean_value,
                    rstd_value,
                )
                pl.store(dx, dx_tile, [row, 0])

    return dx_kernel


@pl.vector_function
def _column_gradients(
    x_tile, dy_tile, mean_tile, rstd_tile, dgamma_tile, dbeta_tile, rows, active_cols
):
    mask = vf.update_mask(active_cols, dtype=FP32)
    sum_dgamma = vf.full(0.0, mask)
    sum_dbeta = vf.full(0.0, mask)

    for row in pl.range(rows):
        xv = vf.load_align(x_tile, [row, 0])
        dyv = vf.load_align(dy_tile, [row, 0])
        mean_value = vf.load_align(mean_tile, row, dist=pl.LoadDist.BRC_B32)
        rstd_value = vf.load_align(rstd_tile, row, dist=pl.LoadDist.BRC_B32)
        centered = vf.sub(xv, mean_value, mask)
        xhat = vf.mul(centered, rstd_value, mask)
        contribution = vf.mul(dyv, xhat, mask)
        sum_dgamma = vf.add(sum_dgamma, contribution, mask)
        sum_dbeta = vf.add(sum_dbeta, dyv, mask)

    vf.store_align(dgamma_tile, sum_dgamma, mask, [0, 0])
    vf.store_align(dbeta_tile, sum_dbeta, mask, [0, 0])


@lru_cache(maxsize=None)
def _make_column_kernel(rows, cols):
    chunks = (cols + VL - 1) // VL
    padded_rows = ((rows + VL - 1) // VL) * VL
    input_tile_bytes = rows * VL * 4
    stats_tile_bytes = padded_rows * 4
    out_tile_bytes = VL * 4
    mean_addr = 2 * input_tile_bytes
    rstd_addr = mean_addr + stats_tile_bytes
    dgamma_addr = rstd_addr + stats_tile_bytes
    dbeta_addr = dgamma_addr + out_tile_bytes

    @pl.jit(auto_mutex=True)
    def column_kernel(
        x: pl.Tensor[[rows, cols], FP32],
        dy: pl.Tensor[[rows, cols], FP32],
        mean: pl.Tensor[[1, rows], FP32],
        rstd: pl.Tensor[[1, rows], FP32],
        dgamma: pl.Tensor[[1, cols], FP32],
        dbeta: pl.Tensor[[1, cols], FP32],
    ):
        with pl.section_vector():
            x_group = pl.make_tile_group(
                type=pl.TileType(
                    shape=[rows, VL],
                    dtype=FP32,
                    target_memory=pl.MemorySpace.Vec,
                    valid_shape=[-1, -1],
                ),
                addrs=[0],
                mutex_ids=[0],
            )
            dy_group = pl.make_tile_group(
                type=pl.TileType(
                    shape=[rows, VL],
                    dtype=FP32,
                    target_memory=pl.MemorySpace.Vec,
                    valid_shape=[-1, -1],
                ),
                addrs=[input_tile_bytes],
                mutex_ids=[1],
            )
            mean_group = pl.make_tile_group(
                type=pl.TileType(
                    shape=[1, padded_rows],
                    dtype=FP32,
                    target_memory=pl.MemorySpace.Vec,
                    valid_shape=[-1, -1],
                ),
                addrs=[mean_addr],
                mutex_ids=[2],
            )
            rstd_group = pl.make_tile_group(
                type=pl.TileType(
                    shape=[1, padded_rows],
                    dtype=FP32,
                    target_memory=pl.MemorySpace.Vec,
                    valid_shape=[-1, -1],
                ),
                addrs=[rstd_addr],
                mutex_ids=[3],
            )
            dgamma_group = pl.make_tile_group(
                type=pl.TileType(
                    shape=[1, VL],
                    dtype=FP32,
                    target_memory=pl.MemorySpace.Vec,
                    valid_shape=[-1, -1],
                ),
                addrs=[dgamma_addr],
                mutex_ids=[4],
            )
            dbeta_group = pl.make_tile_group(
                type=pl.TileType(
                    shape=[1, VL],
                    dtype=FP32,
                    target_memory=pl.MemorySpace.Vec,
                    valid_shape=[-1, -1],
                ),
                addrs=[dbeta_addr],
                mutex_ids=[5],
            )

            mean_tile = mean_group.current()
            rstd_tile = rstd_group.current()
            pl.set_validshape(mean_tile, [1, rows])
            pl.set_validshape(rstd_tile, [1, rows])
            pl.load(mean_tile, mean, [0, 0])
            pl.load(rstd_tile, rstd, [0, 0])

            for chunk in pl.range(pl.get_block_idx(), chunks, pl.get_block_num()):
                col = chunk * VL
                active_cols = pl.min(VL, cols - col)
                x_tile = x_group.next()
                dy_tile = dy_group.next()
                dgamma_tile = dgamma_group.next()
                dbeta_tile = dbeta_group.next()
                pl.set_validshape(x_tile, [rows, active_cols])
                pl.set_validshape(dy_tile, [rows, active_cols])
                pl.set_validshape(dgamma_tile, [1, active_cols])
                pl.set_validshape(dbeta_tile, [1, active_cols])
                pl.load(x_tile, x, [0, col])
                pl.load(dy_tile, dy, [0, col])
                _column_gradients(
                    x_tile,
                    dy_tile,
                    mean_tile,
                    rstd_tile,
                    dgamma_tile,
                    dbeta_tile,
                    rows,
                    active_cols,
                )
                pl.store(dgamma, dgamma_tile, [0, col])
                pl.store(dbeta, dbeta_tile, [0, col])

    return column_kernel


def run(x, dy, gamma, mean, rstd, dx, dgamma, dbeta, workers=1):
    """Launch LayerNorm backward; tensors are contiguous FP32 NPU tensors."""
    rows, cols = x.shape
    gamma_2d = gamma.view(1, cols)
    mean_2d = mean.view(1, rows)
    rstd_2d = rstd.view(1, rows)
    dgamma_2d = dgamma.view(1, cols)
    dbeta_2d = dbeta.view(1, cols)

    _make_dx_kernel(rows, cols)[None, workers](
        x,
        dy,
        gamma_2d,
        mean,
        rstd,
        dx,
    )
    _make_column_kernel(rows, cols)[None, workers](
        x,
        dy,
        mean_2d,
        rstd_2d,
        dgamma_2d,
        dbeta_2d,
    )
