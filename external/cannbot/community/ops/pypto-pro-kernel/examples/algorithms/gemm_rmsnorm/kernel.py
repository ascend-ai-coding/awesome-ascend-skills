# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

"""写法：语法糖（``pl.gemm`` + VF，两次原生 kernel launch）。

连续 FP16 矩阵先做 GEMM，FP32 bias 相加后做整行 RMSNorm 并乘 FP32 gamma。
Cube 按有界 M/N 块将 FP32 乘积写到 out；Vector 分批读取多行，在 VF 内归约并原地覆写。
两个任务的同 Stream 顺序保护中间结果，核内 TileGroup 使用显式地址、单槽与自动 mutex ID。
Host 只处理元数据和启动；输入不变，out 在整次调用完成后才是最终结果。
GEMM 的 Mat A 从 0 开始，B 紧随 A 的 2*gemm_m*k 字节；Left、Right、Acc 各自从 0 开始。
归一化的 Vec out/bias/gamma 顺序排布，分别占 4*NORM_ROWS*padded_n、4*padded_n、4*padded_n 字节。
物理列数补至 64 的倍数，有效行列控制搬运和 VF 掩码；M 块选择保持在 32/64 两档。
test.py 使用固定 atol=rtol=2e-3，覆盖同时取域上界、抵消、epsilon 边界、毒化输出和前后守卫。
"""

from functools import lru_cache

import pypto_pro.language as pl
from pypto_pro.language import Vf as vf  # noqa: N813


FP16 = pl.DT_FP16
FP32 = pl.DT_FP32
GEMM_M_BASE = 32
GEMM_M_LARGE = 64
GEMM_N = 64
NORM_ROWS = 8
VF_LANES = 64


def _padded_cols(cols):
    return ((cols + VF_LANES - 1) // VF_LANES) * VF_LANES


def _gemm_m_tile(m, k, n):
    if m >= 128 and k >= 512 and n >= 1024:
        return GEMM_M_LARGE
    return GEMM_M_BASE


@lru_cache(maxsize=None)
def _kernels(m, k, n, gemm_m, gemm_n):
    padded_n = _padded_cols(n)
    reciprocal_n = 1.0 / n

    @pl.vector_function
    def _rmsnorm_rows(tile, bias_tile, gamma_tile, row_count, epsilon, inv_n):
        all_lanes = vf.create_mask(pattern=pl.MaskPattern.ALL, dtype=FP32)
        for row in pl.range(row_count):
            first_active = pl.min(VF_LANES, n)
            first_mask = vf.update_mask(first_active, dtype=FP32)
            first_value = vf.load_align(tile, [row, 0])
            first_bias = vf.load_align(bias_tile, 0)
            first_t = vf.add(first_value, first_bias, first_mask)
            first_square = vf.mul(first_t, first_t, first_mask)
            total = vf.reduce_sum(first_square, first_mask)

            for col in pl.range(VF_LANES, n, VF_LANES):
                active = pl.min(VF_LANES, n - col)
                mask = vf.update_mask(active, dtype=FP32)
                value = vf.load_align(tile, [row, col])
                bias_value = vf.load_align(bias_tile, col)
                t_value = vf.add(value, bias_value, mask)
                square = vf.mul(t_value, t_value, mask)
                partial = vf.reduce_sum(square, mask)
                total = vf.add(total, partial, all_lanes)

            mean_square = vf.muls(total, inv_n, all_lanes)
            mean_square_eps = vf.adds(mean_square, epsilon, all_lanes)
            rms = vf.sqrt(mean_square_eps, all_lanes)
            rms_broadcast = vf.full(rms, all_lanes)

            for col in pl.range(0, n, VF_LANES):
                active = pl.min(VF_LANES, n - col)
                mask = vf.update_mask(active, dtype=FP32)
                value = vf.load_align(tile, [row, col])
                bias_value = vf.load_align(bias_tile, col)
                gamma_value = vf.load_align(gamma_tile, col)
                t_value = vf.add(value, bias_value, mask)
                normalized = vf.div(t_value, rms_broadcast, mask)
                scaled = vf.mul(normalized, gamma_value, mask)
                vf.store_align(tile, scaled, mask, [row, col])

    @pl.jit(auto_mutex=True)
    def _gemm_kernel(
        a: pl.Tensor[[m, k], FP16],
        b: pl.Tensor[[k, n], FP16],
        out: pl.Tensor[[m, n], FP32],
    ):
        a_l1 = pl.make_tile_group(
            type=pl.TileType(
                shape=[gemm_m, k], dtype=FP16, target_memory=pl.MemorySpace.Mat
            ),
            addrs=0,
            mutex_ids="auto",
            depth=1,
        )
        b_l1 = pl.make_tile_group(
            type=pl.TileType(
                shape=[k, gemm_n], dtype=FP16, target_memory=pl.MemorySpace.Mat
            ),
            addrs=2 * gemm_m * k,
            mutex_ids="auto",
            depth=1,
        )
        l0a = pl.make_tile_group(
            type=pl.TileType(
                shape=[gemm_m, k],
                dtype=FP16,
                target_memory=pl.MemorySpace.Left,
                compact=1,
            ),
            addrs=0,
            mutex_ids="auto",
            depth=1,
        )
        l0b = pl.make_tile_group(
            type=pl.TileType(
                shape=[k, gemm_n],
                dtype=FP16,
                target_memory=pl.MemorySpace.Right,
                compact=1,
            ),
            addrs=0,
            mutex_ids="auto",
            depth=1,
        )
        acc = pl.make_tile_group(
            type=pl.TileType(
                shape=[gemm_m, gemm_n],
                dtype=FP32,
                target_memory=pl.MemorySpace.Acc,
                compact=1,
            ),
            addrs=0,
            mutex_ids="auto",
            depth=1,
        )

        with pl.section_cube():
            for row_base in pl.range(
                pl.get_block_idx() * gemm_m,
                m,
                pl.get_block_num() * gemm_m,
            ):
                active_m = pl.min(gemm_m, m - row_base)
                for col_base in pl.range(0, n, gemm_n):
                    active_n = pl.min(gemm_n, n - col_base)
                    cur_a = a_l1.current()
                    cur_b = b_l1.current()
                    cur_acc = acc.current()
                    pl.set_validshape(cur_a, [active_m, k])
                    pl.load(cur_a, a, [row_base, 0])
                    pl.set_validshape(cur_b, [k, active_n])
                    pl.load(cur_b, b, [0, col_base])
                    pl.gemm(
                        cur_acc,
                        cur_a,
                        cur_b,
                        l0a=l0a,
                        l0b=l0b,
                        m=active_m,
                        n=active_n,
                    )
                    pl.store(out, cur_acc, [row_base, col_base])

    @pl.jit(auto_mutex=True)
    def _norm_kernel(
        bias: pl.Tensor[[1, n], FP32],
        gamma: pl.Tensor[[1, n], FP32],
        out: pl.Tensor[[m, n], FP32],
        epsilon: pl.DT_FP32,
    ):
        out_tile_group = pl.make_tile_group(
            type=pl.TileType(
                shape=[NORM_ROWS, padded_n],
                dtype=FP32,
                target_memory=pl.MemorySpace.Vec,
            ),
            addrs=0,
            mutex_ids="auto",
            depth=1,
        )
        bias_tile_group = pl.make_tile_group(
            type=pl.TileType(
                shape=[1, padded_n], dtype=FP32, target_memory=pl.MemorySpace.Vec
            ),
            addrs=4 * NORM_ROWS * padded_n,
            mutex_ids="auto",
            depth=1,
        )
        gamma_tile_group = pl.make_tile_group(
            type=pl.TileType(
                shape=[1, padded_n], dtype=FP32, target_memory=pl.MemorySpace.Vec
            ),
            addrs=4 * (NORM_ROWS + 1) * padded_n,
            mutex_ids="auto",
            depth=1,
        )

        with pl.section_vector():
            bias_tile = bias_tile_group.current()
            gamma_tile = gamma_tile_group.current()
            pl.set_validshape(bias_tile, [1, n])
            pl.load(bias_tile, bias, [0, 0])
            pl.set_validshape(gamma_tile, [1, n])
            pl.load(gamma_tile, gamma, [0, 0])

            for row_base in pl.range(
                pl.get_block_idx() * NORM_ROWS,
                m,
                pl.get_block_num() * NORM_ROWS,
            ):
                active_rows = pl.min(NORM_ROWS, m - row_base)
                out_tile = out_tile_group.current()
                pl.set_validshape(out_tile, [active_rows, n])
                pl.load(out_tile, out, [row_base, 0])
                _rmsnorm_rows(
                    out_tile, bias_tile, gamma_tile, active_rows, epsilon, reciprocal_n
                )
                pl.store(out, out_tile, [row_base, 0])

    return _gemm_kernel, _norm_kernel


def _launch_with_tile(a, b, bias, gamma, out, epsilon, workers, gemm_m, gemm_n):
    m, k = a.shape
    _, n = b.shape
    gemm_kernel, norm_kernel = _kernels(m, k, n, gemm_m, gemm_n)
    gemm_kernel[None, workers](a, b, out)
    norm_kernel[None, workers](
        bias.reshape(1, n), gamma.reshape(1, n), out, float(epsilon)
    )


def run(a, b, bias, gamma, out, epsilon, workers=1):
    """在声明的形状与 dtype 契约下顺序启动 GEMM 和行归一化。"""
    m, k = a.shape
    b_k, n = b.shape
    if b_k != k or bias.shape != (n,) or gamma.shape != (n,) or out.shape != (m, n):
        raise ValueError("incompatible GEMM/RMSNorm tensor shapes")
    if not (1 <= m <= 257 and 16 <= k <= 512 and k % 16 == 0):
        raise ValueError("supported shapes require 1<=M<=257 and K in 16..512 by 16")
    if not (16 <= n <= 2048 and n % 16 == 0):
        raise ValueError("supported shapes require N in 16..2048 by 16")
    if not (1.0e-5 <= float(epsilon) <= 1.0e-2):
        raise ValueError("epsilon must be in [1e-5, 1e-2]")
    if type(workers) is not int or workers < 1:
        raise ValueError("workers must be a positive integer")

    _launch_with_tile(
        a,
        b,
        bias,
        gamma,
        out,
        epsilon,
        workers,
        _gemm_m_tile(m, k, n),
        GEMM_N,
    )
