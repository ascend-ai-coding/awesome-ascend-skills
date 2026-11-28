# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

"""写法：混合。VF 分组 INT4 反量化，Cube 使用 pl.gemm 并融合 bias。

契约：a[M,K] 为 FP16；packed[K/2,N] 为 UINT8，其低/高 nibble 分别
代表偶/奇 K 位置的无符号值；scale[K/G,N] 为 FP32，zero[K/G,N] 为
INT32，bias[N] 与 out[M,N] 为 FP32。先按 FP32 计算 (q-zero)*scale，
RNE 到 FP16 得到 W，再计算 a@W+bias；不能把这次舍入移到 GEMM 出口。
所有张量连续。M=1..129，K=32..512 且为32倍数，N=16..1040 且为16
倍数，G 为32或64并整除K。a/bias 在[-1,1]、scale在[2^-8,2^-3]、
zero在[0,15]。一般结果 atol=rtol=2e-3，one-hot 且 bias=0 精确取回 W。

VF 每次搬入一行 packed 与对应分组元数据，先 UNPK4 到 b32 容器并建立
INT32 视图，最后用 b32 predicate 与 PACK_B32 写两行 FP16。独立原生
GEMM 消费 GM 中间权重；两个任务沿调用流依次执行。L0 分片宽度随实际
K 容量收窄，M/N 尾部使用有效维度。Host 只验证元数据、分配空间和启动。

所有组显式指定地址、depth=1 与自动 mutex ID。反量化的 Vec packed/scale/zero/weight 从 0、padded_n、5*padded_n、9*padded_n 开始，共 13*padded_n 字节。
GEMM 的 Mat A/weights/bias 从 0、2*block_m*k、2*k*(block_m+GEMM_N) 开始；Bias、Left、Right、Acc 各自从 0 开始。
独立参考和测试覆盖量化顺序、舍入边界、分组/偶奇位置、短于分片的形状、抵消输入、多个 worker、完整重写、输入不变和输出守卫。
"""

from functools import lru_cache

import torch
import pypto_pro.language as pl
from pypto_pro.language import Vf as vf  # noqa: N813

FP16 = pl.DT_FP16
FP32 = pl.DT_FP32
INT32 = pl.DT_INT32
UINT8 = pl.DT_UINT8
VECTOR_LANES = 64
GEMM_M = 64
GEMM_N = 64
GEMM_K = 64


@lru_cache(maxsize=None)
def _make_dequant_kernel(k: int, n: int, group_size: int):
    packed_rows = k // 2
    groups = k // group_size
    padded_n = ((n + VECTOR_LANES - 1) // VECTOR_LANES) * VECTOR_LANES
    packed_pair_group = group_size // 2

    @pl.vector_function
    def _unpack_dequant(packed_tile, scale_tile, zero_tile, weight_tile):
        for col in pl.range(0, n, VECTOR_LANES):
            active = pl.min(VECTOR_LANES, n - col)
            p_i32 = vf.update_mask(active, dtype=INT32)
            p_f32 = vf.update_mask(active, dtype=FP32)
            raw_u8 = vf.load_align(packed_tile, col, dist=pl.LoadDist.UNPK4)
            raw_i32 = vf.bit_cast(raw_u8, dtype=INT32)
            nibble_mask = vf.full(15, p_i32, dtype=INT32)
            even_i32 = vf.and_(raw_i32, nibble_mask, p_i32)
            shifted_i32 = vf.shift_right(raw_i32, pl.const(4, INT32), p_i32)
            odd_i32 = vf.and_(shifted_i32, nibble_mask, p_i32)
            even_f32 = vf.astype(even_i32, p_i32, dtype=FP32)
            odd_f32 = vf.astype(odd_i32, p_i32, dtype=FP32)
            zero_i32 = vf.load_align(zero_tile, col)
            zero_f32 = vf.astype(zero_i32, p_i32, dtype=FP32)
            scale_f32 = vf.load_align(scale_tile, col)
            even_delta = vf.sub(even_f32, zero_f32, p_f32)
            odd_delta = vf.sub(odd_f32, zero_f32, p_f32)
            even_scaled = vf.mul(even_delta, scale_f32, p_f32)
            odd_scaled = vf.mul(odd_delta, scale_f32, p_f32)
            even_fp16 = vf.astype(
                even_scaled,
                p_f32,
                dtype=FP16,
                layout=pl.CastLayout.ZERO,
                round_mode=pl.VFRoundMode.CAST_RINT,
            )
            odd_fp16 = vf.astype(
                odd_scaled,
                p_f32,
                dtype=FP16,
                layout=pl.CastLayout.ZERO,
                round_mode=pl.VFRoundMode.CAST_RINT,
            )
            vf.store_align(
                weight_tile, even_fp16, p_f32, [0, col], dist=pl.StoreDist.PACK_B32
            )
            vf.store_align(
                weight_tile, odd_fp16, p_f32, [1, col], dist=pl.StoreDist.PACK_B32
            )

    @pl.jit(auto_mutex=True, name=f"int4_dequant_{k}_{n}_{group_size}")
    def kernel(
        packed: pl.Tensor[[packed_rows, n], UINT8],
        scale: pl.Tensor[[groups, n], FP32],
        zero: pl.Tensor[[groups, n], INT32],
        weights: pl.Tensor[[k, n], FP16],
    ):
        packed_group = pl.make_tile_group(
            type=pl.TileType(
                shape=[1, padded_n], dtype=UINT8, target_memory=pl.MemorySpace.Vec
            ),
            addrs=0,
            mutex_ids="auto",
            depth=1,
        )
        scale_group = pl.make_tile_group(
            type=pl.TileType(
                shape=[1, padded_n], dtype=FP32, target_memory=pl.MemorySpace.Vec
            ),
            addrs=padded_n,
            mutex_ids="auto",
            depth=1,
        )
        zero_group = pl.make_tile_group(
            type=pl.TileType(
                shape=[1, padded_n], dtype=INT32, target_memory=pl.MemorySpace.Vec
            ),
            addrs=5 * padded_n,
            mutex_ids="auto",
            depth=1,
        )
        weight_group = pl.make_tile_group(
            type=pl.TileType(
                shape=[2, padded_n], dtype=FP16, target_memory=pl.MemorySpace.Vec
            ),
            addrs=9 * padded_n,
            mutex_ids="auto",
            depth=1,
        )

        with pl.section_vector():
            for pair in pl.range(pl.get_block_idx(), packed_rows, pl.get_block_num()):
                group = pair // packed_pair_group
                packed_tile = packed_group.next()
                scale_tile = scale_group.next()
                zero_tile = zero_group.next()
                weight_tile = weight_group.next()
                pl.set_validshape(packed_tile, [1, n])
                pl.set_validshape(scale_tile, [1, n])
                pl.set_validshape(zero_tile, [1, n])
                pl.set_validshape(weight_tile, [2, n])
                pl.load(packed_tile, packed, [pair, 0])
                pl.load(scale_tile, scale, [group, 0])
                pl.load(zero_tile, zero, [group, 0])
                _unpack_dequant(packed_tile, scale_tile, zero_tile, weight_tile)
                pl.store(weights, weight_tile, [pair * 2, 0])

    return kernel


@lru_cache(maxsize=None)
def _make_gemm_kernel(m: int, k: int, n: int, block_m: int):
    row_blocks = (m + block_m - 1) // block_m
    col_blocks = (n + GEMM_N - 1) // GEMM_N
    gemm_k = min(GEMM_K, k)

    @pl.jit(auto_mutex=True, name=f"int4_gemm_{m}_{k}_{n}_bm{block_m}")
    def kernel(
        a: pl.Tensor[[m, k], FP16],
        weights: pl.Tensor[[k, n], FP16],
        bias: pl.Tensor[[1, n], FP32],
        out: pl.Tensor[[m, n], FP32],
    ):
        a_l1 = pl.make_tile_group(
            type=pl.TileType(
                shape=[block_m, k], dtype=FP16, target_memory=pl.MemorySpace.Mat
            ),
            addrs=0,
            mutex_ids="auto",
            depth=1,
        )
        weights_l1 = pl.make_tile_group(
            type=pl.TileType(
                shape=[k, GEMM_N], dtype=FP16, target_memory=pl.MemorySpace.Mat
            ),
            addrs=2 * block_m * k,
            mutex_ids="auto",
            depth=1,
        )
        bias_l1 = pl.make_tile_group(
            type=pl.TileType(
                shape=[1, GEMM_N], dtype=FP32, target_memory=pl.MemorySpace.Mat
            ),
            addrs=2 * k * (block_m + GEMM_N),
            mutex_ids="auto",
            depth=1,
        )
        bias_l0 = pl.make_tile_group(
            type=pl.TileType(
                shape=[1, GEMM_N], dtype=FP32, target_memory=pl.MemorySpace.Bias
            ),
            addrs=0,
            mutex_ids="auto",
            depth=1,
        )
        l0a = pl.make_tile_group(
            type=pl.TileType(
                shape=[block_m, gemm_k],
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
                shape=[gemm_k, GEMM_N],
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
                shape=[block_m, GEMM_N],
                dtype=FP32,
                target_memory=pl.MemorySpace.Acc,
                compact=1,
            ),
            addrs=0,
            mutex_ids="auto",
            depth=1,
        )

        with pl.section_cube():
            total_blocks = row_blocks * col_blocks
            for task in pl.range(pl.get_block_idx(), total_blocks, pl.get_block_num()):
                row_block = task // col_blocks
                col_block = task % col_blocks
                row = row_block * block_m
                col = col_block * GEMM_N
                active_m = pl.min(block_m, m - row)
                active_n = pl.min(GEMM_N, n - col)

                a_tile = a_l1.next()
                weights_tile = weights_l1.next()
                bias_tile = bias_l1.next()
                bias_cube = bias_l0.next()
                acc_tile = acc.next()
                pl.set_validshape(a_tile, [active_m, k])
                pl.set_validshape(weights_tile, [k, active_n])
                pl.set_validshape(bias_tile, [1, active_n])
                pl.load(a_tile, a, [row, 0])
                pl.load(weights_tile, weights, [0, col])
                pl.load(bias_tile, bias, [0, col])
                pl.move(bias_cube, bias_tile)
                pl.gemm(
                    acc_tile,
                    a_tile,
                    weights_tile,
                    l0a=l0a,
                    l0b=l0b,
                    splitk=gemm_k,
                    m=active_m,
                    n=active_n,
                    bias=bias_cube,
                )
                pl.store(out, acc_tile, [row, col])

    return kernel


def _run_impl(a, packed, scale, zero, bias, out, group_size, workers, block_m):
    m, k = (int(dim) for dim in a.shape)
    n = int(out.shape[1])
    weights = torch.empty((k, n), dtype=torch.float16, device=a.device)
    _make_dequant_kernel(k, n, group_size)[None, workers](packed, scale, zero, weights)
    _make_gemm_kernel(m, k, n, block_m)[None, workers](a, weights, bias.view(1, n), out)


def run(a, packed, scale, zero, bias, out, group_size, workers=1):
    """Launch grouped unsigned INT4 GEMM; Host reads tensor metadata only."""
    if type(group_size) is not int or group_size not in (32, 64):
        raise ValueError("group_size must be 32 or 64")
    if type(workers) is not int or workers < 1:
        raise ValueError("workers must be a positive integer")
    if a.ndim != 2 or packed.ndim != 2 or scale.ndim != 2 or zero.ndim != 2:
        raise ValueError("a, packed, scale, and zero must be rank-2 tensors")
    if bias.ndim != 1 or out.ndim != 2:
        raise ValueError("bias must be rank 1 and out must be rank 2")
    m, k = (int(dim) for dim in a.shape)
    n = int(out.shape[1])
    if not (1 <= m <= 129 and 32 <= k <= 512 and k % 32 == 0):
        raise ValueError("expected 1 <= M <= 129 and K in 32..512, divisible by 32")
    if not (16 <= n <= 1040 and n % 16 == 0 and k % group_size == 0):
        raise ValueError(
            "expected N in 16..1040, divisible by 16, and K divisible by G"
        )
    if tuple(packed.shape) != (k // 2, n):
        raise ValueError("packed must have shape [K/2,N]")
    if tuple(scale.shape) != (k // group_size, n):
        raise ValueError("scale must have shape [K/G,N]")
    if tuple(zero.shape) != (k // group_size, n):
        raise ValueError("zero must have shape [K/G,N]")
    if tuple(bias.shape) != (n,) or tuple(out.shape) != (m, n):
        raise ValueError("bias must be [N] and out must be [M,N]")
    expected_dtypes = (
        (a, torch.float16),
        (packed, torch.uint8),
        (scale, torch.float32),
        (zero, torch.int32),
        (bias, torch.float32),
        (out, torch.float32),
    )
    if any(tensor.dtype != dtype for tensor, dtype in expected_dtypes):
        raise TypeError(
            "expected a=FP16, packed=UINT8, scale/bias/out=FP32, zero=INT32"
        )
    if any(not tensor.is_contiguous() for tensor, _ in expected_dtypes):
        raise ValueError("all input and output tensors must be contiguous")
    if any(tensor.device != a.device for tensor, _ in expected_dtypes):
        raise ValueError("all tensors must be on the same device")
    _run_impl(a, packed, scale, zero, bias, out, group_size, workers, GEMM_M)
