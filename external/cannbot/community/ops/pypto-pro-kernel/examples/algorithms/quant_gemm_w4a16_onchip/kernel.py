# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

"""写法：混合。VF 分块反量化并交接片上 L1，Cube 用 pl.gemm 融合 bias。

契约：a[M,K] 为 FP16；packed[K/2,N] 为 UINT8，低/高 nibble 分别表示偶/奇 K 的无符号值；scale[K/G,N] 为 FP32，zero[K/G,N] 为 INT32，bias[N] 与
out[M,N] 为 FP32。先按 FP32 计算 (q-zero)*scale，RNE 到 FP16 得到 W，再计算 a@W+bias。所有张量连续、输入输出互不重叠，输入保持不变。

M=1..129，K=64..2048 且为64倍数，N=16..512 且为16倍数，G=32或64。a/bias 在[-1,1]，scale在[2^-8,2^-3]，zero在[0,15]。对实际舍入后的 W 计算 FP64 参考，误差
不超过 2e-3+3e-4*sum(abs(a*W))；one-hot 且 bias=0 时精确取回 W。

一次原生调用、零辅助 GM：每个 AIV 反量化有界 K 片段，ND→NZ 后插入当前 L1 N 条带；自动跨核 stage 完成交接，preload=0、共享 W 单槽。Cube 以 K=64 分片，并在首片融合 bias。完整浮点权重不
落 GM。Host 只检查元数据和启动。

所有组使用显式地址和自动 mutex ID。Mat W 从 0 开始，A 从 2*k*GEMM_N 开始，bias 紧随 A；W 用 fwd_ids=[0]、bwd_ids=[1] 保护单槽交接。最大 K 下 Mat 共320.25
KiB，包含 A 64 KiB、W 256 KiB、bias 256 B。L0A 为2 KiB/槽、L0B 为8 KiB/槽，各两槽；Acc 为4 KiB。Bias、Left、Right、Acc 各自从 0 开始。
Vec 按 packed、scale、zero、ND weight、NZ weight 的顺序紧邻放置，最大19 KiB：packed 2 KiB、scale/zero 各512 B、ND/NZ 临时块各8 KiB。

当前只沿 M 分配 worker，保留 W 条带以复用于该 worker 的后续 M tile。薄 M 不能用满请求的 worker，多 M owner 会重复反量化；这是一种复用/并行度取舍，不是完整 M/N 网格的最优分配。
workers 仅支持1/4/16。

测试覆盖分组和偶奇位置、FP16 RNE 中点、K/N/M 尾部、抵消、不同 worker、重复毒值初始化及输入输出守卫。
"""

from functools import lru_cache

import pypto_pro.language as pl
from pypto_pro.language import Vf as vf  # noqa: N813


FP16 = pl.DT_FP16
FP32 = pl.DT_FP32
INT32 = pl.DT_INT32
UINT8 = pl.DT_UINT8
VECTOR_LANES = 64
GEMM_M = 16
GEMM_N = 64
GEMM_K = 64
DEQUANT_K = 128


@pl.vector_function
def _unpack_dequant(
    packed_tile,
    scale_tile,
    zero_tile,
    weight_tile,
    pair_count,
    group_size,
    k_start,
    group_base,
    active_n,
):
    """Decode packed nibble pairs and RNE the dequantized values into ND Vec."""
    for pair in pl.range(pair_count):
        local_group = (k_start + pair * 2) // group_size - group_base
        for col in pl.range(0, GEMM_N, VECTOR_LANES):
            active = pl.min(VECTOR_LANES, active_n - col)
            p_i32 = vf.update_mask(active, dtype=INT32)
            p_f32 = vf.update_mask(active, dtype=FP32)
            raw_u8 = vf.load_align(packed_tile, [pair, col], dist=pl.LoadDist.UNPK4)
            raw_i32 = vf.bit_cast(raw_u8, dtype=INT32)
            nibble_mask = vf.full(15, p_i32, dtype=INT32)
            even_i32 = vf.and_(raw_i32, nibble_mask, p_i32)
            shifted_i32 = vf.shift_right(raw_i32, pl.const(4, INT32), p_i32)
            odd_i32 = vf.and_(shifted_i32, nibble_mask, p_i32)
            even_f32 = vf.astype(even_i32, p_i32, dtype=FP32)
            odd_f32 = vf.astype(odd_i32, p_i32, dtype=FP32)
            zero_i32 = vf.load_align(zero_tile, [local_group, col])
            zero_f32 = vf.astype(zero_i32, p_i32, dtype=FP32)
            scale_f32 = vf.load_align(scale_tile, [local_group, col])
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
                weight_tile,
                even_fp16,
                p_f32,
                [pair * 2, col],
                dist=pl.StoreDist.PACK_B32,
            )
            vf.store_align(
                weight_tile,
                odd_fp16,
                p_f32,
                [pair * 2 + 1, col],
                dist=pl.StoreDist.PACK_B32,
            )


@pl.pipeline.stage
def _convert_strip(
    strip,
    k,
    n,
    group_size,
    dequant_k,
    packed,
    scale,
    zero,
    packed_group,
    scale_group,
    zero_group,
    weight_group,
    weight_nz_group,
    shared_w,
):
    """Each AIV inserts its K half for each bounded chunk into the shared L1 tile."""
    sub = pl.get_subblock_idx()
    col = strip * GEMM_N
    active_n = pl.min(GEMM_N, n - col)
    shared = shared_w.next()
    pair_count = dequant_k // 4
    group_count = (dequant_k // 2 + group_size - 1) // group_size
    for k_block in pl.range(k // dequant_k):
        k_start = k_block * dequant_k + sub * (dequant_k // 2)
        group_base = k_start // group_size
        packed_tile = packed_group.next()
        scale_tile = scale_group.next()
        zero_tile = zero_group.next()
        weight_tile = weight_group.next()
        weight_nz = weight_nz_group.next()
        pl.set_validshape(packed_tile, [pair_count, active_n])
        pl.set_validshape(scale_tile, [group_count, active_n])
        pl.set_validshape(zero_tile, [group_count, active_n])
        pl.set_validshape(weight_tile, [dequant_k // 2, active_n])
        pl.set_validshape(weight_nz, [dequant_k // 2, active_n])
        pl.load(packed_tile, packed, [k_start // 2, col])
        pl.load(scale_tile, scale, [group_base, col])
        pl.load(zero_tile, zero, [group_base, col])
        _unpack_dequant(
            packed_tile,
            scale_tile,
            zero_tile,
            weight_tile,
            pair_count,
            group_size,
            k_start,
            group_base,
            active_n,
        )
        pl.move(weight_nz, weight_tile)
        pl.insert(shared, weight_nz, [k_block * dequant_k + sub * (dequant_k // 2), 0])


@pl.pipeline.stage
def _gemm_strip(
    strip,
    m,
    k,
    workers,
    n,
    bias,
    a,
    out,
    a_group,
    bias_mat_group,
    bias_cube_group,
    l0a,
    l0b,
    acc_group,
    shared_w,
):
    """Reuse one on-chip W strip across this worker's row tiles."""
    col = strip * GEMM_N
    active_n = pl.min(GEMM_N, n - col)
    w_tile = shared_w.next()
    bias_mat = bias_mat_group.next()
    bias_cube = bias_cube_group.next()
    pl.set_validshape(bias_mat, [1, active_n])
    pl.load(bias_mat, bias, [0, col])
    pl.move(bias_cube, bias_mat)
    core = pl.get_block_idx() // pl.get_subblock_num()
    row_tiles = (m + GEMM_M - 1) // GEMM_M
    for row_block in pl.range(core, row_tiles, workers):
        row = row_block * GEMM_M
        active_m = pl.min(GEMM_M, m - row)
        a_tile = a_group.next()
        acc_tile = acc_group.next()
        pl.set_validshape(a_tile, [active_m, k])
        pl.load(a_tile, a, [row, 0])
        pl.gemm(
            acc_tile,
            a_tile,
            w_tile,
            l0a=l0a,
            l0b=l0b,
            splitk=GEMM_K,
            m=active_m,
            n=active_n,
            bias=bias_cube,
        )
        pl.store(out, acc_tile, [row, col])


@lru_cache(maxsize=None)
def _make_kernel(m: int, k: int, n: int, group_size: int, workers: int):
    dequant_k = DEQUANT_K if k % DEQUANT_K == 0 else 64
    pair_count = dequant_k // 4
    group_count = (dequant_k // 2 + group_size - 1) // group_size
    row_tiles = (m + GEMM_M - 1) // GEMM_M
    launch_workers = min(workers, row_tiles)
    padded_n = GEMM_N
    col_blocks = (n + GEMM_N - 1) // GEMM_N

    @pl.jit(
        auto_mutex=True,
        name=f"w4a16_bounded64_m{m}_k{k}_n{n}_g{group_size}_w{workers}",
        pipeline=pl.pipeline.PipelineConfig(preload=0),
    )
    def kernel(
        a: pl.Tensor[[m, k], FP16],
        packed: pl.Tensor[[k // 2, n], UINT8],
        scale: pl.Tensor[[k // group_size, n], FP32],
        zero: pl.Tensor[[k // group_size, n], INT32],
        bias: pl.Tensor[[1, n], FP32],
        out: pl.Tensor[[m, n], FP32],
    ):
        packed_group = pl.make_tile_group(
            type=pl.TileType(
                shape=[pair_count, padded_n],
                dtype=UINT8,
                target_memory=pl.MemorySpace.Vec,
            ),
            addrs=0,
            mutex_ids="auto",
            depth=1,
        )
        scale_group = pl.make_tile_group(
            type=pl.TileType(
                shape=[group_count, padded_n],
                dtype=FP32,
                target_memory=pl.MemorySpace.Vec,
            ),
            addrs=pair_count * padded_n,
            mutex_ids="auto",
            depth=1,
        )
        zero_group = pl.make_tile_group(
            type=pl.TileType(
                shape=[group_count, padded_n],
                dtype=INT32,
                target_memory=pl.MemorySpace.Vec,
            ),
            addrs=(pair_count + 4 * group_count) * padded_n,
            mutex_ids="auto",
            depth=1,
        )
        weight_group = pl.make_tile_group(
            type=pl.TileType(
                shape=[dequant_k // 2, GEMM_N],
                dtype=FP16,
                target_memory=pl.MemorySpace.Vec,
            ),
            addrs=(pair_count + 8 * group_count) * padded_n,
            mutex_ids="auto",
            depth=1,
        )
        weight_nz_group = pl.make_tile_group(
            type=pl.TileType(
                shape=[dequant_k // 2, GEMM_N],
                dtype=FP16,
                target_memory=pl.MemorySpace.Vec,
                layout=pl.NZ,
            ),
            addrs=(pair_count + 8 * group_count + dequant_k) * padded_n,
            mutex_ids="auto",
            depth=1,
        )
        shared_w = pl.make_tile_group(
            type=pl.TileType(
                shape=[k, GEMM_N],
                dtype=FP16,
                target_memory=pl.MemorySpace.Mat,
                layout=pl.NZ,
            ),
            addrs=0,
            mutex_ids="auto",
            depth=1,
            fwd_ids=[0],
            bwd_ids=[1],
        )
        a_group = pl.make_tile_group(
            type=pl.TileType(
                shape=[GEMM_M, k], dtype=FP16, target_memory=pl.MemorySpace.Mat
            ),
            addrs=2 * k * GEMM_N,
            mutex_ids="auto",
            depth=1,
        )
        bias_mat_group = pl.make_tile_group(
            type=pl.TileType(
                shape=[1, GEMM_N], dtype=FP32, target_memory=pl.MemorySpace.Mat
            ),
            addrs=2 * k * (GEMM_N + GEMM_M),
            mutex_ids="auto",
            depth=1,
        )
        bias_cube_group = pl.make_tile_group(
            type=pl.TileType(
                shape=[1, GEMM_N], dtype=FP32, target_memory=pl.MemorySpace.Bias
            ),
            addrs=0,
            mutex_ids="auto",
            depth=1,
        )
        l0a = pl.make_tile_group(
            type=pl.TileType(
                shape=[GEMM_M, GEMM_K],
                dtype=FP16,
                target_memory=pl.MemorySpace.Left,
                compact=1,
            ),
            addrs=0,
            mutex_ids="auto",
            depth=2,
        )
        l0b = pl.make_tile_group(
            type=pl.TileType(
                shape=[GEMM_K, GEMM_N],
                dtype=FP16,
                target_memory=pl.MemorySpace.Right,
                compact=1,
            ),
            addrs=0,
            mutex_ids="auto",
            depth=2,
        )
        acc_group = pl.make_tile_group(
            type=pl.TileType(
                shape=[GEMM_M, GEMM_N],
                dtype=FP32,
                target_memory=pl.MemorySpace.Acc,
                compact=1,
            ),
            addrs=0,
            mutex_ids="auto",
            depth=1,
        )

        for strip in pl.range(col_blocks):
            with pl.section_vector():
                _convert_strip(
                    strip,
                    k,
                    n,
                    group_size,
                    dequant_k,
                    packed,
                    scale,
                    zero,
                    packed_group,
                    scale_group,
                    zero_group,
                    weight_group,
                    weight_nz_group,
                    shared_w,
                )
            with pl.section_cube():
                _gemm_strip(
                    strip,
                    m,
                    k,
                    launch_workers,
                    n,
                    bias,
                    a,
                    out,
                    a_group,
                    bias_mat_group,
                    bias_cube_group,
                    l0a,
                    l0b,
                    acc_group,
                    shared_w,
                )

    return kernel, launch_workers


def run(a, packed, scale, zero, bias, out, g, workers=4):
    """Run the grouped W4A16 GEMM; the Host uses tensor metadata and launches."""
    if type(g) is not int or g not in (32, 64):
        raise ValueError("g must be 32 or 64")
    if type(workers) is not int or workers not in (1, 4, 16):
        raise ValueError("workers must be 1, 4, or 16")
    if any(t.ndim != 2 for t in (a, packed, scale, zero, out)) or bias.ndim != 1:
        raise ValueError("a, packed, scale, zero, out must be rank 2 and bias rank 1")
    m, k = (int(dim) for dim in a.shape)
    n = int(out.shape[1])
    if not (1 <= m <= 129 and 64 <= k <= 2048 and k % 64 == 0):
        raise ValueError("expected 1 <= M <= 129 and K in 64..2048 divisible by 64")
    if not (16 <= n <= 512 and n % 16 == 0 and k % g == 0):
        raise ValueError("expected N in 16..512 divisible by 16 and K divisible by g")
    if tuple(packed.shape) != (k // 2, n):
        raise ValueError("packed must have shape [K/2,N]")
    if tuple(scale.shape) != (k // g, n) or tuple(zero.shape) != (k // g, n):
        raise ValueError("scale and zero must have shape [K/g,N]")
    if tuple(bias.shape) != (n,) or tuple(out.shape) != (m, n):
        raise ValueError("bias must be [N] and out must be [M,N]")
    kernel, launch_workers = _make_kernel(m, k, n, g, workers)
    kernel[None, launch_workers](a, packed, scale, zero, bias.view(1, n), out)
