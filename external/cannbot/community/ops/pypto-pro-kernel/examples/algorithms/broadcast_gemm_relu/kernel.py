# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

"""写法：语法糖。广播 GEMM 的 bias 初始化与 fixpipe ReLU。

契约：连续 FP16 A[B,M,K]、W[1,K,N]，连续 FP32 bias[N] 与 out[B,M,N]。
每个 Cube worker 处理独立 batch；W 与 bias 每核仅搬运一次，bias 走 Mat→Bias 通路。
每次 gemm 从 bias 初始化累加器，Acc→GM 的 NormalRelu 完成后处理，无额外输出回读。
单 batch 的完整 M/K/N 驻留片上，M 补齐至16并用 m extent 收窄；扩大尺寸前重新检查 L1/L0 容量。
显式 Mat 地址依次放置 A、W、bias，分别占 2*block_m*k、2*k*n、4*n 字节；Bias、Left、Right、Acc 各自从 0 开始。
所有组为单槽，核内 mutex ID 自动分配；没有跨核交接。
test.py 覆盖 B=2/3/5、M=17/19/33、K=32/64/96、N=32/48/80，FP64 参考基于实际 FP16 输入，atol=rtol=3e-4；不覆盖未对齐 K/N。
"""

from functools import lru_cache

import torch
import pypto_pro.language as pl


FP16 = pl.DT_FP16
FP32 = pl.DT_FP32


@lru_cache(maxsize=None)
def _kernel_for(batch: int, m: int, k: int, n: int):
    """Build one statically shaped kernel; batch is distributed across workers."""
    block_m = ((m + 15) // 16) * 16

    @pl.jit(auto_mutex=True, name=f"broadcast_gemm_relu_{batch}_{m}_{k}_{n}")
    def kernel(
        a: pl.Tensor[[batch, m, k], pl.DT_FP16],
        w: pl.Tensor[[1, k, n], pl.DT_FP16],
        bias: pl.Tensor[[1, n], pl.DT_FP32],
        out: pl.Tensor[[batch, m, n], pl.DT_FP32],
    ):
        a_l1 = pl.make_tile_group(
            type=pl.TileType(
                shape=[block_m, k], dtype=FP16, target_memory=pl.MemorySpace.Mat
            ),
            addrs=0,
            mutex_ids="auto",
            depth=1,
        )
        w_l1 = pl.make_tile_group(
            type=pl.TileType(
                shape=[k, n], dtype=FP16, target_memory=pl.MemorySpace.Mat
            ),
            addrs=2 * block_m * k,
            mutex_ids="auto",
            depth=1,
        )
        bias_l1 = pl.make_tile_group(
            type=pl.TileType(
                shape=[1, n], dtype=FP32, target_memory=pl.MemorySpace.Mat
            ),
            addrs=2 * block_m * k + 2 * k * n,
            mutex_ids="auto",
            depth=1,
        )
        bias_l0 = pl.make_tile_group(
            type=pl.TileType(
                shape=[1, n], dtype=FP32, target_memory=pl.MemorySpace.Bias
            ),
            addrs=0,
            mutex_ids="auto",
            depth=1,
        )
        l0a = pl.make_tile_group(
            type=pl.TileType(
                shape=[block_m, k],
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
                shape=[k, n], dtype=FP16, target_memory=pl.MemorySpace.Right
            ),
            addrs=0,
            mutex_ids="auto",
            depth=1,
        )
        acc = pl.make_tile_group(
            type=pl.TileType(
                shape=[block_m, n],
                dtype=FP32,
                target_memory=pl.MemorySpace.Acc,
                compact=1,
            ),
            addrs=0,
            mutex_ids="auto",
            depth=1,
        )

        with pl.section_cube():
            w_tile = w_l1.current()
            pl.load(w_tile, w, [0, 0, 0])
            bias_tile_l1 = bias_l1.current()
            pl.load(bias_tile_l1, bias, [0, 0])
            bias_tile_l0 = bias_l0.current()
            pl.move(bias_tile_l0, bias_tile_l1)

            a_tile = a_l1.current()
            acc_tile = acc.current()
            for batch_idx in pl.range(pl.get_block_idx(), batch, pl.get_block_num()):
                pl.set_validshape(a_tile, [m, k])
                pl.load(a_tile, a, [batch_idx, 0, 0])
                pl.gemm(
                    acc_tile, a_tile, w_tile, l0a=l0a, l0b=l0b, m=m, bias=bias_tile_l0
                )
                pl.store(
                    out,
                    acc_tile,
                    [batch_idx, 0, 0],
                    relu_pre_mode=pl.ReluPreMode.NormalRelu,
                )

    return kernel


def run(a, w, bias, out, workers=1):
    """Enqueue ``relu(A @ W + bias)`` on the NPU; the Host only launches kernels."""
    if type(workers) is not int or workers < 1:
        raise ValueError("workers must be a positive integer")
    if a.ndim != 3 or w.ndim != 3 or bias.ndim != 1 or out.ndim != 3:
        raise ValueError("expected a[B,M,K], w[1,K,N], bias[N], out[B,M,N]")

    batch, m, k = (int(dim) for dim in a.shape)
    if tuple(w.shape) != (1, k, int(out.shape[2])):
        raise ValueError("w must have shape [1,K,N]")
    n = int(out.shape[2])
    if tuple(out.shape) != (batch, m, n) or tuple(bias.shape) != (n,):
        raise ValueError("out must be [B,M,N] and bias must be [N]")
    if (
        a.dtype != torch.float16
        or w.dtype != torch.float16
        or bias.dtype != torch.float32
        or out.dtype != torch.float32
    ):
        raise TypeError("expected FP16 a/w and FP32 bias/out")
    if not out.is_contiguous():
        raise ValueError("out must be contiguous")
    if not (a.device == w.device == bias.device == out.device):
        raise ValueError("a, w, bias, and out must be on the same device")

    # A leading singleton dimension is a metadata-only view for a contiguous bias.
    _kernel_for(batch, m, k, n)[None, workers](a, w, bias.view(1, n), out)
