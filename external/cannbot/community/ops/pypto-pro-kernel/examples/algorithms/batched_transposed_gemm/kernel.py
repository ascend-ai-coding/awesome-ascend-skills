# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

"""写法：语法糖。三维 GM 寻址、L1 转置视图与 M 尾块的组合。

FP16 输入，FP32 输出；显式地址与自动 mutex ID，单槽五组缓冲，batch 分核，pl.gemm 收窄 M。
本例要求 N 为 16 的倍数、K 为 32 的倍数；不展示 N/K 尾块处理。
Mat A 从 0 开始，W 紧随 A 的 2*block_m*block_k 字节；Left、Right、Acc 各自从 0 开始。
容量由框架检查；运行 test.py 检查批次、尾块、不同核数和输出守卫。
无性能结论。
"""

import pypto_pro.language as pl


def make_batched_transposed_gemm(batch, m, n, k):
    """构造 C[B,M,N] = A[B,M,K] @ W[B,N,K]^T。"""
    if min(batch, m, n, k) <= 0 or n % 16 or k % 32:
        raise ValueError("positive dimensions, N % 16 == 0 and K % 32 == 0 required")
    block_m = ((m + 15) // 16) * 16
    block_n = ((n + 15) // 16) * 16
    block_k = ((k + 31) // 32) * 32
    fp16 = pl.DT_FP16
    fp32 = pl.DT_FP32

    @pl.jit(auto_mutex=True)
    def kernel(
        a: pl.Tensor[[batch, m, k], pl.DT_FP16],
        w: pl.Tensor[[batch, n, k], pl.DT_FP16],
        out: pl.Tensor[[batch, m, n], pl.DT_FP32],
    ):
        a_l1 = pl.make_tile_group(
            type=pl.TileType(
                shape=[block_m, block_k], dtype=fp16, target_memory=pl.MemorySpace.Mat
            ),
            addrs=0,
            mutex_ids="auto",
            depth=1,
        )
        w_l1 = pl.make_tile_group(
            type=pl.TileType(
                shape=[block_n, block_k], dtype=fp16, target_memory=pl.MemorySpace.Mat
            ),
            addrs=2 * block_m * block_k,
            mutex_ids="auto",
            depth=1,
        )
        l0a = pl.make_tile_group(
            type=pl.TileType(
                shape=[block_m, block_k],
                dtype=fp16,
                target_memory=pl.MemorySpace.Left,
                compact=1,
            ),
            addrs=0,
            mutex_ids="auto",
            depth=1,
        )
        l0b = pl.make_tile_group(
            type=pl.TileType(
                shape=[block_k, block_n], dtype=fp16, target_memory=pl.MemorySpace.Right
            ),
            addrs=0,
            mutex_ids="auto",
            depth=1,
        )
        acc = pl.make_tile_group(
            type=pl.TileType(
                shape=[block_m, block_n],
                dtype=fp32,
                target_memory=pl.MemorySpace.Acc,
                compact=1,
            ),
            addrs=0,
            mutex_ids="auto",
            depth=1,
        )

        with pl.section_cube():
            cur_a = a_l1.current()
            cur_w = w_l1.current()
            cur_acc = acc.current()
            for batch_idx in pl.range(pl.get_block_idx(), batch, pl.get_block_num()):
                pl.set_validshape(cur_a, [m, k])
                pl.load(cur_a, a, [batch_idx, 0, 0])
                pl.load(cur_w, w, [batch_idx, 0, 0])
                pl.gemm(cur_acc, cur_a, cur_w, l0a=l0a, l0b=l0b, m=m, trans_b=True)
                pl.store(out, cur_acc, [batch_idx, 0, 0])

    return kernel
