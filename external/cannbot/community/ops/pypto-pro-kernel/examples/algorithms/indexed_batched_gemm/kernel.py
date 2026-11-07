# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

"""写法：语法糖。Stage an indexed matrix operand, then consume it with Cube.

C[b, m, j] = dot(X[b, m, :], table[b, indices[b, j], :]).
X[B,M,K] and table[B,R,K] are contiguous FP16/BF16; indices[B,N] is INT32
with values in [0,R), including repeats. C[B,M,N] is FP32. M is positive,
K is a multiple of 64 in [64,1024], and N is a positive multiple of 128.
The caller supplies a disjoint contiguous panel[B,N,K] with the input dtype.
Table and panel bases are 8-byte aligned; integer transport preserves bytes.

All M rows in a batch share its selector. B identifies the selector/operand
group, not necessarily an application batch. Distinct selectors need distinct
groups; mapping a task's original coordinates to these axes is caller work.

The two native launches run on the same stream: complete the panel before
reading it as a dense operand. No on-chip cross-core handover is involved.
Gather owns (batch, selected row, transport word); GEMM owns (batch, M tile,
N tile). Their grids are independent, and both use the actual worker count.
Per Cube worker: Mat=2*K*(64+128) bytes; Left/Right/Acc=8/16/32 KiB.
Mat X starts at 0 and panel at 128*K bytes; Left/Right/Acc each start at 0.
All five groups have depth=1 and automatic mutex IDs; addresses are explicit.
M tails retain physical allocations and narrow compact Left/Acc views.
Panel scratch is 2*B*N*K bytes. This demonstrates an interface, not a tuned
configuration or a performance guarantee for every indexed contraction.
"""

import pypto_pro.language as pl


def make_kernels(
    batch, rows, table_rows, selected_rows, features, dtype_name="float16"
):
    if min(batch, rows, table_rows, selected_rows) < 1:
        raise ValueError("dimensions must be positive")
    if not 64 <= features <= 1024 or features % 64 or selected_rows % 128:
        raise ValueError("K must be a multiple of 64 in [64,1024]; N % 128 == 0")
    if dtype_name not in ("float16", "bfloat16"):
        raise ValueError("dtype_name must be float16 or bfloat16")
    dtype = pl.DT_FP16 if dtype_name == "float16" else pl.DT_BF16
    words, threads = features // 4, 128
    m_tiles, n_tiles = (rows + 63) // 64, selected_rows // 128

    @pl.vector_function(mode="simt", max_threads=threads)
    def copy_words(src, indices, dst):
        start = pl.simt.block_idx().x * threads + pl.simt.linear_thread_idx()
        step = pl.simt.grid_dim().x * pl.simt.block_dim().x
        for linear in pl.range(start, batch * selected_rows * words, step):
            b = linear // (selected_rows * words)
            j = (linear // words) % selected_rows
            k = linear % words
            source_row = indices[b, j]
            dst[b, j, k] = src[b, source_row, k]

    @pl.jit(arch="a5")
    def gather(
        table: pl.Tensor[[batch, table_rows, features], dtype],
        indices: pl.Tensor[[batch, selected_rows], pl.DT_INT32],
        panel: pl.Tensor[[batch, selected_rows, features], dtype],
    ):
        src = pl.make_tensor(table, [batch, table_rows, words], dtype=pl.DT_INT64)
        dst = pl.make_tensor(panel, [batch, selected_rows, words], dtype=pl.DT_INT64)
        with pl.section_vector():
            copy_words[threads](src, indices, dst)

    @pl.jit(arch="a5", auto_mutex=True)
    def product(
        x: pl.Tensor[[batch, rows, features], dtype],
        panel: pl.Tensor[[batch, selected_rows, features], dtype],
        out: pl.Tensor[[batch, rows, selected_rows], pl.DT_FP32],
    ):
        x_g = pl.make_tile_group(
            type=pl.TileType(
                shape=[64, features], dtype=dtype, target_memory=pl.MemorySpace.Mat
            ),
            addrs=0,
            mutex_ids="auto",
            depth=1,
        )
        panel_g = pl.make_tile_group(
            type=pl.TileType(
                shape=[128, features], dtype=dtype, target_memory=pl.MemorySpace.Mat
            ),
            addrs=128 * features,
            mutex_ids="auto",
            depth=1,
        )
        left = pl.make_tile_group(
            type=pl.TileType(
                shape=[64, 64],
                dtype=dtype,
                target_memory=pl.MemorySpace.Left,
                compact=1,
            ),
            addrs=0,
            mutex_ids="auto",
            depth=1,
        )
        right = pl.make_tile_group(
            type=pl.TileType(
                shape=[64, 128], dtype=dtype, target_memory=pl.MemorySpace.Right
            ),
            addrs=0,
            mutex_ids="auto",
            depth=1,
        )
        acc_g = pl.make_tile_group(
            type=pl.TileType(
                shape=[64, 128],
                dtype=pl.DT_FP32,
                target_memory=pl.MemorySpace.Acc,
                compact=1,
            ),
            addrs=0,
            mutex_ids="auto",
            depth=1,
        )
        with pl.section_cube():
            for task in pl.range(
                pl.get_block_idx(), batch * m_tiles * n_tiles, pl.get_block_num()
            ):
                b = task // (m_tiles * n_tiles)
                m0 = ((task // n_tiles) % m_tiles) * 64
                n0 = (task % n_tiles) * 128
                m_valid = pl.min(64, rows - m0)
                a = x_g.current()
                p = panel_g.current()
                acc = acc_g.current()
                pl.set_validshape(a, [m_valid, features])
                pl.set_validshape(acc, [m_valid, 128])
                pl.load(a, x, [b, m0, 0], order=[1, 2])
                pl.load(p, panel, [b, n0, 0], order=[1, 2])
                pl.gemm(
                    acc, a, p, l0a=left, l0b=right, m=m_valid, trans_b=True, splitk=64
                )
                pl.store(out, acc, [b, m0, n0], order=[1, 2])

    return gather, product
