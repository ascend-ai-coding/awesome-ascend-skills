# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

"""写法：语法糖。SIMT gathers an indexed operand into on-chip Cube storage.

example-query: {"aliases": ["SIMT Cube fusion", "on-chip indexed contraction", "SIMT Cube 片上交接"], "facets": {"source_dtype": ["FP16", "BF16"], "destination_dtype": ["FP32"]}}

C[b,m,j] = dot(x[b,m,:], table[b,indices[b,j],:]). Inputs are contiguous,
disjoint FP16/BF16 x[B,M,K], table[B,R,K] and INT32 indices[B,N] in [0,R).
C[B,M,N] is FP32. Values are finite in [-1,1]; the table base is 8-byte
aligned. Reference and tolerance are shared with indexed_batched_gemm.
This bounded example supports B<=3, M<=129, R<=257, N in {128,256,384},
and K in 64..1024 divisible by 64. Overlap is exposed only for K<=320.

make_kernel(..., workers=W, depth=1, preload=0) is the serial control;
(depth,preload)=(2,1) uses a two-slot time-shifted pipeline. Launch the kernel
with kernel[None,W](x,table,indices,out): W is part of this factory's
mapping and must match the Host block count. Each mixed block uses one
AIC and two AIVs on the verified A5 path. The two AIVs each own 64 panel
rows. One SIMT invocation has 128 threads and moves 64-bit words into an
ND Vec tile; a low-precision alias and a real ND-to-NZ move prepare insert.

Both stages receive the declared Mat group. Cube reinterprets its bound
group parameter as ZN before selecting one slot, then uses pl.gemm without
a transpose flag. This preserves pipeline buffer tracking as well as the
backing storage and cursor; passing an untracked external view is unsafe
to assume equivalent on the verified source. The view performs no copy.
Vec uses 16 KiB plus the installed SIMT reservation. Mat uses
2*K*(128*depth+64) bytes: panel slots precede the A tile. Left/Right/Acc
use 8/16/32 KiB, each with one slot; their independent spaces start at zero.
There is no auxiliary GM panel. Each M tile gathers its operand again.

This is a composition example with explicit alternatives, not a default
replacement for the independently partitioned two-kernel implementation.
Compare complete calls at matched AIC/AIV budgets and preserve the better
path for the actual workload. Source behavior was checked at c46548e;
revalidate synchronization, resources and numerical results after upgrades.
"""

from functools import lru_cache
import pypto_pro.language as pl

TM, TN, TK, THREADS = 64, 128, 64, 128


@pl.vector_function(mode="simt", max_threads=THREADS)
def gather_chunk(table, indices, dst, b, n0, k0):
    tid = pl.simt.linear_thread_idx()
    for linear in pl.range(tid, 64 * (TK // 4), THREADS):
        row = linear // (TK // 4)
        kk = linear % (TK // 4)
        selected = indices[b, n0 + row]
        dst[row, kk] = table[b, selected, k0 // 4 + kk]


@pl.pipeline.stage
def consume(
    step,
    batch,
    workers,
    rows,
    mtiles,
    ntiles,
    x,
    out,
    a_group,
    panel_group,
    left,
    right,
    acc_group,
):
    worker = pl.get_block_idx() // pl.get_subblock_num()
    task = step * workers + worker
    panel_group = pl.reinterpret(panel_group, shape=[x.shape[2], TN], layout=pl.ZN)
    panel = panel_group.next()
    pl.set_validshape(panel, [x.shape[2], TN])
    if task < batch * mtiles * ntiles:
        b = task // (mtiles * ntiles)
        m0 = ((task // ntiles) % mtiles) * TM
        n0 = (task % ntiles) * TN
        valid_m = pl.min(TM, rows - m0)
        a = a_group.next()
        acc = acc_group.next()
        pl.set_validshape(a, [valid_m, x.shape[2]])
        pl.set_validshape(acc, [valid_m, TN])
        pl.load(a, x, [b, m0, 0], order=[1, 2])
        pl.gemm(acc, a, panel, l0a=left, l0b=right, m=valid_m, splitk=TK)
        pl.store(out, acc, [b, m0, n0], order=[1, 2])


@lru_cache(None)
def make_kernel(
    batch,
    rows,
    table_rows,
    selected_rows,
    features,
    dtype_name="float16",
    workers=1,
    depth=1,
    preload=0,
):
    if any(
        type(v) is not int or v <= 0
        for v in (batch, rows, table_rows, selected_rows, features, workers)
    ):
        raise ValueError("shape dimensions and workers must be positive integers")
    if batch > 3 or rows > 129 or table_rows > 257:
        raise ValueError("this example requires B<=3, M<=129 and R<=257")
    if (
        selected_rows not in (128, 256, 384)
        or not 64 <= features <= 1024
        or features % 64
    ):
        raise ValueError(
            "this example requires N in {128,256,384} and K in 64..1024 divisible by 64"
        )
    if dtype_name not in ("float16", "bfloat16"):
        raise ValueError("dtype_name must be float16 or bfloat16")
    if (
        type(depth) is not int
        or type(preload) is not int
        or (depth, preload) not in ((1, 0), (2, 1))
    ):
        raise ValueError("this example exposes (depth,preload)=(1,0) or (2,1)")
    if depth == 2 and features > 320:
        raise ValueError("this example's overlap option is bounded to K<=320")
    dtype = pl.DT_FP16 if dtype_name == "float16" else pl.DT_BF16
    mtiles, ntiles = (rows + TM - 1) // TM, selected_rows // TN
    forward_ids = list(range(depth))
    backward_ids = list(range(depth, 2 * depth))

    @pl.pipeline.stage
    def produce(
        step,
        batch,
        workers,
        mtiles,
        ntiles,
        features,
        packed_table,
        indices,
        nd_group,
        nz_group,
        panel_group,
    ):
        worker = pl.get_block_idx() // pl.get_subblock_num()
        task = step * workers + worker
        shared = panel_group.next()
        if task < batch * mtiles * ntiles:
            b = task // (mtiles * ntiles)
            n0 = (task % ntiles) * TN
            sub = pl.get_subblock_idx()
            for k0 in pl.range(0, features, TK):
                nd = nd_group.next()
                nz = nz_group.next()
                gather_chunk[THREADS](packed_table, indices, nd, b, n0 + sub * 64, k0)
                nd_low = pl.reinterpret(nd, dtype=dtype, shape=[64, TK])
                pl.set_validshape(nd_low, [64, TK])
                pl.move(nz, nd_low)
                pl.insert(shared, nz, [sub * 64, k0])

    @pl.jit(
        arch="a5", auto_mutex=True, pipeline=pl.pipeline.PipelineConfig(preload=preload)
    )
    def kernel(
        x: pl.Tensor[[batch, rows, features], dtype],
        table: pl.Tensor[[batch, table_rows, features], dtype],
        indices: pl.Tensor[[batch, selected_rows], pl.DT_INT32],
        out: pl.Tensor[[batch, rows, selected_rows], pl.DT_FP32],
    ):
        packed_table = pl.make_tensor(
            table, [batch, table_rows, features // 4], dtype=pl.DT_INT64
        )
        nd_group = pl.make_tile_group(
            type=pl.TileType(
                shape=[64, TK // 4], dtype=pl.DT_INT64, target_memory=pl.MemorySpace.Vec
            ),
            addrs=0,
            mutex_ids="auto",
            depth=1,
        )
        nz_group = pl.make_tile_group(
            type=pl.TileType(
                shape=[64, TK],
                dtype=dtype,
                target_memory=pl.MemorySpace.Vec,
                layout=pl.NZ,
            ),
            addrs=64 * TK * 2,
            mutex_ids="auto",
            depth=1,
        )
        panel_group = pl.make_tile_group(
            type=pl.TileType(
                shape=[TN, features],
                dtype=dtype,
                target_memory=pl.MemorySpace.Mat,
                layout=pl.NZ,
            ),
            addrs=0,
            mutex_ids="auto",
            depth=depth,
            fwd_ids=forward_ids,
            bwd_ids=backward_ids,
        )
        a_group = pl.make_tile_group(
            type=pl.TileType(
                shape=[TM, features], dtype=dtype, target_memory=pl.MemorySpace.Mat
            ),
            addrs=depth * TN * features * 2,
            mutex_ids="auto",
            depth=1,
        )
        left = pl.make_tile_group(
            type=pl.TileType(
                shape=[TM, TK],
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
                shape=[TK, TN], dtype=dtype, target_memory=pl.MemorySpace.Right
            ),
            addrs=0,
            mutex_ids="auto",
            depth=1,
        )
        acc_group = pl.make_tile_group(
            type=pl.TileType(
                shape=[TM, TN],
                dtype=pl.DT_FP32,
                target_memory=pl.MemorySpace.Acc,
                compact=1,
            ),
            addrs=0,
            mutex_ids="auto",
            depth=1,
        )
        for step in pl.range((batch * mtiles * ntiles + workers - 1) // workers):
            with pl.section_vector():
                produce(
                    step,
                    batch,
                    workers,
                    mtiles,
                    ntiles,
                    features,
                    packed_table,
                    indices,
                    nd_group,
                    nz_group,
                    panel_group,
                )
            with pl.section_cube():
                consume(
                    step,
                    batch,
                    workers,
                    rows,
                    mtiles,
                    ntiles,
                    x,
                    out,
                    a_group,
                    panel_group,
                    left,
                    right,
                    acc_group,
                )

    return kernel
