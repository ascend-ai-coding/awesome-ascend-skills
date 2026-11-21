# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

"""写法：原始。Native device-side row selection followed by an MTE payload copy.

Contract: contiguous, disjoint FP32 table[R,C], INT32 indices[N], FP32 out[N,C];
positive N/R, 1<=C<=4096, valid indices in [0,R), 32-byte aligned tensor bases.
Host uses metadata only. The outer JIT scalar read selects a row on device;
MTE copies its bytes through one padded Tile with automatic reuse ordering.
No SIMT or arithmetic is needed for the payload. Short rows remain a separate
performance control; this example does not promise a general gather speedup.
API scope: source c46548e, CANN 9.2.0 / Ascend950PR; tests check current behavior.
example-query: {"aliases":["device scalar indexed row copy","MTE indirect row payload"],"term_groups":[["indexed row","indirect row","整行间接搬运"],["copy","mte","搬运"]],"exclude":[{"term_groups":[["noncontiguous input","non-contiguous input","noncontiguous output","non-contiguous output","arbitrary input strides","arbitrary output strides","非连续输入","非连续输出","任意输入步长","任意输出步长"]],"reason":"This program requires contiguous FP32 input and output; arbitrary physical strides need a separately verified implementation."}]}
"""

from functools import lru_cache

import pypto_pro.language as pl

F32 = pl.DT_FP32
I32 = pl.DT_INT32


@lru_cache(None)
def make_copy(rows, columns, count):
    if rows <= 0 or count <= 0 or not 1 <= columns <= 4096:
        raise ValueError("require positive rows/count and 1<=columns<=4096")
    physical_columns = (columns + 7) // 8 * 8

    @pl.jit(auto_mutex=True)
    def kernel(
        table: pl.Tensor[[rows, columns], F32],
        indices: pl.Tensor[[count], I32],
        out: pl.Tensor[[count, columns], F32],
    ):
        group = pl.make_tile_group(
            type=pl.TileType(
                shape=[1, physical_columns], dtype=F32, target_memory=pl.MemorySpace.Vec
            ),
            addrs=[0],
            mutex_ids=[0],
        )
        with pl.section_vector():
            for row in pl.range(pl.get_block_idx(), count, pl.get_block_num()):
                selected = indices[row]
                tile = group.next()
                pl.set_validshape(tile, [1, columns])
                pl.load(tile, table, [selected, 0])
                pl.store(out, tile, [row, 0])

    return kernel


def run(table, indices, out, *, workers=8):
    """Copy complete selected rows; callers supply valid indices and disjoint storage."""
    if (
        str(table.dtype) != "torch.float32"
        or str(indices.dtype) != "torch.int32"
        or str(out.dtype) != "torch.float32"
    ):
        raise TypeError("require FP32 table/output and INT32 indices")
    if (
        table.ndim != 2
        or indices.ndim != 1
        or tuple(out.shape) != (indices.shape[0], table.shape[1])
    ):
        raise ValueError("require table[R,C], indices[N] and out[N,C]")
    if not all(t.is_contiguous() for t in (table, indices, out)):
        raise ValueError("require contiguous tensors")
    if type(workers) is not int or workers <= 0:
        raise ValueError("workers must be a positive legal Stream block count")
    make_copy(*table.shape, indices.shape[0])[None, workers](table, indices, out)
