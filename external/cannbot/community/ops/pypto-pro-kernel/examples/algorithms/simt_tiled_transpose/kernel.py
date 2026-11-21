# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

"""写法：原始。SIMT 的连续 GM 读写与共享 UB 转置。

机制：[按真实访问集合比较布局](../../../references/patterns/vf-neighborhood.md#compare-layouts-from-actual-access-sets)。
本例保留相同外部布局下的 UB pitch 控制；若另加全量格式转换，应计入准备/恢复流量。其他 stride 契约先对齐支持域再做完整调用对照。

契约：非空连续 FP32[R,C] 输入、连续 FP32[C,R] 输出，二者不重叠。
每个外层 worker 认领互不重叠的 32x32 tile；SIMT 线程协作加载和转置写回。
GM 两侧按相邻线程的连续列访问，UB 的列读使用可选物理 padding。
padding 改变 UB 行跨度，不改变外部布局；默认 2 是实验配置，不是全平台最优规则。
共享 UB 用满足 Tile 行对齐的一维物理容器，显式元素偏移表示 32×pitch 布局。
每次 VF 调用的两个 block 屏障都位于线程控制流之外，尾线程也参与。
支持有限 FP32 位模式，包括 signed zero 与 subnormal；独立参考按元素转置。
接口核对源码 c46548e；数值验证安装源码 cbd3c4a 的相关实现相同，CANN 9.2.0 / Ascend950PR。
example-query: {"aliases":["SIMT shared UB transpose","SIMT tiled permutation"],"term_groups":[["transpose","rotation","permutation","转置","旋转"],["simt","ub","tile","tensor","layout","张量"]],"exclude":[{"term_groups":[["noncontiguous input","non-contiguous input","arbitrary positive row","arbitrary input strides","非连续输入"]],"reason":"This program requires contiguous input. Resolve example:algorithms/strided_transpose/kernel.py for the separate positive-stride physical-view contract."}]}
"""

from functools import lru_cache

import pypto_pro.language as pl

F32 = pl.DT_FP32
TILE = 32


@lru_cache(None)
def make_transpose(rows, cols, threads=256, padding=2):
    if (
        rows <= 0
        or cols <= 0
        or threads not in (128, 256, 512)
        or padding not in (0, 1, 2, 4)
    ):
        raise ValueError(
            "require positive dimensions, threads 128/256/512 and padding 0/1/2/4"
        )
    tile_cols = (cols + TILE - 1) // TILE
    tiles = ((rows + TILE - 1) // TILE) * tile_cols
    pitch = TILE + padding

    @pl.vector_function(mode="simt", max_threads=threads)
    def transpose_body(x, out, scratch, tile_id):
        tid = pl.simt.linear_thread_idx()
        row_base = (tile_id // tile_cols) * TILE
        col_base = (tile_id % tile_cols) * TILE
        for i in pl.range(tid, TILE * TILE, threads):
            r = i // TILE
            c = i % TILE
            if row_base + r < rows and col_base + c < cols:
                scratch[0, r * pitch + c] = x[row_base + r, col_base + c]
            else:
                scratch[0, r * pitch + c] = pl.simt.cast(0, F32)
        pl.simt.syncthreads()
        for i in pl.range(tid, TILE * TILE, threads):
            r = i // TILE
            c = i % TILE
            if col_base + r < cols and row_base + c < rows:
                out[col_base + r, row_base + c] = scratch[0, c * pitch + r]
        pl.simt.syncthreads()

    @pl.jit(auto_mutex=False)
    def kernel(x: pl.Tensor[[rows, cols], F32], out: pl.Tensor[[cols, rows], F32]):
        scratch = pl.make_tile(
            pl.TileType(
                shape=[1, TILE * pitch], dtype=F32, target_memory=pl.MemorySpace.Vec
            ),
            addr=0,
        )
        with pl.section_vector():
            for tile_id in pl.range(pl.get_block_idx(), tiles, pl.get_block_num()):
                transpose_body[threads](x, out, scratch, tile_id)

    return kernel


def run(x, out, *, workers=4, threads=256, padding=2):
    """Write an ordinary contiguous transpose through native tile ownership."""
    if str(x.dtype) != "torch.float32" or str(out.dtype) != "torch.float32":
        raise TypeError("require FP32 input/output")
    if (
        x.ndim != 2
        or tuple(out.shape) != (x.shape[1], x.shape[0])
        or not x.is_contiguous()
        or not out.is_contiguous()
    ):
        raise ValueError("require contiguous input[R,C] and output[C,R]")
    if type(workers) is not int or workers <= 0:
        raise ValueError("workers must be a positive legal Stream block count")
    make_transpose(*x.shape, threads, padding)[None, workers](x, out)
