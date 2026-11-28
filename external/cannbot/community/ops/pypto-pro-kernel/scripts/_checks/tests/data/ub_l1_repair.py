# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

import pypto_pro.language as pl
from pypto_pro.language import Vf as vf
from pypto_pro.runtime.tilingkey import TilingKeyField

COLS = 128
OUT_COLS = 16


class HandoverKey:
    ROWS = TilingKeyField(bits=2, values=[8, 16, 32])
    DIRECT = TilingKeyField(bits=1, values=[0, 1])


@pl.vector_function
def increment_nd(tile, rows: pl.DT_INT32):
    mask = vf.create_mask(pattern=pl.MaskPattern.ALL, dtype=pl.DT_FP16)
    one = vf.full(1.0, mask)
    for row in pl.range(rows):
        values = vf.load_align(tile, [row, 0])
        result = vf.add(values, one, mask)
        vf.store_align(tile, result, mask, [row, 0])


@pl.vector_function
def increment_nz(src, dst, rows: pl.DT_INT32, pitch: pl.DT_INT32):
    mask = vf.create_mask(pattern=pl.MaskPattern.ALL, dtype=pl.DT_FP16)
    one = vf.full(1.0, mask)
    for row in pl.range(rows):
        values = vf.load_align(src, [row, 0])
        result = vf.add(values, one, mask)
        vf.store_align(
            dst,
            result,
            mask,
            data_copy_mode=pl.DataCopyMode.DATA_BLOCK_COPY,
            block_stride=pitch,
            repeat_stride=1,
            post_update=True,
        )


@pl.jit(arch="a5", auto_mutex=True, tiling_key=HandoverKey)
def handover(
    x: pl.Tensor[[pl.DYNAMIC, COLS], pl.DT_FP16],
    rhs: pl.Tensor[[COLS, OUT_COLS], pl.DT_FP16],
    out: pl.Tensor[[pl.DYNAMIC, OUT_COLS], pl.DT_FP32],
):
    shared = pl.make_tile_group(
        type=pl.TileType(
            shape=[2 * ROWS, COLS],  # noqa: F821 - DSL tiling-key fields.
            dtype=pl.DT_FP16,
            target_memory=pl.MemorySpace.Mat,
            layout=pl.NZ,
        ),
        addrs=0,
        mutex_ids="auto",
    )
    with pl.section_vector():
        physical_rows = (ROWS + 15) // 16 * 16  # noqa: F821 - DSL tiling-key fields.
        pitch = ROWS + 1  # noqa: F821 - DSL tiling-key fields.
        nd_group = pl.make_tile_group(
            type=pl.TileType(
                shape=[physical_rows, COLS],
                valid_shape=[ROWS, COLS],  # noqa: F821 - DSL tiling-key fields.
                dtype=pl.DT_FP16,
                target_memory=pl.MemorySpace.Vec,
                layout=pl.ND,
            ),
            addrs=0,
            mutex_ids="auto",
        )
        nz_group = pl.make_tile_group(
            type=pl.TileType(
                shape=[pitch, COLS],
                valid_shape=[ROWS, COLS],  # noqa: F821 - DSL tiling-key fields.
                dtype=pl.DT_FP16,
                target_memory=pl.MemorySpace.Vec,
                layout=pl.NZ,
            ),
            addrs=16384,
            mutex_ids="auto",
        )
        nd, nz = (nd_group.current(), nz_group.current())
        row_offset = pl.get_subblock_idx() * ROWS  # noqa: F821 - DSL tiling-key fields.
        pl.load(nd, x, [row_offset, 0])
        if DIRECT:  # noqa: F821 - DSL tiling-key fields.
            increment_nz(nd, nz, ROWS, pitch)  # noqa: F821 - DSL tiling-key fields.
        else:
            increment_nd(nd, ROWS)  # noqa: F821 - DSL tiling-key fields.
            pl.move(nz, nd)
        l1 = shared.current()
        pl.insert(l1, nd, [row_offset, 0])
        pl.system.set_cross_core(pipe=pl.PipeType.MTE3, event_id=2)
    with pl.section_cube():
        weights = pl.make_tile_group(
            type=pl.TileType(
                shape=[COLS, OUT_COLS],
                dtype=pl.DT_FP16,
                target_memory=pl.MemorySpace.Mat,
                layout=pl.NZ,
            ),
            addrs=32768,
            mutex_ids="auto",
        )
        left = pl.make_tile_group(
            type=pl.TileType(
                shape=[2 * ROWS, COLS],  # noqa: F821 - DSL tiling-key fields.
                dtype=pl.DT_FP16,
                target_memory=pl.MemorySpace.Left,
                layout=pl.NZ,
            ),
            addrs=0,
            mutex_ids="auto",
        )
        right = pl.make_tile_group(
            type=pl.TileType(
                shape=[COLS, OUT_COLS],
                dtype=pl.DT_FP16,
                target_memory=pl.MemorySpace.Right,
                layout=pl.ZN,
            ),
            addrs=0,
            mutex_ids="auto",
        )
        accumulator = pl.make_tile_group(
            type=pl.TileType(
                shape=[2 * ROWS, OUT_COLS],  # noqa: F821 - DSL tiling-key fields.
                dtype=pl.DT_FP32,
                target_memory=pl.MemorySpace.Acc,
                layout=pl.NZ,
                fractal=1024,
            ),
            addrs=0,
            mutex_ids="auto",
        )
        b = weights.current()
        pl.load(b, rhs, [0, 0])
        pl.system.wait_cross_core(
            pipe=pl.PipeType.MTE1,
            event_id=2,
            sync_mode=pl.CrossCoreSyncMode.INTRA_BLOCK,
        )
        a = shared.current()
        acc = accumulator.current()
        pl.gemm(acc, a, b, l0a=left, l0b=right, splitk=COLS)
        pl.store(out, acc, [0, 0])
