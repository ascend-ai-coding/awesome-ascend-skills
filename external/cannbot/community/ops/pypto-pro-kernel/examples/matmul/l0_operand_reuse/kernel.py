# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

"""写法：混合。两个M块复用同一L0B，与两次pl.gemm保留数值对照。

契约：A[128,128]、B[128,128] FP16，C[128,128] FP32；一次单Cube核调用。每次计算64行，两个Acc槽互不覆盖。资源：L1 A/B各16/32 KiB、L0A双槽合计32 KiB、L0B单槽32
KiB、Acc双槽合计64 KiB。B只读且在两个matmul间不被覆盖，tile-group IDs让auto_mutex看见全部真实读写。
手写链仅用于gemm-desugar的“保留L0操作数”例外：reuse=True在循环前move B一次；False每次gemm都重新搬B。A搬运、两个matmul、drain和算术完全相同。生成代码中的B搬运次数是结构证据，数值测试或减
少一次move都不代表硬件加速。
"""

from functools import lru_cache

import pypto_pro.language as pl


@lru_cache(maxsize=None)
def make_kernel(reuse):
    @pl.jit(auto_mutex=True, name=f"l0_operand_reuse_{reuse}")
    def kernel(
        a: pl.Tensor[[128, 128], pl.DT_FP16],
        b: pl.Tensor[[128, 128], pl.DT_FP16],
        c: pl.Tensor[[128, 128], pl.DT_FP32],
    ):
        ag = pl.make_tile_group(
            type=pl.TileType(
                shape=[64, 128],
                dtype=pl.DT_FP16,
                target_memory=pl.MemorySpace.Mat,
                layout=pl.NZ,
            ),
            addrs=0x0,
            mutex_ids="auto",
            depth=1,
        )
        bg = pl.make_tile_group(
            type=pl.TileType(
                shape=[128, 128],
                dtype=pl.DT_FP16,
                target_memory=pl.MemorySpace.Mat,
                layout=pl.NZ,
            ),
            addrs=0x4000,
            mutex_ids="auto",
            depth=1,
        )
        left = pl.make_tile_group(
            type=pl.TileType(
                shape=[64, 128],
                dtype=pl.DT_FP16,
                target_memory=pl.MemorySpace.Left,
                layout=pl.NZ,
            ),
            addrs=0x0,
            mutex_ids="auto",
            depth=2,
        )
        right = pl.make_tile_group(
            type=pl.TileType(
                shape=[128, 128],
                dtype=pl.DT_FP16,
                target_memory=pl.MemorySpace.Right,
                layout=pl.ZN,
            ),
            addrs=0x0,
            mutex_ids="auto",
            depth=1,
        )
        acc = pl.make_tile_group(
            type=pl.TileType(
                shape=[64, 128],
                dtype=pl.DT_FP32,
                target_memory=pl.MemorySpace.Acc,
                layout=pl.NZ,
            ),
            addrs=0x0,
            mutex_ids="auto",
            depth=2,
        )
        with pl.section_cube():
            bt = bg.current()
            pl.load(bt, b, [0, 0])
            if reuse:
                retained_b = right.current()
                pl.move(retained_b, bt)
            for block in pl.range(2):
                at = ag.next()
                ct = acc.next()
                pl.load(at, a, [block * 64, 0])
                if reuse:
                    al = left.next()
                    pl.move(al, at)
                    pl.matmul(ct, al, retained_b)
                else:
                    pl.gemm(ct, at, bt, l0a=left, l0b=right)
                pl.store(c, ct, [block * 64, 0])

    return kernel
