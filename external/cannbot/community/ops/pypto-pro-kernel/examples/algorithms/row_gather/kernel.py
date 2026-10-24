# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

"""写法：原始。VF 逐行间接读取，包含负索引和尾块。

契约：有限 FP32 x[M,N]，INT32 index[M,K]，索引位于 [-N,N)，负值按 +N 解释，允许重复索引，输出逐位保持输入值。
已验 M=1/7/33、N=65/257、K=31/80。GM 输入保持紧密布局；X 的物理 UB 行按32B补齐，索引/输出按64-lane寄存器补齐，真实范围用 valid_shape 和 VF mask 表达。
固定地址：X@0 最大1056B，索引@0x1000 最大512B，输出@0x2000 最大512B；扩大契约须重新规划这些区域，示例未宣称任意宽度。
同步：三个单槽 tile group 由 auto_mutex 排序；纯 Vector 的行循环使用 get_block_num，不推断双 AIV 数量。gather 的谓词限制实际索引访存，GM store 只写真实 K 列。
来源：2026-10-06 gpt-6-luna/max 独立 skill 试验产物，入库仅整理说明、相对导入与独立测试，Kernel 实现不变。
验证边界：行为版本 872df763f、CANN 9.2.0、Ascend950PR；运行 test.py 检查数值与输出守卫。未测硬件性能。
"""

import pypto_pro.language as pl
from pypto_pro.language import Vf as vf

FP32 = pl.DT_FP32
I32 = pl.DT_INT32
U32 = pl.DT_UINT32


@pl.vector_function
def _gather_row(x_tile, index_tile, out_tile, n, k):
    """从一行 x 按有符号索引收集 K 个 FP32 值。"""
    for col in pl.range(0, k, 64):
        active = pl.min(64, k - col)
        index_mask = vf.update_mask(active, dtype=I32)
        out_mask = vf.update_mask(active, dtype=FP32)

        signed_index = vf.load_align(index_tile, [0, col])
        negative = vf.lt(signed_index, 0, index_mask)
        wrapped_index = vf.adds(signed_index, n, index_mask)
        normalized_index = vf.select(wrapped_index, signed_index, negative)
        gather_index = vf.astype(normalized_index, index_mask, dtype=U32)

        values = vf.gather(x_tile, gather_index, out_mask)
        vf.store_align(out_tile, values, out_mask, [0, col])


def make_row_gather(n, k):
    """Create an M-dynamic kernel specialized for N and K."""
    padded_n = ((n + 7) // 8) * 8
    padded_k = ((k + 63) // 64) * 64

    @pl.jit(auto_mutex=True, name="vf_row_gather")
    def kernel(
        x: pl.Tensor[[pl.DYNAMIC, n], FP32],
        index: pl.Tensor[[pl.DYNAMIC, k], I32],
        y: pl.Tensor[[pl.DYNAMIC, k], FP32],
    ):
        x_group = pl.make_tile_group(
            type=pl.TileType(
                shape=[1, padded_n], dtype=FP32, target_memory=pl.MemorySpace.Vec
            ),
            addrs=[0x0000],
            mutex_ids=[0],
        )
        index_group = pl.make_tile_group(
            type=pl.TileType(
                shape=[1, padded_k], dtype=I32, target_memory=pl.MemorySpace.Vec
            ),
            addrs=[0x1000],
            mutex_ids=[1],
        )
        out_group = pl.make_tile_group(
            type=pl.TileType(
                shape=[1, padded_k], dtype=FP32, target_memory=pl.MemorySpace.Vec
            ),
            addrs=[0x2000],
            mutex_ids=[2],
        )

        with pl.section_vector():
            for row in pl.range(pl.get_block_idx(), x.shape[0], pl.get_block_num()):
                x_tile = x_group.next()
                index_tile = index_group.next()
                out_tile = out_group.next()
                pl.set_validshape(x_tile, [1, n])
                pl.set_validshape(index_tile, [1, k])
                pl.set_validshape(out_tile, [1, k])
                pl.load(x_tile, x, [row, 0])
                pl.load(index_tile, index, [row, 0])
                _gather_row(x_tile, index_tile, out_tile, n, k)
                pl.store(y, out_tile, [row, 0])

    return kernel
