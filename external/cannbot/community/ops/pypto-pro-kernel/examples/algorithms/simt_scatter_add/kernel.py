# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

"""写法：原始。SIMT 初始化与原子累加构成完整 scatter-add。

INT32 base[N]、index[U]、value[U]，index 在 [-N,N) 内；输出从任意内容开始。
同一 Stream 依次运行返回的 initialize、accumulate；U=0 时只需 initialize。
物理尺寸按工厂参数特化，两个 kernel 均按实际 grid 跨步覆盖工作。
索引规范化保持 INT32，要求 0<N<2**31；调用者保证累加结果不溢出 INT32。
显式 grid 必须位于当前 Stream 预算内，也可直接调用使用自动预算。
验证边界：行为版本 872df763f、CANN 9.2.0；数值、冲突与守卫检查见 test.py。
"""

import pypto_pro.language as pl

I32 = pl.DT_INT32
THREADS = 128


def make_scatter_add(n, updates):
    """Return native base-copy and scatter-add kernels specialized for shapes."""
    if not 0 < n < 2**31 or updates < 0:
        raise ValueError("require 0 < N < 2**31 and U >= 0")

    @pl.vector_function(mode="simt", max_threads=THREADS)
    def copy_base_body(base, out):
        start = pl.simt.block_idx().x * THREADS + pl.simt.linear_thread_idx()
        stride = pl.simt.grid_dim().x * pl.simt.block_dim().x
        for i in pl.range(start, n, stride):
            out[i] = base[i]

    @pl.jit()
    def initialize(base: pl.Tensor[[n], I32], out: pl.Tensor[[n], I32]):
        with pl.section_vector():
            copy_base_body[THREADS](base, out)

    if updates == 0:
        return initialize, None

    @pl.vector_function(mode="simt", max_threads=THREADS)
    def scatter_body(index, value, out):
        start = pl.simt.block_idx().x * THREADS + pl.simt.linear_thread_idx()
        stride = pl.simt.grid_dim().x * pl.simt.block_dim().x
        zero = pl.simt.cast(0, I32)
        n_i32 = pl.simt.cast(n, I32)
        for pos in pl.range(start, updates, stride):
            dst = index[pos]
            if dst < zero:
                dst = dst + n_i32
            pl.simt.atomic_add(out[dst], value[pos])

    @pl.jit()
    def accumulate(
        index: pl.Tensor[[updates], I32],
        value: pl.Tensor[[updates], I32],
        out: pl.Tensor[[n], I32],
    ):
        with pl.section_vector():
            scatter_body[THREADS](index, value, out)

    return initialize, accumulate
