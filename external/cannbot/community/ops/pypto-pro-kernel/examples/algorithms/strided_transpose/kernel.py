# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

"""写法：原始。SIMT 显式物理寻址的 FP32 转置。

契约：非空二维正 stride 输入视图，连续转置输出；保留有限 FP32 位模式，含正负零和 subnormal。
Host 只读元数据。SIMT 的静态一维物理 view 包含行 padding，地址按真实元素 stride 计算。
源指针已包含 storage offset。grid-stride 循环为每个输出元素指定唯一线程。
验证边界：行为版本 872df763f、CANN 9.2.0、Ascend950PR；同目录检查完整输入 storage 不变和输出守卫。
"""

from functools import lru_cache

import pypto_pro.language as pl


FP32 = pl.DT_FP32
THREADS = 128


@lru_cache(maxsize=None)
def _make_kernel(rows, cols, row_stride, col_stride):
    total = rows * cols
    input_span = 1 + (rows - 1) * row_stride + (cols - 1) * col_stride

    @pl.vector_function(mode="simt", max_threads=THREADS)
    def _transpose_elements(src, dst):
        block = pl.simt.block_idx().x
        thread = pl.simt.linear_thread_idx()
        blocks = pl.simt.grid_dim().x
        first = block * THREADS + thread
        step = blocks * THREADS

        for output_index in pl.range(first, total, step):
            source_row = output_index % rows
            source_col = output_index // rows
            source_index = source_row * row_stride + source_col * col_stride
            dst[output_index] = src[source_index]

    @pl.jit()
    def _transpose(
        x: pl.Tensor[[pl.DYNAMIC, pl.DYNAMIC], FP32],
        out: pl.Tensor[[pl.DYNAMIC, pl.DYNAMIC], FP32],
    ):
        source = pl.make_tensor(x, [input_span], [1])
        destination = pl.make_tensor(out, [total], [1])
        with pl.section_vector():
            _transpose_elements[THREADS](source, destination)

    return _transpose


def run(x, out, workers=1):
    """Write ``out[n, m] = x[m, n]`` through a native PyPTO kernel.

    Host-side reads are limited to tensor metadata. The input pointer already
    addresses the first logical element, so PyTorch's storage offset is not
    added again.
    """
    if type(workers) is not int or workers <= 0:
        raise ValueError(f"workers must be a positive integer, got {workers!r}")
    if str(x.dtype) != "torch.float32" or str(out.dtype) != "torch.float32":
        raise TypeError("x and out must both have dtype torch.float32")

    rows, cols = (int(dim) for dim in x.shape)
    if tuple(int(dim) for dim in out.shape) != (cols, rows):
        raise ValueError(f"out must have shape {(cols, rows)}, got {tuple(out.shape)}")
    if not out.is_contiguous():
        raise ValueError("out must be contiguous")

    row_stride, col_stride = (int(stride) for stride in x.stride())
    if row_stride <= 0 or col_stride <= 0:
        raise ValueError("x must have positive strides")

    kernel = _make_kernel(rows, cols, row_stride, col_stride)
    kernel[None, workers](x, out)
