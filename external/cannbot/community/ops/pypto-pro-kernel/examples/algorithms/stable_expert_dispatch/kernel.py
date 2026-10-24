# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

"""写法：原始。按 expert 稳定打包连续 FP32 token 行。

两个 SIMT 任务先初始化输出并计数，再由唯一 token owner 求位置及复制 payload。
同 Stream 的任务顺序保护 counts/offsets；区间编号源于输入位置，与 worker 调度无关。
每个 token 串行复制连续列，是元数据复用的简单参考；是否优于协作搬运需按形状和 worker 数测量。
数值验证对应 Ascend950PR、pypto_pro 行为 872df763f、CANN 9.2.0。
"""

from functools import lru_cache

import pypto_pro.language as pl


FP32 = pl.DT_FP32
INT32 = pl.DT_INT32
THREADS = 128
CHUNK = 64


def _make_kernels(rows: int, cols: int, experts: int):
    chunks = (rows + CHUNK - 1) // CHUNK

    @pl.vector_function(mode="simt", max_threads=THREADS)
    def prepare_body(expert_ids, y, permutation, inverse, offsets, counts):
        start = pl.simt.block_idx().x * THREADS + pl.simt.linear_thread_idx()
        stride = pl.simt.grid_dim().x * pl.simt.block_dim().x
        zero_i32 = pl.simt.cast(0, INT32)
        one_i32 = pl.simt.cast(1, INT32)

        for linear in pl.range(start, rows * cols, stride):
            y[linear // cols, linear % cols] = 0.0

        for token in pl.range(start, rows, stride):
            permutation[token] = -1
            inverse[token] = -1

        for cell in pl.range(start, chunks * experts, stride):
            chunk = cell // experts
            expert = cell % experts
            expert_i32 = pl.simt.cast(expert, INT32)
            first = chunk * CHUNK
            count = zero_i32
            for token in pl.range(first, first + CHUNK):
                if token < rows and expert_ids[token] == expert_i32:
                    count = count + one_i32
            counts[chunk, expert] = count

        for expert in pl.range(start, experts + 1, stride):
            expert_i32 = pl.simt.cast(expert, INT32)
            count = zero_i32
            for token in pl.range(0, rows):
                if expert_ids[token] >= zero_i32 and expert_ids[token] < expert_i32:
                    count = count + one_i32
            offsets[expert] = count

    @pl.jit()
    def prepare(
        expert_ids: pl.Tensor[[rows], INT32],
        y: pl.Tensor[[rows, cols], FP32],
        permutation: pl.Tensor[[rows], INT32],
        inverse: pl.Tensor[[rows], INT32],
        offsets: pl.Tensor[[experts + 1], INT32],
        counts: pl.Tensor[[chunks, experts], INT32],
    ):
        with pl.section_vector():
            prepare_body[THREADS](expert_ids, y, permutation, inverse, offsets, counts)

    @pl.vector_function(mode="simt", max_threads=THREADS)
    def scatter_body(x, expert_ids, y, permutation, inverse, offsets, counts):
        start = pl.simt.block_idx().x * THREADS + pl.simt.linear_thread_idx()
        stride = pl.simt.grid_dim().x * pl.simt.block_dim().x
        zero_i32 = pl.simt.cast(0, INT32)
        one_i32 = pl.simt.cast(1, INT32)

        for token in pl.range(start, rows, stride):
            expert = expert_ids[token]
            if expert >= zero_i32:
                chunk = token // CHUNK
                chunk_start = chunk * CHUNK
                rank = zero_i32
                for earlier in pl.range(chunk_start, token):
                    if expert_ids[earlier] == expert:
                        rank = rank + one_i32
                for previous_chunk in pl.range(0, chunk):
                    rank = rank + counts[previous_chunk, expert]
                destination = offsets[expert] + rank
                permutation[destination] = pl.simt.cast(token, INT32)
                inverse[token] = destination
                for col in pl.range(0, cols):
                    y[destination, col] = x[token, col]

    @pl.jit()
    def scatter(
        x: pl.Tensor[[rows, cols], FP32],
        expert_ids: pl.Tensor[[rows], INT32],
        y: pl.Tensor[[rows, cols], FP32],
        permutation: pl.Tensor[[rows], INT32],
        inverse: pl.Tensor[[rows], INT32],
        offsets: pl.Tensor[[experts + 1], INT32],
        counts: pl.Tensor[[chunks, experts], INT32],
    ):
        with pl.section_vector():
            scatter_body[THREADS](
                x, expert_ids, y, permutation, inverse, offsets, counts
            )

    return prepare, scatter


@lru_cache(maxsize=None)
def _cached_kernels(rows: int, cols: int, experts: int):
    return _make_kernels(rows, cols, experts)


def run(x, expert_ids, y, permutation, inverse, offsets, num_experts, workers=1):
    """Host 只验证元数据、分配计数工作区并顺序启动两个原生任务。"""
    rows, cols = map(int, x.shape) if len(x.shape) == 2 else (0, 0)
    experts = int(num_experts)
    if not 1 <= rows <= 8192 or not 1 <= cols <= 257:
        raise ValueError("x must have shape [N,D] with 1<=N<=8192 and 1<=D<=257")
    if not 1 <= experts <= 32:
        raise ValueError("num_experts must be in [1,32]")
    if int(workers) <= 0:
        raise ValueError("workers must be positive")
    if tuple(expert_ids.shape) != (rows,):
        raise ValueError("expert_ids must have shape [N]")
    if tuple(y.shape) != (rows, cols):
        raise ValueError("y must have shape [N,D]")
    if tuple(permutation.shape) != (rows,) or tuple(inverse.shape) != (rows,):
        raise ValueError("permutation and inverse must have shape [N]")
    if tuple(offsets.shape) != (experts + 1,):
        raise ValueError("offsets must have shape [E+1]")
    if str(x.dtype) != "torch.float32" or str(y.dtype) != "torch.float32":
        raise ValueError("x and y must be FP32")
    if any(
        str(t.dtype) != "torch.int32"
        for t in (expert_ids, permutation, inverse, offsets)
    ):
        raise ValueError("expert_ids, permutation, inverse, and offsets must be INT32")
    tensors = (expert_ids, y, permutation, inverse, offsets)
    if not x.is_contiguous() or any(not t.is_contiguous() for t in tensors):
        raise ValueError("all tensors must be contiguous")
    if any(t.device != x.device for t in tensors):
        raise ValueError("all tensors must be on the same device")

    prepare, scatter = _cached_kernels(rows, cols, experts)
    chunks = (rows + CHUNK - 1) // CHUNK
    counts = expert_ids.new_empty((chunks, experts))
    prepare[None, int(workers)](expert_ids, y, permutation, inverse, offsets, counts)
    scatter[None, int(workers)](x, expert_ids, y, permutation, inverse, offsets, counts)
