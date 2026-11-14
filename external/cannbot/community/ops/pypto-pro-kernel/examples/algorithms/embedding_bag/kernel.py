# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

"""写法：原始。运行期分段与间接索引的 SIMT 求和。

FP32 table[V,D]、INT32 indices[T]/offsets[B+1]；-1 为 padding，空段写零。
短工作量使用直接 owner；较大输入使用 partial 和 merge 两个同 Stream 原生 kernel。
Host 仅依据形状、worker 数分配 scratch 和选择分工；不读取分段内容。
scratch 占 B*ceil(T/chunk)*D*4 字节，所有空 partial 也显式写零。
当前阈值与 chunk 是已测样例的选择，不能由 T/D 推断真实段长分布或保证任意输入分布最优。
验证边界：行为版本 872df763f、CANN 9.2.0、Ascend950PR；改变分块仍使用同一 FP64 参考与固定容差。
"""

from functools import lru_cache

import pypto_pro.language as pl

FP32 = pl.DT_FP32
INT32 = pl.DT_INT32
DIRECT_THREADS = 128
SPLIT_THREADS = 256
SINGLE_WORKER_CHUNK_SIZE = 256
MULTI_WORKER_CHUNK_SIZE = 128
SPLIT_MIN_INDICES = 1024
SPLIT_MIN_DIM = 64


@lru_cache(maxsize=None)
def make_direct_embedding_bag(vocab_size, embedding_dim, num_bags, num_indices):
    """Build the one-pass SIMT form used for small reductions."""

    @pl.vector_function(mode="simt", max_threads=DIRECT_THREADS)
    def sum_body(table, indices, offsets, out):
        thread = pl.simt.linear_thread_idx()
        start = pl.simt.block_idx().x * DIRECT_THREADS + thread
        stride = pl.simt.grid_dim().x * pl.simt.block_dim().x
        total = num_bags * embedding_dim
        for linear in pl.range(start, total, stride):
            bag = linear // embedding_dim
            column = linear % embedding_dim
            begin = offsets[bag]
            end = offsets[bag + 1]
            value = pl.simt.cast(0.0, FP32)
            for position in pl.range(begin, end):
                row = indices[position]
                if row >= 0:
                    value = value + table[row, column]
            out[bag, column] = value

    @pl.jit()
    def kernel(
        table: pl.Tensor[[vocab_size, embedding_dim], FP32],
        indices: pl.Tensor[[num_indices], INT32],
        offsets: pl.Tensor[[num_bags + 1], INT32],
        out: pl.Tensor[[num_bags, embedding_dim], FP32],
    ):
        with pl.section_vector():
            sum_body[DIRECT_THREADS](table, indices, offsets, out)

    return kernel


@lru_cache(maxsize=None)
def make_split_embedding_bag(
    vocab_size, embedding_dim, num_bags, num_indices, num_chunks, chunk_size
):
    """Build partial-reduction and merge kernels for long segments."""

    @pl.vector_function(mode="simt", max_threads=SPLIT_THREADS)
    def partial_body(table, indices, offsets, partials):
        thread = pl.simt.linear_thread_idx()
        start = pl.simt.block_idx().x * SPLIT_THREADS + thread
        stride = pl.simt.grid_dim().x * pl.simt.block_dim().x
        total = num_bags * num_chunks * embedding_dim
        for linear in pl.range(start, total, stride):
            bag = linear // (num_chunks * embedding_dim)
            chunk = (linear // embedding_dim) % num_chunks
            column = linear % embedding_dim
            begin = offsets[bag]
            end = offsets[bag + 1]
            chunk_offset = pl.simt.cast(chunk * chunk_size, INT32)
            chunk_begin = begin + chunk_offset
            chunk_limit = pl.simt.cast(chunk_begin + chunk_size, INT32)
            chunk_end = pl.simt.min(end, chunk_limit)

            value = pl.simt.cast(0.0, FP32)
            for position in pl.range(chunk_begin, chunk_end):
                row = indices[position]
                if row >= 0:
                    value = value + table[row, column]
            partials[bag, chunk, column] = value

    @pl.vector_function(mode="simt", max_threads=DIRECT_THREADS)
    def merge_body(partials, out):
        thread = pl.simt.linear_thread_idx()
        start = pl.simt.block_idx().x * DIRECT_THREADS + thread
        stride = pl.simt.grid_dim().x * pl.simt.block_dim().x
        total = num_bags * embedding_dim
        for linear in pl.range(start, total, stride):
            bag = linear // embedding_dim
            column = linear % embedding_dim
            value = pl.simt.cast(0.0, FP32)
            for chunk in pl.range(0, num_chunks):
                value = value + partials[bag, chunk, column]
            out[bag, column] = value

    @pl.jit()
    def partial_kernel(
        table: pl.Tensor[[vocab_size, embedding_dim], FP32],
        indices: pl.Tensor[[num_indices], INT32],
        offsets: pl.Tensor[[num_bags + 1], INT32],
        partials: pl.Tensor[[num_bags, num_chunks, embedding_dim], FP32],
    ):
        with pl.section_vector():
            partial_body[SPLIT_THREADS](table, indices, offsets, partials)

    @pl.jit()
    def merge_kernel(
        partials: pl.Tensor[[num_bags, num_chunks, embedding_dim], FP32],
        out: pl.Tensor[[num_bags, embedding_dim], FP32],
    ):
        with pl.section_vector():
            merge_body[DIRECT_THREADS](partials, out)

    return partial_kernel, merge_kernel


def uses_split(vocab_size, embedding_dim, num_bags, num_indices, workers):
    """Select splitting from public metadata only; offsets remain device data."""
    return num_indices >= SPLIT_MIN_INDICES and embedding_dim >= SPLIT_MIN_DIM


def task_count(vocab_size, embedding_dim, num_bags, num_indices, workers):
    return (
        2
        if uses_split(vocab_size, embedding_dim, num_bags, num_indices, workers)
        else 1
    )


def run(table, indices, offsets, out, workers=1):
    """Launch native kernels; Host code reads tensor metadata only."""
    if table.ndim != 2 or indices.ndim != 1 or offsets.ndim != 1 or out.ndim != 2:
        raise ValueError("expected table/out matrices and indices/offsets vectors")
    vocab_size, embedding_dim = table.shape
    num_indices = indices.shape[0]
    num_bags = out.shape[0]
    if out.shape[1] != embedding_dim:
        raise ValueError("incompatible tensor shapes")
    if offsets.shape != (num_bags + 1,):
        raise ValueError("offsets must have B + 1 elements")
    if workers < 1:
        raise ValueError("workers must be positive")
    if not all(tensor.is_contiguous() for tensor in (table, indices, offsets, out)):
        raise ValueError("all tensors must be contiguous")

    if uses_split(vocab_size, embedding_dim, num_bags, num_indices, workers):
        chunk_size = (
            SINGLE_WORKER_CHUNK_SIZE if workers == 1 else MULTI_WORKER_CHUNK_SIZE
        )
        num_chunks = (num_indices + chunk_size - 1) // chunk_size
        partial_kernel, merge_kernel = make_split_embedding_bag(
            vocab_size, embedding_dim, num_bags, num_indices, num_chunks, chunk_size
        )
        partials = out.new_empty((num_bags, num_chunks, embedding_dim))
        partial_kernel[None, workers](table, indices, offsets, partials)
        merge_kernel[None, workers](partials, out)
    else:
        kernel = make_direct_embedding_bag(
            vocab_size, embedding_dim, num_bags, num_indices
        )
        kernel[None, workers](table, indices, offsets, out)
