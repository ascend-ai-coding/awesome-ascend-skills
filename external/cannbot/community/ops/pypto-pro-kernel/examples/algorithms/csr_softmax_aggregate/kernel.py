# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

"""写法：原始。CSR 分段 softmax 加权聚合，共享权重与独占分片归约。

契约：连续 FP32 logits[E]/table[V,D]、INT32 offsets[R+1]/indices[E]，R=1..257、E=0..65537、V=1..4097、D=1..129；offsets 单调且首0末E，
indices 位于[0,V)。logits∈[-20,20]、table∈[-2,2]且有限。输出 FP32 out[R,D]/lse[R]，空行精确零/负无穷；有限输出 atol=rtol=3e-4。Host 只读元数据、分配和调度。
分工：设备端按 offsets 构造每行的连续 chunk 区间，默认 chunk=128。每个 (chunk,column) 独占一个 partial，column=0 同时写分母，最终每个 (row,column) 独占输出；重复的输入
索引不引入输出写冲突。权重每条边计算一次，供不同列共享。描述符构造扫描段界，代价与段数和描述符上界有关，不能视作零成本调度。
数值：固定参考 c=0，exp 输入在[-20,20]，最小非空分母大于2e-9，最大分母小于3.2e13，加权和绝对值小于6.4e13；全部使用 FP32 系数和累加。同一参考下各 partial 直接相加。范围证明不替代舍入验收，
test.py 另外覆盖端点、常数分数和接近抵消的载荷。
资源：令 S=ceil(E/128)+R，非空输入额外 GM 为 4*E+4*S*(D+4)+8*R 字节，契约最大值673844字节；包含权重、三个任务描述数组、行任务区间、分子和分母 partial。无显式 UB/L1 tile。所有消
费者读取的 partial 都由生产者定义，无效任务写中性值，实际行区间只消费自己的有效任务。
同步与空输入：prepare、partial、merge 是同一 Stream 上顺序执行的三个 native task；无需跨线程原子操作或网格内屏障。E=0 直接用一个原生填充 task，避免声明零长度 Tensor。分片大小与三阶段线
程数来自完整调用测量，其他输入分布或硬件需重新测量。
验证环境：PyPTO 导出行为基线872df763f，CANN 9.2.0、BiSheng 15.0.5、Ascend950PR，torch/torch_npu 2.10.0；当前验收仍需绑定源码哈希和实际工具链。test.py 的
FP64 数学参考不采用实现的固定参考或分块归约顺序。
"""

from functools import lru_cache

import torch
import pypto_pro.language as pl

FP32 = pl.DT_FP32
INT32 = pl.DT_INT32
CHUNK_SIZE = 128
PREP_THREADS = 128
PARTIAL_THREADS = 256
MERGE_THREADS = 128
NEG_INF = float("-inf")


@lru_cache(maxsize=None)
def _make_empty_kernel(rows, dims):
    """Zero-edge specialization avoids illegal zero-sized SIMT tensor shapes."""

    @pl.vector_function(mode="simt", max_threads=MERGE_THREADS)
    def empty_body(out, lse):
        thread = pl.simt.linear_thread_idx()
        start = pl.simt.block_idx().x * MERGE_THREADS + thread
        stride = pl.simt.grid_dim().x * pl.simt.block_dim().x
        for linear in pl.range(start, rows * dims, stride):
            row = linear // dims
            column = linear % dims
            out[row, column] = pl.simt.cast(0.0, FP32)
            if column == 0:
                lse[row] = NEG_INF

    @pl.jit()
    def empty_kernel(
        out: pl.Tensor[[rows, dims], FP32],
        lse: pl.Tensor[[rows], FP32],
    ):
        with pl.section_vector():
            empty_body[MERGE_THREADS](out, lse)

    return empty_kernel


@lru_cache(maxsize=None)
def _make_kernels(rows, edges, vocab, dims, chunk_size):
    """Build shape-specialized stages for bounded, fixed-reference softmax.

    The contract bounds permit c=0: exp([-20,20]) stays normal and finite;
    even 65,537 maximum terms keep the denominator and |table|-weighted sum
    far below FP32 overflow. A partial owns one runtime row chunk and one
    output column. The prepare pass builds compact row/chunk ranges from device
    offsets; no Host value from offsets or indices affects work assignment.
    Scratch is O((ceil(E/chunk_size)+R)*D), with no R-by-E padded rectangle.
    """
    max_tasks = (edges + chunk_size - 1) // chunk_size + rows
    prepare_work_items = max(edges, max_tasks, rows)

    @pl.vector_function(mode="simt", max_threads=PREP_THREADS)
    def prepare_body(
        logits,
        offsets,
        weights,
        task_rows,
        task_begins,
        task_ends,
        row_task_begins,
        row_task_ends,
    ):
        thread = pl.simt.linear_thread_idx()
        start = pl.simt.block_idx().x * PREP_THREADS + thread
        stride = pl.simt.grid_dim().x * pl.simt.block_dim().x
        for linear in pl.range(start, prepare_work_items, stride):
            if linear < edges:
                weights[linear] = pl.simt.exp(logits[linear])

            if linear < max_tasks:
                task = pl.simt.cast(linear, INT32)
                row = pl.simt.cast(0, INT32)
                prefix = pl.simt.cast(0, INT32)
                found = pl.simt.cast(0, INT32)
                task_begin = pl.simt.cast(0, INT32)
                task_end = pl.simt.cast(0, INT32)
                while (row < rows) & (found == 0):
                    segment_begin = offsets[row]
                    segment_end = offsets[row + 1]
                    segment_length = segment_end - segment_begin
                    chunk_count = pl.simt.cast(
                        (segment_length + chunk_size - 1) // chunk_size, INT32
                    )
                    if task < prefix + chunk_count:
                        local_chunk = task - prefix
                        task_begin = pl.simt.cast(
                            segment_begin + local_chunk * chunk_size, INT32
                        )
                        task_end = pl.simt.cast(task_begin + chunk_size, INT32)
                        if task_end > segment_end:
                            task_end = segment_end
                        found = pl.simt.cast(1, INT32)
                    else:
                        prefix = prefix + chunk_count
                        row = row + pl.simt.cast(1, INT32)
                if found == 1:
                    task_rows[task] = row
                    task_begins[task] = task_begin
                    task_ends[task] = task_end
                else:
                    task_rows[task] = pl.simt.cast(-1, INT32)
                    task_begins[task] = pl.simt.cast(0, INT32)
                    task_ends[task] = pl.simt.cast(0, INT32)

            if linear < rows:
                row_index = pl.simt.cast(linear, INT32)
                prefix = pl.simt.cast(0, INT32)
                for prior_row in pl.range(0, row_index):
                    prior_begin = offsets[prior_row]
                    prior_end = offsets[prior_row + 1]
                    prior_length = prior_end - prior_begin
                    prior_chunks = pl.simt.cast(
                        (prior_length + chunk_size - 1) // chunk_size, INT32
                    )
                    prefix = prefix + prior_chunks
                segment_begin = offsets[row_index]
                segment_end = offsets[row_index + 1]
                segment_length = segment_end - segment_begin
                chunk_count = pl.simt.cast(
                    (segment_length + chunk_size - 1) // chunk_size, INT32
                )
                row_task_begins[row_index] = prefix
                row_task_ends[row_index] = prefix + chunk_count

    @pl.vector_function(mode="simt", max_threads=PARTIAL_THREADS)
    def partial_body(
        table,
        indices,
        weights,
        task_rows,
        task_begins,
        task_ends,
        partials,
        denominators,
    ):
        thread = pl.simt.linear_thread_idx()
        start = pl.simt.block_idx().x * PARTIAL_THREADS + thread
        stride = pl.simt.grid_dim().x * pl.simt.block_dim().x
        total = max_tasks * dims
        for linear in pl.range(start, total, stride):
            task = linear // dims
            column = linear % dims
            value = pl.simt.cast(0.0, FP32)
            if task_rows[task] >= 0:
                task_begin = task_begins[task]
                task_end = task_ends[task]
                if column == 0:
                    denominator = pl.simt.cast(0.0, FP32)
                    for position in pl.range(task_begin, task_end):
                        weight = weights[position]
                        denominator = denominator + weight
                        value = value + weight * table[indices[position], column]
                    denominators[task] = denominator
                else:
                    for position in pl.range(task_begin, task_end):
                        value = (
                            value + weights[position] * table[indices[position], column]
                        )
            else:
                if column == 0:
                    denominators[task] = pl.simt.cast(0.0, FP32)
            partials[task, column] = value

    @pl.vector_function(mode="simt", max_threads=MERGE_THREADS)
    def merge_body(row_task_begins, row_task_ends, partials, denominators, out, lse):
        thread = pl.simt.linear_thread_idx()
        start = pl.simt.block_idx().x * MERGE_THREADS + thread
        stride = pl.simt.grid_dim().x * pl.simt.block_dim().x
        total = rows * dims
        for linear in pl.range(start, total, stride):
            row = linear // dims
            column = linear % dims
            task_begin = row_task_begins[row]
            task_end = row_task_ends[row]
            numerator = pl.simt.cast(0.0, FP32)
            denominator = pl.simt.cast(0.0, FP32)
            for task in pl.range(task_begin, task_end):
                numerator = numerator + partials[task, column]
                denominator = denominator + denominators[task]
            if denominator > 0.0:
                out[row, column] = numerator / denominator
                if column == 0:
                    lse[row] = pl.simt.log(denominator)
            else:
                out[row, column] = pl.simt.cast(0.0, FP32)
                if column == 0:
                    lse[row] = NEG_INF

    @pl.jit(name=f"csr_prepare_chunk{chunk_size}")
    def prepare_kernel(
        logits: pl.Tensor[[edges], FP32],
        offsets: pl.Tensor[[rows + 1], INT32],
        weights: pl.Tensor[[edges], FP32],
        task_rows: pl.Tensor[[max_tasks], INT32],
        task_begins: pl.Tensor[[max_tasks], INT32],
        task_ends: pl.Tensor[[max_tasks], INT32],
        row_task_begins: pl.Tensor[[rows], INT32],
        row_task_ends: pl.Tensor[[rows], INT32],
    ):
        with pl.section_vector():
            prepare_body[PREP_THREADS](
                logits,
                offsets,
                weights,
                task_rows,
                task_begins,
                task_ends,
                row_task_begins,
                row_task_ends,
            )

    @pl.jit(name=f"csr_partial_chunk{chunk_size}")
    def partial_kernel(
        table: pl.Tensor[[vocab, dims], FP32],
        indices: pl.Tensor[[edges], INT32],
        weights: pl.Tensor[[edges], FP32],
        task_rows: pl.Tensor[[max_tasks], INT32],
        task_begins: pl.Tensor[[max_tasks], INT32],
        task_ends: pl.Tensor[[max_tasks], INT32],
        partials: pl.Tensor[[max_tasks, dims], FP32],
        denominators: pl.Tensor[[max_tasks], FP32],
    ):
        with pl.section_vector():
            partial_body[PARTIAL_THREADS](
                table,
                indices,
                weights,
                task_rows,
                task_begins,
                task_ends,
                partials,
                denominators,
            )

    @pl.jit(name=f"csr_merge_chunk{chunk_size}")
    def merge_kernel(
        row_task_begins: pl.Tensor[[rows], INT32],
        row_task_ends: pl.Tensor[[rows], INT32],
        partials: pl.Tensor[[max_tasks, dims], FP32],
        denominators: pl.Tensor[[max_tasks], FP32],
        out: pl.Tensor[[rows, dims], FP32],
        lse: pl.Tensor[[rows], FP32],
    ):
        with pl.section_vector():
            merge_body[MERGE_THREADS](
                row_task_begins, row_task_ends, partials, denominators, out, lse
            )

    return prepare_kernel, partial_kernel, merge_kernel, max_tasks


def _run_with_chunk_size(
    logits, offsets, indices, table, out, lse, workers, chunk_size
):
    """Internal complete call used to compare reduction work-unit sizes."""
    if any(
        tensor.ndim != ndim
        for tensor, ndim in (
            (logits, 1),
            (offsets, 1),
            (indices, 1),
            (table, 2),
            (out, 2),
            (lse, 1),
        )
    ):
        raise ValueError(
            "expected logits/offsets/indices/lse vectors and table/out matrices"
        )
    edges = logits.shape[0]
    rows = out.shape[0]
    vocab, dims = table.shape
    if indices.shape != (edges,):
        raise ValueError("indices must have E elements")
    if offsets.shape != (rows + 1,):
        raise ValueError("offsets must have R + 1 elements")
    if out.shape != (rows, dims) or lse.shape != (rows,):
        raise ValueError("out and lse shapes do not match R and D")
    if not (1 <= rows <= 257 and 0 <= edges <= 65537):
        raise ValueError("R or E is outside the supported range")
    if not (1 <= vocab <= 4097 and 1 <= dims <= 129):
        raise ValueError("V or D is outside the supported range")
    if workers < 1:
        raise ValueError("workers must be positive")
    if chunk_size < 1:
        raise ValueError("chunk_size must be positive")
    tensors = (logits, offsets, indices, table, out, lse)
    if not all(tensor.is_contiguous() for tensor in tensors):
        raise ValueError("all tensors must be contiguous")
    if len({tensor.device for tensor in tensors}) != 1:
        raise ValueError("all tensors must be on the same device")
    if logits.dtype != torch.float32 or table.dtype != torch.float32:
        raise ValueError("logits and table must be FP32")
    if out.dtype != torch.float32 or lse.dtype != torch.float32:
        raise ValueError("out and lse must be FP32")
    if offsets.dtype != torch.int32 or indices.dtype != torch.int32:
        raise ValueError("offsets and indices must be INT32")

    if edges == 0:
        _make_empty_kernel(rows, dims)[None, workers](out, lse)
        return

    prepare_kernel, partial_kernel, merge_kernel, max_tasks = _make_kernels(
        rows, edges, vocab, dims, chunk_size
    )
    device = logits.device
    weights = torch.empty((edges,), dtype=torch.float32, device=device)
    task_rows = torch.empty((max_tasks,), dtype=torch.int32, device=device)
    task_begins = torch.empty((max_tasks,), dtype=torch.int32, device=device)
    task_ends = torch.empty((max_tasks,), dtype=torch.int32, device=device)
    row_task_begins = torch.empty((rows,), dtype=torch.int32, device=device)
    row_task_ends = torch.empty((rows,), dtype=torch.int32, device=device)
    partials = torch.empty((max_tasks, dims), dtype=torch.float32, device=device)
    denominators = torch.empty((max_tasks,), dtype=torch.float32, device=device)

    prepare_kernel[None, workers](
        logits,
        offsets,
        weights,
        task_rows,
        task_begins,
        task_ends,
        row_task_begins,
        row_task_ends,
    )
    partial_kernel[None, workers](
        table,
        indices,
        weights,
        task_rows,
        task_begins,
        task_ends,
        partials,
        denominators,
    )
    merge_kernel[None, workers](
        row_task_begins, row_task_ends, partials, denominators, out, lse
    )


def run(logits, offsets, indices, table, out, lse, workers=4):
    """Launch CSR softmax aggregation; Host reads tensor metadata only."""
    return _run_with_chunk_size(
        logits, offsets, indices, table, out, lse, workers, CHUNK_SIZE
    )


def run_with_chunk_size(
    logits, offsets, indices, table, out, lse, chunk_size, workers=4
):
    """Run a work-unit variant for measured implementation comparisons."""
    return _run_with_chunk_size(
        logits, offsets, indices, table, out, lse, workers, chunk_size
    )
