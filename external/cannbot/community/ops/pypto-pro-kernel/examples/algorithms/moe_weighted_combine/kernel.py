# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

"""写法：原始。INT64 定点归约实现可重复的 MoE 加权 combine。

契约：连续 expert[R,D]/weights[R]/out[T,D] 为 FP32，token_ids[R] 为 INT32；
T=1..4096、R=0..32768、D=1..512，expert在[-1,1]、weights在[0,1]。
有效token属于[0,T)，-1被忽略；out[t,d]汇总该token的加权贡献，未命中为零。
相对FP64数学参考，每元素误差不超过2e-5+2e-5*sum(abs(weights*expert))；
同输入重复调用逐位一致，输入不变；workers支持1/4/16，Host只看元数据与分配。

每个record由一个worker内的一个lane组覆盖，组内lane分摊D；贡献以2^-30为步长
最近舍入，再做INT64原子和。任意中间整数和的绝对值不超过R*2^30<=2^45。
令u=2^-24、A为真实贡献绝对值之和，误差上界为(1+u)*R*2^-31+(2*u+u*u)*A，
另有不影响契约余量的FP32下溢绝对误差。R项量化部分至多1.526e-5。
这种近似仅适用于本例给定范围和容差，不能替代要求精确浮点结果的契约。

三次同Stream原生调用依次初始化、分散累加、归并；初始化完成后才能原子更新。
记录组映射到worker内的分区；不同组仍可能共用分区，因此保留原子加。
输出归并由一个线程独占每个元素，所有分区用整数相加后只转换一次FP32。
SIMT不分配UB/L1；GM临时空间为workers*shards*T*D*8字节，最大256 MiB。
shards由lane组数和形状约束，初始化与归并开销随临时空间增长；R=0时单任务清零。
减少热点冲突会增加临时空间与固定开销，均匀、热点、全无效路由须分别测量。
验证工具链：PyPTO行为基线872df763f、CANN 9.2.0、Ascend950PR、
BiSheng 15.0.5、torch/torch_npu 2.10.0；当前结果仍须绑定源文件哈希。
"""

from functools import lru_cache

import torch
import pypto_pro.language as pl

FP32 = pl.DT_FP32
INT32 = pl.DT_INT32
INT64 = pl.DT_INT64

THREADS = 256
SCALE = 1073741824.0  # 2**30
INV_SCALE = 1.0 / SCALE


@lru_cache(maxsize=None)
def _make_zero_kernel(tokens, dims):
    @pl.vector_function(mode="simt", max_threads=THREADS)
    def zero_body(out):
        thread = pl.simt.linear_thread_idx()
        start = pl.simt.block_idx().x * THREADS + thread
        stride = pl.simt.grid_dim().x * pl.simt.block_dim().x
        zero = pl.simt.cast(0.0, FP32)
        for linear in pl.range(start, tokens * dims, stride):
            token = linear // dims
            column = linear % dims
            out[token, column] = zero

    @pl.jit(name=f"moe_combine_zero_t{tokens}_d{dims}")
    def zero_kernel(out: pl.Tensor[[tokens, dims], FP32]):
        with pl.section_vector():
            zero_body[THREADS](out)

    return zero_kernel


@lru_cache(maxsize=None)
def _make_kernels(tokens, records, dims, workers, shards, group_width):
    groups_per_block = THREADS // group_width

    @pl.vector_function(mode="simt", max_threads=THREADS)
    def initialize_body(partials):
        thread = pl.simt.linear_thread_idx()
        start = pl.simt.block_idx().x * THREADS + thread
        stride = pl.simt.grid_dim().x * pl.simt.block_dim().x
        total = workers * shards * tokens * dims
        zero = pl.simt.cast(0, INT64)
        for linear in pl.range(start, total, stride):
            column = linear % dims
            token = (linear // dims) % tokens
            shard = (linear // (dims * tokens)) % shards
            worker = linear // (dims * tokens * shards)
            partials[worker, shard, token, column] = zero

    @pl.vector_function(mode="simt", max_threads=THREADS)
    def accumulate_body(expert, token_ids, weights, partials):
        thread = pl.simt.linear_thread_idx()
        block = pl.simt.block_idx().x
        group = thread // group_width
        lane = thread % group_width
        groups = pl.simt.grid_dim().x * groups_per_block
        record_start = block * groups_per_block + group
        record_step = groups
        shard = group % shards
        for record in pl.range(record_start, records, record_step):
            token = token_ids[record]
            if token >= 0:
                weight = weights[record]
                for column in pl.range(lane, dims, group_width):
                    product = weight * expert[record, column]
                    scaled = pl.simt.rint(product * SCALE)
                    contribution = pl.simt.cast(scaled, INT64)
                    pl.simt.atomic_add(
                        partials[block, shard, token, column], contribution
                    )

    @pl.vector_function(mode="simt", max_threads=THREADS)
    def merge_body(partials, out):
        thread = pl.simt.linear_thread_idx()
        start = pl.simt.block_idx().x * THREADS + thread
        stride = pl.simt.grid_dim().x * pl.simt.block_dim().x
        total = tokens * dims
        zero = pl.simt.cast(0, INT64)
        for linear in pl.range(start, total, stride):
            token = linear // dims
            column = linear % dims
            value = zero
            for worker in pl.range(0, workers):
                for shard_id in pl.range(0, shards):
                    value = value + partials[worker, shard_id, token, column]
            out[token, column] = pl.simt.cast(value, FP32) * INV_SCALE

    @pl.jit(name=f"moe_combine_init_t{tokens}_d{dims}_w{workers}_s{shards}")
    def initialize(partials: pl.Tensor[[workers, shards, tokens, dims], INT64]):
        with pl.section_vector():
            initialize_body[THREADS](partials)

    @pl.jit(
        name=f"moe_combine_acc_t{tokens}_r{records}_d{dims}_w{workers}_s{shards}_g{group_width}"
    )
    def accumulate(
        expert: pl.Tensor[[records, dims], FP32],
        token_ids: pl.Tensor[[records], INT32],
        weights: pl.Tensor[[records], FP32],
        partials: pl.Tensor[[workers, shards, tokens, dims], INT64],
    ):
        with pl.section_vector():
            accumulate_body[THREADS](expert, token_ids, weights, partials)

    @pl.jit(name=f"moe_combine_merge_t{tokens}_d{dims}_w{workers}_s{shards}")
    def merge(
        partials: pl.Tensor[[workers, shards, tokens, dims], INT64],
        out: pl.Tensor[[tokens, dims], FP32],
    ):
        with pl.section_vector():
            merge_body[THREADS](partials, out)

    return initialize, accumulate, merge


def _shard_count(tokens, dims, group_width):
    """Bound replicated scratch while exposing several private update slots."""
    groups_per_block = THREADS // group_width
    cells = tokens * dims
    scratch_budget_cells = 262144
    by_budget = max(1, scratch_budget_cells // max(1, cells))
    return min(groups_per_block, 4, by_budget)


def run(expert, token_ids, weights, out, workers=4):
    if expert.ndim != 2 or out.ndim != 2:
        raise ValueError("expert and out must be matrices")
    if token_ids.ndim != 1 or weights.ndim != 1:
        raise ValueError("token_ids and weights must be vectors")
    records, dims = expert.shape
    tokens = out.shape[0]
    if not (1 <= tokens <= 4096 and 0 <= records <= 32768 and 1 <= dims <= 512):
        raise ValueError("T, R, or D is outside the supported range")
    if token_ids.shape != (records,) or weights.shape != (records,):
        raise ValueError("token_ids and weights must have R elements")
    if out.shape[1] != dims:
        raise ValueError("out must have shape [T, D]")
    if workers not in (1, 4, 16):
        raise ValueError("workers must be one of 1, 4, or 16")
    tensors = (expert, token_ids, weights, out)
    if not all(tensor.is_contiguous() for tensor in tensors):
        raise ValueError("all tensors must be contiguous")
    if len({tensor.device for tensor in tensors}) != 1:
        raise ValueError("all tensors must be on the same device")
    if expert.dtype != torch.float32 or weights.dtype != torch.float32:
        raise ValueError("expert and weights must be FP32")
    if token_ids.dtype != torch.int32:
        raise ValueError("token_ids must be INT32")
    if out.dtype != torch.float32:
        raise ValueError("out must be FP32")

    if records == 0:
        _make_zero_kernel(tokens, dims)[None, workers](out)
        return

    if dims >= 64:
        group_width = 64
    elif dims >= 16:
        group_width = 16
    elif dims >= 4:
        group_width = 4
    else:
        group_width = 1
    shards = _shard_count(tokens, dims, group_width)
    initialize, accumulate, merge = _make_kernels(
        tokens, records, dims, workers, shards, group_width
    )
    partials = torch.empty(
        (workers, shards, tokens, dims), dtype=torch.int64, device=out.device
    )
    initialize[None, workers](partials)
    accumulate[None, workers](expert, token_ids, weights, partials)
    merge[None, workers](partials, out)
