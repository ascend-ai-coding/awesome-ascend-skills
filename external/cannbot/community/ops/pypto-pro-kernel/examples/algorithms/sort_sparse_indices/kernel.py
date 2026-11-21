# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

"""写法：混合。有界 INT32 索引排序，独立缓冲与精确 key 逆变换。

example-query: {"aliases": ["sparse index sorting", "sort index values", "稀疏索引排序"], "term_groups": [["sort", "sorting", "排序"], ["index", "indices", "索引"]], "facets": {"source_dtype": ["INT32"], "destination_dtype": ["INT32"]}, "exclude": [{"term_groups": [["arbitrary int32", "full int32 range", "任意INT32"]], "reason": "Both key encodings are qualified only for values in [0,2**24]."}, {"term_groups": [["associated payload", "独立payload", "附带payload"]], "reason": "This example emits index values, not independently associated payloads."}]}

连续、32 B 对齐的 INT32 [R,512]，R>0，值在 [0,2**24]；允许重复，不保证稳定顺序。
默认 ordered_bits 将 x-0x40800000 的 INT32 位重解释为 FP32：其位区间
0xBF800000..0xC0800000 对应严格递减的有限负数 -1..-4。硬件降序排序后，
按位恢复并加回偏移得到精确升序索引，两次整数加法均不溢出。
numeric 控制使用精确的 -FP32(x)，再做逆数值转换。两条路径使用同一数学参考。

目标值可从 key 精确恢复，sort32 的 UINT32 tie 槽只需在每个 worker 初始化一次。
独立的 input/output、key、tie 各 2 KiB，三个 record 缓冲各 4 KiB，共 20 KiB UB。
它们有独立 mutex ID；auto_mutex 管理 load、VF、排序与 store 及下一行复用的依赖。
输入不被输出写回复用，不额外加入全管线 barrier；VF 内没有依赖自身存储的回读。
每个 worker 跨步拥有完整行。完整调用为一个 native task，默认使用合法的自动 launch。

sort32 产生 16 个 32-record run；两级 mrgsort.block_len=64/256 的单位为 32-bit 槽，
而非 record。独立 payload 不能从 key 恢复时仍须保留其载体和最后读取边界。
范围、排序单位及传输成本见
[sort records](../../../references/patterns/ordered-selection.md#native-sort-records-and-index-payloads)。
"""

from functools import lru_cache

import torch
import pypto_pro.language as pl
from pypto_pro.language import Vf as vf


I32, U32, F32 = pl.DT_INT32, pl.DT_UINT32, pl.DT_FP32
KEY_BIAS = 0x40800000


@pl.vector_function
def _zero_ties(ties):
    lanes = vf.create_mask(pattern=pl.MaskPattern.ALL, dtype=U32)
    zero = vf.full(0, lanes, dtype=U32)
    for block in pl.range(8):
        vf.store_align(ties, zero, lanes, [0, block * 64])


@pl.vector_function
def _bit_keys(source, keys):
    lanes = vf.create_mask(pattern=pl.MaskPattern.ALL, dtype=I32)
    for block in pl.range(8):
        value = vf.load_align(source, [0, block * 64])
        bits = vf.adds(value, -KEY_BIAS, lanes)
        key = vf.bit_cast(bits, dtype=F32)
        vf.store_align(keys, key, lanes, [0, block * 64])


@pl.vector_function
def _numeric_keys(source, keys):
    lanes = vf.create_mask(pattern=pl.MaskPattern.ALL, dtype=I32)
    for block in pl.range(8):
        value = vf.load_align(source, [0, block * 64])
        numeric = vf.astype(value, lanes, dtype=F32)
        key = vf.neg(numeric, lanes)
        vf.store_align(keys, key, lanes, [0, block * 64])


@pl.vector_function
def _recover_bit_keys(records, output):
    lanes = vf.create_mask(pattern=pl.MaskPattern.ALL, dtype=I32)
    for block in pl.range(8):
        key, _ = vf.load_align(records, [0, block * 128], dist=pl.LoadDist.DINTLV_B32)
        bits = vf.bit_cast(key, dtype=I32)
        value = vf.adds(bits, KEY_BIAS, lanes)
        vf.store_align(output, value, lanes, [0, block * 64])


@pl.vector_function
def _recover_numeric_keys(records, output):
    lanes = vf.create_mask(pattern=pl.MaskPattern.ALL, dtype=I32)
    for block in pl.range(8):
        key, _ = vf.load_align(records, [0, block * 128], dist=pl.LoadDist.DINTLV_B32)
        numeric = vf.neg(key, lanes)
        value = vf.astype(numeric, lanes, dtype=I32)
        vf.store_align(output, value, lanes, [0, block * 64])


@lru_cache(maxsize=None)
def make_kernel(key_encoding="ordered_bits"):
    if key_encoding not in ("ordered_bits", "numeric"):
        raise ValueError("key_encoding must be ordered_bits or numeric")
    encode = _bit_keys if key_encoding == "ordered_bits" else _numeric_keys
    recover = (
        _recover_bit_keys if key_encoding == "ordered_bits" else _recover_numeric_keys
    )

    @pl.jit(arch="a5", auto_mutex=True, name=f"sort_indices_{key_encoding}")
    def kernel(
        indices: pl.Tensor[[pl.DYNAMIC, 512], I32],
        output: pl.Tensor[[pl.DYNAMIC, 512], I32],
    ):
        value_type = pl.TileType(
            shape=[1, 512], dtype=I32, target_memory=pl.MemorySpace.Vec
        )
        key_type = pl.TileType(
            shape=[1, 512], dtype=F32, target_memory=pl.MemorySpace.Vec
        )
        tie_type = pl.TileType(
            shape=[1, 512], dtype=U32, target_memory=pl.MemorySpace.Vec
        )
        record_type = pl.TileType(
            shape=[1, 1024], dtype=F32, target_memory=pl.MemorySpace.Vec
        )
        source_group = pl.make_tile_group(type=value_type, addrs=0, mutex_ids=[0])
        key_group = pl.make_tile_group(type=key_type, addrs=2048, mutex_ids=[1])
        tie_group = pl.make_tile_group(type=tie_type, addrs=4096, mutex_ids=[2])
        run_group = pl.make_tile_group(type=record_type, addrs=6144, mutex_ids=[3])
        merge_group = pl.make_tile_group(type=record_type, addrs=10240, mutex_ids=[4])
        sorted_group = pl.make_tile_group(type=record_type, addrs=14336, mutex_ids=[5])
        output_group = pl.make_tile_group(type=value_type, addrs=18432, mutex_ids=[6])
        with pl.section_vector():
            source = source_group.current()
            keys = key_group.current()
            ties = tie_group.current()
            runs = run_group.current()
            merged = merge_group.current()
            sorted_records = sorted_group.current()
            result = output_group.current()
            _zero_ties(ties)
            for row in pl.range(
                pl.get_block_idx(), indices.shape[0], pl.get_block_num()
            ):
                pl.load(source, indices, [row, 0])
                encode(source, keys)
                pl.sort32(runs, keys, ties)
                pl.mrgsort(merged, runs, block_len=64)
                pl.mrgsort(sorted_records, merged, block_len=256)
                recover(sorted_records, result)
                pl.store(output, result, [row, 0])

    return kernel


sort_kernel = make_kernel()


def sort_sparse_indices(indices, *, key_encoding="ordered_bits"):
    """Caller guarantees [0,2**24]; preserve the input and return sorted rows."""
    if indices.dtype != torch.int32 or indices.ndim != 2:
        raise ValueError("expected INT32 [rows,512]")
    if indices.shape[0] <= 0 or indices.shape[1] != 512 or not indices.is_contiguous():
        raise ValueError("expected nonempty contiguous [rows,512]")
    if indices.device.type != "npu":
        raise ValueError("expected an NPU tensor")
    if indices.data_ptr() % 32:
        raise ValueError("expected a 32-byte-aligned input")
    kernel = make_kernel(key_encoding)
    output = torch.empty_like(indices)
    kernel(indices, output)
    return output
