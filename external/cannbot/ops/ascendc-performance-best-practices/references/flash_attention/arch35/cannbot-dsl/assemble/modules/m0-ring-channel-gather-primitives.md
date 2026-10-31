> FA 族 CANNBot-DSL 第 4 层「积木模块」的 SFA 实例，按类边界拆分为独立文件以便逐模块复用；定位与使用规则见 [组装层索引](../index.md)。

# 积木模块 M0：跨核环形通道与 gather/NZ 基础原语（源 L1-186）

> **为什么**：全部是形状/深度/滞后常量（常数四分类里的"代码常量"）。`MM_FEATURE_CHUNK` 与 `TILE_S2` 刻意解耦——前者是 L0A/L0B 内维（D=512=8×64），后者是 KV token tile；解耦后缩小 ds 块（B_TOPK=64）不会破坏 Cube 通路。改任何常量前先过 [容量方程](../../optimize/3-capacity-equations.md) 的整除链。
```python
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.


"""Sparse Flash Attention for MLA-absorb.

The kernel gathers sparse KV rows, computes tiled QK attention with online
softmax, and accumulates the absorbed Value output. It supports FP16/BF16,
BSND/TND queries, and BSND/TND/PA_BSND KV storage.
"""

from __future__ import annotations
from cannbotdsl.lang.host import host
import cannbotdsl

import torch

from sparse_flash_attention_validation import (
    DIM_NOPE,
    DIM_ROPE,
    INT64_MAX,
    SPARSE_MODE_DENSE,
    SPARSE_MODE_RIGHT_DOWN_CAUSAL,
    SUPPORTED_ATTENTION_MODE,
    SUPPORTED_SPARSE_BLOCK_SIZE,
    validate_sparse_flash_attention_args,
)

from cannbotdsl import (
    TensorViewSpec, ViewLayout,
    copy_options,
    MemLoc,
    PIPE,
    Tensor,
    dtypes,
    matmul,
    mem_copy,
)
from cannbotdsl import reg as rr
from cannbotdsl.buffer import Buffer
from cannbotdsl.ring_buffer import RingBuffer


class ChannelKind:
    """Kept so existing call sites still name the cross-core slots.

    RingBuffer synchronization is derived from the accesses themselves.
    """

    CrossCore = "cross_core"


class Channel:
    """Old produce/consume ring, backed by RingBuffer on this package.

    produce() advances the write cursor (head_next). consume() advances the
    read cursor (tail_next). kind is accepted and ignored.
    """

    def __init__(
        self, mem_loc, shape, dtype, depth=1, data_format=None, n1_pad=0, kind=None, view=None,
    ):
        del kind
        extra = {}
        if data_format is not None:
            extra["data_format"] = data_format
        if n1_pad:
            extra["n1_pad"] = n1_pad
        if view is not None:
            self._ring = RingBuffer(mem_loc, depth=depth, view=view)
        else:
            self._ring = RingBuffer(mem_loc, shape, dtype, depth=depth, **extra)

    def produce(self):
        return self._ring.head_next()

    def consume(self):
        return self._ring.tail_next()
from cannbotdsl.lang.constexpr import const_expr, range_constexpr
from cannbotdsl.lang.jit import jit
from cannbotdsl.lang.kernel import kernel
from cannbotdsl.lang.vf import vf
from cannbotdsl.ops.arch import get_block_idx, get_subblock_id
from cannbotdsl.ops.cube import set_mmad_direction
from cannbotdsl.ops.sync import (
    cube_sync_block_wait,
    vec_sync_block_arrive,
)
from cannbotdsl.tensor import (
    make_tiler,
    reinterpret,
    tile_slice,
)
from cannbotdsl.types.delay_line import DelayLineGroup


INT32_MAX = 2_147_483_647

INVALID_SPARSE_INDEX = -1
EMPTY_QUERY_ID = -1

TILE_M = 64
TILE_S2 = 128
# Width of one matmul D/K feature column block (DIM_NOPE split into 4). This is
# the L0A/L0B inner dim and the PV output D-chunk, decoupled from the KV token
# tile TILE_S2 so TILE_S2 can shrink (ds B_TOPK=64) without breaking the cube.
MM_FEATURE_CHUNK = 64
CUBE_K_TILE = 128
GATHER_CHUNK_ROWS = 16
AIV_ROW_CAPACITY = 32
AIV_COUNT = 2
SPARSE_INDEX_PAIR_SIZE = 2
DIM_QK = DIM_NOPE + DIM_ROPE

PIPELINE_SLOT_COUNT = 3
SOFTMAX_STATE_SLOTS = 2
PIPELINE_STAGE_COUNT = 4
PIPELINE_DRAIN_TICKS = PIPELINE_STAGE_COUNT - 1

BYTES_PER_16BIT_ELEMENT = 2
KV_ROW_BYTES = DIM_NOPE * BYTES_PER_16BIT_ELEMENT
NZ_BLOCK_BYTES = 32
VECTOR_REGISTER_BITS = 2048
FP32_BITS = 32
VL_T = VECTOR_REGISTER_BITS // FP32_BITS
SOFTMAX_NEG_INF = -1.0e30
EMPTY_SOFTMAX_THRESHOLD = -1.0e29


@jit
def _copy_kv_token(stage, key_token, rope_token, row):
    """Copy one sparse KV row with NoPE followed by RoPE in UB."""
    dst = tile_slice(stage, (1, DIM_QK), (row, 0))
    mem_copy(tile_slice(dst[None, :DIM_NOPE], (1, DIM_NOPE), (0, 0)), key_token)
    mem_copy(tile_slice(dst[None, DIM_NOPE:DIM_QK], (1, DIM_ROPE), (0, 0)), rope_token)


@jit
def _copy_kv_token_no_rope(stage, key_token, row):
    """Copy one sparse K row when RoPE is disabled."""
    dst = tile_slice(stage, (1, DIM_NOPE), (row, 0))
    mem_copy(dst, key_token)


@jit
def _copy_kv_pair(stage, key0, key1, rope0, rope1, pair_index):
    """Copy two sparse KV rows together, separately for NoPE and RoPE."""
    key_dst = tile_slice(
        stage[None, :DIM_NOPE], (SPARSE_INDEX_PAIR_SIZE, DIM_NOPE),
        (pair_index, 0),
    )
    rope_dst = tile_slice(
        stage[None, DIM_NOPE:DIM_QK], (SPARSE_INDEX_PAIR_SIZE, DIM_ROPE),
        (pair_index, 0),
    )
    mem_copy(key_dst, (key0, key1))
    mem_copy(rope_dst, (rope0, rope1))


@jit
def _copy_kv_pair_no_rope(stage, key0, key1, pair_index):
    """Copy two sparse K rows when RoPE is disabled."""
    dst = tile_slice(
        stage, (SPARSE_INDEX_PAIR_SIZE, DIM_NOPE), (pair_index, 0)
    )
    mem_copy(dst, (key0, key1))


@jit
def _clamp_nonneg(value: int) -> int:
    """Clamp a dynamic mask length before converting it to an unsigned value."""
    return 0 if value < 0 else value


def _nz_params(p_ub):
    """Derive NZ fractal addressing from p_ub physical layout stride."""
    s = p_ub.physical_stride
    s_n1, s_m1, s_m0, s_n0 = s[0], s[1], s[2], s[3]
    m0 = s_m1 // s_m0
    n0 = s_m0 // s_n0
    return m0, s_m1, s_m0, s_n1 // n0
```

