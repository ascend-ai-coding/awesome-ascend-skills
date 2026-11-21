# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

"""写法：原始。门控外积更新与查询，矩阵状态在片上跨时间步保留。

机制：[不变量改写](../../../references/api/numerics.md#prove-an-invariant-before-removing-work)与[特例资格](../../../references/workflows/authoring.md#generalization-and-specialization)。
本例实际维护完整矩阵，只示范外积递推的驻留与 D 容量/R、T 实际范围；秩一矩阵多项式降为标量需要另一份算法与数值证明。

example-query: {"aliases": ["gated outer-product recurrence", "on-chip recurrent matrix state", "门控外积状态递推"], "facets": {"source_dtype": ["FP16", "FP32"], "destination_dtype": ["FP32"]}}

q/k/v 为 FP16 [R,T,D]，g 为 FP32 [R,T]，s0 为 FP32 [R,D,D]。
状态从 s0 开始，每步先 S = g*S + k[:,None]*v[None,:]，再 y = q^T@S。
返回 FP32 y[R,T,D] 与最终状态；输入不变，Host 仅检查元数据与分配。
范围为 R=1..33、T=1..257、D∈{16,32,64,128}，q/k/v/s0 绝对值<=0.125，g∈[0,1]。
独立 FP64 参考的 y 容差为 atol=2e-3/rtol=1e-4，最终状态为 atol=rtol=1e-4。

worker 独占 r，完整调用只有一次 native launch；每个 r 的状态只读入、写出一次。
每步输入行及输出累加器使用独立 Vec 缓冲组，auto_mutex 管理跨 VF 调用的读写依赖。
D 决定状态 tile 的物理容量，R/T 使用运行时元数据；padded_dim=max(64,D)。
峰值 Vec 字节为 4*D*padded_dim + 10*padded_dim，最大 66816 B，无 GM scratch。
UNPK 扩精度与后续 FP32 指令复用相同物理谓词。逐状态行的 VF 调用是正确性起点，
调用开销与其他计算映射需要单独计时，此例不声称性能优势。
验证工具链为 PyPTO cbd3c4a778、CANN 9.2.0、Ascend950PR；运行证据须绑定当前源码与安装身份。
"""

from functools import lru_cache

import torch
import pypto_pro.language as pl
from pypto_pro.language import Vf as vf  # noqa: N813

FP16 = pl.DT_FP16
FP32 = pl.DT_FP32
VECTOR_LANES = 64
MAX_WORKERS = 32


@pl.vector_function
def _zero_row(tile, cols):
    for col in pl.range(0, cols, VECTOR_LANES):
        active = pl.min(VECTOR_LANES, cols - col)
        mask = vf.update_mask(active, dtype=FP32)
        zeros = vf.full(0.0, mask, dtype=FP32)
        vf.store_align(tile, zeros, mask, [0, col])


@pl.vector_function
def _update_state_row_chunk(state, v, accum, row, col, active, gate, k_i, q_i):
    mask32 = vf.update_mask(active, dtype=FP32)

    old = vf.load_align(state, [row, col])
    raw_v = vf.load_align(v, [0, col], dist=pl.LoadDist.UNPK)
    v32 = vf.astype(raw_v, mask32, dtype=FP32, layout=pl.CastLayout.ZERO)

    gated = vf.muls(old, gate, mask32)
    outer = vf.muls(v32, k_i, mask32)
    updated = vf.add(gated, outer, mask32)
    vf.store_align(state, updated, mask32, [row, col])

    partial = vf.muls(updated, q_i, mask32)
    previous_total = vf.load_align(accum, [0, col])
    next_total = vf.add(previous_total, partial, mask32)
    vf.store_align(accum, next_total, mask32, [0, col])


@lru_cache(maxsize=4)
def _kernel_for_dim(dim):
    if dim not in (16, 32, 64, 128):
        raise ValueError(f"unsupported D={dim}")
    padded_dim = max(VECTOR_LANES, dim)
    state_bytes = dim * padded_dim * 4
    input_bytes = padded_dim * 2
    q_addr = state_bytes
    k_addr = q_addr + input_bytes
    v_addr = k_addr + input_bytes
    accum_addr = v_addr + input_bytes

    @pl.jit(auto_mutex=True, name=f"gated_outer_state_d{dim}")
    def kernel(
        q: pl.Tensor[[pl.DYNAMIC, pl.DYNAMIC, dim], FP16],
        k: pl.Tensor[[pl.DYNAMIC, pl.DYNAMIC, dim], FP16],
        v: pl.Tensor[[pl.DYNAMIC, pl.DYNAMIC, dim], FP16],
        g: pl.Tensor[[pl.DYNAMIC, pl.DYNAMIC], FP32],
        s0: pl.Tensor[[pl.DYNAMIC, dim, dim], FP32],
        y: pl.Tensor[[pl.DYNAMIC, pl.DYNAMIC, dim], FP32],
        s_final: pl.Tensor[[pl.DYNAMIC, dim, dim], FP32],
    ):
        state_group = pl.make_tile_group(
            type=pl.TileType(
                shape=[dim, padded_dim], dtype=FP32, target_memory=pl.MemorySpace.Vec
            ),
            addrs=0,
            mutex_ids="auto",
            depth=1,
        )
        q_group = pl.make_tile_group(
            type=pl.TileType(
                shape=[1, padded_dim], dtype=FP16, target_memory=pl.MemorySpace.Vec
            ),
            addrs=q_addr,
            mutex_ids="auto",
            depth=1,
        )
        k_group = pl.make_tile_group(
            type=pl.TileType(
                shape=[1, padded_dim], dtype=FP16, target_memory=pl.MemorySpace.Vec
            ),
            addrs=k_addr,
            mutex_ids="auto",
            depth=1,
        )
        v_group = pl.make_tile_group(
            type=pl.TileType(
                shape=[1, padded_dim], dtype=FP16, target_memory=pl.MemorySpace.Vec
            ),
            addrs=v_addr,
            mutex_ids="auto",
            depth=1,
        )
        accum_group = pl.make_tile_group(
            type=pl.TileType(
                shape=[1, padded_dim], dtype=FP32, target_memory=pl.MemorySpace.Vec
            ),
            addrs=accum_addr,
            mutex_ids="auto",
            depth=1,
        )

        with pl.section_vector():
            state = state_group.current()
            q_tile = q_group.current()
            k_tile = k_group.current()
            v_tile = v_group.current()
            accum = accum_group.current()

            pl.set_validshape(state, [dim, dim])
            pl.set_validshape(q_tile, [1, dim])
            pl.set_validshape(k_tile, [1, dim])
            pl.set_validshape(v_tile, [1, dim])
            pl.set_validshape(accum, [1, dim])

            for r in pl.range(pl.get_block_idx(), q.shape[0], pl.get_block_num()):
                pl.load(state, s0, [r, 0, 0])
                for t in pl.range(0, q.shape[1]):
                    pl.load(q_tile, q, [r, t, 0])
                    pl.load(k_tile, k, [r, t, 0])
                    pl.load(v_tile, v, [r, t, 0])
                    _zero_row(accum, dim)

                    gate = pl.getval(g, r * q.shape[1] + t)
                    for i in pl.range(0, dim):
                        k_i = pl.astype(pl.getval(k_tile, i), FP32)
                        q_i = pl.astype(pl.getval(q_tile, i), FP32)
                        for col in pl.range(0, dim, VECTOR_LANES):
                            active = pl.min(VECTOR_LANES, dim - col)
                            _update_state_row_chunk(
                                state, v_tile, accum, i, col, active, gate, k_i, q_i
                            )

                    pl.store(y, accum, [r, t, 0])

                pl.store(s_final, state, [r, 0, 0])

    return kernel


def run(q, k, v, g, s0):
    """Run the recurrence; host work is metadata validation and output allocation."""
    if q.ndim != 3:
        raise ValueError("q must have shape [R,T,D]")
    r_count, time_steps, dim = q.shape
    if not (1 <= r_count <= 33 and 1 <= time_steps <= 257):
        raise ValueError("R and T are outside the supported domain")
    if dim not in (16, 32, 64, 128):
        raise ValueError("D must be one of 16, 32, 64, 128")
    if tuple(k.shape) != tuple(q.shape) or tuple(v.shape) != tuple(q.shape):
        raise ValueError("q, k, and v must have matching [R,T,D] shapes")
    if tuple(g.shape) != (r_count, time_steps):
        raise ValueError("g must have shape [R,T]")
    if tuple(s0.shape) != (r_count, dim, dim):
        raise ValueError("s0 must have shape [R,D,D]")

    expected_dtypes = (
        torch.float16,
        torch.float16,
        torch.float16,
        torch.float32,
        torch.float32,
    )
    inputs = (q, k, v, g, s0)
    if tuple(t.dtype for t in inputs) != expected_dtypes:
        raise TypeError("q/k/v must be FP16 and g/s0 must be FP32")
    if any(not tensor.is_contiguous() for tensor in inputs):
        raise ValueError("all inputs must be contiguous")
    if any(tensor.device != q.device for tensor in inputs):
        raise ValueError("all inputs must be on the same device")
    if q.device.type != "npu":
        raise ValueError("run requires NPU tensors")

    y = torch.empty((r_count, time_steps, dim), dtype=torch.float32, device=q.device)
    s_final = torch.empty((r_count, dim, dim), dtype=torch.float32, device=q.device)
    workers = min(r_count, MAX_WORKERS)
    _kernel_for_dim(dim)[None, workers](q, k, v, g, s0, y, s_final)
    return y, s_final
