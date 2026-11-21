# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

"""写法：原始。FP16 向量的两两平方距离，差值、乘法与归约使用 FP32。

example-query: {"aliases": ["pairwise squared distance", "squared Euclidean distance", "两两平方欧氏距离"], "facets": {"source_dtype": ["FP16"], "destination_dtype": ["FP32"]}}

输入为连续 x[P,D]、z[Q,D]；输出 FP32 y[P,Q] = sum_d (x-z)^2。
范围为 P/Q=1..257、D=1..1025，输入绝对值<=8；相同向量返回精确零。
FP64 参考使用实际量化输入，容差为 atol=1e-5、rtol=2e-5。
逐项扩精度后计算差值，避免近邻数据中范数展开的消减误差。

run_rows 每个 pair 读取一行 z 并调用 VF，打包 8 个标量结果后写回；
run_batched 一次搬运最多 16 行 z，共享当前 x 行并在一个 VF 调用内计算。
两种路径均以 UNPK 和 FP32 谓词扩精度，以 FIRST_ELEMENT_B32 打包标量输出。
各 worker 独占输出片段，三个单槽缓冲组使用独立地址与 mutex IDs。
最大 Vec 有效分配为 37056 B，最高地址端点为 0xA040，无 GM scratch。

run 使用当前平台 vector_core_num 限制 worker 数；批量路径的工作组不足以覆盖
这些 worker 时采用逐行路径，避免输入复用缩减并行度。此规则只是一种配置选择，
不同设备和工作负载仍需完整调用计时。原生接口与配置在 PyPTO cbd3c4a778、
CANN 9.2.0、Ascend950PR 验证，当前证据须绑定源码与安装身份。
"""

import torch
import torch_npu  # noqa: F401
import pypto_pro.language as pl
from pypto_pro.language import Vf as vf
from pypto_pro.runtime.platform import get_platform_info


FP16 = pl.DT_FP16
FP32 = pl.DT_FP32
VL = 64
PAD_D = 1088
ROW_Q_BLOCK = 8
BATCH_Q_BLOCK = 16


@pl.vector_function
def _distance_pair(x_tile, z_tile, out_tile, dim: pl.DT_INT64, q_slot: pl.DT_INT64):
    """Compute one pair into a slot of the current output tile."""
    all_lanes = vf.create_mask(pattern=pl.MaskPattern.ALL, dtype=FP32)
    total = vf.full(0.0, all_lanes, dtype=FP32)
    for d in pl.range(0, dim, VL):
        active = pl.min(VL, dim - d)
        mask = vf.update_mask(active, dtype=FP32)
        x16 = vf.load_align(x_tile, d, dist=pl.LoadDist.UNPK)
        z16 = vf.load_align(z_tile, d, dist=pl.LoadDist.UNPK)
        xv = vf.astype(x16, mask, dtype=FP32, layout=pl.CastLayout.ZERO)
        zv = vf.astype(z16, mask, dtype=FP32, layout=pl.CastLayout.ZERO)
        delta = vf.sub(xv, zv, mask)
        square = vf.mul(delta, delta, mask)
        part = vf.reduce_sum(square, mask)
        total = vf.add(total, part, all_lanes)

    first = vf.update_mask(1, dtype=FP32)
    vf.store_align(
        out_tile,
        total,
        first,
        [0, q_slot],
        dist=pl.StoreDist.FIRST_ELEMENT_B32,
    )


@pl.vector_function
def _distance_batch(x_tile, z_tile, out_tile, dim: pl.DT_INT64, q_count: pl.DT_INT64):
    """Reduce a loaded block of right-hand rows against one resident row."""
    all_lanes = vf.create_mask(pattern=pl.MaskPattern.ALL, dtype=FP32)
    first = vf.update_mask(1, dtype=FP32)
    for q in pl.range(q_count):
        total = vf.full(0.0, all_lanes, dtype=FP32)
        for d in pl.range(0, dim, VL):
            active = pl.min(VL, dim - d)
            mask = vf.update_mask(active, dtype=FP32)
            x16 = vf.load_align(x_tile, d, dist=pl.LoadDist.UNPK)
            z16 = vf.load_align(z_tile, [q, d], dist=pl.LoadDist.UNPK)
            xv = vf.astype(x16, mask, dtype=FP32, layout=pl.CastLayout.ZERO)
            zv = vf.astype(z16, mask, dtype=FP32, layout=pl.CastLayout.ZERO)
            delta = vf.sub(xv, zv, mask)
            square = vf.mul(delta, delta, mask)
            part = vf.reduce_sum(square, mask)
            total = vf.add(total, part, all_lanes)
        vf.store_align(
            out_tile,
            total,
            first,
            [0, q],
            dist=pl.StoreDist.FIRST_ELEMENT_B32,
        )


@pl.jit(auto_mutex=True, name="pairwise_distance_rows_q8")
def _rows_kernel(
    x: pl.Tensor[[pl.DYNAMIC, pl.DYNAMIC], FP16],
    z: pl.Tensor[[pl.DYNAMIC, pl.DYNAMIC], FP16],
    y: pl.Tensor[[pl.DYNAMIC, pl.DYNAMIC], FP32],
):
    x_group = pl.make_tile_group(
        type=pl.TileType(
            shape=[1, PAD_D],
            dtype=FP16,
            target_memory=pl.MemorySpace.Vec,
            valid_shape=[-1, -1],
        ),
        addrs=[0x0000],
        mutex_ids=[0],
    )
    z_group = pl.make_tile_group(
        type=pl.TileType(
            shape=[1, PAD_D],
            dtype=FP16,
            target_memory=pl.MemorySpace.Vec,
            valid_shape=[-1, -1],
        ),
        addrs=[0x1000],
        mutex_ids=[1],
    )
    out_group = pl.make_tile_group(
        type=pl.TileType(
            shape=[1, ROW_Q_BLOCK],
            dtype=FP32,
            target_memory=pl.MemorySpace.Vec,
            valid_shape=[-1, -1],
        ),
        addrs=[0x2000],
        mutex_ids=[2],
    )

    with pl.section_vector():
        p = x.shape[0]
        q = z.shape[0]
        dim = x.shape[1]
        q_groups = (q + ROW_Q_BLOCK - 1) // ROW_Q_BLOCK
        total_groups = p * q_groups
        for task in pl.range(pl.get_block_idx(), total_groups, pl.get_block_num()):
            row = task // q_groups
            q_group = task % q_groups
            q_start = q_group * ROW_Q_BLOCK
            q_count = pl.min(ROW_Q_BLOCK, q - q_start)

            x_tile = x_group.next()
            pl.set_validshape(x_tile, [1, dim])
            pl.load(x_tile, x, [row, 0])
            out_tile = out_group.next()
            pl.set_validshape(out_tile, [1, q_count])
            for q_local in pl.range(q_count):
                z_tile = z_group.next()
                pl.set_validshape(z_tile, [1, dim])
                pl.load(z_tile, z, [q_start + q_local, 0])
                _distance_pair(x_tile, z_tile, out_tile, dim, q_local)
            pl.store(y, out_tile, [row, q_start])


@pl.jit(auto_mutex=True, name="pairwise_distance_batched_q16")
def _batched_kernel(
    x: pl.Tensor[[pl.DYNAMIC, pl.DYNAMIC], FP16],
    z: pl.Tensor[[pl.DYNAMIC, pl.DYNAMIC], FP16],
    y: pl.Tensor[[pl.DYNAMIC, pl.DYNAMIC], FP32],
):
    x_group = pl.make_tile_group(
        type=pl.TileType(
            shape=[1, PAD_D],
            dtype=FP16,
            target_memory=pl.MemorySpace.Vec,
            valid_shape=[-1, -1],
        ),
        addrs=[0x0000],
        mutex_ids=[0],
    )
    z_group = pl.make_tile_group(
        type=pl.TileType(
            shape=[BATCH_Q_BLOCK, PAD_D],
            dtype=FP16,
            target_memory=pl.MemorySpace.Vec,
            valid_shape=[-1, -1],
        ),
        addrs=[0x1000],
        mutex_ids=[1],
    )
    out_group = pl.make_tile_group(
        type=pl.TileType(
            shape=[1, BATCH_Q_BLOCK],
            dtype=FP32,
            target_memory=pl.MemorySpace.Vec,
            valid_shape=[-1, -1],
        ),
        addrs=[0xA000],
        mutex_ids=[2],
    )

    with pl.section_vector():
        p = x.shape[0]
        q = z.shape[0]
        dim = x.shape[1]
        q_groups = (q + BATCH_Q_BLOCK - 1) // BATCH_Q_BLOCK
        total_groups = p * q_groups
        for task in pl.range(pl.get_block_idx(), total_groups, pl.get_block_num()):
            row = task // q_groups
            q_group = task % q_groups
            q_start = q_group * BATCH_Q_BLOCK
            q_count = pl.min(BATCH_Q_BLOCK, q - q_start)

            x_tile = x_group.next()
            pl.set_validshape(x_tile, [1, dim])
            pl.load(x_tile, x, [row, 0])
            z_tile = z_group.next()
            pl.set_validshape(z_tile, [q_count, dim])
            pl.load(z_tile, z, [q_start, 0])
            out_tile = out_group.next()
            pl.set_validshape(out_tile, [1, q_count])
            _distance_batch(x_tile, z_tile, out_tile, dim, q_count)
            pl.store(y, out_tile, [row, q_start])


def _validate_inputs(x, z):
    if x.device.type != "npu" or z.device != x.device:
        raise ValueError("x and z must be on the same NPU")
    if x.dtype != torch.float16 or z.dtype != torch.float16:
        raise ValueError("x and z must be FP16")
    if x.ndim != 2 or z.ndim != 2 or x.shape[1] != z.shape[1]:
        raise ValueError("x and z must be rank-2 with the same feature width")
    if not x.is_contiguous() or not z.is_contiguous():
        raise ValueError("x and z must be contiguous")
    p, dim = x.shape
    q = z.shape[0]
    if not (1 <= p <= 257 and 1 <= q <= 257 and 1 <= dim <= 1025):
        raise ValueError("supported shapes: 1<=P,Q<=257 and 1<=D<=1025")


def _run_with(kernel, q_block, x, z):
    _validate_inputs(x, z)
    p, dim = x.shape
    q = z.shape[0]
    y = torch.empty((p, q), dtype=torch.float32, device=x.device)
    groups = (q + q_block - 1) // q_block
    workers = min(int(get_platform_info().vector_core_num), p * groups)
    kernel[None, workers](x, z, y)
    return y


def run_rows(x, z):
    """Read one z row and invoke VF per pair, packing row outputs."""
    return _run_with(_rows_kernel, ROW_Q_BLOCK, x, z)


def run_batched(x, z):
    """Load and compute a Q16 batch of right-hand rows."""
    return _run_with(_batched_kernel, BATCH_Q_BLOCK, x, z)


def run(x, z):
    """Return FP32 distances, batching only when enough groups remain active."""
    _validate_inputs(x, z)
    p, _ = x.shape
    q = z.shape[0]
    cores = int(get_platform_info().vector_core_num)
    candidate_groups = p * ((q + BATCH_Q_BLOCK - 1) // BATCH_Q_BLOCK)
    if candidate_groups < cores:
        return _run_with(_rows_kernel, ROW_Q_BLOCK, x, z)
    return _run_with(_batched_kernel, BATCH_Q_BLOCK, x, z)
