# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

"""写法：语法糖。设备端变长分组 GEMM，小组复用权重、长组跨 worker 分片。

契约：连续 FP16 A[T,K]/W[G,K,N]，输入有限且在[-1,1]；INT32 offsets[G+1]单调、首0末T，输出 FP32 C[T,N]。T=1..4096、G=1..32，K/N各为
16/32/64/128/256。Host 只检查元数据与调度，不读取分组内容。每元素误差上限为1e-3+3e-4*sum(abs(A*W))，参考消费实际 FP16 输入。
分工：默认 M块128，K步长min(K,128)。设备读取组长；tile数大于workers的长组按M tile轮转分配，其他组由group_id%workers的单一owner处理。一个worker进入某组后只载入一次该组W并复用。空组
无读写，所有有效输出行恰由一个worker写入。短组保留完整所有权有复用收益，也可能受空组位置和组间长短分布影响；该条件不是通用最优调度规则。
物理资源：A_L1位于0，最大65536 B；W_L1位于0x20000，最大131072 B，Mat地址高水位262144 B。L0A最大32768 B、L0B最大65536 B、Acc最大131072 B，均为单槽，三个L0空间各自从0
开始；不需要GM scratch。K/N为16的倍数，M尾部在L1装载前设置有效形状，并通过pl.gemm(m=...)限制计算与写回；被收窄的L0A和Acc使用compact=1。
数值与同步：乘积绝对值至多1，K项绝对和至多256，FP32累加不会溢出；舍入误差由test.py的独立FP64项和判据验收。pl.gemm(splitk=...)在首段初始化、后续累加同一Acc。auto_mutex管理MTE2、
MTE1、M、FIX及槽位复用，完整调用为一个native task，无跨worker数据依赖。
验证：边界测试覆盖workers=1/4、全部25个K/N组合、非零存储偏移、输入输出守卫、输出毒值和同地址offsets更新。默认块形状与调度来自完整调用比较，其他输入分布和硬件须重新测量。工具链为PyPTO导出行为基线
872df763f、CANN 9.2.0、BiSheng 15.0.5、Ascend950PR、torch/torch_npu 2.10.0；当前验收仍需绑定源码哈希。
"""

from __future__ import annotations

from functools import lru_cache

import pypto_pro.language as pl


_VALID_DIMS = (16, 32, 64, 128, 256)
_M_BLOCK = 128


def _make_kernel(
    k_dim: int, n_dim: int, *, balance_tiles: bool, m_block: int = _M_BLOCK
):
    """Build one K/N specialization; T and G remain runtime dimensions."""
    split_k = min(k_dim, 128)
    strategy = "adaptive" if balance_tiles else "group"
    kernel_name = f"ragged_grouped_gemm_{strategy}_m{m_block}_k{k_dim}_n{n_dim}"

    @pl.jit(arch="a5", auto_mutex=True, name=kernel_name)
    def kernel(
        a: pl.Tensor[[pl.DYNAMIC, k_dim], pl.DT_FP16],
        w: pl.Tensor[[pl.DYNAMIC, k_dim, n_dim], pl.DT_FP16],
        offsets: pl.Tensor[[pl.DYNAMIC], pl.DT_INT32],
        c: pl.Tensor[[pl.DYNAMIC, n_dim], pl.DT_FP32],
    ):
        # Per worker, the peak live storage is: L1 A 128xK FP16 (<=64 KiB),
        # L1 W KxN FP16 (<=128 KiB), L0A 128xmin(K,128) FP16 (<=32 KiB),
        # L0B min(K,128)xN FP16 (<=64 KiB), and Acc 128xN FP32 (<=128 KiB).
        # One slot per tile is enough because tiles have no dependency; no GM
        # partial or merge pass is needed. Runtime m tails are narrowed by gemm.
        with pl.section_cube():
            a_l1 = pl.make_tile_group(
                type=pl.TileType(
                    shape=[m_block, k_dim],
                    dtype=pl.DT_FP16,
                    target_memory=pl.MemorySpace.Mat,
                    layout=pl.NZ,
                    valid_shape=[-1, -1],
                ),
                addrs=[0],
                mutex_ids=[0],
            )
            w_l1 = pl.make_tile_group(
                type=pl.TileType(
                    shape=[k_dim, n_dim],
                    dtype=pl.DT_FP16,
                    target_memory=pl.MemorySpace.Mat,
                    layout=pl.NZ,
                ),
                addrs=[0x20000],
                mutex_ids=[1],
            )
            l0a = pl.make_tile_group(
                type=pl.TileType(
                    shape=[m_block, split_k],
                    dtype=pl.DT_FP16,
                    target_memory=pl.MemorySpace.Left,
                    layout=pl.NZ,
                    compact=1,
                ),
                addrs=[0],
                mutex_ids=[2],
            )
            l0b = pl.make_tile_group(
                type=pl.TileType(
                    shape=[split_k, n_dim],
                    dtype=pl.DT_FP16,
                    target_memory=pl.MemorySpace.Right,
                    layout=pl.ZN,
                    compact=1,
                ),
                addrs=[0],
                mutex_ids=[3],
            )
            acc = pl.make_tile_group(
                type=pl.TileType(
                    shape=[m_block, n_dim],
                    dtype=pl.DT_FP32,
                    target_memory=pl.MemorySpace.Acc,
                    layout=pl.NZ,
                    fractal=1024,
                    compact=1,
                ),
                addrs=[0],
                mutex_ids=[4],
            )

            worker = pl.get_block_idx()
            workers = pl.get_block_num()

            if balance_tiles:
                # Keep short groups intact on one worker. A longer segment is
                # striped across workers to prevent a concentrated workload tail.
                for g in pl.range(0, w.shape[0]):
                    row_start = pl.getval(offsets, g)
                    row_end = pl.getval(offsets, g + 1)
                    row_count = row_end - row_start
                    tile_count = (row_count + m_block - 1) // m_block
                    if tile_count > workers:
                        w_tile = w_l1.next()
                        pl.load(w_tile, w, [g, 0, 0])
                        for tile_idx in pl.range(worker, tile_count, workers):
                            row_in_group = tile_idx * m_block
                            valid_rows = pl.min(m_block, row_count - row_in_group)
                            a_tile = a_l1.next()
                            acc_tile = acc.next()
                            pl.set_validshape(a_tile, [valid_rows, k_dim])
                            pl.load(a_tile, a, [row_start + row_in_group, 0])
                            pl.gemm(
                                acc_tile,
                                a_tile,
                                w_tile,
                                l0a=l0a,
                                l0b=l0b,
                                m=valid_rows,
                                splitk=split_k,
                            )
                            pl.store(c, acc_tile, [row_start + row_in_group, 0])
                    else:
                        if g % workers == worker:
                            if tile_count > 0:
                                w_tile = w_l1.next()
                                pl.load(w_tile, w, [g, 0, 0])
                                for tile_idx in pl.range(0, tile_count):
                                    row_in_group = tile_idx * m_block
                                    valid_rows = pl.min(
                                        m_block, row_count - row_in_group
                                    )
                                    a_tile = a_l1.next()
                                    acc_tile = acc.next()
                                    pl.set_validshape(a_tile, [valid_rows, k_dim])
                                    pl.load(a_tile, a, [row_start + row_in_group, 0])
                                    pl.gemm(
                                        acc_tile,
                                        a_tile,
                                        w_tile,
                                        l0a=l0a,
                                        l0b=l0b,
                                        m=valid_rows,
                                        splitk=split_k,
                                    )
                                    pl.store(c, acc_tile, [row_start + row_in_group, 0])
            else:
                # Measurement baseline: a group belongs to one worker; each
                # worker visits groups in a strided sequence and owns all M tiles.
                for g in pl.range(worker, w.shape[0], workers):
                    row_start = pl.getval(offsets, g)
                    row_end = pl.getval(offsets, g + 1)
                    row_count = row_end - row_start
                    tile_count = (row_count + m_block - 1) // m_block
                    if tile_count > 0:
                        w_tile = w_l1.next()
                        pl.load(w_tile, w, [g, 0, 0])
                        for tile_idx in pl.range(0, tile_count):
                            row_in_group = tile_idx * m_block
                            valid_rows = pl.min(m_block, row_count - row_in_group)
                            a_tile = a_l1.next()
                            acc_tile = acc.next()
                            pl.set_validshape(a_tile, [valid_rows, k_dim])
                            pl.load(a_tile, a, [row_start + row_in_group, 0])
                            pl.gemm(
                                acc_tile,
                                a_tile,
                                w_tile,
                                l0a=l0a,
                                l0b=l0b,
                                m=valid_rows,
                                splitk=split_k,
                            )
                            pl.store(c, acc_tile, [row_start + row_in_group, 0])

    return kernel


@lru_cache(maxsize=None)
def _tile_kernel(k_dim: int, n_dim: int, m_block: int = _M_BLOCK):
    return _make_kernel(k_dim, n_dim, balance_tiles=True, m_block=m_block)


@lru_cache(maxsize=None)
def _group_kernel(k_dim: int, n_dim: int, m_block: int = _M_BLOCK):
    return _make_kernel(k_dim, n_dim, balance_tiles=False, m_block=m_block)


def _metadata(a, w, offsets, c, workers: int) -> tuple[int, int, int, int]:
    """Validate public metadata without reading tensor contents."""
    if (
        len(a.shape) != 2
        or len(w.shape) != 3
        or len(offsets.shape) != 1
        or len(c.shape) != 2
    ):
        raise ValueError("expected A[T,K], W[G,K,N], offsets[G+1], and C[T,N]")
    t_dim, k_dim = a.shape
    groups, w_k, n_dim = w.shape
    if not (1 <= t_dim <= 4096 and 1 <= groups <= 32):
        raise ValueError("T must be in [1,4096] and G in [1,32]")
    if k_dim != w_k or k_dim not in _VALID_DIMS or n_dim not in _VALID_DIMS:
        raise ValueError("K and N must match and belong to {16,32,64,128,256}")
    if tuple(offsets.shape) != (groups + 1,) or tuple(c.shape) != (t_dim, n_dim):
        raise ValueError("offsets must have shape [G+1] and C must have shape [T,N]")
    if workers < 1 or workers > 32:
        raise ValueError("workers must be in [1,32]")
    if str(a.dtype) != "torch.float16" or str(w.dtype) != "torch.float16":
        raise TypeError("A and W must be FP16")
    if str(offsets.dtype) != "torch.int32" or str(c.dtype) != "torch.float32":
        raise TypeError("offsets must be INT32 and C must be FP32")
    if not (
        a.is_contiguous()
        and w.is_contiguous()
        and offsets.is_contiguous()
        and c.is_contiguous()
    ):
        raise ValueError("all tensors must be contiguous")
    if not (a.device == w.device == offsets.device == c.device):
        raise ValueError("all tensors must be on the same device")
    return t_dim, groups, k_dim, n_dim


def run(a, w, offsets, c, workers: int = 1):
    """Launch one native grouped GEMM call and return ``c`` (asynchronously)."""
    _, _, k_dim, n_dim = _metadata(a, w, offsets, c, workers)
    _tile_kernel(k_dim, n_dim, _M_BLOCK)[None, workers](a, w, offsets, c)
    return c


def run_group_owned(a, w, offsets, c, workers: int = 1):
    """Single-launch comparison baseline that assigns each group to one worker."""
    _, _, k_dim, n_dim = _metadata(a, w, offsets, c, workers)
    kernel = _group_kernel(k_dim, n_dim, _M_BLOCK)
    kernel[None, workers](a, w, offsets, c)
    return c


def run_tile_size_for_benchmark(a, w, offsets, c, workers: int, m_block: int):
    """Run another fixed M tile size for the isolated A/B measurement."""
    _, _, k_dim, n_dim = _metadata(a, w, offsets, c, workers)
    if m_block not in (64, 128):
        raise ValueError("benchmark M block must be 64 or 128")
    _tile_kernel(k_dim, n_dim, m_block)[None, workers](a, w, offsets, c)
    return c
