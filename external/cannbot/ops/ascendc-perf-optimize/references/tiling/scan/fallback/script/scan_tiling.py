#!/usr/bin/env python3
# ----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# Scan / Cum 族 Tiling 参考实现（简化版）：合轴 (M,R,N) → 流式选块 → 并行度补偿 → 跨核借 R 判定。
# 与 kernel 侧 scan_tiling.template 对应。仅用于建模/选型说明，非生产 tiling。
# ----------------------------------------------------------------------------------------------------------
from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

VL = 64  # 256B / 4B
W_TARGET = 1024  # 流式模板目标向量宽


def ceil_div(a: int, b: int) -> int:
    return a if b == 0 else (a + b - 1) // b


def fuse_shape(shape: list[int], dim: int) -> Tuple[int, int, int]:
    """任意 rank 沿 dim 融合成 (M, R, N)。dim=-1 → N=1；dim=0 → M=1。"""
    nd = len(shape)
    if dim < 0:
        dim += nd
    m = 1
    for i in range(dim):
        m *= shape[i]
    r = shape[dim]
    n = 1
    for i in range(dim + 1, nd):
        n *= shape[i]
    return m, r, n


@dataclass
class ScanTiling:
    template: str  # "streaming" | "oneway" | "twoway" | "core_partition"
    m: int
    r: int
    n: int
    m_blk: int
    n_tile_len: int
    m_outer: int
    n_outer: int
    total_work: int
    block_dim: int
    with_index: bool
    r_borrow: int = 0  # core_partition 时的 R 分段数


def select_scan_tiling(shape: list[int], dim: int, op_type: str, core_num: int, ub_size: int) -> ScanTiling:
    """
    op_type: 'sum'|'prod'|'min'|'max'|'logcumsumexp'
    带索引（min/max）→ 流式模板；否则 oneway/twoway。并行度不足 + 大 R → core_partition。
    """
    m, r, n = fuse_shape(shape, dim)
    with_index = op_type in ("min", "max")

    # 流式选块（带索引主力；无索引亦可用作简化建模）
    if n >= W_TARGET:
        m_blk, n_tile_len = 1, W_TARGET
    else:
        n_tile_len = n
        m_blk = max(1, min(m, W_TARGET // max(1, n)))
    m_outer = ceil_div(m, m_blk)
    n_outer = ceil_div(n, n_tile_len)
    items = m_outer * n_outer

    # 并行度补偿：work-item 太少时缩小分块以用满核
    if items < core_num:
        if n >= W_TARGET:
            n_target = ceil_div(core_num, m_outer)
            n_tile_len = ceil_div(n, n_target)
            n_tile_len = max(n_tile_len, VL)
            n_tile_len = ceil_div(n_tile_len, VL) * VL
            n_tile_len = min(n_tile_len, n)
        else:
            m_target = ceil_div(core_num, n_outer)
            m_blk = max(1, ceil_div(m, m_target))
        m_outer = ceil_div(m, m_blk)
        n_outer = ceil_div(n, n_tile_len)
        items = m_outer * n_outer

    # 跨核借 R 判定：M×N 极小 + R 超长，切行仍严重用不满核
    if items * 2 < core_num and r >= 4096:
        r_borrow = min(core_num, ceil_div(r, 1024))
        return ScanTiling(
            "core_partition",
            m,
            r,
            n,
            1,
            max(1, n),
            m,
            ceil_div(n, max(1, n)),
            r_borrow,
            r_borrow,
            with_index,
            r_borrow=r_borrow,
        )

    template = "streaming" if with_index else ("oneway" if n * 4 > 256 // 4 else "twoway")
    used = min(items, core_num)
    return ScanTiling(template, m, r, n, m_blk, n_tile_len, m_outer, n_outer, items, max(1, used), with_index)


if __name__ == "__main__":
    # cann-bench cummin 代表用例演示
    demos = [
        ([1024, 1024], -1, "min"),  # M=1024,R=1024,N=1 → streaming 切行满核
        ([16384, 16384], 0, "min"),  # M=1,R=16384,N=16384 → streaming 借 N
        ([1000003], -1, "min"),  # M=1,R=1e6,N=1 → core_partition（借 R）
        ([363, 367, 373], 1, "min"),  # M=363,R=367,N=373 → streaming
        ([2048, 2048], -1, "sum"),  # 无索引 → oneway/twoway
    ]
    for shp, d, op in demos:
        t = select_scan_tiling(shp, d, op, core_num=72, ub_size=192 * 1024)
        print(
            f"shape={shp} dim={d} op={op:4s} -> {t.template:14s} "
            f"(M={t.m},R={t.r},N={t.n}) mBlk={t.m_blk} nTileLen={t.n_tile_len} "
            f"items={t.total_work} blockDim={t.block_dim} rBorrow={t.r_borrow}"
        )
