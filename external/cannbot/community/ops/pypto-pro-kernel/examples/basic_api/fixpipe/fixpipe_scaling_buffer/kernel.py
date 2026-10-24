# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

"""写法：原始。通过 Scaling Tile 提供每列独立的 Fixpipe 量化参数。

本例 M=N=K=64，FP16 输入、FP32 Acc，输出覆盖 FP16/INT8；FP32→FP32 的带 scale 路径按当前框架契约拒绝。Cube 输入与 Acc 的声明复用
[riders](../fixpipe_riders/kernel.py) 的同形状生产者。

资源均显式放置并使用 depth=1、``mutex_ids="auto"``。L1 的 A/B 位于 0、0x2000，scale 中转表位于 0x4000，互不重叠。Scaling 表从独立 FB 地址 0 开始，[1,64]
INT64 占 512B；Left、Right、Acc、Vec 在各自空间从 0 开始。

准备与消费依次为：GM→L1（MTE2）、等待 MTE2→FIX、L1→Scaling（FIX）、Acc→Vec/GM 的逐列量化（FIX）。自动 mutex 跟踪这些读写；源码保留显式 MTE2→FIX flag 对，展示表数据就绪的边
界。示例分别取得 L1 与 Scaling 句柄，保持数据通路清晰。

INT64 表项的低 32 bit 是 FP32 scale 的 IEEE 位模式，bit46 控制 INT8 的有符号饱和区间。``reference.py`` 编码该位域，并从数学定义独立计算逐列缩放、取整和饱和。scale 使用 2
的幂，数值对照保持 rtol=atol=0。

表的合法形状为 [1,N]、N 是 16 的倍数且不超过 512，dtype 必须 INT64，空间必须 Scaling。超限的表由 scale 消费操作在解析期拒绝；其余非法形状、类型、GM Tensor 和
per-channel+atomic 组合也只做解析期验证。
"""

import pypto_pro.language as pl

from ..fixpipe_riders.kernel import _cube_groups

FP16 = pl.DT_FP16
FP32 = pl.DT_FP32
INT8 = pl.DT_INT8
INT64 = pl.DT_INT64

M = N = K = 64

ND = pl.TensorLayout.ND


def _scale_groups():
    """scale 表的两站：L1 中转 + FB 终点。两边都是 `[1, N] INT64`。"""
    scale_mat = pl.make_tile_group(
        type=pl.TileType(
            shape=[1, N], dtype=INT64, target_memory=pl.MemorySpace.Mat, layout=ND
        ),
        addrs=0x4000,
        mutex_ids="auto",
        depth=1,
    )
    scale_fb = pl.make_tile_group(
        type=pl.TileType(
            shape=[1, N], dtype=INT64, target_memory=pl.MemorySpace.Scaling
        ),
        addrs=0x0,
        mutex_ids="auto",
        depth=1,
    )
    return scale_mat, scale_fb


# ---------------------------------------------------------------------------
# 一、Acc → Vec：per-channel move
# ---------------------------------------------------------------------------


def make_per_channel_move(dst_dtype):
    """`pl.move(t, ac, scale=fp_tile)`——每列一个 scale。

    目的 dtype 由闭包给，这样三种目的类型共用一份 kernel 代码：

    | dst | 为什么选它 |
    | --- | --- |
    | `FP32` | per-tensor 的 `FP32 → FP32` **会静默丢 scale**（见 defect）；per-channel 会不会 |
    | `FP16` | 降位 + 缩放，最常用的一格 |
    | `INT8` | bit46 那个 signed-int8 标志唯一有意义的地方；顺带验取整与饱和 |

    走 Acc→**Vec**，和 `fixpipe_quant_matrix` 的 11 格同一条路，可以直接对比。
    """

    @pl.jit(auto_mutex=True)
    def kernel(
        a: pl.Tensor[[M, K], FP16],
        b: pl.Tensor[[K, N], FP16],
        scales: pl.Tensor[[1, N], INT64],
        out: pl.Tensor[[M, N], dst_dtype],
    ):
        ub = pl.make_tile_group(
            type=pl.TileType(
                shape=[M, N],
                dtype=dst_dtype,
                target_memory=pl.MemorySpace.Vec,
                layout=ND,
            ),
            addrs=0x0,
            mutex_ids="auto",
            depth=1,
        )
        a_l1, b_l1, l0a, l0b, acc = _cube_groups()
        scale_mat, scale_fb = _scale_groups()
        t = ub.current()
        with pl.section_cube():
            cur_a, cur_b, ac = a_l1.current(), b_l1.current(), acc.current()
            # L1 staging and Scaling consumption use distinct memory spaces.
            fp_mat = scale_mat.current()
            fp_tile = scale_fb.current()
            pl.load(cur_a, a, [0, 0])
            pl.load(cur_b, b, [0, 0])
            pl.load(fp_mat, scales, [0, 0])
            pl.gemm(ac, cur_a, cur_b, l0a=l0a, l0b=l0b)
            pl.system.sync_src(
                set_pipe=pl.PipeType.MTE2, wait_pipe=pl.PipeType.FIX, event_id=1
            )
            pl.system.sync_dst(
                set_pipe=pl.PipeType.MTE2, wait_pipe=pl.PipeType.FIX, event_id=1
            )
            pl.move(fp_tile, fp_mat)
            pl.move(t, ac, scale=fp_tile)
            pl.system.set_cross_core(pipe=pl.PipeType.FIX, event_id=0)
        with pl.section_vector():
            pl.system.wait_cross_core(pipe=pl.PipeType.MTE3, event_id=0)
            if pl.get_subblock_idx() == 0:
                pl.store(out, t, [0, 0])
            pl.system.bar_all()

    return kernel


move_to_fp32 = make_per_channel_move(FP32)
move_to_fp16 = make_per_channel_move(FP16)
move_to_int8 = make_per_channel_move(INT8)


# ---------------------------------------------------------------------------
# 二、Acc → GM：per-channel store
# ---------------------------------------------------------------------------


@pl.jit(auto_mutex=True)
def store_fp16(
    a: pl.Tensor[[M, K], FP16],
    b: pl.Tensor[[K, N], FP16],
    scales: pl.Tensor[[1, N], INT64],
    out: pl.Tensor[[M, N], FP16],
):
    """`pl.store(out, ac, [0, 0], scale=fp_tile)`——**同一张表，另一条路**。

    Acc→GM 不需要跨核交接（对照 §一 那条），所以这是 per-channel 最短的写法。
    Scaling Tile 的搭法一模一样，五步一步不少。
    """
    a_l1, b_l1, l0a, l0b, acc = _cube_groups()
    scale_mat, scale_fb = _scale_groups()
    with pl.section_cube():
        cur_a, cur_b, ac = a_l1.current(), b_l1.current(), acc.current()
        fp_mat = scale_mat.current()
        fp_tile = scale_fb.current()
        pl.load(cur_a, a, [0, 0])
        pl.load(cur_b, b, [0, 0])
        pl.load(fp_mat, scales, [0, 0])
        pl.gemm(ac, cur_a, cur_b, l0a=l0a, l0b=l0b)
        pl.system.sync_src(
            set_pipe=pl.PipeType.MTE2, wait_pipe=pl.PipeType.FIX, event_id=1
        )
        pl.system.sync_dst(
            set_pipe=pl.PipeType.MTE2, wait_pipe=pl.PipeType.FIX, event_id=1
        )
        pl.move(fp_tile, fp_mat)
        pl.store(out, ac, [0, 0], scale=fp_tile)


@pl.jit(auto_mutex=True)
def store_with_atomic(
    a: pl.Tensor[[M, K], FP16],
    b: pl.Tensor[[K, N], FP16],
    scales: pl.Tensor[[1, N], INT64],
    out: pl.Tensor[[M, N], FP16],
):
    """per-channel + `atomic`——**解析期拒**（`block_ops.py:367`）。

    「the fixpipe quantization path does not emit atomic instructions;
    per-tensor scalar scale + atomic is supported」——所以这是
    **per-channel 独有**的限制：per-tensor + atomic 在
    [`fixpipe_riders`](../fixpipe_riders/kernel.py) 里是跑通的。
    """
    a_l1, b_l1, l0a, l0b, acc = _cube_groups()
    scale_mat, scale_fb = _scale_groups()
    with pl.section_cube():
        cur_a, cur_b, ac = a_l1.current(), b_l1.current(), acc.current()
        fp_mat = scale_mat.current()
        fp_tile = scale_fb.current()
        pl.load(cur_a, a, [0, 0])
        pl.load(cur_b, b, [0, 0])
        pl.load(fp_mat, scales, [0, 0])
        pl.gemm(ac, cur_a, cur_b, l0a=l0a, l0b=l0b)
        pl.system.sync_src(
            set_pipe=pl.PipeType.MTE2, wait_pipe=pl.PipeType.FIX, event_id=1
        )
        pl.system.sync_dst(
            set_pipe=pl.PipeType.MTE2, wait_pipe=pl.PipeType.FIX, event_id=1
        )
        pl.move(fp_tile, fp_mat)
        pl.store(out, ac, [0, 0], scale=fp_tile, atomic=pl.AtomicType.AtomicAdd)


# ---------------------------------------------------------------------------
# 三、六条形状 / 类型约束，每条一个 kernel
# ---------------------------------------------------------------------------


def make_bad_scale_tile(*, rows=1, cols=N, dtype=INT64):
    """Create a malformed Scaling Tile for parse-only shape and dtype checks."""

    @pl.jit(auto_mutex=True)
    def kernel(
        a: pl.Tensor[[M, K], FP16],
        b: pl.Tensor[[K, N], FP16],
        scales: pl.Tensor[[1, N], INT64],
        out: pl.Tensor[[M, N], FP16],
    ):
        a_l1, b_l1, l0a, l0b, acc = _cube_groups()
        bad = pl.make_tile_group(
            type=pl.TileType(
                shape=[rows, cols], dtype=dtype, target_memory=pl.MemorySpace.Scaling
            ),
            addrs=0x0,
            mutex_ids="auto",
            depth=1,
        )
        with pl.section_cube():
            cur_a, cur_b, ac = a_l1.current(), b_l1.current(), acc.current()
            pl.load(cur_a, a, [0, 0])
            pl.load(cur_b, b, [0, 0])
            pl.gemm(ac, cur_a, cur_b, l0a=l0a, l0b=l0b)
            pl.store(out, ac, [0, 0], scale=bad.current())

    return kernel


#: 行不是 1——按行（per-token）量化硬件不支持。
scale_two_rows = make_bad_scale_tile(rows=2)

#: 列数不是 16 的倍数——INT64 × 16 = 128B 对齐。
scale_unaligned_cols = make_bad_scale_tile(cols=24)

#: 列数超过 512——FB 只有 4KB。
scale_too_many_cols = make_bad_scale_tile(cols=1024)

#: dtype 是 FP32——报错里会给 `npu_trans_quant_param` 的用法。
scale_fp32_dtype = make_bad_scale_tile(dtype=FP32)

#: dtype 是 FP16——同一条检查的另一支。
scale_fp16_dtype = make_bad_scale_tile(dtype=FP16)


@pl.jit(auto_mutex=True)
def scale_in_l1(
    a: pl.Tensor[[M, K], FP16],
    b: pl.Tensor[[K, N], FP16],
    scales: pl.Tensor[[1, N], INT64],
    out: pl.Tensor[[M, N], FP16],
):
    """A scale Tile in L1 is rejected: the consumer requires MemorySpace.Scaling."""
    a_l1, b_l1, l0a, l0b, acc = _cube_groups()
    bad = pl.make_tile_group(
        type=pl.TileType(
            shape=[1, N], dtype=INT64, target_memory=pl.MemorySpace.Mat, layout=ND
        ),
        addrs=0x4000,
        mutex_ids="auto",
        depth=1,
    )
    with pl.section_cube():
        cur_a, cur_b, ac = a_l1.current(), b_l1.current(), acc.current()
        pl.load(cur_a, a, [0, 0])
        pl.load(cur_b, b, [0, 0])
        pl.gemm(ac, cur_a, cur_b, l0a=l0a, l0b=l0b)
        pl.store(out, ac, [0, 0], scale=bad.current())


@pl.jit(auto_mutex=True)
def scale_from_gm_tensor(
    a: pl.Tensor[[M, K], FP16],
    b: pl.Tensor[[K, N], FP16],
    scales: pl.Tensor[[1, N], INT64],
    out: pl.Tensor[[M, N], FP16],
):
    """直接把 GM 上的 `scales` 当 `scale=` 传——**解析期拒**。

    自动 per-channel 路径（框架替你分配 Mat 中转 + 插同步）**已经删掉了**,
    所以这条路现在必须手搭。报错文本里就写着那五步。
    """
    a_l1, b_l1, l0a, l0b, acc = _cube_groups()
    with pl.section_cube():
        cur_a, cur_b, ac = a_l1.current(), b_l1.current(), acc.current()
        pl.load(cur_a, a, [0, 0])
        pl.load(cur_b, b, [0, 0])
        pl.gemm(ac, cur_a, cur_b, l0a=l0a, l0b=l0b)
        pl.store(out, ac, [0, 0], scale=scales)
