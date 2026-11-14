# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

"""Per-channel Fixpipe scaling with an explicitly placed Scaling Tile.

The CPU reference encodes FP32 scale bits and the signed INT8 flag, then independently computes per-column scaling,
rounding and saturation. Powers of two and small integer products preserve the existing rtol=atol=0 criterion.

Numerical controls cover FP16/INT8 move, direct GM store, column-dependent tables and bit46 saturation. Parse-only
checks cover unsupported FP32 output, shape/dtype/space restrictions, FB capacity, a GM Tensor passed as scale and
per-channel atomic updates. The scale consumer rejects N > 512 before launch, since INT64 tables must fit the 4096B
Scaling space.

The kernel uses MTE2→FIX synchronization before L1→Scaling. The shared Cube producer and scale table occupy disjoint L1
ranges; the declaration helper is owned by fixpipe_riders.
"""

from pathlib import Path

import numpy as np
import pytest
import torch
from pypto_pro._errors import (
    InvalidOperation,
    InvalidTile,
    InvalidType,
    InvalidVal,
    NotSupported,
    OutOfRange,
)

from . import reference as ref
from .kernel import (
    K,
    M,
    N,
    move_to_fp16,
    move_to_fp32,
    move_to_int8,
    scale_fp16_dtype,
    scale_fp32_dtype,
    scale_from_gm_tensor,
    scale_in_l1,
    scale_too_many_cols,
    scale_two_rows,
    scale_unaligned_cols,
    store_fp16,
    store_with_atomic,
)

#: INT8 那格用更小的 scale，好让取整和两端饱和都验得到。
INT8_EXPONENTS = (-3, -2, -1, 0, 2)


def _operands(device):
    a, b = ref.make_operands(M, K, N)
    return a, b, torch.from_numpy(a).to(device), torch.from_numpy(b).to(device)


def _run(kernel, device, scales, torch_dtype, *, flag=True):
    """跑一个 kernel，返回 (acc 参考, 设备输出)。"""
    a, b, ta, tb = _operands(device)
    enc = torch.from_numpy(ref.encode_scales(scales, signed_int8_flag=flag)).to(device)
    out = torch.zeros((M, N), dtype=torch_dtype, device=device)
    kernel(ta, tb, enc, out)
    torch.npu.synchronize()
    return ref.accumulate(a, b), out.cpu().to(torch.float32).numpy().astype(np.float64)


# ---------------------------------------------------------------------------
# 一、数值：两条路 × 三种目的 dtype
# ---------------------------------------------------------------------------


def test_per_channel_move_to_fp16(device):
    """Acc→Vec，`FP32 → FP16`，每列一个 scale——逐位精确。"""
    scales = ref.column_scales(N)
    acc, got = _run(move_to_fp16, device, scales, torch.float16)
    np.testing.assert_allclose(
        got,
        ref.scaled_per_column(acc, scales),
        rtol=0,
        atol=0,
        err_msg="per-channel FP16 的输出不是逐列缩放",
    )


def test_per_channel_move_to_int8(device):
    """Acc→Vec，`FP32 → INT8`：逐列缩放 → 半偶取整 → `[-128, 127]` 饱和。"""
    scales = ref.column_scales(N, exponents=INT8_EXPONENTS)
    acc, got = _run(move_to_int8, device, scales, torch.int8)
    np.testing.assert_allclose(
        got,
        ref.quantized_to_int8(acc, scales),
        rtol=0,
        atol=0,
        err_msg="per-channel INT8 的输出不是「缩放→半偶取整→饱和」",
    )


def test_the_int8_case_actually_saturates_both_ends(device):
    """防空测：INT8 那条要**两端都饱和到**才算验全。

    不饱和的话，这条测试和 FP16 那条验的是同一件事。
    """
    a, b = ref.make_operands(M, K, N)
    scales = ref.column_scales(N, exponents=INT8_EXPONENTS)
    raw = ref.scaled_per_column(ref.accumulate(a, b), scales)
    lo, hi = ref.INT8_RANGE
    assert (raw > hi).any(), "没有一个值超过 127——上界没验到，调大某一列的 scale"
    assert (raw < lo).any(), "没有一个值低于 -128——下界没验到"


def test_per_channel_store_to_gm(device):
    """Acc→GM，`FP32 → FP16`——**同一张表，另一条路**，也逐位精确。

    这条不需要跨核交接，是 per-channel 最短的写法。
    """
    scales = ref.column_scales(N)
    acc, got = _run(store_fp16, device, scales, torch.float16)
    np.testing.assert_allclose(
        got,
        ref.scaled_per_column(acc, scales),
        rtol=0,
        atol=0,
        err_msg="per-channel store 的输出不是逐列缩放",
    )


def test_per_channel_fp32_to_fp32_is_rejected(device):
    """`FP32 → FP32` 带 per-channel 的 scale 走 `pl.move`，**解析期被拒**。

    原来的行为是**静默丢弃**，和 per-tensor 那一格一样;
    当时这一条的价值在于把缺陷的范围收窄：**per-channel 走的是完全不同的编码路**
    （FB 里的位域，不是指令里的立即数），两条路都丢,
    所以坏的是「目的是 UB 且目的 dtype 也是 FP32」这条路，不是 scale 怎么编。

    **2026-09-27 起 pypto 在解析期拒掉这一格**（根因在 CANN 的 PTO 头，先不修）。
    拦截不看 scale 是标量还是 Tile，所以两条编码路一起被拦——这条测试钉的就是
    「per-channel 也被拦住了」，别只拦了 per-tensor 那一半。

    根因见 [defect](../../../../references/diagnostics/defects/fixpipe-fp32-to-fp32-drops-scale.md),
    处置见 [`diagnostics/defects/fixpipe-overview.md`](../../../../references/diagnostics/defects/fixpipe-overview.md)。
    """
    scales = ref.column_scales(N)
    a, b, ta, tb = _operands(device)
    enc = torch.from_numpy(ref.encode_scales(scales)).to(device)
    out = torch.zeros((M, N), dtype=torch.float32, device=device)

    with pytest.raises(Exception) as excinfo:
        move_to_fp32(ta, tb, enc, out)
    assert "silently dropped" in str(excinfo.value), (
        f"报错不是那条拦截：{excinfo.value}"
    )


# ---------------------------------------------------------------------------
# 二、bit46：signed-int8 标志
# ---------------------------------------------------------------------------


def test_the_signed_int8_flag_selects_the_saturation_range(device):
    """不置 bit46 时，INT8 目的按 **`[0, 255]`** 饱和，字节再按 int8 读。

    逐位相等，见模块 docstring §二。这一条同时说明了那一位的名字是准确的。
    """
    scales = ref.column_scales(N, exponents=INT8_EXPONENTS)
    acc, got = _run(move_to_int8, device, scales, torch.int8, flag=False)

    raw = ref.scaled_per_column(acc, scales)
    unsigned = np.clip(np.rint(raw), 0, 255)
    as_int8 = (unsigned + 128) % 256 - 128
    np.testing.assert_allclose(
        got,
        as_int8,
        rtol=0,
        atol=0,
        err_msg="不置 bit46 的输出不是「无符号饱和后按 int8 读」",
    )

    assert not np.array_equal(got, ref.quantized_to_int8(acc, scales)), (
        "带 flag 和不带 flag 的结果相同——这条验空了"
    )
    assert (got[raw < 0] == 0).all(), "负数位置没有全部变成 0"


def test_the_flag_does_not_affect_a_float_destination(device):
    """浮点目的带不带 bit46 **逐位相同**——那一位只管整型。"""
    scales = ref.column_scales(N)
    acc, with_flag = _run(move_to_fp16, device, scales, torch.float16, flag=True)
    _, without = _run(move_to_fp16, device, scales, torch.float16, flag=False)

    np.testing.assert_allclose(
        with_flag, without, rtol=0, atol=0, err_msg="浮点目的下 bit46 改变了结果"
    )
    np.testing.assert_allclose(
        with_flag,
        ref.scaled_per_column(acc, scales),
        rtol=0,
        atol=0,
        err_msg="带 flag 的浮点结果本身就不对",
    )


# ---------------------------------------------------------------------------
# 三、per-channel 真的是「逐列」
# ---------------------------------------------------------------------------


def test_the_table_is_read_per_column_not_broadcast(device):
    """**只有第 0 列的 scale 是 2，其余列是 1** —— 输出必须只有第 0 列翻倍。

    这一条排掉「实现只读了表的第一个元素然后广播」。用的是**同一个 kernel**——
    差异全在 host 侧那张表里，所以它验的确实是表的读法，不是另一段 kernel 代码。
    """
    scales = np.ones(N, dtype=np.float64)
    scales[0] = 2.0
    acc, got = _run(move_to_fp16, device, scales, torch.float16)

    np.testing.assert_allclose(
        got,
        ref.scaled_per_column(acc, scales),
        rtol=0,
        atol=0,
        err_msg="只改第 0 列的 scale，输出却不是「只有第 0 列变了」",
    )
    np.testing.assert_allclose(
        got[:, 1:],
        acc[:, 1:],
        rtol=0,
        atol=0,
        err_msg="第 0 列的 scale 泄漏到了别的列（广播？）",
    )


def test_the_columns_of_the_default_table_really_differ():
    """不需要卡。防空测：默认那张表不能所有列都一样，否则「逐列」验不到。"""
    scales = ref.column_scales(N)
    assert len(set(scales.tolist())) > 1, "整张表只有一个值——换 exponents"
    assert scales[0] != scales[1], "相邻两列相同——广播实现也能过"


# ---------------------------------------------------------------------------
# 四、六条形状 / 类型约束 + 两条组合约束，全在解析期
# ---------------------------------------------------------------------------


def _reject(kernel, exc, pattern):
    """在 host 上调一次，断言解析期就拒。传 CPU 张量就够。"""
    a, b = ref.make_operands(M, K, N)
    enc = torch.from_numpy(ref.encode_scales(ref.column_scales(N)))
    with pytest.raises(exc, match=pattern):
        kernel(
            torch.from_numpy(a),
            torch.from_numpy(b),
            enc,
            torch.zeros(M, N, dtype=torch.float16),
        )


def test_a_scale_tile_with_two_rows_is_rejected(device):
    """行必须是 1——按行（per-token）量化要在 vector 域做。"""
    _reject(scale_two_rows, NotSupported, r"must have shape \[1, N\]")


def test_a_column_count_not_divisible_by_16_is_rejected(device):
    """`N % 16 == 0`：INT64 × 16 = 128B 对齐。"""
    _reject(scale_unaligned_cols, InvalidTile, "must be a multiple of 16")


def test_too_many_columns_is_rejected_by_the_scale_consumer(device):
    """The scale consumer rejects N=1024 under the 4096B INT64 table limit."""
    _reject(scale_too_many_cols, OutOfRange, r"N must be <= 512.*got \[1, 1024\]")


def test_an_fp32_scale_tile_is_rejected_with_a_conversion_hint(device):
    """dtype 不能是 FP32，报错里给的路是 `npu_trans_quant_param`。

    **断言里带上那个函数名**：这条报错的价值就在于它告诉你下一步做什么。
    """
    _reject(scale_fp32_dtype, InvalidType, "npu_trans_quant_param")


def test_a_non_int64_scale_tile_is_rejected(device):
    """dtype 必须是 INT64——FP16 走的是另一支报错。"""
    _reject(scale_fp16_dtype, InvalidType, "must be INT64")


def test_a_scale_tile_outside_the_scaling_space_is_rejected(device):
    """放在 L1 里不算——必须是 `MemorySpace.Scaling`。"""
    _reject(scale_in_l1, InvalidOperation, "must be allocated in MemorySpace.Scaling")


def test_a_gm_tensor_as_scale_is_rejected(device):
    """不能把 GM 上的表直接当 `scale=` 传——自动 per-channel 路径已经删了。"""
    _reject(scale_from_gm_tensor, InvalidVal, "scale Tensor is not supported")


def test_per_channel_cannot_be_combined_with_atomic(device):
    """per-channel + `atomic` 拒。**per-tensor + atomic 是可以的**
    （[`fixpipe_riders`](../fixpipe_riders/test.py) 验过），所以这是 per-channel 独有的限制。
    """
    _reject(
        store_with_atomic,
        InvalidOperation,
        r"scale \(per-channel\) cannot be combined with atomic",
    )


# ---------------------------------------------------------------------------
# 五、结构检查
# ---------------------------------------------------------------------------


def test_the_five_step_preparation_appears_in_every_working_kernel():
    """不需要卡。防漏：能跑的 kernel 里，五步流程的每一步都得出现。

    查的是「必须出现」——这个方向安全（`references/repeated-mistakes.md` 第 8 条）。
    """
    source = (Path(__file__).parent / "kernel.py").read_text()
    for step in (
        "pl.load(fp_mat, scales,",
        "pl.move(fp_tile, fp_mat)",
        "set_pipe=pl.PipeType.MTE2, wait_pipe=pl.PipeType.FIX",
        "scale=fp_tile",
    ):
        assert step in source, f"五步流程里缺了 `{step}`"
