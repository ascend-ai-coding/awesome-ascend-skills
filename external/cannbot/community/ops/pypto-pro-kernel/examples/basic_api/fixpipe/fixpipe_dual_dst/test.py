# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

"""AccToVecMode 的单目标/双目标语义与非法组合的拒绝检查。

有效模式使用两个独立 UB 区域、不同预置值和小整数 GEMM 输入，以 rtol=atol=0 验证目的子块及
M/N 分割。kernel 的混合/纯 Vector 几何使用独立探针检查。

非法 scale、dtype 转换、DN 目的与 Relu 组合只做解析检查，要求明确的 NotSupported 诊断。
当前官方 master 缺少 dtype 转换、DN 目的和 Relu 的相应解析门禁，因此这三项使用 strict xfail；其他诊断仍算失败，解析门禁修复后以严格 XPASS 提醒移除标记。已知危险组合不提交到设备。
"""

from pathlib import Path

import numpy as np
import pytest
import torch

import pypto_pro as pp
from pypto_pro._errors import NotSupported

from . import reference as ref
from .kernel import (
    HALF_M,
    HALF_N,
    K,
    M,
    N,
    dual_dn_destination,
    dual_narrowing_dtype,
    dual_with_relu,
    dual_split_m,
    dual_split_n,
    dual_with_scale,
    default_mode,
    probe_mixed_subblock_shape,
    probe_subblock_shape,
    single_vec0_and_vec1,
)


def _dev(array, device):
    return torch.from_numpy(array).to(device)


def _operands(device):
    a, b = ref.make_operands(M, K, N)
    return a, b, _dev(a, device), _dev(b, device)


def _zeros(device, shape=(M, N), dtype=torch.float32):
    return torch.zeros(shape, dtype=dtype, device=device)


def _host(tensor):
    return tensor.cpu().to(torch.float32).numpy().astype(np.float64)


# ---------------------------------------------------------------------------
# 一、前提：有两个 vector 子块
# ---------------------------------------------------------------------------


def test_a_mixed_kernel_has_two_vector_subblocks(device):
    """`AccToVecMode` 的 Vec0 / Vec1 要存在，`get_subblock_num()` 就得是 2。

    **不靠「ST 里写了 `sub_id == 0/1` 所以一定有两个」推**，直接量。
    这一条挂了的话，下面所有「两个子块各存一份」的测试都验不到东西。
    """
    out = torch.zeros(2, dtype=torch.int32, device=device)
    a, b, ta, tb = _operands(device)
    probe_mixed_subblock_shape(ta, tb, out)
    torch.npu.synchronize()
    block_num, subblock_num = out.cpu().tolist()

    assert subblock_num == 2, (
        f"混合 kernel 的 subblock_num 是 {subblock_num}，双目标就无从谈起"
    )
    assert block_num > 1, f"只启动了 {block_num} 个 block"


def test_a_vector_only_kernel_reports_a_different_shape(device):
    """对照：**纯 vector kernel 的两个数不一样**（实测 56 / 1）。

    这条不是为了钉住 56 和 1，是为了钉住「这两个数取决于 kernel 的形状」——
    所以不能拿一种形状量出来的数去解释另一种。
    [`fixpipe_riders`](../fixpipe_riders/test.py) §五 那个 28 就是 cube 段的数。
    """
    out = torch.zeros(2, dtype=torch.int32, device=device)
    probe_subblock_shape(out)
    torch.npu.synchronize()
    vector_only = out.cpu().tolist()

    mixed_out = torch.zeros(2, dtype=torch.int32, device=device)
    a, b, ta, tb = _operands(device)
    probe_mixed_subblock_shape(ta, tb, mixed_out)
    torch.npu.synchronize()

    assert vector_only != mixed_out.cpu().tolist(), (
        f"两种形状量出同一组数 {vector_only}——那这条测试没验到东西"
    )
    assert vector_only[0] * vector_only[1] == mixed_out.cpu().tolist()[0] * 2, (
        f"纯 vector 的 block_num×subblock_num 应当等于混合形状的 vector 核总数，"
        f"实测 {vector_only} vs {mixed_out.cpu().tolist()}"
    )


# ---------------------------------------------------------------------------
# 二、SingleModeVec0 / SingleModeVec1
# ---------------------------------------------------------------------------


def test_the_two_single_modes_target_different_buffers(device):
    """Vec0 收 `2 × acc`，Vec1 收 `1 × acc`，两个子块各读到自己那份。

    **模式如果被忽略（两次 move 落到同一个 UB），两个子块会看到同一个值**，
    这条就挂。不依赖任何「未初始化的 UB 里是什么」的假设——
    那种假设被 UB 残留骗过（`vf_unalign` 那次）。
    """
    a, b, ta, tb = _operands(device)
    out0, out1 = _zeros(device), _zeros(device)
    single_vec0_and_vec1(ta, tb, out0, out1)
    torch.npu.synchronize()

    acc = ref.accumulate(a, b)
    np.testing.assert_allclose(
        _host(out0),
        acc * 2.0,
        rtol=0,
        atol=0,
        err_msg="SingleModeVec0 那份不是 2 × acc",
    )
    np.testing.assert_allclose(
        _host(out1), acc, rtol=0, atol=0, err_msg="SingleModeVec1 那份不是 1 × acc"
    )


def test_the_default_mode_is_single_vec0(device):
    """**不带 `acc_to_vec_mode` 的 move 默认写 Vec0 的 UB。**

    两个 UB 都先种成 `1 × acc`，再不带 mode 地写 `2 × acc`：
    `out0 == 2×` 且 `out1 == 1×` 就说明默认只写了 Vec0。
    四种可能的结果都可分辨，见 kernel.py 的表。

    **这一格是 `fixpipe_paths` / `fixpipe_quant_matrix` 那 39 条的隐含前提**——
    它们都用 `if sub_id == 0` 存，正好是默认落点。
    """
    a, b, ta, tb = _operands(device)
    out0, out1 = _zeros(device), _zeros(device)
    default_mode(ta, tb, out0, out1)
    torch.npu.synchronize()

    acc = ref.accumulate(a, b)
    got0, got1 = _host(out0), _host(out1)
    np.testing.assert_allclose(
        got0,
        acc * 2.0,
        rtol=0,
        atol=0,
        err_msg="Vec0 没拿到不带 mode 那次写的 2×——默认不是 SingleModeVec0",
    )
    np.testing.assert_allclose(
        got1,
        acc,
        rtol=0,
        atol=0,
        err_msg="Vec1 的内容被改了——默认那次 move 也写了 Vec1（广播？）",
    )


# ---------------------------------------------------------------------------
# 三、DualModeSplitM / DualModeSplitN
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("kernel,axis", [(dual_split_m, "M"), (dual_split_n, "N")])
def test_a_dual_mode_splits_the_accumulator_across_both_buffers(device, kernel, axis):
    """半尺寸的 UB tile + 两个子块 → 拼出完整结果。

    **两个模式互为判别式**：SplitN 如果按 M 拆，它的子块会把行切片写进列切片，
    数值就不对。所以「两条都过」比「一条过」强得多。
    """
    a, b, ta, tb = _operands(device)
    out = _zeros(device)
    kernel(ta, tb, out)
    torch.npu.synchronize()

    np.testing.assert_allclose(
        _host(out),
        ref.accumulate(a, b),
        rtol=0,
        atol=0,
        err_msg=f"按 {axis} 拆的两半拼起来不等于参考",
    )


def test_the_two_halves_are_not_interchangeable(device):
    """防空测：参考矩阵的两个半边必须不同，否则拆错也看不出来。

    行半边相同 → SplitM 拆反了也过；列半边相同 → SplitN 拆反了也过。
    """
    a, b = ref.make_operands(M, K, N)
    acc = ref.accumulate(a, b)
    assert not np.array_equal(acc[:HALF_M, :], acc[HALF_M:, :]), (
        "上下两半相同——SplitM 那条验空了"
    )
    assert not np.array_equal(acc[:, :HALF_N], acc[:, HALF_N:]), (
        "左右两半相同——SplitN 那条验空了"
    )


# ---------------------------------------------------------------------------
# 四、dual：四类不安全组合的解析拒绝要求
# ---------------------------------------------------------------------------


def test_a_dual_mode_rejects_scale_at_parse_time():
    """非法双目标组合必须在解析期拒绝，不提交到设备。"""
    with pytest.raises(NotSupported, match='scale cannot be combined with dual-mode'):
        dual_with_scale.to_kernel_def().parse_target_program(pp.ir.SectionKind.Cube)


@pytest.mark.xfail(
    strict=True,
    raises=pytest.fail.Exception,
    reason="PyPTO bca90a6c lacks the dual-destination dtype-conversion parse guard",
)
def test_a_dual_mode_rejects_a_narrowing_dtype():
    """非法双目标组合必须在解析期拒绝，不提交到设备。"""
    with pytest.raises(NotSupported, match='dual-mode acc_to_vec_mode cannot convert'):
        dual_narrowing_dtype.to_kernel_def().parse_target_program(pp.ir.SectionKind.Cube)


@pytest.mark.xfail(
    strict=True,
    raises=pytest.fail.Exception,
    reason="PyPTO bca90a6c lacks the dual-destination DN-layout parse guard",
)
def test_a_dual_mode_rejects_a_dn_destination():
    """非法双目标组合必须在解析期拒绝，不提交到设备。"""
    with pytest.raises(NotSupported, match='does not support a DN destination layout'):
        dual_dn_destination.to_kernel_def().parse_target_program(pp.ir.SectionKind.Cube)


@pytest.mark.xfail(
    strict=True,
    raises=pytest.fail.Exception,
    reason="PyPTO bca90a6c lacks the dual-destination Relu parse guard; launching this route is unsafe",
)
def test_a_dual_mode_with_relu_is_rejected():
    """非法双目标组合必须在解析期拒绝，不提交到设备。"""
    with pytest.raises(NotSupported, match='relu_pre_mode cannot be combined with dual-mode'):
        dual_with_relu.to_kernel_def().parse_target_program(pp.ir.SectionKind.Cube)


# ---------------------------------------------------------------------------
# 五、结构检查
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "name", ["SingleModeVec0", "SingleModeVec1", "DualModeSplitM", "DualModeSplitN"]
)
def test_every_mode_appears_in_this_example(name):
    """不需要卡。防漏：`AccToVecMode` 四个值每个都得在 kernel 源码里用过。

    查的是「必须出现」——这个方向安全，注释里提到只会加强它
    （反方向的教训见 `references/repeated-mistakes.md` 第 8 条）。
    """
    source = (Path(__file__).parent / "kernel.py").read_text()
    assert f"AccToVecMode.{name}" in source, f"kernel.py 里没有用到 {name}"
