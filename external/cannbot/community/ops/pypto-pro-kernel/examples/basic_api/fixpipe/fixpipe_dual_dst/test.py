# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

"""fixpipe 的双目标控制：`AccToVecMode` 四个值 + 「dual 不承载 rider」。

**写法：原始。** 前置样例：[`fixpipe_paths`](../fixpipe_paths/test.py)（Acc→Vec 的跨核交接）、
[`fixpipe_riders`](../fixpipe_riders/test.py)（rider 要配对照组）。
四个值各自是什么、怎么证明「模式真的选了目的地」，在 [`kernel.py`](kernel.py)。

## 一、量出来的四件事

| 问题 | 答案 | 怎么量的 |
| --- | --- | --- |
| 混合 kernel 里有几个 vector 子块 | `block_num=28`、`subblock_num=2`（纯 vector kernel 是 56 / 1） | 两个探针各写一次 GM |
| `SingleModeVec0` / `Vec1` 是不同的 UB 吗 | **是** | 两个 UB 写进不同的已知值，两个子块各存自己那份 |
| 不带 `acc_to_vec_mode` 的 move 默认是哪个 | **`SingleModeVec0`** | 先把两个 UB 都种成 1×，再不带 mode 写 2× |
| dual + `relu_pre_mode` | **当时抛 aicore 异常**（`device error type 3 / 507015`），现在解析期就拒 | 当时在子进程里量，见 §三 |

第三条不是学术问题：`fixpipe_paths` / `fixpipe_quant_matrix` 那 39 条全是
「不带 mode 的 move + `if sub_id == 0` 的存」，一直依赖这一格的答案，只是没写下来。

## 二、「dual 不承载 rider」的三种形式

ascriptor 的规则是一句话：双目标不承载任何 rider。落到这一版 pypto_pro 上是三种后果：

ascriptor 的规则是一句话：**双目标只承载同类型平凡拷贝**。「不承载任何 rider」
落到实际有三种形式，2026-09-27 之前它们**拦在三个不同的层**：

| 组合 | 那时拦在哪 | 现在 |
| --- | --- | --- |
| dual + `scale` | Python 解析期 | 不变 |
| dual + 降位 / 任何 dtype 转换 | **C++ 编译期**（十几秒 bisheng，报错指不回源码） | **Python 解析期** |
| dual + `relu_pre_mode` | **谁都没拦**，运行期 aicore 异常 507015 并毒化进程 | **Python 解析期** |
| dual + `DN` 目的布局 | **C++ 编译期** | **Python 解析期** |

前三条是同一条 ISA static_assert（`quantPre == NoQuant`）的三种形式,
第四条是 `GetDualDstCtl` 的**另一条** static_assert
（`"Dual Dst Mode is not support in nz2dn."`）。**四条合起来才是完整规则**——
前三条是被问出来的，第四条是拿 ISA 的 static_assert 清单**对账**找出来的
（`references/diagnostics/defects/fixpipe-overview.md` §二点六）。

三种形式里第二种最容易漏：
**`FP32 → FP16` 在 ISA 里本身就是一个 QuantMode（`F322F16`）**，
所以它撞的是同一条「dual 模式要求 `NoQuant`」
（`static_assert(quantPre == NoQuant, "Quant is not support in dual Dst Mode.")`）。
原来那条 Python 检查只看 `scale` 参数，看不到 **dtype 推出来的** QuantMode。

第三条原来是真缺口：`TMov.hpp:233` 把 `dualDstCtl` 和 `reluMode` **并列**传给
`pto_copy_matrix_cc_to_ub`，两边都没有交叉检查，于是生成了一条硬件不接受的指令。
那次实测与根因记在
[defect](../../../../references/diagnostics/defects/fixpipe-dual-dst-plus-relu-faults-aicore.md)。

## 三、relu 那条当初是怎么量的（现在拦在解析期，这段是记录）

aicore 异常之后，**同一个进程里任何 NPU 操作都失败**——连 `torch.zeros` 都报
`aclnnInplaceZero`。第一版探针把七格放在一个进程里，relu 那格炸掉之后
「降位」那格报了一个与它无关的错，误判了一轮。所以那次是用 `subprocess`
一格一个进程量的（形状抄
[`support/debug_probe`](../../support/debug_probe/test.py) 的 `_run_child`,
那里仍然在用）。

**拦到解析期之后这条路走不到运行期了**，所以这个文件不再开子进程,
`kernel.py` 里的 `run_case` 也随之删掉。要亲眼再看一次那个异常,
得先把 `block_ops.py` 里那条检查注释掉——别在共享的卡上这么做。

## 四、没量的：dual + `phase`

`TMov.hpp:233` 里 `unitFlagCtrl`（就是 `phase`）和 `dualDstCtl` 也是并列传的,
同样没有交叉检查，所以**形状上和 relu 那条一模一样**，大概率也是一条坏指令。

没测，理由是风险不对等：`phase` 的 UF 链不平衡会让 AI Core **挂住**
（`test_scale_phase_atomic_order.py` 的注释写着 "leaves the UF state machine
unbalanced and hangs the AI Core"）——挂住要重置卡，而 relu 那条只是抛异常。
拿一张共享的卡去换一格已经能从源码读出结论的信息，不值。

## 五、判据

`rtol=0, atol=0`。目的类型 FP32、不带 scale，输入是小整数，`a @ b` 在 FP32 精确。
"""

from pathlib import Path

import numpy as np
import pytest
import torch
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
# 四、dual + rider：三种后果
# ---------------------------------------------------------------------------


def test_a_dual_mode_rejects_scale_at_parse_time(device):
    """dual + `scale`：**Python 解析期就拒**，这是三条里最好的一种后果。

    传 CPU 张量就够——拒绝发生在 parse。

    **这个 kernel 的目的是 `FP32 → FP32`，所以它同时触犯了另一条拦截**
    （那一格带 scale 会被静默丢，见 `fixpipe_quant_matrix` §三）。断言的是
    **dual 那条先报**：dual 不承载 scale 是硬件限制，任何 dtype 都一样,
    而 FP32 那条会让人以为换成 FP16 就好了。

    2026-09-27 这条测试抓到了框架里检查顺序反了——它是先报 FP32 那条的。
    对应的 pypto 侧回归：`test_dual_mode_is_reported_before_the_fp32_defect`。
    """
    a, b = ref.make_operands(M, K, N)
    with pytest.raises(NotSupported, match="scale cannot be combined with dual-mode"):
        dual_with_scale(torch.from_numpy(a), torch.from_numpy(b), torch.zeros(M, N))


def test_a_dual_mode_rejects_a_narrowing_dtype(device):
    """dual + `FP32 → FP16`（**不带 scale**）：**解析期就拒**。

    `FP32 → FP16` 在不带 scale 的组合表里是合法的（`fixpipe_quant_matrix` 量过），
    所以这一格看起来只是个 dtype 转换。但 ISA 实例化出来 `quantPre = F322F16`,
    撞的是
    `static_assert(quantPre == QuantMode_t::NoQuant, "Quant is not support in dual Dst Mode.")`
    （`pto/npu/a5/common.hpp:187`）。

    **这说明「降位」本身在 ISA 里就是一种 QuantMode**，不是一个独立的 dtype 转换——
    这才是 ascriptor 那句「连不带 deqScalar 的浮点降位也不支持」的机制,
    也是「dual 只承载同类型平凡拷贝」这条规则的完整形式。

    **2026-09-27 之前它拦在 C++ 编译期**：这条测试原来断言
    `RuntimeFailure: Failed to compile kernel`，要等十几秒 bisheng，
    而且报错消息里没有用户源码的行号。现在拦在解析期,
    所以这条测试也从「本文件第二慢」变成了瞬时。
    """
    a, b = ref.make_operands(M, K, N)
    out = _zeros(device, dtype=torch.float16)
    with pytest.raises(NotSupported, match="dual-mode acc_to_vec_mode cannot convert"):
        dual_narrowing_dtype(_dev(a, device), _dev(b, device), out)


def test_a_dual_mode_rejects_a_dn_destination(device):
    """dual + **`DN` 目的布局**：解析期拒。

    前三条排除规则都避开了（同类型、不带 scale、不带 relu），撞的是
    `GetDualDstCtl` 的**第二条** static_assert：
    `"Dual Dst Mode is not support in nz2dn."`（`common.hpp:188`）。
    `DN` 的 Tile 正好是 `!isRowMajor && SFractal == NoneBox`。

    **这一格是对账找出来的，不是撞出来的。** 把 `CheckTMovAccValid` 与
    `GetDualDstCtl` 的八条 static_assert 逐条对 pypto 的覆盖时，
    发现这一条谁都没拦、本章也从没量过（所有 kernel 都用 `ND`）。
    方法记在 [`diagnostics/defects/fixpipe-overview.md`](../../../../references/diagnostics/defects/fixpipe-overview.md) §二点
    六。
    """
    a, b = ref.make_operands(M, K, N)
    out = _zeros(device)
    with pytest.raises(NotSupported, match="does not support a DN destination layout"):
        dual_dn_destination(_dev(a, device), _dev(b, device), out)


def test_a_dual_mode_with_relu_is_rejected(device):
    """dual + `relu_pre_mode`：**解析期就拒**（2026-09-27 起）。

    在这之前**谁都没拦**：`TMov.hpp:233` 把 `dualDstCtl` 和 `reluMode` **并列**
    传给 `pto_copy_matrix_cc_to_ub`，两边都没有交叉检查，于是生成了一条硬件不接受
    的指令——运行期 aicore 异常 507015，而且**异常之后同一个进程里任何 NPU 操作
    都失败**。那次实测是在子进程里做的，记在
    [defect](../../../../references/diagnostics/defects/fixpipe-dual-dst-plus-relu-faults-aicore.md)。

    我当时的预测也错了：按「C++ 的 `TExtractL0CToUBDualDst` 不收 relu 模板参数」
    推「静默丢弃」，实际是设备异常。**静态读 C++ 谓词第三次给出错的预测**
    （前两次见设计文档 §十四点三）。

    **判据只匹配「dual 不承载 relu」这半句，不匹配 507015。** 原因是这条测试
    自己踩过的坑：改成解析期拒之后，旧判据 `assert "507015" in text` **仍然通过**,
    因为拦截的错误消息里引用了那个异常号——判据变成了同义反复。
    记在 `references/verification/evidence.md`。
    """
    a, b = ref.make_operands(M, K, N)
    out = _zeros(device, dtype=torch.float32)
    with pytest.raises(
        NotSupported, match="relu_pre_mode cannot be combined with dual-mode"
    ):
        dual_with_relu(_dev(a, device), _dev(b, device), out)


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
