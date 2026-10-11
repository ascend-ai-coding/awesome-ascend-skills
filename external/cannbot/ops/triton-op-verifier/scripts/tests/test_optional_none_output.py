# ----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ----------------------------------------------------------------------------------------------------------

"""run_single_case 对"可选输出为 None"的判定规则单元测试。

背景：大量 benchmark 参考实现带可选输出（`output_final_state` / `has_q` 等开关），
关闭时返回 None。原规则 `if fw_out is None or impl_out is None: raise` 会把
"framework 与 impl 都返回 None"（语义完全一致）也判为失败。

规则修正为：
- 双方都是 None  → 该输出语义一致，跳过比对，继续检查后续输出；
- 仅单侧是 None  → 输出契约不一致，仍然报错（不放宽）。

运行环境仅需 CPU torch（无需 NPU）。
"""
import torch
import pytest

import verify


CPU = torch.device("cpu")


class _FixedOutputModel:
    """按固定序列返回输出的桩模型（忽略入参）。"""

    def __init__(self, outputs):
        self._outputs = outputs

    def __call__(self, *args, **kwargs):
        return self._outputs


def _run(fw_outputs, impl_outputs, case_idx=1, total_cases=1):
    """以给定的 framework / impl 输出序列跑一次 run_single_case。"""
    models = verify.ModelPair(
        framework=_FixedOutputModel(fw_outputs),
        impl=_FixedOutputModel(impl_outputs),
    )
    ctx = verify.CaseContext(case_idx=case_idx, total_cases=total_cases)
    verify.run_single_case(models, [torch.ones(4)], CPU, ctx)


def test_both_none_is_skipped():
    """双方均为 None 的可选输出应被跳过，不抛异常。"""
    out = torch.ones(8)
    _run((out, None), (out.clone(), None))


def test_framework_none_impl_tensor_still_fails():
    """framework 返回 None 而 impl 返回张量 —— 契约不一致，必须报错。"""
    out = torch.ones(8)
    with pytest.raises(AssertionError, match=r"输出 1 为 None"):
        _run((out, None), (out.clone(), torch.empty(1)))


def test_impl_none_framework_tensor_still_fails():
    """impl 返回 None 而 framework 返回张量 —— 实现漏写输出，必须报错。"""
    out = torch.ones(8)
    with pytest.raises(AssertionError, match=r"输出 1 为 None"):
        _run((out, torch.ones(8)), (out.clone(), None))


def test_skip_does_not_swallow_later_mismatch():
    """跳过 None 输出后，其后输出的精度比对必须照常执行。"""
    good = torch.ones(8)
    bad = torch.full((8,), 5.0)
    with pytest.raises((AssertionError, verify.AccuracyError)):
        _run((good, None, good), (good.clone(), None, bad))


def test_multiple_none_slots_all_skipped():
    """多个可选输出同时关闭时全部跳过。"""
    out = torch.ones(8)
    _run((out, None, None), (out.clone(), None, None))


def test_none_only_output_is_skipped():
    """唯一输出即为 None（整组输出都是可选）时也不应报错。"""
    _run((None,), (None,))
