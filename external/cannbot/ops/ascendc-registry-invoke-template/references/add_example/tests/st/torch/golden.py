# ----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ----------------------------------------------------------------------------------------------------------
"""
CPU Golden 计算模块

提供算子参考实现的 CPU 端计算，用于精度比对。
每个算子对应一个 compute_golden_* 函数，输入输出均为 torch.Tensor（CPU）。
"""

import torch


def compute_golden_add(x1: torch.Tensor, x2: torch.Tensor) -> torch.Tensor:
    """计算逐元素加法的 golden 结果: out = x1 + x2"""
    return x1 + x2


# ============================================================================
# Golden 正确性自测
# ============================================================================

def test_golden_correctness() -> bool:
    """验证 golden 函数本身的正确性"""
    test_cases = [
        {
            "name": "浮点数加法",
            "x1": torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0]),
            "x2": torch.tensor([5.0, 4.0, 3.0, 2.0, 1.0]),
            "expected": torch.tensor([6.0, 6.0, 6.0, 6.0, 6.0]),
        },
        {
            "name": "正负数抵消",
            "x1": torch.tensor([-1.0, 2.5, -3.7, 0.0, 100.5]),
            "x2": torch.tensor([1.0, -2.5, 3.7, 0.0, -100.5]),
            "expected": torch.tensor([0.0, 0.0, 0.0, 0.0, 0.0]),
        },
        {
            "name": "INT32 整数加法",
            "x1": torch.tensor([100, 200, -300, 0, 500], dtype=torch.int32),
            "x2": torch.tensor([50, -100, 300, 0, -500], dtype=torch.int32),
            "expected": torch.tensor([150, 100, 0, 0, 0], dtype=torch.int32),
        },
    ]

    all_passed = True
    for tc in test_cases:
        result = compute_golden_add(tc["x1"], tc["x2"])
        passed = torch.equal(result, tc["expected"])
        print(f"  {tc['name']}: {'PASS' if passed else 'FAIL'}")
        all_passed = all_passed and passed

    return all_passed
