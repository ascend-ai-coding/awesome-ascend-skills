# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

"""上板对照：``silu(A @ B + bias)``，``[256,256]`` FP16 matmul + ``[1,256]`` FP32 bias 广播。

**解决什么.** 融合算子 ``C = silu(A@B+bias)`` 的数值正确性：矩阵乘（FP16 输入、FP32
累加）、bias 融合（cube 侧随 matmul 一次性铺入累加器）、silu 激活（vector 侧 VF）
三段拼接是否算对。

**物理访问.** 见 ``kernel.py`` 模块 docstring：cube 整块算完 ``[256,256]`` 累加器后
片上直传（``DualModeSplitM``，不经 GM）给两个 AIV 各自的半区 UB tile，一次性跨核
信号后，两个 AIV 各自 VF 算 silu、写到 ``out`` 对应半区。GM 中转（独立 ``ws``
scratch）是 tested-and-rejected 形态，「跨核 GM 中转不得复用输出张量」的行为边界归
``references/api/crosscore-sync.md`` 静默错法清单；本版片上直传后不存在任何 GM 中转形参。

**为何这样同步.** 见 ``kernel.py``：cube -> vector 只发生一次、不在循环里，按
``references/api/crosscore-sync.md`` 的决策树走 C 层手写、只需正向同步。

**如何独立验证.**

1. 主测试：随机 A/B/bias，与 CPU FP32 参考比，``rtol=atol=2e-2``（依据见下）。
2. bias 隔离：bias 全零时，kernel 输出应等于同一参考函数在 ``bias=0`` 下的输出——
   把"bias 通路有没有副作用"和"matmul+silu 本身对不对"分开看。bias 融合搬运
   （``Mat -> Bias``）如果顺带改了别的东西，只测随机 bias 未必露出来。

**容差依据.** ``rtol=atol=2e-2`` 沿用本仓 ``examples/matmul/{split_k,gemm_vs_matmul,
fundamental_usage/bias_fusion}/test.py`` 对 FP16 输入、FP32 累加矩阵乘的既有取值——
三个样例都已上板验证过这个阈值对同一数值情形（FP16 matmul + FP32 累加，
K=128~384）合适，不是本次临时选的数。误差来源：(a) 主导项——FP16 矩阵乘在
K=256 项上的 FP32 求和顺序，硬件树形规约与 torch CPU 规约顺序不同；(b) 次要项
——``vf.exp``/``vf.div`` 的超越函数近似，按 ``references/api/vf.md`` §二点三 实测
在正常输入范围下约 1 ulp（相对误差 ~1e-7），比 (a) 小 5 个数量级，不改变整体
容差量级。**验证失败时不放宽这个值**——按 [skill constraints](../../../SKILL.md#constraints-and-workspace-policy)，容差要有出处，
不因为一次不过就现改；不过就如实报告失败与诊断。
"""

from __future__ import annotations

import torch

from .kernel import LAUNCH_BLOCK_DIM, K, M, N, gemm_bias_silu
from .reference import gemm_bias_silu_reference


def _run(a: torch.Tensor, b: torch.Tensor, bias_1d: torch.Tensor) -> torch.Tensor:
    """host 侧把 ``bias`` 从任务约定的 ``[256]`` reshape 成 kernel 要的 ``[1, 256]``。

    连续内存下 reshape 是视图，不拷贝、不改变数据——语义仍是"每行加同一个
    256 长的向量"。kernel 片上直传 cube -> vector（见 ``kernel.py`` 模块
    docstring），不再有 GM 中转 scratch 形参。
    """
    bias_2d = bias_1d.reshape(1, N)
    out = torch.zeros([M, N], device=a.device, dtype=torch.float32)
    gemm_bias_silu[None, LAUNCH_BLOCK_DIM](a, b, bias_2d, out)
    torch.npu.synchronize()
    return out


def test_gemm_bias_silu_matches_reference(device):
    torch.manual_seed(0)
    a = torch.randn([M, K], device=device, dtype=torch.float16)
    b = torch.randn([K, N], device=device, dtype=torch.float16)
    bias = torch.randn([N], device=device, dtype=torch.float32)

    out = _run(a, b, bias)
    expected = gemm_bias_silu_reference(a, b, bias.reshape(1, N))

    torch.testing.assert_close(out.cpu(), expected, rtol=2e-2, atol=2e-2)


def test_zero_bias_matches_plain_gemm_silu(device):
    """bias 全零：把 bias 通路的副作用单独钉住，不被一般输入的误差空间盖过。"""
    torch.manual_seed(1)
    a = torch.randn([M, K], device=device, dtype=torch.float16)
    b = torch.randn([K, N], device=device, dtype=torch.float16)
    bias = torch.zeros([N], device=device, dtype=torch.float32)

    out = _run(a, b, bias)
    expected = gemm_bias_silu_reference(a, b, bias.reshape(1, N))

    torch.testing.assert_close(out.cpu(), expected, rtol=2e-2, atol=2e-2)
