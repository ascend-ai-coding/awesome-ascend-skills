#!/usr/bin/env python3
# coding: utf-8
# Copyright (c) 2024-2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

"""PyPTO {op} golden reference implementation.

模板说明：
  - 所有 {op} 占位符需替换为实际算子名称。
  - golden 是基于 torch + torch_npu 的 NPU 参考实现，计算在 NPU 上执行。
  - torch_npu 未安装时直接报错引导安装；仅无 NPU 硬件时回退 CPU。
  - 导出函数 {op}_golden() 供 test_{op}.py 调用。
  - 参考 golden 函数风格（activation、layernorm 等）：用 skill `pypto-docs-search` 搜索 golden 用法范本。
"""

import os
import torch

_DEVICE = None
_HAS_NPU = False

try:
    import torch_npu
    _HAS_NPU = torch.npu.is_available() and torch.npu.device_count() > 0
except ImportError:
    raise ImportError(
        "torch_npu is not installed. Please install it first:\n"
        "  pip install torch_npu"
    )

def _get_device() -> torch.device:
    global _DEVICE
    if _DEVICE is None:
        if not _HAS_NPU:
            _DEVICE = torch.device("cpu")
        else:
            device_id = int(os.environ.get("TILE_FWK_DEVICE_ID", "0"))
            torch.npu.set_device(device_id)
            _DEVICE = torch.device(f"npu:{device_id}")
    return _DEVICE


# ─────────────────────────────────────────────
# Golden 参考实现（NPU torch）
# ─────────────────────────────────────────────

def {op}_golden(x: torch.Tensor) -> torch.Tensor:
    """PyTorch 参考实现。

    根据算子规格中的数学公式实现。
    仅使用 torch 标准操作，不依赖 pypto。

    Args:
        x: 输入 tensor。
           根据实际算子需求调整参数列表（可多输入、可带 gamma/beta/eps 等参数）。

    Returns:
        计算结果 tensor。
        根据实际算子需求调整返回值（可多输出、可返回 tuple）。

    示例（SiLU）::

        return x * torch.sigmoid(x)

    示例（LayerNorm）::

        mean = x.mean(dim=-1, keepdim=True)
        var = x.var(dim=-1, keepdim=True, unbiased=False)
        normalized = (x - mean) / torch.sqrt(var + eps)
        return normalized * gamma + beta
    """
    device = _get_device()
    x = x.to(device)
    # TODO: 替换为实际 golden 逻辑（示例见本函数 docstring）。
    return x


# ==========================================
# 输入构造（供 profiling --factory 使用）
# ==========================================

def _make_inputs(device):
    """构造 P0 典型输入，供验证和性能采集共用。

    对于有语义约束的算子（block table 索引、状态缓存、位置编码等），
    必须在此函数中构造合法值，不能使用随机值。

    SPEC.md 中有多个性能 P0 shape 时，必须为每个 shape 生成一组 case。

    Returns:
        单 P0 shape:
            (args_list, kwargs_dict)
        多 P0 shape:
            [(case_name, args_list, kwargs_dict), ...]

    注意: 所有 tensor 必须带 device=device。

    示例（单 P0 shape）::

        x = torch.randn(8, 1024, dtype=torch.bfloat16, device=device)
        return [x], {}

    示例（多 P0 shape）::

        cases = []
        x = torch.randn(8, 1024, dtype=torch.bfloat16, device=device)
        cases.append(("perf_p0_small", [x], {}))
        x = torch.randn(16, 2048, dtype=torch.bfloat16, device=device)
        cases.append(("perf_p0_large", [x], {}))
        return cases

    示例（复杂算子，有语义约束）::

        x = torch.randn(2, 2, 64, dtype=torch.bfloat16, device=device)
        block_table = _gen_block_table(2, device)  # 合法索引
        state = torch.zeros(5, 128, 32, dtype=torch.float32, device=device)
        return [x, block_table, state], {"eps": 1e-6}
    """
    # TODO: 根据算子规格构造合法输入（示例见本函数 docstring）。
    x = torch.randn(8, 1024, dtype=torch.bfloat16, device=device)
    return [x], {}


# ==========================================
# 验证
# ==========================================

def _validate():
    """自动生成的验证函数 - 运行时动态生成验证报告"""

    device = _get_device()
    print("=" * 60)
    print("{op}_golden 验证报告")
    print("=" * 60)
    print(f"Device: {{device}}")

    # -- 1. 典型 case 验证（来自算子规格中的典型配置）--
    print("\n[典型 case 验证]")
    # TODO: 按算子规格中的典型配置生成验证
    #   注意: torch.randn 等创建 tensor 时必须带 device=device，使输入直接位于目标设备

    # -- 2. 泛化 case 验证（来自算子规格中的动态轴范围）--
    print("\n[泛化 case 验证]")
    # TODO: 按动态轴采样范围验证

    # -- 3. 值域检查（从公式推导）--
    print("\n[值域检查]")
    # TODO: 验证输出值域

    # -- 4. 数值稳定性检查 --
    print("\n[数值稳定性检查]")
    # TODO: 大值、小值、零值等极端输入

    # -- 5. API 对比（如适用）--
    print("\n[API 对比]")
    # TODO: 与 PyTorch 等价 API 对比
    #   注意: 对比双方 tensor 必须在相同 device 上

    print("\n" + "=" * 60)
    print("验证完成")
    print("=" * 60)


if __name__ == "__main__":
    _validate()
