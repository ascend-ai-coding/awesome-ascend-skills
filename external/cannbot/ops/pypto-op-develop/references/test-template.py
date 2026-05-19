#!/usr/bin/env python3
# coding: utf-8
# ----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ----------------------------------------------------------------------------------------------------------

"""PyPTO {op} operator test.

模板说明：
  - 本文件是 test_{op}.py 的固定模板，由 pypto-op-develop 在 Stage 3 生成。
  - 所有 {op} 占位符需替换为实际算子名称。
  - test_{op}.py 只做 import + 调用 + 精度对比，不包含 golden 或 kernel 实现代码。
  - golden 实现来自 {op}_golden.py（Stage 2A 由 pypto-golden-generate 生成）。
  - kernel 实现来自 {op}_impl.py（Stage 3 由 pypto-op-develop 生成）。
  - 精度对比必须使用 numpy.testing.assert_allclose，禁止手写 assert max_diff < tolerance。
  - 模式参照 examples/ 与 models/ 的统一规范。
"""

import os
import sys
import argparse

import torch
import numpy as np
from numpy.testing import assert_allclose

from {op}_golden import {op}_golden
from {op}_impl import {op}_wrapper

# ─────────────────────────────────────────────
# 1. 环境工具
# ─────────────────────────────────────────────

def get_device_id():
    """从环境变量获取 TILE_FWK_DEVICE_ID。"""
    if "TILE_FWK_DEVICE_ID" not in os.environ:
        print("Please set: export TILE_FWK_DEVICE_ID=0")
        return None
    try:
        return int(os.environ["TILE_FWK_DEVICE_ID"])
    except ValueError:
        print(f"ERROR: TILE_FWK_DEVICE_ID must be int, got: {os.environ['TILE_FWK_DEVICE_ID']}")
        return None

# ─────────────────────────────────────────────
# 2. 测试函数
# ─────────────────────────────────────────────

def test_{op}_level0(device_id=None, run_mode="npu"):
    """Level 0: 小数据量基础功能验证（8-16 元素）。"""
    print("=" * 60)
    print("Test: {op} Level 0 (basic)")
    print("=" * 60)

    device = f"npu:{device_id}" if (run_mode == "npu" and device_id is not None) else "cpu"

    # 测试数据
    torch.manual_seed(0)
    shape = (1, 16)  # Level 0: 小 shape
    dtype = torch.float32
    x = torch.randn(shape, dtype=dtype, device=device)

    # 执行 kernel wrapper
    result = {op}_wrapper(x)

    # 执行 golden
    golden = {op}_golden(x)

    # 精度对比（必须使用 assert_allclose）
    print(f"  Input shape : {x.shape}")
    print(f"  Output shape: {result.shape}")
    max_diff = np.abs(result.cpu().numpy() - golden.cpu().numpy()).max()
    print(f"  Max diff    : {max_diff:.6e}")

    if run_mode == "npu":
        assert_allclose(
            result.cpu().numpy(),
            golden.cpu().numpy(),
            rtol=1e-3, atol=1e-3,
        )

    print("  ✓ Passed\n")


def test_{op}_level1(device_id=None, run_mode="npu"):
    """Level 1: 典型场景验证（1K 元素）。"""
    print("=" * 60)
    print("Test: {op} Level 1 (typical)")
    print("=" * 60)

    device = f"npu:{device_id}" if (run_mode == "npu" and device_id is not None) else "cpu"

    torch.manual_seed(42)
    shape = (4, 1024)  # Level 1: 典型 shape
    dtype = torch.float32
    x = torch.randn(shape, dtype=dtype, device=device)

    result = {op}_wrapper(x)
    golden = {op}_golden(x)

    max_diff = np.abs(result.cpu().numpy() - golden.cpu().numpy()).max()
    print(f"  Shape: {shape}, Max diff: {max_diff:.6e}")

    if run_mode == "npu":
        assert_allclose(
            result.cpu().numpy(),
            golden.cpu().numpy(),
            rtol=1e-3, atol=1e-3,
        )

    print("  ✓ Passed\n")

# ─────────────────────────────────────────────
# 3. CLI 入口
# ─────────────────────────────────────────────

# 用例注册表
EXAMPLES = {
    "{op}::test_{op}_level0": {
        "name": "{op} Level 0",
        "description": "小数据量基础功能验证",
        "function": test_{op}_level0,
    },
    "{op}::test_{op}_level1": {
        "name": "{op} Level 1",
        "description": "典型场景验证",
        "function": test_{op}_level1,
    },
}


def main():
    parser = argparse.ArgumentParser(
        description="PyPTO {op} operator test",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s {op}::test_{op}_level0    Run Level 0
  %(prog)s --list                    List all cases
        """,
    )
    parser.add_argument("example_id", type=str, nargs="?", help="Case ID to run")
    parser.add_argument("--list", action="store_true", help="List available cases")
    parser.add_argument(
        "--run_mode", "--run-mode",
        type=str, default="npu", choices=["npu", "sim"],
        help="Run mode (default: npu)",
    )
    args = parser.parse_args()

    # --list
    if args.list:
        print("\nAvailable cases:\n")
        for key, info in sorted(EXAMPLES.items()):
            print(f"  {key}  — {info['description']}")
        return

    # 选择用例
    if args.example_id:
        if args.example_id not in EXAMPLES:
            print(f"ERROR: unknown case '{args.example_id}'")
            print(f"Valid: {', '.join(sorted(EXAMPLES))}")
            sys.exit(1)
        to_run = [(args.example_id, EXAMPLES[args.example_id])]
    else:
        to_run = list(sorted(EXAMPLES.items()))

    # NPU 设备初始化
    device_id = None
    if args.run_mode == "npu":
        device_id = get_device_id()
        if device_id is None:
            return
        import torch_npu
        torch.npu.set_device(device_id)

    # 执行
    try:
        for key, info in to_run:
            print(f"\n▸ Running {key}: {info['name']}")
            info["function"](device_id, args.run_mode)
        print("\n" + "=" * 60)
        print("All tests passed!")
        print("=" * 60)
    except Exception as e:
        print(f"\nError: {e}")
        raise


if __name__ == "__main__":
    main()
