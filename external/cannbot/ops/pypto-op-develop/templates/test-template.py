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
  - 本文件是 test_{op}.py 的固定模板，由 pypto-op-develop 生成。
  - 所有 {op} 占位符需替换为实际算子名称。
  - 测试用例信息放在 test_cases.json，遍历读取执行。
  - test_{op}.py 只做 import + 遍历 + 调用 + 精度对比，不包含 golden 或 kernel 实现代码。
  - golden 实现来自 {op}_golden.py（由 pypto-golden-generate 生成）。
  - kernel 实现来自 {op}_impl.py（由 pypto-op-develop 生成）。
  - 精度对比必须使用 numpy.testing.assert_allclose，禁止手写 assert max_diff < tolerance。
  - 模式参照 examples/ 与 models/ 的统一规范。
"""

import os
import sys
import json
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
        print("Please set: export TILE_FWK_DEVICE_ID={device_id}")
        return 0
    try:
        return int(os.environ["TILE_FWK_DEVICE_ID"])
    except ValueError:
        print(f"ERROR: TILE_FWK_DEVICE_ID must be int, got: {os.environ['TILE_FWK_DEVICE_ID']}")
        return 0

def load_test_cases(json_path="test_cases.json"):
    """加载测试用例。"""
    if not os.path.exists(json_path):
        print(f"ERROR: {json_path} not found")
        sys.exit(1)
    with open(json_path, "r") as f:
        return json.load(f)

# ─────────────────────────────────────────────
# 2. 测试执行
# ─────────────────────────────────────────────

def run_single_case(case_data, device_id=None, run_mode="npu"):
    """执行单个测试用例。"""
    case_id = case_data["id"]
    description = case_data.get("description", "")
    
    print("=" * 60)
    print(f"Test: {case_id} — {description}")
    print("=" * 60)
    
    device = f"npu:{device_id}" if (run_mode == "npu" and device_id is not None) else "cpu"
    
    # 构造输入数据
    torch.manual_seed(case_data.get("seed", 42))
    
    # 根据 test_cases.json 格式构造输入（需根据具体算子适配）
    # 示例：单输入算子
    input_shape = case_data["input"]["shape"]
    input_dtype = case_data["input"]["dtype"]
    
    dtype_map = {
        "float16": torch.float16,
        "float32": torch.float32,
        "bfloat16": torch.bfloat16,
    }
    dtype = dtype_map.get(input_dtype, torch.float32)
    x = torch.randn(input_shape, dtype=dtype, device=device)
    
    # 执行 kernel wrapper
    result = {op}_wrapper(x)
    
    # 执行 golden
    golden = {op}_golden(x)
    
    # 精度对比
    print(f"  Input shape : {x.shape}")
    print(f"  Output shape: {result.shape}")
    max_diff = np.abs(result.cpu().numpy() - golden.cpu().numpy()).max()
    print(f"  Max diff    : {max_diff:.6e}")
    
    if run_mode == "npu":
        try:
            assert_allclose(
                result.cpu().numpy(),
                golden.cpu().numpy(),
                rtol=case_data.get("rtol", 1e-3),
                atol=case_data.get("atol", 1e-3),
            )
            print("[PRECISION_PASS]")
        except AssertionError as e:
            print(f"[PRECISION_FAIL] {e}", file=sys.stderr)
            raise
        except Exception as e:
            print(f"Runtime error: {e}", file=sys.stderr)
            raise
    
    print("  ✓ Passed\n")

# ─────────────────────────────────────────────
# 3. CLI 入口
# ─────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="PyPTO {op} operator test",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s                     Run all cases
  %(prog)s case_001            Run specific case
  %(prog)s --list              List all cases
        """,
    )
    parser.add_argument("case_id", type=str, nargs="?", help="Case ID to run")
    parser.add_argument("--list", action="store_true", help="List available cases")
    parser.add_argument(
        "--run_mode", "--run-mode",
        type=str, default="npu", choices=["npu", "sim"],
        help="Run mode (default: npu)",
    )
    parser.add_argument(
        "--json", type=str, default="test_cases.json",
        help="Test cases JSON file (default: test_cases.json)",
    )
    args = parser.parse_args()
    
    # 加载测试用例
    test_cases = load_test_cases(args.json)
    cases = test_cases.get("test_cases", [])
    
    if not cases:
        print("ERROR: No test cases found in JSON")
        sys.exit(1)
    
    # --list
    if args.list:
        print(f"\nTest cases from {args.json}:\n")
        for case in cases:
            case_id = case["id"]
            desc = case.get("description", "")
            shape = case["input"]["shape"]
            dtype = case["input"]["dtype"]
            print(f"  {case_id}  — {desc}  [{dtype} {shape}]")
        return
    
    # 选择用例
    if args.case_id:
        case_data = None
        for case in cases:
            if case["id"] == args.case_id:
                case_data = case
                break
        if case_data is None:
            print(f"ERROR: unknown case '{args.case_id}'")
            print(f"Valid: {', '.join([c['id'] for c in cases])}")
            sys.exit(1)
        to_run = [case_data]
    else:
        to_run = cases
    
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
        passed = 0
        for case_data in to_run:
            print(f"\n▸ Running {case_data['id']}")
            run_single_case(case_data, device_id, args.run_mode)
            passed += 1
        
        print("\n" + "=" * 60)
        print(f"All tests passed! ({passed}/{len(to_run)} cases)")
        print("=" * 60)
    except Exception as e:
        print(f"\nError: {e}")
        raise


if __name__ == "__main__":
    main()
