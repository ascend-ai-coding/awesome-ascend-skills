#!/usr/bin/env python3
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
ST 测试 - ACLNN 两段式接口 PyTorch 接入验证

数据流：
    PyTorch Tensor (NPU) -> torch_adapter.so -> ACLNN -> NPU 结果
    CPU golden (golden.py)  <-compare.py->  NPU 实际结果（取回 CPU 后比对）

使用方法：
    python3 test.py --lib /path/to/libtorch_adapter.so
    python3 test.py --lib libtorch_adapter.so --case 0
"""

import torch
import argparse
import sys
import os
import time

from golden import compute_golden_add, test_golden_correctness
from compare import compare_results

CASE_COOLDOWN_MS = int(os.environ.get("CASE_COOLDOWN_MS", 100))

# 测试用例原始数据定义（用于每次创建全新 tensor）
TEST_CASES_DATA = [
    {
        "name": "FP32 基础加法",
        "shape": (2, 3),
        "dtype": torch.float32,
        "x1_data": [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]],
        "x2_data": [[10.0, 20.0, 30.0], [40.0, 50.0, 60.0]],
    },
    {
        "name": "FP32 正负数混合",
        "shape": (6,),
        "dtype": torch.float32,
        "x1_data": [-1.5, 2.5, -3.7, 0.0, 100.5, -200.8],
        "x2_data": [1.5, -2.5, 3.7, 0.0, -100.5, 200.8],
    },
    {
        "name": "FP32 大 shape",
        "shape": (32,),
        "dtype": torch.float32,
        "x1_data": [1.5] * 32,
        "x2_data": [2.5] * 32,
    },
    {
        "name": "FP32 多维张量",
        "shape": (2, 3, 4),
        "dtype": torch.float32,
        "x1_data": [[[1.0]*4]*3]*2,
        "x2_data": [[[2.0]*4]*3]*2,
    },
    {
        "name": "INT32 基础加法",
        "shape": (2, 3),
        "dtype": torch.int32,
        "x1_data": [[10, 20, 30], [40, 50, 60]],
        "x2_data": [[5, 10, 15], [20, 25, 30]],
    },
    {
        "name": "INT32 正负数混合",
        "shape": (6,),
        "dtype": torch.int32,
        "x1_data": [-100, 200, -300, 0, 500, -600],
        "x2_data": [100, -200, 300, 0, -500, 600],
    },
    {
        "name": "INT32 大 shape",
        "shape": (64,),
        "dtype": torch.int32,
        "x1_data": [100] * 64,
        "x2_data": [200] * 64,
    },
    {
        "name": "边界条件 - 单个元素",
        "shape": (1,),
        "dtype": torch.float32,
        "x1_data": [123.456],
        "x2_data": [876.544],
    },
    {
        "name": "FP32 极小值和极大值",
        "shape": (6,),
        "dtype": torch.float32,
        "x1_data": [
            torch.finfo(torch.float32).tiny,
            torch.finfo(torch.float32).max,
            torch.finfo(torch.float32).eps,
            0.0, 0.0, 0.0
        ],
        "x2_data": [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
    },
    {
        "name": "FP32 零值测试",
        "shape": (4,),
        "dtype": torch.float32,
        "x1_data": [0.0, -0.0, 1.5, -2.5],
        "x2_data": [0.0, 0.0, 0.0, 0.0],
    },
]


def get_test_cases():
    """返回测试用例（包含原始数据用于 NPU 测试）"""
    return [
        {
            "name": tc["name"],
            "shape": tc["shape"],
            "dtype": tc["dtype"],
            "x1_data": tc["x1_data"],
            "x2_data": tc["x2_data"],
            "x1": torch.tensor(tc["x1_data"], dtype=tc["dtype"]),
            "x2": torch.tensor(tc["x2_data"], dtype=tc["dtype"]),
        }
        for tc in TEST_CASES_DATA
    ]


# ============================================================================
# 单条测试执行
# ============================================================================

def run_test(tc):
    print(f"\n测试: {tc['name']}")
    print(f"  Shape: {tc['shape']}, Dtype: {tc['dtype']}")

    x1_cpu = tc["x1"].clone()
    x2_cpu = tc["x2"].clone()
    assert x1_cpu.shape == tc["shape"], (
        f"用例 '{tc['name']}' x1 shape 不匹配: {x1_cpu.shape} vs {tc['shape']}"
    )
    assert x2_cpu.shape == tc["shape"], (
        f"用例 '{tc['name']}' x2 shape 不匹配: {x2_cpu.shape} vs {tc['shape']}"
    )

    if not torch.npu.is_available():
        print("  [SKIP] NPU 不可用")
        return None

    # 每次创建全新 tensor，避免复用导致的缓存问题
    x1 = torch.tensor(tc["x1_data"], dtype=tc["dtype"], device="npu")
    x2 = torch.tensor(tc["x2_data"], dtype=tc["dtype"], device="npu")

    # 执行算子（异步，通过 OpCommand 入 queue 保证顺序）
    result = torch.ops.add_example.forward(x1, x2)

    # 取回结果（cpu() 内部会隐式同步）
    result_cpu = result.cpu()

    golden = compute_golden_add(x1_cpu, x2_cpu)

    passed = compare_results(golden, result_cpu)

    print(f"  结果: {'PASS' if passed else 'FAIL'}")

    # 用例间冷却，避免 NPU 内存池复用导致的时序问题
    if CASE_COOLDOWN_MS > 0:
        time.sleep(CASE_COOLDOWN_MS / 1000.0)

    return passed


# ============================================================================
# 主函数
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description="ST 测试 - ACLNN 两段式接口 PyTorch 接入验证")
    parser.add_argument("--lib", required=True, help="共享库路径 (libtorch_adapter.so)")
    parser.add_argument("--case", type=int, default=None,
                        help="执行指定测试用例编号 (0-based)")
    args = parser.parse_args()

    print(f"\n加载共享库: {args.lib}")
    args.lib = os.path.realpath(args.lib)
    if not os.path.exists(args.lib):
        print(f"错误: 文件不存在 {args.lib}")
        sys.exit(1)

    torch.ops.load_library(args.lib)
    print("共享库加载成功")

    golden_passed = test_golden_correctness()
    print(f"\nGolden 自测: {'PASS' if golden_passed else 'FAIL'}")

    test_cases = get_test_cases()

    if args.case is not None:
        if args.case < 0 or args.case >= len(test_cases):
            print(f"错误: 测试用例编号超出范围 (0-{len(test_cases)-1})")
            sys.exit(1)
        test_cases = [test_cases[args.case]]

    results = {}

    print(f"\n{'=' * 60}")
    print("设备: NPU")
    print(f"{'=' * 60}")

    passed_count = 0
    failed_count = 0
    skipped_count = 0

    for i, tc in enumerate(test_cases):
        result = run_test(tc)
        if result is None:
            skipped_count += 1
            continue
        elif result:
            passed_count += 1
        else:
            failed_count += 1
            results[tc['name']] = False
            continue

        results[tc['name']] = result

    print(f"\nNPU 测试汇总:")
    print(f"  通过: {passed_count}")
    print(f"  失败: {failed_count}")
    print(f"  跳过: {skipped_count}")

    print(f"\n{'=' * 60}")
    print("总体测试报告")
    print(f"{'=' * 60}")

    total_passed = sum(1 for v in results.values() if v)
    total_failed = sum(1 for v in results.values() if not v)

    print(f"总计: {len(results)}")
    print(f"通过: {total_passed}")
    print(f"失败: {total_failed}")
    print(f"{'=' * 60}")

    if total_failed > 0:
        print("\n失败的测试用例:")
        for name, passed in results.items():
            if not passed:
                print(f"  - {name}")

    return 0 if total_failed == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
