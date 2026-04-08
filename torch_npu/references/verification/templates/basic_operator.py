#!/usr/bin/env python3
"""
基础算子验证模板

提供通用的算子精度验证函数和示例用法。
"""

import torch
from typing import Callable, List, Dict, Any


def verify_operator(
    op_func: Callable,
    inputs_spec: List[Dict[str, Any]],
    rtol: float = 1e-4,
    atol: float = 1e-5,
    verbose: bool = True,
) -> bool:
    """
    验证算子在 CPU 和 NPU 上的输出精度

    Args:
        op_func: 待验证的算子函数，签名: op_func(*inputs) -> Tensor
        inputs_spec: 输入规格列表，每个元素是字典，包含 'shape' 和 'dtype'
        rtol: 相对误差容差
        atol: 绝对误差容差
        verbose: 是否打印详细验证信息

    Returns:
        bool: 验证是否通过（True: 通过，False: 失败）

    示例:
        >>> def add_func(a, b):
        ...     return a + b
        >>> inputs_spec = [
        ...     {'shape': (2, 3, 4), 'dtype': torch.float32},
        ...     {'shape': (2, 3, 4), 'dtype': torch.float32}
        ... ]
        >>> verify_operator(add_func, inputs_spec)
    """
    if not torch.npu.is_available():
        raise RuntimeError(
            "NPU is not available. Please check if NPU is connected and drivers are installed."
        )

    # 1. 构造输入数据
    inputs_cpu = [torch.randn(**spec) for spec in inputs_spec]
    inputs_npu = [inp.clone().npu() for inp in inputs_cpu]

    # 2. CPU 执行
    try:
        output_cpu = op_func(*inputs_cpu)
    except Exception as e:
        raise RuntimeError(f"CPU execution failed: {e}")

    # 3. NPU 执行
    try:
        output_npu = op_func(*inputs_npu).cpu()
    except Exception as e:
        raise RuntimeError(f"NPU execution failed: {e}")

    # 4. 精度对比
    diff = torch.abs(output_cpu - output_npu)
    rel_diff = diff / (torch.abs(output_cpu) + 1e-8)

    passed = (diff < atol).all() and (rel_diff < rtol).all()

    # 5. 输出结果
    if verbose:
        print(f"Operator: {op_func.__name__}")
        print(f"  Inputs: {inputs_spec}")
        print(f"  Max abs diff: {diff.max().item():.6e}")
        print(f"  Max rel diff: {rel_diff.max().item():.6e}")
        print(f"  Result: {'✓ PASS' if passed else '✗ FAIL'}")
        if not passed:
            print(f"  CPU output: {output_cpu[:3]}")
            print(f"  NPU output: {output_npu[:3]}")

    return passed


# ============================================
# 使用示例
# ============================================


def example_add_operator():
    """验证加法算子"""

    def add_func(a, b):
        return a + b

    inputs_spec = [
        {"shape": (2, 3, 4), "dtype": torch.float32},
        {"shape": (2, 3, 4), "dtype": torch.float32},
    ]

    print("=" * 50)
    print("Example: Add Operator")
    print("=" * 50)
    passed = verify_operator(add_func, inputs_spec)
    return passed


def example_matmul_operator():
    """验证矩阵乘法算子"""

    def matmul_func(a, b):
        return torch.matmul(a, b)

    inputs_spec = [
        {"shape": (4, 5), "dtype": torch.float16},
        {"shape": (5, 6), "dtype": torch.float16},
    ]

    print("\n" + "=" * 50)
    print("Example: MatMul Operator")
    print("=" * 50)
    passed = verify_operator(matmul_func, inputs_spec)
    return passed


def example_conv2d_operator():
    """验证卷积算子"""

    def conv2d_func(input_tensor, weight, bias=None, stride=1, padding=1):
        return torch.nn.functional.conv2d(
            input_tensor, weight, bias, stride=stride, padding=padding
        )

    inputs_spec = [
        {"shape": (1, 3, 8, 8), "dtype": torch.float32},
        {"shape": (16, 3, 3, 3), "dtype": torch.float32},
        {"shape": (16,), "dtype": torch.float32},
    ]

    print("\n" + "=" * 50)
    print("Example: Conv2d Operator")
    print("=" * 50)
    passed = verify_operator(conv2d_func, inputs_spec, rtol=1e-4, atol=1e-5)
    return passed


def example_elementwise_operations():
    """验证多种算术运算"""
    ops = [
        (lambda a, b: a - b, "Subtract", [(3, 4), (3, 4)]),
        (lambda a, b: a * b, "Multiply", [(2, 3), (2, 3)]),
        (
            lambda a, b: torch.div(a, b, rounding_mode="floor"),
            "Divide",
            [(4, 3), (4, 3)],
        ),
        (lambda a: torch.abs(a), "Abs", [(3, 4)]),
        (lambda a: torch.sqrt(a), "Sqrt", [(3, 2)]),
    ]

    print("\n" + "=" * 50)
    print("Example: Element-wise Operations")
    print("=" * 50)

    results = []
    for op_func, op_name, input_shapes in ops:
        inputs_spec = [
            {"shape": shape, "dtype": torch.float32} for shape in input_shapes
        ]
        passed = verify_operator(op_func, inputs_spec, verbose=False)
        results.append((op_name, passed))

    for op_name, passed in results:
        status = "✓ PASS" if passed else "✗ FAIL"
        print(f"  {op_name}: {status}")

    return all(passed for _, passed in results)


# ============================================
# 主程序入口
# ============================================

if __name__ == "__main__":
    print("PyTorch NPU Operator Verification Template")
    print("=" * 50)

    try:
        # 运行所有示例
        results = []

        results.append(("Add Operator", example_add_operator()))
        results.append(("MatMul Operator", example_matmul_operator()))
        results.append(("Conv2d Operator", example_conv2d_operator()))
        results.append(("Element-wise Ops", example_elementwise_operations()))

        # 汇总结果
        print("\n" + "=" * 50)
        print("Summary")
        print("=" * 50)
        for op_name, passed in results:
            status = "✓ PASS" if passed else "✗ FAIL"
            print(f"  {op_name}: {status}")

        all_passed = all(passed for _, passed in results)
        print(f"\nOverall: {'✓ ALL PASSED' if all_passed else '✗ SOME FAILED'}")

    except RuntimeError as e:
        print(f"\nError: {e}")
        exit(1)
