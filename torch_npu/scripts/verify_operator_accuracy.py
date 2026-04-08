#!/usr/bin/env python3
"""
PyTorch NPU 运算精度验证命令行工具

支持通过命令行参数指定算子、输入形状和数据类型进行验证。

用法:
    python3 verify_operator_accuracy.py --op add --shapes 2x3x4 --dtypes fp32
    python3 verify_operator_accuracy.py --op matmul --shapes 4x5x5 --dtypes fp16,bf16
    python3 verify_operator_accuracy.py --op conv2d --shapes 1x3x8x8,16x3x3x3 --dtypes fp32
"""

import argparse
import sys
import os
import torch
from typing import List, Dict, Any, Callable

# 添加父目录到路径以便导入模板
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "references", "verification", "templates"))
from basic_operator import verify_operator


# ============================================
# 预定义算子映射
# ============================================

OPERATORS = {
    "add": lambda a, b: a + b,
    "sub": lambda a, b: a - b,
    "mul": lambda a, b: a * b,
    "div": lambda a, b: torch.div(a, b, rounding_mode="floor"),
    "matmul": lambda a, b: torch.matmul(a, b),
    "conv2d": lambda input_tensor,
    weight,
    bias=None,
    stride=1,
    padding=1: torch.nn.functional.conv2d(
        input_tensor, weight, bias, stride=stride, padding=padding
    ),
    "relu": lambda a: torch.nn.functional.relu(a),
    "sigmoid": lambda a: torch.nn.functional.sigmoid(a),
    "softmax": lambda a: torch.nn.functional.softmax(a, dim=-1),
    "abs": lambda a: torch.abs(a),
    "sqrt": lambda a: torch.sqrt(a),
}


# ============================================
# 数据类型映射
# ============================================

DTYPES = {
    "fp32": torch.float32,
    "fp16": torch.float16,
    "bf16": torch.bfloat16,
    "int32": torch.int32,
    "int64": torch.int64,
}


# ============================================
# 工具函数
# ============================================


def parse_shape(shape_str: str) -> tuple:
    """
    解析形状字符串，例如 "2x3x4" -> (2, 3, 4)

    Args:
        shape_str: 形状字符串，用 'x' 或 'x' 分隔

    Returns:
        tuple: 形状元组
    """
    try:
        return tuple(map(int, shape_str.lower().replace("x", " ").split()))
    except ValueError as e:
        raise ValueError(
            f"Invalid shape string: {shape_str}. Expected format: '2x3x4' or '2 3 4'"
        )


def parse_shapes_list(shapes_str: str) -> List[tuple]:
    """
    解析多个形状字符串，用 ',' 或 ',' 分隔

    Args:
        shapes_str: 形状字符串列表，例如 "2x3x4,4x5x6"

    Returns:
        List[tuple]: 形状元组列表
    """
    shapes_strs = shapes_str.lower().replace(" ", "").split(",")
    return [parse_shape(s) for s in shapes_strs]


def parse_dtypes_list(dtypes_str: str) -> List[torch.dtype]:
    """
    解析数据类型字符串列表

    Args:
        dtypes_str: 数据类型字符串，例如 "fp32,fp16,bf16"

    Returns:
        List[torch.dtype]: 数据类型列表
    """
    dtypes = []
    for dtype_str in dtypes_str.lower().replace(" ", "").split(","):
        if dtype_str not in DTYPES:
            raise ValueError(
                f"Unsupported dtype: {dtype_str}. Supported: {list(DTYPES.keys())}"
            )
        dtypes.append(DTYPES[dtype_str])
    return dtypes


# ============================================
# 验证函数
# ============================================


def verify_with_config(
    op_name: str,
    input_shapes: List[tuple],
    dtypes: List[torch.dtype],
    rtol: float = 1e-4,
    atol: float = 1e-5,
    verbose: bool = True,
) -> bool:
    """
    根据配置执行验证

    Args:
        op_name: 算子名称
        input_shapes: 输入形状列表
        dtypes: 数据类型列表
        rtol: 相对误差容差
        atol: 绝对误差容差
        verbose: 是否打印详细信息

    Returns:
        bool: 是否通过验证
    """
    if op_name not in OPERATORS:
        raise ValueError(
            f"Unsupported operator: {op_name}. "
            f"Supported operators: {list(OPERATORS.keys())}"
        )

    op_func = OPERATORS[op_name]

    # 构造输入规格
    inputs_spec = [
        {"shape": shape, "dtype": dtype} for shape in input_shapes for dtype in dtypes
    ]

    print(f"\n{'=' * 60}")
    print(f"Verification: {op_name.upper()}")
    print(f"{'=' * 60}")
    print(f"Operator: {op_func.__name__}")
    print(f"Input shapes: {input_shapes}")
    print(f"Dtypes: {dtypes}")
    print(f"Tolerances: rtol={rtol}, atol={atol}")
    print(f"{'-' * 60}")

    # 执行验证
    try:
        passed = verify_operator(
            op_func, inputs_spec, rtol=rtol, atol=atol, verbose=verbose
        )
        return passed
    except Exception as e:
        print(f"\n✗ Verification failed with error: {e}")
        return False


def verify_multiple_operators(
    operators: List[str],
    input_shapes: List[tuple],
    dtypes: List[torch.dtype],
    rtol: float = 1e-4,
    atol: float = 1e-5,
    verbose: bool = True,
) -> Dict[str, bool]:
    """
    验证多个算子

    Args:
        operators: 算子名称列表
        input_shapes: 输入形状列表
        dtypes: 数据类型列表
        rtol: 相对误差容差
        atol: 绝对误差容差
        verbose: 是否打印详细信息

    Returns:
        Dict[str, bool]: 算子名称到验证结果的映射
    """
    results = {}

    for op_name in operators:
        passed = verify_with_config(op_name, input_shapes, dtypes, rtol, atol, verbose)
        results[op_name] = passed

    return results


# ============================================
# 主程序入口
# ============================================


def main():
    parser = argparse.ArgumentParser(
        description="PyTorch NPU Operator Verification Tool",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Verify add operator with single shape and dtype
  python3 verify_operator_accuracy.py --op add --shapes 2x3x4 --dtypes fp32

  # Verify matmul operator with multiple shapes and dtypes
  python3 verify_operator_accuracy.py --op matmul --shapes 4x5x5 --dtypes fp16,bf16

  # Verify multiple operators
  python3 verify_operator_accuracy.py --op add,mul,div --shapes 3x4 --dtypes fp32

  # Verify conv2d operator with custom tolerances
  python3 verify_operator_accuracy.py --op conv2d --shapes 1x3x8x8,16x3x3x3 --dtypes fp32 --rtol 1e-4 --atol 1e-5
        """,
    )

    # 必需参数
    parser.add_argument(
        "--op",
        type=str,
        required=True,
        help="Operator name(s) to verify. Use comma to separate multiple operators. "
        f"Supported: {', '.join(OPERATORS.keys())}",
    )

    parser.add_argument(
        "--shapes",
        type=str,
        required=True,
        help="Input shape(s) to verify. Use comma to separate multiple shapes. "
        'Format: "shape1,shape2" or "2x3x4,4x5x6"',
    )

    parser.add_argument(
        "--dtypes",
        type=str,
        required=True,
        help="Data type(s) to verify. Use comma to separate multiple dtypes. "
        f"Supported: {', '.join(DTYPES.keys())}",
    )

    # 可选参数
    parser.add_argument(
        "--rtol",
        type=float,
        default=1e-4,
        help="Relative error tolerance (default: 1e-4)",
    )

    parser.add_argument(
        "--atol",
        type=float,
        default=1e-5,
        help="Absolute error tolerance (default: 1e-5)",
    )

    parser.add_argument("--verbose", action="store_true", help="Enable verbose output")

    args = parser.parse_args()

    # 解析参数
    operators = [op.strip().lower() for op in args.op.split(",")]
    input_shapes = parse_shapes_list(args.shapes)
    dtypes = parse_dtypes_list(args.dtypes)

    # 验证参数
    if len(input_shapes) < 1:
        raise ValueError("At least one input shape is required")

    if len(dtypes) < 1:
        raise ValueError("At least one dtype is required")

    # 检查 NPU 可用性
    if not torch.npu.is_available():
        print(
            "Error: NPU is not available. Please check if NPU is connected and drivers are installed."
        )
        sys.exit(1)

    print(f"{'=' * 60}")
    print("PyTorch NPU Operator Verification Tool")
    print(f"{'=' * 60}")
    print(f"PyTorch version: {torch.__version__}")
    print(f"NPU available: {torch.npu.is_available()}")
    print(f"NPU count: {torch.npu.device_count()}")
    print(f"{'-' * 60}")

    # 执行验证
    if len(operators) > 1:
        # 验证多个算子
        results = verify_multiple_operators(
            operators, input_shapes, dtypes, args.rtol, args.atol, args.verbose
        )

        # 打印汇总结果
        print(f"\n{'=' * 60}")
        print("Summary")
        print(f"{'=' * 60}")
        for op_name, passed in results.items():
            status = "✓ PASS" if passed else "✗ FAIL"
            print(f"  {op_name}: {status}")

        all_passed = all(passed for passed in results.values())
        print(f"\nOverall: {'✓ ALL PASSED' if all_passed else '✗ SOME FAILED'}")
        sys.exit(0 if all_passed else 1)
    else:
        # 验证单个算子
        passed = verify_with_config(
            operators[0], input_shapes, dtypes, args.rtol, args.atol, args.verbose
        )
        sys.exit(0 if passed else 1)


if __name__ == "__main__":
    main()
