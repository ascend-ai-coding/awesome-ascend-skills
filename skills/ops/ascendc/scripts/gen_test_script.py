#!/usr/bin/env python3
"""
根据算子名称和参数，自动生成 mssanitizer 内存检测测试脚本。

使用方法:
    python3 gen_test_script.py \
        --operator gelu_custom \
        --fallback gelu \
        --dtypes float16 float32 \
        --output /path/to/<op>_mssanitizer_test.py

生成的测试脚本会:
  1. 打印环境信息
  2. 对多种 shape / dtype 组合调用自定义算子（不可用时自动 fallback）
  3. 输出 JSON 测试报告
"""

import argparse
import textwrap
from pathlib import Path

DTYPE_MAP = {
    "float16": "torch.float16",
    "float32": "torch.float32",
    "bfloat16": "torch.bfloat16",
}

SHAPES = [
    ("小尺寸", "(128,)"),
    ("中尺寸", "(1024,)"),
    ("大尺寸", "(4096,)"),
    ("非对齐", "(3,)"),
    ("非对齐", "(5,)"),
    ("2D", "(32, 512)"),
    ("2D", "(64, 768)"),
    ("3D", "(8, 16, 64)"),
]


def build_test_cases(dtypes: list[str]) -> str:
    lines = []
    idx = 0
    for label, shape in SHAPES:
        for dt in dtypes:
            torch_dt = DTYPE_MAP.get(dt, f"torch.{dt}")
            name = f"{label} {dt.upper()} {shape}"
            lines.append(f'        ("{name}", {shape}, {torch_dt}),')
            idx += 1
    return "\n".join(lines)


def generate(operator: str, fallback: str, dtypes: list[str]) -> str:
    test_cases = build_test_cases(dtypes)

    return textwrap.dedent(f'''\
        #!/usr/bin/env python3
        """
        {operator} mssanitizer 内存检测测试（自动生成）
        """

        import argparse
        import json
        import os
        from datetime import datetime

        import torch
        import torch_npu


        def setup_environment():
            os.environ["ASCEND_SLOG_PRINT_TO_STDOUT"] = "1"
            os.environ["ASCEND_GLOBAL_LOG_LEVEL"] = "3"

            print("=" * 80)
            print("{operator} mssanitizer 内存检测测试")
            print("=" * 80)
            print(f"PyTorch: {{torch.__version__}}")
            print(f"torch_npu: {{torch_npu.__version__}}")
            print(f"NPU 可用: {{torch.npu.is_available()}}")
            if torch.npu.is_available():
                print(f"NPU 设备数量: {{torch.npu.device_count()}}")
            print("=" * 80)


        def call_operator(x):
            """调用自定义算子；不可用时 fallback 到 torch.nn.functional"""
            if hasattr(torch.ops, "customize") and hasattr(torch.ops.customize, "{operator}"):
                return torch.ops.customize.{operator}(x)
            return torch.nn.functional.{fallback}(x)


        def test_basic_functionality():
            if not torch.npu.is_available():
                print("NPU 不可用，跳过测试")
                return []

            device = torch.device("npu:0")
            results = []

            test_cases = [
        {test_cases}
            ]

            for name, shape, dtype in test_cases:
                print(f"\\n测试: {{name}} - 形状: {{shape}} - 数据类型: {{dtype}}")
                try:
                    x = torch.randn(shape, dtype=dtype, device=device)
                    result = call_operator(x)
                    torch.npu.synchronize()
                    print(f"  ✓ 通过 - 输出形状: {{result.shape}}")
                    results.append({{"name": name, "shape": list(shape), "dtype": str(dtype), "status": "PASS"}})
                except Exception as exc:
                    print(f"  ✗ 失败 - 错误: {{exc}}")
                    results.append({{"name": name, "shape": list(shape), "dtype": str(dtype),
                                     "status": "FAIL", "error": str(exc)}})

            return results


        def generate_report(results, output_file):
            report = {{
                "timestamp": datetime.now().isoformat(),
                "test_type": "mssanitizer 内存检测测试",
                "operator": "{operator}",
                "total_tests": len(results),
                "passed": sum(1 for r in results if r["status"] == "PASS"),
                "failed": sum(1 for r in results if r["status"] == "FAIL"),
                "results": results,
            }}
            with open(output_file, "w", encoding="utf-8") as f:
                json.dump(report, f, indent=2, ensure_ascii=False, default=str)
            print(f"\\n报告已保存到: {{output_file}}")
            print(f"总计: {{report['total_tests']}}, 通过: {{report['passed']}}, 失败: {{report['failed']}}")


        if __name__ == "__main__":
            parser = argparse.ArgumentParser()
            parser.add_argument("--output", "-o", required=True, help="输出报告文件")
            args = parser.parse_args()

            setup_environment()
            items = test_basic_functionality()
            generate_report(items, args.output)
    ''')


def main():
    parser = argparse.ArgumentParser(description="生成 mssanitizer 测试脚本")
    parser.add_argument("--operator", required=True, help="算子名称，如 gelu_custom")
    parser.add_argument("--fallback", required=True,
                        help="torch.nn.functional 中的 fallback 函数名，如 gelu")
    parser.add_argument("--dtypes", nargs="+", default=["float16", "float32"],
                        help="要测试的数据类型列表")
    parser.add_argument("--output", "-o", required=True, help="输出脚本路径")
    args = parser.parse_args()

    script = generate(args.operator, args.fallback, args.dtypes)
    Path(args.output).write_text(script, encoding="utf-8")
    print(f"已生成测试脚本: {args.output}")


if __name__ == "__main__":
    main()
