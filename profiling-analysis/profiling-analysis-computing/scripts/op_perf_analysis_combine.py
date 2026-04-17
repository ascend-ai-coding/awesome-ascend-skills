#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
算子性能分析整合脚本（完整版本）

实际功能由以下三个独立脚本实现：
1. op_high_time_selector.py - 高耗时算子筛选
2. op_pivot_table_analyzer.py - 数据透视表分析与瓶颈定位
3. extract_op_shapes.py - 算子形状解析（如MatMul系列）

用法：
    python op_perf_analysis_combine.py --input-path <input_path> --output-path <output_dir> --top-n <number>
    
参数：
    --input-path - profiling文件路径，包含PROF_*目录的根路径
    --output-path - 输出结果目录，用于保存生成的分析报告和CSV文件
    --top-n - 选取的高耗时算子数量，默认3
"""

import os
import sys
import argparse
import subprocess

def main():
    """主函数"""
    # 解析命令行参数
    parser = argparse.ArgumentParser(description='算子性能分析整合脚本')
    parser.add_argument('--input-path', required=True, help='profiling文件路径，包含PROF_*目录的根路径')
    parser.add_argument('--output-path', required=False, help='输出结果目录，用于保存生成的分析报告和CSV文件')
    parser.add_argument('--top-n', type=int, default=3, help='选取的高耗时算子数量，默认3')
    
    args = parser.parse_args()
    
    # 获取脚本目录
    script_dir = os.path.dirname(os.path.abspath(__file__))
    
    # 构建第一个脚本命令（高耗时算子筛选）
    selector_script = os.path.join(script_dir, "op_high_time_selector.py")
    selector_cmd = [
        sys.executable, selector_script,
        "--input-path", args.input_path,
        "--top-n", str(args.top_n)
    ]
    
    if args.output_path:
        selector_cmd.extend(["--output-path", args.output_path])
    
    # 执行第一个脚本
    print("=" * 50)
    print("执行高耗时算子筛选...")
    print("=" * 50)
    result1 = subprocess.run(selector_cmd, capture_output=True, text=True)
    print(result1.stdout)
    
    if result1.stderr:
        print("错误输出:", result1.stderr)
    
    if result1.returncode != 0:
        print("高耗时算子筛选失败")
        return 1
    
    # 确定输出目录
    if args.output_path:
        output_dir = args.output_path
    else:
        # 如果没有指定输出路径，则在输入路径下创建output文件夹
        output_dir = os.path.join(args.input_path, 'output')
    
    # 构建第二个脚本命令（数据透视表分析）
    analyzer_script = os.path.join(script_dir, "op_pivot_table_analyzer.py")
    analyzer_cmd = [
        sys.executable, analyzer_script,
        "--input-path", args.input_path,
        "--top-n", str(args.top_n)
    ]
    
    if args.output_path:
        analyzer_cmd.extend(["--output-path", args.output_path])
    
    # 执行第二个脚本
    print("\n" + "=" * 50)
    print("执行数据透视表分析与瓶颈定位...")
    print("=" * 50)
    result2 = subprocess.run(analyzer_cmd, capture_output=True, text=True)
    print(result2.stdout)
    
    if result2.stderr:
        print("错误输出:", result2.stderr)
    
    if result2.returncode != 0:
        print("数据透视表分析失败")
        return 1
    
    # 构建第三个脚本命令（算子形状解析）
    extract_script = os.path.join(script_dir, "extract_op_shapes.py")
    extract_output = os.path.join(output_dir, "matmul_shapes.csv")
    html_file = os.path.join(output_dir, "op_analysis_combined.html")
    
    extract_cmd = [
        sys.executable, extract_script,
        "--input", args.input_path,
        "--output", extract_output,
        "--op", "matmul"
    ]
    
    # 如果存在HTML文件，则添加HTML形状替换参数
    if os.path.exists(html_file):
        extract_cmd.extend(["--html-file", html_file])
    
    # 执行第三个脚本
    print("\n" + "=" * 50)
    print("执行算子形状解析...")
    print("=" * 50)
    result3 = subprocess.run(extract_cmd, capture_output=True, text=True)
    print(result3.stdout)
    
    if result3.stderr:
        print("错误输出:", result3.stderr)
    
    if result3.returncode != 0:
        print("算子形状解析失败，但不影响整体分析结果")
        # 形状解析失败不影响整体流程，继续执行
    
    print("\n" + "=" * 50)
    print("完整分析流程完成！")
    print("=" * 50)
    return 0

if __name__ == "__main__":
    exit(main())