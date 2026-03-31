#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
算子数据透视表分析脚本

根据op_total_duration.csv文件中的高耗时算子和op_summary_*.csv或kernel_details.csv文件，生成数据透视表。
功能：
1. 从op_total_duration.csv文件中读取高耗时算子列表
2. 读取op_summary_*.csv或kernel_details.csv文件
3. 为每个高耗时算子生成数据透视表
4. 输出HTML格式的分析报告

用法：
    python op_pivot_table_analyzer.py --input-path <input_path> --output-path <output_dir> --top-n <number>
    
参数：
    --input-path - profiling文件路径，包含PROF_*目录的根路径
    --output-path - 输出结果目录，用于保存分析报告
    --top-n - 选取的高耗时算子数量，默认3
"""

import pandas as pd
import os
import glob
import sys
import argparse
import json

def generate_op_pivot_tables(profiling_path, output_dir, top_n=3):
    """
    生成算子数据透视表
    
    参数：
        profiling_path: profiling文件路径
        output_dir: 输出结果目录
        top_n: 选取的高耗时算子数量
        
    返回：
        success: 是否生成成功
    """
    # 从op_total_duration.csv文件中读取高耗时算子列表
    total_duration_file = os.path.join(output_dir, "op_total_duration.csv")
    if not os.path.exists(total_duration_file):
        print(f"op_total_duration.csv文件不存在: {total_duration_file}")
        return False
    
    # 读取总览文件
    op_stats_df = pd.read_csv(total_duration_file)
    
    # 确保必要的列存在
    if "OP Type" not in op_stats_df.columns:
        print("op_total_duration.csv文件缺少必要的列: OP Type")
        return False
    
    # 获取前N个高耗时算子
    top_ops = op_stats_df["OP Type"].head(top_n).tolist()
    
    if not top_ops:
        print("高耗时算子列表为空")
        return False
    
    print(f"从op_total_duration.csv中读取到{len(top_ops)}个高耗时算子: {top_ops}")
    
    # 查找op_summary_*.csv文件
    summary_search_pattern = os.path.join(profiling_path, "PROF_*", "mindstudio_profiler_output", "op_summary_*.csv")
    print(f"搜索op_summary_*.csv模式: {summary_search_pattern}")
    op_summary_files = glob.glob(summary_search_pattern)
    
    # 查找kernel_details.csv文件
    kernel_search_pattern = os.path.join(profiling_path, "PROF_*", "mindstudio_profiler_output", "kernel_details.csv")
    print(f"搜索kernel_details.csv模式: {kernel_search_pattern}")
    kernel_details_files = glob.glob(kernel_search_pattern)
    
    # 确定使用哪种文件类型
    if op_summary_files:
        print(f"找到{len(op_summary_files)}个op_summary文件")
        # 读取所有op_summary文件
        dfs = []
        for file in op_summary_files:
            print(f"读取文件: {file}")
            df = pd.read_csv(file)
            dfs.append(df)
        df = pd.concat(dfs, ignore_index=True)
    elif kernel_details_files:
        print(f"未找到op_summary_*.csv文件，使用kernel_details.csv文件")
        print(f"找到{len(kernel_details_files)}个kernel_details文件")
        # 读取所有kernel_details文件
        dfs = []
        for file in kernel_details_files:
            print(f"读取文件: {file}")
            df = pd.read_csv(file)
            dfs.append(df)
        df = pd.concat(dfs, ignore_index=True)
    else:
        print("未找到op_summary_*.csv或kernel_details.csv文件")
        return False
    
    print(f"总数据行数: {len(df)}")
    
    # 确保必要的列存在
    required_columns = ["OP Type", "Task Duration(us)", "Input Shapes", 
                       "aic_mac_ratio", "aic_scalar_ratio", "aic_mte1_ratio", "aic_mte2_ratio", "aic_fixpipe_ratio",
                       "aiv_vec_ratio", "aiv_scalar_ratio", "aiv_mte2_ratio", "aiv_mte3_ratio"]
    
    for col in required_columns:
        if col not in df.columns:
            print(f"缺少必要的列: {col}")
            return False
    
    print("所有必要列都存在")
    
    # 初始化合并后的HTML输出
    combined_html_output = "<html><head><title>算子性能分析</title></head><body><h1>算子性能分析报告</h1><br><br>"
    
    # 初始化分析详情数据
    analysis_details = []
    
    # 为每个算子生成透视表
    for op in top_ops:
        print(f"处理算子: {op}")
        op_df = df[df["OP Type"] == op]
        
        # 按Input Shapes分组，计算各列的平均值
        pivot_df = op_df.groupby("Input Shapes").agg({
            "aic_mac_ratio": "mean",
            "aic_scalar_ratio": "mean",
            "aic_mte1_ratio": "mean",
            "aic_mte2_ratio": "mean",
            "aic_fixpipe_ratio": "mean",
            "aiv_vec_ratio": "mean",
            "aiv_scalar_ratio": "mean",
            "aiv_mte2_ratio": "mean",
            "aiv_mte3_ratio": "mean"
        }).reset_index()
        
        print(f"  该算子有{len(pivot_df)}种不同的Input Shapes")
        
        # 生成HTML输出，标记每行最大值为红色
        combined_html_output += f"<h2>算子: {op}</h2>"
        combined_html_output += "<table border='1'>"
        # 表头
        combined_html_output += "<tr>"
        combined_html_output += "<th>Input Shapes</th>"
        for col in pivot_df.columns[1:]:
            combined_html_output += f"<th>{col}</th>"
        combined_html_output += "</tr>"
        
        # 数据行
        for _, row in pivot_df.iterrows():
            combined_html_output += "<tr>"
            combined_html_output += f"<td>{row['Input Shapes']}</td>"
            # 找到当前行的最大值
            max_val = row[1:].max()
            max_col = row[1:].idxmax()
            
            # 保存分析详情
            detail_row = {"OP Type": op, "Input Shapes": row['Input Shapes']}
            for col in pivot_df.columns[1:]:
                detail_row[col] = row[col]
                if row[col] == max_val:
                    combined_html_output += f"<td style='color: red; font-weight: bold;'>{row[col]:.4f}</td>"
                else:
                    combined_html_output += f"<td>{row[col]:.4f}</td>"
            analysis_details.append(detail_row)
            combined_html_output += "</tr>"
        combined_html_output += "</table><br><br>"
    
    # 完成HTML输出
    combined_html_output += "</body></html>"
    
    # 保存合并后的HTML文件
    combined_html_file = os.path.join(output_dir, "op_analysis_combined.html")
    with open(combined_html_file, "w", encoding="utf-8") as f:
        f.write(combined_html_output)
    print(f"已生成合并后的HTML文件: {combined_html_file}")
    
    # 生成分析详情CSV文件
    if analysis_details:
        analysis_df = pd.DataFrame(analysis_details)
        analysis_csv_file = os.path.join(output_dir, "op_analysis_details.csv")
        analysis_df.to_csv(analysis_csv_file, index=False)
        print(f"已生成分析详情CSV文件: {analysis_csv_file}")
    
    return True

def main():
    """主函数"""
    # 解析命令行参数
    parser = argparse.ArgumentParser(description='算子数据透视表分析脚本')
    parser.add_argument('--input-path', required=True, help='profiling文件路径，包含PROF_*目录的根路径')
    parser.add_argument('--output-path', required=False, help='输出结果目录，用于保存生成的分析报告')
    parser.add_argument('--top-n', type=int, default=3, help='选取的高耗时算子数量，默认3')
    
    args = parser.parse_args()
    
    # 输入路径
    profiling_path = args.input_path
    top_n = args.top_n
    
    # 确定输出目录
    if args.output_path:
        output_dir = args.output_path
    else:
        # 如果没有指定输出路径，则在输入路径下创建output文件夹
        output_dir = os.path.join(profiling_path, 'output')
    
    # 确保输出目录存在
    os.makedirs(output_dir, exist_ok=True)
    
    # 生成算子数据透视表
    success = generate_op_pivot_tables(profiling_path, output_dir, top_n)
    
    if success:
        print("算子数据透视表生成完成！")
        return 0
    else:
        print("算子数据透视表生成失败！")
        return 1

if __name__ == "__main__":
    exit(main())
