#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
算子性能透视表生成脚本

根据SKILL.md文件中的分析步骤，对profiling文件创建算子透视表。
功能：
1. 读取op_summary_*.csv文件
2. 统计各个算子的总耗时
3. 选取总耗时最高的前3个算子
4. 为每个算子生成透视表，计算各列的平均值
5. 输出HTML格式的结果，标记最大值为红色
6. 生成总览CSV文件

用法：
    python op_analysis_final.py <profiling_path> <output_dir>
    
参数：
    profiling_path - profiling文件路径，例如：C:/Users/lan/Doc/profiling/p-perf-huawei-05_110439_20250728062428118_ascend_pt
    output_dir - 输出结果目录，例如：C:/Users/lan/Code/agent/performance_analysis/output
"""

import pandas as pd
import os
import glob
import sys

def main():
    """主函数"""
    # 检查命令行参数
    if len(sys.argv) != 3:
        print("用法: python op_analysis_final.py <profiling_path> <output_dir>")
        return 1
    
    # 输入路径
    profiling_path = sys.argv[1]
    output_dir = sys.argv[2]
    
    # 确保输出目录存在
    os.makedirs(output_dir, exist_ok=True)
    
    # 查找op_summary_*.csv文件
    search_pattern = os.path.join(profiling_path, "PROF_*", "mindstudio_profiler_output", "op_summary_*.csv")
    print(f"搜索模式: {search_pattern}")
    op_summary_files = glob.glob(search_pattern)
    
    print(f"找到的文件: {op_summary_files}")
    
    if not op_summary_files:
        print("未找到op_summary_*.csv文件")
        return 1
    
    print(f"找到{len(op_summary_files)}个op_summary文件")
    
    # 读取所有op_summary文件
    dfs = []
    for file in op_summary_files:
        print(f"读取文件: {file}")
        df = pd.read_csv(file)
        dfs.append(df)
    df = pd.concat(dfs, ignore_index=True)
    
    print(f"总数据行数: {len(df)}")
    
    # 确保必要的列存在
    required_columns = ["OP Type", "Task Duration(us)", "Input Shapes", 
                       "aic_mac_ratio", "aic_scalar_ratio", "aic_mte1_ratio", "aic_mte2_ratio", "aic_fixpipe_ratio",
                       "aiv_vec_ratio", "aiv_scalar_ratio", "aiv_mte2_ratio", "aiv_mte3_ratio"]
    
    for col in required_columns:
        if col not in df.columns:
            print(f"缺少必要的列: {col}")
            return 1
    
    print("所有必要列都存在")
    
    # 统计各个算子的总耗时
    print("统计各个算子的总耗时...")
    op_total_duration = df.groupby("OP Type")["Task Duration(us)"].sum().sort_values(ascending=False)
    
    # 选取总耗时最高的前3个算子
    top_ops = op_total_duration.head(3).index.tolist()
    print(f"总耗时最高的3个算子: {top_ops}")
    
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
    
    # 生成总览CSV文件
    total_duration_df = op_total_duration.reset_index()
    total_duration_df.columns = ["OP Type", "Total Duration(us)"]
    total_duration_file = os.path.join(output_dir, "op_total_duration.csv")
    total_duration_df.to_csv(total_duration_file, index=False)
    print(f"已生成总览文件: {total_duration_file}")
    
    print("分析完成！")
    return 0

if __name__ == "__main__":
    exit(main())
