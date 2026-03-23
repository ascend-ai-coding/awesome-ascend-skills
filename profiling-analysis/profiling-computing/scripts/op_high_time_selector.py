#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
高耗时算子筛选脚本

根据输入的op_statistic_*.csv、op_summary_*.csv或kernel_details.csv文件，筛选出耗时最高的N个算子。
功能：
1. 读取op_statistic_*.csv、op_summary_*.csv或kernel_details.csv文件
2. 筛选高耗时算子：
   - 若使用op_statistic_*.csv，取Ratio(%)列前N个算子
   - 若使用op_summary_*.csv或kernel_details.csv，取总Task Duration前N个算子
3. 输出高耗时算子列表和相关统计信息

用法：
    python op_high_time_selector.py --input-path <input_path> --output-path <output_dir> --top-n <number>
    
参数：
    --input-path - profiling文件路径，包含PROF_*目录的根路径
    --output-path - 输出结果目录，用于保存生成的算子列表
    --top-n - 选取的高耗时算子数量，默认3
"""

import pandas as pd
import os
import glob
import sys
import argparse
import json

def select_high_time_ops(profiling_path, top_n=3):
    """
    筛选高耗时算子
    
    参数：
        profiling_path: profiling文件路径
        top_n: 选取的高耗时算子数量
        
    返回：
        top_ops: 高耗时算子列表
        op_stats: 算子统计信息
    """
    # 查找op_statistic_*.csv文件
    statistic_search_pattern = os.path.join(profiling_path, "PROF_*", "mindstudio_profiler_output", "op_statistic_*.csv")
    print(f"搜索op_statistic_*.csv模式: {statistic_search_pattern}")
    op_statistic_files = glob.glob(statistic_search_pattern)
    
    # 查找op_summary_*.csv文件
    summary_search_pattern = os.path.join(profiling_path, "PROF_*", "mindstudio_profiler_output", "op_summary_*.csv")
    print(f"搜索op_summary_*.csv模式: {summary_search_pattern}")
    op_summary_files = glob.glob(summary_search_pattern)
    
    # 查找kernel_details.csv文件
    kernel_search_pattern = os.path.join(profiling_path, "PROF_*", "mindstudio_profiler_output", "kernel_details.csv")
    print(f"搜索kernel_details.csv模式: {kernel_search_pattern}")
    kernel_details_files = glob.glob(kernel_search_pattern)
    
    # 确定使用哪种文件类型
    if op_statistic_files:
        print(f"找到{len(op_statistic_files)}个op_statistic文件")
        # 读取所有op_statistic文件
        dfs = []
        for file in op_statistic_files:
            print(f"读取文件: {file}")
            df = pd.read_csv(file)
            dfs.append(df)
        df = pd.concat(dfs, ignore_index=True)
        
        print(f"总数据行数: {len(df)}")
        
        # 确保必要的列存在
        if "OP Type" not in df.columns or "Ratio(%)" not in df.columns:
            print("op_statistic_*.csv文件缺少必要的列: OP Type 或 Ratio(%)")
            return None, None
        
        print("所有必要列都存在")
        
        # 根据Ratio(%)列选取前N个算子
        print(f"根据Ratio(%)列选取前{top_n}个高耗时算子...")
        op_statistic = df.sort_values(by="Ratio(%)", ascending=False)
        top_ops = op_statistic.head(top_n)["OP Type"].tolist()
        print(f"Ratio(%)最高的{top_n}个算子: {top_ops}")
        
        # 收集统计信息
        op_stats = op_statistic.head(top_n).to_dict('records')
        
    elif op_summary_files:
        print(f"未找到op_statistic_*.csv文件，使用op_summary_*.csv文件")
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
        if "OP Type" not in df.columns or "Task Duration(us)" not in df.columns:
            print("op_summary_*.csv文件缺少必要的列: OP Type 或 Task Duration(us)")
            return None, None
        
        print("所有必要列都存在")
        
        # 统计各个算子的总耗时
        print(f"统计各个算子的总耗时，选取前{top_n}个...")
        op_total_duration = df.groupby("OP Type")["Task Duration(us)"].sum().sort_values(ascending=False)
        
        # 选取总耗时最高的前N个算子
        top_ops = op_total_duration.head(top_n).index.tolist()
        print(f"总耗时最高的{top_n}个算子: {top_ops}")
        
        # 收集统计信息
        op_stats = []
        for op_type, duration in op_total_duration.head(top_n).items():
            op_stats.append({
                "OP Type": op_type,
                "Total Duration(us)": duration
            })
    elif kernel_details_files:
        print(f"未找到op_statistic_*.csv或op_summary_*.csv文件，使用kernel_details.csv文件")
        print(f"找到{len(kernel_details_files)}个kernel_details文件")
        
        # 读取所有kernel_details文件
        dfs = []
        for file in kernel_details_files:
            print(f"读取文件: {file}")
            df = pd.read_csv(file)
            dfs.append(df)
        df = pd.concat(dfs, ignore_index=True)
        
        print(f"总数据行数: {len(df)}")
        
        # 确保必要的列存在
        if "OP Type" not in df.columns or "Task Duration(us)" not in df.columns:
            print("kernel_details.csv文件缺少必要的列: OP Type 或 Task Duration(us)")
            return None, None
        
        print("所有必要列都存在")
        
        # 统计各个算子的总耗时
        print(f"统计各个算子的总耗时，选取前{top_n}个...")
        op_total_duration = df.groupby("OP Type")["Task Duration(us)"].sum().sort_values(ascending=False)
        
        # 选取总耗时最高的前N个算子
        top_ops = op_total_duration.head(top_n).index.tolist()
        print(f"总耗时最高的{top_n}个算子: {top_ops}")
        
        # 收集统计信息
        op_stats = []
        for op_type, duration in op_total_duration.head(top_n).items():
            op_stats.append({
                "OP Type": op_type,
                "Total Duration(us)": duration
            })
    else:
        print("未找到op_statistic_*.csv、op_summary_*.csv或kernel_details.csv文件")
        return None, None
    
    return top_ops, op_stats

def main():
    """主函数"""
    # 解析命令行参数
    parser = argparse.ArgumentParser(description='高耗时算子筛选脚本')
    parser.add_argument('--input-path', required=True, help='profiling文件路径，包含PROF_*目录的根路径')
    parser.add_argument('--output-path', required=False, help='输出结果目录，用于保存生成的算子列表')
    parser.add_argument('--top-n', type=int, default=3, help='选取的高耗时算子数量，默认3')
    
    args = parser.parse_args()
    
    # 输入路径
    profiling_path = args.input_path
    
    # 确定输出目录
    if args.output_path:
        output_dir = args.output_path
    else:
        # 如果没有指定输出路径，则在输入路径下创建output文件夹
        output_dir = os.path.join(profiling_path, 'output')
    
    # 确保输出目录存在
    os.makedirs(output_dir, exist_ok=True)
    
    # 筛选高耗时算子
    top_ops, op_stats = select_high_time_ops(profiling_path, args.top_n)
    
    if not top_ops:
        print("筛选高耗时算子失败")
        return 1
    
    # 保存总览CSV文件
    if op_stats:
        op_stats_df = pd.DataFrame(op_stats)
        total_duration_file = os.path.join(output_dir, "op_total_duration.csv")
        op_stats_df.to_csv(total_duration_file, index=False)
        print(f"已生成总览文件: {total_duration_file}")
    
    print("高耗时算子筛选完成！")
    return 0

if __name__ == "__main__":
    exit(main())
