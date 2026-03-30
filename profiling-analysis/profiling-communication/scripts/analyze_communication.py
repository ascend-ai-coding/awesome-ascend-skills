#!/usr/bin/env python3
import os
import pandas as pd
import argparse


def analyze_communication(input_folder):
    """
    分析通信瓶颈：
    1. 查找所有相关的profiling数据文件
    2. 分析通信耗时的分布和原因
    3. 识别主要通信操作类型及其耗时占比
    4. 分析通信热点和潜在瓶颈
    5. 提供优化建议
    """
    # 1. 查找所有相关的profiling数据文件
    csv_files = []
    for root, dirs, files in os.walk(input_folder):
        for file in files:
            # 查找与通信相关的csv文件
            if (file.endswith(".csv") and 
                ("communication" in file.lower() or 
                 "step_trace" in file.lower() or 
                 "op_trace" in file.lower())):
                csv_files.append(os.path.join(root, file))

    if not csv_files:
        print("❌ 未找到任何与通信相关的CSV文件")
        return

    print(f"✅ 找到 {len(csv_files)} 个通信相关文件")
    print("-" * 100)

    # 存储所有文件的分析结果
    file_results = []

    # 2. 逐个分析文件
    for file_path in csv_files:
        print(f"正在分析：{file_path}")

        try:
            df = pd.read_csv(file_path)
        except Exception as e:
            print(f"⚠️  读取失败：{e}")
            continue

        print(f"   包含字段：{list(df.columns)}")

        # 根据不同的文件类型进行分析
        if "step_trace" in file_path.lower():
            # 分析step_trace_time.csv文件
            analyze_step_trace(df, file_path, file_results)
        elif "communication" in file_path.lower():
            # 分析通信专门的csv文件
            analyze_communication_csv(df, file_path, file_results)
        elif "op_trace" in file_path.lower():
            # 分析op_trace.csv文件中的通信操作
            analyze_op_trace(df, file_path, file_results)

        print("-" * 100)

    if not file_results:
        print("❌ 没有可分析的有效通信数据")
        return

    # 3. 整体分析结论
    print("\n" + "=" * 100)
    print("📊 通信瓶颈分析结论")
    print("=" * 100)

    # 计算平均通信占比
    avg_comm_ratios = [result["comm_ratio"] for result in file_results if "comm_ratio" in result]
    if avg_comm_ratios:
        avg_comm = round(sum(avg_comm_ratios) / len(avg_comm_ratios), 2)
        print(f"平均通信耗时占比：{avg_comm}%")

    # 找出通信耗时最高的文件
    if file_results:
        max_comm_file = max(file_results, key=lambda x: x.get("comm_ratio", 0))
        print(f"通信耗时最高文件：{max_comm_file['path']} ({max_comm_file.get('comm_ratio', 0)}%)")

    # 4. 优化建议
    print("\n💡 优化建议")
    print("-" * 80)
    print("1. 增加通信与计算的重叠")
    print("2. 优化数据传输格式和大小")
    print("3. 使用更高效的通信算法")
    print("4. 调整并行度和批处理大小")
    print("5. 考虑使用异步通信机制")
    print("6. 检查网络带宽和延迟")
    print("=" * 100)


def analyze_step_trace(df, file_path, file_results):
    """分析step_trace_time.csv文件"""
    # 检查必须字段
    required_cols = ["Computing", "Communication(Not Overlapped)", "Free"]
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        print(f"   ⚠️  缺失字段：{missing}，跳过")
        return

    # 计算总和
    compute_sum = df["Computing"].sum()
    comm_sum = df["Communication(Not Overlapped)"].sum()
    free_sum = df["Free"].sum()
    total = compute_sum + comm_sum + free_sum

    if total <= 0:
        print("   ⚠️  总耗时为0，跳过")
        return

    # 计算占比
    compute_ratio = round(compute_sum / total * 100, 2)
    comm_ratio = round(comm_sum / total * 100, 2)
    free_ratio = round(free_sum / total * 100, 2)

    print(f"   计算占比：{compute_ratio}%")
    print(f"   通信占比：{comm_ratio}%")
    print(f"   空闲占比：{free_ratio}%")

    # 保存结果
    file_results.append({
        "path": file_path,
        "file_type": "step_trace",
        "computing": compute_ratio,
        "comm_ratio": comm_ratio,
        "free": free_ratio
    })


def analyze_communication_csv(df, file_path, file_results):
    """分析通信专门的csv文件"""
    # 检查是否有时间相关字段
    time_cols = [col for col in df.columns if "time" in col.lower() or "duration" in col.lower()]
    if not time_cols:
        print(f"   ⚠️  未找到时间相关字段，跳过")
        return

    print(f"   时间相关字段：{time_cols}")

    # 计算总通信时间
    total_comm_time = 0
    for col in time_cols:
        try:
            total_comm_time += df[col].sum()
        except (TypeError, ValueError):
            continue

    print(f"   总通信时间：{total_comm_time:.2f} ms")

    # 保存结果
    file_results.append({
        "path": file_path,
        "file_type": "communication_csv",
        "total_comm_time": total_comm_time
    })


def analyze_op_trace(df, file_path, file_results):
    """分析op_trace.csv文件中的通信操作"""
    # 查找通信相关的操作
    if "op_type" in df.columns:
        # 筛选通信相关的操作
        comm_ops = df[df["op_type"].str.contains("comm|send|recv|broadcast|reduce", case=False, na=False)]
        if not comm_ops.empty:
            print(f"   发现 {len(comm_ops)} 个通信操作")
            
            # 计算通信操作总耗时
            if "duration" in df.columns:
                total_comm_time = comm_ops["duration"].sum()
                print(f"   通信操作总耗时：{total_comm_time:.2f} ms")
                
                # 保存结果
                file_results.append({
                    "path": file_path,
                    "file_type": "op_trace",
                    "comm_ops_count": len(comm_ops),
                    "total_comm_time": total_comm_time
                })


# ==================== 运行入口 ====================
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="通信瓶颈分析工具")
    parser.add_argument("--input", type=str, required=True, help="输入profiling数据文件夹路径")
    
    args = parser.parse_args()
    analyze_communication(args.input)