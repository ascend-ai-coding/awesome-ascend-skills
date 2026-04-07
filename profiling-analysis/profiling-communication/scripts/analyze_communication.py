#!/usr/bin/env python3
import os
import pandas as pd
import argparse

# 理论带宽配置 (Gbps)
THEORETICAL_BANDWIDTH = {
    "A2": 200,    # 单卡间带宽
    "A3": 400,    # 单卡间带宽
    "8*A2": 640,  # 8卡集群总带宽
    "8*A3": 1280  # 8卡集群总带宽
}

# MOE模型通信算子
MOE_OPS = ["hcom_reduceScatter_", "hcom_allGather_"]

# 常规模型通信算子
REGULAR_OPS = ["allreduce", "allgather", "broadcast", "reduce", "scatter", "gather"]

# 通用通信算子
COMMON_OPS = ["send", "recv", "barrier"]


def analyze_communication(input_folder, detail=False, model_type=None):
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
    communication_ops_stats = []  # 用于存储详细的通信算子统计

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
            analyze_communication_csv(df, file_path, file_results, communication_ops_stats, detail)
        elif "op_trace" in file_path.lower():
            # 分析op_trace.csv文件中的通信操作
            analyze_op_trace(df, file_path, file_results, communication_ops_stats, detail, model_type)

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

    # 4. 如果需要详细分析，展示通信算子统计
    if detail and communication_ops_stats:
        print("\n" + "=" * 100)
        print("🔍 通信算子详细分析")
        print("=" * 100)
        
        # 合并所有文件的通信算子统计
        ops_summary = {}
        for stats in communication_ops_stats:
            for op_name, op_stats in stats.items():
                if op_name not in ops_summary:
                    ops_summary[op_name] = {
                        "count": 0,
                        "total_duration": 0,
                        "max_duration": 0,
                        "model_type": op_stats.get("model_type", "unknown")
                    }
                ops_summary[op_name]["count"] += op_stats["count"]
                ops_summary[op_name]["total_duration"] += op_stats["total_duration"]
                if op_stats["max_duration"] > ops_summary[op_name]["max_duration"]:
                    ops_summary[op_name]["max_duration"] = op_stats["max_duration"]
        
        # 打印通信算子统计
        print(f"{'算子类型':<30} {'调用次数':<10} {'总耗时(ms)':<15} {'平均耗时(ms)':<15} {'最大耗时(ms)':<15} {'模型类型':<10}")
        print("-" * 100)
        
        for op_name, stats in sorted(ops_summary.items(), key=lambda x: x[1]["total_duration"], reverse=True):
            avg_duration = round(stats["total_duration"] / stats["count"], 2) if stats["count"] > 0 else 0
            print(f"{op_name:<30} {stats['count']:<10} {stats['total_duration']:<15.2f} {avg_duration:<15.2f} {stats['max_duration']:<15.2f} {stats['model_type']:<10}")
        
        # 带宽分析
        print("\n" + "-" * 100)
        print("📡 带宽分析")
        print("-" * 100)
        
        # 计算总通信数据量（假设每个通信操作的数据量，实际需要从profiling数据中获取）
        # 这里使用估算值作为示例
        estimated_data_size = 0  # bytes
        for op_name, stats in ops_summary.items():
            if "reduceScatter" in op_name or "allGather" in op_name:
                # MOE算子假设数据量
                estimated_data_size += stats["count"] * 1024 * 1024 * 16  # 每次16MB
            elif "allreduce" in op_name:
                # Allreduce假设数据量
                estimated_data_size += stats["count"] * 1024 * 1024 * 32  # 每次32MB
            else:
                # 其他算子假设数据量
                estimated_data_size += stats["count"] * 1024 * 1024 * 8  # 每次8MB
        
        total_comm_time = sum(stats["total_duration"] for stats in ops_summary.values()) / 1000  # 转换为秒
        
        if total_comm_time > 0:
            actual_bandwidth = (estimated_data_size * 8) / (total_comm_time * 10**9)  # Gbps
            print(f"估算总通信数据量：{estimated_data_size / (1024*1024):.2f} MB")
            print(f"总通信时间：{total_comm_time:.2f} 秒")
            print(f"实际通信带宽：{actual_bandwidth:.2f} Gbps")
            
            # 与理论带宽对比
            for device, theoretical in THEORETICAL_BANDWIDTH.items():
                utilization = (actual_bandwidth / theoretical) * 100
                print(f"{device}理论带宽利用率：{utilization:.2f}%")

    # 5. 优化建议
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


def analyze_communication_csv(df, file_path, file_results, communication_ops_stats, detail=False):
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

    # 详细分析：如果有算子类型信息，进行更详细的统计
    if detail and "op_type" in df.columns:
        ops_stats = {}
        for _, row in df.iterrows():
            op_type = row["op_type"]
            duration = float(row.get("duration", 0))
            
            if op_type not in ops_stats:
                ops_stats[op_type] = {
                    "count": 0,
                    "total_duration": 0,
                    "max_duration": 0
                }
            
            ops_stats[op_type]["count"] += 1
            ops_stats[op_type]["total_duration"] += duration
            if duration > ops_stats[op_type]["max_duration"]:
                ops_stats[op_type]["max_duration"] = duration
        
        communication_ops_stats.append(ops_stats)


def analyze_op_trace(df, file_path, file_results, communication_ops_stats, detail=False, model_type=None):
    """分析op_trace.csv文件中的通信操作"""
    # 查找通信相关的操作
    if "op_type" in df.columns:
        # 根据模型类型筛选通信算子
        if model_type == "moe":
            pattern = "|" .join(MOE_OPS + COMMON_OPS)
        elif model_type == "regular":
            pattern = "|" .join(REGULAR_OPS + COMMON_OPS)
        else:
            # 默认匹配所有通信算子
            pattern = "comm|send|recv|broadcast|reduce|allgather|scatter|gather|barrier|hcom_reduceScatter_|hcom_allGather_"
        
        # 筛选通信相关的操作
        comm_ops = df[df["op_type"].str.contains(pattern, case=False, na=False)]
        
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
            
            # 详细分析：统计不同类型的通信算子
            if detail:
                ops_stats = {}
                for _, row in comm_ops.iterrows():
                    op_type = row["op_type"]
                    duration = float(row.get("duration", 0))
                    
                    # 确定算子所属模型类型
                    op_model_type = "unknown"
                    if any(moe_op in op_type for moe_op in MOE_OPS):
                        op_model_type = "moe"
                    elif any(reg_op in op_type for reg_op in REGULAR_OPS):
                        op_model_type = "regular"
                    
                    if op_type not in ops_stats:
                        ops_stats[op_type] = {
                            "count": 0,
                            "total_duration": 0,
                            "max_duration": 0,
                            "model_type": op_model_type
                        }
                    
                    ops_stats[op_type]["count"] += 1
                    ops_stats[op_type]["total_duration"] += duration
                    if duration > ops_stats[op_type]["max_duration"]:
                        ops_stats[op_type]["max_duration"] = duration
                
                communication_ops_stats.append(ops_stats)
                
                # 打印当前文件的通信算子统计
                print(f"   {'算子类型':<30} {'调用次数':<10} {'总耗时(ms)':<15} {'平均耗时(ms)':<15}")
                print(f"   {'-' * 70}")
                
                for op_name, stats in sorted(ops_stats.items(), key=lambda x: x[1]["total_duration"], reverse=True):
                    avg_duration = round(stats["total_duration"] / stats["count"], 2) if stats["count"] > 0 else 0
                    print(f"   {op_name:<30} {stats['count']:<10} {stats['total_duration']:<15.2f} {avg_duration:<15.2f}")


# ==================== 运行入口 ====================
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="通信瓶颈分析工具")
    parser.add_argument("--input", type=str, required=True, help="输入profiling数据文件夹路径")
    parser.add_argument("--detail", action="store_true", help="显示详细的通信算子分析")
    parser.add_argument("--model-type", type=str, choices=["moe", "regular"], help="指定模型类型（moe或regular）")
    
    args = parser.parse_args()
    analyze_communication(args.input, args.detail, args.model_type)