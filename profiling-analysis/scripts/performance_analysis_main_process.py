import os
import pandas as pd


def analyze_performance(input_folder):
    """
    从指定文件夹下查找所有 step_trace_time.csv，分析耗时占比，并判断瓶颈类型
    新增：展示 free / computing / communication 各自占比最高的文件
    """
    # 1. 查找所有 step_trace_time.csv 文件（递归查找）
    csv_files = []
    for root, dirs, files in os.walk(input_folder):
        for file in files:
            if file == "step_trace_time.csv":
                csv_files.append(os.path.join(root, file))

    if not csv_files:
        print("❌ 未找到任何 step_trace_time.csv 文件")
        return

    print(f"✅ 找到 {len(csv_files)} 个目标文件")
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

        # 检查必须字段
        required_cols = ["Computing", "Communication(Not Overlapped)", "Free"]
        missing = [c for c in required_cols if c not in df.columns]
        if missing:
            print(f"⚠️  缺失字段：{missing}，跳过")
            continue

        # 计算总和
        compute_sum = df["Computing"].sum()
        comm_sum = df["Communication(Not Overlapped)"].sum()
        free_sum = df["Free"].sum()
        total = compute_sum + comm_sum + free_sum

        if total <= 0:
            print("⚠️  总耗时为0，跳过")
            continue

        # 计算占比
        compute_ratio = round(compute_sum / total * 100, 2)
        comm_ratio = round(comm_sum / total * 100, 2)
        free_ratio = round(free_sum / total * 100, 2)

        print(f"计算占比：{compute_ratio}%")
        print(f"通信占比：{comm_ratio}%")
        print(f"空闲占比：{free_ratio}%")

        # 保存结果
        file_results.append({
            "path": file_path,
            "computing": compute_ratio,
            "communication": comm_ratio,
            "free": free_ratio
        })

        print("-" * 100)

    if not file_results:
        print("❌ 没有可分析的有效数据")
        return

    # ==================== 计算平均值 ====================
    avg_compute = round(sum([f["computing"] for f in file_results]) / len(file_results), 2)
    avg_comm = round(sum([f["communication"] for f in file_results]) / len(file_results), 2)
    avg_free = round(sum([f["free"] for f in file_results]) / len(file_results), 2)

    # ==================== 找出各项最大值（新增功能） ====================
    max_free = max(file_results, key=lambda x: x["free"])
    max_compute = max(file_results, key=lambda x: x["computing"])
    max_comm = max(file_results, key=lambda x: x["communication"])

    # ==================== 瓶颈判定 ====================
    conclusion = ""
    sub_skill = ""
    sub_skill_path = ""

    if avg_free > 20:
        conclusion = "空闲占比超过20% → 判定为【下发问题】"
        sub_skill = "/profiling-analysis-hostbound"
        # sub_skill_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "profiling-analysis-hostbound", "scripts", "hostbound_analysis.py")
    elif avg_compute > 85:
        conclusion = "计算占比超过85% → 判定为【计算问题】"
        sub_skill = "/profiling-analysis-computing"
        sub_skill_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "profiling-analysis-computing", "scripts", "op_perf_analysis_combine.py")
    elif avg_comm > 10:
        conclusion = "通信占比超过10% → 判定为【通信问题】"
        sub_skill = "/profiling-analysis-communication"
        # sub_skill_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "profiling-analysis-communication", "scripts", "communication_analysis.py")
    else:
        conclusion = "无明显性能瓶颈，系统运行正常"
        sub_skill = "无"

    # ==================== 最终打屏展示 ====================
    print("\n" + "=" * 100)
    print("📊 整体分析结论")
    print("=" * 100)
    print(f"平均计算耗时：{avg_compute}%")
    print(f"平均通信耗时：{avg_comm}%")
    print(f"平均空闲耗时：{avg_free}%")

    print("\n🔥 各项耗时占比 最高文件（新增）")
    print("-" * 80)
    print(f"【空闲 Free 最高】{max_free['free']}% → {max_free['path']}")
    print(f"【计算 Computing 最高】{max_compute['computing']}% → {max_compute['path']}")
    print(f"【通信 Communication 最高】{max_comm['communication']}% → {max_comm['path']}")

    print(f"\n结论：{conclusion}")
    print(f"请调用子Skill进行深入分析：{sub_skill}")
    print("=" * 100)

    # ==================== 自动调用子技能 ====================
    if sub_skill != "无" and os.path.exists(sub_skill_path):
        print(f"\n🤖 正在自动调用子技能进行深入分析...")
        print(f"📁 调用子技能：{sub_skill}")
        print(f"📄 执行脚本：{sub_skill_path}")
        print("-" * 100)
        
        import subprocess
        
        try:
            # 构建命令行参数
            cmd = ["python", sub_skill_path, "--input-path", input_folder]
            
            # 执行子脚本并实时输出结果
            process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
            
            for line in process.stdout:
                print(line.strip())
            
            # 等待进程结束并获取返回值
            process.wait()
            
            if process.returncode == 0:
                print(f"\n✅ 子技能调用成功")
            else:
                print(f"\n❌ 子技能调用失败，返回码：{process.returncode}")
                
        except Exception as e:
            print(f"\n❌ 调用子技能时发生错误：{e}")
            print(f"💡 请尝试手动调用子技能：{sub_skill}")
    elif sub_skill != "无":
        print(f"\n⚠️  子技能脚本不存在：{sub_skill_path}")
        print(f"💡 请尝试手动调用子技能：{sub_skill}")


# ==================== 运行入口 ====================
if __name__ == "__main__":
    import argparse
    
    # 创建命令行参数解析器
    parser = argparse.ArgumentParser(description='昇腾NPU Profiling性能分析主脚本')
    parser.add_argument('--input-path', required=True, help='Profiling数据文件夹路径')
    
    # 解析命令行参数
    args = parser.parse_args()
    
    # 调用分析函数
    analyze_performance(args.input_path)