#!/usr/bin/env python3
import os
import sys
import argparse
import subprocess
import zipfile
import sqlite3
import tempfile


def check_msprof_installed():
    """检查msprof-analyze工具是否安装"""
    try:
        result = subprocess.run(["msprof-analyze", "--help"],
                                capture_output=True, text=True)
        return result.returncode == 0
    except FileNotFoundError:
        return False


def install_msprof():
    """安装msprof-analyze工具"""
    print("正在安装msprof-analyze工具...")

    # 创建临时目录
    with tempfile.TemporaryDirectory() as temp_dir:
        try:
            # 安装wheel
            subprocess.run([sys.executable, "-m", "pip", "install", "wheel"],
                           check=True, capture_output=True)

            # 克隆源码
            subprocess.run(["git", "clone", "-b", "master",
                            "https://gitee.com/ascend/mstt.git"],
                           cwd=temp_dir, check=True)

            # 编译whl包
            msprof_dir = os.path.join(temp_dir, "mstt", "profiler", "msprof_analyze")
            subprocess.run([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"],
                           cwd=msprof_dir, check=True)
            subprocess.run([sys.executable, "setup.py", "bdist_wheel"],
                           cwd=msprof_dir, check=True)

            # 安装whl包
            dist_dir = os.path.join(msprof_dir, "dist")
            whl_files = [f for f in os.listdir(dist_dir) if f.endswith(".whl")]
            if not whl_files:
                print("❌ 编译失败，未生成whl文件")
                return False

            subprocess.run([sys.executable, "-m", "pip", "install",
                            os.path.join(dist_dir, whl_files[0])],
                           check=True)

            print("✅ msprof-analyze工具安装成功")
            return True

        except subprocess.CalledProcessError as e:
            print(f"❌ 安装失败：{e}")
            print(f"错误输出：{e.stderr.decode() if hasattr(e, 'stderr') else '无详细信息'}")
            return False


def extract_zip(zip_path, output_dir):
    """解压zip文件"""
    print(f"正在解压 {zip_path} 到 {output_dir}...")

    try:
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(output_dir)
        print(f"✅ 解压完成")
        return True
    except Exception as e:
        print(f"❌ 解压失败：{e}")
        return False


def run_slow_rank_detection(profiling_dir, output_dir):
    """运行快慢卡检测"""
    print(f"正在运行快慢卡检测...")
    print(f"输入目录：{profiling_dir}")
    print(f"输出目录：{output_dir}")

    try:
        result = subprocess.run(["msprof-analyze", "cluster",
                                 "-d", profiling_dir,
                                 "-m", "slow_rank",
                                 "-o", output_dir],
                                check=True, capture_output=True, text=True)

        print("✅ 快慢卡检测完成")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ 快慢卡检测失败：{e}")
        print(f"错误输出：{e.stderr}")
        return False


def get_actual_ranks(profiling_dir):
    """通过遍历device_x子文件夹获取实际的rank数量和ID列表"""
    rank_ids = set()
    
    # 遍历profiling目录下的所有子目录
    for root, dirs, files in os.walk(profiling_dir):
        for dir_name in dirs:
            # 查找device_开头的文件夹
            if dir_name.startswith("device_"):
                try:
                    # 提取device_后面的数字作为rankId
                    rank_id = int(dir_name.split("_")[1])
                    rank_ids.add(rank_id)
                except (IndexError, ValueError):
                    continue
    
    # 返回排序后的rankId列表
    return sorted(rank_ids)


def analyze_slow_rank_results(db_path, profiling_dir=None):
    """分析快慢卡检测结果"""
    print(f"\n📊 快慢卡检测结果分析")
    print("-" * 80)
    
    if not os.path.exists(db_path):
        print(f"❌ 结果文件不存在：{db_path}")
        return
    
    try:
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        
        # 查询所有rank的快慢卡影响次数
        cursor.execute("SELECT rankId, slowAffectCount FROM SlowRank")
        db_results = cursor.fetchall()
        
        # 将数据库结果转换为字典
        slow_counts_dict = {rank_id: count for rank_id, count in db_results}
        
        # 获取实际的rank列表
        if profiling_dir and os.path.exists(profiling_dir):
            actual_ranks = get_actual_ranks(profiling_dir)
            print(f"\n🔍 实际检测到的卡数：{len(actual_ranks)}")
            print(f"   rank列表：{actual_ranks}")
        else:
            # 如果没有提供profiling目录，使用数据库中的rank
            actual_ranks = sorted(slow_counts_dict.keys())
        
        # 合并实际rank和数据库结果，确保所有rank都有记录
        results = []
        for rank_id in actual_ranks:
            count = slow_counts_dict.get(rank_id, 0)
            results.append((rank_id, count))
        
        if not results:
            print("未发现快慢卡问题")
            return

        # 按快慢卡影响次数排序
        sorted_results = sorted(results, key=lambda x: x[1], reverse=True)

        print("各rank快慢卡影响次数统计：")
        print("rankId | slowAffectCount")
        print("-" * 25)
        for rank_id, count in sorted_results:
            print(f"{rank_id:<6} | {count:<15}")

        # 统计分析
        slow_counts = [count for _, count in results]
        avg_count = sum(slow_counts) / len(slow_counts)
        max_count = max(slow_counts)
        min_count = min(slow_counts)

        # 计算标准差
        import statistics
        if len(slow_counts) > 1:
            std_dev = statistics.stdev(slow_counts)
        else:
            std_dev = 0

        print(f"\n📈 统计分析：")
        print(f"平均影响次数：{avg_count:.2f}")
        print(f"最大影响次数：{max_count}")
        print(f"最小影响次数：{min_count}")
        print(f"标准差：{std_dev:.2f}")

        # 识别异常值（快慢卡标准）
        # 结合绝对次数和Z-score方法进行综合判断
        print(f"\n🔍 快慢卡检测（综合判定）：")
        print(f"- 平均影响次数：{avg_count:.2f}")
        print(f"- 标准差：{std_dev:.2f}")
        print(f"- 绝对次数阈值：")
        print(f"  * 轻微慢卡：1-5 次")
        print(f"  * 中度慢卡：6-20 次")
        print(f"  * 严重慢卡：>20 次")
        print(f"- Z-score阈值：")
        print(f"  * 轻微慢卡：>0.5")
        print(f"  * 中度慢卡：>1.5")
        print(f"  * 严重慢卡：>3.0")

        slow_ranks = []
        for rank_id, count in results:
            z_score = 0
            if std_dev > 0:
                z_score = (count - avg_count) / std_dev

            # 综合判定逻辑
            severity = "正常"
            if count == 0:
                severity = "正常"
            elif (count >= 1 and count <= 5) or (z_score > 0.5 and z_score <= 1.5):
                severity = "轻微慢卡"
            elif (count >= 6 and count <= 20) or (z_score > 1.5 and z_score <= 3):
                severity = "中度慢卡"
            elif count > 20 or z_score > 3:
                severity = "严重慢卡"

            if severity != "正常":
                slow_ranks.append((rank_id, count, z_score, severity))

        if slow_ranks:
            slow_ranks.sort(key=lambda x: (x[1], x[2]), reverse=True)
            print(f"\n🔥 检测到 {len(slow_ranks)} 个异常rank：")
            print("rankId | slowAffectCount | Z-score | 严重程度")
            print("-" * 50)
            for rank_id, count, z_score, severity in slow_ranks:
                print(f"{rank_id:<6} | {count:<15} | {z_score:>7.2f} | {severity:>8}")

            # 统计各严重程度的慢卡数量
            severity_counts = {}
            for _, _, _, severity in slow_ranks:
                severity_counts[severity] = severity_counts.get(severity, 0) + 1

            print(f"\n📊 慢卡严重程度分布：")
            for severity in ["轻微慢卡", "中度慢卡", "严重慢卡"]:
                if severity in severity_counts:
                    print(f"- {severity}: {severity_counts[severity]} 个")

            # 分析慢卡影响程度
            max_affected_rank = slow_ranks[0]
            print(f"\n💡 影响最严重的rank：")
            print(f"rankId: {max_affected_rank[0]}")
            print(f"影响次数: {max_affected_rank[1]}")
            print(f"Z-score: {max_affected_rank[2]:.2f}")
            print(f"严重程度: {max_affected_rank[3]}")
            if max_affected_rank[1] > avg_count and avg_count > 0:
                print(
                    f"超过平均值: {(max_affected_rank[1] - avg_count):.2f} ({((max_affected_rank[1] - avg_count) / avg_count * 100):.1f}%)")
        else:
            print("\n✅ 未检测到明显的快慢卡问题")

        # 提供优化建议
        print(f"\n💡 优化建议：")
        print("-" * 80)
        if slow_ranks:
            # 检查是否有严重慢卡
            has_severe = any(severity == "严重慢卡" for _, _, _, severity in slow_ranks)
            has_moderate = any(severity == "中度慢卡" for _, _, _, severity in slow_ranks)

            if has_severe:
                print("🔥 严重慢卡优化建议：")
                print("1. 立即检查慢卡的硬件状态（温度、风扇、内存、PCIe连接）")
                print("2. 考虑重启慢卡或更换硬件")
                print("3. 检查网络连接和带宽，确保网络稳定")
                print("4. 临时将该rank的任务迁移到其他节点")
                print("5. 分析应用代码，优化负载均衡策略")
            elif has_moderate:
                print("⚠️  中度慢卡优化建议：")
                print("1. 检查慢卡的资源使用情况（内存、CPU、GPU利用率）")
                print("2. 优化数据分布，确保各rank负载均衡")
                print("3. 调整通信策略，减少同步等待时间")
                print("4. 考虑使用异步通信机制")
                print("5. 检查是否有其他进程占用了慢卡资源")
            else:
                print("📋 轻微慢卡优化建议：")
                print("1. 持续监控慢卡的性能变化")
                print("2. 检查应用代码中的数据传输效率")
                print("3. 考虑微调通信参数")
                print("4. 确保所有节点的固件和驱动版本一致")
                print("5. 定期清理系统缓存和临时文件")

            # 通用建议
            print("\n📌 通用优化建议：")
            print("1. 使用npu-smi工具监控各卡的硬件状态")
            print("2. 调整应用的并行度和批处理大小")
            print("3. 优化内存访问模式，减少数据拷贝")
            print("4. 考虑使用混合并行策略")
            print("5. 定期更新固件和驱动版本")

        conn.close()

    except sqlite3.Error as e:
        print(f"❌ 分析结果失败：{e}")
    except ImportError:
        print(f"⚠️  无法进行高级统计分析（缺少statistics模块）")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="快慢卡检测工具")
    parser.add_argument("--input", type=str, required=True,
                        help="输入profiling数据路径（可以是zip文件或目录）")
    parser.add_argument("--output", type=str, default="./slow_rank_result",
                        help="输出结果目录")
    parser.add_argument("--install", action="store_true",
                        help="强制重新安装msprof-analyze工具")

    args = parser.parse_args()

    # 1. 检查或安装msprof-analyze工具
    if args.install or not check_msprof_installed():
        if not install_msprof():
            sys.exit(1)
    else:
        print("✅ msprof-analyze工具已安装")

    # 2. 处理输入数据
    profiling_dir = args.input
    temp_dir = None

    if args.input.endswith(".zip"):
        # 创建临时目录解压zip文件
        temp_dir = tempfile.mkdtemp()
        profiling_dir = os.path.join(temp_dir, "profiling_data")
        os.makedirs(profiling_dir, exist_ok=True)

        if not extract_zip(args.input, profiling_dir):
            sys.exit(1)

    # 3. 运行快慢卡检测
    if not run_slow_rank_detection(profiling_dir, args.output):
        sys.exit(1)

    # 4. 分析结果
    db_path = os.path.join(args.output, "cluster_analysis.db")
    
    # 检查默认路径，如果不存在，尝试在cluster_analysis_output子目录中查找
    if not os.path.exists(db_path):
        output_subdir = os.path.join(args.output, "cluster_analysis_output")
        db_path = os.path.join(output_subdir, "cluster_analysis.db")
        if os.path.exists(db_path):
            print(f"✅ 找到结果文件在子目录中：{db_path}")
    
    analyze_slow_rank_results(db_path, profiling_dir)

    # 5. 清理临时文件
    if temp_dir and os.path.exists(temp_dir):
        import shutil

        shutil.rmtree(temp_dir)

    print("\n" + "=" * 100)
    print(f"检测完成！结果保存至：{args.output}")
    print("=" * 100)
