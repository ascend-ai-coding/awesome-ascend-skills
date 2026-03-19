import os
import pandas as pd
import numpy as np
from typing import Dict, List, Union

class ProfilingPerformanceSkill:
    def __init__(self):
        self.skill_name = "profiling-performance-bottleneck-analysis"
        self.target_file = "step_trace_time.csv"  # 目标解析文件名
        self.required_cols = ["Computing", "Communication(Not Overlapped)", "Free"]  # 核心必填字段

    def find_csv_files(self, input_path: str) -> List[str]:
        """
        从输入路径中检索所有step_trace_time.csv文件
        :param input_path: 输入路径（文件/文件夹）
        :return: 目标文件路径列表
        """
        csv_files = []
        # 校验输入路径是否存在
        if not os.path.exists(input_path):
            raise FileNotFoundError(f"输入路径不存在：{input_path}")

        # 单文件场景
        if os.path.isfile(input_path):
            if os.path.basename(input_path) == self.target_file:
                csv_files.append(input_path)
        # 文件夹场景（递归检索）
        else:
            for root, _, files in os.walk(input_path):
                for file in files:
                    if file == self.target_file:
                        csv_files.append(os.path.join(root, file))

        # 无目标文件场景
        if not csv_files:
            raise FileNotFoundError(f"路径中未找到{self.target_file}文件：{input_path}")
        return csv_files

    def analyze_single_file(self, file_path: str) -> Dict[str, float]:
        """
        解析单个step_trace_time.csv文件，计算耗时占比
        :param file_path: 单个文件路径
        :return: 该文件的耗时占比（计算、通信、空闲）
        """
        # 读取CSV文件
        try:
            df = pd.read_csv(file_path)
        except Exception as e:
            raise RuntimeError(f"读取文件失败{file_path}：{str(e)}")

        # 校验核心字段
        missing_cols = [col for col in self.required_cols if col not in df.columns]
        if missing_cols:
            raise ValueError(f"文件缺失核心字段：{', '.join(missing_cols)}，路径：{file_path}")

        # 计算各字段总耗时
        computing_sum = df["Computing"].sum()
        communication_sum = df["Communication(Not Overlapped)"].sum()
        free_sum = df["Free"].sum()
        total_time = computing_sum + communication_sum + free_sum

        # 避免除零错误
        if total_time <= 0:
            return {"computing_ratio": 0.0, "communication_ratio": 0.0, "free_ratio": 0.0}

        # 计算占比并保留2位小数
        computing_ratio = round((computing_sum / total_time) * 100, 2)
        communication_ratio = round((communication_sum / total_time) * 100, 2)
        free_ratio = round((free_sum / total_time) * 100, 2)

        return {
            "computing_ratio": computing_ratio,
            "communication_ratio": communication_ratio,
            "free_ratio": free_ratio
        }

    def execute(self, input_path: str) -> Dict:
        """
        技能核心执行入口
        :param input_path: 输入路径（文件/文件夹）
        :return: 分析结果字典
        """
        try:
            # 步骤1：检索目标文件
            csv_files = self.find_csv_files(input_path)
            file_count = len(csv_files)

            # 步骤2：解析所有文件并收集耗时指标
            all_metrics = []
            for file in csv_files:
                single_metrics = self.analyze_single_file(file)
                all_metrics.append(single_metrics)

            # 步骤3：计算全局平均耗时占比
            avg_computing = round(np.mean([m["computing_ratio"] for m in all_metrics]), 2)
            avg_communication = round(np.mean([m["communication_ratio"] for m in all_metrics]), 2)
            avg_free = round(np.mean([m["free_ratio"] for m in all_metrics]), 2)
            avg_metrics = {
                "computing_ratio": avg_computing,
                "communication_ratio": avg_communication,
                "free_ratio": avg_free
            }

            # 步骤4：判定瓶颈类型与匹配后续技能
            bottleneck_type = "normal"
            next_skill = ""
            if avg_free > 20:
                bottleneck_type = "scheduling"
                next_skill = "Hostbound_skill.md"
            elif avg_computing > 85:
                bottleneck_type = "computing"
                next_skill = "Computing_skill.md"
            elif avg_communication > 10:
                bottleneck_type = "communication"
                next_skill = "Communication_skill.md"

            # 步骤5：生成分析结论
            analysis_msg = self._generate_analysis_message(avg_metrics, bottleneck_type)

            # 组装最终结果
            return {
                "skill_name": self.skill_name,
                "status": "success",
                "file_count": file_count,
                "metrics": avg_metrics,
                "bottleneck_type": bottleneck_type,
                "next_skill": next_skill,
                "message": analysis_msg
            }

        except Exception as e:
            # 异常捕获与返回
            return {
                "skill_name": self.skill_name,
                "status": "failed",
                "error": str(e),
                "bottleneck_type": "unknown",
                "next_skill": "",
                "message": f"分析失败：{str(e)}"
            }

    def _generate_analysis_message(self, metrics: Dict[str, float], bottleneck_type: str) -> str:
        """
        生成人类可读的分析结论描述
        :param metrics: 平均耗时占比
        :param bottleneck_type: 瓶颈类型
        :return: 分析结论字符串
        """
        base_msg = (f"性能分析结果：计算耗时占比={metrics['computing_ratio']}%，"
                    f"通信耗时占比={metrics['communication_ratio']}%，"
                    f"空闲耗时占比={metrics['free_ratio']}%。")

        if bottleneck_type == "scheduling":
            return f"{base_msg} 空闲耗时占比超过20%，判定为下发问题，请参考Hostbound_skill.md进行分析。"
        elif bottleneck_type == "computing":
            return f"{base_msg} 计算耗时占比超过85%，判定为计算/算子问题，请参考Computing_skill.md进行分析。"
        elif bottleneck_type == "communication":
            return f"{base_msg} 通信耗时占比超过10%，判定为通信问题，请参考Communication_skill.md进行分析。"
        else:
            return f"{base_msg} 未检测到明显性能瓶颈，系统运行正常。"

# 技能入口函数（兼容Agent框架）
def run_skill(input_params: Dict) -> Dict:
    """
    技能执行入口
    :param input_params: 输入参数字典（需包含input_path）
    :return: 技能执行结果
    """
    input_path = input_params.get("input_path")
    if not input_path:
        return {
            "skill_name": "profiling-performance-bottleneck-analysis",
            "status": "failed",
            "error": "缺失必填参数：input_path",
            "bottleneck_type": "unknown",
            "next_skill": "",
            "message": "分析失败：请传入有效的文件或文件夹路径"
        }

    # 初始化技能并执行
    skill = ProfilingPerformanceSkill()
    return skill.execute(input_path)