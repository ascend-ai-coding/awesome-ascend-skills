#!/usr/bin/env python3
# coding: utf-8
# Copyright (c) 2025-2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""
PyPTO 性能分析脚本
从bubble_analysis.log中提取性能数据并计算性能指标
"""

import logging
import os
import re
import sys
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple


@dataclass
class CoreMetrics:
    """核心性能指标"""

    core_name: str
    task_num: int
    total_work_time: float
    total_wait_time: float
    wait_schedule_time: float
    wait_predecessor_time: float

    @property
    def aicore_time(self) -> float:
        """核心实际工作时间"""
        return self.total_work_time - self.total_wait_time

    @property
    def core_utilization(self) -> float:
        """核心利用率"""
        total_time = self.aicore_time + self.total_wait_time
        if total_time == 0:
            return 0.0
        return (self.aicore_time / total_time) * 100

    @property
    def bubble_rate(self) -> float:
        """气泡率"""
        total_time = self.aicore_time + self.wait_schedule_time
        if total_time == 0:
            return 0.0
        return (self.wait_schedule_time / total_time) * 100


def parse_bubble_analysis(log_path: str) -> List[CoreMetrics]:
    """解析bubble_analysis.log文件"""
    cores = []

    with open(log_path, "r") as f:
        content = f.read()

    # 匹配核心信息
    core_pattern = (
        r"\[(AIC_\d+|AIV_\d+)\] Execute task num:(\d+)\s+Core Total Work Time: ([\d.]+)\s+"
        r"Total Wait Time: ([\d.]+)\s+Wait Schedule Time: ([\d.]+)\s+"
        r"Wait Predecessor Time: ([\d.]+)"
    )

    matches = re.findall(core_pattern, content)

    for match in matches:
        core_name = match[0]
        task_num = int(match[1])
        total_work_time = float(match[2])
        total_wait_time = float(match[3])
        wait_schedule_time = float(match[4])
        wait_predecessor_time = float(match[5])

        core = CoreMetrics(
            core_name=core_name,
            task_num=task_num,
            total_work_time=total_work_time,
            total_wait_time=total_wait_time,
            wait_schedule_time=wait_schedule_time,
            wait_predecessor_time=wait_predecessor_time,
        )
        cores.append(core)

    return cores


def calculate_performance_metrics(cores: List[CoreMetrics]) -> Dict:
    """计算性能指标"""
    # 分离AIC和AIV核心
    aic_cores = [c for c in cores if c.core_name.startswith("AIC")]
    aiv_cores = [c for c in cores if c.core_name.startswith("AIV")]

    # 计算平均核心利用率
    avg_core_utilization = _average([c.core_utilization for c in cores])

    # 计算平均气泡率
    avg_bubble_rate = _average([c.bubble_rate for c in cores])

    # 计算AIC核心平均利用率
    avg_aic_utilization = _average([c.core_utilization for c in aic_cores])

    # 计算AIV核心平均利用率
    avg_aiv_utilization = _average([c.core_utilization for c in aiv_cores])

    # 计算AIC核心平均气泡率
    avg_aic_bubble_rate = _average([c.bubble_rate for c in aic_cores])

    # 计算AIV核心平均气泡率
    avg_aiv_bubble_rate = _average([c.bubble_rate for c in aiv_cores])

    # 算子实际执行时间（所有核心最大工作时间）
    max_work_time = max(c.total_work_time for c in cores) if cores else 0

    # 核心负载均衡度（标准差）
    load_balance = _calculate_load_balance(aic_cores)

    return {
        "avg_core_utilization": avg_core_utilization,
        "avg_bubble_rate": avg_bubble_rate,
        "avg_aic_utilization": avg_aic_utilization,
        "avg_aiv_utilization": avg_aiv_utilization,
        "avg_aic_bubble_rate": avg_aic_bubble_rate,
        "avg_aiv_bubble_rate": avg_aiv_bubble_rate,
        "max_work_time": max_work_time,
        "load_balance": load_balance,
        "aic_cores": aic_cores,
        "aiv_cores": aiv_cores,
        "all_cores": cores,
    }


def _average(values: List[float]) -> float:
    """计算列表平均值，空列表返回 0。"""
    return sum(values) / len(values) if values else 0


def _calculate_load_balance(aic_cores: List[CoreMetrics]) -> float:
    """根据 AIC 实际工作时间计算负载均衡度。"""
    if len(aic_cores) <= 1:
        return 100

    aic_times = [core.aicore_time for core in aic_cores]
    mean_time = _average(aic_times)
    if mean_time <= 0:
        return 0

    variance = _average([(work_time - mean_time) ** 2 for work_time in aic_times])
    return (1 - variance**0.5 / mean_time) * 100


def _higher_is_better_rating(value: float) -> Tuple[str, str]:
    ratings = (
        (90, "⭐⭐⭐⭐⭐", "优秀"),
        (80, "⭐⭐⭐⭐", "良好"),
        (60, "⭐⭐⭐", "一般"),
        (50, "⭐⭐", "较差"),
    )
    for threshold, stars, description in ratings:
        if value > threshold:
            return stars, description
    return "⭐", "很差"


def _lower_is_better_rating(value: float) -> Tuple[str, str]:
    ratings = (
        (2, "⭐⭐⭐⭐⭐", "优秀"),
        (5, "⭐⭐⭐⭐", "良好"),
        (10, "⭐⭐⭐", "一般"),
        (20, "⭐⭐", "较差"),
    )
    for threshold, stars, description in ratings:
        if value < threshold:
            return stars, description
    return "⭐", "很差"


def get_rating(value: float, metric_type: str) -> Tuple[str, str]:
    """获取性能评级"""
    if metric_type in ("core_utilization", "load_balance"):
        return _higher_is_better_rating(value)
    if metric_type == "bubble_rate":
        return _lower_is_better_rating(value)
    return "⭐", "未知"


def _utilization_bottleneck(value: float) -> Optional[Dict]:
    if value < 50:
        return {
            "type": "核心利用率低",
            "severity": "高",
            "description": f"平均核心利用率仅为 {value:.2f}%，远低于理想值",
            "impact": "严重影响算子性能",
            "suggestion": "建议调整Tilesize增大算术强度，或使用L2亲和调度",
        }
    if value < 70:
        return {
            "type": "核心利用率偏低",
            "severity": "中",
            "description": f"平均核心利用率为 {value:.2f}%，有优化空间",
            "impact": "影响算子性能",
            "suggestion": "建议检查任务调度策略，优化内存访问",
        }
    return None


def _bubble_bottleneck(value: float) -> Optional[Dict]:
    if value > 20:
        return {
            "type": "气泡率过高",
            "severity": "高",
            "description": f"平均气泡率为 {value:.2f}%，存在大量调度等待",
            "impact": "严重影响算子性能",
            "suggestion": "建议增大任务粒度，使用loop_unroll优化",
        }
    if value > 10:
        return {
            "type": "气泡率偏高",
            "severity": "中",
            "description": f"平均气泡率为 {value:.2f}%，存在调度等待",
            "impact": "影响算子性能",
            "suggestion": "建议优化调度策略，使用L1Reuse优化",
        }
    return None


def _load_balance_bottleneck(value: float) -> Optional[Dict]:
    if value < 60:
        return {
            "type": "核心负载不均衡",
            "severity": "高",
            "description": f"核心负载均衡度为 {value:.2f}%，核心间负载差异大",
            "impact": "严重影响算子性能",
            "suggestion": "建议调整任务分配策略，优化tile size",
        }
    if value < 80:
        return {
            "type": "核心负载略有不均",
            "severity": "中",
            "description": f"核心负载均衡度为 {value:.2f}%，核心间负载有一定差异",
            "impact": "影响算子性能",
            "suggestion": "建议检查任务分配是否均匀",
        }
    return None


def _predecessor_bottleneck(aic_cores: List[CoreMetrics]) -> Optional[Dict]:
    if not aic_cores:
        return None

    max_wait_pred = max(core.wait_predecessor_time for core in aic_cores)
    if max_wait_pred <= 500:
        return None
    return {
        "type": "等待前驱时间过长",
        "severity": "中",
        "description": f"最大等待前驱时间为 {max_wait_pred:.2f} us，任务依赖较多",
        "impact": "影响算子性能",
        "suggestion": "建议减少任务依赖，使用sg_set_scope合并子图",
    }


def analyze_bottlenecks(metrics: Dict) -> List[Dict]:
    """分析性能瓶颈"""
    candidates = (
        _utilization_bottleneck(metrics["avg_core_utilization"]),
        _bubble_bottleneck(metrics["avg_bubble_rate"]),
        _load_balance_bottleneck(metrics["load_balance"]),
        _predecessor_bottleneck(metrics["aic_cores"]),
    )
    return [bottleneck for bottleneck in candidates if bottleneck is not None]


def generate_optimization_suggestions(metrics: Dict, bottlenecks: List[Dict]) -> Dict:
    """生成优化建议"""
    suggestions = {"high_priority": [], "medium_priority": [], "low_priority": []}

    # 根据瓶颈生成建议
    for bottleneck in bottlenecks:
        if bottleneck["severity"] == "高":
            suggestions["high_priority"].append(bottleneck)
        elif bottleneck["severity"] == "中":
            suggestions["medium_priority"].append(bottleneck)
        else:
            suggestions["low_priority"].append(bottleneck)

    # 添加具体优化代码示例
    if metrics["avg_core_utilization"] < 50:
        suggestions["high_priority"].append(
            {
                "type": "使用L2亲和调度",
                "code": '@pypto.frontend.jit(runtime_options={"device_sched_mode": 1})',
                "description": "启用L2亲和调度，减少核心间通信开销",
            }
        )
        suggestions["high_priority"].append(
            {
                "type": "调整Cube Tilesize",
                "code": "pypto.set_cube_tile_shapes([128, 128], [128, 512], [128, 128])",
                "description": "增大Tilesize，提高算术强度",
            }
        )

    if metrics["avg_bubble_rate"] > 10:
        suggestions["high_priority"].append(
            {
                "type": "使用loop_unroll",
                "code": (
                    """for b, k in pypto.loop_unroll(A.shape[0] // 64, unroll_list=[64, 16, 4], name="A", idx_name='b'):
    if k <= 16:
        pypto.set_vec_tile_shapes(16, 64)
    else:
        pypto.set_vec_tile_shapes(64, 64)
    tile_a = A[b * 64:(b + k) * 64, :]
    tile_a = tile_a + 2
    B[b * 64:, :] = tile_a"""
                ),
                "description": "对于循环类任务动态轴范围较广时开启loop_unroll",
            }
        )
        suggestions["medium_priority"].append(
            {
                "type": "使用L1Reuse优化",
                "code": "pypto.set_pass_options(cube_l1_reuse_setting={0: 8})",
                "description": "启用L1缓存复用，减少内存访问",
            }
        )

    if metrics["load_balance"] < 80:
        suggestions["medium_priority"].append(
            {
                "type": "优化任务分配",
                "code": "# 调整tile size使任务更均匀\npypto.set_vec_tile_shapes(64, 64)",
                "description": "调整tile size使任务分配更均匀",
            }
        )

    return suggestions


def _rating_score(description: str) -> int:
    """将评级描述转换为综合评分。"""
    return {"优秀": 5, "良好": 4, "一般": 3, "较差": 2}.get(description, 1)


def _overall_rating(descriptions: List[str]) -> Tuple[str, str]:
    avg_score = _average([_rating_score(description) for description in descriptions])
    ratings = (
        (4.5, "⭐⭐⭐⭐⭐", "优秀"),
        (3.5, "⭐⭐⭐⭐", "良好"),
        (2.5, "⭐⭐⭐", "一般"),
        (1.5, "⭐⭐", "较差"),
    )
    for threshold, stars, description in ratings:
        if avg_score >= threshold:
            return stars, description
    return "⭐", "很差"


def _format_core_rows(cores: List[CoreMetrics]) -> str:
    rows = []
    for core in cores:
        rows.append(
            f"| {core.core_name} | {core.task_num} | {core.total_work_time:.2f} | "
            f"{core.total_wait_time:.2f} | {core.wait_schedule_time:.2f} | "
            f"{core.wait_predecessor_time:.2f} | {core.aicore_time:.2f} | "
            f"{core.core_utilization:.2f}% | {core.bubble_rate:.2f}% |\n"
        )
    return "".join(rows)


def _format_metrics_section(
    metrics: Dict, ratings: Dict, overall: Tuple[str, str]
) -> str:
    util_rating, util_desc = ratings["utilization"]
    bubble_rating, bubble_desc = ratings["bubble"]
    balance_rating, balance_desc = ratings["balance"]
    overall_rating, overall_desc = overall
    return f"""# PyPTO 算子性能分析报告

## 1. 核心性能指标

### 算子实际执行时间
**{metrics["max_work_time"]:.2f} us**

### AIC 核心性能指标

| 核心 | 任务数 | 总工作时间 | 总等待时间 | 等待调度时间 | 等待前驱时间 | AicoreTime | 核心利用率 | 气泡率 |
|------|--------|------------|------------|--------------|--------------|------------|------------|--------|
{_format_core_rows(metrics["aic_cores"])}
### AIV 核心性能指标

| 核心 | 任务数 | 总工作时间 | 总等待时间 | 等待调度时间 | 等待前驱时间 | AicoreTime | 核心利用率 | 气泡率 |
|------|--------|------------|------------|--------------|--------------|------------|------------|--------|
{_format_core_rows(metrics["aiv_cores"])}
## 2. 性能指标统计

| 指标 | 数值 |
|------|------|
| 平均核心利用率 | {metrics["avg_core_utilization"]:.2f}% |
| 平均气泡率 | {metrics["avg_bubble_rate"]:.2f}% |
| AIC平均核心利用率 | {metrics["avg_aic_utilization"]:.2f}% |
| AIV平均核心利用率 | {metrics["avg_aiv_utilization"]:.2f}% |
| AIC平均气泡率 | {metrics["avg_aic_bubble_rate"]:.2f}% |
| AIV平均气泡率 | {metrics["avg_aiv_bubble_rate"]:.2f}% |
| 核心负载均衡度 | {metrics["load_balance"]:.2f}% |

## 3. 性能评级

| 指标 | 当前值 | 目标值(⭐⭐⭐⭐⭐) | 评级 | 描述 |
|------|--------|----------------|------|------|
| 核心利用率 | {metrics["avg_core_utilization"]:.2f}% | >90% | {util_rating} | {util_desc} |
| 气泡率 | {metrics["avg_bubble_rate"]:.2f}% | <2% | {bubble_rating} | {bubble_desc} |
| 负载均衡度 | {metrics["load_balance"]:.2f}% | >90% | {balance_rating} | {balance_desc} |

### 综合评级
**{overall_rating} ({overall_desc})**

"""


def _format_bottleneck_section(bottlenecks: List[Dict]) -> str:
    section = "## 4. 性能瓶颈分析\n\n"
    if not bottlenecks:
        return section + "未发现明显的性能瓶颈，性能表现良好。\n\n"

    for index, bottleneck in enumerate(bottlenecks, 1):
        section += f"{index}. **{bottleneck['type']}** ({bottleneck['severity']})\n"
        section += f"   - 描述: {bottleneck['description']}\n"
        section += f"   - 影响: {bottleneck['impact']}\n"
        section += f"   - 建议: {bottleneck['suggestion']}\n\n"
    return section


def _format_suggestion_group(
    title: str, suggestions: List[Dict], include_code: bool
) -> str:
    if not suggestions:
        return ""

    section = f"### {title}\n\n"
    for index, suggestion in enumerate(suggestions, 1):
        section += f"{index}. **{suggestion['type']}**\n"
        section += f"   - 描述: {suggestion.get('description', '')}\n"
        if include_code and "code" in suggestion:
            section += f"   - 代码:\n```python\n{suggestion['code']}\n```\n"
        section += "\n"
    return section


def _format_suggestion_section(suggestions: Dict) -> str:
    return "".join(
        (
            "## 5. 性能优化建议\n\n",
            _format_suggestion_group(
                "高优先级优化", suggestions["high_priority"], True
            ),
            _format_suggestion_group(
                "中优先级优化", suggestions["medium_priority"], True
            ),
            _format_suggestion_group(
                "低优先级优化", suggestions["low_priority"], False
            ),
        )
    )


def _format_data_locations(output_dir: str) -> str:
    return f"""
## 6. 性能数据文件位置

- 泳道图: {output_dir}/merged_swimlane.json
- 气泡分析: {output_dir}/bubble_analysis.log
- 性能追踪: {output_dir}/machine_runtime_operator_trace.json

可在 https://ui.perfetto.dev/ 上传泳道图文件进行可视化分析。
"""


def _format_frontend_tuning(metrics: Dict) -> str:
    if metrics["avg_core_utilization"] < 50 or metrics["avg_bubble_rate"] > 20:
        return (
            "### 7.1 开箱性能调优（推荐优先）\n\n"
            "**是否需要**: 是\n\n"
            "**原因**: 核心利用率偏低或气泡率过高，需先优化基础代码写法\n\n"
            "**调优重点**:\n"
            "- Loop 写法优化\n- TileShape 设置优化\n- 数据操作优化\n\n"
            "**详细指南**: 加载 `tune-frontend` 子技能\n\n"
        )
    return ""


def _format_swimlane_tuning(metrics: Dict) -> str:
    if metrics["avg_bubble_rate"] > 10 or metrics["load_balance"] < 80:
        reason_parts = []
        if metrics["avg_bubble_rate"] > 10:
            reason_parts.append(
                f"气泡率 {metrics['avg_bubble_rate']:.2f}% 偏高，需优化调度策略"
            )
        if metrics["load_balance"] < 80:
            reason_parts.append(
                f"负载均衡度 {metrics['load_balance']:.2f}% 偏低，需优化任务分配"
            )
        return (
            "### 7.2 深度性能调优\n\n"
            "**是否需要**: 是\n\n"
            f"**原因**: {'; '.join(reason_parts)}\n\n"
            "**调优重点**:\n"
            "- Stitch 调优\n- TileShape 深度调优\n- 合图调优\n- 调度策略调优\n\n"
            "**详细指南**: 加载 `tune-swimlane` 子技能\n\n"
        )
    return ""


def _format_incore_tuning(metrics: Dict) -> str:
    section = "### 7.3 核内性能调优\n\n"
    if metrics["avg_core_utilization"] >= 70 and metrics["avg_bubble_rate"] < 10:
        section += "**是否需要**: 可能需要\n\n"
        section += "**原因**: 调度和利用率已接近合理水平，进一步优化需分析核内指令\n\n"
    else:
        section += "**是否需要**: 待开箱和深度调优完成后评估\n\n"
        section += "**原因**: 需先解决高优先级瓶颈\n\n"
    section += "**调优重点**:\n"
    section += "- 特殊 Shape 处理\n- 冗余计算优化\n- 尾轴优化\n- Operation 实现检查\n\n"
    section += "**详细指南**: 加载 `tune-incore` 子技能\n\n"
    return section


def _format_tuning_section(metrics: Dict) -> str:
    return "".join(
        (
            "## 7. 调优方向建议\n\n",
            _format_frontend_tuning(metrics),
            _format_swimlane_tuning(metrics),
            _format_incore_tuning(metrics),
        )
    )


def _format_summary(
    overall: Tuple[str, str], bottlenecks: List[Dict], suggestions: Dict
) -> str:
    overall_rating, overall_desc = overall
    high_priority = suggestions["high_priority"]
    tuning_direction = (
        ", ".join(suggestion["type"] for suggestion in high_priority[:3])
        if high_priority
        else "无特定建议"
    )
    return f"""## 8. 调优记录

| 轮次 | 优化内容 | 修改前执行时间(us) | 修改后执行时间(us) | 提升比例 | 精度结果 |
|------|---------|-------------------|-------------------|---------|---------|
| （调优过程中填写） | | | | | |

## 9. 总结

### 9.1 当前状态

- **性能评级**: {overall_rating} ({overall_desc})
- **主要瓶颈**: {", ".join(b["type"] for b in bottlenecks) if bottlenecks else "无明显瓶颈"}
- **调优方向**: {tuning_direction}

### 9.2 下一步行动

1. 按调优方向建议（第 7 节）顺序执行调优
2. 每次优化后验证精度并记录到调优记录（第 8 节）
3. 达到性能目标后生成最终报告
"""


def generate_report(
    metrics: Dict, bottlenecks: List[Dict], suggestions: Dict, output_dir: str
) -> str:
    """生成性能分析报告"""
    ratings = {
        "utilization": get_rating(metrics["avg_core_utilization"], "core_utilization"),
        "bubble": get_rating(metrics["avg_bubble_rate"], "bubble_rate"),
        "balance": get_rating(metrics["load_balance"], "load_balance"),
    }
    overall = _overall_rating([description for _, description in ratings.values()])
    return "".join(
        (
            _format_metrics_section(metrics, ratings, overall),
            _format_bottleneck_section(bottlenecks),
            _format_suggestion_section(suggestions),
            _format_data_locations(output_dir),
            _format_tuning_section(metrics),
            _format_summary(overall, bottlenecks, suggestions),
        )
    )


def find_bubble_analysis_log(output_dir: str) -> Optional[str]:
    """查找 bubble_analysis.log 文件，支持自动递归搜索"""
    direct_path = os.path.join(output_dir, "bubble_analysis.log")
    if os.path.exists(direct_path):
        return direct_path

    candidates = []
    for root, _, files in os.walk(output_dir):
        if "bubble_analysis.log" in files:
            candidates.append(os.path.join(root, "bubble_analysis.log"))

    if candidates:
        candidates.sort()
        return candidates[0]

    return None


class AnalysisInputError(Exception):
    """Raised for an invalid CLI input without terminating from a helper."""


def _require_output_dir(argv: List[str]) -> str:
    if len(argv) < 2:
        logging.info("Usage: python analyze_perf.py <output_dir>")
        logging.info(
            "Example: python analyze_perf.py custom/<op>/output/output_20260304_171658_543682_529508"
        )
        logging.info("")
        logging.info(
            "Note: output_dir should be the directory containing bubble_analysis.log"
        )
        logging.info(
            "      If not found, the script will search subdirectories automatically."
        )
        raise AnalysisInputError()

    output_dir = argv[1]
    if not os.path.exists(output_dir):
        logging.error(f"Error: directory not found: {output_dir}")
        logging.info("")
        logging.info(
            "Hint: output directory is relative to where you ran the operator command."
        )
        logging.info(
            "  If you ran 'python3 custom/op/op.py --run-mode npu' from project root:"
        )
        logging.info(f"    try: {os.path.abspath('output')}")
        logging.info("  If you ran from the operator directory:")
        logging.info("    try: <operator_dir>/output/<timestamp_dir>")
        logging.info("")
        logging.info(
            "Quick find: run 'find . -name bubble_analysis.log' from project root"
        )
        raise AnalysisInputError()
    return output_dir


def _require_bubble_log(output_dir: str) -> str:
    bubble_log_path = find_bubble_analysis_log(output_dir)
    if bubble_log_path is not None:
        return bubble_log_path

    logging.error(f"Error: bubble_analysis.log not found in {output_dir}")
    logging.info("")
    logging.info("Possible reasons:")
    logging.info(
        "  1. The operator has not been run with debug_options={'runtime_debug_mode': 1}"
    )
    logging.info("  2. The output directory is in a different location than expected")
    logging.info("")
    logging.info(
        "Hint: output directory is relative to where you ran the operator command."
    )
    logging.info("  If you ran from the operator directory (e.g. custom/op/):")
    logging.info(f"    try: {output_dir}/output/  (if output_dir is the operator dir)")
    logging.info(
        "  Quick find: run 'find . -name bubble_analysis.log' from project root"
    )
    raise AnalysisInputError()


def _save_report(report: str, output_dir: str) -> str:
    report_path = os.path.join(output_dir, "performance_analysis_report.md")
    with open(report_path, "w") as report_file:
        report_file.write(report)
    return report_path


def _log_metrics_summary(metrics: Dict, bottlenecks: List[Dict]) -> None:
    logging.info("\n=== Performance metrics summary ===")
    logging.info(f"Average core utilization: {metrics['avg_core_utilization']:.2f}%")
    logging.info(f"Average bubble rate: {metrics['avg_bubble_rate']:.2f}%")
    logging.info(f"Core load balance: {metrics['load_balance']:.2f}%")
    logging.info(f"Operator actual execution time: {metrics['max_work_time']:.2f} us")
    logging.info(f"Found {len(bottlenecks)} performance bottlenecks")


def main() -> int:
    """主函数"""
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    try:
        output_dir = _require_output_dir(sys.argv)
        bubble_log_path = _require_bubble_log(output_dir)
    except AnalysisInputError:
        return 1

    real_output_dir = os.path.dirname(bubble_log_path)
    logging.info(f"Analyzing performance data: {bubble_log_path}")

    # 解析性能数据
    cores = parse_bubble_analysis(bubble_log_path)
    logging.info(f"Found {len(cores)} cores")

    # 计算性能指标
    metrics = calculate_performance_metrics(cores)

    # 分析性能瓶颈
    bottlenecks = analyze_bottlenecks(metrics)

    # 生成优化建议
    suggestions = generate_optimization_suggestions(metrics, bottlenecks)

    # 生成报告
    report = generate_report(metrics, bottlenecks, suggestions, real_output_dir)

    report_path = _save_report(report, real_output_dir)
    logging.info(f"Performance analysis report generated: {report_path}")
    _log_metrics_summary(metrics, bottlenecks)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
