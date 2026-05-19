#!/usr/bin/env python3
# coding: utf-8
# ----------------------------------------------------------------------------------------------------------
# Copyright (c) 2025-2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ----------------------------------------------------------------------------------------------------------
"""
PyPTO 性能分析脚本
从bubble_analysis.log中提取性能数据并计算性能指标
"""

import logging
import os
import re
import sys
from dataclasses import dataclass
from typing import Dict, List, Tuple


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

    with open(log_path, 'r') as f:
        content = f.read()

    # 匹配核心信息
    core_pattern = (
        r'\[(AIC_\d+|AIV_\d+)\] Execute task num:(\d+)\s+Core Total Work Time: ([\d.]+)\s+'
        r'Total Wait Time: ([\d.]+)\s+Wait Schedule Time: ([\d.]+)\s+'
        r'Wait Predecessor Time: ([\d.]+)'
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
            wait_predecessor_time=wait_predecessor_time
        )
        cores.append(core)

    return cores


def calculate_performance_metrics(cores: List[CoreMetrics]) -> Dict:
    """计算性能指标"""
    # 分离AIC和AIV核心
    aic_cores = [c for c in cores if c.core_name.startswith('AIC')]
    aiv_cores = [c for c in cores if c.core_name.startswith('AIV')]

    # 计算平均核心利用率
    avg_core_utilization = sum(c.core_utilization for c in cores) / len(cores) if cores else 0

    # 计算平均气泡率
    avg_bubble_rate = sum(c.bubble_rate for c in cores) / len(cores) if cores else 0

    # 计算AIC核心平均利用率
    avg_aic_utilization = sum(c.core_utilization for c in aic_cores) / len(aic_cores) if aic_cores else 0

    # 计算AIV核心平均利用率
    avg_aiv_utilization = sum(c.core_utilization for c in aiv_cores) / len(aiv_cores) if aiv_cores else 0

    # 计算AIC核心平均气泡率
    avg_aic_bubble_rate = sum(c.bubble_rate for c in aic_cores) / len(aic_cores) if aic_cores else 0

    # 计算AIV核心平均气泡率
    avg_aiv_bubble_rate = sum(c.bubble_rate for c in aiv_cores) / len(aiv_cores) if aiv_cores else 0

    # 算子实际执行时间（所有核心最大工作时间）
    max_work_time = max(c.total_work_time for c in cores) if cores else 0

    # 核心负载均衡度（标准差）
    if len(aic_cores) > 1:
        aic_times = [c.aicore_time for c in aic_cores]
        mean_time = sum(aic_times) / len(aic_times)
        variance = sum((t - mean_time) ** 2 for t in aic_times) / len(aic_times)
        std_dev = variance ** 0.5
        load_balance = (1 - std_dev / mean_time) * 100 if mean_time > 0 else 0
    else:
        load_balance = 100

    return {
        'avg_core_utilization': avg_core_utilization,
        'avg_bubble_rate': avg_bubble_rate,
        'avg_aic_utilization': avg_aic_utilization,
        'avg_aiv_utilization': avg_aiv_utilization,
        'avg_aic_bubble_rate': avg_aic_bubble_rate,
        'avg_aiv_bubble_rate': avg_aiv_bubble_rate,
        'max_work_time': max_work_time,
        'load_balance': load_balance,
        'aic_cores': aic_cores,
        'aiv_cores': aiv_cores,
        'all_cores': cores
    }


def get_rating(value: float, metric_type: str) -> Tuple[str, str]:
    """获取性能评级"""
    if metric_type == 'core_utilization':
        if value > 90:
            return '⭐⭐⭐⭐⭐', '优秀'
        elif value > 80:
            return '⭐⭐⭐⭐', '良好'
        elif value > 60:
            return '⭐⭐⭐', '一般'
        elif value > 50:
            return '⭐⭐', '较差'
        else:
            return '⭐', '很差'
    elif metric_type == 'bubble_rate':
        if value < 2:
            return '⭐⭐⭐⭐⭐', '优秀'
        elif value < 5:
            return '⭐⭐⭐⭐', '良好'
        elif value < 10:
            return '⭐⭐⭐', '一般'
        elif value < 20:
            return '⭐⭐', '较差'
        else:
            return '⭐', '很差'
    elif metric_type == 'load_balance':
        if value > 90:
            return '⭐⭐⭐⭐⭐', '优秀'
        elif value > 80:
            return '⭐⭐⭐⭐', '良好'
        elif value > 60:
            return '⭐⭐⭐', '一般'
        elif value > 50:
            return '⭐⭐', '较差'
        else:
            return '⭐', '很差'
    else:
        return '⭐', '未知'


def analyze_bottlenecks(metrics: Dict) -> List[Dict]:
    """分析性能瓶颈"""
    bottlenecks = []

    # 分析核心利用率
    if metrics['avg_core_utilization'] < 50:
        bottlenecks.append({
            'type': '核心利用率低',
            'severity': '高',
            'description': f'平均核心利用率仅为 {metrics["avg_core_utilization"]:.2f}%，远低于理想值',
            'impact': '严重影响算子性能',
            'suggestion': '建议调整Tilesize增大算术强度，或使用L2亲和调度'
        })
    elif metrics['avg_core_utilization'] < 70:
        bottlenecks.append({
            'type': '核心利用率偏低',
            'severity': '中',
            'description': f'平均核心利用率为 {metrics["avg_core_utilization"]:.2f}%，有优化空间',
            'impact': '影响算子性能',
            'suggestion': '建议检查任务调度策略，优化内存访问'
        })

    # 分析气泡率
    if metrics['avg_bubble_rate'] > 20:
        bottlenecks.append({
            'type': '气泡率过高',
            'severity': '高',
            'description': f'平均气泡率为 {metrics["avg_bubble_rate"]:.2f}%，存在大量调度等待',
            'impact': '严重影响算子性能',
            'suggestion': '建议增大任务粒度，使用loop_unroll优化'
        })
    elif metrics['avg_bubble_rate'] > 10:
        bottlenecks.append({
            'type': '气泡率偏高',
            'severity': '中',
            'description': f'平均气泡率为 {metrics["avg_bubble_rate"]:.2f}%，存在调度等待',
            'impact': '影响算子性能',
            'suggestion': '建议优化调度策略，使用L1Reuse优化'
        })

    # 分析负载均衡
    if metrics['load_balance'] < 60:
        bottlenecks.append({
            'type': '核心负载不均衡',
            'severity': '高',
            'description': f'核心负载均衡度为 {metrics["load_balance"]:.2f}%，核心间负载差异大',
            'impact': '严重影响算子性能',
            'suggestion': '建议调整任务分配策略，优化tile size'
        })
    elif metrics['load_balance'] < 80:
        bottlenecks.append({
            'type': '核心负载略有不均',
            'severity': '中',
            'description': f'核心负载均衡度为 {metrics["load_balance"]:.2f}%，核心间负载有一定差异',
            'impact': '影响算子性能',
            'suggestion': '建议检查任务分配是否均匀'
        })

    # 分析等待前驱时间
    aic_cores = metrics['aic_cores']
    if aic_cores:
        max_wait_pred = max(c.wait_predecessor_time for c in aic_cores)
        if max_wait_pred > 500:
            bottlenecks.append({
                'type': '等待前驱时间过长',
                'severity': '中',
                'description': f'最大等待前驱时间为 {max_wait_pred:.2f} us，任务依赖较多',
                'impact': '影响算子性能',
                'suggestion': '建议减少任务依赖，使用sg_set_scope合并子图'
            })

    return bottlenecks


def generate_optimization_suggestions(metrics: Dict, bottlenecks: List[Dict]) -> Dict:
    """生成优化建议"""
    suggestions = {
        'high_priority': [],
        'medium_priority': [],
        'low_priority': []
    }

    # 根据瓶颈生成建议
    for bottleneck in bottlenecks:
        if bottleneck['severity'] == '高':
            suggestions['high_priority'].append(bottleneck)
        elif bottleneck['severity'] == '中':
            suggestions['medium_priority'].append(bottleneck)
        else:
            suggestions['low_priority'].append(bottleneck)

    # 添加具体优化代码示例
    if metrics['avg_core_utilization'] < 50:
        suggestions['high_priority'].append({
             'type': '使用L2亲和调度',
             'code': '@pypto.jit(runtime_options={"device_sched_mode": 1})',
             'description': '启用L2亲和调度，减少核心间通信开销'
        })
        suggestions['high_priority'].append({
            'type': '调整Cube Tilesize',
            'code': 'pypto.set_cube_tile_shapes([128, 128], [128, 512], [128, 128])',
            'description': '增大Tilesize，提高算术强度'
        })

    if metrics['avg_bubble_rate'] > 10:
        suggestions['high_priority'].append({
            'type': '使用loop_unroll',
            'code': '''for b, k in pypto.loop_unroll(A.shape[0] // 64, unroll_list=[64, 16, 4], name="A", idx_name='b'):
    if k <= 16:
        pypto.set_vec_tile_shapes(16, 64)
    else:
        pypto.set_vec_tile_shapes(64, 64)
    tile_a = A[b * 64:(b + k) * 64, :]
    tile_a = tile_a + 2
    B[b * 64:, :] = tile_a''',
            'description': '对于循环类任务动态轴范围较广时开启loop_unroll'
        })
        suggestions['medium_priority'].append({
            'type': '使用L1Reuse优化',
            'code': 'pypto.set_pass_options(cube_l1_reuse_setting={0: 8})',
            'description': '启用L1缓存复用，减少内存访问'
        })

    if metrics['load_balance'] < 80:
        suggestions['medium_priority'].append({
            'type': '优化任务分配',
            'code': '# 调整tile size使任务更均匀\npypto.set_vec_tile_shapes(64, 64)',
            'description': '调整tile size使任务分配更均匀'
        })

    return suggestions


def generate_report(metrics: Dict, bottlenecks: List[Dict], suggestions: Dict, output_dir: str) -> str:
    """生成性能分析报告"""
    # 获取评级
    util_rating, util_desc = get_rating(metrics['avg_core_utilization'], 'core_utilization')
    bubble_rating, bubble_desc = get_rating(metrics['avg_bubble_rate'], 'bubble_rate')
    balance_rating, balance_desc = get_rating(metrics['load_balance'], 'load_balance')

    # 综合评级
    util_score = (
        5 if util_desc == '优秀' else
        4 if util_desc == '良好' else
        3 if util_desc == '一般' else
        2 if util_desc == '较差' else
        1
    )
    bubble_score = (
        5 if bubble_desc == '优秀' else
        4 if bubble_desc == '良好' else
        3 if bubble_desc == '一般' else
        2 if bubble_desc == '较差' else
        1
    )
    balance_score = (
        5 if balance_desc == '优秀' else
        4 if balance_desc == '良好' else
        3 if balance_desc == '一般' else
        2 if balance_desc == '较差' else
        1
    )
    avg_rating_score = (util_score + bubble_score + balance_score) / 3

    if avg_rating_score >= 4.5:
        overall_rating = '⭐⭐⭐⭐⭐'
        overall_desc = '优秀'
    elif avg_rating_score >= 3.5:
        overall_rating = '⭐⭐⭐⭐'
        overall_desc = '良好'
    elif avg_rating_score >= 2.5:
        overall_rating = '⭐⭐⭐'
        overall_desc = '一般'
    elif avg_rating_score >= 1.5:
        overall_rating = '⭐⭐'
        overall_desc = '较差'
    else:
        overall_rating = '⭐'
        overall_desc = '很差'

    report = f"""# PyPTO 算子性能分析报告

## 1. 核心性能指标

### 算子实际执行时间
**{metrics['max_work_time']:.2f} us**

### AIC 核心性能指标

| 核心 | 任务数 | 总工作时间 | 总等待时间 | 等待调度时间 | 等待前驱时间 | AicoreTime | 核心利用率 | 气泡率 |
|------|--------|------------|------------|--------------|--------------|------------|------------|--------|
"""

    for core in metrics['aic_cores']:
        report += (
            f"| {core.core_name} | {core.task_num} | {core.total_work_time:.2f} | "
            f"{core.total_wait_time:.2f} | {core.wait_schedule_time:.2f} | "
            f"{core.wait_predecessor_time:.2f} | {core.aicore_time:.2f} | "
            f"{core.core_utilization:.2f}% | {core.bubble_rate:.2f}% |\n"
        )

    report += """
### AIV 核心性能指标

| 核心 | 任务数 | 总工作时间 | 总等待时间 | 等待调度时间 | 等待前驱时间 | AicoreTime | 核心利用率 | 气泡率 |
|------|--------|------------|------------|--------------|--------------|------------|------------|--------|
"""

    for core in metrics['aiv_cores']:
        report += (
            f"| {core.core_name} | {core.task_num} | {core.total_work_time:.2f} | "
            f"{core.total_wait_time:.2f} | {core.wait_schedule_time:.2f} | "
            f"{core.wait_predecessor_time:.2f} | {core.aicore_time:.2f} | "
            f"{core.core_utilization:.2f}% | {core.bubble_rate:.2f}% |\n"
        )

    report += f"""
## 2. 性能指标统计

| 指标 | 数值 |
|------|------|
| 平均核心利用率 | {metrics['avg_core_utilization']:.2f}% |
| 平均气泡率 | {metrics['avg_bubble_rate']:.2f}% |
| AIC平均核心利用率 | {metrics['avg_aic_utilization']:.2f}% |
| AIV平均核心利用率 | {metrics['avg_aiv_utilization']:.2f}% |
| AIC平均气泡率 | {metrics['avg_aic_bubble_rate']:.2f}% |
| AIV平均气泡率 | {metrics['avg_aiv_bubble_rate']:.2f}% |
| 核心负载均衡度 | {metrics['load_balance']:.2f}% |

## 3. 性能评级

| 指标 | 当前值 | 目标值(⭐⭐⭐⭐⭐) | 评级 | 描述 |
|------|--------|----------------|------|------|
| 核心利用率 | {metrics['avg_core_utilization']:.2f}% | >90% | {util_rating} | {util_desc} |
| 气泡率 | {metrics['avg_bubble_rate']:.2f}% | <2% | {bubble_rating} | {bubble_desc} |
| 负载均衡度 | {metrics['load_balance']:.2f}% | >90% | {balance_rating} | {balance_desc} |

### 综合评级
**{overall_rating} ({overall_desc})**

## 4. 性能瓶颈分析

"""

    if bottlenecks:
        for i, bottleneck in enumerate(bottlenecks, 1):
            report += f"{i}. **{bottleneck['type']}** ({bottleneck['severity']})\n"
            report += f"   - 描述: {bottleneck['description']}\n"
            report += f"   - 影响: {bottleneck['impact']}\n"
            report += f"   - 建议: {bottleneck['suggestion']}\n\n"
    else:
        report += "未发现明显的性能瓶颈，性能表现良好。\n\n"

    report += "## 5. 性能优化建议\n\n"

    if suggestions['high_priority']:
        report += "### 高优先级优化\n\n"
        for i, suggestion in enumerate(suggestions['high_priority'], 1):
            report += f"{i}. **{suggestion['type']}**\n"
            report += f"   - 描述: {suggestion.get('description', '')}\n"
            if 'code' in suggestion:
                report += f"   - 代码:\n```python\n{suggestion['code']}\n```\n"
            report += "\n"

    if suggestions['medium_priority']:
        report += "### 中优先级优化\n\n"
        for i, suggestion in enumerate(suggestions['medium_priority'], 1):
            report += f"{i}. **{suggestion['type']}**\n"
            report += f"   - 描述: {suggestion.get('description', '')}\n"
            if 'code' in suggestion:
                report += f"   - 代码:\n```python\n{suggestion['code']}\n```\n"
            report += "\n"

    if suggestions['low_priority']:
        report += "### 低优先级优化\n\n"
        for i, suggestion in enumerate(suggestions['low_priority'], 1):
            report += f"{i}. **{suggestion['type']}**\n"
            report += f"   - 描述: {suggestion.get('description', '')}\n"
            report += "\n"

    report += f"""
## 6. 性能数据文件位置

- 泳道图: {output_dir}/merged_swimlane.json
- 气泡分析: {output_dir}/bubble_analysis.log
- 性能追踪: {output_dir}/machine_runtime_operator_trace.json

可在 https://ui.perfetto.dev/ 上传泳道图文件进行可视化分析。
"""

    return report


def main():
    """主函数"""
    logging.basicConfig(level=logging.INFO, format='%(message)s')

    if len(sys.argv) < 2:
        logging.info("Usage: python analyze_perf.py <output_dir>")
        logging.info("Example: python analyze_perf.py models/glm_v4_5/output/output_20260304_171658_543682_529508")
        sys.exit(1)

    output_dir = sys.argv[1]
    bubble_log_path = os.path.join(output_dir, 'bubble_analysis.log')

    if not os.path.exists(bubble_log_path):
        logging.info(f"Error: bubble_analysis.log not found in {output_dir}")
        sys.exit(1)

    logging.info(f"正在分析性能数据: {bubble_log_path}")

    # 解析性能数据
    cores = parse_bubble_analysis(bubble_log_path)
    logging.info(f"找到 {len(cores)} 个核心")

    # 计算性能指标
    metrics = calculate_performance_metrics(cores)

    # 分析性能瓶颈
    bottlenecks = analyze_bottlenecks(metrics)

    # 生成优化建议
    suggestions = generate_optimization_suggestions(metrics, bottlenecks)

    # 生成报告
    report = generate_report(metrics, bottlenecks, suggestions, output_dir)

    # 保存报告
    report_path = os.path.join(output_dir, 'performance_analysis_report.md')
    with open(report_path, 'w') as f:
        f.write(report)

    logging.info(f"性能分析报告已生成: {report_path}")

    # 输出关键指标摘要
    logging.info("\n=== 性能指标摘要 ===")
    logging.info(f"平均核心利用率: {metrics['avg_core_utilization']:.2f}%")
    logging.info(f"平均气泡率: {metrics['avg_bubble_rate']:.2f}%")
    logging.info(f"核心负载均衡度: {metrics['load_balance']:.2f}%")
    logging.info(f"算子实际执行时间: {metrics['max_work_time']:.2f} us")
    logging.info(f"发现 {len(bottlenecks)} 个性能瓶颈")


if __name__ == '__main__':
    main()
