---
name: profiling-analysis-profiling-communication
description: Skill for analyzing communication performance bottlenecks, focusing on data transfer efficiency between host and device in Ascend NPU systems.
---

# Profiling 通信瓶颈分析 Skill

## 功能概述

该Skill用于分析系统中的通信瓶颈问题，当主分析Skill检测到通信耗时占比超过10%时自动触发。

## 分析内容

- **通信时间分析**：详细分析通信耗时的分布和原因
- **数据传输优化**：识别数据传输过程中的瓶颈
- **并行通信分析**：分析通信与计算的重叠情况

## 输出结果

- 通信时间的详细分布报告
- 可能的通信瓶颈点识别
- 针对性的优化建议

## 使用方式

该Skill通常由主分析Skill `/profiling-analysis-profiling-main` 自动调用，也可以单独使用：

```python
# 运行通信瓶颈分析
python scripts/analyze_communication.py --input <path_to_csv_files>
```