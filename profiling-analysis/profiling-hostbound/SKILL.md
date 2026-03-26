---
name: profiling-analysis-profiling-hostbound
description: Skill for analyzing hostbound performance bottlenecks, focusing on free time and下发 issues in Ascend NPU systems.
---

# Profiling 下发瓶颈分析 Skill

## 功能概述

该Skill用于分析系统中的下发瓶颈问题，当主分析Skill检测到空闲耗时占比超过20%时自动触发。

## 分析内容

- **空闲时间分析**：详细分析系统空闲时间的分布和原因
- **下发延迟分析**：识别下发过程中的延迟问题
- **任务调度分析**：分析任务调度策略的合理性

## 输出结果

- 空闲时间的详细分布报告
- 可能的下发瓶颈点识别
- 针对性的优化建议

## 使用方式

该Skill通常由主分析Skill `/profiling-analysis-profiling-main` 自动调用，也可以单独使用：

```python
# 运行下发瓶颈分析
python scripts/analyze_hostbound.py --input <path_to_csv_files>
```
