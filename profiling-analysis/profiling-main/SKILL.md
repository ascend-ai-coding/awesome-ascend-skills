---
name: profiling-analysis-profiling-main
description: Main skill for profiling performance analysis, identifying bottlenecks in computing, communication, and hostbound operations.
---

# Profiling 性能分析主 Skill

## 功能概述

该Skill用于对Profiling生成的`step_trace_time.csv`文件进行性能分析，自动识别系统性能瓶颈类型（计算、通信、下发），并触发相应的子Skill进行深入分析。

## 输入参数

| 参数名称   | 类型   | 是否必填 | 描述                                                         |
| ---------- | ------ | -------- | ------------------------------------------------------------ |
| input_path | string | 是       | 输入路径：支持单个`step_trace_time.csv`文件路径，或包含多个该文件的文件夹路径 |

## 使用示例

```python
# 将路径修改为你的实际文件夹路径
TARGET_FOLDER = r"./"

# 运行分析
python scripts/performance_analysis_main_process.py
```

## 分析流程

1. **扫描CSV文件**：递归遍历指定目录，收集所有`step_trace_time.csv`文件
2. **解析数据**：验证文件结构，提取核心字段（Computing、Communication、Free）
3. **计算占比**：计算各阶段耗时占比
4. **瓶颈判定**：根据阈值判定瓶颈类型
5. **触发子Skill**：调用相应的子Skill进行深入分析

## 瓶颈判定规则

1. **下发问题（Hostbound）**：空闲耗时占比 > 10% → 触发 `/profiling-analysis-profiling-hostbound`
2. **计算问题（Computing）**：计算耗时占比 > 85% → 触发 `/profiling-analysis-profiling-computing`
3. **通信问题（Communication）**：通信耗时占比 > 10% → 触发 `/profiling-analysis-profiling-communication`

## 子Skill调用

如果遇到性能瓶颈问题，请调用子Agent运行对应的 /profiling-analysis-xxx-skill：

- **下发瓶颈问题**：调用 `/profiling-analysis-profiling-hostbound`
- **计算瓶颈问题**：调用 `/profiling-analysis-profiling-computing`
- **通信瓶颈问题**：调用 `/profiling-analysis-profiling-communication`