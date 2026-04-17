---
name: profiling-analysis
description: 华为昇腾NPU性能分析技能集，**当用户提供Profiling文件或目录路径时必须使用**，用于分析Profiling数据识别计算、通信、下发等性能瓶颈，支持step_trace_time.csv、op_statistic.csv、kernel_details.csv等多种数据格式，提供完整的性能分析流程和优化建议。
keywords:
    - 性能分析
    - Profiling
    - 昇腾性能分析
    - NPU性能优化
    - 计算瓶颈
    - 通信瓶颈
    - 下发问题
    - hostbound
    - 高耗时算子
    - step_trace_time.csv
    - op_statistic.csv
    - kernel_details.csv
    - 性能瓶颈识别
    - 算子性能分析
    - 通信性能分析
    - Host侧性能分析
---

# 昇腾NPU Profiling性能分析技能集

## 功能概述

profiling-analysis 是一套完整的华为昇腾NPU性能分析技能集，用于分析Profiling生成的性能数据，自动识别系统性能瓶颈类型（计算、通信、下发），并提供深入分析能力和优化建议。

## 核心功能与执行流程

profiling-analysis 技能通过以下流程自动完成性能分析：

### 1. 文件扫描与收集
- **输入**：目标文件夹路径
- **处理**：递归遍历目录，自动查找所有 `step_trace_time.csv` 文件
- **验证**：未找到文件时输出错误信息并终止

### 2. 性能数据解析与计算
- **处理**：验证文件结构，提取计算（Computing）、通信（Communication）、空闲（Free）等核心指标
- **计算**：计算各阶段耗时占比（保留2位小数）
- **异常处理**：跳过读取失败、字段缺失或总耗时为0的文件

### 3. 全局统计与瓶颈判定
- **统计**：计算所有文件的平均耗时占比，识别各项占比最高的文件
- **判定逻辑**（优先级从高到低）：
  - 下发问题（Hostbound）：空闲耗时占比 > 20%
  - 计算问题（Computing）：计算耗时占比 > 85%
  - 通信问题（Communication）：通信耗时占比 > 10%
  - 无明显瓶颈：各项占比均在正常范围

### 4. 子技能自动调用
根据判定的瓶颈类型，自动调用对应子技能进行深入分析：
- **Hostbound** → 调用 `profiling-analysis-hostbound` 子技能
- **Computing** → 调用 `profiling-analysis-computing` 子技能，执行完整分析流程
- **Communication** → 调用 `profiling-analysis-communication` 子技能

### 5. 分析报告生成
输出完整的性能分析报告，包括：
- 文件扫描结果
- 单文件性能分析数据
- 全局统计与最高占比文件
- 瓶颈判定结论
- 子技能深入分析结果

### 执行流程图
```
┌─────────────────────────────────────────────────┐
│ 文件扫描与收集                                 │
│ 递归查找 step_trace_time.csv 文件               │
└─────────────────────────────────────────────────┘
                      ↓
┌─────────────────────────────────────────────────┐
│ 性能数据解析与计算                             │
│ 提取核心指标并计算占比                         │
└─────────────────────────────────────────────────┘
                      ↓
┌─────────────────────────────────────────────────┐
│ 全局统计与瓶颈判定                             │
│ 计算平均占比并识别瓶颈类型                     │
└─────────────────────────────────────────────────┘
                      ↓
┌─────────────────────────────────────────────────┐
│ 子技能自动调用                                 │
│ 根据瓶颈类型触发对应子技能                     │
└─────────────────────────────────────────────────┘
                      ↓
┌─────────────────────────────────────────────────┐
│ 分析报告生成                                   │
│ 输出完整性能分析结果                           │
└─────────────────────────────────────────────────┘
```

## 输入参数

| 参数名称   | 类型   | 是否必填 | 描述                                                         |
| ---------- | ------ | -------- | ------------------------------------------------------------ |
| input_path | string | 是       | 输入路径：支持单个Profiling文件路径，或包含多个Profiling文件的文件夹路径 |

## 使用示例

### 通过AI工具调用（推荐）
```
调用 skill "profiling-analysis" 分析 "input_path" 中的性能数据
```

### 参数说明
- `input_path`：替换为实际的Profiling数据文件夹路径
- 该调用将自动完成文件扫描、性能分析、瓶颈判定和子技能调用的完整流程

### 子技能独立调用（可选）
如果需要直接调用特定子技能进行深入分析：

```
# 分析计算瓶颈
调用 skill "profiling-analysis-computing" 分析 "input_path" 中的高耗时算子

# 分析通信瓶颈
调用 skill "profiling-analysis-communication" 分析 "input_path" 中的通信性能

# 分析下发瓶颈
调用 skill "profiling-analysis-hostbound" 分析 "input_path" 中的Host侧问题
```

## 子技能介绍

profiling-analysis 包含以下子技能，用于不同类型的性能分析。当主技能检测到对应瓶颈时，会自动调用子技能进行深入分析，无需手动执行：

| 子技能名称 | 功能描述 | 适用场景 |
|---------|---------|---------|
| **profiling-analysis** | 主分析技能，识别性能瓶颈类型 | 首次分析Profiling数据，快速定位主要瓶颈 |
| **profiling-analysis-computing(.\profiling-analysis-computing\SKILL.md)** | 计算瓶颈分析，包含：<br>- 高耗时算子筛选（Top-N算子）<br>- 算子性能数据透视表分析<br>- 关键算子（如MatMul系列）形状解析与优化建议 | 计算占比过高时，深入分析具体的高耗时算子及其性能特征 |
| **profiling-analysis-communication(.\profiling-analysis-communication\SKILL.md)** | 通信瓶颈分析，分析集合通信性能 | 通信占比过高时，分析通信操作的性能 |
| **profiling-analysis-hostbound(.\profiling-analysis-hostbound\SKILL.md)** | 下发瓶颈分析，分析Host侧性能问题 | 空闲占比过高时，分析Host侧的下发问题 |

## 支持的数据格式

- `step_trace_time.csv`：包含计算、通信、空闲等时间占比信息
- `op_statistic.csv`：包含算子执行统计信息
- `kernel_details.csv`：包含算子内核详细执行信息
- `op_summary.csv`：包含算子执行摘要信息
- `analysis.db`：Profiling数据库文件

## 安装与依赖

- Python 3.8+
- 依赖库：pandas, numpy, matplotlib (可选)

## 官方文档

- [华为昇腾Profiling工具文档](https://www.hiascend.com/document/detail/zh/canncommercial/81RC1/perftools/profiler/index.html)
- [MindStudio Profiling使用指南](https://www.hiascend.com/document/detail/zh/mindstudio/60RC1/profiling/index.html)
