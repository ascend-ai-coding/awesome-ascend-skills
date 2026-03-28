---
name: profiling-analysis-profiling-hostbound
description: 分析MindStudio Insight采集的profiling数据识别快慢卡，以及分析系统trace文件识别Host侧进程性能问题。当用户需要分析NPU卡间性能差异或Host侧进程瓶颈时调用。
---

# NPU 快慢卡与Host侧性能分析 Skill

## 功能概述

该Skill提供两个独立的分析功能，可根据用户需求选择使用：

1. **快慢卡性能分析**：通过解析MindStudio Insight采集的analysis.db数据库文件，分析各NPU卡的执行时间和通信时间，识别性能表现不一致的快慢卡。
2. **Host侧Trace分析**：通过解析系统trace文件，分析Host侧进程性能，识别存在问题的进线程及主要瓶颈。

两个功能独立运行，不互相调用。用户可以根据需要单独或同时使用两种分析功能。

**核心功能：**
- 多维度快慢卡识别（计算时间+通信时间）
- Host侧进程性能分析（上下文切换、CPU迁移、中断等）
- 详细的Excel报告生成
- 问题进线程及瓶颈定位

## 支持的数据格式

- MindStudio Insight采集的profiling数据（analysis.db数据库文件）
- 系统trace文件（如Linux ftrace生成的trace文件）
- 支持Ascend NPU系统的性能分析

## 子技能说明

该Skill包含两个子技能，用户可根据具体需求选择使用：

### 1. 快慢卡性能分析
详细分析NPU卡间性能差异，识别计算或通信异常的快慢卡。

**主要功能：**
- 基于计算时间和通信时间的多维度快慢卡识别
- 卡间性能差异百分比计算
- 详细的Excel报告生成

**详细文档：** [快慢卡性能分析](./reference/slow-cards-analysis.md)

### 2. Host侧Trace分析
分析系统trace文件，识别Host侧进程性能问题。

**主要功能：**
- 进程运行时间统计
- 上下文切换次数检测
- CPU迁移次数分析
- 中断时间统计
- 问题进线程识别

**详细文档：** [Host侧Trace分析](./reference/ftrace-analysis.md)

## 使用方式

### 1. 由主分析Skill自动调用

该Skill通常由主分析Skill `/profiling-analysis-profiling-main` 自动触发。当主Skill检测到需要分析NPU卡间性能差异或Host侧进程瓶颈时，会自动调用相应的子技能进行深入分析。

### 2. 单独使用

根据具体需求，选择使用相应的子技能：

#### 2.1 使用快慢卡性能分析
```bash
python scripts/analyze_hostbound.py --db-path <analysis.db路径> [--output <报告输出路径>] [--speed-threshold <阈值>]
```

**详细说明：** 请参考 [快慢卡性能分析文档](./reference/slow-cards-analysis.md)

#### 2.2 使用Host侧Trace分析
```bash
python scripts/trace_analyzer.py --input <trace文件路径> [--output <报告输出路径>]
```

**详细说明：** 请参考 [Host侧Trace分析文档](./reference/ftrace-analysis.md)

## 使用建议

**功能选择**：
- 当需要分析NPU卡间性能差异时，使用**快慢卡性能分析**子技能
- 当需要分析Host侧进程性能问题时，使用**Host侧Trace分析**子技能
- 若需全面了解系统性能，可同时运行两种分析子技能

**分析优化路径**：
- 若发现卡间性能不均衡，可进一步分析Host侧Trace查找可能的进程调度或资源竞争问题
- 若Host侧存在性能瓶颈，可结合快慢卡分析确认是否影响了NPU卡的整体性能发挥

**报告解读**：
- 快慢卡分析报告重点关注卡间性能差异
- Host侧Trace分析报告重点关注进程行为和资源竞争情况
- 综合两份报告可获得更全面的系统性能洞察

## 依赖要求
- Python 3.8+
- 依赖库：pandas, openpyxl, matplotlib, sqlite3