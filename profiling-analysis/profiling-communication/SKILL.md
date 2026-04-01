---
name: profiling-analysis-profiling-communication
description: Skill for analyzing communication performance bottlenecks and detecting slow/fast rank issues in Ascend NPU systems. Use this skill whenever you need to analyze communication efficiency, data transfer bottlenecks, or identify slow/fast rank problems using profiling data.
---

# Profiling 通信瓶颈分析与快慢卡检测 Skill

## 功能概述

该Skill用于分析系统中的通信瓶颈问题和检测快慢卡现象，当主分析Skill检测到通信耗时占比超过10%时自动触发。支持对集群环境下的通信性能进行深入分析，识别影响性能的关键因素。

该Skill包含两个独立的分析子技能，用户可根据具体需求选择使用：

1. **通信算子分析**：从profiling的"communication"泳道提取通信算子，分析其性能特征，计算实际带宽并与理论带宽对比，识别通信瓶颈
2. **快慢卡检测**：检测集群环境下的快慢卡现象，分析其对整体性能的影响，定位导致快慢卡的具体原因

## 支持的数据格式

- CSV格式的profiling数据文件
- 完整的profiling数据文件夹
- MindStudio Insight采集的profiling数据
- 支持Ascend NPU系统的性能分析

## 子技能说明

### 1. 通信算子分析

深入分析系统中的通信操作，提取各类通信算子，统计其性能特征，计算实际通信带宽并与理论带宽对比，识别通信瓶颈。

**主要功能：**
- 从"communication"泳道提取各类通信算子
- 分析不同模型类型的通信模式（MOE模型/常规模型）
- 计算实际通信带宽并与理论带宽对比
- 识别带宽利用率低的通信操作
- 提供针对性的通信优化建议

**支持的通信算子：**
- MOE模型（开启EP）：hcom_reduceScatter_、hcom_allGather_
- 常规模型（未开启EP）：allreduce、allgather、broadcast、reduce、scatter、gather
- 其他通用通信算子：send、recv、barrier等

**详细文档：** [通信算子分析](./reference/communication-operator-analysis.md)

### 2. 快慢卡检测

分析集群环境下的快慢卡现象，识别影响性能的异常rank，并提供针对性的优化建议。

**主要功能：**
- 检测集群环境下的快慢卡现象
- 分析快慢卡对整体性能的影响
- 定位导致快慢卡的具体原因
- 基于Z-score统计方法的异常检测
- 提供针对性的优化建议

**详细文档：** [快慢卡检测](./reference/slow-rank-detection.md)

## 使用方式

### 1. 由主分析Skill自动调用

该Skill通常由主分析Skill `/profiling-analysis-profiling-main` 自动触发。当主Skill检测到通信耗时占比超过10%时，会自动调用相应的子技能进行深入分析。

### 2. 单独使用

根据具体需求，选择使用相应的子技能：

#### 2.1 使用通信算子分析

```bash
# 运行通信算子分析
python scripts/analyze_communication.py --input <path_to_csv_files> --detail
```

**详细说明：** 请参考 [通信算子分析文档](./reference/communication-operator-analysis.md)

#### 2.2 使用快慢卡检测

```bash
# 检测快慢卡问题
msprof-analyze cluster -d ./profiling_data -m slow_rank -o ./result
```

**详细说明：** 请参考 [快慢卡检测文档](./reference/slow-rank-detection.md)

## 使用建议

**功能选择：**
- 当需要分析通信算子性能和带宽利用率时，使用**通信算子分析**子技能
- 当需要检测集群中性能不一致的rank时，使用**快慢卡检测**子技能
- 若需全面了解通信性能问题，可同时运行两种分析子技能

**分析优化路径：**
1. 首先使用**通信算子分析**识别主要的通信瓶颈点
2. 然后使用**快慢卡检测**确认是否存在卡间性能差异
3. 结合两份分析结果制定综合优化策略

## 依赖要求

- Python 3.8+
- 通信算子分析依赖：pandas, argparse
- 快慢卡检测依赖：msprof_analyze工具

## 注意事项

1. 确保安装了正确版本的快慢卡检测工具
2. 提供完整的profiling数据以获得准确的分析结果
3. 对于大规模集群，可能需要更长的分析时间
4. 结合系统架构和应用场景综合分析结果
5. 不同模型类型（MOE/常规）的通信模式差异较大，需针对性分析