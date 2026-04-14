---
name: profiling-analysis-profiling-communication
description: Skill for analyzing communication performance bottlenecks and detecting slow/fast rank issues in Ascend NPU systems. Use this skill whenever you need to analyze communication efficiency, data transfer bottlenecks, or identify slow/fast rank problems using profiling data.
---

# Profiling 通信瓶颈分析与快慢卡检测 Skill

## 功能概述

该Skill用于分析系统中的通信瓶颈问题和检测快慢卡现象，当主分析Skill检测到通信耗时占比超过10%时自动触发。支持对集群环境下的通信性能进行深入分析，识别影响性能的关键因素。

该Skill采用条件分支的工作流程：

1. **运行一次mstt工具**获取各rank的slowAffectCount值
2. **基于slowAffectCount最大值进行分支判断**：
   - **分支1：slowAffectCount最大值 > 20**
     - 计算Z-score进行统计分析
     - 若Z-score > 0.5 → 判定为存在快慢卡现象，引导使用host快慢卡问题skill
     - 否则 → 建议检查plog日志
   - **分支2：slowAffectCount最大值 ≤ 20**
     - 转到通信算子异常分析分支进行进一步分析

该Skill包含两个主要的分析功能模块：

1. **通信瓶颈分析**：从profiling的"communication"泳道提取通信操作数据，分析通信耗时分布和原因
2. **快慢卡检测**：基于mstt工具和Z-score统计方法检测集群环境下的快慢卡现象

## 支持的数据格式

- profiling数据文件夹

## 功能模块说明

### 1. 通信算子异常分析

处理主分析Skill判定的非快慢卡问题的通信算子异常情况，直接建议用户查看plog日志以进一步分析通信问题。

**主要功能：**
- 接收主分析Skill的非快慢卡通信异常判定
- 提示用户检查plog日志以分析通信异常
- 提供plog日志检查方法和异常处理建议
- 提供通用的通信性能优化建议

**详细文档：** [通信算子异常分析](./reference/communication-operator-analysis.md)

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

#### 2.1 使用通信瓶颈分析

通信瓶颈分析功能通过主分析Skill自动调用，无需手动执行脚本。

**详细说明：** 请参考 [通信瓶颈分析文档](./reference/communication-operator-analysis.md)

#### 2.2 使用快慢卡检测

```bash
# 检测快慢卡问题
msprof-analyze cluster -d ./profiling_data -m slow_rank -o ./result
```

**详细说明：** 请参考 [快慢卡检测文档](./reference/slow-rank-detection.md)

## 使用建议

**自动分支流程：**
该Skill会自动执行以下分析流程，用户无需手动选择功能模块：

1. **自动运行**：当主分析Skill检测到通信耗时占比超过10%时自动触发
2. **一次检测**：运行一次mstt工具获取快慢卡次数
3. **智能分支**：基于slowAffectCount最大值自动选择分析路径
4. **结果输出**：提供针对性的分析结果和建议

**优化分析路径：**
1. 首先运行通信瓶颈分析获取初步结果
2. 根据分析结果中的分支建议采取下一步行动：
   - 若判定为快慢卡现象 → 使用host快慢卡问题skill深入分析
   - 否则 → 检查plog日志确认是否存在hccl异常
3. 结合多维度分析结果制定综合优化策略

## 依赖要求

- 快慢卡检测依赖：msprof_analyze工具

## 注意事项

1. 确保安装了正确版本的快慢卡检测工具
2. 提供完整的profiling数据以获得准确的分析结果
3. 对于大规模集群，可能需要更长的分析时间
4. 结合系统架构和应用场景综合分析结果
5. 当分析结果要求提供plog日志时，请确保收集完整的日志文件