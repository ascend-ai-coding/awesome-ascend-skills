---
name: profiling-analysis-profiling-computing
description: 用于分析Ascend NPU系统中计算性能瓶颈的技能，专注于算子效率和计算优化
keywords:
    - profiling
    - 计算瓶颈
    - 算子分析
    - Ascend NPU
---

# Profiling 计算瓶颈分析 Skill

## 功能概述

该Skill用于分析Ascend NPU系统中的计算瓶颈问题，当主分析Skill检测到计算耗时占比超过85%时自动触发。包含两个独立的脚本：

1. **op_high_time_selector.py**：从op_statistic_*.csv、op_summary_*.csv或kernel_details.csv文件中筛选高耗时算子
2. **op_pivot_table_analyzer.py**：基于高耗时算子列表，生成数据透视表

同时提供整合版本**op_perf_pivot_table.py**，用于保持向后兼容性，自动调用上述两个脚本。

## 输入参数

### 高耗时算子筛选脚本 (op_high_time_selector.py)

| 参数名称   | 类型   | 是否必填 | 描述                                                         |
| ---------- | ------ | -------- | ------------------------------------------------------------ |
| input_path | string | 是       | Profiling文件路径，包含PROF_*目录的根路径，例如：./profiling/p-perf-huawei-05_110439_20250728062428118_ascend_pt。当skill由主分析Skill触发时，优先分析主Skill提供的profiling文件，以确保性能分析使用同源数据 |
| output_path | string | 否       | 输出结果目录，用于保存生成的算子列表。若不指定，将在输入路径下自动创建output文件夹 |
| top_n      | int    | 否       | 选取的高耗时算子数量，默认3个                                 |

### 数据透视表分析脚本 (op_pivot_table_analyzer.py)

| 参数名称   | 类型   | 是否必填 | 描述                                                         |
| ---------- | ------ | -------- | ------------------------------------------------------------ |
| input_path | string | 是       | Profiling文件路径，包含PROF_*目录的根路径。当skill由主分析Skill触发时，优先分析主Skill提供的profiling文件，以确保性能分析使用同源数据 |
| output_path | string | 否       | 输出结果目录，用于保存分析报告。若不指定，将在输入路径下自动创建output文件夹 |
| top_n      | int    | 否       | 选取的高耗时算子数量，默认3个                                 |

## 使用方式

### 1. 由主分析Skill自动调用

该Skill通常由主分析Skill `/profiling-analysis-profiling-main` 自动触发。当主Skill检测到计算耗时占比超过85%时，会自动调用该Skill进行深入分析。

### 2. 单独使用（分步骤）

**第一步：筛选高耗时算子**

```bash
# 基本用法
python scripts/op_high_time_selector.py --input-path "./p-perf-huawei-05_110439_20250728062428118_ascend_pt" --output-path "./output"

# 指定分析前5个高耗时算子
python scripts/op_high_time_selector.py --input-path "./p-perf-huawei-05_110439_20250728062428118_ascend_pt" --output-path "./output" --top-n 5
```

**第二步：数据透视表分析与瓶颈定位**

```bash
# 基本用法
python scripts/op_pivot_table_analyzer.py --input-path "./p-perf-huawei-05_110439_20250728062428118_ascend_pt" --output-path "./output"

# 指定分析前5个高耗时算子
python scripts/op_pivot_table_analyzer.py --input-path "./p-perf-huawei-05_110439_20250728062428118_ascend_pt" --output-path "./output" --top-n 5
```

### 3. 单独使用（整合版本）

```bash
# 基本用法
python scripts/op_perf_pivot_table.py --input-path "./p-perf-huawei-05_110439_20250728062428118_ascend_pt" --output-path "./output"

# 指定分析前5个高耗时算子
python scripts/op_perf_pivot_table.py --input-path "./p-perf-huawei-05_110439_20250728062428118_ascend_pt" --output-path "./output" --top-n 5
```

## 分析内容

- **识别高耗时算子**：识别计算耗时最高的N个算子（N=3或由用户指定）
- **统计高耗时算子瓶颈**：根据筛选出的高耗时算子，分析各类型指令占比和瓶颈

## 准备工作

### 数据文件结构

分析所需的op_statistic_*.csv、op_summary_*.csv、kernel_details.csv文件位于以下目录结构中：

```
└─*_ascend_pt
    ├─ASCEND_PROFILER_OUTPUT
    ├─FRAMEWORK
    ├─logs
    └─PROF_*
        ├─device_*
        │  └─data
        ├─host
        │  └─data
        ├─mindstudio_profiler_log
        └─mindstudio_profiler_output  # op_statistic_*.csv、op_summary_*.csv、kernel_details.csv文件位于此目录
```

## 分析步骤详解

### 1. 高耗时算子筛选

根据输入文件类型的不同，采用不同的筛选策略：

#### 1.1 使用op_statistic_*.csv文件筛选

1. **文件读取**：搜索并读取所有op_statistic_*.csv文件。
2. **数据验证**：确保文件包含必要的"OP Type"和"Ratio(%)"列。
3. **筛选高耗时算子**：根据"Ratio(%)"列对算子进行降序排序，选取前N个（N=3或由用户指定）高耗时算子。
4. **读取详细数据**：搜索并读取对应的op_summary_*.csv或kernel_details.csv文件，用于后续分析。

#### 1.2 使用op_summary_*.csv文件筛选

1. **文件读取**：搜索并读取所有op_summary_*.csv或kernel_details.csv文件。
2. **数据验证**：确保文件包含必要的列，如"OP Type"、"Task Duration(us)"、"Input Shapes"等。
3. **算子耗时统计**：基于"Op Type"列对算子分类，统计各个算子的总耗时。
4. **筛选高耗时算子**：根据算子总耗时对算子降序排序，选取前N个（N=3或由用户指定）高耗时算子。

### 2. 数据透视表分析

1. **数据读取**：搜索并读取所有op_summary_*.csv或kernel_details.csv文件。

2. **数据透视分析**：对于每个筛选出的高耗时算子，统计它在各个"Input Shapes"下以下各列的平均值：aic_mac_ratio、aic_saclar_ratio、aic_mte1_ratio、aic_mte2_ratio、aic_fixpipe_ratio、aiv_vec_ratio、aiv_saclar_ratio、aiv_mte2_ratio、aiv_mte3_ratio。

3. **结果输出**：以表格形式输出各个"Op Types"和"Input Shapes"组合中，指定列的均值，并用红色标记每行中的最大值。
    - 输出文件：op_total_duration.csv、op_analysis_details.csv、op_analysis_combined.html
    - 输出路径：用户指定路径，或在输入csv文件所在的文件夹下自动创建output文件夹

## 输出结果

- **算子耗时统计表**：op_total_duration.csv - 按耗时排序的算子列表
- **高耗时算子指令占比透视表**：op_analysis_details.csv - 各算子在不同输入形状下的指令占比
- **算子性能数据透视表**：op_analysis_combined.html - 带可视化标记的算子性能数据透视表

## 参考文档

**官方文档**：
- [性能数据文件参考/op_summary（算子详细信息）](https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/81RC1alpha001/devaids/devtools/profiling/atlasprofiling_16_0067.html)
- [Ascend Profiler用户指南](https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/81RC1alpha001/devaids/devtools/profiling/atlasprofiling_16_0001.html)