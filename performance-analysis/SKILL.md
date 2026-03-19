

# Profiling 性能分析 Agent (下发 / 通信 / 算子瓶颈分析)

## 1. Skill 基本信息

| 项目           | 描述                                                         |
| -------------- | ------------------------------------------------------------ |
| **Skill 名称** | Profiling 性能瓶颈分析（下发 / 通信 / 算子）                 |
| **所属领域**   | AI 性能分析、Ascend 平台调优                                 |
| **适用场景**   | 基于 step_trace_time.csv 的 Profiling 数据，自动识别下发 / 通信 / 算子性能瓶颈 |
| **前置依赖**   | Python 3.6+、step_trace_time.csv 文件（支持多文件夹分布）    |
| **触发条件**   | 扫描到 step_trace_time.csv 文件，且各耗时占比超过预设阈值    |

## 2. Skill 功能描述

### 2.1 核心能力

该 Skill 通过解析 Profiling 生成的`step_trace_time.csv`文件，提取`Computing`、`Communication(Not Overlapped)`、`Free`三个核心字段，计算各阶段耗时占比，并根据阈值自动判定性能瓶颈类型，触发对应子分析流程：

**下发问题**：Free 耗时占比 > 20% → 触发 hostbound_analysis_skill.md 分析

**通信问题**：通信耗时占比 > 10% → 触发 communication_analysis_skill.md 分析

**算子（计算）问题**：计算耗时占比 > 85% → 触发 computing_analysis_skill.md 分析

支持多文件夹下`step_trace_time.csv`的批量扫描与独立分析。

### 2.2 输入输出

**输入参数**

| 参数名称   | 类型   | 是否必填 | 描述                                                         |
| ---------- | ------ | -------- | ------------------------------------------------------------ |
| input_path | string | 是       | 输入路径：支持单个`step_trace_time.csv`文件路径，或包含多个该文件的文件夹路径 |

**输出参数**

| 参数名称        | 类型   | 描述                                                         |
| --------------- | ------ | ------------------------------------------------------------ |
| analysis_result | object | 完整性能分析结果：包含耗时指标、瓶颈类型、后续技能路径等核心信息 |
| bottleneck_type | string | 识别到的瓶颈类型：可选值为`hostbound`（下发问题）、`computing`（计算 / 算子问题）、`communication`（通信问题）、`normal`（无明显瓶颈） |
| next_skill      | string | 后续分析技能文档路径：匹配`Hostbound_skill.md`、`Computing_skill.md`、`Communication_skill.md`或空（正常状态） |
| metrics         | object | 耗时占比指标：包含`computing_ratio`（计算耗时占比）、`communication_ratio`（通信耗时占比）、`free_ratio`（空闲耗时占比） |
| file_count      | int    | 实际分析的`step_trace_time.csv`文件数量                      |
| message         | string | 人类可读的分析结论描述                                       |

## 3. 实现步骤

## 3.1 关键步骤详情

### 步骤 1：扫描 CSV 文件

递归遍历指定根目录，收集所有名为step_trace_time.csv的文件路径，支持多层子文件夹分布。

### 步骤 2：解析 CSV 数据

验证文件是否包含Computing、Communication(Not Overlapped)、Free核心字段；
累加各字段的总耗时，跳过非数值数据行，处理文件读取异常。

### 步骤 3：计算耗时占比

总耗时 = Computing总耗时 + Communication总耗时 + Free总耗时 

各阶段占比 = (该阶段总耗时 / 总耗时) × 100%

### 步骤 4：判定瓶颈并触发子 Skill

按阈值判定瓶颈类型（支持多瓶颈同时触发）；
读取对应子 Skill 的 md 文件，输出内容预览；
处理子 Skill 文件不存在的异常。

按优先级顺序执行瓶颈判定，一旦满足条件即确定瓶颈类型并匹配后续技能：

1. 若**空闲耗时占比 > 20%**，判定为**下发问题**，后续技能指向`Hostbound_skill.md`
2. 若**计算耗时占比 > 85%**，判定为**计算 / 算子问题**，后续技能指向`Computing_skill.md`
3. 若**通信耗时占比 > 10%**，判定为**通信问题**，后续技能指向`Communication_skill.md`
4. 未满足以上任一条件，判定为**运行正常**，无后续技能匹配

参考代码见performance_analysis.py

## 3.2 使用示例

#### 输入

```
{
  "input_path": "/home/user/profiling_results"
}
```

#### 输出

```
{
  "skill_name": "profiling-performance-bottleneck-analysis",
  "status": "success",
  "file_count": 3,
  "metrics": {
    "computing_ratio": 72.8,
    "communication_ratio": 6.5,
    "free_ratio": 20.7
  },
  "bottleneck_type": "scheduling",
  "next_skill": "Hostbound_skill.md",
  "message": "性能分析结果：计算耗时占比=72.8%，通信耗时占比=6.5%，空闲耗时占比=20.7%。 空闲耗时占比超过20%，判定为下发问题，请参考Hostbound_skill.md进行分析。"
}
```



## 4. 依赖说明

1. 运行环境：Python 3.8 及以上版本
2. 第三方库：
   - pandas >= 1.3.0（用于 CSV 文件解析与数据计算）
   - numpy >= 1.21.0（用于多文件指标均值计算）

## 5. 注意事项

1. 输入路径支持绝对路径和相对路径，既支持单文件直接解析，也支持文件夹批量解析
2. 多文件分析时采用**均值计算**，避免单文件异常数据导致整体判定偏差
3. 瓶颈判定优先级：**下发问题（空闲占比） > 计算 / 算子问题（计算占比） > 通信问题（通信占比）**，高优先级条件满足后不再执行低优先级判定
4. 运行前需确保`step_trace_time.csv`文件包含`Computing`、`Communication(Not Overlapped)`、`Free`三个核心字段，否则会直接抛出字段缺失异常
5. 技能会自动捕获路径不存在、文件读取失败、字段缺失等异常，并返回清晰的错误信息，便于问题排查