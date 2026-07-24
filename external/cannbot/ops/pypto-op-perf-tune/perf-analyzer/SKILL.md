---
name: external-cannbot-ops-pypto-op-perf-tune-perf-analyzer
description: 分析 PyPTO 算子的性能指标。用于分析 PyPTO 算子的性能指标，从性能数据文件中提取关键指标，计算性能评级，并提供性能瓶颈分析和优化建议。
  当需要分析 PyPTO 算子性能数据、计算性能评级或定位性能瓶颈时使用此技能。
original-name: perf-analyzer
synced-from: https://gitcode.com/cann/cannbot-skills
synced-date: '2026-07-24'
synced-commit: bff73845607ac78808e2af3e0014d7eb72094ef3
license: UNKNOWN
---

# PyPTO 性能指标分析技能

## 概述

此技能专注于分析 PyPTO 算子的性能指标，从性能数据文件中提取关键指标，计算性能评级，并提供性能瓶颈分析和优化建议。

## 使用场景

当用户需要：
- 分析已生成的性能数据文件
- 评估算子性能表现
- 识别性能瓶颈
- 获取性能优化建议

## 工作流程

## 步骤 0：数据提取方法

使用技能中的性能分析脚本自动生成报告：

**重要提示**
- 性能分析脚本位置：`pypto-op-perf-tune/perf-analyzer/scripts/analyze_perf.py`
- 性能数据目录位于**执行算子命令时的工作目录**下的 `output/output_*/` 目录中

### 使用方法

```bash
# 传入完整的 output 目录路径（推荐使用绝对路径）
python3 <脚本路径>/analyze_perf.py <output_dir>

# 示例1：在算子目录下执行的，output 就在算子目录下
python3 pypto/.agents/skills/pypto-op-perf-tune/perf-analyzer/scripts/analyze_perf.py custom/operator_name/output/output_20260304_171658_543682_529508

# 示例2：使用绝对路径
python3 pypto/.agents/skills/pypto-op-perf-tune/perf-analyzer/scripts/analyze_perf.py /home/user/project/custom/operator_name/output/output_20260304_171658_543682_529508
```

**⚠️ 路径说明**：
- `output_dir` 应为包含 `bubble_analysis.log` 的具体 output 时间戳目录
- 如果不确定 output 目录位置，先使用 `find <搜索范围> -name "bubble_analysis.log"` 查找
- 脚本支持自动递归查找：如果传入的目录下没有 `bubble_analysis.log`，脚本会自动在子目录中搜索 `output_*/bubble_analysis.log`

### 步骤 1：定位性能数据文件

**⚠️ 性能数据文件位置取决于执行算子命令时的工作目录**：

| 执行场景 | output 目录位置 |
|---------|----------------|
| 在算子目录下执行 `python3 op.py --run-mode npu` | `<算子目录>/output/output_*/` |
| 在项目根目录执行 `python3 custom/op/op.py --run-mode npu` | `./output/output_*/` |

性能数据文件位于 `output/output_<时间戳>/` 目录下：

- `bubble_analysis.log` - 气泡分析报告
- `merged_swimlane.json` - 泳道图数据文件
- `machine_runtime_operator_trace.json` - 性能追踪文件

**查找最新输出目录**（需在对应的工作目录下执行）：
```bash
# 方法1：直接列出 output 下的目录
ls -lt output/ | head -n 2

# 方法2：不确定位置时，从项目根目录搜索
find . -name "bubble_analysis.log" -type f
```

### 步骤 2：提取核心性能指标

从 `bubble_analysis.log` 中提取以下指标：

**AIC 核心指标（AI Cube核心）：**
- Core Total Work Time: 核心总工作时间
- Total Wait Time: 总等待时间
- Wait Schedule Time: 等待调度时间（气泡）
- Wait Predecessor Time: 等待前驱时间

**AIV 核心指标（AI Vector核心）：**
- 同上指标

**算子实际执行时间：**
所有核心中 Core Total Work Time 的最大值

**AicoreTime(核心实际工作时间)：**
AicoreTime = 核心总工作时间 - 总等待时间


### 步骤 3：计算性能指标

#### 3.1 核心利用率

```
核心利用率 = AicoreTime / (AicoreTime + 等待总时间) × 100%
```

#### 3.2 气泡率

```
气泡率 = 等待调度时间 / (AicoreTime + 等待调度时间) × 100%
```

#### 3.3 平均核心利用率

```
平均核心利用率 = 所有核心核心利用率之和 / 核心数量
```

#### 3.4 平均气泡率

```
平均气泡率 = 所有核心气泡率之和 / 核心数量
```

### 步骤 4：性能评级

根据以下标准进行性能评级：

| 评分 | 核心利用率 | 气泡率 | 内存效率 | 控制开销占比 |
|------|-----------|--------|---------|-------------|
| ⭐⭐⭐⭐⭐ | >90% | <2% | >70% | <30% |
| ⭐⭐⭐⭐ | >80% | <5% | >50% | <50% |
| ⭐⭐⭐ | >60% | <10% | >30% | <70% |
| ⭐⭐ | >50% | <20% | >20% | <80% |
| ⭐ | <50% | >20% | <20% | >80% |

### 步骤 5：性能瓶颈分析

#### 5.1 气泡率分析

**高气泡率（>20%）可能原因：**
- 任务粒度过小
- 调度策略不当
- stitch 参数过小

**分析要点：**
- 识别气泡率最高的核心
- 检查 Top 3 tasks in waiting schedule time
- 分析任务分布情况

#### 5.2 核心利用率分析

**低核心利用率（<50%）可能原因：**
- 等待时间过长
- 任务调度不均衡
- 任务粒度过小
- 任务依赖过多

**分析要点：**
- 检查 Total Wait Time 是否过高
- 检查 Wait Predecessor Time 是否过长
- 检查核心间工作时间差异

#### 5.3 核心负载均衡分析

**分析方法：**
- 计算所有AicoreTime的标准差
- 识别工作时间最大和最小的核心
- 分析差异原因

#### 5.4 任务依赖分析

**分析方法：**
- 检查 Wait Predecessor Time
- 识别依赖关系复杂的任务
- 分析任务执行顺序

### 步骤 6：生成性能分析报告

性能分析报告模板："templates/performance_report_template.md"
性能分析报告应包含以下内容：

#### 6.1 核心性能指标

- 算子实际执行时间（所有核心最大工作时间）
- 各核心的AicoreTime
- 各核心的气泡时间
- 各核心的等待时间

#### 6.2 性能指标统计

- 平均核心利用率
- 平均气泡率
- 最大/最小核心利用率
- 最大/最小气泡率
- 核心负载均衡度

#### 6.3 性能评级

- 核心利用率评级
- 气泡率评级
- 综合评级

#### 6.4 性能瓶颈分析

- 主要瓶颈识别
- 瓶颈原因分析
- 影响程度评估


## 性能分析报告模板

```markdown
## {算子名称} 性能分析报告

### 1. 核心性能指标

**算子实际执行时间:** {最大工作时间} us

**AIC AicoreTime:**
- AIC_1: {时间} us
- AIC_2: {时间} us
- ...

**AIV AicoreTime:**
- AIV_1: {时间} us
- AIV_2: {时间} us
- ...

### 2. 核心利用率分析

| 核心 | AIcoreTime | 等待时间 | 核心利用率 |
|------|---------|---------|-----------|
| AIC_1 | {时间} | {时间} | {利用率}% |
| ...

**平均核心利用率: {平均利用率}%**

### 3. 气泡分析

| 核心 | 气泡率 | 等待调度时间 |
|------|--------|------------|
| AIC_1 | {气泡率}% | {时间} us |
| ...

**平均气泡率: {平均气泡率}%**
**最大气泡率: {最大气泡率}% ({核心名称})**

### 4. 性能评级

| 指标 | 当前值 | 目标值(⭐⭐⭐⭐⭐) | 评级 |
|------|--------|----------------|------|
| 核心利用率 | {当前值}% | >98% | {星级} |
| 气泡率 | {当前值}% | <2% | {星级} |

**综合评级: {星级} ({评级描述})**

### 5. 性能瓶颈分析

1. **{瓶颈1名称}**
   - 描述: {描述}
   - 影响: {影响程度}

2. **{瓶颈2名称}**
   - 描述: {描述}
   - 影响: {影响程度}

### 6. 性能优化建议

**高优先级优化:**

1. **{优化1}**
   - 描述: {描述}
   - 代码: {代码示例}

**中优先级优化:**

2. **{优化2}**
   - 描述: {描述}
   - 代码: {代码示例}

### 7. 性能数据文件位置

- 泳道图: {泳道图路径}
- 气泡分析: {气泡分析路径}
- 性能追踪: {性能追踪路径}

可在 https://ui.perfetto.dev/ 上传泳道图文件进行可视化分析。
```
