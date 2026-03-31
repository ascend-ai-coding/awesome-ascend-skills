---
name: profiling-analysis-profiling-hostbound-slow-cards
---

# 快慢卡性能分析

## 功能概述

通过解析MindStudio Insight采集的analysis.db数据库文件，分析各NPU卡的执行时间和通信时间，识别性能表现不一致的快慢卡。

## 支持的数据格式

- MindStudio Insight采集的profiling数据（analysis.db数据库文件）

## 分析原理

基于各卡平均计算时间和通信时间的统计分析，识别卡间性能差异：

### 计算时间维度
- **慢卡**：计算时间超过整体平均+20%标准差（可通过`--speed-threshold`调整）
- **快卡**：计算时间低于整体平均-20%标准差（可通过`--speed-threshold`调整）

### 通信时间维度
- **慢卡**：通信时间短表示该卡计算慢，被其他卡等待
- **快卡**：通信时间长表示该卡计算快，在等待其他卡完成计算

### 综合判定
只要在任一维度表现异常，就认为是快慢卡

## 问题识别

当检测到快慢卡不均衡时，会提供以下详细信息：
- 综合慢卡、快卡和正常卡的具体设备ID列表
- 计算时间维度和通信时间维度的单独分析结果
- 各卡的平均计算时间、通信时间、标准差和任务数
- 整体性能统计（平均计算时间、通信时间、标准差、阈值信息）
- 卡间性能差异百分比

## 使用方式

```bash
python scripts/analyze_hostbound.py --db-path <analysis.db路径> [--output <报告输出路径>] [--speed-threshold <阈值>]
```

### 配置参数

| 参数 | 描述 | 默认值 |
|------|------|--------|
| `--db-path` | Profiling数据库文件路径(analysis.db) | 必填 |
| `--output` | 分析报告输出路径 | card_performance_analysis.xlsx |
| `--speed-threshold` | 快慢卡判定标准差倍数（用于计算慢卡/快卡阈值） | 0.2 (20%) |

### 使用示例

```bash
# 基本快慢卡分析
python scripts/analyze_hostbound.py --db-path "path\ASCEND_PROFILER_OUTPUT\analysis.db"

# 自定义阈值分析
python scripts/analyze_hostbound.py \
  --db-path "path\ASCEND_PROFILER_OUTPUT\analysis.db" \
  --output "performance_analysis.xlsx" \
  --speed-threshold 0.15
```

## 输出结果

生成包含以下内容的Excel报告：

### 分析摘要
- **快慢卡检测结果**：是否存在快慢卡不均衡及具体卡列表
- **核心性能指标**：整体平均计算时间、标准差、慢/快卡阈值

### 卡性能详情
- **设备ID**：NPU卡的设备标识
- **平均计算时间**：卡的平均计算耗时
- **计算时间标准差**：卡内计算时间的稳定性
- **任务数**：分析的任务数量

### 问题识别输出

**快慢卡识别**：
   - 慢卡和快卡的设备ID及具体性能数据
   - 各卡与整体平均值的差异百分比
   - 性能差异的严重程度评估

所有表格默认启用筛选功能，便于进一步分析和比较。