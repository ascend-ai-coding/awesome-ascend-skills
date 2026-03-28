---
name: profiling-analysis-profiling-communication
description: Skill for analyzing communication performance bottlenecks and detecting slow/fast rank issues in Ascend NPU systems. Use this skill whenever you need to analyze communication efficiency, data transfer bottlenecks, or identify slow/fast rank problems using profiling data.
---

# Profiling 通信瓶颈分析与快慢卡检测 Skill

## 功能概述

该Skill用于分析系统中的通信瓶颈问题和检测快慢卡现象，当主分析Skill检测到通信耗时占比超过10%时自动触发。支持对集群环境下的通信性能进行深入分析，识别影响性能的关键因素。

## 分析内容

### 1. 通信时间分析
- 详细分析通信耗时的分布和原因
- 识别主要通信操作类型及其耗时占比
- 分析通信热点和潜在瓶颈

### 2. 数据传输优化
- 识别数据传输过程中的瓶颈点
- 分析数据传输效率和带宽利用率
- 提供数据传输优化建议

### 3. 并行通信分析
- 分析通信与计算的重叠情况
- 评估并行通信的效率
- 识别并行通信中的同步问题

### 4. 快慢卡检测
- 检测集群环境下的快慢卡现象
- 分析快慢卡对整体性能的影响
- 定位导致快慢卡的具体原因

## 工具依赖

### 快慢卡检测工具 (msprof_analyze)

#### 安装步骤
1. 安装依赖
```shell
pip3 install wheel
```

2. 下载源码
```shell
git clone -b master https://gitcode.com/Ascend/mstt
```

3. 编译whl包
```shell
cd mstt/profiler/msprof_analyze
pip3 install -r requirements.txt && python3 setup.py bdist_wheel
```

4. 安装工具
```shell
cd dist
pip3 install ./msprof_analyze-{version}-py3-none-any.whl
```

## 使用方式

### 1. 单独使用

#### 通信瓶颈分析
```python
# 运行通信瓶颈分析
python scripts/analyze_communication.py --input <path_to_csv_files>
```

#### 快慢卡检测
```shell
# 检测快慢卡问题
msprof-analyze cluster -d ./profiling_data -m slow_rank -o ./result
```

### 2. 自动调用

该Skill通常由主分析Skill `/profiling-analysis-profiling-main` 自动调用，当检测到通信耗时占比超过10%时触发。

## 输入数据格式

### 通信瓶颈分析
- CSV格式的profiling数据文件
- 包含通信操作的时间戳和时长信息

### 快慢卡检测
- 完整的profiling数据文件夹
- 支持从Qwen3-32B等模型的profiling数据中检测快慢卡问题

## 输出结果

### 1. 通信瓶颈分析报告
- 通信时间的详细分布
- 主要通信操作的耗时统计
- 通信瓶颈点识别
- 针对性的优化建议

### 2. 快慢卡检测报告
- 各rank的快慢卡影响次数统计
- 快慢卡检测标准（基于Z-score方法）
  - 计算各rank的slowAffectCount的平均值和标准差
  - 使用Z-score > 2作为异常阈值
  - 识别出明显偏离正常水平的慢卡
- 快慢卡对性能的影响评估
- 可能导致快慢卡的原因分析
- 针对性的优化建议

## 快慢卡检测标准

我们采用**结合绝对次数和Z-score统计方法**来判断slowAffectCount是否符合快慢卡标准：

### 1. 绝对次数判定标准
根据Ascend NPU集群的实际运行经验，slowAffectCount的绝对次数判定标准如下：

| slowAffectCount次数范围 | 判定结果 | 建议措施 |
|-----------------------|---------|---------|
| 0 | 正常 | 无需处理 |
| 1-5 | 轻微慢卡 | 监控观察 |
| 6-20 | 中度慢卡 | 分析原因并优化 |
| >20 | 严重慢卡 | 立即处理 |

### 2. 统计分析（Z-score方法）
为了适应不同规模集群和工作负载，同时使用Z-score统计方法：

- **平均值（μ）**：所有rank的slowAffectCount的平均值
- **标准差（σ）**：反映数据的离散程度

### 3. 异常检测
综合绝对次数和Z-score进行判断：

1. **基础判断**：当slowAffectCount > 5时，进入详细分析
2. **Z-score计算**：`Z = (slowAffectCount - μ) / σ`
3. **综合判定**：
   - 当slowAffectCount > 5 且 Z-score > 1.5 时，视为轻微慢卡
   - 当slowAffectCount > 10 且 Z-score > 2 时，视为中度慢卡
   - 当slowAffectCount > 20 或 Z-score > 3 时，视为严重慢卡
4. **特殊情况**：当标准差为0（只有一个rank或所有rank值相同）时，直接使用绝对次数判定

### 4. 结果解释
- **正常**：slowAffectCount = 0，系统运行正常
- **轻微影响**：slowAffectCount 1-5 或 Z-score > 1.5，性能影响较小
- **明显影响**：slowAffectCount 6-20 或 Z-score > 2，性能明显下降
- **严重影响**：slowAffectCount > 20 或 Z-score > 3，性能严重受损，需立即处理

## 案例分析

### 快慢卡检测案例

使用Qwen3-32B模型的profiling数据进行快慢卡检测：

```shell
# 解压profiling数据
unzip Qwen3-32B.zip -d ./qwen_profiling

# 运行快慢卡检测
msprof-analyze cluster -d ./qwen_profiling -m slow_rank -o ./qwen_result
```

检测结果将保存在`./qwen_result/cluster_analysis.db`中，包含各rank的快慢卡影响次数统计。

## 优化建议

### 通信瓶颈优化
1. 增加通信与计算的重叠
2. 优化数据传输格式和大小
3. 使用更高效的通信算法
4. 调整并行度和批处理大小

### 快慢卡问题优化
1. 检查硬件配置是否一致
2. 优化数据分布和负载均衡
3. 调整通信策略和同步机制
4. 升级固件和驱动版本

## 注意事项

1. 确保安装了正确版本的快慢卡检测工具
2. 提供完整的profiling数据以获得准确的分析结果
3. 对于大规模集群，可能需要更长的分析时间
4. 结合系统架构和应用场景综合分析结果