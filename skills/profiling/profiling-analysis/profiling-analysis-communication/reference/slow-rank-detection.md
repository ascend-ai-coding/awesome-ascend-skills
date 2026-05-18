# 快慢卡检测子技能

## 功能概述

快慢卡检测功能用于分析集群环境下的快慢卡现象，识别影响性能的异常rank，并提供针对性的优化建议。当检测到通信耗时占比超过10%时，通常会自动触发该子技能进行分析。

## 检测原理

采用**基于阈值和Z-score统计方法**来判断slowAffectCount是否符合快慢卡标准：

### 1. 统计分析（Z-score方法）

为了适应不同规模集群和工作负载，同时使用Z-score统计方法：

- **平均值（μ）**：所有rank的slowAffectCount的平均值
- **标准差（σ）**：反映数据的离散程度

### 2. 异常检测流程

采用简化的条件分支逻辑进行异常检测：

1. **获取快慢卡次数**：使用mstt工具运行一次，获取各rank的slowAffectCount
2. **阈值判断**：以20次为基准进行判断
3. **条件分支处理**：
   - **情况1：slowAffectCount最大值 > 20**
     - 计算Z-score：`Z = (slowAffectCount - μ) / σ`
     - 若Z-score > 0.5 → 判定为存在快慢卡现象，建议使用host快慢卡问题skill进行深入分析
     - 否则 → 建议检查plog日志以进一步分析
   - **情况2：slowAffectCount最大值 ≤ 20**
     - 转到 [通信算子异常分析](./communication-operator-analysis.md) 分支进行进一步分析
     - 通信算子异常分析分支将提供：
       - plog日志检查建议
       - 异常处理指导
       - 通用通信性能优化建议

### 3. 结果解释

- **完整卡数检测**：通过遍历profiling文件夹中的device_x子文件夹，确保所有卡（包括快慢卡次数为0的卡）都被检测和显示
- **快慢卡判定**：当slowAffectCount最大值 > 20且存在Z-score > 2的节点时，判定为存在快慢卡现象
- **正常卡显示**：快慢卡次数为0的卡会被标记为"正常"，确保分析结果包含所有实际存在的卡

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

### 命令行使用

```shell
# 检测快慢卡问题
msprof-analyze cluster -d ./profiling_data -m slow_rank -o ./result
```

### 输入数据格式

- 完整的profiling数据文件夹
- 支持从Qwen3-32B等模型的profiling数据中检测快慢卡问题

### 输出结果

- 各rank的快慢卡影响次数统计
- 快慢卡检测标准（基于Z-score方法）
- 快慢卡对性能的影响评估
- 可能导致快慢卡的原因分析
- 针对性的优化建议

## 案例分析

使用Qwen3-32B模型的profiling数据进行快慢卡检测：

```shell
# 解压profiling数据
unzip Qwen3-32B.zip -d ./qwen_profiling

# 运行快慢卡检测
msprof-analyze cluster -d ./qwen_profiling -m slow_rank -o ./qwen_result
```

检测结果将保存在`./qwen_result/cluster_analysis.db`中，包含各rank的快慢卡影响次数统计。

## 优化建议

1. 检查硬件配置是否一致
2. 优化数据分布和负载均衡
3. 调整通信策略和同步机制
4. 升级固件和驱动版本
5. 检查网络连接和带宽
6. 优化应用层的负载分配策略