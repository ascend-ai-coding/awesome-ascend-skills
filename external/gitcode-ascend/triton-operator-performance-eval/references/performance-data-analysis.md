# 性能数据分析详解

本文档提供 msprof op 输出数据的详细分析方法。

## 输出文件结构

```
output/
├── ArithmeticUtilization.csv # Cube和Vector类型的指令耗时和占比
├── L2Cache.csv               # L2 Cache命中率
├── Memory.csv                # UB/L1/L2/主存储器读写带宽速率
├── MemoryL0.csv              # L0A/L0B/L0C读写带宽速率
├── MemoryUB.csv              # mte/vector/scalar采集UB读写带宽速率
├── OpBasicInfo.csv           # 算子基础信息（名称、block dim、耗时等）
├── PipeUtilization.csv       # 计算单元和搬运单元耗时和占比
├── ResourceConflictRatio.csv # UB上的bank group、bank conflict占比
└── visualize_data.bin        # 可视化数据二进制文件
```

## 关键指标解读

### 1. 判断瓶颈类型

**Arithmetic Intensity (AI) = FLOPs / Bytes**

- **Memory-Bound**：AI 远低于硬件平衡点（昇腾910约几十到一百多）
- **Compute-Bound**：AI 接近或高于平衡点

**从 Memory.csv 和 ArithmeticUtilization.csv 分析：**

```python
# 计算 AI
flops = ...  # 从 ArithmeticUtilization.csv 获取
bytes_accessed = ...  # 从 Memory.csv 获取
ai = flops / bytes_accessed

# 判断瓶颈
if ai < 50:  # 经验阈值
    print("Memory-Bound: 优化内存访问")
else:
    print("Compute-Bound: 优化计算逻辑")
```

### 2. 内存带宽利用率

**从 Memory.csv 和 MemoryUB.csv 分析：**

```python
# 实际带宽 vs 理论带宽
actual_bandwidth = ...  # 从 Memory.csv 获取
theoretical_bandwidth = 1.2e12  # 1.2 TB/s (A2/A3)

utilization = actual_bandwidth / theoretical_bandwidth

if utilization < 0.3:
    print("低带宽利用率：检查内存访问模式")
```

### 3. 计算单元利用率

**从 PipeUtilization.csv 和 ArithmeticUtilization.csv 分析：**

```python
# Cube 单元利用率（矩阵运算）
cube_utilization = ...  # 从 ArithmeticUtilization.csv

# Vector 单元利用率（向量运算）
vector_utilization = ...  # 从 PipeUtilization.csv

if cube_utilization < 0.5:
    print("Cube 利用率低：检查矩阵分块策略")
```

### 4. UB 冲突分析

**从 ResourceConflictRatio.csv 分析：**

```python
# Bank conflict 比例
bank_conflict_ratio = ...  # 从 ResourceConflictRatio.csv

if bank_conflict_ratio > 0.1:
    print("高 Bank Conflict：优化数据布局")
```

## 完整分析示例

```python
import pandas as pd

def analyze_msprof_op_output(output_dir):
    """分析 msprof op 输出数据"""
    
    # 读取算术利用率
    arith_util = pd.read_csv(f'{output_dir}/ArithmeticUtilization.csv')
    
    # 读取内存数据
    memory = pd.read_csv(f'{output_dir}/Memory.csv')
    
    # 读取流水线利用率
    pipe_util = pd.read_csv(f'{output_dir}/PipeUtilization.csv')
    
    # 读取资源冲突率
    resource_conflict = pd.read_csv(f'{output_dir}/ResourceConflictRatio.csv')
    
    # 计算 AI
    total_flops = arith_util['flops'].sum()
    total_bytes = memory['bytes_accessed'].sum()
    ai = total_flops / total_bytes
    
    print(f"算术强度 (AI): {ai:.2f} FLOPs/Byte")
    
    # 判断瓶颈
    if ai < 50:
        print("瓶颈类型: Memory-Bound")
        print("优化方向: 优化内存访问模式")
    else:
        print("瓶颈类型: Compute-Bound")
        print("优化方向: 优化计算逻辑")
    
    # 计算带宽利用率
    actual_bandwidth = memory['bandwidth'].mean()
    theoretical_bandwidth = 1.2e12  # 1.2 TB/s
    bandwidth_util = actual_bandwidth / theoretical_bandwidth
    
    print(f"\n带宽利用率: {bandwidth_util * 100:.1f}%")
    
    # Cube 利用率
    cube_util = arith_util[arith_util['unit'] == 'Cube']['utilization'].mean()
    print(f"Cube 利用率: {cube_util * 100:.1f}%")
    
    # Bank conflict
    bank_conflict = resource_conflict['bank_conflict_ratio'].mean()
    print(f"Bank Conflict 比例: {bank_conflict * 100:.1f}%")
    
    # 诊断建议
    if bandwidth_util < 0.3:
        print("\n⚠️  低带宽利用率：检查内存访问模式")
    
    if cube_util < 0.5:
        print("⚠️  Cube 利用率低：检查矩阵分块策略")
    
    if bank_conflict > 0.1:
        print("⚠️  高 Bank Conflict：优化数据块大小")

# 使用示例
analyze_msprof_op_output('./profiling_result')
```
