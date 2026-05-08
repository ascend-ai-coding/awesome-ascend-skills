# 性能数据分析详解

本文档提供 msprof op 输出数据的详细分析方法。

## 输出文件结构

```
output/
└── OPPROF_<timestamp>/
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

## 关键 CSV 列名参考

### ArithmeticUtilization.csv 关键列

| 列名 | 说明 |
|------|------|
| `aic_cube_fops` | Cube 浮点运算次数 |
| `aic_cube_ratio` | Cube 单元占比 |
| `aiv_vec_ratio` | Vector 单元占比 |
| `aiv_vec_fp32_ratio` | Vector FP32 占比 |
| `aiv_vec_fp16_ratio` | Vector FP16 占比 |
| `aiv_vec_fops` | Vector 浮点运算次数 |

### Memory.csv 关键列

| 列名 | 说明 |
|------|------|
| `aiv_gm_to_ub_bw(GB/s)` | GM → UB 读带宽 |
| `aiv_ub_to_gm_bw(GB/s)` | UB → GM 写带宽 |
| `aiv_main_mem_read_bw(GB/s)` | 主存读带宽 |
| `aiv_main_mem_write_bw(GB/s)` | 主存写带宽 |
| `GM_to_UB_datas(KB)` | GM→UB 数据量 |
| `UB_to_GM_datas(KB)` | UB→GM 数据量 |
| `GM_to_UB_bw_usage_rate(%)` | GM→UB 带宽利用率 |
| `UB_to_GM_bw_usage_rate(%)` | UB→GM 带宽利用率 |

### ResourceConflictRatio.csv 关键列

| 列名 | 说明 |
|------|------|
| `aiv_vec_total_cflt_ratio` | 总冲突率 |
| `aiv_vec_bankgroup_cflt_ratio` | Bank Group 冲突率 |
| `aiv_vec_bank_cflt_ratio` | Bank 冲突率 |
| `aiv_vec_resc_cflt_ratio` | 资源冲突率 |

### OpBasicInfo.csv 关键列

| 列名 | 说明 |
|------|------|
| `Op Name` | 算子名称 |
| `Op Type` | 算子类型（vector/cube） |
| `Task Duration(us)` | Task 耗时（微秒） |
| `Block Dim` | Block 维度 |

## 关键指标解读

### 1. 判断瓶颈类型

**Arithmetic Intensity (AI) = FLOPs / Bytes**

- **Memory-Bound**：AI 远低于硬件平衡点（昇腾910约几十到一百多）
- **Compute-Bound**：AI 接近或高于平衡点

**从 Memory.csv 和 ArithmeticUtilization.csv 分析：**

```python
import pandas as pd

arith = pd.read_csv(f'{output_dir}/ArithmeticUtilization.csv')
memory = pd.read_csv(f'{output_dir}/Memory.csv')

# 计算 FLOPs（Cube + Vector）
total_flops = arith['aic_cube_fops'].sum() + arith['aiv_vec_fops'].sum()

# 计算数据量（GM→UB + UB→GM，单位 KB）
total_bytes_kb = memory['GM_to_UB_datas(KB)'].sum() + memory['UB_to_GM_datas(KB)'].sum()
total_bytes = total_bytes_kb * 1024  # 转 Byte

ai = total_flops / total_bytes if total_bytes > 0 else 0

# 判断瓶颈
if ai < 50:  # 经验阈值
    print("Memory-Bound: 优化内存访问")
else:
    print("Compute-Bound: 优化计算逻辑")
```

### 2. 内存带宽利用率

**从 Memory.csv 分析：**

```python
# 实际带宽 vs 理论带宽
# 注意：列名带单位后缀 (GB/s)
gm_to_ub_bw = memory['aiv_gm_to_ub_bw(GB/s)'].mean()
ub_to_gm_bw = memory['aiv_ub_to_gm_bw(GB/s)'].mean()

# 带宽利用率（百分比列）
gm_to_ub_util = memory['GM_to_UB_bw_usage_rate(%)'].mean()
ub_to_gm_util = memory['UB_to_GM_bw_usage_rate(%)'].mean()

print(f"GM→UB 带宽: {gm_to_ub_bw:.2f} GB/s, 利用率: {gm_to_ub_util:.1f}%")
print(f"UB→GM 带宽: {ub_to_gm_bw:.2f} GB/s, 利用率: {ub_to_gm_util:.1f}%")

if gm_to_ub_util < 30:
    print("低带宽利用率：检查内存访问模式")
```

### 3. 计算单元利用率

**从 ArithmeticUtilization.csv 分析：**

```python
# Cube 单元利用率（矩阵运算），纯 vector 算子可能为 NaN
cube_ratio = arith['aic_cube_ratio'].mean(skipna=True)

# Vector 单元利用率（向量运算）
vec_ratio = arith['aiv_vec_ratio'].mean(skipna=True)

if pd.notna(cube_ratio):
    print(f"Cube 占比: {cube_ratio * 100:.1f}%")
    if cube_ratio < 0.5:
        print("Cube 利用率低：检查矩阵分块策略")
else:
    print("Cube 占比: N/A (纯 vector 算子)")
```

### 4. UB 冲突分析

**从 ResourceConflictRatio.csv 分析：**

```python
resource_conflict = pd.read_csv(f'{output_dir}/ResourceConflictRatio.csv')

# Bank conflict 比例
bankgroup_cflt = resource_conflict['aiv_vec_bankgroup_cflt_ratio'].mean()
bank_cflt = resource_conflict['aiv_vec_bank_cflt_ratio'].mean()

print(f"Bank Group Conflict: {bankgroup_cflt * 100:.1f}%")
print(f"Bank Conflict: {bank_cflt * 100:.1f}%")

if bankgroup_cflt > 0.1:
    print("高 Bank Conflict：优化数据布局")
```

## 完整分析示例

```python
import os
import glob
import pandas as pd

def analyze_msprof_op_output(output_root):
    """分析 msprof op 输出数据"""
    
    # 定位 OPPROF 目录
    prof_dirs = glob.glob(os.path.join(output_root, 'OPPROF_*'))
    if not prof_dirs:
        print(f"未找到 OPPROF 目录于 {output_root}")
        return
    output_dir = prof_dirs[0]
    
    # 读取算子基础信息
    op_info = pd.read_csv(f'{output_dir}/OpBasicInfo.csv')
    print(f"算子: {op_info['Op Name'].iloc[0]}")
    print(f"类型: {op_info['Op Type'].iloc[0]}")
    print(f"耗时: {op_info['Task Duration(us)'].iloc[0]:.2f} us")
    print(f"Block Dim: {op_info['Block Dim'].iloc[0]}")
    
    # 读取算术利用率
    arith = pd.read_csv(f'{output_dir}/ArithmeticUtilization.csv')
    
    # 读取内存数据
    memory = pd.read_csv(f'{output_dir}/Memory.csv')
    
    # 读取资源冲突率
    resource_conflict = pd.read_csv(f'{output_dir}/ResourceConflictRatio.csv')
    
    # 计算 AI
    total_flops = arith['aic_cube_fops'].sum() + arith['aiv_vec_fops'].sum()
    total_bytes_kb = memory['GM_to_UB_datas(KB)'].sum() + memory['UB_to_GM_datas(KB)'].sum()
    total_bytes = total_bytes_kb * 1024
    ai = total_flops / total_bytes if total_bytes > 0 else 0
    
    print(f"\n算术强度 (AI): {ai:.2f} FLOPs/Byte")
    
    # 判断瓶颈
    if ai < 50:
        print("瓶颈类型: Memory-Bound")
        print("优化方向: 优化内存访问模式")
    else:
        print("瓶颈类型: Compute-Bound")
        print("优化方向: 优化计算逻辑")
    
    # 带宽利用率
    gm_to_ub_util = memory['GM_to_UB_bw_usage_rate(%)'].mean()
    ub_to_gm_util = memory['UB_to_GM_bw_usage_rate(%)'].mean()
    print(f"\nGM→UB 带宽利用率: {gm_to_ub_util:.1f}%")
    print(f"UB→GM 带宽利用率: {ub_to_gm_util:.1f}%")
    
    # Cube 利用率（纯 vector 算子可能全为 NaN）
    cube_ratio = arith['aic_cube_ratio'].mean(skipna=True)
    vec_ratio = arith['aiv_vec_ratio'].mean(skipna=True)
    print(f"\nCube 占比: {cube_ratio * 100:.1f}%" if pd.notna(cube_ratio) else "\nCube 占比: N/A (纯 vector 算子)")
    print(f"Vector 占比: {vec_ratio * 100:.1f}%")
    
    # Bank conflict
    bankgroup_cflt = resource_conflict['aiv_vec_bankgroup_cflt_ratio'].mean(skipna=True)
    print(f"\nBank Group Conflict: {bankgroup_cflt * 100:.1f}%")
    
    # 诊断建议
    if gm_to_ub_util < 30:
        print("\n⚠️  低带宽利用率：检查内存访问模式")
    
    if pd.notna(cube_ratio) and cube_ratio < 0.5:
        print("⚠️  Cube 利用率低：检查矩阵分块策略")
    
    if bankgroup_cflt > 0.1:
        print("⚠️  高 Bank Conflict：优化数据块大小")

# 使用示例
analyze_msprof_op_output('./op_profiling_result')
```
