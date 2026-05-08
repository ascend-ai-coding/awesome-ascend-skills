# msprof 函数级性能采集详解

本文档提供 msprof 函数级性能采集的详细使用方法和示例。

## 基本用法

```bash
# 基本用法：采集整个 Python 应用的性能数据
msprof --application="python my_script.py" --output=./profiling_result

# 完整参数示例
msprof \
    --application="python my_script.py" \
    --output=./profiling_result \
    --ai-core=on \
    --aic-metrics=Default
```

## 关键参数说明

| 参数 | 说明 | 默认值 |
|------|------|--------|
| `--application` | **必需**，指定要分析的应用程序 | - |
| `--output` | 输出目录 | 当前目录 |
| `--ai-core` | AI Core 数据采集开关 | on |
| `--aic-metrics` | AI Core 性能指标（PipeUtilization, ArithmeticUtilization, Memory 等） | Default |

## 输出文件结构

```
profiling_result/
└── PROF_<id>_<timestamp>/
    ├── device_<N>/                       # 设备侧采集数据
    │   ├── data/                         # 二进制采集原始数据
    │   └── sqlite/                       # SQLite 数据库
    │       ├── ai_core_op_summary.db     # AI Core 算子摘要
    │       └── ...
    ├── host/                             # Host 侧采集数据
    │   ├── data/                         # 二进制采集原始数据
    │   └── sqlite/                       # SQLite 数据库
    │       ├── runtime.db                # Runtime API 数据
    │       └── ...
    ├── mindstudio_profiler_output/       # 导出的可读 CSV/JSON 文件
    │   ├── op_summary_<ts>.csv          # 算子级性能统计
    │   ├── op_statistic_<ts>.csv        # 算子调用统计
    │   ├── api_statistic_<ts>.csv       # API 调用统计
    │   ├── task_time_<ts>.csv           # Task 耗时数据
    │   └── msprof_<ts>.json             # 时间线数据（可用于 MindStudio Insight 可视化）
    ├── mindstudio_profiler_log/          # 采集与分析日志
    └── msprof_<ts>.db                   # 汇总数据库
```

> **注意**：`--application` 模式下 `--export` 参数无效，导出会自动执行。
> CSV 文件位于 `mindstudio_profiler_output/` 子目录中，文件名带时间戳后缀。

## 适用场景

- 对比多个 PyTorch 算子 vs 融合 Triton 算子的性能
- 分析函数级别的性能瓶颈
- 生成可视化性能报告
- 全链路性能分析（Host + Device）

## 完整示例：对比 PyTorch 原生算子 vs Triton 融合算子

### 测试脚本

```python
# my_script.py
import torch
import triton

# PyTorch 原生实现
def torch_implementation(x, y):
    return torch.add(x, y)

# Triton 融合实现
def triton_implementation(x, y):
    return fused_add_relu(x, y)

# 测试数据
x = torch.randn(1024, 1024, device='npu', dtype=torch.float16)
y = torch.randn(1024, 1024, device='npu', dtype=torch.float16)

# 预热
for _ in range(5):
    torch_implementation(x, y)
    triton_implementation(x, y)

# 性能测试
torch.npu.synchronize()
for _ in range(10):
    result_torch = torch_implementation(x, y)

torch.npu.synchronize()
for _ in range(10):
    result_triton = triton_implementation(x, y)

torch.npu.synchronize()
```

### 运行性能分析

```bash
# 运行性能分析
msprof --application="python my_script.py" --output=./profiling_result

# 输出目录中的 PROF_<id>_<timestamp>/mindstudio_profiler_output/ 下包含导出的 CSV 和 JSON
# 使用 MindStudio Insight 打开 msprof_*.json 可视化时间线
```

## 性能数据分析

### 读取算子性能摘要

```python
import os
import glob
import pandas as pd

# 找到导出目录中的 op_summary CSV
output_dir = './profiling_result'
prof_dirs = [d for d in os.listdir(output_dir) if d.startswith('PROF_')]
prof_path = os.path.join(output_dir, prof_dirs[0], 'mindstudio_profiler_output')

op_summary_files = glob.glob(os.path.join(prof_path, 'op_summary_*.csv'))
op_summary = pd.read_csv(op_summary_files[0])

# 查看耗时最长的算子
top_ops = op_summary.nlargest(10, 'Total Time(us)')
print("耗时最长的算子:")
print(top_ops[['Op Name', 'Total Time(us)', 'Count']])

# 对比 PyTorch vs Triton 算子
torch_ops = op_summary[op_summary['Op Name'].str.contains('aten', na=False)]
triton_ops = op_summary[op_summary['Op Name'].str.contains('triton', na=False)]

print(f"\nPyTorch 算子总耗时: {torch_ops['Total Time(us)'].sum():.2f} us")
print(f"Triton 算子总耗时: {triton_ops['Total Time(us)'].sum():.2f} us")
if len(torch_ops) > 0 and len(triton_ops) > 0:
    print(f"性能提升: {(1 - triton_ops['Total Time(us)'].sum() / torch_ops['Total Time(us)'].sum()) * 100:.1f}%")
```

### 分析时间线数据

```python
import os
import glob
import json

output_dir = './profiling_result'
prof_dirs = [d for d in os.listdir(output_dir) if d.startswith('PROF_')]
prof_path = os.path.join(output_dir, prof_dirs[0], 'mindstudio_profiler_output')

# 读取时间线 JSON
timeline_files = glob.glob(os.path.join(prof_path, 'msprof_*.json'))
with open(timeline_files[0], 'r') as f:
    timeline = json.load(f)

# 分析算子执行顺序和耗时
for event in timeline:
    if event.get('ph') == 'X':
        print(f"算子: {event['name']}, 耗时: {event['dur']} us")
```
