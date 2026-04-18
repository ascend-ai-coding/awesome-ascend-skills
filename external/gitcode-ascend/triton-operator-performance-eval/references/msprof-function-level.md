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
    --trace-level=1 \
    --aic-metrics=Default \
    --profile-iterations=10 \
    --warmup-iterations=5
```

## 关键参数说明

| 参数 | 说明 | 默认值 |
|------|------|--------|
| `--application` | **必需**，指定要分析的应用程序 | - |
| `--output` | 输出目录 | 当前目录 |
| `--trace-level` | 采集级别（0=仅时间，1=包含AI Core指标） | 1 |
| `--aic-metrics` | AI Core 性能指标 | Default |
| `--profile-iterations` | 性能采集迭代次数 | 10 |
| `--warmup-iterations` | 预热迭代次数 | 5 |

## 输出文件结构

```
profiling_result/
├── summary.csv                    # 性能摘要
├── op_summary.csv                 # 算子级性能统计
├── op_detail.csv                  # 算子详细性能数据
├── timeline_trace.json            # 时间线数据（可视化）
├── report.html                    # 可视化报告
└── msprof.log                     # 采集日志
```

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

# 查看可视化报告
firefox profiling_result/report.html
```

## 性能数据分析

### 读取算子性能摘要

```python
import pandas as pd

# 读取算子性能摘要
op_summary = pd.read_csv('./profiling_result/op_summary.csv')

# 查看耗时最长的算子
top_ops = op_summary.nlargest(10, 'total_time_us')
print("耗时最长的算子:")
print(top_ops[['op_name', 'total_time_us', 'call_count']])

# 对比 PyTorch vs Triton 算子
torch_ops = op_summary[op_summary['op_name'].str.contains('aten')]
triton_ops = op_summary[op_summary['op_name'].str.contains('triton')]

print(f"\nPyTorch 算子总耗时: {torch_ops['total_time_us'].sum():.2f} us")
print(f"Triton 算子总耗时: {triton_ops['total_time_us'].sum():.2f} us")
print(f"性能提升: {(1 - triton_ops['total_time_us'].sum() / torch_ops['total_time_us'].sum()) * 100:.1f}%")
```

### 分析时间线数据

```python
import json

# 读取时间线数据
with open('./profiling_result/timeline_trace.json', 'r') as f:
    timeline = json.load(f)

# 分析算子执行顺序和重叠情况
for event in timeline['traceEvents']:
    if event['ph'] == 'X':  # Complete event
        print(f"算子: {event['name']}, 耗时: {event['dur']} us")
```
