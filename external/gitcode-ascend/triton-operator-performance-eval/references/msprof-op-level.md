# msprof op 算子级性能采集详解

本文档提供 msprof op 算子级深度分析的详细使用方法。

## 基本用法

```bash
# 必须指定 kernel-name 进行算子级性能采集
msprof op --kernel-name={jit_kernel_name} {application}

# 示例：分析名为 "add_kernel" 的 Triton kernel
msprof op --kernel-name=add_kernel python my_triton_script.py
```

## 完整参数示例

```bash
msprof op \
    --kernel-name=my_kernel \
    --output=/path/to/output \
    --ai-core=on \
    --aic-metrics=Default \
    python my_script.py
```

## 关键参数说明

| 参数 | 说明 | 默认值 |
|------|------|--------|
| `--kernel-name` | **必需**，指定要分析的 kernel 名称 | - |
| `--output` | 输出目录 | 当前目录 |
| `--ai-core` | AI Core 数据采集开关 | on |
| `--aic-metrics` | AI Core 性能指标 | Default |

## 可选的性能指标

```bash
--aic-metrics=Default               # 整体采集（默认）
--aic-metrics=PipeUtilization       # 流水线利用率
--aic-metrics=ArithmeticUtilization # 算术单元利用率
--aic-metrics=Memory                # 内存访问分析
--aic-metrics=MemoryL0              # L0 内存分析
--aic-metrics=ResourceConflictRatio # 资源冲突率
```

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

## Triton Kernel 名称获取

### 方法：直接从 kernel 函数名获取

```python
import torch
import triton

# 准备数据
x = torch.randn(1024, device='npu')
y = torch.empty_like(x)

# 启动 kernel
grid = lambda meta: (triton.cdiv(1024, meta['BLOCK_SIZE']),)
my_kernel[grid](x, y, 1024, BLOCK_SIZE=256)

# kernel 名称为: my_kernel
# 使用: msprof op --kernel-name=my_kernel python script.py
```
