# 性能分析工具指南

本文档提供 Ascend NPU 上 Triton 算子性能分析的完整工具链指南。

## 核心工具概览

| 工具 | 用途 | 适用场景 |
|------|------|---------|
| `msprof op` | 算子级性能采集 | 单个 Triton kernel 性能分析 |
| `triton.testing.do_bench` | 基准测试 | 快速性能对比 |

## msprof op 使用指南

### 基本用法

```bash
# 必须指定 kernel-name 进行算子级性能采集
msprof op --kernel-name={jit_kernel_name} {application}

# 示例：分析名为 "add_kernel" 的 Triton kernel
msprof op --kernel-name=add_kernel python my_triton_script.py
```

### 关键参数

```bash
# 完整参数示例
msprof op \
    --kernel-name=my_kernel \
    --output=/path/to/output \
    --ai-core=on \
    --aic-metrics=Default \
    python my_script.py
```

| 参数 | 说明 | 默认值 |
|------|------|--------|
| `--kernel-name` | **必需**，指定要分析的 kernel 名称 | - |
| `--output` | 输出目录 | 当前目录 |
| `--ai-core` | AI Core 数据采集开关 | on |
| `--aic-metrics` | AI Core 性能指标 | Default |

### AI Core 性能指标

```bash
# 可选的 aic-metrics 值
--aic-metrics=Default               # 整体采集（默认）
--aic-metrics=PipeUtilization       # 流水线利用率
--aic-metrics=ArithmeticUtilization # 算术单元利用率
--aic-metrics=Memory                # 内存访问分析
--aic-metrics=MemoryL0              # L0 内存分析
--aic-metrics=ResourceConflictRatio # 资源冲突率
```

### 性能数据输出

```
output/
├── dump
├── ArithmeticUtilization.csv # Cube和Vector类型的指令耗时和占比
├── L2Cache.csv               # L2 Cache命中率
├── Memory.csv                # UB/L1/L2/主存储器采集内存读写带宽速率
├── MemoryL0.csv              # L0A/L0B/L0C采集内存读写带宽速率
├── MemoryUB.csv              # mte/vector/scalar采集ub读写带宽速率
├── OpBasicInfo.csv           # 算子基础信息，包含算子名称、block dim和耗时等信息
├── PipeUtilization.csv       # 采集计算单元和搬运单元耗时和占比
├── ResourceConflictRatio.csv # UB上的bank group、bank conflict和资源冲突在所有指令中的占比
├── visualize_data.bin        # 可视化数据二进制文件
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
```

## 性能分析工作流

### Step 1: 基准测试

```python
import triton.testing as tt
import torch

def benchmark_triton_kernel():
    """使用 Triton 内置基准测试工具"""
    n = 1024 * 1024
    x = torch.randn(n, device='npu')
    y = torch.empty_like(y)
    
    # 基准测试
    time_ms = tt.do_bench(
        lambda: my_kernel[(triton.cdiv(n, 1024),)](x, y, n, BLOCK_SIZE=1024),
        warmup=25,      # 预热时间 (ms)
        rep=100,        # 测试时间 (ms)
        return_mode="median"
    )
    
    print(f"执行时间: {time_ms:.3f} ms")
    print(f"带宽: {n * 4 * 2 / time_ms / 1e6:.2f} GB/s")  # 假设 FP32
    
    return time_ms
```

### Step 2: 详细性能分析

```python
import subprocess
import os

def profile_with_msprof(script_path, kernel_name, output_dir):
    """使用 msprof op 进行详细性能分析"""
    cmd = f"msprof op --kernel-name={kernel_name} --output={output_dir} python {script_path}"
    
    result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
    
    if result.returncode != 0:
        print(f"错误: {result.stderr}")
        return None
    
    print(f"性能数据已保存到: {output_dir}")
    return output_dir
```

### Step 3: 性能数据分析

```python
import pandas as pd

def analyze_performance_data(output_dir):
    """分析 msprof 输出的性能数据"""
    
    # 读取摘要数据
    summary_path = os.path.join(output_dir, "summary.csv")
    if os.path.exists(summary_path):
        df = pd.read_csv(summary_path)
        print("性能摘要:")
        print(df)
    
    # 读取 AI Core 指标
    ai_core_path = os.path.join(output_dir, "ai_core_metrics")
    if os.path.exists(ai_core_path):
        cube_util = pd.read_csv(os.path.join(ai_core_path, "cube_utilization.csv"))
        vector_util = pd.read_csv(os.path.join(ai_core_path, "vector_utilization.csv"))
        
        print("\nCube 利用率:", cube_util['utilization'].mean())
        print("Vector 利用率:", vector_util['utilization'].mean())
```

## 性能瓶颈诊断

### 瓶颈类型判断

```python
def diagnose_bottleneck(kernel_time_ms, flops, bytes_accessed):
    """诊断性能瓶颈类型"""
    
    # 计算实际性能
    actual_tflops = flops / (kernel_time_ms * 1e-3) / 1e12
    actual_bandwidth = bytes_accessed / (kernel_time_ms * 1e-3) / 1e9
    
    # 硬件峰值（Ascend 910B 示例）
    peak_tflops = 256.0  # FP16
    peak_bandwidth = 1200.0  # GB/s
    
    # 计算利用率
    compute_util = actual_tflops / peak_tflops
    memory_util = actual_bandwidth / peak_bandwidth
    
    # 判断瓶颈类型
    arithmetic_intensity = flops / bytes_accessed  # FLOPs/Byte
    balance_point = peak_tflops / peak_bandwidth  # ~0.21
    
    if arithmetic_intensity < balance_point:
        bottleneck = "Memory-Bound"
        recommendation = "优化内存访问模式，提高数据复用"
    else:
        bottleneck = "Compute-Bound"
        recommendation = "优化计算指令，提高 Cube/Vector 利用率"
    
    return {
        'bottleneck_type': bottleneck,
        'compute_utilization': compute_util,
        'memory_utilization': memory_util,
        'arithmetic_intensity': arithmetic_intensity,
        'recommendation': recommendation
    }
```

### 常见性能问题诊断

| 问题 | 症状 | 诊断方法 | 解决方案 |
|------|------|---------|---------|
| Cube 利用率低 | 矩阵乘法性能差 | 检查 `PipeUtilization.csv` | BLOCK_M/N/K 设为 16 倍数 |
| Vector 利用率低 | 向量操作性能差 | 检查 `PipeUtilization.csv` | 优化向量化，连续访存 |
| 内存带宽低 | 带宽利用率 < 50% | 检查 `Memory.csv` | 连续访存，减少随机访问 |
| UB 溢出 | 编译失败或性能差 | 检查编译日志 | 减小 BLOCK_SIZE，连续访存 |

## 性能报告生成

### 自动化性能报告

```python
import triton.testing as tt

@tt.perf_report(
    tt.Benchmark(
        x_names=['size'],
        x_vals=[2**i for i in range(10, 20)],
        line_arg='backend',
        line_vals=['pytorch', 'triton'],
        line_names=['PyTorch', 'Triton'],
        plot_name='Kernel Performance',
        args={},
        xlabel='Size',
        ylabel='Time (ms)'
    )
)
def benchmark_comparison(size, backend):
    x = torch.randn(size, device='npu')
    y = torch.empty_like(size, device='npu')
    
    if backend == 'pytorch':
        fn = lambda: torch.mul(x, 2, out=y)
    else:
        fn = lambda: my_kernel[(triton.cdiv(size, 1024),)](x, y, size, BLOCK_SIZE=1024)
    
    ms = tt.do_bench(fn, return_mode="median")
    return ms

# 运行并生成报告
benchmark_comparison.run(show_plots=True, print_data=True)
```

## 性能分析最佳实践

### 1. 分层分析策略

```
┌─────────────────────────────────────────┐
│  Level 1: 快速基准测试 (do_bench)        │
│  - 快速定位性能问题                      │
│  - 对比不同实现                          │
└─────────────────────────────────────────┘
              ↓
┌─────────────────────────────────────────┐
│  Level 2: 详细性能分析 (msprof op)       │
│  - 定位具体瓶颈                          │
│  - 分析硬件利用率                        │
└─────────────────────────────────────────┘
```

### 2. 性能分析检查清单

**基准测试前**：
- [ ] 确保数据在 NPU 上（`device='npu'`）
- [ ] 预热运行（warmup）避免冷启动影响
- [ ] 多次测量取中位数或均值

**性能分析时**：
- [ ] 使用 `--kernel-name` 指定目标 kernel
- [ ] 选择合适的 `--aic-metrics`
- [ ] 检查输出目录权限

**分析结果时**：
- [ ] 对比理论峰值和实际性能
- [ ] 判断瓶颈类型（Memory/Compute）
- [ ] 检查硬件利用率指标

### 3. 性能优化迭代流程

```python
def optimization_workflow():
    """性能优化迭代流程"""
    
    # 1. 基准测试
    baseline_time = benchmark_kernel()
    print(f"基线性能: {baseline_time:.3f} ms")
    
    # 2. 性能分析
    profile_with_msprof("my_script.py", "my_kernel", "./prof_data")
    
    # 3. 分析瓶颈
    bottleneck = diagnose_bottleneck(baseline_time, flops, bytes)
    print(f"瓶颈类型: {bottleneck['bottleneck_type']}")
    print(f"建议: {bottleneck['recommendation']}")
    
    # 4. 应用优化
    # ... 修改 kernel 代码 ...
    
    # 5. 验证优化效果
    optimized_time = benchmark_kernel()
    improvement = (baseline_time - optimized_time) / baseline_time * 100
    print(f"性能提升: {improvement:.1f}%")
    
    # 6. 迭代直到满意
    if improvement < 20:
        return optimization_workflow()  # 继续优化
```

## 常见问题

### Q1: 性能数据为空

```bash
# 确保 kernel 实际执行了
# 添加同步点
torch.npu.synchronize()

# 确保输出目录有写权限
chmod 755 /path/to/output
```

### Q2: 性能开销过大

```bash
# 减少采集项
msprof op --kernel-name=my_kernel --ai-core=off python my_script.py

# 使用 sample-based 模式
msprof op --kernel-name=my_kernel --aic-mode=sample-based --aic-freq=10 python my_script.py
```

## 参考资源

- [msprof 官方文档](https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/900beta1/devaids/optool/atlasopdev_16_0082.html)
- [Triton-Ascend Profiling 指南](https://gitcode.com/Ascend/triton-ascend/blob/main/docs/zh/debug_guide/profiling.md)
- [Triton Testing API](https://triton-lang.org/main/python-api/triton.testing.html)
