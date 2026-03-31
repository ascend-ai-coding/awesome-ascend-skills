# 昇腾NPU最佳实践指南

## 1. 昇腾NPU架构概述

### 1.1 硬件架构
- **昇腾910**: 华为最强AI处理器，7nm工艺，32GB HBM
- **昇腾310**: 面向推理场景，8GB LPDDR4
- **架构特点**: 采用达芬奇架构，集成Vector/Cube两种计算单元

### 1.2 软件栈
```
┌─────────────────────────────────────┐
│         应用层 (PyTorch/TF)          │
├─────────────────────────────────────┤
│      PyTorch NPU Adapter (torch_npu) │
├─────────────────────────────────────┤
│      Ascend Transformer Backend (ATB) │
├─────────────────────────────────────┤
│         CANN (Compute Architecture)   │
├─────────────────────────────────────┤
│         Driver & Firmware            │
└─────────────────────────────────────┘
```

## 2. 环境配置

### 2.1 CANN安装
```bash
# 1. 下载CANN包
wget https://ascend-repo.obs.cn-east-2.myhuaweicloud.com/CANN/7.0/Ascend-cann-toolkit_7.0.RC1_linux-x86_64.run

# 2. 安装
bash Ascend-cann-toolkit_7.0.RC1_linux-x86_64.run --install

# 3. 配置环境变量
source /usr/local/Ascend/ascend-toolkit/set_env.sh
```

### 2.2 PyTorch NPU支持
```bash
# 安装torch和torch_npu
pip install torch torch_npu

# 或从源码编译
git clone https://gitee.com/ascend/pytorch.git
cd pytorch
git submodule update --init
python setup.py install
```

### 2.3 验证安装
```python
import torch
import torch_npu

# 检查NPU可用性
print(f"NPU available: {torch_npu.is_npu_available()}")

# 获取设备信息
print(f"NPU device count: {torch_npu.device_count()}")
print(f"Current device: {torch_npu.current_device()}")

# 创建张量
x = torch.randn(2, 3).npu()
print(f"Tensor device: {x.device}")
```

## 3. API映射表

### 3.1 设备管理

| GPU (CUDA) | NPU (Ascend) | 说明 |
|-----------|-------------|------|
| `torch.device('cuda')` | `torch.device('npu')` | 设备定义 |
| `torch.cuda.device_count()` | `torch_npu.device_count()` | 设备数量 |
| `torch.cuda.current_device()` | `torch_npu.current_device()` | 当前设备 |
| `torch.cuda.set_device(i)` | `torch_npu.set_device(i)` | 设置设备 |
| `torch.cuda.device_name(i)` | `torch_npu.get_device_name(i)` | 设备名称 |

### 3.2 张量操作

| GPU (CUDA) | NPU (Ascend) | 说明 |
|-----------|-------------|------|
| `tensor.cuda()` | `tensor.npu()` | 移到NPU |
| `tensor.cuda(non_blocking=True)` | `tensor.npu(non_blocking=True)` | 异步传输 |
| `torch.cuda.FloatTensor` | `torch_npu.FloatTensor` | 直接创建 |
| `torch.zeros_like(tensor)` | `torch.zeros_like(tensor)` | 兼容（自动适配）|

### 3.3 内存管理

| GPU (CUDA) | NPU (Ascend) | 说明 |
|-----------|-------------|------|
| `torch.cuda.empty_cache()` | `torch_npu.empty_cache()` | 清空缓存 |
| `torch.cuda.memory_allocated()` | `torch_npu.memory_allocated()` | 已分配内存 |
| `torch.cuda.memory_reserved()` | `torch_npu.memory_reserved()` | 保留内存 |
| `torch.cuda.synchronize()` | `torch_npu.synchronize()` | 同步 |

### 3.4 流和事件

| GPU (CUDA) | NPU (Ascend) | 说明 |
|-----------|-------------|------|
| `torch.cuda.Stream()` | `torch_npu.Stream()` | 创建流 |
| `torch.cuda.Event()` | `torch_npu.Event()` | 创建事件 |
| `stream.synchronize()` | `stream.synchronize()` | 流同步 |

### 3.5 随机数

| GPU (CUDA) | NPU (Ascend) | 说明 |
|-----------|-------------|------|
| `torch.cuda.manual_seed()` | `torch_npu.manual_seed()` | 设置种子 |
| `torch.cuda.manual_seed_all()` | `torch_npu.manual_seed_all()` | 所有设备种子 |

### 3.6 AMP (自动混合精度)

```python
# GPU AMP
from torch.cuda.amp import autocast, GradScaler

scaler = GradScaler()
with autocast():
    output = model(input)
scaler.scale(loss).backward()

# NPU AMP
from torch_npu.npu import npu_format_cast

# 使用NPU AMP
scaler = torch_npu.amp.GradScaler()
with torch_npu.amp.autocast():
    output = model(input)
```

## 4. 常用算子适配

### 4.1 Flash Attention

```python
# GPU: 使用flash-attn库
from flash_attn import flash_attn_func
output = flash_attn_func(q, k, v)

# NPU: 使用torch_npu或ATB
# 方案1: torch_npu
output = torch_npu.npu_flash_attention(q, k, v)

# 方案2: ATB
from ascend_transformer_bindings import flash_attention
output = flash_attention(q, k, v)
```

### 4.2 Transformer优化

```python
# 使用ATB (Ascend Transformer Backend)
from ascend_transformer_bindings import (
    PagedAttention,
    Linear,
    LayerNorm,
    RotaryEmbedding
)

# Paged Attention (替代Flash Attention用于LLM)
attn_output = PagedAttention(
    q, k, v,
    block_size=16,
    num_kv_heads=8
)
```

### 4.3 矩阵乘法

```python
# GPU: cuBLAS
result = torch.matmul(a, b)

# NPU: 自动兼容（底层调用aclBLAS）
result = torch.matmul(a, b)  # 兼容

# 或显式使用NPU
result = torch_npu.matmul(a, b)
```

## 5. 性能优化

### 5.1 算子融合
```python
# 使用ATC进行算子融合
# 1. 导出模型为.onnx
torch.onnx.export(model, input, "model.onnx")

# 2. 使用ATC融合
atc --model=model.onnx --framework=5 --output=model_fused \
    --soc_version=Ascend310 --insert_op_conf=aipp.config
```

### 5.2 内存优化
```python
# 1. 使用梯度检查点
from torch.utils.checkpoint import checkpoint

# 2. 动态形状优化
torch._inductor.config.dynamic_shapes = True

# 3. 内存复用
torch.npu.set_option("opt_level", 2)
torch.npu.set_option("memory_fraction", 0.8)
```

### 5.3 编译优化
```bash
# 使用TorchScript编译
torch.jit.trace(model, input)
torch.jit.freeze(model)

# 使用torch_npu优化
model = model.to(npu)
model = torch_npu.optimize(model, dtype=torch.float16)
```

## 6. 调试与诊断

### 6.1 常用调试命令
```bash
# 1. 检查NPU状态
npu-smi info

# 2. 查看NPU进程
npu-smi ps

# 3. 检查内存使用
npu-smi meminfo

# 4. 设置日志级别
export ASCEND_S_LOG=3
```

### 6.2 Python调试
```python
import torch_npu

# 1. 启用详细日志
torch_npu.set_option("debug_level", 1)

# 2. 查看算子执行信息
with torch_npu.profiler.profile(
    activities=[
        torch_npu.profiler.ProfilerActivity.CPU,
        torch_npu.profiler.ProfilerActivity.NPU
    ]
) as prof:
    model(input)

print(prof.key_averages().table())
```

### 6.3 常见错误处理

| 错误码 | 错误描述 | 解决方案 |
|-------|---------|---------|
| E00100 | 内存分配失败 | 减小batch_size，清理缓存 |
| E00200 | 算子不支持 | 使用ATB替代或自研算子 |
| E00300 | 精度溢出 | 检查数据类型，降低精度 |
| E00400 | 形状不匹配 | 检查输入维度 |

## 7. 最佳实践总结

1. **渐进式迁移**: 先从简单的张量操作开始，逐步迁移复杂算子
2. **保持兼容性**: 使用设备检测，保留CUDA回退
3. **性能优先**: 优先使用ATB优化库，减少Python-C++边界
4. **充分测试**: 验证精度和性能，确保与GPU等效
5. **持续监控**: 使用profiler持续优化性能
