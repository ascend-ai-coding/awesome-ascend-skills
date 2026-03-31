# CANN迁移指南 - 从CUDA到Ascend NPU

## 1. 概述

本文档提供从NVIDIA CUDA迁移到华为昇腾CANN的详细指南。涵盖常见场景的迁移路径、最佳实践和代码示例。

## 2. 核心概念映射

### 2.1 计算架构

| CUDA概念 | CANN概念 | 说明 |
|---------|---------|------|
| CUDA Driver | NPU Driver | 硬件驱动层 |
| cuBLAS | ACLBLAS | BLAS线性代数库 |
| cuDNN | ACLE | CNN引擎 |
| cuFFT | ACFFT | FFT库 |
| cuSolver | ACLSolver | 稀疏矩阵求解器 |
| TensorRT | ATC/ATB | 模型优化与推理 |
| Nsight | MindX DL Profiler | 性能分析工具 |

### 2.2 编程模型

```python
# CUDA编程模型
import torch
device = torch.device('cuda:0')
x = torch.randn(100, 100).to(device)
y = torch.matmul(x, x)
y.cpu()

# CANN编程模型（兼容模式）
import torch
import torch_npu
device = torch.device('npu:0')
x = torch.randn(100, 100).to(device)
y = torch.matmul(x, x)
y.cpu()
```

## 3. 迁移步骤

### 3.1 环境准备

```bash
# 1. 安装CANN
# 下载: https://www.hiascend.com/software/aiengine
bash install.sh --all

# 2. 设置环境变量
source /usr/local/Ascend/ascend-toolkit/set_env.sh

# 3. 安装torch_npu
pip install torch_npu

# 4. 验证
python -c "import torch; print(torch.npu.is_available())"
```

### 3.2 代码迁移检查清单

- [ ] 替换设备字符串 `'cuda'` → `'npu'`
- [ ] 替换张量方法 `.cuda()` → `.npu()`
- [ ] 检查CUDA特定API调用
- [ ] 替换torch.cuda模块为torch_npu
- [ ] 验证第三方库NPU支持
- [ ] 测试精度和性能

## 4. 常见迁移场景

### 4.1 张量创建与移动

```python
# CUDA
x = torch.randn(3, 4, device='cuda')
y = torch.zeros(5, 6, device='cuda:0')
z = torch.ones(2, 3).cuda()

# NPU
x = torch.randn(3, 4, device='npu')
y = torch.zeros(5, 6, device='npu:0')
z = torch.ones(2, 3).npu()
```

### 4.2 模型迁移

```python
# CUDA
model = MyModel().cuda()
output = model(input)

# NPU
model = MyModel().npu()  # 方式1
model = MyModel().to('npu')  # 方式2
output = model(input)  # 自动在NPU执行
```

### 4.3 数据加载

```python
# CUDA
from torch.utils.data import DataLoader
loader = DataLoader(dataset, batch_size=32, shuffle=True)

# NPU
# 相同代码，DataLoader会自动处理NPU张量
from torch.utils.data import DataLoader
loader = DataLoader(dataset, batch_size=32, shuffle=True)
```

### 4.4 分布式训练

```python
# CUDA
import torch.distributed as dist
dist.init_process_group('nccl', rank=rank, world_size=world_size)

# NPU
import torch.distributed as dist
# CANN支持NCCL后端，直接兼容
dist.init_process_group('nccl', rank=rank, world_size=world_size)
```

### 4.5 混合精度训练

```python
# CUDA
from torch.cuda.amp import autocast, GradScaler
scaler = GradScaler()
with autocast():
    loss = model(input)
scaler.scale(loss).backward()
scaler.step(optimizer)
scaler.update()

# NPU (推荐方式)
from torch_npu.npu import NPUAppliedOptimizer
# torch_npu自动处理混合精度
optimizer = NPUAppliedOptimizer(optimizer)

# 或使用AMP
from torch_npu.amp import autocast, GradScaler
scaler = GradScaler()
with autocast():
    loss = model(input)
scaler.scale(loss).backward()
scaler.step(optimizer)
scaler.update()
```

## 5. 第三方库迁移

### 5.1 DeepSpeed

```python
# CUDA DeepSpeed
import deepspeed
model, optimizer, _, _ = deepspeed.initialize(
    model=model,
    optimizer=optimizer,
    config=ds_config
)

# NPU DeepSpeed
# 使用NPU特定配置
ds_config = {
    "train_batch_size": 32,
    "fp16": {"enabled": True},
    "zero_optimization": {
        "stage": 1,
        "reduce_bucket_size": 1e8
    },
    "npu": {
        "enabled": True,
        "zero_stage": 2
    }
}
model, optimizer, _, _ = deepspeed.initialize(
    model=model,
    optimizer=optimizer,
    config=ds_config
)
```

### 5.2 Accelerate

```python
# CUDA
from accelerate import Accelerator
accelerator = Accelerator()
model, optimizer, data = accelerator.prepare(
    model, optimizer, dataloader
)

# NPU
from accelerate import Accelerator
accelerator = Accelerator(
    device_placement=True,
    kwargs_handlers=[...]  # NPU特定配置
)
```

### 5.3 Flash Attention

```python
# CUDA flash-attn
from flash_attn import flash_attn_func
output = flash_attn_func(q, k, v)

# NPU方案
# 方案1: 使用torch_npu
import torch_npu
output = torch_npu.npu_flash_attention(q, k, v)

# 方案2: 使用ATB
from ascend_transformer_bindings import flash_attention
output = flash_attention(q, k, v)

# 方案3: 使用MindSpore
import mindspore
output = mindspore.ops.nn.flash_attention(q, k, v)
```

## 6. 性能优化技巧

### 6.1 算子融合

```python
# 使用torch.compile
model = model.to('npu')
model = torch.compile(model, backend='inductor')

# 或使用torch_npu优化
model = torch_npu.optimize(model, dtype=torch.float16)
```

### 6.2 内存优化

```python
# 梯度累积（减少batch size）
accumulation_steps = 4
for i, (input, target) in enumerate(dataloader):
    output = model(input)
    loss = criterion(output, target)
    loss = loss / accumulation_steps
    loss.backward()
    
    if (i + 1) % accumulation_steps == 0:
        optimizer.step()
        optimizer.zero_grad()
```

### 6.3 I/O优化

```python
# 使用内存映射
weights = torch.load('model.pt', map_location='npu')

# 使用异步加载
prefetcher = DataLoaderPrefetcher(dataloader, device='npu')
```

## 7. 迁移验证

### 7.1 功能测试

```python
def verify_npu_equivalence():
    # 创建相同输入
    x = torch.randn(8, 128, device='cpu')
    
    # CUDA输出
    model_cuda = Model().cuda()
    model_cuda.eval()
    with torch.no_grad():
        out_cuda = model_cuda(x.cuda()).cpu()
    
    # NPU输出
    model_npu = Model().npu()
    model_npu.eval()
    with torch.no_grad():
        out_npu = model_npu(x.npu()).cpu()
    
    # 比较结果
    diff = (out_cuda - out_npu).abs().max()
    print(f"Max difference: {diff}")
    return diff < 1e-4
```

### 7.2 性能测试

```python
import time

def benchmark():
    model = Model().npu()
    x = torch.randn(1, 128, 512).npu()
    
    # Warmup
    for _ in range(10):
        _ = model(x)
    
    # Benchmark
    start = time.time()
    for _ in range(100):
        _ = model(x)
    end = time.time()
    
    print(f"Average time: {(end-start)/100*1000:.2f}ms")
```

## 8. 常见问题

### 8.1 精度问题

**问题**: NPU输出与CUDA有差异

**解决方案**:
1. 检查数据类型是否一致
2. 确认随机种子设置相同
3. 检查算子实现是否有差异
4. 使用`torch.allclose`验证

### 8.2 性能问题

**问题**: NPU性能不如预期

**解决方案**:
1. 使用`torch_npu.optimize()`
2. 启用算子融合
3. 调整batch size
4. 使用ATB优化库

### 8.3 内存问题

**问题**: 内存溢出

**解决方案**:
1. 减小batch size
2. 使用梯度累积
3. 启用内存复用
4. 清理缓存
