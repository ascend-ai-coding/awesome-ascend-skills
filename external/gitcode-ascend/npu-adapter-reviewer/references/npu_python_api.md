# NPU Python API 参考

## 1. 设备管理API

### 1.1 基础设备操作

```python
import torch
import torch_npu

# 检查NPU可用性
torch_npu.is_npu_available()  # -> bool
torch_npu.device_count()      # -> int

# 获取当前设备
torch_npu.current_device()    # -> int
torch_npu.get_device_name(0)  # -> str

# 设置当前设备
torch_npu.set_device(0)

# 设备上下文管理器
with torch_npu.device(0):
    x = torch.randn(3, 4)
```

### 1.2 流管理

```python
# 创建流
stream = torch_npu.Stream()
stream = torch_npu.Stream(priority=0)

# 流同步
torch_npu.synchronize()
stream.synchronize()

# 事件
start_event = torch_npu.Event(enable_timing=True)
end_event = torch_npu.Event(enable_timing=True)

start_event.record()
# ... 执行计算 ...
end_event.record()
end_event.synchronize()

elapsed_time = start_event.elapsed_time(end_event)  # ms
```

## 2. 张量操作API

### 2.1 张量创建

```python
# 创建NPU张量
x = torch.randn(3, 4, device='npu')
y = torch.zeros(5, 6, device='npu:0')
z = torch.ones(2, 3, dtype=torch.float16, device='npu')

# 使用NPU特定方法
x = torch_npu.FloatTensor(3, 4)
x = torch_npu.IntTensor(3, 4)
x = torch_npu.HalfTensor(3, 4)  # float16
```

### 2.2 张量迁移

```python
# 移动到NPU
tensor = tensor.npu()
tensor = tensor.to('npu')
tensor = tensor.to(device='npu:0')

# 异步移动
tensor = tensor.npu(non_blocking=True)
tensor = tensor.to('npu', non_blocking=True)

# 从NPU移回CPU
tensor = tensor.cpu()
tensor = tensor.to('cpu')
```

### 2.3 内存管理

```python
# 清空缓存
torch_npu.empty_cache()

# 内存信息
torch_npu.memory_allocated()      # 当前分配 bytes
torch_npu.memory_reserved()       # 保留缓存 bytes
torch_npu.max_memory_allocated() # 最大分配
torch_npu.max_memory_reserved()  # 最大保留

# 重置峰值统计
torch_npu.reset_peak_memory_stats()
torch_npu.reset_peak_memory_stats(device=None)
```

## 3. 神经网络API

### 3.1 层与模块

```python
import torch.nn as nn
import torch_npu.nn as npu_nn

# 标准层自动支持NPU
linear = nn.Linear(128, 64).npu()
conv = nn.Conv2d(3, 64, 3).npu()
layernorm = nn.LayerNorm(64).npu()

# NPU优化层
npu_linear = npu_nn.Linear(128, 64)
npu_conv = npu_nn.Conv2d(3, 64, 3)
```

### 3.2 激活函数

```python
import torch_npu.nn as npu_nn

# NPU优化的激活函数
act = npu_npu.GELU()
act = npu_npu.ReLU()
act = npu_npu.SiLU()
```

## 4. 优化器API

### 4.1 NPU优化器

```python
import torch.optim as optim
from torch_npu.npu import NPUAppliedOptimizer

# 标准优化器自动适配NPU
optimizer = optim.Adam(model.parameters(), lr=1e-3)
optimizer = optim.SGD(model.parameters(), lr=1e-3)

# NPU特定优化器
optimizer = NPUAppliedOptimizer(optimizer)
```

### 4.2 分布式优化器

```python
import torch.distributed as dist

# 分布式训练
model = torch.nn.parallel.DistributedDataParallel(
    model,
    device_ids=[local_rank],
    output_device=local_rank
)
```

## 5. 混合精度API

### 5.1 AMP

```python
from torch_npu.amp import autocast, GradScaler

# 创建GradScaler
scaler = GradScaler(
    init_scale=2.0**16,
    growth_factor=2.0,
    backoff_factor=0.5,
    growth_interval=2000
)

# 前向传播
with autocast():
    output = model(input)
    loss = criterion(output, target)

# 反向传播
scaler.scale(loss).backward()
scaler.step(optimizer)
scaler.update()
```

### 5.2 精度控制

```python
# 设置计算精度
torch_npu.set_compile_mode(jit=False)

# 启用/禁用FP16
torch_npu.set_flush_denorm(True)
```

## 6. 数据加载API

### 6.1 NPU DataLoader

```python
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler

# NPU DataLoader (标准API，自动适配)
loader = DataLoader(
    dataset,
    batch_size=32,
    shuffle=True,
    num_workers=4,
    pin_memory=True  # NPU自动处理
)

# 分布式采样
sampler = DistributedSampler(
    dataset,
    num_replicas=world_size,
    rank=rank
)
```

### 6.2 异步数据加载

```python
# 使用NPU异步数据预取
class DataPrefetcher:
    def __init__(self, loader, device):
        self.loader = iter(loader)
        self.device = device
        self.stream = torch_npu.Stream()
        self.preload()
    
    def preload(self):
        try:
            self.next_input, self.next_target = next(self.loader)
        except StopIteration:
            self.next_input = None
            self.next_target = None
            return
        
        with torch_npu.stream(self.stream):
            self.next_input = self.next_input.to(self.device, non_blocking=True)
            self.next_target = self.next_target.to(self.device, non_blocking=True)
    
    def next(self):
        torch_npu.synchronize()
        input, target = self.next_input, self.next_target
        self.preload()
        return input, target
```

## 7. 算子API

### 7.1 Flash Attention

```python
import torch_npu

# NPU Flash Attention
output = torch_npu.npu_flash_attention(
    query,           # (B, N, H, D)
    key,
    value,
    head_dim_num=64,  # attention head dimension
    head_dim_value=64,
    next_token_type=0,
    actual_seq_len_q=None,
    actual_seq_len_kv=None,
    is_causal=True,
    dropout_p=0.0,
    is_fused_head=True,
    padding_mask=None,
    attn_mask=None,
    scale=1.0
)
```

### 7.2 矩阵运算

```python
import torch_npu

# 矩阵乘法 (自动优化)
result = torch_npu.matmul(a, b)

# 批量矩阵乘法
result = torch_npu.bmm(a, b)

# 转置矩阵乘法
result = torch_npu.mm(a, b.t())

# BMM with alpha
result = torch_npu.baddbmm(
    input, batch1, batch2,
    alpha=1.0,
    beta=0.0
)
```

### 7.3 归约操作

```python
import torch_npu

# Softmax
result = torch_npu.npu_softmax(x, dim=-1)

# LayerNorm
result = torch_npu.npu_layer_norm(
    x,
    normalized_shape,
    weight,
    bias,
    eps=1e-5
)

# BatchNorm (in-place)
torch_npu.npu_batch_norm(
    input,
    running_mean,
    running_var,
    weight,
    bias,
    training,
    momentum,
    eps
)
```

## 8. Profiler API

### 8.1 性能分析

```python
import torch_npu.profiler

# 创建Profiler
with torch_npu.profiler.profile(
    activities=[
        torch_npu.profiler.ProfilerActivity.CPU,
        torch_npu.profiler.ProfilerActivity.NPU,
    ],
    schedule=torch_npu.profiler.schedule(
        wait=1,
        warmup=1,
        active=3,
        repeat=1
    ),
    on_trace_ready=torch_npu.profiler.tensorboard_trace_handler('./logs'),
    record_shapes=True,
    profile_memory=True,
    with_stack=True
) as prof:
    for step, batch in enumerate(dataloader):
        with torch_npu.profiler.record_function("data_loading"):
            input = batch.to('npu')
        
        with torch_npu.profiler.record_function("forward"):
            output = model(input)
        
        with torch_npu.profiler.record_function("backward"):
            loss.backward()
        
        prof.step()

# 查看结果
print(prof.key_averages().table(
    sort_by="npu_time_total",
    row_limit=20
))

# 导出
prof.export_chrome_trace("trace.json")
```

### 8.2 内存分析

```python
# 内存使用追踪
with torch_npu.profiler.profile(
    activities=[
        torch_npu.profiler.ProfilerActivity.CPU,
        torch_npu.profiler.ProfilerActivity.NPU,
    ],
    profile_memory=True
) as prof:
    output = model(input)
    prof.step()

# 查看内存统计
print(prof.key_averages().table(
    sort_by="self_npu_memory_usage",
    row_limit=20
))
```

## 9. 工具函数

### 9.1 设备检测

```python
# 检测是否在NPU上运行
def is_npu_available():
    return torch_npu.is_npu_available()

# 获取NPU设备属性
def get_device_properties(device_id=0):
    props = torch_npu.get_device_properties(device_id)
    return {
        "name": props.name,
        "total_memory": props.total_memory,
        "major": props.major,
        "minor": props.minor
    }
```

### 9.2 格式转换

```python
# NPU格式转换
# NCHW -> NHWC
x = torch_npu.npu_format_cast(x, 2)  # 2 = NHWC

# 数据类型转换
x = x.to(dtype=torch.float16)
x = x.to(dtype=torch.float32)
```

### 9.3 随机数

```python
# 设置随机种子
torch_npu.manual_seed(42)
torch_npu.manual_seed_all(42)

# 获取随机状态
state = torch_npu.get_rng_state()
torch_npu.set_rng_state(state)
```

## 10. 配置选项

### 10.1 运行时配置

```python
# 设置选项
torch_npu.set_option("opt_level", 2)      # 优化级别 0-3
torch_npu.set_option("memory_fraction", 0.8)  # 内存使用比例
torch_npu.set_option("enable_profiling", True)  # 启用profiling

# 获取选项
torch_npu.get_option("opt_level")
```

### 10.2 编译选项

```python
# TorchScript编译
model = torch.jit.trace(model, input)
model = torch.jit.freeze(model)

# torch.compile (PyTorch 2.0+)
model = torch.compile(model, backend='inductor')
```
