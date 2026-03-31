# GPU与NPU架构差异详解

## 核心架构对比

| 维度 | GPU（NVIDIA） | 昇腾（Ascend） |
|------|---------------|----------------|
| **grid本质** | 逻辑任务维度（和物理核解耦） | 物理核组映射（绑定AI Core拓扑） |
| **核数/维度限制** | grid维度/大小无硬限制 | grid大小≤AI Core总数 |
| **核心结构** | CUDA Core + Tensor Core | AI Core = Cube Core + Vector Core |
| **执行模式** | 每个线程执行一次kernel | 每个核执行一次Block，支持重复调度 |

## 并发任务配置原则

### Vector-only算子
- 并发任务数 = Vector Core数
- 适用于：简单向量运算、element-wise操作

### 含tl.dot算子
- 并发任务数 = AI Core数
- 适用于：矩阵乘法、卷积等需要Cube Core的算子

## 内存层次结构

### GPU内存层次
```
Global Memory (HBM)
    ↓
Shared Memory (SRAM, 可编程)
    ↓
Registers
```

### NPU内存层次
```
Global Memory (HBM)
    ↓
Unified Buffer (UB, 192KB-256KB)
    ↓
Vector Registers / Cube Buffer
```

## Grid配置差异

### GPU Grid配置
```python
# GPU: grid大小可以很大，由硬件调度
grid = lambda meta: (triton.cdiv(n_elements, meta['BLOCK_SIZE']),)
# 可以有数千个block
```

### NPU Grid配置
```python
# NPU: grid大小受限于物理核数
device = torch_npu.npu.current_device()
properties = driver.active.utils.get_device_properties(device)
NUM_AICORE = properties["num_aicore"]  # 通常30-108个

# 固定核数，使用跨步分配
grid = (NUM_AICORE,)
```

## 数据类型支持

### Vector指令支持的数据类型
- ✅ float16, bfloat16, float32
- ✅ int8, int16, int32
- ⚠️ int64: 会退化为scalar操作

### Cube指令支持的数据类型
- ✅ float16, bfloat16 (矩阵乘法)
- ✅ int8 (量化计算)

## 内存对齐要求

| 算子类型 | 对齐要求 | 说明 |
|---------|---------|------|
| VV算子（Vector-Vector） | 32字节 | 仅使用Vector Core |
| CV算子（Cube-Vector） | 512字节 | 使用Cube Core |

## 性能优化要点

### GPU优化重点
1. Coalesced memory access
2. Shared memory bank conflicts
3. Occupancy最大化

### NPU优化重点
1. UB空间利用（尽量用满192KB）
2. Tiling策略（避免UB溢出）
3. care_padding=False（提升并行度）
4. 避免离散访存

## 执行模型差异

### GPU执行模型
```
每个thread执行一次kernel
thread间通过shared memory通信
```

### NPU执行模型
```
每个AI Core执行一次Block
支持重复调度（一个核处理多个Block）
使用跨步分配任务
```

## 代码示例对比

### GPU代码
```python
@triton.jit
def kernel_gpu(x_ptr, output_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(axis=0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    x = tl.load(x_ptr + offsets, mask=mask)
    output = x * 2
    tl.store(output_ptr + offsets, output, mask=mask)

grid = lambda meta: (triton.cdiv(n_elements, meta['BLOCK_SIZE']),)
kernel_gpu[grid](x, output, n_elements, BLOCK_SIZE=1024)
```

### NPU代码
```python
@triton.jit
def kernel_npu(x_ptr, output_ptr, n_elements, NUM_CORES: tl.constexpr, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(axis=0)
    NUM_BLOCKS = tl.cdiv(n_elements, BLOCK_SIZE)
    
    # 跨步分配任务
    for block_idx in range(pid, NUM_BLOCKS, NUM_CORES):
        offsets = block_idx * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
        mask = offsets < n_elements
        x = tl.load(x_ptr + offsets, mask=mask, care_padding=False)
        output = x * 2
        tl.store(output_ptr + offsets, output, mask=mask)

grid = (NUM_AICORE,)
kernel_npu[grid](x, output, n_elements, NUM_CORES=NUM_AICORE, BLOCK_SIZE=1024)
```
