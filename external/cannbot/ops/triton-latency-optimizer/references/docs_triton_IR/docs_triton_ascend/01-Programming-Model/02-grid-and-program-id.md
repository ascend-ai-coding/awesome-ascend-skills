# Grid/Program ID 与 AI Core 对应关系

## 概述

在 Triton 编程模型中，Grid 定义了并行 program 实例的维度布局，`tl.program_id(axis)` 返回当前 program 实例在指定轴上的索引，`tl.num_programs(axis)` 返回指定轴上的 program 总数。在昇腾 NPU 上，这些概念与 AI Core 的物理分配直接关联：每个 program 实例绑定一个 AI Core 执行，Grid 配置决定了核的分配方式。

理解 Grid/Program ID 在 NPU 上的语义，是编写正确且高效的多核并行 Triton 算子的基础。与 GPU 不同，NPU 的 Grid 配置应优先对齐物理核数，2D/3D Grid 会被合并为 1D 执行，多个小 Grid 可以合并为一个大 Grid 以减少调度开销。

**关键词**：Grid、program_id、num_programs、AI Core、核分配、Grid 合并、1D Grid

## 关键概念

| 概念 | API | NPU 语义 | 说明 |
|------|-----|----------|------|
| Program ID | `tl.program_id(axis)` | 当前 AI Core 的索引 | axis 为 0/1/2，对应 Grid 的 x/y/z 维度 |
| Program 数量 | `tl.num_programs(axis)` | 指定维度上的 AI Core 总数 | 应与物理核数对齐 |
| Grid 配置 | `kernel[grid](...)` | AI Core 分配方案 | 推荐使用 1D Grid |
| coreDim | `grid_x * grid_y * grid_z` | 总 program 实例数 | 不能超过 65535 |
| Grid 合并 | 编译器优化 | 多个小 Grid 合并为一个大 Grid | 减少核启动开销 |

## 详细内容

### 1. tl.program_id(axis) 在 NPU 上的语义

`tl.program_id(axis)` 返回当前 program 实例在 Grid 指定维度上的索引，取值范围为 `[0, num_programs(axis))`。

源码定义（[core.py](https://github.com/triton-lang/triton-ascend/tree/main/python/triton/language/core.py#L1151-L1166)）：

```python
@builtin
def program_id(axis, _builder=None):
    """
    Returns the id of the current program instance along the given :code:`axis`.

    :param axis: The axis of the 3D launch grid. Must be 0, 1 or 2.
    :type axis: int
    """
    axis = _constexpr_to_value(axis)
    return semantic.program_id(axis, _builder)
```

在 NPU 上的语义映射：

| `tl.program_id(axis)` | NPU 含义 |
|------------------------|----------|
| `tl.program_id(0)` | 当前 AI Core 在 x 维度的索引 |
| `tl.program_id(1)` | 当前 AI Core 在 y 维度的索引 |
| `tl.program_id(2)` | 当前 AI Core 在 z 维度的索引 |

**重要**：在 NPU 上，2D/3D Grid 会被编译器合并为 1D 执行。因此，推荐直接使用 1D Grid 和 `tl.program_id(0)`，避免多维索引的复杂性。

### 2. Grid 配置（x, y, z 维度）与 AI Core 分配

#### 2.1 Grid 的三种配置方式

```python
# 方式一：1D Grid（推荐）
grid_1d = (num_cores,)
kernel[grid_1d](...)

# 方式二：2D Grid
grid_2d = (num_cores_x, num_cores_y)
kernel[grid_2d](...)

# 方式三：3D Grid
grid_3d = (num_cores_x, num_cores_y, num_cores_z)
kernel[grid_3d](...)
```

#### 2.2 NPU 上的 Grid 合并机制

在 NPU 上，2D 和 3D Grid 会被合并为 1D 执行。例如：

```python
# 2D Grid (4, 5) 在 NPU 上等效于 1D Grid (20,)
grid_2d = (4, 5)
# 编译器内部将 (pid_x, pid_y) 映射为 flat_pid = pid_x + pid_y * 4

# 3D Grid (2, 3, 4) 在 NPU 上等效于 1D Grid (24,)
grid_3d = (2, 3, 4)
# 编译器内部将 (pid_x, pid_y, pid_z) 映射为 flat_pid = pid_x + pid_y * 2 + pid_z * 6
```

**结论**：由于 NPU 的 Grid 合并机制，2D Grid `(4, 5)` 与 1D Grid `(20,)` 效果相同。推荐直接使用 1D Grid，代码更简洁且避免不必要的维度转换。

#### 2.3 Grid 大小与物理核数的关系

| Grid 大小 | 与物理核数关系 | 行为 |
|-----------|--------------|------|
| Grid < 物理核数 | 部分核空闲 | 资源浪费，性能下降 |
| Grid = 物理核数 | 充分利用所有核 | **最优配置** |
| Grid > 物理核数 | 分批调度 | 额外调度开销 |

### 3. num_programs 的含义

`tl.num_programs(axis)` 返回指定维度上启动的 program 总数。

源码定义（[core.py](https://github.com/triton-lang/triton-ascend/tree/main/python/triton/language/core.py#L1170-L1178)）：

```python
@builtin
def num_programs(axis, _builder=None):
    """
    Returns the number of program instances launched along the given :code:`axis`.

    :param axis: The axis of the 3D launch grid. Must be 0, 1 or 2.
    :type axis: int
    """
    axis = _constexpr_to_value(axis)
    return semantic.num_programs(axis, _builder)
```

#### 3.1 典型用法

`tl.num_programs` 最常见的用途是实现跨步分配任务：

```python
@triton.jit
def strided_kernel(data_ptr, N, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    NUM_CORE = tl.num_programs(0)
    NUM_BLOCKS = tl.cdiv(N, BLOCK_SIZE)

    # 跨步分配：每个核从自己的 pid 开始，步长为总核数
    for block_idx in range(pid, NUM_BLOCKS, NUM_CORE):
        offsets = block_idx * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
        mask = offsets < N
        data = tl.load(data_ptr + offsets, mask=mask)
        data = data * 2.0
        tl.store(data_ptr + offsets, data, mask=mask)
```

#### 3.2 多维 Grid 中的 num_programs

```python
# 2D Grid 示例
@triton.jit
def kernel_2d(data_ptr, M, N, BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    num_m = tl.num_programs(0)
    num_n = tl.num_programs(1)

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    # ...

# 启动
grid = (4, 5)
kernel_2d[grid](data_ptr, M, N, BLOCK_M=64, BLOCK_N=64)
# tl.num_programs(0) = 4, tl.num_programs(1) = 5
```

### 4. Grid 合并优化

#### 4.1 多个小 Grid 合并为一个大 Grid

当同一个 kernel 被多次调用（例如循环中），每次调用都会产生核启动开销。可以将多个小 Grid 合并为一个大 Grid，减少调度次数：

```python
# 优化前：多次小 Grid 启动
for i in range(num_iterations):
    grid = (num_cores,)
    my_kernel[grid](data_ptr + i * chunk_size, chunk_size, BLOCK_SIZE)

# 优化后：单次大 Grid 启动，核内循环处理
@triton.jit
def merged_kernel(data_ptr, total_size, chunk_size, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    NUM_CORE = tl.num_programs(0)
    total_blocks = tl.cdiv(total_size, BLOCK_SIZE)

    for block_idx in range(pid, total_blocks, NUM_CORE):
        offsets = block_idx * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
        mask = offsets < total_size
        data = tl.load(data_ptr + offsets, mask=mask)
        data = data * 2.0
        tl.store(data_ptr + offsets, data, mask=mask)

grid = (num_cores,)
merged_kernel[grid](data_ptr, total_size, chunk_size, BLOCK_SIZE)
```

#### 4.2 TRITON_ALL_BLOCKS_PARALLEL 优化

当逻辑 Grid 大小大于物理核数时，可以启用 `TRITON_ALL_BLOCKS_PARALLEL` 环境变量，让编译器自动调整逻辑核数为物理核数：

```bash
export TRITON_ALL_BLOCKS_PARALLEL=1
```

启用条件：
- 逻辑核间可并行（无数据依赖）
- 逻辑核数大于物理核数

编译器行为：
- 自动将逻辑核数调整为物理核数
- 在核内添加循环处理剩余任务
- 可通过 `auto_blockify_size` 编译选项指定扩展的左起第一个维度的大小

```python
# autotune 中配置 auto_blockify_size
@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE': 1024, 'auto_blockify_size': 20}),
    ],
    key=['N'],
)
@triton.jit
def my_kernel(...):
    ...
```

### 5. 代码示例：不同 Grid 配置模式

#### 5.1 1D Grid 模式（推荐）

```python
import torch
import torch_npu
import triton
import triton.language as tl
import triton.runtime.driver as driver

@triton.jit
def vector_add_kernel(x_ptr, y_ptr, out_ptr, N, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    NUM_CORE = tl.num_programs(0)
    NUM_BLOCKS = tl.cdiv(N, BLOCK_SIZE)

    for block_idx in range(pid, NUM_BLOCKS, NUM_CORE):
        offsets = block_idx * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
        mask = offsets < N
        x = tl.load(x_ptr + offsets, mask=mask)
        y = tl.load(y_ptr + offsets, mask=mask)
        tl.store(out_ptr + offsets, x + y, mask=mask)

def vector_add(x, y):
    out = torch.empty_like(x)
    N = x.numel()

    device = torch_npu.npu.current_device()
    props = driver.active.utils.get_device_properties(device)
    NUM_CORE = props["num_vectorcore"]

    BLOCK_SIZE = 1024
    grid = (NUM_CORE,)
    vector_add_kernel[grid](x, y, out, N, BLOCK_SIZE)
    return out
```

#### 5.2 2D Grid 模式（矩阵乘法）

```python
@triton.jit
def matmul_kernel(A, B, C, M, N, K,
                  stride_am, stride_ak,
                  stride_bk, stride_bn,
                  stride_cm, stride_cn,
                  BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    for k in range(0, K, BLOCK_K):
        a = tl.load(A + offs_m[:, None] * stride_am + (offs_k[None, :] + k) * stride_ak,
                     mask=(offs_m[:, None] < M) & (offs_k[None, :] + k < K), other=0.0)
        b = tl.load(B + (offs_k[:, None] + k) * stride_bk + offs_n[None, :] * stride_bn,
                     mask=(offs_k[:, None] + k < K) & (offs_n[None, :] < N), other=0.0)
        acc += tl.dot(a, b)

    tl.store(C + offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn,
             acc, mask=(offs_m[:, None] < M) & (offs_n[None, :] < N))

# 启动 - 注意 NPU 上应对齐物理核数
device = torch_npu.npu.current_device()
props = driver.active.utils.get_device_properties(device)
aicore_num = props["num_aicore"]

BLOCK_M, BLOCK_N = 64, 64
grid = (triton.cdiv(M, BLOCK_M), triton.cdiv(N, BLOCK_N))
# 如果 grid 乘积 > aicore_num，考虑使用 1D Grid + 核内循环
```

#### 5.3 多维任务展平为 1D Grid（推荐 NPU 模式）

对于多维任务（如 batch * heads * blocks），推荐展平为 1D 后使用跨步分配：

```python
@triton.jit
def attention_kernel(Q, K, V, Out,
                     stride_qz, stride_qh,
                     Z: tl.constexpr, H: tl.constexpr,
                     N_CTX: tl.constexpr,
                     HEAD_DIM: tl.constexpr,
                     BLOCK_M: tl.constexpr,
                     BLOCK_N: tl.constexpr):
    NUM_BLOCKS_M = N_CTX // BLOCK_M
    NUM_BLOCKS = NUM_BLOCKS_M * Z * H

    pid = tl.program_id(0)
    NUM_CORE = tl.num_programs(0)

    for block_idx in range(pid, NUM_BLOCKS, NUM_CORE):
        task_hz_idx = block_idx // NUM_BLOCKS_M
        task_m_idx = block_idx % NUM_BLOCKS_M
        off_z = task_hz_idx // H
        off_h = task_hz_idx % H
        qvk_offset = off_z.to(tl.int64) * stride_qz + off_h.to(tl.int64) * stride_qh
        # ... 处理当前 block 的计算

device = torch_npu.npu.current_device()
props = driver.active.utils.get_device_properties(device)
NUM_CORE = props["num_aicore"]

grid = (NUM_CORE,)
attention_kernel[grid](Q, K, V, Out, stride_qz, stride_qh,
                        Z=Z, H=H, N_CTX=N_CTX, HEAD_DIM=HEAD_DIM,
                        BLOCK_M=64, BLOCK_N=64)
```

#### 5.4 使用 libentry 装饰器

`libentry` 是 Triton-Ascend 提供的装饰器，自动处理 Grid 配置和核数对齐：

```python
from triton.runtime import libentry

@libentry()
@triton.jit
def auto_kernel(data_ptr, N, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < N
    data = tl.load(data_ptr + offsets, mask=mask)
    data = data * 2.0
    tl.store(data_ptr + offsets, data, mask=mask)
```

## NPU 适配要点

1. **优先使用 1D Grid**：2D/3D Grid 在 NPU 上会被合并为 1D，直接使用 1D Grid 避免不必要的维度转换
2. **Grid 大小对齐物理核数**：使用 `tl.num_programs(0)` 获取实际核数，而非硬编码
3. **使用跨步分配模式**：`range(pid, NUM_BLOCKS, NUM_CORE)` 确保任务均匀分配
4. **多维任务展平为 1D**：将 batch/heads/blocks 等维度展平后用 1D Grid + 核内循环处理
5. **注意 coreDim 上限**：确保 Grid 乘积不超过 65535
6. **善用 autotune**：通过 `@triton.autotune` 自动搜索最优 BLOCK_SIZE 配置

## 常见问题

**Q1: `tl.program_id(0)` 在 2D Grid 中返回什么？**

A: 返回当前 program 在 x 维度的索引。在 NPU 上，2D Grid 会被合并为 1D，因此 `tl.program_id(0)` 仍然返回 x 维度的索引。如果需要全局唯一 ID，需要手动计算：`flat_pid = pid_x + pid_y * num_programs(0)`。

**Q2: 为什么推荐 1D Grid 而不是 2D Grid？**

A: NPU 的 2D Grid 会被编译器合并为 1D 执行，没有性能优势。1D Grid 更直观，代码更简洁，且避免了多维索引的复杂性。对于矩阵乘法等天然 2D 问题，推荐将 2D 任务展平为 1D 后使用跨步分配。

**Q3: `tl.num_programs(0)` 和 `get_device_properties` 获取的核数有什么区别？**

A: `tl.num_programs(0)` 是 kernel 内部获取的 Grid 大小（即启动的 program 数量），而 `get_device_properties` 获取的是物理核数。当 Grid 大小与物理核数对齐时，两者相等。建议在 kernel 内部使用 `tl.num_programs(0)`，在 host 端使用 `get_device_properties`。

**Q4: Grid 合并优化有什么限制？**

A: Grid 合并要求合并前后的数据访问模式不变，且核间无数据依赖。如果 kernel 内部使用了 `tl.program_id(1)` 或 `tl.program_id(2)` 进行条件分支，合并后可能导致逻辑错误。

**Q5: 如何调试 Grid 配置问题？**

A: 设置 `TRITON_DEBUG=1` 可以查看编译中间产物。如果遇到 coreDim 超限错误，检查 `grid_x * grid_y * grid_z` 是否超过 65535。如果遇到性能问题，检查 Grid 大小是否对齐物理核数。

## 相关文档

- [SPMD 模型在 NPU 上的映射](./01-spmd-on-npu.md)
- [NPU 内存层次](./03-memory-model.md)
- [数据类型支持矩阵与约束](./04-data-types.md)
- [硬件架构概览](../../docs_ascendnpu_ir/00-Architecture/01-npu-hardware-overview.md)

## 源文件参考

- [core.py - program_id](https://github.com/triton-lang/triton-ascend/tree/main/python/triton/language/core.py#L1151-L1166) - program_id 函数定义
- [core.py - num_programs](https://github.com/triton-lang/triton-ascend/tree/main/python/triton/language/core.py#L1170-L1178) - num_programs 函数定义
- [programming_guide.md](https://github.com/triton-lang/triton-ascend/tree/main/docs/zh/programming_guide.md) - Triton 算子开发指南
- [architecture_difference.md](https://github.com/triton-lang/triton-ascend/tree/main/docs/zh/migration_guide/architecture_difference.md) - 架构差异分析
