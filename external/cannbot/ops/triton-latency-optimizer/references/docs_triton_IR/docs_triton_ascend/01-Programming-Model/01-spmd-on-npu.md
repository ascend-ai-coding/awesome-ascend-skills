# SPMD 模型在 NPU 上的映射

## 概述

Triton 采用 SPMD（Single Program, Multiple Data）编程模型，开发者编写一份 kernel 程序，由多个 program 实例并行执行，每个实例处理不同的数据分片。在 GPU 上，SPMD 通过 CUDA Thread Block 实现；在华为昇腾 NPU 上，SPMD 模型映射为 **AI Core 物理核并行**，每个 program 实例对应一个 AI Core 的执行单元。

理解 SPMD 在 NPU 上的映射机制，是从 GPU 迁移 Triton 算子、编写高性能 NPU 算子的核心基础。关键差异在于：GPU 的 grid 是逻辑维度，与物理 SM 解耦；而 NPU 的 grid 是**物理核强绑定**模式，grid 大小应与 AI Core 数量对齐。

**关键词**：SPMD、AI Core、多核并行、Grid 配置、物理核绑定、SIMD、SIMT、coreDim

## 关键概念

| 概念 | GPU 语义 | NPU 语义 | 说明 |
|------|----------|----------|------|
| Program 实例 | Thread Block（线程块） | AI Core 执行单元 | NPU 上每个 program 绑定一个物理核 |
| 并行粒度 | Warp（32 线程） | AI Core（含 Cube + 2x Vector） | NPU 以 AI Core 为最小调度单位 |
| 执行模型 | SIMT（单指令多线程） | SIMD（单指令多数据） | NPU Vector 单元为 SIMD 模式 |
| Grid 本质 | 逻辑任务维度，与物理核解耦 | 物理核组映射，绑定 AI Core 拓扑 | NPU grid 大小应 <= AI Core 总数 |
| 核数上限 | 数千个 SM | 数十个 AI Core（20~36） | NPU 核数远少于 GPU |
| 跨核调度 | 硬件自动调度 | 超出物理核数时分批调度 | 分批调度引入额外开销 |

## 详细内容

### 1. Triton SPMD 编程模型

Triton 的 SPMD 模型中，开发者通过 `@triton.jit` 定义 kernel 函数，通过 `kernel[grid](...)` 语法启动并行执行。Grid 指定了 program 实例的数量和维度布局，每个 program 实例通过 `tl.program_id(axis)` 获取自身标识，从而确定要处理的数据范围。

```python
import triton
import triton.language as tl

@triton.jit
def my_kernel(x_ptr, N, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < N
    x = tl.load(x_ptr + offsets, mask=mask)
    x = x * 2.0
    tl.store(x_ptr + offsets, x, mask=mask)

N = 1024
BLOCK_SIZE = 256
grid = (triton.cdiv(N, BLOCK_SIZE),)
my_kernel[grid](x_ptr, N, BLOCK_SIZE)
```

### 2. NPU 上的 SPMD 执行模型

在昇腾 NPU 上，SPMD 的 program 实例直接映射到 AI Core：

- **每个 program 实例对应一个 AI Core**：AI Core 是 NPU 的基本计算单元，包含 1 个 Cube Core（矩阵计算）和 2 个 Vector Core（向量计算），此比例适用于所有当前 Ascend NPU 型号
- **物理核强绑定**：Grid 配置中的 program 数量直接决定了使用的 AI Core 数量
- **分批调度机制**：当 program 数量超过物理 AI Core 数量时，运行时会将任务分为多个批次调度执行，每个批次的并行度不超过物理核数

#### AI Core 数量参考

| NPU 型号 | AI Core 数 | Cube Core 数 | Vector Core 数 |
|----------|-----------|-------------|---------------|
| Ascend910B1 | 24 | 24 | 48 |
| Ascend910B3 | 20 | 20 | 40 |
| Ascend910_9381 | 24 | 24 | 48 |
| Ascend910_9581 | 32 | 32 | 64 |
| Ascend910_9599 | 36 | 36 | 72 |

获取当前设备的核数信息：

```python
import torch
import torch_npu
import triton.runtime.driver as driver

device = torch_npu.npu.current_device()
properties = driver.active.utils.get_device_properties(device)
vectorcore_num = properties["num_vectorcore"]
aicore_num = properties["num_aicore"]
print(f"AI Core: {aicore_num}, Vector Core: {vectorcore_num}")
```

### 3. 多核并行策略：Grid 配置与 AI Core 数量的关系

#### 3.1 核心原则

**将分核数量直接固定为硬件的物理核数**，在核内做更为细致的数据分块。这是 NPU 上获取最佳性能的关键策略。

- 对于**纯 Vector 算子**（不包含 `tl.dot`）：分核数等于 **Vector Core 数量**
- 对于 **CV 融合算子**（包含 `tl.dot`）：分核数等于 **AI Core 数量**（即 Cube Core 数量），算子执行时会按 1:2 的比例调用 Vector Core

#### 3.2 推荐的 Grid 配置模式

```python
import torch
import torch_npu
import triton.runtime.driver as driver

device = torch_npu.npu.current_device()
properties = driver.active.utils.get_device_properties(device)
vectorcore_num = properties["num_vectorcore"]
aicore_num = properties["num_aicore"]

NUM_CORE = vectorcore_num  # 纯 Vector 算子
# NUM_CORE = aicore_num   # CV 融合算子（含 tl.dot）

grid = (NUM_CORE,)
my_kernel[grid](...)
```

#### 3.3 核内循环处理任务分块

固定核数后，每个核需要通过内部循环处理多个数据分块，采用"跨步分配"策略确保任务均匀分布：

```python
@triton.jit
def parallel_kernel(data_ptr, N, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    NUM_CORE = tl.num_programs(0)
    NUM_BLOCKS = tl.cdiv(N, BLOCK_SIZE)

    for block_idx in range(pid, NUM_BLOCKS, NUM_CORE):
        block_start = block_idx * BLOCK_SIZE
        offsets = block_start + tl.arange(0, BLOCK_SIZE)
        mask = offsets < N
        data = tl.load(data_ptr + offsets, mask=mask)
        data = data * 2.0
        tl.store(data_ptr + offsets, data, mask=mask)
```

**跨步分配原理**：
- 起始值 `pid`：每个核从自己的 ID 开始取任务，避免任务重叠
- 步长 `NUM_CORE`：按总核数跨步，确保任务均匀分配到各个核
- `range(pid, NUM_BLOCKS, NUM_CORE)` 实现了循环轮转式任务分配

### 4. 与 GPU SPMD 模型的关键差异

#### 4.1 Warp vs AI Core

| 维度 | GPU Warp | NPU AI Core |
|------|----------|-------------|
| 组成 | 32 个 CUDA Thread | 1 Cube Core + 2 Vector Core |
| 执行模式 | SIMT（每线程独立 PC） | SIMD（数据并行，无独立线程） |
| 同步 | `__syncthreads()` 块内同步 | `sync_block_all` / `sync_block_set` / `sync_block_wait` |
| 调度 | 硬件自动 Warp 调度 | 编译器静态调度 |

#### 4.2 SIMT vs SIMD

GPU 采用 SIMT（Single Instruction, Multiple Thread）模型，每个 CUDA Thread 拥有独立的执行上下文，可以走不同的控制流路径（divergent branch）。NPU 的 Vector Core 采用 SIMD 模型，所有数据元素在同一时刻执行相同操作：

```python
# GPU SIMT: 每个线程可以独立判断
# if thread_id < N:
#     data = load(...)
#     store(...)

# NPU SIMD: 使用 mask 实现条件操作
offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
mask = offsets < N
data = tl.load(ptr + offsets, mask=mask)
result = tl.where(mask, data * 2.0, 0.0)
tl.store(ptr + offsets, result, mask=mask)
```

#### 4.3 Grid 维度差异

```python
# GPU: 可以自由定义大量逻辑 grid
grid_gpu = (1024, 1, 1)  # 1024 个 Thread Block，GPU 硬件自动调度

# NPU: grid 应对齐物理核数
grid_npu = (20, 1, 1)    # 对齐 Ascend910B3 的 20 个 AI Core
# 等效于
grid_npu_1d = (20,)       # 推荐使用 1D grid
```

### 5. coreDim 概念及约束

`coreDim` 是 NPU 上 Grid 维度乘积的值，代表总的 program 实例数量。NPU 硬件对 coreDim 有严格限制：

- **coreDim 不能超过 UINT16_MAX（65535）**
- 当 `coreDim = grid_x * grid_y * grid_z > 65535` 时，编译会报错

#### 5.1 coreDim 超限的典型场景

当数据规模很大而 BLOCK_SIZE 较小时，`coreDim = ceil(N / BLOCK_SIZE)` 可能超过 65535：

```
N = 1073741824, BLOCK_SIZE = 2048
coreDim = ceil(1073741824 / 2048) = 524288 > 65535  # 超限！
```

#### 5.2 解决方案

**方案一：设置环境变量自动优化**

```bash
export TRITON_ALL_BLOCKS_PARALLEL=1
```

启用后，编译器自动将逻辑核数调整为物理核数，减少调度开销。仅在逻辑核间可并行时方可启用。

**方案二：增大 BLOCK_SIZE**

```python
import triton

N = 1073741824
min_block_size = triton.next_power_of_2(triton.cdiv(N, 65535))
BLOCK_SIZE = max(32768, min_block_size)
# coreDim = ceil(1073741824 / 32768) = 32768 <= 65535  # 合规
```

**方案三：引入子块分块（BLOCK_SIZE_SUB）**

当增大 BLOCK_SIZE 导致 UB 溢出时，使用子块分块策略：

```python
@triton.jit
def safe_kernel(data_ptr, N,
                BLOCK_SIZE: tl.constexpr,
                BLOCK_SIZE_SUB: tl.constexpr):
    pid = tl.program_id(0)
    base_offset = pid * BLOCK_SIZE
    num_sub_blocks = tl.cdiv(BLOCK_SIZE, BLOCK_SIZE_SUB)

    for sub_idx in range(num_sub_blocks):
        offsets = base_offset + sub_idx * BLOCK_SIZE_SUB + tl.arange(0, BLOCK_SIZE_SUB)
        mask = offsets < N
        data = tl.load(data_ptr + offsets, mask=mask)
        data = data * 2.0
        tl.store(data_ptr + offsets, data, mask=mask)

MAIN_BLOCK = 32768
SUB_BLOCK = 1024
grid = lambda meta: (triton.cdiv(N, MAIN_BLOCK),)
safe_kernel[grid](data_ptr, N, MAIN_BLOCK, SUB_BLOCK)
```

### 6. 代码示例：配置 Grid 以充分利用 NPU 多核

以下是一个完整的示例，展示如何根据算子类型选择合适的 Grid 配置：

```python
import torch
import torch_npu
import triton
import triton.language as tl
import triton.runtime.driver as driver

def get_npu_core_count(use_dot=False):
    device = torch_npu.npu.current_device()
    properties = driver.active.utils.get_device_properties(device)
    if use_dot:
        return properties["num_aicore"]
    else:
        return properties["num_vectorcore"]

@triton.jit
def optimized_kernel(data_ptr, out_ptr, N,
                     BLOCK_SIZE: tl.constexpr,
                     BLOCK_SIZE_SUB: tl.constexpr):
    pid = tl.program_id(0)
    NUM_CORE = tl.num_programs(0)
    NUM_BLOCKS = tl.cdiv(N, BLOCK_SIZE)

    for block_idx in range(pid, NUM_BLOCKS, NUM_CORE):
        base_offset = block_idx * BLOCK_SIZE
        num_sub = tl.cdiv(BLOCK_SIZE, BLOCK_SIZE_SUB)

        for sub_idx in range(num_sub):
            offsets = base_offset + sub_idx * BLOCK_SIZE_SUB + tl.arange(0, BLOCK_SIZE_SUB)
            mask = offsets < N
            data = tl.load(data_ptr + offsets, mask=mask)
            result = data * 2.0 + 1.0
            tl.store(out_ptr + offsets, result, mask=mask)

def launch_optimized(data, out, use_dot=False):
    N = data.numel()
    NUM_CORE = get_npu_core_count(use_dot=use_dot)

    BLOCK_SIZE = max(32768, triton.next_power_of_2(triton.cdiv(N, 65535)))
    BLOCK_SIZE_SUB = 1024

    grid = (NUM_CORE,)
    optimized_kernel[grid](data, out, N, BLOCK_SIZE, BLOCK_SIZE_SUB)

x = torch.randn(1024 * 1024, device='npu', dtype=torch.float16)
y = torch.empty_like(x)
launch_optimized(x, y, use_dot=False)
```

## NPU 适配要点

1. **Grid 必须对齐物理核数**：不要使用 GPU 的 `cdiv(N, BLOCK_SIZE)` 作为 Grid 大小，应固定为物理核数并在核内循环
2. **优先使用 1D Grid**：2D Grid 在 NPU 上会被合并为 1D，直接使用 1D Grid 更直观高效
3. **注意 coreDim 上限**：确保 `grid_x * grid_y * grid_z <= 65535`，超出时使用 `TRITON_ALL_BLOCKS_PARALLEL` 或增大 BLOCK_SIZE
4. **区分 Vector 算子和 CV 算子**：纯 Vector 算子用 Vector Core 数量，含 `tl.dot` 的算子用 AI Core 数量
5. **避免分批调度开销**：Grid 大小超过物理核数时会产生分批调度，引入额外核启动和初始化开销
6. **使用 autotune 优化 BLOCK_SIZE**：通过 `@triton.autotune` 自动搜索最优分块参数

## 常见问题

**Q1: 为什么不能像 GPU 那样用 `cdiv(N, BLOCK_SIZE)` 作为 Grid？**

A: GPU 有数百个 SM，硬件自动调度 Thread Block，逻辑 Grid 与物理核解耦。NPU 只有 20~36 个 AI Core，超出物理核数的 program 会分批调度，引入核启动和初始化开销。固定 Grid 为物理核数并在核内循环是最优策略。

**Q2: 2D Grid `(4, 5)` 和 1D Grid `(20,)` 在 NPU 上有区别吗？**

A: 没有本质区别。2D Grid 在 NPU 适配中会被合并为 1D，`(4, 5)` 等效于 `(20,)`。推荐直接使用 1D Grid，代码更简洁。

**Q3: `TRITON_ALL_BLOCKS_PARALLEL` 环境变量什么时候可以使用？**

A: 仅当逻辑核间可并行（无数据依赖）时方可启用。启用后编译器自动将逻辑核数调整为物理核数。如果核间存在数据依赖，使用该选项可能导致正确性问题。

**Q4: 如何判断算子是 Vector 算子还是 CV 算子？**

A: 如果 kernel 中使用了 `tl.dot`，则为 CV 算子，Grid 应设为 AI Core 数量；否则为纯 Vector 算子，Grid 应设为 Vector Core 数量。

**Q5: coreDim 超限但增大 BLOCK_SIZE 又导致 UB 溢出怎么办？**

A: 使用两级分块策略：外层 BLOCK_SIZE 控制 coreDim 合规，内层 BLOCK_SIZE_SUB 控制 UB 使用量。参见上文"方案三"的代码示例。

## 相关文档

- [Grid/Program ID 与 AI Core 对应关系](./02-grid-and-program-id.md)
- [NPU 内存层次](./03-memory-model.md)
- [数据类型支持矩阵与约束](./04-data-types.md)
- [硬件架构概览](../../docs_ascendnpu_ir/00-Architecture/01-npu-hardware-overview.md)

## 源文件参考

- [programming_guide.md](https://github.com/triton-lang/triton-ascend/tree/main/docs/zh/programming_guide.md) - Triton 算子开发指南
- [migrate_from_gpu.md](https://github.com/triton-lang/triton-ascend/tree/main/docs/zh/migration_guide/migrate_from_gpu.md) - GPU 迁移指南
- [architecture_difference.md](https://github.com/triton-lang/triton-ascend/tree/main/docs/zh/migration_guide/architecture_difference.md) - 架构差异分析
- [core.py](https://github.com/triton-lang/triton-ascend/tree/main/python/triton/language/core.py) - Triton 语言核心定义
