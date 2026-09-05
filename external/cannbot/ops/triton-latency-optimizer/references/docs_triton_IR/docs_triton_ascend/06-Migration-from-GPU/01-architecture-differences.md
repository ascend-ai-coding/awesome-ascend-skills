# GPU vs NPU 架构差异

## 概述

将 Triton 算子从 GPU 迁移到华为昇腾 NPU 时，理解两种硬件架构的根本差异至关重要。GPU 采用 SIMT（Single Instruction Multiple Threads）执行模型，而 NPU 采用 SIMD（Single Instruction Multiple Data）执行模型，这种底层差异直接影响编程模型、内存管理和性能优化策略。本文系统梳理 GPU 与 NPU 在架构层面的核心差异，为迁移工作提供理论基础。

## 关键概念

| 概念 | GPU（NVIDIA） | NPU（Ascend） |
|------|---------------|---------------|
| 计算核心 | SM（Streaming Multiprocessor） | AI Core（Cube + Vector） |
| 执行模型 | SIMT（单指令多线程） | SIMD（单指令多数据）；910_95 额外支持 SIMT 模式 |
| 矩阵计算单元 | Tensor Core | Cube Unit |
| 向量计算单元 | CUDA Core | Vector Unit |
| 片上共享内存 | Shared Memory | UB（Unified Buffer） |
| 片上私有缓存 | L1 Cache | L1 Buffer |
| 全局内存 | Global Memory（HBM） | GM（Global Memory） |
| 线程组织 | Thread → Warp → Block → Grid | 无线程概念（A2/A3）；910_95 SIMT 模式下有线程概念 |
| Grid 本质 | 逻辑任务维度（与物理核解耦） | 物理核组映射（绑定 AI Core 拓扑） |
| 核数/维度限制 | Grid 维度/大小无硬限制 | Grid 大小 <= AI Core 总数，coreDim <= 65535 |

## 详细内容

### 1. 计算核心架构对比

#### GPU：SM（Streaming Multiprocessor）

GPU 的基本计算单元是 SM。每个 SM 包含多个 CUDA Core 和 Tensor Core：

- **CUDA Core**：标量/向量计算单元，负责逐元素运算（加、减、乘、除、激活函数等）
- **Tensor Core**：矩阵乘法加速单元，执行 D = A * B + C 形式的矩阵运算
- **Warp**：32 个线程组成一个 Warp，是 SM 的最小调度单位，Warp 内所有线程执行相同指令
- 一个 GPU 芯片通常包含数十到数百个 SM

#### NPU：AI Core（Cube + Vector）

NPU 的基本计算单元是 AI Core。每个 AI Core 包含两类计算单元：

- **Cube Unit（立方体计算单元）**：矩阵乘法加速单元，类似 GPU 的 Tensor Core，执行矩阵乘法运算
- **Vector Unit（向量计算单元）**：向量计算单元，类似 GPU 的 CUDA Core，执行逐元素运算
- AI Core 内部 Cube 和 Vector 以 1:2 比例配置（所有当前型号），一个 Cube Core 对应两个 Vector Core
- 一个 NPU 芯片通常包含数十个 AI Core

```text
GPU SM 结构:
┌─────────────────────────────────┐
│            SM                   │
│  ┌──────────┐  ┌──────────────┐ │
│  │CUDA Cores│  │ Tensor Cores │ │
│  │ (128个)  │  │   (4个)      │ │
│  └──────────┘  └──────────────┘ │
│  ┌──────────────────────────────┐│
│  │     Shared Memory / L1       ││
│  └──────────────────────────────┘│
└─────────────────────────────────┘

NPU AI Core 结构:
┌─────────────────────────────────┐
│          AI Core                │
│  ┌──────────┐  ┌──────────────┐ │
│  │Cube Unit │  │Vector Unit 0 │ │
│  │  (1个)   │  │  Vector Unit 1│ │
│  └──────────┘  └──────────────┘ │
│  ┌──────────────────────────────┐│
│  │     UB / L1 Buffer           ││
│  └──────────────────────────────┘│
└─────────────────────────────────┘
```

### 2. 执行模型差异：SIMT vs SIMD

#### SIMT（GPU）

SIMT 模型下，每个线程拥有独立的执行上下文（PC、寄存器），但同一 Warp 内的线程必须执行相同指令：

- 每个线程可以有不同的数据路径（分支发散时，Warp 内线程串行执行各分支）
- Warp 是最小调度单位，32 个线程同步执行
- 编程模型以线程为单位，开发者通过 `tl.arange` 等构造线程索引

#### SIMD（NPU）

SIMD 模式下，一条指令同时处理多个数据元素，没有线程的概念：

- Vector Unit 一次处理一个向量（多个元素），所有元素执行相同操作
- 不存在分支发散问题，但也不支持 Warp 级别的线程独立执行
- 编程模型以 AI Core 为单位，每个 AI Core 执行一个 Block

> **910_95 SIMT 模式**：910_95 系列采用 Reg-based 架构，额外支持 SIMT（Single Instruction Multiple Threads）执行模式。在 SIMT 模式下，Vector 操作基于寄存器进行，每个线程独立执行标量操作，类似 GPU 的线程级并行。SIMT 模式适合控制流密集的操作（如条件分支、动态索引），可通过 `compile_mode="simt_only"` 或 `compile_mode="unstructured_in_simt"` 启用。

```python
# GPU SIMT 思维：每个线程独立计算
pid = tl.program_id(axis=0)
offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
# tl.arange(0, BLOCK_SIZE) 在 GPU 上对应 BLOCK_SIZE 个线程的 ID

# NPU SIMD 思维：一个 AI Core 处理一个向量
pid = tl.program_id(axis=0)
offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
# tl.arange(0, BLOCK_SIZE) 在 NPU 上对应一个向量中的 BLOCK_SIZE 个元素索引
```

### 3. 内存层次对比

| 内存层级 | GPU | 容量（典型） | NPU | 容量（A2/A3） | 容量（910_95） |
|---------|-----|-------------|-----|--------------|---------------|
| 全局内存 | Global Memory (HBM) | 40-80 GB | GM (Global Memory) | 32-64 GB | 32-64 GB |
| 片上共享内存 | Shared Memory | 48-164 KB/SM | UB (Unified Buffer) | 192 KB/AI Core | 256 KB/AI Core |
| 片上私有缓存 | L1 Cache | 与 Shared Memory 共享 | L1 Buffer | 与 UB 独立 | 与 UB 独立 |
| 寄存器 | Register File | 256 KB/SM | - | - | RF 128KB + DCache 32~120KB（SIMT 模式） |

**关键差异**：

1. **UB vs Shared Memory**：NPU 的 UB 是 Vector Unit 的专用缓存，A2/A3 系列容量为 192 KB，910_95 系列为 256 KB。GPU 的 Shared Memory 由 Warp 内线程共享。UB 的使用需要严格管理，超出容量会导致编译错误（UB overflow）
2. **对齐要求**：NPU Vector 算子场景要求 32 字节访存对齐，Cube-Vector 融合算子场景要求 512 字节对齐。GPU 对对齐要求相对宽松
3. **访存模式**：NPU 更偏好连续访存，离散访存会导致性能严重下降甚至 scalar 退化

### 4. 计算单元对比：Tensor Core vs Cube Unit

| 维度 | GPU Tensor Core | NPU Cube Unit |
|------|----------------|---------------|
| 功能 | 矩阵乘法加速 | 矩阵乘法加速 |
| 典型操作 | D = A * B + C | D = A * B + C |
| Triton 映射 | `tl.dot()` | `tl.dot()` |
| 数据类型 | fp16, bf16, int8, tf32, fp8 | fp16, bf16, int8, fp32, fp8（910_95） |
| 输出精度 | 可配置（tf32, tf32x3, ieee, hf32） | 可配置 |
| 最小矩阵尺寸 | 16x16x16 | 16x16x16 |

**Triton 中的映射**：在 GPU 和 NPU 上，矩阵乘法均通过 `tl.dot()` 实现，但底层硬件执行方式不同。GPU 上 `tl.dot()` 由 Tensor Core 执行，NPU 上由 Cube Unit 执行。

### 5. 编程模型差异：Warp 级并行 vs AI Core 级并行

#### GPU 编程模型

```text
Grid → Block → Warp → Thread
 │      │      │      │
 │      │      │      └─ 最小执行单位
 │      │      └──────── 32个线程同步执行
 │      └─────────────── SM 上的调度单位
 └────────────────────── 整个 kernel 的线程组织
```

- `num_warps` 参数控制每个 Block 使用的 Warp 数量（默认 4，即 128 个线程）
- `num_stages` 参数控制软件流水线的阶段数
- Grid 可以是 1D/2D/3D，逻辑维度与物理核解耦

#### NPU 编程模型

```text
Grid → AI Core (Block)
 │      │
 │      └── 最小执行单位，一个 AI Core 执行一个 Block
 └────────── 整个 kernel 的核组织
```

- `num_warps` 参数在 NPU 上无效（硬件架构差异）
- `num_stages` 参数在 NPU 上无效
- Grid 优先使用 1D，2D 会被合并为 1D
- Grid 大小应对齐物理核数，超过物理核数会分批调度

```python
# GPU 写法：num_warps 控制并行度
kernel[grid](x, y, out, N, BLOCK_SIZE=1024, num_warps=8)

# NPU 写法：直接指定核数
kernel[n, 1, 1](x, y, out, N, BLOCK_SIZE=1024)
# n 为使用的 AI Core 数量，需 <= 物理核数
```

### 6. 并行调度差异

| 维度 | GPU | NPU |
|------|-----|-----|
| Grid 本质 | 逻辑任务维度（与物理核解耦） | 物理核组映射（绑定 AI Core 拓扑） |
| 核数/维度限制 | Grid 维度/大小无硬限制 | Grid 大小 <= AI Core 总数，coreDim <= 65535 |
| 超核调度 | 硬件自动调度，开销较小 | 分批调度，额外设备侧开销大 |
| 并发任务数 | 可远超物理 SM 数 | 最大 65535，建议等于物理核数 |
| 算子类型与核数 | 编译器和硬件自动决定 | Vector-only 算子 = Vector Core 数；含 tl.dot 算子 = AI Core 数 |

**核心原则**：NPU 上应将并发任务数配置为物理核数，避免分批调度开销。可通过 `driver.active.utils.get_device_properties` 接口获取物理核数。

### 7. 数据搬运流水线差异

#### GPU 数据搬运

GPU 的数据搬运通过 CUDA 的显式拷贝（`cudaMemcpy`）或隐式缓存完成：

- Global Memory → Shared Memory：通过 `__shared__` 内存和显式加载完成
- Shared Memory → Register：Warp 内线程各自加载所需数据
- 软件流水线：通过 `num_stages` 参数控制，编译器自动生成多阶段重叠执行

#### NPU 数据搬运

NPU 的数据搬运通过专用的 DMA 引擎完成，具有明确的流水线阶段：

- **MTE2（Memory Transfer Engine 2）**：GM → UB 的数据搬入
- **MTE3（Memory Transfer Engine 3）**：UB → GM 的数据搬出
- **MTE1**：L1 → UB 的数据搬运
- **Vector/Cube**：在 UB 上执行计算
- **multibuffer**：NPU 特有的流水并行数据搬运优化，通过 `triton.Config({'multibuffer': True})` 启用

```text
NPU 数据搬运流水线:
GM ──MTE2──> UB ──Vector/Cube──> UB ──MTE3──> GM
              ↑                      |
              └── multibuffer 双缓冲 ──┘
```

**关键差异**：
1. GPU 的 Shared Memory 由线程显式管理，NPU 的 UB 由编译器自动管理
2. NPU 的 multibuffer 机制需要额外 UB 空间（双缓冲时可用空间减半）
3. NPU 的数据搬运和计算可以并行（通过 multibuffer），但需要编译器支持

### 8. 同步机制差异

| 同步类型 | GPU | NPU |
|---------|-----|-----|
| Block 内同步 | `__syncthreads()` | `tl.debug_barrier()` |
| Warp 内同步 | 隐式同步（锁步执行） | 不适用（无 Warp 概念） |
| Block 间同步 | 不支持（需通过 Global Memory） | `sync_block_set/wait/all`（Ascend 扩展） |
| 流同步 | `cudaStreamSynchronize` | Host 侧同步 |
| 事件同步 | `cudaEvent` | NPU 事件机制 |

**NPU 特有的同步操作**（通过 `tl.extra.cann.extension` 提供）：

- `sync_block_set`：设置同步事件
- `sync_block_wait`：等待同步事件
- `sync_block_all`：全局同步

这些同步操作在解释器模式下为 no-op，仅在 NPU 实际运行时生效。

### 9. 数据类型支持差异

| 数据类型 | GPU | NPU (A2/A3) | NPU (910_95) | 说明 |
|---------|-----|-------------|-------------|------|
| int8 | 支持 | 支持 | 支持 | - |
| int16 | 支持 | 支持 | 支持 | - |
| int32 | 支持 | 支持 | 支持 | - |
| int64 | 支持 | 支持 | 支持 | Vector ADD/CMP 不支持，退化为 scalar |
| uint8 | 支持 | 部分不支持 | 部分不支持 | Block Pointer 等场景不支持 |
| uint16 | 支持 | 不支持 | 不支持 | 硬件限制 |
| uint32 | 支持 | 不支持 | 不支持 | 硬件限制 |
| uint64 | 支持 | 不支持 | 不支持 | 硬件限制 |
| fp16 | 支持 | 支持 | 支持 | - |
| fp32 | 支持 | 支持 | 支持 | - |
| fp64 | 支持 | 不支持 | 不支持 | 硬件限制 |
| bf16 | 支持 | 支持 | 支持 | - |
| fp8 | 支持 | 不支持 | 支持 | 910_95 支持 FP8 类型转换和 dot_scaled |
| bool | 不支持 | 不支持 | 不支持 | 两者均不支持 |

## NPU 适配要点

1. **放弃 GPU 逻辑 Grid 自由定义**：转为昇腾物理核组绑定模式，Grid 大小应对齐物理核数
2. **访存对齐**：Vector 算子场景 32 字节对齐，Cube-Vector 融合算子场景 512 字节对齐
3. **移除 GPU 专属参数**：`num_warps`、`num_stages`、`cache_modifier`、`eviction_policy` 等参数在 NPU 上无效
4. **Grid 优先 1D**：2D Grid 会被合并为 1D，建议直接使用 1D Grid
5. **UB 容量限制**：A2/A3 系列 UB 为 192 KB，910_95 系列为 256 KB，需严格控制单次处理的数据量
6. **数据类型选择**：避免使用 uint16/uint32/uint64/fp64，int64 在 Vector ADD/CMP 中会退化为 scalar

## 常见问题（Q&A）

**Q1：为什么 GPU 上正常的算子迁移到 NPU 后性能大幅下降？**

A：最常见的原因是 Grid 分核数过多。GPU 上大量 Block 可以由硬件自动调度到 SM 上执行，但 NPU 上超过物理核数的任务会分批调度，产生额外开销。建议将并发任务数设置为物理核数，并在核内做更细致的数据分块。

**Q2：NPU 上的 `num_warps` 参数有什么作用？**

A：由于 NPU 没有 Warp 的概念，`num_warps` 参数在 NPU 上无效。NPU 的并行粒度是 AI Core 级别，通过 Grid 的第一个维度控制使用的核数。

**Q3：为什么 NPU 不支持通过调整 stride 实现转置？**

A：NPU 的 Cube Unit 和 Vector Unit 对内存布局有严格要求，转置语义只能通过调整 `order` 参数来表达，不能通过交换 `stride` 参数实现。这是硬件层面的限制。

**Q4：A2/A3 和 910_95 的架构差异大吗？**

A: A2 和 A3 在编程模型上基本一致，主要差异在于具体的核数、频率和 UB 容量等硬件参数。A2/A3 系列 UB 均为 192 KB，Ascend910_95/950 系列为 256 KB。910_95 采用 Reg-based 架构，支持 SIMT 模式、L0C->UB 直通路径和 FP8 数据类型，与 A2/A3（Mem-based 架构）有显著差异。

## 相关文档

- [02-代码迁移模式](./02-code-migration-patterns.md)
- [03-迁移常见问题](./03-common-issues.md)
- [04-Block-Pointer-迁移注意事项](./04-block-pointer-migration.md)
- [architecture_difference.md](https://github.com/triton-lang/triton-ascend/tree/main/docs/zh/migration_guide/architecture_difference.md)
- [migrate_from_gpu.md](https://github.com/triton-lang/triton-ascend/tree/main/docs/zh/migration_guide/migrate_from_gpu.md)
