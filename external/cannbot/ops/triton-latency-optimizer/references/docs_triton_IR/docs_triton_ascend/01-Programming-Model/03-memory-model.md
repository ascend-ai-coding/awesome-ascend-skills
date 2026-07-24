# NPU 内存层次

## 概述

昇腾 NPU 采用多级存储架构，从大容量高延迟的全局内存到小容量低延迟的片上缓存，形成了 GM -> L1 -> L0A/L0B -> L0C（Cube 计算路径）和 GM -> UB（Vector 计算路径）两条核心数据通路。理解 NPU 的内存层次和数据通路，是编写高性能 Triton 算子的关键——它决定了数据搬运策略、分块大小选择以及存算并行的实现方式。

与 GPU 的 Global Memory -> Shared Memory -> Register 三级层次不同，NPU 的存储层次更加丰富，Cube 计算和 Vector 计算使用不同的片上缓存，且各级存储有严格的对齐要求。`tl.load`/`tl.store` 在 NPU 上映射为 GM 与 UB 之间的数据搬运，而 `tl.dot` 则走 GM -> L1 -> L0A/L0B -> Cube -> L0C -> GM 的完整 Cube 数据通路。

**关键词**：内存层次、GM、L1、UB、L0A、L0B、L0C、数据通路、MTE2、MTE1、FIX、对齐、tl.load、tl.store

## 关键概念

| 存储空间 | IR 标识符 | 大小（910B/93） | 大小（910_95） | 对齐要求 | 用途 |
|----------|-----------|-----------------|----------------|----------|------|
| GM (Global Memory) | `gm` | 数十 GB (HBM) | 数十 GB (HBM) | - | 外部全局存储 |
| L1 (Local Memory) | `cbuf` | 512KB | 512KB | 32B | Cube 输入缓存 |
| UB (Unified Buffer) | `ub` | 192KB | 256KB (编译器可用248KB) | 32B | Vector 工作区 |
| L0A | `ca` | 64KB | 64KB | 512B | 矩阵 A 输入缓存 |
| L0B | `cb` | 64KB | 64KB | 512B | 矩阵 B 输入缓存 |
| L0C | `cc` | 128KB | 256KB | 512B | 矩阵乘法结果缓存 |
| BT Buffer | - | 1KB | 1KB | 64B | Bias 数据缓存 |
| FP Buffer | - | 7KB | 7KB | 128B | FixPipe 中间缓冲区 |

## 详细内容

### 1. Ascend NPU 存储层次详解

#### 1.1 GM (Global Memory)

GM 是 NPU 外部的全局存储，通常为 HBM（High Bandwidth Memory），容量大但访问延迟高。所有输入数据和输出结果都存储在 GM 中。

- **容量**：数十 GB（取决于具体卡型号）
- **访问方式**：通过 MTE2/MTE3 流水线与片上存储交互
- **IR 地址空间**：`#hivm.address_space<gm>`（枚举值 1）

#### 1.2 L1 (Local Memory / cbuf)

L1 是 Cube 单元使用的一级缓存，存储矩阵乘法的输入数据。

- **容量**：512KB（所有系列）
- **对齐要求**：32B
- **数据来源**：通过 MTE2 从 GM 加载
- **数据去向**：通过 MTE1 搬运到 L0A/L0B/BT Buffer
- **IR 地址空间**：`#hivm.address_space<cbuf>`（枚举值 2）
- **布局转换**：支持 ND -> NZ 格式转换（GM -> L1 时）

#### 1.3 UB (Unified Buffer)

UB 是 Vector 单元使用的统一缓冲区，所有 Vector 计算的输入和输出都经过 UB。

- **容量**：192KB（910B/93 系列），256KB（910_95 系列；编译器可用 248KB，预留 8KB）
- **对齐要求**：32B
- **数据来源**：通过 MTE2 从 GM 加载
- **数据去向**：通过 MTE3 写回 GM
- **IR 地址空间**：`#hivm.address_space<ub>`（枚举值 6）
- **重要约束**：
  - Vector 算子场景要求 Tensor 尾轴大小能被 32B 整除
  - CV 算子场景要求 Tensor 尾轴大小能被 512B 整除
  - 开启 double buffer 后可用容量减半

#### 1.4 L0A / L0B

L0A 和 L0B 分别是矩阵 A 和矩阵 B 的输入缓存，仅供 Cube 单元使用。

- **容量**：各 64KB（所有系列）
- **对齐要求**：512B
- **数据来源**：通过 MTE1 从 L1 加载
- **IR 地址空间**：`#hivm.address_space<ca>`（枚举值 3）/ `#hivm.address_space<cb>`（枚举值 4）

#### 1.5 L0C

L0C 是矩阵乘法的结果缓存，存储 Cube 单元的输出。

- **容量**：128KB（910B/93 系列），256KB（910_95 系列）
- **对齐要求**：512B
- **数据来源**：Cube 计算结果
- **数据去向**：通过 FIX (FixPipe) 输出到 GM/L1/UB
- **IR 地址空间**：`#hivm.address_space<cc>`（枚举值 5）

### 2. 各级存储的对齐要求与访问延迟

| 存储级别 | 对齐要求 | 相对延迟 | 访问模式 | 备注 |
|----------|----------|----------|----------|------|
| GM | 无硬性要求 | 最高 | 随机/连续 | 外部存储，DMA 搬运 |
| L1 | 32B | 中 | 连续优先 | Cube 输入缓存 |
| UB | 32B | 中低 | 连续优先 | Vector 工作区 |
| L0A/L0B | 512B | 低 | 固定模式 | Cube 专用输入 |
| L0C | 512B | 低 | 固定模式 | Cube 专用输出 |

**对齐约束详解**：

- **VV 类算子**（纯 Vector）：UB 要求 Tensor 尾轴大小能被 32B 整除
- **CV 类算子**（Cube+Vector）：要求 Tensor 尾轴大小能被 512B 整除
- 尾轴长度不足时会自动补齐，可能导致性能恶化

### 3. 数据通路

#### 3.1 流水线类型定义

| 流水线 | 枚举值 | 说明 |
|--------|--------|------|
| PIPE_S | 0 | 标量流水线 |
| PIPE_V | 1 | 向量计算流水线 |
| PIPE_M | 2 | 矩阵计算流水线 |
| PIPE_MTE1 | 3 | L1 到 L0A/L0B/BT Buffer 的单向数据通路 |
| PIPE_MTE2 | 4 | GM/L2 到 L1/UB 的双向数据通路 |
| PIPE_MTE3 | 5 | UB 到 GM 的单向数据通路 |
| PIPE_FIX | 10 | FixPipe 数据通路（L0C -> GM/L1/UB） |

#### 3.2 Cube 计算路径（矩阵乘法）

```
GM --[MTE2]--> L1 --[MTE1]--> L0A/L0B/BT Buffer --[M]--> L0C --[FIX]--> GM
```

详细步骤：

1. **MTE2**：从 GM 加载矩阵 A、矩阵 B 数据到 L1
2. **MTE1**：从 L1 加载矩阵 A 数据到 L0A，矩阵 B 数据到 L0B
3. **M**：Cube 执行矩阵乘法，结果写入 L0C
4. **FIX**：L0C 数据通过 FixPipe 输出到 GM（可附带量化/激活操作）

#### 3.3 Vector 计算路径

```
GM --[MTE2]--> UB --[V]--> UB --[MTE3]--> GM
```

详细步骤：

1. **MTE2**：从 GM 加载数据到 UB
2. **V**：Vector 单元执行向量计算
3. **MTE3**：计算结果从 UB 存储到 GM

#### 3.4 Cube-Vector 混合计算路径

**Ascend910_95 系列**（使用紧耦合缓冲区）：

```
GM --[MTE2]--> L1 --[MTE1]--> L0A/L0B --[M]--> L0C --[FIX]--> UB --[V]--> UB --[MTE3]--> GM
```

910_95 系列支持 L0C -> UB 直通路径，Cube 计算结果可直接通过 FixPipe 输出到 UB 供 Vector 操作使用，无需经过 GM 中转。

**非 Ascend910_95 系列**：

```
GM --[MTE2]--> L1 --[MTE1]--> L0A/L0B --[M]--> L0C --[FIX]--> GM --[MTE2]--> UB --[V]--> UB --[MTE3]--> GM
```

非 910_95 系列不支持 L0C -> UB 通路，需要先将 L0C 输出到 GM，再通过 MTE2 加载到 UB 进行向量计算，多了一次 GM 往返。

#### 3.5 数据通路汇总表

| 数据通路 | 流水线 | 操作 | 说明 |
|----------|--------|------|------|
| GM -> L1 | MTE2 | `copy_gm_to_cbuf` | 全局内存到 L1 缓存 |
| L1 -> GM | MTE2 | `copy_cbuf_to_gm` | L1 缓存到全局内存 |
| GM -> UB | MTE2 | `hivm.load` | 全局内存到统一缓冲区 |
| UB -> GM | MTE3 | `hivm.store` | 统一缓冲区到全局内存 |
| UB -> UB | V | 向量计算或 `hivm.copy` | 向量计算或 UB 内复制 |
| UB -> L1 | MTE3 | `hivm.copy` | UB 到 L1（910_95） |
| L1 -> L0A | MTE1 | 内部指令 | L1 到矩阵 A 缓存 |
| L1 -> L0B | MTE1 | 内部指令 | L1 到矩阵 B 缓存 |
| L0A/L0B -> L0C | M | Cube 计算 | 矩阵乘法 |
| L0C -> GM | FIX | `hivm.fixpipe` | L0C 到全局内存 |
| L0C -> L1 | FIX | `hivm.fixpipe` | L0C 到 L1 缓存 |
| L0C -> UB | FIX | `hivm.fixpipe` | L0C 到 UB（910_95 特有） |

### 4. 与 GPU 内存层次的对比

| 维度 | GPU | NPU (Ascend) |
|------|-----|-------------|
| 全局存储 | Global Memory (HBM) | GM (HBM) |
| 片上共享存储 | Shared Memory (可编程) | L1 (Cube 用) / UB (Vector 用) |
| 寄存器文件 | Register File | 无（A2/A3）；DCache + RF 128KB（910_95 SIMT 模式） |
| 矩阵计算缓存 | Tensor Core 内部缓存 | L0A/L0B/L0C 专用缓存 |
| 编程模型 | 手动管理 Shared Memory | 编译器自动管理片上存储 |
| 数据搬运 | 显式 `__ldg` / `__stg` | `tl.load` / `tl.store`（编译器映射为 DMA） |
| 存算并行 | Software Pipelining | MultiBuffer（默认开启） |

**核心差异**：

1. **GPU 的 Shared Memory 由开发者手动管理**，NPU 的 L1/UB 由编译器自动管理
2. **GPU 的 Register File 是线程私有的**，NPU 的 A2/A3 系列没有 Register File 概念，UB 兼具 Shared Memory 和 Register 的功能；910_95 系列采用 Reg-based 架构，在 SIMT 模式下有 128KB RF（Register File）和 32~120KB DCache
3. **NPU 的 Cube 和 Vector 使用不同的片上存储**，而 GPU 的 CUDA Core 和 Tensor Core 共享同一套 Shared Memory
4. **NPU 有专用的 L0A/L0B/L0C 缓存**用于矩阵乘法，GPU 的 Tensor Core 内部缓存对开发者不可见

### 5. tl.load/tl.store 在 NPU 上的数据通路映射

#### 5.1 tl.load 映射

`tl.load` 在 NPU 上映射为 **MTE2 流水线的 GM -> UB 数据搬运**：

```python
# Triton 代码
x = tl.load(x_ptr + offsets, mask=mask)

# NPU 内部映射
# 1. MTE2: GM -> UB (通过 hivm.load 操作)
# 2. 支持 Padding 和隐式转置
```

#### 5.2 tl.store 映射

`tl.store` 在 NPU 上映射为 **MTE3 流水线的 UB -> GM 数据搬运**：

```python
# Triton 代码
tl.store(out_ptr + offsets, result, mask=mask)

# NPU 内部映射
# 1. MTE3: UB -> GM (通过 hivm.store 操作)
# 2. 支持原子操作和隐式转置
```

#### 5.3 tl.dot 映射

`tl.dot` 在 NPU 上走完整的 Cube 数据通路：

```python
# Triton 代码
acc = tl.dot(a, b, acc)

# NPU 内部映射
# 1. MTE2: GM -> L1 (加载矩阵 A、B)
# 2. MTE1: L1 -> L0A/L0B (搬运到 Cube 输入缓存)
# 3. M: L0A * L0B -> L0C (Cube 执行矩阵乘法)
# 4. FIX: L0C -> GM/L1/UB (FixPipe 输出结果)
```

### 6. 代码示例：显式内存管理（使用 Buffer 模型）

Triton-Ascend 提供了 Buffer 模型扩展 API，允许开发者显式管理片上存储。这些 API 定义在 [core.py](https://github.com/triton-lang/triton-ascend/tree/main/third_party/ascend/language/cann/extension/core.py) 中。

#### 6.1 地址空间定义

```python
from triton.language.core import int64
from triton.extension.buffer import language as bl
from triton.backends.ascend.language.cann.extension.core import (
    ascend_address_space, CORE, PIPE, copy, fixpipe,
    FixpipeDMAMode, FixpipeDualDstMode, sub_vec_id, sub_vec_num,
    sync_block_set, sync_block_wait, sync_block_all, debug_barrier,
    SYNC_IN_VF
)

# 地址空间常量
GM = ascend_address_space.GM     # 全局内存
L1 = ascend_address_space.L1     # Local Memory (cbuf)
UB = ascend_address_space.UB     # Unified Buffer
L0A = ascend_address_space.L0A   # 矩阵 A 缓存
L0B = ascend_address_space.L0B   # 矩阵 B 缓存
L0C = ascend_address_space.L0C   # 矩阵乘法结果缓存
```

#### 6.2 UB 到 L1 的数据拷贝

```python
@builtin
def copy(src, dst, _builder=None):
    """
    将数据从 UB 拷贝到 UB 或 L1。
    src: 位于 UB 的源数据
    dst: 位于 UB 或 L1 的目标缓冲区
    """
    return semantic.copy(src, dst, _builder)
```

#### 6.3 FixPipe 操作（L0C -> UB，仅 910_95 系列）

```python
from triton.backends.ascend.language.cann.extension.core import (
    fixpipe, FixpipeDMAMode, FixpipeDualDstMode
)

@triton.jit
def cv_fusion_kernel(...):
    # ... Cube 计算后，L0C 中有矩阵乘法结果

    # 使用 FixPipe 将 L0C 结果直接搬运到 UB
    fixpipe(
        src=l0c_tensor,           # L0C 中的源数据
        dst=ub_buffer,            # UB 中的目标缓冲区
        dma_mode=FixpipeDMAMode.NZ2ND,      # NZ 到 ND 布局转换
        dual_dst_mode=FixpipeDualDstMode.NO_DUAL  # 不使用双目标模式
    )

    # 之后 Vector 可以直接在 UB 上操作 Cube 的结果
    result = vector_op(ub_buffer)
```

#### 6.4 Cube-Vector 同步

```python
from triton.backends.ascend.language.cann.extension.core import (
    sync_block_set, sync_block_wait, sync_block_all
)

@triton.jit
def cv_sync_kernel(...):
    event_id = 0  # 0 ~ 15

    # Cube 端设置同步信号
    sync_block_set("cube", "vector", event_id)

    # ... Cube 计算完成后 ...

    # Vector 端等待同步信号
    sync_block_wait("cube", "vector", event_id)

    # 所有 Vector Core 同步
    sync_block_all("all_vector", event_id)

    # 所有 Cube Core 同步
    sync_block_all("all_cube", event_id)
```

#### 6.5 获取子 Vector Core ID

```python
from triton.backends.ascend.language.cann.extension.core import sub_vec_id, sub_vec_num

@triton.jit
def sub_vector_kernel(...):
    # 每个 AI Core 有 2 个 Vector Core
    vec_id = sub_vec_id()      # 获取当前 Vector Core 索引 (0 或 1)
    vec_num = sub_vec_num()    # 获取每个 AI Core 的 Vector Core 数量 (2)
```

### 7. 存算并行与 MultiBuffer

#### 7.1 存算并行原理

Triton-Ascend 支持两种数据处理模式：

- **存算串行**：先从 GM 搬运数据到片上内存，完成计算后，再搬运下一批数据。存在空闲等待时间
- **存算并行**：在搬运第一批数据的同时，已开始对其执行计算；继续搬运第二批数据，形成"搬运+计算"重叠的流水线操作

#### 7.2 MultiBuffer 机制

编译器默认配置 `multibuffer=True`（A2/A3 系列）或 `multibuffer=False`（910_95 系列），自动开启/关闭存算并行。开启后片上内存可用容量减半（因为需要双缓冲）：

| 配置 | UB 可用容量 | 说明 |
|------|-----------|------|
| multibuffer=True（A2/A3 默认） | 96KB（910B），128KB（910_95） | 双缓冲，存算并行 |
| multibuffer=False（910_95 默认） | 192KB（910B），256KB（910_95） | 单缓冲，存算串行 |

```python
# 在 autotune 中配置 multibuffer
@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE': 1024, 'multibuffer': True}),
        triton.Config({'BLOCK_SIZE': 2048, 'multibuffer': False}),
    ],
    key=['N'],
)
@triton.jit
def my_kernel(...):
    ...
```

#### 7.3 Tiling 优化示例

当数据量超过片上内存容量时，需要进行 Tiling（分块）处理：

```python
@triton.jit
def tiled_kernel(data_ptr, N,
                 BLOCK_SIZE: tl.constexpr,
                 BLOCK_SIZE_SUB: tl.constexpr):
    pid = tl.program_id(0)
    base_offset = pid * BLOCK_SIZE
    num_sub_blocks = BLOCK_SIZE // BLOCK_SIZE_SUB

    for sub_idx in range(num_sub_blocks):
        sub_offset = base_offset + sub_idx * BLOCK_SIZE_SUB
        offsets = sub_offset + tl.arange(0, BLOCK_SIZE_SUB)
        mask = offsets < N
        data = tl.load(data_ptr + offsets, mask=mask)
        data = data * 2.0
        tl.store(data_ptr + offsets, data, mask=mask)
```

## NPU 适配要点

1. **UB 容量有限**：910B 系列为 192KB，开启 double buffer 后仅 96KB；910_95 系列为 256KB，开启 double buffer 后为 128KB；注意控制单次搬运的数据量
2. **对齐要求严格**：VV 算子 32B 对齐，CV 算子 512B 对齐；尾轴不足会自动补齐，影响性能
3. **Cube 和 Vector 使用不同存储**：Cube 用 L1/L0A/L0B/L0C，Vector 用 UB；CV 融合算子需要通过 FixPipe 或 GM 中转
4. **910_95 系列优势**：支持 L0C -> UB 直通路径，CV 融合算子性能更优
5. **910_95 系列默认关闭 MultiBuffer**：A2/A3 系列默认开启存算并行，910_95 系列默认关闭；开启后可用 UB 容量减半
6. **UB OVERFLOW 处理**：遇到 `ub overflow` 错误时，减小 BLOCK_SIZE 或引入 BLOCK_SIZE_SUB 子块分块

## 常见问题

**Q1: 为什么会出现 UB OVERFLOW 错误？**

A: 当单次搬运的数据量加上计算中间结果超过了 UB 的可用容量时触发。910B 系列 UB 为 192KB（开启 double buffer 后仅 96KB），910_95 系列 UB 为 256KB（开启 double buffer 后为 128KB）。解决方法是减小 BLOCK_SIZE 或引入 BLOCK_SIZE_SUB 子块分块。

典型错误信息：
```
ub overflow, requires 3072256 bits while 1572864 bits available!
```

**Q2: tl.load 在 NPU 上走哪条数据通路？**

A: `tl.load` 映射为 MTE2 流水线的 GM -> UB 数据搬运。数据从全局内存通过 DMA 搬运到统一缓冲区，供 Vector 单元使用。

**Q3: tl.dot 在 NPU 上的完整数据通路是什么？**

A: `tl.dot` 走完整的 Cube 数据通路：GM -> L1（MTE2）-> L0A/L0B（MTE1）-> Cube 计算（M）-> L0C -> GM/L1/UB（FIX）。

**Q4: 910_95 系列的 L0C -> UB 通路有什么优势？**

A: 在 CV 融合算子中，910_95 系列的 L0C -> UB 通路允许 Cube 计算结果直接通过 FixPipe 输出到 UB 供 Vector 操作使用，无需经过 GM 中转，减少了一次 GM 往返，大幅降低数据搬运开销。

**Q5: 如何选择是否开启 MultiBuffer？**

A: 默认开启。当数据量较小、UB 容量紧张时，可以关闭 MultiBuffer 以获得更大的可用 UB 空间，但会失去存算并行的性能优势。建议通过 autotune 对比两种配置的性能。

**Q6: 如何处理尾轴不对齐导致的性能问题？**

A: 对于 shape 为 (2048, 3) 的 Tensor，尾轴 3 会被自动补齐到 32B 对齐，导致性能恶化。可以使用"借轴转置"技巧：将长轴裂出一根对齐轴借给短轴，让两个轴都对齐。参见 [programming_guide.md](https://github.com/triton-lang/triton-ascend/tree/main/docs/zh/programming_guide.md) 中的详细示例。

## 相关文档

- [SPMD 模型在 NPU 上的映射](./01-spmd-on-npu.md)
- [Grid/Program ID 与 AI Core 对应关系](./02-grid-and-program-id.md)
- [数据类型支持矩阵与约束](./04-data-types.md)
- [硬件架构概览](../../docs_ascendnpu_ir/00-Architecture/01-npu-hardware-overview.md)

## 源文件参考

- [00-Architecture](../../docs_ascendnpu_ir/00-Architecture/) - 硬件架构概览
- [programming_guide.md](https://github.com/triton-lang/triton-ascend/tree/main/docs/zh/programming_guide.md) - Triton 算子开发指南
- [core.py (extension)](https://github.com/triton-lang/triton-ascend/tree/main/third_party/ascend/language/cann/extension/core.py) - NPU 扩展 API
- [NPUTargetSpec.td](../../AscendNPU-IR/bishengir/include/bishengir/Dialect/HACC/Targets/NPUTargetSpec.td) - NPU 目标规格定义
- [HIVMDMAOps.td](../../AscendNPU-IR/bishengir/include/bishengir/Dialect/HIVM/IR/HIVMDMAOps.td) - DMA 操作定义
