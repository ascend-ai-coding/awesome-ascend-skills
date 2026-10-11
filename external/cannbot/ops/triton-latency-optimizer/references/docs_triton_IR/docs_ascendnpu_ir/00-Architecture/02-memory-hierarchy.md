# 内存层次详解

> 关键词：AddressSpace, GM, L1, L0A, L0B, L0C, UB, MTE2, MTE1, MTE3, FIX, 数据通路, 随路操作, 紧耦合缓冲区

## 概述

Ascend NPU 的内存层次是其计算性能的关键基础。与通用处理器不同，NPU 的各级存储空间与特定的计算单元和执行流水线紧密绑定，数据在不同存储层次之间的搬运由专用的 DMA 引擎（MTE 系列 Pipeline）完成，而非通过统一的缓存层次自动管理。

理解内存层次对于编写高效的 HIVM IR 至关重要：每个 `memref` 必须通过 `#hivm.address_space<...>` 属性标注其所在的存储空间，每个数据搬运操作必须使用正确的 Pipeline，且不同存储空间之间的数据通路存在严格的硬件约束。

本文档从 [NPUTargetSpec.td](https://gitcode.com/Ascend/AscendNPU-IR/tree/master/bishengir/include/bishengir/Dialect/HACC/Targets/NPUTargetSpec.td) 和 [HIVMAttrs.td](https://gitcode.com/Ascend/AscendNPU-IR/tree/master/bishengir/include/bishengir/Dialect/HIVM/IR/HIVMAttrs.td) 中精确提取各级存储的容量、对齐要求和数据通路信息。

## 完整内存层次

### 通用可寻址存储空间

| 层次 | IR 标识符 | 枚举值 | 说明 | 所属计算单元 |
|------|-----------|--------|------|-------------|
| GM | `gm` | 1 | 全局内存 (HBM/L2)，设备外部存储 | 所有单元共享 |
| L1 | `cbuf` | 2 | 一级缓存 | Cube 单元 |
| L0A | `ca` | 3 | 矩阵 A 输入缓存 | Cube A 端 |
| L0B | `cb` | 4 | 矩阵 B 输入缓存 | Cube B 端 |
| L0C | `cc` | 5 | 矩阵乘法结果缓存 | Cube C 端 |
| UB | `ub` | 6 | 统一缓冲区 | Vector 单元 |

### 专用硬件缓冲区

| 缓冲区 | 大小 | 对齐 | 说明 | 访问方式 |
|--------|------|------|------|----------|
| BT Buffer (BiasTable) | 1KB | 64B | 存放矩阵乘法的 Bias 数据 | 通过 `copy_cbuf_to_bt` 从 L1 拷贝 |
| FP Buffer (FixPipe) | 7KB | 128B | FixPipe 流水线的中间缓冲区 | 通过 `hivm.fixpipe` 隐式使用 |

## 各级存储的容量和对齐要求

### Ascend910B / 910_93 系列

源文件：[NPUTargetSpec.td:64-74](https://gitcode.com/Ascend/AscendNPU-IR/tree/master/bishengir/include/bishengir/Dialect/HACC/Targets/NPUTargetSpec.td#L64-L74)

| 存储空间 | 大小 | 对齐要求 | 源码值（bits） |
|----------|------|----------|---------------|
| UB | 192KB | 32B | UbSize=1572864, UbAlignSize=256 |
| L1 | 512KB | 32B | L1Size=4194304, L1AlignSize=256 |
| L0A | 64KB | - | L0aSize=524288 |
| L0B | 64KB | - | L0bSize=524288 |
| L0C | 128KB | 512B | L0cSize=1048576, L0cAlignSize=4096 |

### Ascend310B 系列

源文件：[NPUTargetSpec.td:156-166](https://gitcode.com/Ascend/AscendNPU-IR/tree/master/bishengir/include/bishengir/Dialect/HACC/Targets/NPUTargetSpec.td#L156-L166)

| 存储空间 | 大小 | 对齐要求 | 源码值（bits） |
|----------|------|----------|---------------|
| UB | 256KB | 32B | UbSize=2097152, UbAlignSize=256 |
| L1 | 1024KB | 32B | L1Size=8388608, L1AlignSize=256 |
| L0A | 64KB | - | L0aSize=524288 |
| L0B | 64KB | - | L0bSize=524288 |
| L0C | 128KB | 512B | L0cSize=1048576, L0cAlignSize=4096 |

### Ascend910_95 / 950PR / 950DT 系列

源文件：[NPUTargetSpec.td:196-208](https://gitcode.com/Ascend/AscendNPU-IR/tree/master/bishengir/include/bishengir/Dialect/HACC/Targets/NPUTargetSpec.td#L196-L208)

| 存储空间 | 大小 | 对齐要求 | 源码值（bits） |
|----------|------|----------|---------------|
| UB | 248KB（预留 8KB） | 32B | UbSize=2031616, UbAlignSize=256 |
| DCache | 32KB ~ 120KB | - | MinimalDCacheSize=262144, MaximumDCacheSize=983040 |
| L1 | 512KB | 32B | L1Size=4194304, L1AlignSize=256 |
| L0A | 64KB | - | L0aSize=524288 |
| L0B | 64KB | - | L0bSize=524288 |
| L0C | 256KB | 512B | L0cSize=2097152, L0cAlignSize=4096 |

## 数据通路详解

### AddressSpace 枚举与硬件存储的完整映射表

源文件：[HIVMAttrs.td:171-197](https://gitcode.com/Ascend/AscendNPU-IR/tree/master/bishengir/include/bishengir/Dialect/HIVM/IR/HIVMAttrs.td#L171-L197)

| 枚举值 | C++ 符号 | 数值 | IR 标识符 | 硬件存储 | 说明 |
|--------|---------|------|-----------|---------|------|
| Zero | `AddressSpace::Zero` | 0 | `zero` | - | 默认/零地址空间 |
| GM | `AddressSpace::GM` | 1 | `gm` | HBM/L2 | 全局内存 |
| L1 | `AddressSpace::L1` | 2 | `cbuf` | L1 Cache | Cube 一级缓存 |
| L0A | `AddressSpace::L0A` | 3 | `ca` | L0A Buffer | 矩阵 A 输入缓存 |
| L0B | `AddressSpace::L0B` | 4 | `cb` | L0B Buffer | 矩阵 B 输入缓存 |
| L0C | `AddressSpace::L0C` | 5 | `cc` | L0C Buffer | 矩阵乘法结果缓存 |
| UB | `AddressSpace::UB` | 6 | `ub` | UB | 统一缓冲区 |

IR 使用示例：

```mlir
memref<?x?x?x?xf32, #hivm.address_space<cbuf>>
memref<?x?x?x?xf32, #hivm.address_space<cc>>
memref<256x256xf16, #hivm.address_space<gm>>
```

### 源-目标地址空间到 Pipeline 的映射

源文件：[HIVMDMAOps.cpp:616-622](https://gitcode.com/Ascend/AscendNPU-IR/tree/master/bishengir/lib/Dialect/HIVM/IR/HIVMDMAOps.cpp#L616-L622)

| 源地址空间 | 目标地址空间 | Pipeline | IR 操作 | 说明 |
|-----------|-------------|----------|---------|------|
| GM | L1 | PIPE_MTE2 | `hivm.nd2nz` | GM 到 L1，支持 ND->NZ 转换 |
| GM | UB | PIPE_MTE2 | `hivm.load` | GM 到 UB，支持 Padding |
| L1 | GM | PIPE_MTE2 | `copy_cbuf_to_gm` | L1 到 GM，支持 NZ->ND 转换 |
| L1 | L0A | PIPE_MTE1 | 内部指令 | L1 到矩阵 A 缓存 |
| L1 | L0B | PIPE_MTE1 | 内部指令 | L1 到矩阵 B 缓存 |
| L1 | BT Buffer | PIPE_MTE1 | `copy_cbuf_to_bt` | L1 到 Bias Table 缓存 |
| L1 | UB | PIPE_MTE1 | `hivm.l12ub` | L1 到 UB |
| L0A/L0B | L0C | PIPE_M | Cube 计算 | 矩阵乘法 |
| L0C | GM | PIPE_FIX | `hivm.fixpipe` | L0C 到全局内存 |
| L0C | L1 | PIPE_FIX | `hivm.fixpipe` | L0C 到 L1 缓存 |
| L0C | UB | PIPE_FIX | `hivm.fixpipe` | L0C 到 UB（仅 950 系列） |
| UB | UB | PIPE_V | `hivm.copy` | UB 内复制 |
| UB | GM | PIPE_MTE3 | `hivm.store` | UB 到全局内存 |
| UB | L1 | PIPE_MTE3 | `hivm.copy` | UB 到 L1（仅 950 系列） |

### Cube 数据通路

Cube 计算路径涉及从 GM 加载数据到 L1，再从 L1 加载到 L0A/L0B，经 Cube 计算后结果写入 L0C，最后通过 FixPipe 输出。

```
GM ──[MTE2]──▶ L1 ──[MTE1]──▶ L0A/L0B/BT Buffer ──[M]──▶ L0C ──[FIX]──▶ GM/L1/UB
```

详细步骤：

1. **MTE2**: 从 GM 加载矩阵 A、矩阵 B 数据到 L1
2. **MTE2**: 从 GM 加载 Bias 数据到 L1
3. **MTE1**: 从 L1 加载矩阵 A 数据到 L0A
4. **MTE1**: 从 L1 加载矩阵 B 数据到 L0B
5. **MTE1**: 从 L1 加载 Bias 数据到 BT Buffer
6. **M**: Cube 执行矩阵乘法，结果写入 L0C
7. **FIX**: L0C 数据通过 FixPipe 输出到 GM/L1/UB

### Vector 数据通路

Vector 计算路径从 GM 加载数据到 UB，在 UB 中完成向量计算后写回 GM。

```
GM ──[MTE2]──▶ UB ──[V]──▶ UB ──[MTE3]──▶ GM
```

### 910_95 特殊通路

Ascend950 架构引入了两条特殊数据通路：

**L0C -> UB 直通通路**：Cube 计算结果可直接通过 FixPipe 输出到 UB，无需经过 GM 中转。这使得 Cube-Vector 混合计算路径更高效：

```
910_95: GM ──[MTE2]──▶ L1 ──[MTE1]──▶ L0A/L0B ──[M]──▶ L0C ──[FIX]──▶ UB ──[V]──▶ UB ──[MTE3]──▶ GM
非910_95: GM ──[MTE2]──▶ L1 ──[MTE1]──▶ L0A/L0B ──[M]──▶ L0C ──[FIX]──▶ GM ──[MTE2]──▶ UB ──[V]──▶ UB ──[MTE3]──▶ GM
```

**UB -> L1 通路**：Vector 处理后的数据可从 UB 搬运到 L1，供后续 Cube 操作使用。

## 随路操作汇总表

随路操作是指在数据搬运过程中，由硬件 DMA 引擎自动完成的附加操作，无需额外的计算指令。

| Pipeline | 数据流向 | 支持的随路操作 | IR 属性/操作 | 备注 |
|----------|---------|---------------|-------------|------|
| **MTE1** | L1 -> L0A/L0B | 矩阵转置 | `a_transpose`/`b_transpose` | 在 `mmadL1` 等操作中设置 |
| **MTE1** | L1 -> L0A/L0B | 布局转换 | zN <-> nZ | 支持格式互转 |
| **MTE2** | GM -> L1 | ND -> NZ 布局转换 | `hivm.nd2nz` | 将 ND 格式转为 NZ 格式 |
| **MTE2** | L1 -> GM | NZ -> ND 布局转换 | `hivm.nz2nd` | 将 NZ 格式转为 ND 格式 |
| **MTE2** | GM -> UB | Padding | `pad_mode`, `pad_value`, `left_padding_num`, `right_padding_num` | 在 `hivm.load` 中设置 |
| **MTE2** | GM -> UB | 隐式转置 | `may_implicit_transpose_with_last_axis` | 在 `hivm.load` 中设置 |
| **MTE3** | UB -> GM | 原子操作 | `atomic_kind` | 支持 add, max, min, and, or, xor, CAS, XCHG |
| **MTE3** | UB -> GM | 隐式转置 | `may_implicit_transpose_with_last_axis` | 在 `hivm.store` 中设置 |
| **FIX** | L0C -> GM/L1/UB | 预量化 | `pre_quant` | FP32->FP16, FP32->BF16, INT32->INT8 |
| **FIX** | L0C -> GM/L1/UB | 预激活 | `pre_relu` | ReLU, Leaky ReLU, P-ReLU |
| **FIX** | L0C -> GM/L1/UB | 布局转换 | `dma_mode` | NZ2ND, NZ2DN, NZ2NZ |
| **FIX** | L0C -> UB | 双目标模式 | `dual_dst_mode` | ROW_SPLIT, COLUMN_SPLIT（仅 950 系列） |
| **FIX** | L0C -> GM/L1/UB | Channel Split | `channel_split` | 仅 950 系列 |

## 紧耦合缓冲区（TightlyCoupledBuffer）

源文件：[HIVMAttrs.td:1010-1017](https://gitcode.com/Ascend/AscendNPU-IR/tree/master/bishengir/include/bishengir/Dialect/HIVM/IR/HIVMAttrs.td#L1010-L1017)

紧耦合缓冲区是 **Ascend950 架构特有** 的 CV（Cube-Vector）通信机制，用于在 Cube 操作和 Vector 操作之间高效传递数据，无需经过全局内存中转。

**IR 表示**：

```mlir
#hivm.tightly_coupled_buffer<id : optional<i32>>
```

**工作原理**：

紧耦合缓冲区通过 `InsertCVTightCoupledBuffer` Pass 在 Fixpipe 和 Vector 操作之间插入，支持两种数据搬运模式：

| 模式 | 数据流向 | 说明 |
|------|---------|------|
| MoveToUb | L0C -> UB | 将 Cube 计算结果从 L0C 直接搬运到 UB |
| MoveToL1 | UB -> L1 | 将 Vector 处理后的数据从 UB 搬运到 L1 |

**Pipeline 选择逻辑**：

```
if (isAscend950(target)) {
    if (enableLayoutOptimization) {
        InsertCVDataMovement    // A5 新布局优化路径
    } else {
        InsertCVTightCoupledBuffer  // 传统紧耦合缓冲区路径
    }
} else {
    InsertLoadStoreForMixCV    // 非 950 设备的混合 CV 路径
}
```

## 数据流 ASCII 图

### Ascend910B / 910_93 架构

```
+-----------------------------------------------------------------------------+
|                          Global Memory (GM / HBM)                           |
+---------------------------------------+-------------------------------------+
                                        |
                    +-------------------+-------------------+
                    |                                       |
              +-----v-----+                           +-----v-----+
              |   MTE2    |                           |   MTE2    |
              |  GM -> L1 |                           |  GM -> UB |
              |  (双向)   |                           |  (单向)   |
              +-----+-----+                           +-----+-----+
                    |                                       |
                    v                                       v
        +-----------------------+                   +-----------------+
        |          L1           |                   |       UB        |
        |    (cbuf, 512KB)      |                   |   (ub, 192KB)   |
        |    Cube输入缓存       |                   |  Vector工作区   |
        +-----------+-----------+                   +--------+--------+
                    |                                        |
        +-----------+-----------+                            |
        |           |           |                            |
  +-----v-----+ +---v---+ +-----v-----+                      |
  |   MTE1    | | MTE1  | |   MTE1    |                      |
  | L1 -> L0A | |L1->L0B| |L1 -> BT Buf|                     |
  +-----+-----+ +---+---+ +-----+-----+                      |
        |           |           |                            |
        v           v           v                            |
  +-----------+ +-----------+ +-----------+                   |
  |    L0A    | |    L0B    | | BT Buffer |                   |
  | (ca,64KB) | | (cb,64KB) | |   (1KB)   |                   |
  | 矩阵A输入  | | 矩阵B输入  | | Bias数据  |                   |
  +-----+-----+ +-----+-----+ +-----+-----+                   |
        |             |             |                          |
        +-------------+-------------+                          |
                      |                                        |
                      v                                        |
            +------------------+                               |
            |      Cube        |                               |
            |   (MatMul)       |                               |
            +--------+---------+                               |
                     |                                         |
                     v                                         |
            +------------------+                               |
            |      L0C         |                               |
            |   (cc, 128KB)    |                               |
            |   矩阵乘法结果    |                               |
            +--------+---------+                               |
                     |                                         |
         +-----------+-----------+                              |
         |           |           |                              |
   +-----v-----+ +---v---+                                     |
   |    FIX    | |  FIX  |                                     |
   | L0C -> GM | |L0C->L1|                                     |
   +-----+-----+ +---+---+                                     |
         |           |                                         |
         v           v                                         |
   +-----------+ +-----------+                                 |
   |    GM     | |    L1     |                                 |
   +-----------+ +-----------+                                 |
                                                               |
                               +-------------------------------+
                               |
                         +-----v-----+
                         |   MTE3    |
                         |  UB -> GM |
                         |  (单向)   |
                         +-----+-----+
                               |
                               v
                         +-----------+
                         |    GM     |
                         +-----------+
```

### Ascend910_95 / 950PR / 950DT 架构

```
+-----------------------------------------------------------------------------+
|                          Global Memory (GM / HBM)                           |
+---------------------------------------+-------------------------------------+
                                        |
                    +-------------------+-------------------+
                    |                                       |
              +-----v-----+                           +-----v-----+
              |   MTE2    |                           |   MTE2    |
              |  GM -> L1 |                           |  GM -> UB |
              |  (双向)   |                           |  (单向)   |
              +-----+-----+                           +-----+-----+
                    |                                       |
                    v                                       v
        +-----------------------+                   +----------------------+
        |          L1           |                   |         UB           |
        |    (cbuf, 512KB)      |                   |  (ub, 248KB,预留8KB)  |
        |    Cube输入缓存       |                   |    Vector工作区      |
        +-----------+-----------+                   +--------+-------------+
                    |                                        |
        +-----------+-----------+                            |
        |           |           |                            |
  +-----v-----+ +---v---+ +-----v-----+                      |
  |   MTE1    | | MTE1  | |   MTE1    |                      |
  | L1 -> L0A | |L1->L0B| |L1 -> BT Buf|                     |
  +-----+-----+ +---+---+ +-----+-----+                      |
        |           |           |                            |
        v           v           v                            |
  +-----------+ +-----------+ +-----------+                   |
  |    L0A    | |    L0B    | | BT Buffer |                   |
  | (ca,64KB) | | (cb,64KB) | |   (1KB)   |                   |
  | 矩阵A输入  | | 矩阵B输入  | | Bias数据  |                   |
  +-----+-----+ +-----+-----+ +-----+-----+                   |
        |             |             |                          |
        +-------------+-------------+                          |
                      |                                        |
                      v                                        |
            +------------------+                               |
            |      Cube        |                               |
            |   (MatMul)       |                               |
            +--------+---------+                               |
                     |                                         |
                     v                                         |
            +------------------+                               |
            |      L0C         |                               |
            |   (cc, 256KB)    |                               |
            |   矩阵乘法结果    |                               |
            +--------+---------+                               |
                     |                                         |
         +-----------+-----------+-----------+                  |
         |           |           |           |                  |
   +-----v-----+ +---v---+ +-----v-----+     |                  |
   |    FIX    | |  FIX  | |    FIX    |     |                  |
   | L0C -> GM | |L0C->L1| | L0C -> UB |<----+                  |
   |           | |       | | (950特有) |                        |
   +-----+-----+ +---+---+ +-----+-----+                        |
         |           |           |                               |
         v           v           v                               |
   +-----------+ +-----------+ +----------------------+          |
   |    GM     | |    L1     | |  UB (紧耦合缓冲区)    |<---------+
   +-----------+ +-----------+ +----------+-----------+          |
                                          |                     |
                                    +-----v-----+               |
                                    |   MTE3    |               |
                                    |  UB -> GM |               |
                                    +-----+-----+               |
                                          |                     |
                                          v                     |
                                    +-----------+               |
                                    |    GM     |               |
                                    +-----------+               |
```

## 常见问题

**Q: 为什么 L0A/L0B 没有对齐要求字段？**
A: 在 [NPUTargetSpec.td](https://gitcode.com/Ascend/AscendNPU-IR/tree/master/bishengir/include/bishengir/Dialect/HACC/Targets/NPUTargetSpec.td) 的 `TargetSpec` 基类中，只定义了 `UbAlignSize`、`L1AlignSize` 和 `L0cAlignSize` 三个对齐字段，L0A/L0B 的对齐要求未在 TableGen 中显式描述。

**Q: 950 系列的 UB -> L1 通路使用哪个 Pipeline？**
A: 根据源码 [HIVMDMAOps.cpp:616-622](https://gitcode.com/Ascend/AscendNPU-IR/tree/master/bishengir/lib/Dialect/HIVM/IR/HIVMDMAOps.cpp#L616-L622)，`{AddressSpace::UB, AddressSpace::L1}` 映射到 `PIPE::PIPE_MTE3`。

**Q: `hivm.copy` 操作支持哪些地址空间组合？**
A: 根据源码 [HIVMDMAOps.cpp:440-448](https://gitcode.com/Ascend/AscendNPU-IR/tree/master/bishengir/lib/Dialect/HIVM/IR/HIVMDMAOps.cpp#L440-L448)，`copy` 支持的组合为：`UB -> UB` 和 `GM -> L1`。对于 950 系列还额外支持 `UB -> L1`。

**Q: 紧耦合缓冲区和普通数据通路有什么区别？**
A: 紧耦合缓冲区是 950 架构特有的 CV 通信机制，它允许 Cube 和 Vector 之间直接传递数据而无需经过 GM 中转。普通数据通路中，非 950 设备的 Cube 结果必须先写回 GM，再由 Vector 从 GM 加载到 UB。

## 相关文档

- 源码参考：[NPUTargetSpec.td](https://gitcode.com/Ascend/AscendNPU-IR/tree/master/bishengir/include/bishengir/Dialect/HACC/Targets/NPUTargetSpec.td)
- 源码参考：[HIVMAttrs.td](https://gitcode.com/Ascend/AscendNPU-IR/tree/master/bishengir/include/bishengir/Dialect/HIVM/IR/HIVMAttrs.td)
- 源码参考：[HIVMDMAOps.td](https://gitcode.com/Ascend/AscendNPU-IR/tree/master/bishengir/include/bishengir/Dialect/HIVM/IR/HIVMDMAOps.td)
- 源码参考：[HIVMDMAOps.cpp](https://gitcode.com/Ascend/AscendNPU-IR/tree/master/bishengir/lib/Dialect/HIVM/IR/HIVMDMAOps.cpp)
- 上一节：[01-npu-hardware-overview.md](./01-npu-hardware-overview.md) — NPU 硬件架构总览
- 下一节：[03-pipeline-execution-model.md](./03-pipeline-execution-model.md) — Pipeline 执行模型
