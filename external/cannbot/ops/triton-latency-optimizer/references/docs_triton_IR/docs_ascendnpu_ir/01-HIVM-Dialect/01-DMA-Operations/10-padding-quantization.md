# 随路 Padding / 量化 / 激活

> 关键词：PadMode、FixpipePreQuantMode、FixpipePreReluMode、随路功能、量化、ReLU

## 概述

HIVM 方言的 DMA 操作支持多种随路（on-the-fly）功能，这些功能在数据搬运过程中同时执行，无需额外的计算操作。随路功能是 Ascend NPU 硬件的重要特性，可以显著减少计算开销和内存访问次数。

本文档汇总所有 DMA 操作的随路功能，包括：
- **Padding（填充）**：在数据搬运时填充目标缓冲区
- **Quantization（量化）**：在 Fixpipe 搬运时执行类型转换
- **Activation（激活）**：在 Fixpipe 搬运时执行激活函数

## Padding（填充）

### 概述

Padding 功能允许在 DMA 搬运过程中，将目标缓冲区中未被源数据覆盖的位置填充为指定值。这对于处理边界不齐的数据非常有用，例如在 Tiling 后的边界 Tile 中，源数据可能小于目标缓冲区。

### PadMode 枚举

定义于 [HIVMAttrs.td:L329-L349](https://gitcode.com/Ascend/AscendNPU-IR/tree/master/bishengir/include/bishengir/Dialect/HIVM/IR/HIVMAttrs.td#L329-L349)

| 枚举值 | 数值 | IR 字面量 | 说明 |
|--------|------|----------|------|
| `PadNull` | 0 | `PadNull` | 不填充，源和目标形状必须相同 |
| `PadFirstElem` | 1 | `PadFirstElem` | 使用源数据的第一个元素填充 |
| `PadValue` | 2 | `PadValue` | 使用指定的 pad_value 填充 |

### IR 表示

```mlir
#hivm.padmode<PadNull>
#hivm.padmode<PadFirstElem>
#hivm.padmode<PadValue>
```

### 支持 Padding 的操作

| 操作 | 左侧 Padding | 右侧 Padding | PadFirstElem | PadValue |
|------|-------------|-------------|-------------|---------|
| `hir.load` | 支持 | 支持 | 支持 | 支持 |
| `hir.copy` | 支持 | 不支持 | 支持 | 支持 |

### 使用模式

#### PadValue 模式

```mlir
%val = arith.constant 0.0 : f16
hivm.hir.load ins(%src : memref<16x15xf16, #hivm.address_space<gm>>)
               outs(%dst : memref<16x16xf16, #hivm.address_space<ub>>)
               pad_mode = #hivm.padmode<PadValue>
               pad_value = %val : f16
```

当仅指定 `pad_value` 而不指定 `pad_mode` 时，编译器自动推断为 `PadValue` 模式：

```mlir
hivm.hir.load ins(%src : memref<16x15xf16, #hivm.address_space<gm>>)
               outs(%dst : memref<16x16xf16, #hivm.address_space<ub>>)
               pad_value = %val : f16
```

#### PadFirstElem 模式

```mlir
hivm.hir.load ins(%src : memref<16x15xf16, #hivm.address_space<gm>>)
               outs(%dst : memref<16x16xf16, #hivm.address_space<ub>>)
               pad_mode = #hivm.padmode<PadFirstElem>
```

#### 左侧 Padding

```mlir
%c0 = arith.constant 0 : index
hivm.hir.load ins(%src : memref<16x16xf16, #hivm.address_space<gm>>)
               outs(%dst : memref<16x16xf16, #hivm.address_space<ub>>)
               pad_mode = #hivm.padmode<PadValue>
               pad_value = %val : f16
               left_padding_num = %c0 : index
```

### Padding 与 init_out_buffer 的关系

`init_out_buffer` 属性控制是否在搬运前初始化整个目标缓冲区。当 Padding 无法覆盖所有位置时（例如右侧 Padding 之外的区域），`init_out_buffer` 可以确保缓冲区的干净状态。

```mlir
hivm.hir.load ins(%src : memref<16x15xf16, #hivm.address_space<gm>>)
               outs(%dst : memref<16x16xf16, #hivm.address_space<ub>>)
               init_out_buffer = true
               pad_value = %val : f16
```

### PaddingOption 枚举（Gather/Scatter 专用）

定义于 [HIVMAttrs.td:L920-L928](https://gitcode.com/Ascend/AscendNPU-IR/tree/master/bishengir/include/bishengir/Dialect/HIVM/IR/HIVMAttrs.td#L920-L928)

| 枚举值 | 数值 | IR 字面量 | 说明 |
|--------|------|----------|------|
| `PAD_ZERO` | 1 | `zero` | 用零填充越界位置 |
| `PAD_NAN` | 2 | `nan` | 用 NaN 填充越界位置 |

此枚举仅用于 `hir.gather_load` 操作的 `padding` 属性。

## Quantization（量化）

### 概述

量化功能是 `hir.fixpipe` 操作的独有随路功能，在数据从 L0C 搬运到其他内存层级时，同时执行类型转换。这避免了额外的向量计算操作，可以显著提升性能。

### FixpipePreQuantMode 枚举

定义于 [HIVMAttrs.td:L783-L801](https://gitcode.com/Ascend/AscendNPU-IR/tree/master/bishengir/include/bishengir/Dialect/HIVM/IR/HIVMAttrs.td#L783-L801)

| 枚举值 | 数值 | IR 字面量 | 源类型 | 目标类型 | 说明 |
|--------|------|----------|--------|---------|------|
| `NO_QUANT` | 0 | `NO_QUANT` | - | - | 不执行量化（默认） |
| `F322F16` | 1 | `F322F16` | F32 | F16 | F32 到 F16 转换 |
| `S322I8` | 9 | `S322I8` | F32 | I8 | F32 到 I8 量化 |
| `QF322F32_PRE` | 15 | `QF322F32_PRE` | F32 | F32 | 带 scale 的 F32 预量化 |
| `F322BF16` | 16 | `F322BF16` | F32 | BF16 | F32 到 BF16 转换 |

### IR 表示

```mlir
#hivm.fixpipe_pre_quant_mode<NO_QUANT>
#hivm.fixpipe_pre_quant_mode<F322F16>
#hivm.fixpipe_pre_quant_mode<S322I8>
#hivm.fixpipe_pre_quant_mode<QF322F32_PRE>
#hivm.fixpipe_pre_quant_mode<F322BF16>
```

### 量化模式详解

#### F322F16

将 F32 累加结果转换为 F16，适用于混合精度训练和推理：

```mlir
%l0c = tensor.empty() : tensor<256x128xf32>
%dst = tensor.empty() : tensor<256x128xf16>
%result = hivm.hir.fixpipe {pre_quant = #hivm.fixpipe_pre_quant_mode<F322F16>}
                         ins(%l0c : tensor<256x128xf32>)
                         outs(%dst : tensor<256x128xf16>)
                         -> tensor<256x128xf16>
```

#### S322I8

将 F32 累加结果量化为 I8，适用于量化推理：

```mlir
%l0c = tensor.empty() : tensor<256x128xf32>
%dst = tensor.empty() : tensor<256x128xi8>
%result = hivm.hir.fixpipe {pre_quant = #hivm.fixpipe_pre_quant_mode<S322I8>}
                         ins(%l0c : tensor<256x128xf32>)
                         outs(%dst : tensor<256x128xi8>)
                         -> tensor<256x128xi8>
```

#### QF322F32_PRE

带缩放因子的 F32 到 F32 预量化，需要提供 `quant_scale` 参数：

```mlir
%scale = arith.constant 0.125 : f32
%l0c = tensor.empty() : tensor<256x128xf32>
%dst = tensor.empty() : tensor<256x128xf32>
%result = hivm.hir.fixpipe {pre_quant = #hivm.fixpipe_pre_quant_mode<QF322F32_PRE>}
                         ins(%l0c : tensor<256x128xf32>)
                         outs(%dst : tensor<256x128xf32>)
                         quant_scale = %scale : f32
                         -> tensor<256x128xf32>
```

#### F322BF16

将 F32 累加结果转换为 BF16：

```mlir
%l0c = tensor.empty() : tensor<256x128xf32>
%dst = tensor.empty() : tensor<256x128xbf16>
%result = hivm.hir.fixpipe {pre_quant = #hivm.fixpipe_pre_quant_mode<F322BF16>}
                         ins(%l0c : tensor<256x128xf32>)
                         outs(%dst : tensor<256x128xbf16>)
                         -> tensor<256x128xbf16>
```

### 量化与激活的组合

量化与激活可以同时使用，执行顺序为：先量化，后激活：

```mlir
%l0c = tensor.empty() : tensor<256x128xf32>
%dst = tensor.empty() : tensor<256x128xf16>
%result = hivm.hir.fixpipe {pre_quant = #hivm.fixpipe_pre_quant_mode<F322F16>,
                            pre_relu = #hivm.fixpipe_pre_relu_mode<NORMAL_RELU>}
                         ins(%l0c : tensor<256x128xf32>)
                         outs(%dst : tensor<256x128xf16>)
                         -> tensor<256x128xf16>
```

## Activation（激活）

### 概述

激活功能是 `hir.fixpipe` 操作的独有随路功能，在数据从 L0C 搬运时同时执行激活函数。与量化类似，这避免了额外的向量计算操作。

### FixpipePreReluMode 枚举

定义于 [HIVMAttrs.td:L803-L819](https://gitcode.com/Ascend/AscendNPU-IR/tree/master/bishengir/include/bishengir/Dialect/HIVM/IR/HIVMAttrs.td#L803-L819)

| 枚举值 | 数值 | IR 字面量 | 公式 | 说明 |
|--------|------|----------|------|------|
| `NO_RELU` | 0 | `NO_RELU` | - | 不执行激活（默认） |
| `NORMAL_RELU` | 1 | `NORMAL_RELU` | max(0, x) | 标准 ReLU |
| `LEAKY_RELU` | 2 | `LEAKY_RELU` | x > 0 ? x : alpha * x | Leaky ReLU |
| `P_RELU` | 3 | `P_RELU` | x > 0 ? x : p * x | Parametric ReLU |

### IR 表示

```mlir
#hivm.fixpipe_pre_relu_mode<NO_RELU>
#hivm.fixpipe_pre_relu_mode<NORMAL_RELU>
#hivm.fixpipe_pre_relu_mode<LEAKY_RELU>
#hivm.fixpipe_pre_relu_mode<P_RELU>
```

### 激活模式详解

#### NORMAL_RELU

标准 ReLU 激活，将负值截断为 0：

```mlir
%l0c = tensor.empty() : tensor<256x128xf16>
%dst = tensor.empty() : tensor<256x128xf16>
%result = hivm.hir.fixpipe {pre_relu = #hivm.fixpipe_pre_relu_mode<NORMAL_RELU>}
                         ins(%l0c : tensor<256x128xf16>)
                         outs(%dst : tensor<256x128xf16>)
                         -> tensor<256x128xf16>
```

#### LEAKY_RELU

Leaky ReLU 激活，负值乘以一个小的斜率因子：

```mlir
%l0c = tensor.empty() : tensor<256x128xf16>
%dst = tensor.empty() : tensor<256x128xf16>
%result = hivm.hir.fixpipe {pre_relu = #hivm.fixpipe_pre_relu_mode<LEAKY_RELU>}
                         ins(%l0c : tensor<256x128xf16>)
                         outs(%dst : tensor<256x128xf16>)
                         -> tensor<256x128xf16>
```

#### P_RELU

Parametric ReLU，负值乘以可学习的参数：

```mlir
%l0c = tensor.empty() : tensor<256x128xf16>
%dst = tensor.empty() : tensor<256x128xf16>
%result = hivm.hir.fixpipe {pre_relu = #hivm.fixpipe_pre_relu_mode<P_RELU>}
                         ins(%l0c : tensor<256x128xf16>)
                         outs(%dst : tensor<256x128xf16>)
                         -> tensor<256x128xf16>
```

## 随路功能组合

### Fixpipe 随路功能执行顺序

```
L0C 数据
    |
    v
[Pre-Quant] (可选)     -- 类型转换：F32->F16/F32->I8/F32->BF16/F32->F32(scale)
    |
    v
[Pre-ReLU] (可选)      -- 激活函数：ReLU/LeakyReLU/P-ReLU
    |
    v
[DMA Mode]             -- 布局转换：NZ2ND/NZ2DN/NZ2NZ
    |
    v
[Dual Dst] (可选)      -- 双目标拆分：ROW_SPLIT/COLUMN_SPLIT
    |
    v
目标缓冲区 (GM/UB/L1)
```

### 组合约束

| 组合 | 是否支持 | 说明 |
|------|---------|------|
| 量化 + 激活 | 支持 | 先量化后激活 |
| 量化 + NZ2ND | 支持 | 先量化/激活后布局转换 |
| 量化 + NZ2DN | 支持 | 仅 Ascend950 |
| 量化 + 双目标 | 支持 | 仅 Ascend950，dst=UB |
| 激活 + NZ2ND | 支持 | - |
| 激活 + 双目标 | 支持 | 仅 Ascend950，dst=UB |
| NZ2DN + 双目标 | 不支持 | 互斥 |

### Eviction Policy（缓存驱逐策略）

定义于 [HIVMAttrs.td:L356-L372](https://gitcode.com/Ascend/AscendNPU-IR/tree/master/bishengir/include/bishengir/Dialect/HIVM/IR/HIVMAttrs.td#L356-L372)

| 枚举值 | 数值 | IR 字面量 | 说明 |
|--------|------|----------|------|
| `EvictFirst` | 0 | `EvictFirst` | 优先驱逐，适用于一次性数据 |
| `EvictLast` | 1 | `EvictLast` | 最后驱逐，适用于反复访问数据 |

Eviction Policy 仅用于 `hir.load` 操作，控制加载的数据在缓存中的保留策略。

## 随路功能在各操作中的支持矩阵

| 功能 | hir.load | hir.store | hir.copy | hir.fixpipe | hir.nd2nz |
|------|---------|----------|---------|------------|----------|
| PadValue | 支持 | - | 支持 | - | - |
| PadFirstElem | 支持 | - | 支持 | - | - |
| 左侧 Padding | 支持 | - | 支持 | - | - |
| 右侧 Padding | 支持 | - | - | - | - |
| init_out_buffer | 支持 | - | - | - | 支持 |
| Eviction Policy | 支持 | - | - | - | - |
| 量化 (PreQuant) | - | - | - | 支持 | - |
| 激活 (PreReLU) | - | - | - | 支持 | - |
| 布局转换 | - | - | - | 支持 | - |
| 双目标 | - | - | - | 支持 | - |
| 原子操作 | - | 支持 | - | - | - |

## 常见问题

### Q: 为什么 copy 只支持左侧 Padding 而 load 支持双侧？

A: 这是硬件 DMA 引擎的限制。Copy 操作的硬件实现在 Padding 方面功能较弱，仅支持左侧填充。

### Q: 量化和激活可以同时使用吗？

A: 可以。Fixpipe 支持同时执行随路量化和随路激活，执行顺序为先量化后激活。这是 Fixpipe 的核心优势之一。

### Q: QF322F32_PRE 量化模式中 quant_scale 的作用是什么？

A: `quant_scale` 是一个缩放因子，在 F32 到 F32 的预量化过程中，将结果乘以该缩放因子。这常用于量化感知训练中，需要在保持 F32 精度的同时模拟量化效果。

### Q: PadFirstElem 的典型使用场景是什么？

A: `PadFirstElem` 常用于需要用边界值填充的场景，例如在卷积的 Padding 中，用边缘像素值填充边界，而不是用零值。

## 相关文档

- hir.load：[01-load.md](01-load.md)
- hir.copy：[05-copy.md](05-copy.md)
- hir.fixpipe：[06-fixpipe.md](06-fixpipe.md)
- 源码参考：
  - [HIVMAttrs.td](https://gitcode.com/Ascend/AscendNPU-IR/tree/master/bishengir/include/bishengir/Dialect/HIVM/IR/HIVMAttrs.td) - 枚举定义
  - [HIVMDMAOps.td](https://gitcode.com/Ascend/AscendNPU-IR/tree/master/bishengir/include/bishengir/Dialect/HIVM/IR/HIVMDMAOps.td) - DMA 操作定义
