# NPU 硬件架构总览

> 关键词：AI Core, Cube, Vector, NPUTargetSpec, AddressSpace, TCoreType, VFMode, dav-c220, dav-c310, Reg-based, Mem-based

## 概述

华为昇腾 NPU 是面向 AI 计算的专用处理器，其核心设计理念是通过多级存储层次和专用计算单元实现高吞吐的矩阵与向量计算。AscendNPU-IR 项目通过 HACC 和 HIVM 两个 Dialect 将 NPU 硬件特征映射到 MLIR IR 中，使得编译器能够精确描述和优化针对特定硬件的计算逻辑。

每个 AI Core 是 NPU 的基本计算单元，内部包含 Cube（矩阵乘法）和 Vector（向量计算）两类核心。不同 NPU 型号在 AI Core 数量、存储容量、数据通路等方面存在差异，这些差异通过 [NPUTargetSpec.td](https://gitcode.com/Ascend/AscendNPU-IR/tree/master/bishengir/include/bishengir/Dialect/HACC/Targets/NPUTargetSpec.td) 中的 TableGen 定义精确描述，并在编译时通过 `hacc.DeviceSpec` 属性传递给 IR。

本文档从源码中精确提取所有 NPU 型号规格、核心类型枚举和硬件概念在 IR 中的映射关系，为理解 AscendNPU-IR 的后续文档奠定基础。

## AI Core 架构

每个 AI Core 包含以下组件：

| 组件 | 功能 | 说明 |
|------|------|------|
| **AIC (AI Cube)** | 包含 Cube 单元，用于矩阵乘法计算 | 每个 AI Core 有 1 个 AIC |
| **AIV (AI Vector)** | 包含 Vector 单元，用于向量计算 | 每个 AI Core 有 2 个 AIV |
| **Scalar** | 标量计算单元，拥有 DCache 和 ICache | 每个 AIC/AIV 都有自己的 Scalar 单元 |

**核心数量关系**：
- 每个 AI Core = 1 个 Cube Core + 2 个 Vector Core
- 即 `VectorCoreCount = 2 * CubeCoreCount = 2 * AiCoreCount`

## NPU 型号规格

### Ascend910B 系列

源文件：[NPUTargetSpec.td:64-98](https://gitcode.com/Ascend/AscendNPU-IR/tree/master/bishengir/include/bishengir/Dialect/HACC/Targets/NPUTargetSpec.td#L64-L98)

基础规格（`Ascend910B_BaseSpec`）：UB=192KB, L1=512KB, L0A=64KB, L0B=64KB, L0C=128KB, 架构代号=`dav-c220`

| 型号 | AI Core | Cube Core | Vector Core | UB | L1 | L0A | L0B | L0C | Arch |
|------|---------|-----------|-------------|-----|-----|-----|-----|-----|------|
| Ascend910B1 | 24 | 24 | 48 | 192KB | 512KB | 64KB | 64KB | 128KB | dav-c220 |
| Ascend910B2 | 24 | 24 | 48 | 192KB | 512KB | 64KB | 64KB | 128KB | dav-c220 |
| Ascend910B3 | 20 | 20 | 40 | 192KB | 512KB | 64KB | 64KB | 128KB | dav-c220 |
| Ascend910B4 | 20 | 20 | 40 | 192KB | 512KB | 64KB | 64KB | 128KB | dav-c220 |

### Ascend910_93 系列

源文件：[NPUTargetSpec.td:104-150](https://gitcode.com/Ascend/AscendNPU-IR/tree/master/bishengir/include/bishengir/Dialect/HACC/Targets/NPUTargetSpec.td#L104-L150)

基础规格（`Ascend910_93_BaseSpec`）：与 910B 系列相同，UB=192KB, L1=512KB, L0A=64KB, L0B=64KB, L0C=128KB, 架构代号=`dav-c220`

| 型号 | AI Core | Cube Core | Vector Core | UB | L1 | L0A | L0B | L0C | Arch |
|------|---------|-----------|-------------|-----|-----|-----|-----|-----|------|
| Ascend910_9362 | 20 | 20 | 40 | 192KB | 512KB | 64KB | 64KB | 128KB | dav-c220 |
| Ascend910_9372 | 20 | 20 | 40 | 192KB | 512KB | 64KB | 64KB | 128KB | dav-c220 |
| Ascend910_9381 | 24 | 24 | 48 | 192KB | 512KB | 64KB | 64KB | 128KB | dav-c220 |
| Ascend910_9382 | 24 | 24 | 48 | 192KB | 512KB | 64KB | 64KB | 128KB | dav-c220 |
| Ascend910_9391 | 24 | 24 | 48 | 192KB | 512KB | 64KB | 64KB | 128KB | dav-c220 |
| Ascend910_9392 | 24 | 24 | 48 | 192KB | 512KB | 64KB | 64KB | 128KB | dav-c220 |

### Ascend310B 系列

源文件：[NPUTargetSpec.td:156-190](https://gitcode.com/Ascend/AscendNPU-IR/tree/master/bishengir/include/bishengir/Dialect/HACC/Targets/NPUTargetSpec.td#L156-L190)

基础规格（`Ascend310B_BaseSpec`）：UB=256KB, L1=1024KB, L0A=64KB, L0B=64KB, L0C=128KB, 架构代号=`dav-m300`

| 型号 | AI Core | Cube Core | Vector Core | UB | L1 | L0A | L0B | L0C | Arch |
|------|---------|-----------|-------------|-----|------|-----|-----|-----|------|
| Ascend310B1 | 1 | 1 | 1 | 256KB | 1024KB | 64KB | 64KB | 128KB | dav-m300 |
| Ascend310B2 | 1 | 1 | 1 | 256KB | 1024KB | 64KB | 64KB | 128KB | dav-m300 |
| Ascend310B3 | 1 | 1 | 1 | 256KB | 1024KB | 64KB | 64KB | 128KB | dav-m300 |
| Ascend310B4 | 1 | 1 | 1 | 256KB | 1024KB | 64KB | 64KB | 128KB | dav-m300 |

### Ascend910_95 系列

源文件：[NPUTargetSpec.td:196-262](https://gitcode.com/Ascend/AscendNPU-IR/tree/master/bishengir/include/bishengir/Dialect/HACC/Targets/NPUTargetSpec.td#L196-L262)

基础规格（`Ascend950_BaseSpec`）：UB=248KB（预留 8KB 给编译器）, DCache=32KB~120KB, L1=512KB, L0A=64KB, L0B=64KB, L0C=256KB, 架构代号=`dav-c310`

| 型号 | AI Core | Cube Core | Vector Core | UB | DCache | L1 | L0A | L0B | L0C | Arch |
|------|---------|-----------|-------------|------|--------|-----|-----|-----|-----|------|
| Ascend910_950z | 4 | 4 | 8 | 248KB | 32~120KB | 512KB | 64KB | 64KB | 256KB | dav-c310 |
| Ascend910_9579 | 28 | 28 | 56 | 248KB | 32~120KB | 512KB | 64KB | 64KB | 256KB | dav-c310 |
| Ascend910_957b | 28 | 28 | 56 | 248KB | 32~120KB | 512KB | 64KB | 64KB | 256KB | dav-c310 |
| Ascend910_957d | 28 | 28 | 56 | 248KB | 32~120KB | 512KB | 64KB | 64KB | 256KB | dav-c310 |
| Ascend910_9581 | 32 | 32 | 64 | 248KB | 32~120KB | 512KB | 64KB | 64KB | 256KB | dav-c310 |
| Ascend910_9589 | 32 | 32 | 64 | 248KB | 32~120KB | 512KB | 64KB | 64KB | 256KB | dav-c310 |
| Ascend910_958a | 32 | 32 | 64 | 248KB | 32~120KB | 512KB | 64KB | 64KB | 256KB | dav-c310 |
| Ascend910_958b | 32 | 32 | 64 | 248KB | 32~120KB | 512KB | 64KB | 64KB | 256KB | dav-c310 |
| Ascend910_9599 | 36 | 36 | 72 | 248KB | 32~120KB | 512KB | 64KB | 64KB | 256KB | dav-c310 |

### Ascend950PR 系列

源文件：[NPUTargetSpec.td:264-346](https://gitcode.com/Ascend/AscendNPU-IR/tree/master/bishengir/include/bishengir/Dialect/HACC/Targets/NPUTargetSpec.td#L264-L346)

基础规格与 Ascend910_95 系列相同（`Ascend950_BaseSpec`）。

| 型号 | AI Core | Cube Core | Vector Core |
|------|---------|-----------|-------------|
| Ascend950PR_950z | 4 | 4 | 8 |
| Ascend950PR_9579 | 28 | 28 | 56 |
| Ascend950PR_957a | 28 | 28 | 56 |
| Ascend950PR_957b | 28 | 28 | 56 |
| Ascend950PR_957c | 28 | 28 | 56 |
| Ascend950PR_957d | 28 | 28 | 56 |
| Ascend950PR_9589 | 32 | 32 | 64 |
| Ascend950PR_958a | 32 | 32 | 64 |
| Ascend950PR_958b | 32 | 32 | 64 |
| Ascend950PR_958c | 32 | 32 | 64 |
| Ascend950PR_958d | 32 | 32 | 64 |
| Ascend950PR_9599 | 36 | 36 | 72 |
| Ascend950PR_959a | 36 | 36 | 72 |
| Ascend950PR_959b | 36 | 36 | 72 |

### Ascend950DT 系列

源文件：[NPUTargetSpec.td:348-490](https://gitcode.com/Ascend/AscendNPU-IR/tree/master/bishengir/include/bishengir/Dialect/HACC/Targets/NPUTargetSpec.td#L348-L490)

基础规格与 Ascend910_95 系列相同（`Ascend950_BaseSpec`）。

| 型号 | AI Core | Cube Core | Vector Core |
|------|---------|-----------|-------------|
| Ascend950DT_950x | 8 | 8 | 16 |
| Ascend950DT_950y | 8 | 8 | 16 |
| Ascend950DT_9571 | 28 | 28 | 56 |
| Ascend950DT_9572 | 28 | 28 | 56 |
| Ascend950DT_9573 | 28 | 28 | 56 |
| Ascend950DT_9574 | 28 | 28 | 56 |
| Ascend950DT_9575 | 28 | 28 | 56 |
| Ascend950DT_9576 | 28 | 28 | 56 |
| Ascend950DT_9577 | 28 | 28 | 56 |
| Ascend950DT_9578 | 28 | 28 | 56 |
| Ascend950DT_9581 | 32 | 32 | 64 |
| Ascend950DT_9582 | 32 | 32 | 64 |
| Ascend950DT_9583 | 32 | 32 | 64 |
| Ascend950DT_9584 | 32 | 32 | 64 |
| Ascend950DT_9585 | 32 | 32 | 64 |
| Ascend950DT_9586 | 32 | 32 | 64 |
| Ascend950DT_9587 | 32 | 32 | 64 |
| Ascend950DT_9588 | 32 | 32 | 64 |
| Ascend950DT_9591 | 36 | 36 | 72 |
| Ascend950DT_9592 | 36 | 36 | 72 |
| Ascend950DT_9595 | 36 | 36 | 72 |
| Ascend950DT_9596 | 36 | 36 | 72 |
| Ascend950DT_95A1 | 36 | 36 | 72 |
| Ascend950DT_95A2 | 36 | 36 | 72 |

## 存储层次概览

Ascend NPU 采用多级存储架构，包含通用可寻址存储空间和专用硬件缓冲区两类。

### 通用可寻址存储空间

| 存储空间 | IR 标识符 | 说明 | 910B/910_93 | 910_95/950PR/950DT | 310B |
|----------|-----------|------|-------------|---------------------|------|
| GM | `gm` | 全局内存 (HBM/L2)，设备外部存储 | - | - | - |
| L1 | `cbuf` | Cube 单元的一级缓存 | 512KB | 512KB | 1024KB |
| L0A | `ca` | 矩阵 A 输入缓存 | 64KB | 64KB | 64KB |
| L0B | `cb` | 矩阵 B 输入缓存 | 64KB | 64KB | 64KB |
| L0C | `cc` | 矩阵乘法结果缓存 | 128KB | 256KB | 128KB |
| UB | `ub` | 统一缓冲区，Vector 单元使用 | 192KB | 248KB | 256KB |

### 对齐要求

源文件：[NPUTargetSpec.td:64-74](https://gitcode.com/Ascend/AscendNPU-IR/tree/master/bishengir/include/bishengir/Dialect/HACC/Targets/NPUTargetSpec.td#L64-L74) 和 [NPUTargetSpec.td:196-208](https://gitcode.com/Ascend/AscendNPU-IR/tree/master/bishengir/include/bishengir/Dialect/HACC/Targets/NPUTargetSpec.td#L196-L208)

| 存储空间 | 对齐要求（所有架构一致） | 源码值（bits） |
|----------|------------------------|---------------|
| UB | 32B | 256 bits |
| L1 | 32B | 256 bits |
| L0C | 512B | 4096 bits |

### 专用硬件缓冲区

| 缓冲区 | 大小 | 对齐 | 说明 | 访问方式 |
|--------|------|------|------|----------|
| BT Buffer (BiasTable) | 1KB | 64B | 存放矩阵乘法的 Bias 数据 | 通过 `copy_cbuf_to_bt` 从 L1 拷贝 |
| FP Buffer (FixPipe) | 7KB | 128B | FixPipe 流水线的中间缓冲区 | 通过 `hivm.fixpipe` 隐式使用 |

### Ascend950 架构特有存储

| 存储空间 | 大小范围 | 说明 |
|----------|---------|------|
| DCache (SIMT) | 32KB ~ 120KB | SIMT Vector 的数据缓存，大小可配置 |

## 核心类型枚举

源文件：[HIVMAttrs.td:298-317](https://gitcode.com/Ascend/AscendNPU-IR/tree/master/bishengir/include/bishengir/Dialect/HIVM/IR/HIVMAttrs.td#L298-L317)

### TCoreType — 操作级核心类型

用于标注 HIVM 操作在哪种核心上执行。

| 枚举值 | C++ 符号 | 数值 | 说明 |
|--------|---------|------|------|
| CUBE | `TCoreType::CUBE` | 1 | 操作在 Cube 核心上执行 |
| VECTOR | `TCoreType::VECTOR` | 2 | 操作在 Vector 核心上执行 |
| CUBE_OR_VECTOR | `TCoreType::CUBE_OR_VECTOR` | 3 | 操作可在 Cube 或 Vector 核心上执行 |
| CUBE_AND_VECTOR | `TCoreType::CUBE_AND_VECTOR` | 4 | 操作需要在 Cube 和 Vector 核心上同时执行 |

IR 语法示例：

```mlir
#hivm.tcore_type<CUBE>
#hivm.tcore_type<VECTOR>
```

### TFuncCoreType — 函数级核心类型

源文件：[HIVMAttrs.td:250-269](https://gitcode.com/Ascend/AscendNPU-IR/tree/master/bishengir/include/bishengir/Dialect/HIVM/IR/HIVMAttrs.td#L250-L269)

| 枚举值 | C++ 符号 | 数值 | 说明 |
|--------|---------|------|------|
| AIC | `TFuncCoreType::AIC` | 1 | 函数运行在 AI Cube 核心上 |
| AIV | `TFuncCoreType::AIV` | 2 | 函数运行在 AI Vector 核心上 |
| MIX | `TFuncCoreType::MIX` | 3 | 函数混合使用 Cube 和 Vector 核心 |
| AIC_OR_AIV | `TFuncCoreType::AIC_OR_AIV` | 4 | 函数可在 Cube 或 Vector 核心上运行 |

### TModuleCoreType — 模块级核心类型

源文件：[HIVMAttrs.td:271-292](https://gitcode.com/Ascend/AscendNPU-IR/tree/master/bishengir/include/bishengir/Dialect/HIVM/IR/HIVMAttrs.td#L271-L292)

| 枚举值 | C++ 符号 | 数值 | 说明 |
|--------|---------|------|------|
| AIC | `TModuleCoreType::AIC` | 1 | 模块内所有函数均为 AIC 类型 |
| AIV | `TModuleCoreType::AIV` | 2 | 模块内所有函数均为 AIV 类型 |
| MIX | `TModuleCoreType::MIX` | 3 | 模块内函数混合使用 AIC 和 AIV |

推断规则（源自源码注释）：
- 若模块内所有函数的 `func_core_type` 均为 `AIV`，则模块核心类型为 `AIV`
- 若模块内所有函数的 `func_core_type` 均为 `AIC`，则模块核心类型为 `AIC`
- 否则，模块核心类型为 `MIX`

## VFMode 枚举

源文件：[HIVMAttrs.td:948-960](https://gitcode.com/Ascend/AscendNPU-IR/tree/master/bishengir/include/bishengir/Dialect/HIVM/IR/HIVMAttrs.td#L948-L960)

VFMode 用于描述 Vector Function 的执行模式。

| 枚举值 | C++ 符号 | 数值 | 说明 |
|--------|---------|------|------|
| SIMD | `VFMode::SIMD` | 0 | 单指令多数据模式，传统 Vector 执行方式 |
| SIMT | `VFMode::SIMT` | 1 | 单指令多线程模式，类似 GPU 的线程级并行 |
| MIX | `VFMode::MIX` | 2 | 混合模式，同时使用 SIMD 和 SIMT |

IR 语法示例：

```mlir
#hivm.vf_mode<SIMD>
#hivm.vf_mode<SIMT>
#hivm.vf_mode<MIX>
```

## 硬件概念在 IR 中的映射表

| 硬件概念 | IR 表示 | 源文件位置 |
|----------|---------|-----------|
| NPU 型号规格 | `hacc.DeviceSpec` 属性 | [NPUTargetSpec.td](https://gitcode.com/Ascend/AscendNPU-IR/tree/master/bishengir/include/bishengir/Dialect/HACC/Targets/NPUTargetSpec.td) |
| 内存空间 | `#hivm.address_space<gm/cbuf/ca/cb/cc/ub>` | [HIVMAttrs.td:171-197](https://gitcode.com/Ascend/AscendNPU-IR/tree/master/bishengir/include/bishengir/Dialect/HIVM/IR/HIVMAttrs.td#L171-L197) |
| 操作核心类型 | `#hivm.tcore_type<CUBE/VECTOR/...>` | [HIVMAttrs.td:298-317](https://gitcode.com/Ascend/AscendNPU-IR/tree/master/bishengir/include/bishengir/Dialect/HIVM/IR/HIVMAttrs.td#L298-L317) |
| 函数核心类型 | `#hivm.func_core_type<AIC/AIV/MIX/...>` | [HIVMAttrs.td:250-269](https://gitcode.com/Ascend/AscendNPU-IR/tree/master/bishengir/include/bishengir/Dialect/HIVM/IR/HIVMAttrs.td#L250-L269) |
| 模块核心类型 | `#hivm.module_core_type<AIC/AIV/MIX>` | [HIVMAttrs.td:271-292](https://gitcode.com/Ascend/AscendNPU-IR/tree/master/bishengir/include/bishengir/Dialect/HIVM/IR/HIVMAttrs.td#L271-L292) |
| Vector 执行模式 | `#hivm.vf_mode<SIMD/SIMT/MIX>` | [HIVMAttrs.td:948-960](https://gitcode.com/Ascend/AscendNPU-IR/tree/master/bishengir/include/bishengir/Dialect/HIVM/IR/HIVMAttrs.td#L948-L960) |
| 执行流水线 | `#hivm.pipe<PIPE_V/PIPE_M/...>` | [HIVMAttrs.td:203-244](https://gitcode.com/Ascend/AscendNPU-IR/tree/master/bishengir/include/bishengir/Dialect/HIVM/IR/HIVMAttrs.td#L203-L244) |
| 紧耦合缓冲区 | `#hivm.tightly_coupled_buffer<id : ...>` | [HIVMAttrs.td:1010-1017](https://gitcode.com/Ascend/AscendNPU-IR/tree/master/bishengir/include/bishengir/Dialect/HIVM/IR/HIVMAttrs.td#L1010-L1017) |
| 数据布局 | `#hivm.data_layout<ND/nZ/zN/...>` | [HIVMAttrs.td:84-165](https://gitcode.com/Ascend/AscendNPU-IR/tree/master/bishengir/include/bishengir/Dialect/HIVM/IR/HIVMAttrs.td#L84-L165) |

## 架构规格对比汇总

| 特性 | Ascend910B / 910_93 | Ascend310B | Ascend910_95 / 950PR / 950DT |
|------|---------------------|------------|-------------------------------|
| 架构代号 | dav-c220 | dav-m300 | dav-c310 |
| UB 大小 | 192KB | 256KB | 248KB（预留 8KB） |
| L1 大小 | 512KB | 1024KB | 512KB |
| L0A 大小 | 64KB | 64KB | 64KB |
| L0B 大小 | 64KB | 64KB | 64KB |
| L0C 大小 | 128KB | 128KB | 256KB |
| UB 对齐 | 32B | 32B | 32B |
| L1 对齐 | 32B | 32B | 32B |
| L0C 对齐 | 512B | 512B | 512B |
| DCache | 无 | 无 | 32KB ~ 120KB（SIMT 可配置） |
| 紧耦合缓冲区 | 不支持 | 不支持 | 支持（MoveToUb / MoveToL1） |
| L0C -> UB 通路 | 不支持 | 不支持 | 支持 |
| Fixpipe Dual Dst | 不支持 | 不支持 | 支持（ROW_SPLIT / COLUMN_SPLIT） |

## 架构分类：Reg-based 与 Mem-based

Ascend NPU 的编译器将硬件架构分为 **Reg-based（寄存器基）** 和 **Mem-based（内存基）** 两大类。这一分类的根本依据是**是否支持 SIMT（Single Instruction Multiple Threads）VF 模式**：Reg-based 架构支持 SIMT，向量操作可基于寄存器进行；Mem-based 架构不支持 SIMT，向量操作全部基于内存（UB 缓冲区）。两种架构在同步机制、算子降级、内存规划等方面也存在系统性差异。

源码参考：[Utils.cpp](https://gitcode.com/Ascend/AscendNPU-IR/tree/master/bishengir/lib/Dialect/HACC/Utils/Utils.cpp)、[Utility.h](https://gitcode.com/Ascend/AscendNPU-IR/tree/master/bishengir/include/bishengir/Dialect/HIVM/Transforms/GraphSyncSolver/Utility.h)

### 分类定义

```cpp
// Architecture is memory based (A2/A3).
const bool isMemBasedArch;

// Architecture is register based (A5).
const bool isRegBasedArch;
```

| 架构类型 | 代号 | 对应芯片系列 | 架构代号 |
|---------|------|------------|---------|
| **Mem-based**（内存基） | A2/A3 | Ascend910B, Ascend910_93 | dav-c220 |
| **Reg-based**（寄存器基） | A5 | Ascend310B, Ascend950 | dav-m300, dav-c310 |

### 判断函数

编译器通过 `isRegBasedArch` 和 `isMemBasedArch` 两个函数判断目标架构类型：

```cpp
bool isRegBasedArch(TargetDevice targetDevice) {
  return isAscend310B(targetDevice) || isAscend950(targetDevice);
}

bool isMemBasedArch(TargetDevice targetDevice) {
  return isAscend910B(targetDevice) || isAscend910_93(targetDevice);
}
```

当 IR 中未指定目标设备时，默认按 910B（Mem-based）处理。

### 核心区别：SIMT 支持

两种架构最根本的区别在于**是否支持 SIMT VF 模式**：

| 特性 | Mem-based (A2/A3) | Reg-based (A5) |
|------|-------------------|-----------------|
| SIMT VF 模式 | **不支持** | **支持**（Ascend310B 和 Ascend950 均支持） |
| VFMode 推断 | 不运行 InferVFMode | 运行 InferVFMode，推断 SIMD/SIMT/MIX |
| 数据访问模型 | 全部基于 UB 缓冲区（SIMD 模式） | SIMT 基于寄存器，SIMD 基于 UB |
| SIMT 编译路径 | 无 | SIMT VF 拆分后走 Triton GPU 编译路径 |
| SIMD/SIMT 混合 | 不适用 | 通过 `--enable-simd-simt-mix-compile` 启用 |
| DCache | 无 | 有（950 系列 32-120KB） |

SIMT VF 模式下，向量操作基于寄存器进行，每个线程独立执行标量操作，需要 `LocalLoadOp`/`LocalStoreOp` 在 UB 和寄存器之间转移数据。编译器仅在 Reg-based 架构上运行 `InferVFModePass` 和 `InsertInferVFModeFuncPass`，Mem-based 架构直接跳过这些 Pass。

### 附加区别：同步机制

两种架构的核间同步实现方式也不同：

| 特性 | Mem-based (A2/A3) | Reg-based (A5) |
|------|-------------------|-----------------|
| 同步方式 | 基于内存的 FFTS（Fast Flag Transmit Storage） | 基于寄存器的 SetFlag/WaitFlag 指令 |
| 跨核同步 | 需要设置 FFTS base addr，使用 `SetCrossCoreInstrOp` | 使用 `SetFlagOp`/`WaitFlagOp` 寄存器级指令 |
| 块同步降低 | 使用 `SetCrossCoreInstrOp` | 使用 `IntraBlockSet`/`IntraBlockRegInstrOp` |
| Pipe Barrier | 对所有 Pipe 生成 barrier | 跳过 PIPE_V 的 barrier |

### 编译器行为差异

架构类型在以下编译 Pass 中产生不同的行为：

| 编译 Pass | Mem-based 行为 | Reg-based 行为 |
|-----------|---------------|----------------|
| **InjectSync** | 对所有 Pipe 设置 barrier | 跳过 PIPE_V 的 barrier |
| **CrossCoreGSS** | 需要获取并设置 FFTS base addr | 不需要 FFTS |
| **MmadL1 同步** | 标准 SetFlag/WaitFlag | 额外注入 PIPE_M → PIPE_MTE1 的 SetFlag/WaitFlag |
| **VReduceOp 标量降级** | 基本归约仅 i64 降级；argmax/argmin 条件更多 | 除 argmax/argmin 内存对齐问题外，基本归约不降级 |
| **VReduceOp Extra Buffer** | 需要额外临时缓冲区 | 不需要额外缓冲区 |
| **Normalize** | 应用 CmpVne 规范化（vcmp NE → vnot(vcmp EQ)） | 不应用 CmpVne 规范化 |
| **Stride 对齐** | 标准 stride 对齐规则 | 对非单位最后维度 stride 有更严格的对齐要求 |
| **内存规划** | SIMT/MIX 模式下不需要动态调整 UB | SIMT/MIX 模式下需要动态调整 UB 空间（考虑 DCache） |
| **入口内核配置** | `configureEntryForMembaseArch` | `configureEntryForRegbaseArch` |

### 对 Triton 算子优化的影响

- **归约操作**：Reg-based 架构（A5）对基本归约（sum/prod/max/min）有更好的向量硬件支持，除 argmax/argmin 的内存对齐问题外不会标量降级。Mem-based 架构（A2/A3）下 i64 归约和整数 argmax/argmin 会标量降级
- **比较操作规范化**：Mem-based 架构会将 `vcmp(NE)` 规范化为 `vnot(vcmp(EQ))` 以正确处理 NaN；Reg-based 架构不做此规范化
- **SIMT 模式**：仅 Reg-based 架构（Ascend310B 和 Ascend950）支持 SIMT VF 模式，Mem-based 架构（910B/910_93）不支持

> 详细的标量降级差异见 [11-scalar-lowering.md](../../docs_ascendnpu_ir/01-HIVM-Dialect/02-Vector-Operations/11-scalar-lowering.md)

## 常见问题

**Q: Ascend910B1 和 Ascend910B2 的硬件规格有什么区别？**
A: 从 [NPUTargetSpec.td](https://gitcode.com/Ascend/AscendNPU-IR/tree/master/bishengir/include/bishengir/Dialect/HACC/Targets/NPUTargetSpec.td) 的定义来看，两者的 AI Core 数量（24/24/48）和存储规格完全相同。区别可能在于芯片频率、HBM 容量等 TableGen 中未描述的特性。

**Q: 为什么 950 系列的 UB 是 248KB 而不是 256KB？**
A: 源码注释明确标注 `UbSize = 2031616; // bits = 248KB, reserve 8KB for compiler`，即预留了 8KB 给编译器内部使用。

**Q: CUBE_OR_VECTOR 和 CUBE_AND_VECTOR 有什么区别？**
A: `CUBE_OR_VECTOR` 表示操作可以在 Cube 或 Vector 任一核心上执行（选择其一），`CUBE_AND_VECTOR` 表示操作需要 Cube 和 Vector 核心同时参与执行。

**Q: VFMode 的 SIMT 模式在哪些设备上可用？**
A: SIMT VF 模式是 Reg-based 架构（A5 代）的核心特性，Ascend310B 和 Ascend950 均支持。Mem-based 架构（A2/A3 代：Ascend910B/910_93）不支持 SIMT，编译器在这些设备上会跳过 `InferVFModePass` 等相关 Pass。

**Q: Reg-based 和 Mem-based 架构对 Triton 算子编写有什么影响？**
A: 最显著的影响在归约操作上：Reg-based 架构（A5 代：Ascend310B/950）对基本归约有更好的向量硬件支持，而 Mem-based 架构（A2/A3 代：Ascend910B/910_93）下 i64 归约和整数 argmax/argmin 会标量降级，性能损失较大。此外，Mem-based 架构会将 `vcmp(NE)` 规范化为 `vnot(vcmp(EQ))` 以正确处理 NaN。详见 [11-scalar-lowering.md](../../docs_ascendnpu_ir/01-HIVM-Dialect/02-Vector-Operations/11-scalar-lowering.md)。

## 相关文档

- 源码参考：[NPUTargetSpec.td](https://gitcode.com/Ascend/AscendNPU-IR/tree/master/bishengir/include/bishengir/Dialect/HACC/Targets/NPUTargetSpec.td)
- 源码参考：[HIVMAttrs.td](https://gitcode.com/Ascend/AscendNPU-IR/tree/master/bishengir/include/bishengir/Dialect/HIVM/IR/HIVMAttrs.td)
- 下一节：[02-memory-hierarchy.md](./02-memory-hierarchy.md) — 内存层次详解
- 下一节：[03-pipeline-execution-model.md](./03-pipeline-execution-model.md) — Pipeline 执行模型
