# 紧耦合缓冲区 — Tightly Coupled Buffer

## 概述

紧耦合缓冲区（Tightly Coupled Buffer）是 Ascend NPU 在 Mix CV（Cube-Vector 混合）场景下用于 Cube 和 Vector 核心之间高效数据传递的机制。它利用硬件的紧耦合存储特性，避免数据通过 GM（全局内存）中转，减少延迟和带宽消耗。

## 相关 Pass

### hivm-insert-cv-tight-coupled-buffer

- **Pass 名**：`hivm-insert-cv-tight-coupled-buffer`
- **作用域**：`func::FuncOp`
- **构造函数**：`mlir::hivm::createInsertCVTightCoupledBufferPass()`
- **功能**：为 Mix CV 场景插入 Cube-Vector 紧耦合缓冲区。在 Cube 操作的输出和 Vector 操作的输入之间插入紧耦合缓冲区，使数据可以直接从 Cube 核心传递到 Vector 核心，无需经过全局内存。
- **依赖方言**：`hivm::HIVMDialect`, `bufferization::BufferizationDialect`, `annotation::AnnotationDialect`
- **源码参考**：[Passes.td:L559-L566](https://gitcode.com/Ascend/AscendNPU-IR/tree/master/bishengir/include/bishengir/Dialect/HIVM/Transforms/Passes.td#L559-L566)

## TightlyCoupledBufferAttr

### 属性定义

- **属性名**：`hivm.tightly_coupled_buffer`
- **Mnemonic**：`tightly_coupled_buffer`
- **参数**：
  - `id`：`std::optional<int32_t>`，紧耦合缓冲区的标识符
- **Assembly 格式**：`<id>`

### 属性描述

标记一个缓冲区为紧耦合缓冲区，用于 Cube 和 Vector 核心之间的数据传递。

### IR 示例

```mlir
// 紧耦合缓冲区标记
annotation.mark %buffer {hivm.tightly_coupled_buffer = #hivm.tightly_coupled_buffer<0>}
  : memref<16x16xf16, #hivm.address_space<ub>>
```

源码参考：[HIVMAttrs.td:L1010-L1017](https://gitcode.com/Ascend/AscendNPU-IR/tree/master/bishengir/include/bishengir/Dialect/HIVM/IR/HIVMAttrs.td#L1010-L1017)

## 工作原理

### Mix CV 数据流

在 Mix CV 场景下，Cube 核心执行矩阵乘法，Vector 核心执行元素级操作。两者之间的数据传递路径决定了性能：

```
方式 1（无紧耦合缓冲区）：
  Cube L0C → fixpipe → GM → load → Vector UB
  （需要经过全局内存，延迟高、带宽受限）

方式 2（有紧耦合缓冲区）：
  Cube L0C → fixpipe → TightlyCoupledBuffer → Vector UB
  （直接传递，低延迟、高带宽）
```

### 紧耦合缓冲区的内存规划

PlanMemory Pass 的 `disable-tightly-coupled-buffer-reuse` 选项控制是否允许紧耦合缓冲区的内存复用：

- **默认（false）**：允许复用，紧耦合缓冲区可以与其他缓冲区共享物理内存
- **设为 true**：禁止复用，紧耦合缓冲区独占物理内存，确保数据传递的可靠性

## 相关 Pass

### hivm-insert-load-store-for-mix-cv

- **Pass 名**：`hivm-insert-load-store-for-mix-cv`
- **作用域**：`func::FuncOp`
- **构造函数**：`mlir::hivm::createInsertLoadStoreForMixCVPass()`
- **功能**：为 Mix CV 场景插入 load/store 操作，处理 Cube 和 Vector 核心之间的数据搬运。
- **依赖方言**：`hivm::HIVMDialect`, `bishengir::memref_ext::MemRefExtDialect`
- **源码参考**：[Passes.td:L552-L557](https://gitcode.com/Ascend/AscendNPU-IR/tree/master/bishengir/include/bishengir/Dialect/HIVM/Transforms/Passes.td#L552-L557)

### hivm-split-mixed-if-conditionals

- **Pass 名**：`hivm-split-mixed-if-conditionals`
- **作用域**：`func::FuncOp`
- **构造函数**：`mlir::hivm::createSplitMixedIfConditionalsPass()`
- **功能**：为 Mix CV 场景拆分 if 条件语句，确保 Cube 和 Vector 操作在各自的核心上正确执行。
- **依赖方言**：`arith::ArithDialect`, `hivm::HIVMDialect`, `tensor::TensorDialect`
- **源码参考**：[Passes.td:L544-L550](https://gitcode.com/Ascend/AscendNPU-IR/tree/master/bishengir/include/bishengir/Dialect/HIVM/Transforms/Passes.td#L544-L550)

### hivm-insert-cv-data-movement

- **Pass 名**：`hivm-insert-cv-data-movement`
- **作用域**：`func::FuncOp`
- **构造函数**：`mlir::hivm::createInsertCVDataMovementPass()`
- **功能**：插入 Cube-Vector 数据搬运操作。
- **依赖方言**：`hivm::HIVMDialect`
- **源码参考**：[Passes.td:L802-L806](https://gitcode.com/Ascend/AscendNPU-IR/tree/master/bishengir/include/bishengir/Dialect/HIVM/Transforms/Passes.td#L802-L806)

## 源码参考

- InsertCVTightCoupledBuffer Pass：[Passes.td:L559-L566](https://gitcode.com/Ascend/AscendNPU-IR/tree/master/bishengir/include/bishengir/Dialect/HIVM/Transforms/Passes.td#L559-L566)
- TightlyCoupledBufferAttr：[HIVMAttrs.td:L1010-L1017](https://gitcode.com/Ascend/AscendNPU-IR/tree/master/bishengir/include/bishengir/Dialect/HIVM/IR/HIVMAttrs.td#L1010-L1017)
- InsertLoadStoreForMixCV Pass：[Passes.td:L552-L557](https://gitcode.com/Ascend/AscendNPU-IR/tree/master/bishengir/include/bishengir/Dialect/HIVM/Transforms/Passes.td#L552-L557)
