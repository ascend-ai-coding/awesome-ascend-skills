# Bufferization — Tensor 到 MemRef 的转换

## 概述

Bufferization 是内存管理的第一阶段，负责将基于 Tensor 的 SSA 值转换为基于 MemRef 的缓冲区操作。此阶段消除了 Tensor 的值语义，引入 `memref.alloc` 分配逻辑缓冲区，为后续的内存规划和物理地址分配奠定基础。

## 核心概念

### Tensor vs MemRef

| 特性 | Tensor | MemRef |
|------|--------|--------|
| 语义 | 值语义（SSA） | 引用语义（指针） |
| 内存分配 | 隐式 | 显式 `memref.alloc` |
| 地址空间 | 无 | `#hivm.address_space<...>` |
| 可变性 | 不可变 | 可变 |
| 生命周期 | 由 SSA 支配 | 由 alloc/dealloc 决定 |

### BufferizableOpInterface

MLIR 的 Bufferization 框架通过 `BufferizableOpInterface` 接口让每个操作定义自己的 bufferization 行为。实现该接口的操作需要指定：
- 如何将 Tensor 操作数映射到 MemRef
- 是否支持 inplace bufferization（输出复用输入缓冲区）
- bufferization 后的 IR 结构

## IR 变换效果

### 变换前（Tensor IR）

```mlir
%0 = tensor.empty() : tensor<16x16xf16>
%1 = hivm.hir.vadd ins(%arg0, %arg1 : tensor<16x16xf16>, tensor<16x16xf16>)
                  outs(%0 : tensor<16x16xf16>) -> tensor<16x16xf16>
```

### 变换后（MemRef IR）

```mlir
%0 = memref.alloc() : memref<16x16xf16, #hivm.address_space<ub>>
hivm.hir.vadd ins(%arg0, %arg1 : memref<16x16xf16, #hivm.address_space<ub>>,
                      memref<16x16xf16, #hivm.address_space<ub>>)
              outs(%0 : memref<16x16xf16, #hivm.address_space<ub>>)
```

## 相关 Pass

### hivm-clone-tensor-empty

- **Pass 名**：`hivm-clone-tensor-empty`
- **作用域**：`func::FuncOp`
- **功能**：为不同的 HIVM 操作输出克隆不同的 `tensor.empty`，确保每个输出有独立的 Tensor 分配，避免 Bufferization 时的 inplace 冲突。
- **依赖方言**：`hivm::HIVMDialect`, `tensor::TensorDialect`
- **源码参考**：[Passes.td:L102-L107](https://gitcode.com/Ascend/AscendNPU-IR/tree/master/bishengir/include/bishengir/Dialect/HIVM/Transforms/Passes.td#L102-L107)

### hivm-clone-scf-if-yield-operand

- **Pass 名**：`hivm-clone-scf-if-yield-operand`
- **作用域**：`func::FuncOp`
- **功能**：克隆 `scf.if` 的 yield 操作数，当多个 yield 返回相同的值或该值在 `scf.if` 之后仍被使用时，避免 PlanMemory 的 inplace 优化导致错误。
- **依赖方言**：`hivm::HIVMDialect`, `scf::SCFDialect`, `tensor::TensorDialect`
- **源码参考**：[Passes.td:L121-L128](https://gitcode.com/Ascend/AscendNPU-IR/tree/master/bishengir/include/bishengir/Dialect/HIVM/Transforms/Passes.td#L121-L128)

### hivm-opt-func-output

- **Pass 名**：`hivm-opt-func-output`
- **作用域**：`ModuleOp`
- **功能**：在 Bufferization 之后优化函数输出，移除不必要的地址返回。当函数返回的 MemRef 已经通过参数传入时，不需要再返回该地址。
- **依赖方言**：`memref::MemRefDialect`
- **源码参考**：[Passes.td:L352-L357](https://gitcode.com/Ascend/AscendNPU-IR/tree/master/bishengir/include/bishengir/Dialect/HIVM/Transforms/Passes.td#L352-L357)

### hivm-inline-otf-load-store

- **Pass 名**：`hivm-inline-otf-load-store`
- **作用域**：`func::FuncOp`
- **功能**：即时内联 Load 和 Store 操作，减少不必要的缓冲区中间拷贝。
- **依赖方言**：`bufferization::BufferizationDialect`, `memref::MemRefDialect`, `tensor::TensorDialect`, `hivm::HIVMDialect`
- **源码参考**：[Passes.td:L660-L666](https://gitcode.com/Ascend/AscendNPU-IR/tree/master/bishengir/include/bishengir/Dialect/HIVM/Transforms/Passes.td#L660-L666)

## Inplace Bufferization

Bufferization 的关键优化是 Inplace——当操作的输出可以复用输入缓冲区时，避免额外的 `memref.alloc`。HIVM 的结构化操作通过 `HIVMStructuredOpInterface` 支持 inplace 分析。

在 PlanMemory 阶段，Inplace 优化进一步扩展到物理地址级别：当两个缓冲区的生命周期不重叠时，它们可以共享同一块物理内存偏移。

## 源码参考

- Pass 定义：[Passes.td](https://gitcode.com/Ascend/AscendNPU-IR/tree/master/bishengir/include/bishengir/Dialect/HIVM/Transforms/Passes.td)
- Bufferization 测试：[buffer-opt.mlir](https://gitcode.com/Ascend/AscendNPU-IR/tree/master/bishengir/test/Dialect/HIVM/buffer-opt.mlir)
