# 额外缓冲区 — Extra Buffer

## 概述

某些 HIVM 操作在执行时需要额外的临时缓冲区（temp_buffer），例如类型转换中的中间存储、broadcast 操作的展开缓冲区等。额外缓冲区 Pass 负责分配和管理这些临时存储。

## 相关 Pass

### hivm-alloc-extra-buffer

- **Pass 名**：`hivm-alloc-extra-buffer`
- **作用域**：`func::FuncOp`
- **构造函数**：`mlir::hivm::createAllocExtraBufferPass()`
- **功能**：为需要额外临时缓冲区的操作分配临时缓冲区。某些 HIVM 操作（如带 broadcast 的二元操作、某些类型转换操作）需要额外的 temp_buffer 来存储中间结果。
- **依赖方言**：`hivm::HIVMDialect`
- **源码参考**：[Passes.td:L320-L324](https://gitcode.com/Ascend/AscendNPU-IR/tree/master/bishengir/include/bishengir/Dialect/HIVM/Transforms/Passes.td#L320-L324)

### hivm-outline-alloc-in-VF

- **Pass 名**：`hivm-outline-alloc-in-VF`
- **作用域**：`ModuleOp`
- **构造函数**：`mlir::hivm::createOutlineAllocInVFPass()`
- **功能**：将 VF（Vector Function）中具有静态形状的 `memref.alloc` 提取到函数外部，减少 VF 内部的内存分配开销。
- **依赖方言**：`hivm::HIVMDialect`
- **源码参考**：[Passes.td:L326-L330](https://gitcode.com/Ascend/AscendNPU-IR/tree/master/bishengir/include/bishengir/Dialect/HIVM/Transforms/Passes.td#L326-L330)

### hivm-constantize-buffer-size

- **Pass 名**：`hivm-constantize-buffer-size`
- **作用域**：`func::FuncOp`
- **构造函数**：`mlir::hivm::createConstantizeBufferSizePass()`
- **功能**：尝试将动态形状的缓冲区常量化。通过对原始形状进行上界估计，如果成功则创建新的静态形状分配，并通过 subview 转换为原始形状供后续使用。
- **依赖方言**：`memref::MemRefDialect`
- **源码参考**：[Passes.td:L339-L350](https://gitcode.com/Ascend/AscendNPU-IR/tree/master/bishengir/include/bishengir/Dialect/HIVM/Transforms/Passes.td#L339-L350)

### hivm-set-buffer-size

- **Pass 名**：`hivm-set-buffer-size`
- **作用域**：`func::FuncOp`
- **构造函数**：`mlir::hivm::createSetBufferSizePass()`
- **依赖方言**：`arith::ArithDialect`, `memref::MemRefDialect`
- **源码参考**：[Passes.td:L409-L412](https://gitcode.com/Ascend/AscendNPU-IR/tree/master/bishengir/include/bishengir/Dialect/HIVM/Transforms/Passes.td#L409-L412)

### hivm-auto-infer-buffer-size

- **Pass 名**：`hivm-auto-infer-buffer-size`
- **作用域**：`func::FuncOp`
- **构造函数**：`mlir::hivm::createAutoInferBufferSizePass()`
- **功能**：通过插入 `annotation.mark` 操作自动推断缓冲区大小。
- **依赖方言**：`annotation::AnnotationDialect`
- **源码参考**：[Passes.td:L636-L642](https://gitcode.com/Ascend/AscendNPU-IR/tree/master/bishengir/include/bishengir/Dialect/HIVM/Transforms/Passes.td#L636-L642)

## temp_buffer 机制

某些 HIVM 操作支持 `temp_buffer` 操作数，用于指定操作执行时需要的额外临时存储：

```mlir
// 带 temp_buffer 的 vsub 操作
hivm.hir.vsub ins(%alloc, %alloc_0 : memref<64x128xf32, #hivm.address_space<ub>>,
                                    memref<64x1xf32, #hivm.address_space<ub>>)
  outs(%alloc_1 : memref<64x128xf32, #hivm.address_space<ub>>)
  temp_buffer(%alloc_2 : memref<512xf32, #hivm.address_space<ub>>)
  broadcast = [1]
```

### temp_buffer 的典型场景

1. **Broadcast 操作**：当二元操作的输入需要 broadcast 时，temp_buffer 用于存储 broadcast 后的中间结果。
2. **类型转换**：某些类型转换需要中间存储。
3. **Decompose 操作**：`hivm-decompose-op` 在动态情况下会创建带有 `buffer_size_in_byte` 标记的额外缓冲区。

## ExtraBufferOpInterface

`ExtraBufferOpInterface` 是 HIVM 的操作接口，用于声明操作是否需要额外缓冲区以及缓冲区的大小。实现该接口的操作在 `hivm-alloc-extra-buffer` Pass 中会被自动分配 temp_buffer。

源码参考：[ExtraBufferOpInterface.td](https://gitcode.com/Ascend/AscendNPU-IR/tree/master/bishengir/include/bishengir/Dialect/HIVM/Interfaces/ExtraBufferOpInterface.td)

## 源码参考

- AllocExtraBuffer Pass：[Passes.td:L320-L324](https://gitcode.com/Ascend/AscendNPU-IR/tree/master/bishengir/include/bishengir/Dialect/HIVM/Transforms/Passes.td#L320-L324)
- OutlineAllocInVF Pass：[Passes.td:L326-L330](https://gitcode.com/Ascend/AscendNPU-IR/tree/master/bishengir/include/bishengir/Dialect/HIVM/Transforms/Passes.td#L326-L330)
- ConstantizeBufferSize Pass：[Passes.td:L339-L350](https://gitcode.com/Ascend/AscendNPU-IR/tree/master/bishengir/include/bishengir/Dialect/HIVM/Transforms/Passes.td#L339-L350)
- AutoInferBufferSize Pass：[Passes.td:L636-L642](https://gitcode.com/Ascend/AscendNPU-IR/tree/master/bishengir/include/bishengir/Dialect/HIVM/Transforms/Passes.td#L636-L642)
