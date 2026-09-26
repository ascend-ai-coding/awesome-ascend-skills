# 内存对齐 — Memory Alignment

## 概述

Ascend NPU 硬件对内存访问有严格的对齐要求。内存对齐 Pass 确保缓冲区分配和 stride 满足硬件约束，避免未对齐访问导致的性能下降或执行错误。

## 相关 Pass

### hivm-align-alloc-size

- **Pass 名**：`hivm-align-alloc-size`
- **作用域**：`mlir::ModuleOp`
- **构造函数**：`mlir::hivm::createAlignAllocSizePass()`
- **功能**：自动对齐 `memref.alloc` 的大小。某些 HIVM 操作的访问大小只能对齐到硬件单元大小，此 Pass 调整 `memref.alloc` 的大小以避免越界访问。
- **依赖方言**：`mlir::memref::MemRefDialect`, `hivm::HIVMDialect`, `annotation::AnnotationDialect`
- **源码参考**：[Passes.td:L437-L449](https://gitcode.com/Ascend/AscendNPU-IR/tree/master/bishengir/include/bishengir/Dialect/HIVM/Transforms/Passes.td#L437-L449)

### hivm-mark-stride-align

- **Pass 名**：`hivm-mark-stride-align`
- **作用域**：`func::FuncOp`
- **构造函数**：`mlir::hivm::createMarkStrideAlignPass()`
- **功能**：自动为 HIVM 操作的 memref 操作数标注 stride 对齐标记。分析所有 HIVM 操作，为其 memref 操作数自动添加 `stride_align_dims` 和 `stride_align_value_in_byte` 标注。
- **依赖方言**：`mlir::memref::MemRefDialect`, `hivm::HIVMDialect`, `annotation::AnnotationDialect`
- **源码参考**：[Passes.td:L451-L461](https://gitcode.com/Ascend/AscendNPU-IR/tree/master/bishengir/include/bishengir/Dialect/HIVM/Transforms/Passes.td#L451-L461)

### hivm-enable-stride-align

- **Pass 名**：`hivm-enable-stride-align`
- **作用域**：`ModuleOp`
- **构造函数**：`mlir::hivm::createEnableStrideAlignPass()`
- **功能**：根据 stride 对齐标注重新分配 memref。读取 `annotation.mark` 中的 `stride_align_dims` 和 `stride_align_value_in_byte` 标注，重新分配 memref 以满足 stride 对齐要求。
- **依赖方言**：`mlir::memref::MemRefDialect`, `annotation::AnnotationDialect`, `arith::ArithDialect`
- **源码参考**：[Passes.td:L463-L473](https://gitcode.com/Ascend/AscendNPU-IR/tree/master/bishengir/include/bishengir/Dialect/HIVM/Transforms/Passes.td#L463-L473)

### hivm-lift-lowest-stride

- **Pass 名**：`hivm-lift-lowest-stride`
- **作用域**：`func::FuncOp`
- **构造函数**：`mlir::hivm::createLiftLowestStridePass()`
- **功能**：提升 HIVM 操作操作数的最低 stride。对于大多数 HIVM 结构化操作，如果最后一个维度不连续，则提升最低 stride。例外：MacroOp 和 VArangeOp。
- **依赖方言**：`mlir::memref::MemRefDialect`, `hivm::HIVMDialect`
- **源码参考**：[Passes.td:L475-L490](https://gitcode.com/Ascend/AscendNPU-IR/tree/master/bishengir/include/bishengir/Dialect/HIVM/Transforms/Passes.td#L475-L490)

#### LiftLowestStride 示例

```mlir
// 变换前：最后一个维度不连续（stride = 8，但元素大小为 2 字节）
%0 : memref<16xf16, strided<[8]>>

// 变换后：提升最低 stride，使最后一个维度连续
%0 : memref<16x1xf32, strided<[8, 1]>>
```

## 对齐相关属性

### StrideAlignDims

- **属性名**：`hivm.stride_align_dims`
- **功能**：标记需要 stride 对齐的维度
- **源码参考**：[HIVMAttrs.td:L968-L972](https://gitcode.com/Ascend/AscendNPU-IR/tree/master/bishengir/include/bishengir/Dialect/HIVM/IR/HIVMAttrs.td#L968-L972)

### StrideAlignValueInByte

- **属性名**：`hivm.stride_align_value_in_byte`
- **功能**：标记 stride 对齐的字节值
- **源码参考**：[HIVMAttrs.td:L974-L979](https://gitcode.com/Ascend/AscendNPU-IR/tree/master/bishengir/include/bishengir/Dialect/HIVM/IR/HIVMAttrs.td#L974-L979)

### AllocAlignDims

- **属性名**：`hivm.alloc_align_dims`
- **功能**：标记分配时需要对齐的维度
- **源码参考**：[HIVMAttrs.td:L981-L985](https://gitcode.com/Ascend/AscendNPU-IR/tree/master/bishengir/include/bishengir/Dialect/HIVM/IR/HIVMAttrs.td#L981-L985)

### AllocAlignValueInByte

- **属性名**：`hivm.alloc_align_value_in_byte`
- **功能**：标记分配对齐的字节值
- **源码参考**：[HIVMAttrs.td:L987-L992](https://gitcode.com/Ascend/AscendNPU-IR/tree/master/bishengir/include/bishengir/Dialect/HIVM/IR/HIVMAttrs.td#L987-L992)

## 硬件对齐约束

不同地址空间有不同的对齐要求，定义在 [NPUTargetSpec.td](https://gitcode.com/Ascend/AscendNPU-IR/tree/master/bishengir/include/bishengir/Dialect/HACC/Targets/NPUTargetSpec.td) 中：

| 地址空间 | 对齐大小 | 说明 |
|----------|----------|------|
| UB | 256 bits (32B) | `UbAlignSize = 256` |
| L1 | 256 bits (32B) | `L1AlignSize = 256` |
| L0C | 4096 bits (512B) | `L0cAlignSize = 4096` |

## 相关 Pass

### hivm-reduce-rank-subview

- **Pass 名**：`hivm-reduce-rank-subview`
- **作用域**：`func::FuncOp`
- **功能**：使用 subview 降低秩，有助于满足对齐约束。
- **依赖方言**：`mlir::memref::MemRefDialect`, `hivm::HIVMDialect`
- **源码参考**：[Passes.td:L492-L497](https://gitcode.com/Ascend/AscendNPU-IR/tree/master/bishengir/include/bishengir/Dialect/HIVM/Transforms/Passes.td#L492-L497)

## 源码参考

- AlignAllocSize Pass：[Passes.td:L437-L449](https://gitcode.com/Ascend/AscendNPU-IR/tree/master/bishengir/include/bishengir/Dialect/HIVM/Transforms/Passes.td#L437-L449)
- MarkStrideAlign Pass：[Passes.td:L451-L461](https://gitcode.com/Ascend/AscendNPU-IR/tree/master/bishengir/include/bishengir/Dialect/HIVM/Transforms/Passes.td#L451-L461)
- EnableStrideAlign Pass：[Passes.td:L463-L473](https://gitcode.com/Ascend/AscendNPU-IR/tree/master/bishengir/include/bishengir/Dialect/HIVM/Transforms/Passes.td#L463-L473)
- LiftLowestStride Pass：[Passes.td:L475-L490](https://gitcode.com/Ascend/AscendNPU-IR/tree/master/bishengir/include/bishengir/Dialect/HIVM/Transforms/Passes.td#L475-L490)
- 对齐属性定义：[HIVMAttrs.td:L968-L992](https://gitcode.com/Ascend/AscendNPU-IR/tree/master/bishengir/include/bishengir/Dialect/HIVM/IR/HIVMAttrs.td#L968-L992)
