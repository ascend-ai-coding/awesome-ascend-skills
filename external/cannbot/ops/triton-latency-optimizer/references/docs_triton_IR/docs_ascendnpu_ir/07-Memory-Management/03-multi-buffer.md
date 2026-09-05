# 多缓冲区 — Multi-Buffer

## 概述

多缓冲区（Multi-Buffer）是一种流水线优化技术，通过为同一逻辑缓冲区分配多个物理缓冲区实例，使 DMA 数据传输与计算操作重叠执行，从而提升流水线效率。在 Ascend NPU 上，这通常表现为双缓冲（Double Buffer）或多缓冲。

## 相关 Pass

### hivm-mark-multi-buffer

- **Pass 名**：`hivm-mark-multi-buffer`
- **作用域**：`func::FuncOp`
- **构造函数**：`mlir::hivm::createMarkMultiBufferPass()`
- **功能**：为 HIVM 操作标记多缓冲区。当 `enable-auto` 选项为 true 时，自动为符合条件的操作标记 `hivm.multi_buffer` 属性；当 `enable-auto` 为 false 时，不做任何操作。
- **说明**：L0C 地址空间的缓冲区不会被标记为多缓冲区。
- **依赖方言**：`hivm::HIVMDialect`, `annotation::AnnotationDialect`
- **源码参考**：[Passes.td:L55-L90](https://gitcode.com/Ascend/AscendNPU-IR/tree/master/bishengir/include/bishengir/Dialect/HIVM/Transforms/Passes.td#L55-L90)

#### 选项

| 选项名 | 类型 | 默认值 | 说明 |
|--------|------|--------|------|
| `enable-auto` | `bool` | `false` | 自动标记多缓冲区 |
| `limit-auto-multi-buffer-only-for-local-buffer` | `bool` | `false` | 禁止 workspace 的自动多缓冲标记 |
| `limit-auto-multi-buffer-of-local-buffer` | `MultiBufferStrategy` | `CUBE_NO_L0C` | 限制 local buffer 的自动多缓冲策略 |
| `limit-mix-auto-multi-buffer-buffer` | `MultiBufferStrategy` | `ONLY_CUBE` | 限制 Mix 场景的自动多缓冲策略 |
| `set-workspace-multibuffer` | `unsigned` | `2` | 覆盖 workspace 的多缓冲数量 |

#### MultiBufferStrategy 枚举值

| 策略 | 说明 |
|------|------|
| `no-limit` | 对 local 多缓冲不加限制 |
| `no-l0c` | 禁止 L0C 的多缓冲 |

### hivm-enable-multi-buffer

- **Pass 名**：`hivm-enable-multi-buffer`
- **作用域**：`func::FuncOp`
- **构造函数**：`mlir::hivm::createEnableMultiBufferPass()`
- **功能**：为被标记了 `hivm.multi_buffer` 属性的操作启用多缓冲区。将单缓冲区分配转换为多缓冲区分配。
- **依赖方言**：`mlir::affine::AffineDialect`
- **源码参考**：[Passes.td:L92-L100](https://gitcode.com/Ascend/AscendNPU-IR/tree/master/bishengir/include/bishengir/Dialect/HIVM/Transforms/Passes.td#L92-L100)

## 工作流程

```
1. hivm-mark-multi-buffer
   - 自动或手动为缓冲区标记 hivm.multi_buffer = N 属性
   - annotation.mark %buf {hivm.multi_buffer = 2 : i32}

2. hivm-enable-multi-buffer
   - 读取 hivm.multi_buffer 属性
   - 将单缓冲区分配转换为多缓冲区分配

3. hivm-plan-memory
   - 为多缓冲区分配多个物理偏移
   - pointer_cast(offset0, offset1, ..., offsetN-1)
```

## IR 表示

### 标记多缓冲区

```mlir
%src_ub = memref.alloc() : memref<16xf16, #hivm.address_space<ub>>
annotation.mark %src_ub {hivm.multi_buffer = 2 : i32} : memref<16xf16, #hivm.address_space<ub>>
```

### PlanMemory 后的多缓冲区

双缓冲区分配后，`pointer_cast` 包含两个偏移：

```mlir
// 双缓冲区
%src_ub = hivm.hir.pointer_cast(%const0, %const1) : memref<16xf16, #hivm.address_space<ub>>
```

四缓冲区分配后：

```mlir
// 四缓冲区
%src_ub = hivm.hir.pointer_cast(%const0, %const1, %const2, %const3) : memref<16xf16, #hivm.address_space<ub>>
```

## MultiBufferAttr

`hivm.multi_buffer` 属性用于标记一个缓冲区需要多缓冲区支持：

- **属性名**：`hivm.multi_buffer`
- **类型**：`i32` 整数属性
- **含义**：缓冲区实例数量（2 = 双缓冲，4 = 四缓冲等）

源码参考：[HIVMAttrs.td:L962-L966](https://gitcode.com/Ascend/AscendNPU-IR/tree/master/bishengir/include/bishengir/Dialect/HIVM/IR/HIVMAttrs.td#L962-L966)

## CVPipelining Pass

`cv-pipelining` Pass 与多缓冲区配合使用，实现 Cube 和 Vector 核心的流水线化：

- **Pass 名**：`cv-pipelining`
- **作用域**：`func::FuncOp`
- **功能**：对多缓冲化的 Mix CV 操作进行 Cube-Vector 流水线调度
- **选项**：
  - `pipeline-depth`（`int`，默认 `-1`）：指定流水线深度

源码参考：[Passes.td:L750-L759](https://gitcode.com/Ascend/AscendNPU-IR/tree/master/bishengir/include/bishengir/Dialect/HIVM/Transforms/Passes.td#L750-L759)

## 源码参考

- MarkMultiBuffer Pass 定义：[Passes.td:L55-L90](https://gitcode.com/Ascend/AscendNPU-IR/tree/master/bishengir/include/bishengir/Dialect/HIVM/Transforms/Passes.td#L55-L90)
- EnableMultiBuffer Pass 定义：[Passes.td:L92-L100](https://gitcode.com/Ascend/AscendNPU-IR/tree/master/bishengir/include/bishengir/Dialect/HIVM/Transforms/Passes.td#L92-L100)
- CVPipelining Pass 定义：[Passes.td:L750-L759](https://gitcode.com/Ascend/AscendNPU-IR/tree/master/bishengir/include/bishengir/Dialect/HIVM/Transforms/Passes.td#L750-L759)
- MultiBufferAttr 定义：[HIVMAttrs.td:L962-L966](https://gitcode.com/Ascend/AscendNPU-IR/tree/master/bishengir/include/bishengir/Dialect/HIVM/IR/HIVMAttrs.td#L962-L966)
