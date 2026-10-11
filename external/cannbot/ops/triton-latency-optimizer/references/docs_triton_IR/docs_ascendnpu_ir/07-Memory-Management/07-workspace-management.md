# 工作空间管理 — Workspace Management

## 概述

工作空间（Workspace）是 Ascend NPU 编译模型中用于全局内存分配的机制。与局部内存（`memref.alloc`）不同，工作空间的生命周期跨越整个模型执行，由运行时统一分配和管理。工作空间管理 Pass 负责工作空间的大小推断、参数绑定和 Mix CV 场景的工作空间插入。

## 相关 Pass

### hivm-insert-infer-workspace-size-func

- **Pass 名**：`hivm-insert-infer-workspace-size-func`
- **作用域**：`ModuleOp`
- **构造函数**：`mlir::hivm::createInsertInferWorkSpaceSizeFuncPass()`
- **功能**：插入推断工作空间大小的函数。为模块中的每个函数生成一个对应的工作空间大小推断函数，该函数根据缓冲区分配计算出所需的工作空间总大小。
- **依赖方言**：`hivm::HIVMDialect`, `func::FuncDialect`, `arith::ArithDialect`, `memref::MemRefDialect`
- **源码参考**：[Passes.td:L163-L169](https://gitcode.com/Ascend/AscendNPU-IR/tree/master/bishengir/include/bishengir/Dialect/HIVM/Transforms/Passes.td#L163-L169)

### hivm-bind-workspace-arg

- **Pass 名**：`hivm-bind-workspace-arg`
- **作用域**：`ModuleOp`
- **构造函数**：`mlir::hivm::createBindWorkSpaceArgPass()`
- **功能**：将工作空间绑定到函数参数。为模块中的每个函数添加工作空间参数（`memref_ext.alloc_workspace`），并将函数内的 `memref.alloc` 替换为工作空间内的偏移分配。
- **依赖方言**：`hivm::HIVMDialect`, `func::FuncDialect`, `memref::MemRefDialect`, `bishengir::memref_ext::MemRefExtDialect`
- **源码参考**：[Passes.td:L171-L178](https://gitcode.com/Ascend/AscendNPU-IR/tree/master/bishengir/include/bishengir/Dialect/HIVM/Transforms/Passes.td#L171-L178)

### insert-workspace-for-mix-cv

- **Pass 名**：`insert-workspace-for-mix-cv`
- **作用域**：`ModuleOp`
- **构造函数**：`mlir::hivm::createInsertWorkSpaceForMixCVPass()`
- **功能**：为 Mix CV 场景插入工作空间。在 Cube-Vector 混合执行模式下，需要额外的工作空间来存储 Cube 和 Vector 核心之间的中间数据。
- **依赖方言**：`hivm::HIVMDialect`, `func::FuncDialect`, `arith::ArithDialect`, `memref::MemRefDialect`, `bishengir::memref_ext::MemRefExtDialect`
- **源码参考**：[Passes.td:L180-L188](https://gitcode.com/Ascend/AscendNPU-IR/tree/master/bishengir/include/bishengir/Dialect/HIVM/Transforms/Passes.td#L180-L188)

## 工作空间分配流程

```
1. hivm-plan-memory (global-workspace-plan)
   - 为 memref_ext.alloc_workspace 分配全局工作空间偏移

2. hivm-insert-infer-workspace-size-func
   - 生成 infer_workspace_size 函数
   - 计算每个函数所需的工作空间总大小

3. hivm-bind-workspace-arg
   - 将工作空间作为函数参数传入
   - 替换 memref.alloc 为工作空间偏移

4. insert-workspace-for-mix-cv (Mix CV 场景)
   - 为 Cube-Vector 混合执行插入额外工作空间
```

## MemRefExt 方言相关操作

工作空间管理使用 MemRefExt 方言的操作：

### memref_ext.alloc_workspace

分配全局工作空间缓冲区。

```mlir
%workspace = memref_ext.alloc_workspace() : memref<256xi8>
```

### memref_ext.bind_workspace_arg

将工作空间绑定到函数参数。

## 工作空间复用

PlanMemory Pass 的 `enable-global-workspace-reuse` 选项控制全局工作空间的复用：

- **默认（false）**：不启用复用，每个缓冲区独立分配偏移
- **设为 true**：启用复用，生命周期不重叠的缓冲区共享工作空间偏移

## 工作空间大小推断

`hivm-insert-infer-workspace-size-func` Pass 生成的推断函数会：

1. 遍历函数内所有 `memref.alloc` 和 `memref_ext.alloc_workspace`
2. 计算每个分配的大小（考虑对齐）
3. 考虑 Inplace 复用后的实际大小
4. 生成推断函数，返回所需工作空间的总字节数

推断函数的签名：

```mlir
func.func @infer_workspace_size() -> i64 {
  %size = arith.constant 65536 : i64
  return %size : i64
}
```

## 与局部内存规划的关系

| 特性 | 局部内存规划 (local-mem-plan) | 全局工作空间规划 (global-workspace-plan) |
|------|------|------|
| 分配方式 | `memref.alloc` → `pointer_cast(offset)` | `memref_ext.alloc_workspace` → 偏移绑定 |
| 生命周期 | 函数内 | 跨函数 |
| 地址空间 | UB/L1/L0C 等 | GM |
| 管理方式 | 编译时静态分配 | 运行时动态分配 |
| 复用 | Inplace 复用 | 全局复用 |

## 源码参考

- InsertInferWorkSpaceSizeFunc Pass：[Passes.td:L163-L169](https://gitcode.com/Ascend/AscendNPU-IR/tree/master/bishengir/include/bishengir/Dialect/HIVM/Transforms/Passes.td#L163-L169)
- BindWorkSpaceArg Pass：[Passes.td:L171-L178](https://gitcode.com/Ascend/AscendNPU-IR/tree/master/bishengir/include/bishengir/Dialect/HIVM/Transforms/Passes.td#L171-L178)
- InsertWorkSpaceForMixCV Pass：[Passes.td:L180-L188](https://gitcode.com/Ascend/AscendNPU-IR/tree/master/bishengir/include/bishengir/Dialect/HIVM/Transforms/Passes.td#L180-L188)
- PlanMemory Pass 选项：[Passes.td:L130-L161](https://gitcode.com/Ascend/AscendNPU-IR/tree/master/bishengir/include/bishengir/Dialect/HIVM/Transforms/Passes.td#L130-L161)
