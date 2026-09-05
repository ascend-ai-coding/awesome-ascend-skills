# MemRefExt 方言

## 1. 概述

MemRefExt 方言扩展了 MLIR 标准的 MemRef 方言，提供 NPU 设备特有的内存分配操作，特别是工作空间（Workspace）内存的分配。

- **方言名称**：`memref_ext`
- **C++ 命名空间**：`::bishengir::memref_ext`
- **依赖方言**：`arith::ArithDialect`

> 源码参考：[MemRefExtOps.td](https://gitcode.com/Ascend/AscendNPU-IR/tree/master/bishengir/include/bishengir/Dialect/MemRefExt/IR/MemRefExtOps.td)、[MemRefExtBase.td](https://gitcode.com/Ascend/AscendNPU-IR/tree/master/bishengir/include/bishengir/Dialect/MemRefExt/IR/MemRefExtBase.td)

## 2. 方言定义

```tablegen
def MemRefExt_Dialect : Dialect {
  let name = "memref_ext";
  let cppNamespace = "::bishengir::memref_ext";
  let description = [{
    extended memref dialect
  }];
  let dependentDialects = ["arith::ArithDialect"];
  let hasConstantMaterializer = 1;
}
```

## 3. 操作定义

### 3.1 memref_ext.alloc_workspace

#### 功能

分配工作空间内存。工作空间用于 Kernel 执行过程中的临时存储，其大小由 Host 端的 `hacc.infer_workspace_shape_function` 推断。

#### 操作签名

| 操作数/结果 | 类型 | 说明 |
|-------------|------|------|
| `workspaceArg` | `Optional<AnyMemRef>` | 已有的工作空间参数（复用场景） |
| `dynamicSize` | `Variadic<Index>` | 动态维度大小 |
| `offset` | `Variadic<Index>` | 偏移量 |
| `memref` | `AnyMemRef` | 分配的 MemRef 结果 |

#### 内存效果

- `MemAlloc<DefaultResource, 0, FullEffect>`

#### MLIR 示例

静态大小分配：

```mlir
%ws = memref_ext.alloc_workspace() : memref<100xi8>
```

动态大小分配：

```mlir
%ws = memref_ext.alloc_workspace(%dynamic) : memref<2x?xi32>
```

从已有工作空间分配（带偏移）：

```mlir
%ws = memref_ext.alloc_workspace(%dynamic) from %arg offset = [%offset]
  : from memref<?xi8> to memref<2x?xi32>
```

## 4. Lowering

MemRefExt 操作在编译流程中会被 lowering 为标准 MemRef 操作或 HIVM 层的内存操作：

```
memref_ext.alloc_workspace
  │
  ├── Lowering 到 HIVM 层的内存分配
  │
  └── Lowering 到标准 memref.alloc + 偏移计算
```

## 5. 与 HACC 方言的协作

工作空间内存的生命周期由 HACC 方言管理：

1. Host 端 `hacc.infer_workspace_shape_function` 计算工作空间大小
2. Host 端分配工作空间内存
3. 工作空间通过 `hacc.arg_type = #hacc.arg_type<workspace>` 传递给 Device Kernel
4. Device Kernel 内部使用 `memref_ext.alloc_workspace` 从传入的工作空间中分配子区域
