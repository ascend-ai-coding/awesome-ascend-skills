# hir.store

> 关键词：Store、DMA、MTE3、UB 到 GM、原子操作

## 概述

`hir.store` 是 HIVM 方言中的核心 DMA 操作，用于将数据从本地缓冲区（UB）存储到全局内存（GM）。该操作映射到硬件的 MTE3 Pipeline，是大多数内核中数据写回的主要方式。

`hir.store` 支持原子操作模式，可以在存储时执行原子加、原子最大值、原子最小值等操作，用于实现 Triton 中的原子存储语义。

> Python API 对应：tl.store -- 详见 [docs_triton_ascend/02-Core-API/01-memory-ops.md](../../../docs_triton_ascend/02-Core-API/01-memory-ops.md)

## IR 操作定义

### TableGen 定义

来源：[HIVMDMAOps.td:L138-L190](https://gitcode.com/Ascend/AscendNPU-IR/tree/master/bishengir/include/bishengir/Dialect/HIVM/IR/HIVMDMAOps.td#L138-L190)

```tablegen
def StoreOp : HIVM_DmaOp<"store", [
  StaticMaxRankTrait<3>,
  SinglePipeOpTrait, OpPipeTrait<"PIPE::PIPE_MTE3">,
  DeclareOpInterfaceMethods<HIVMInferCoreTypeInterface, ["inferCoreType"]>,
  UniformReassociationFlattenTrait,
  DeclareOpInterfaceMethods<FlattenInterface, ["getLimitedAxes"]>,
  OperElemTypeConstraints<[0], [I8, UI8, I16, UI16, F16, BF16,
                                I32, UI32, F32, UI64, I64, F8E4M3FN, F8E5M2]>,
  DeclareOpInterfaceMethods<HIVMStructuredOpInterface, ["getIndexingMaps"]>,
]> {
  let summary = "HIVM data store operation";
  let description = [{
    Stores the data on local buffer to global memory.
    Currently only support storing data on the unified buffer.

    Examples:
    ```mlir
    hivm.store ins(%src : memref<16x16xf16, #hivm.address_space<ub>>) outs(%dst : memref<16x16xf16, #hivm.address_space<gm>>)
    ```

    Constraints:
    - `src` and `dst` are expected to have the same element type.
    - If `atomic_kind` is set, the kind is one of `add`, `max`, `min`.
  }];
  let arguments = (ins TensorOrMemref:$src,
                       TensorOrMemref:$dst,
                       OptionalAttr<HIVM_AtomicKindAttr>:$atomic_kind
  );
  let results = (outs Optional<AnyRankedTensor>:$result_tensor);
  let builders = [
    OpBuilder<(ins "TypeRange":$res, "Value":$src, "Value":$dst)>
  ];
  let assemblyFormat = [{
    `ins` `(` $src `:` type($src) `)`
    `outs` `(` $dst `:` type($dst) `)`
    attr-dict
    (`atomic` `=` $atomic_kind^)?
    (`->` type($result_tensor)^)?
  }];
  let hasFolder = 1;
  let hasVerifier = 1;
  let extraClassDeclaration = DmaOpBaseDecl # [{
    // Return whether atomic store is enabled.
    bool isAtomic();

    // Return whether block-sync atomic store enabled is implemented by hardware.
    bool isHWAtomic();

    // Return whether block-sync atomic store enabled is implemented by software.
    bool isSWAtomic();
  }];
}
```

### MLIR 语法

```mlir
hivm.hir.store ins(%src : memref<MxNxf16, #hivm.address_space<ub>>)
                outs(%dst : memref<MxNxf16, #hivm.address_space<gm>>)

hivm.hir.store ins(%src : memref<MxNxf16, #hivm.address_space<ub>>)
                outs(%dst : memref<MxNxf16, #hivm.address_space<gm>>)
                atomic = #hivm.atomic_kind<add>
```

### 参数说明

| 参数 | 类型 | 必选 | 说明 | 约束 |
|------|------|------|------|------|
| `src` | TensorOrMemref | 是 | 源数据缓冲区 | 必须为 UB 地址空间（memref 语义下） |
| `dst` | TensorOrMemref | 是 | 目标数据缓冲区 | 必须为 GM 地址空间（memref 语义下） |
| `atomic_kind` | HIVM_AtomicKindAttr | 否 | 原子操作类型 | NONE / ADD / MAX / MIN / AND / OR / XOR / CAS / XCHG |

### 结果说明

| 结果 | 类型 | 说明 |
|------|------|------|
| `result_tensor` | Optional<AnyRankedTensor> | Tensor 语义下的结果张量 |

### 属性说明

| 属性 | 类型 | 默认值 | 说明 | 可选值 |
|------|------|--------|------|--------|
| `atomic_kind` | HIVM_AtomicKindAttr | 无 | 原子操作类型 | 见下表 |

#### AtomicKind 枚举

定义于 [HIVMAttrs.td:L637-L664](https://gitcode.com/Ascend/AscendNPU-IR/tree/master/bishengir/include/bishengir/Dialect/HIVM/IR/HIVMAttrs.td#L637-L664)

| 枚举值 | 数值 | IR 字面量 | 说明 | 实现方式 |
|--------|------|----------|------|---------|
| `NONE` | 0 | `none` | 非原子操作 | - |
| `ADD` | 1 | `add` | 原子加 | 硬件实现 |
| `MAX` | 2 | `max` | 原子最大值 | 硬件实现 |
| `MIN` | 3 | `min` | 原子最小值 | 硬件实现 |
| `AND` | 4 | `and` | 原子与 | 软件实现 |
| `OR` | 5 | `or` | 原子或 | 软件实现 |
| `XOR` | 6 | `xor` | 原子异或 | 软件实现 |
| `CAS` | 7 | `cas` | 原子比较并交换 | 软件实现 |
| `XCHG` | 8 | `xchg` | 原子交换 | 软件实现 |

### 数据类型约束

来源：`OperElemTypeConstraints<[0], [...]>`

| 支持的元素类型 | 说明 |
|---------------|------|
| `i8`, `ui8` | 8 位整数 |
| `i16`, `ui16` | 16 位整数 |
| `f16` | 半精度浮点 |
| `bf16` | BFloat16 |
| `i32`, `ui32` | 32 位整数 |
| `f32` | 单精度浮点 |
| `ui64`, `i64` | 64 位整数 |
| `f8E4M3FN` | 8 位浮点 E4M3 格式 |
| `f8E5M2` | 8 位浮点 E5M2 格式 |

## IR 示例

### 基础存储

最简单的 UB 到 GM 数据存储：

```mlir
func.func @hivm_memref_store_ub_to_gm() {
  %src = memref.alloc() : memref<16x16xf16, #hivm.address_space<ub>>
  %dst = memref.alloc() : memref<16x16xf16, #hivm.address_space<gm>>
  hivm.hir.store ins(%src : memref<16x16xf16, #hivm.address_space<ub>>)
                outs(%dst : memref<16x16xf16, #hivm.address_space<gm>>)
  return
}
```

### 原子加存储

```mlir
func.func @hivm_memref_atomic_add_store() {
  %src = memref.alloc() : memref<16x16xf32, #hivm.address_space<ub>>
  %dst = memref.alloc() : memref<16x16xf32, #hivm.address_space<gm>>
  hivm.hir.store ins(%src : memref<16x16xf32, #hivm.address_space<ub>>)
                outs(%dst : memref<16x16xf32, #hivm.address_space<gm>>)
                atomic = #hivm.atomic_kind<add>
  return
}
```

### 原子最大值存储

```mlir
func.func @hivm_memref_atomic_max_store() {
  %src = memref.alloc() : memref<16x16xf16, #hivm.address_space<ub>>
  %dst = memref.alloc() : memref<16x16xf16, #hivm.address_space<gm>>
  hivm.hir.store ins(%src : memref<16x16xf16, #hivm.address_space<ub>>)
                outs(%dst : memref<16x16xf16, #hivm.address_space<gm>>)
                atomic = #hivm.atomic_kind<max>
  return
}
```

## IR 层约束与验证

来源：[HIVMDMAOps.cpp:L348-L376](https://gitcode.com/Ascend/AscendNPU-IR/tree/master/bishengir/lib/Dialect/HIVM/IR/HIVMDMAOps.cpp#L348-L376)

1. **元素类型一致性**：`src` 和 `dst` 的元素类型必须相同
2. **Rank 一致性**：`src` 和 `dst` 必须具有相同的 rank
3. **地址空间约束**（memref 语义下）：
   - `src` 必须为 UB 地址空间
   - `dst` 必须为 GM 地址空间
4. **Tensor 语义约束**：
   - `result_tensor` 的元素类型必须与 `dst` 相同
   - `result_tensor` 的 rank 必须与 `dst` 相同

### 原子操作分类

来源：[HIVMDMAOps.cpp:L378-L394](https://gitcode.com/Ascend/AscendNPU-IR/tree/master/bishengir/lib/Dialect/HIVM/IR/HIVMDMAOps.cpp#L378-L394)

- **硬件原子操作**（`isHWAtomic()`）：ADD、MAX、MIN -- 由硬件直接支持
- **软件原子操作**（`isSWAtomic()`）：AND、OR、XOR、CAS、XCHG -- 通过软件锁机制实现

## 与其他 IR 操作的关系

### 从 Triton 到 HIVM

```
tt.store          -->  hivm.hir.store
tt.store (atomic) -->  hivm.hir.store atomic = #hivm.atomic_kind<add>
```

### 后续 Lowering

`hir.store` 最终被 lowering 为库函数调用，函数名格式为：

```
store_ubuf_to_gm_{rank}d_{datatype}
```

## 常见问题

### Q: 为什么 store 只支持 UB -> GM？

A: 这是当前硬件的限制。如果需要将 L1 数据写回 GM，应使用 `hir.nz2nd`；如果需要将 L0C 数据写出，应使用 `hir.fixpipe`。

### Q: 硬件原子和软件原子有什么区别？

A: 硬件原子（ADD/MAX/MIN）由 Ascend NPU 的 DMA 引擎直接支持，性能更高。软件原子（AND/OR/XOR/CAS/XCHG）需要通过软件锁机制实现，开销较大。在性能敏感场景中，应优先使用硬件原子操作。

### Q: 如何在 store 之前设置全局原子模式？

A: 可以使用 `hir.set_atomic` 操作设置全局原子模式，后续的 store 操作将自动启用原子语义。使用 `hir.set_atomic kind= #hivm.atomic_kind<NONE>` 重置。

## 相关文档

- Python API：[docs_triton_ascend/02-Core-API/01-memory-ops.md](../../../docs_triton_ascend/02-Core-API/01-memory-ops.md)
- DMA 操作总览：[00-overview.md](00-overview.md)
- 原子操作详解：[07-atomic.md](07-atomic.md)
- 源码参考：
  - [HIVMDMAOps.td](https://gitcode.com/Ascend/AscendNPU-IR/tree/master/bishengir/include/bishengir/Dialect/HIVM/IR/HIVMDMAOps.td) - TableGen 定义
  - [HIVMDMAOps.cpp](https://gitcode.com/Ascend/AscendNPU-IR/tree/master/bishengir/lib/Dialect/HIVM/IR/HIVMDMAOps.cpp) - 验证逻辑实现
