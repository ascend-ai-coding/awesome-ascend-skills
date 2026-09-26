# hir.atomic_cas / hir.atomic_xchg

> 关键词：Atomic、CAS、Compare-And-Swap、XCHG、Exchange、原子操作

## 概述

HIVM 方言提供了两个独立的原子操作：`hir.atomic_cas`（原子比较并交换）和 `hir.atomic_xchg`（原子交换）。这两个操作用于在全局内存（GM）上执行不可分割的读-修改-写操作，是实现并发同步原语的基础。

与 `hir.store` 的原子模式不同，`atomic_cas` 和 `atomic_xchg` 是独立的操作，不依赖于 `hir.set_atomic` 的全局原子模式设置。

> Python API 对应：tl.atomic_cas / tl.atomic_xchg -- 详见 [docs_triton_ascend/02-Core-API/01-memory-ops.md](../../../docs_triton_ascend/02-Core-API/01-memory-ops.md)

## hir.atomic_cas

### 概述

原子比较并交换（Compare-And-Swap, CAS）操作。该操作读取内存位置 V 的值，如果等于期望值 A，则将其更新为新值 B，并返回 V 的原始值。整个过程是原子的，不会被其他线程中断。

### TableGen 定义

来源：[HIVMDMAOps.td:L406-L443](https://gitcode.com/Ascend/AscendNPU-IR/tree/master/bishengir/include/bishengir/Dialect/HIVM/IR/HIVMDMAOps.td#L406-L443)

```tablegen
def AtomicCasOp : HIVM_Op<"atomic_cas",
    [NoLibraryFunctionTrait,
     OperElemTypeConstraints<[0], [I8, I16, I32, F16, F32, I64]>
    ]> {
  let summary = "Atomic Compare-And-Swap (CAS) Op";
  let description = [{
    Compare-And-Swap (CAS) is an atomic operation that consists of three operands:
    Memory location (V), Expected old value (A), New value (B).
    The semantics of the operation are: the value of V is updated to B,
    only if the value of memory location V is equal to the expected old value A.
    The operation returns the original value of V regardless of whether it is updated or not.

    Constraints:
      1. The input memref and output memref must have the same rank
         and the same element type.

    Arguments:
      * `src0`: expected old value
      * `src1`: new value
      * `dst`: memory location in GM

    Examples:
    ```mlir
    hivm.hir.atomic_cas ins(%src0, %src1 : memref<?xf32>, memref<?xf32>) outs(%dst : memref<?xf32>)
    %result = hivm.hir.atomic_cas ins(%src0, %src1 : tensor<?xf32>, tensor<?xf32>) outs(%dst : tensor<?xf32>) -> tensor<?xf32>
    ```
  }];
  let arguments = (ins Variadic<AnyType>:$src,
                       TensorOrMemref:$dst
  );
  let results = (outs Optional<AnyRankedTensor>:$result_tensor);
  let assemblyFormat = [{
    attr-dict
    `ins` `(` $src `:` type($src) `)`
    `outs` `(` $dst `:` type($dst) `)`
    (`->` type($result_tensor)^)?
  }];
}
```

### MLIR 语法

```mlir
hivm.hir.atomic_cas ins(%src0, %src1 : memref<?xf32>, memref<?xf32>)
                    outs(%dst : memref<?xf32>)

%result = hivm.hir.atomic_cas ins(%src0, %src1 : tensor<?xf32>, tensor<?xf32>)
                              outs(%dst : tensor<?xf32>) -> tensor<?xf32>
```

### 参数说明

| 参数 | 类型 | 必选 | 说明 | 约束 |
|------|------|------|------|------|
| `src` | Variadic<AnyType> | 是 | 源操作数（2 个） | src0=期望旧值, src1=新值 |
| `dst` | TensorOrMemref | 是 | 目标内存位置 | GM 地址空间 |

### 结果说明

| 结果 | 类型 | 说明 |
|------|------|------|
| `result_tensor` | Optional<AnyRankedTensor> | 返回 V 的原始值（Tensor 语义） |

### 数据类型约束

来源：`OperElemTypeConstraints<[0], [...]>`

| 支持的元素类型 | 说明 |
|---------------|------|
| `i8` | 8 位有符号整数 |
| `i16` | 16 位有符号整数 |
| `i32` | 32 位有符号整数 |
| `i64` | 64 位有符号整数 |
| `f16` | 半精度浮点 |
| `f32` | 单精度浮点 |

### IR 示例

#### Memref 语义

```mlir
func.func @test_atomic_cas_memref() {
  %src0 = memref.alloc() : memref<16xf32, #hivm.address_space<ub>>
  %src1 = memref.alloc() : memref<16xf32, #hivm.address_space<ub>>
  %dst = memref.alloc() : memref<16xf32, #hivm.address_space<gm>>
  hivm.hir.atomic_cas ins(%src0, %src1 : memref<16xf32, #hivm.address_space<ub>>, memref<16xf32, #hivm.address_space<ub>>)
                      outs(%dst : memref<16xf32, #hivm.address_space<gm>>)
  return
}
```

#### Tensor 语义

```mlir
func.func @test_atomic_cas_tensor() -> tensor<16xi32> {
  %src0 = tensor.empty() : tensor<16xi32>
  %src1 = tensor.empty() : tensor<16xi32>
  %dst = tensor.empty() : tensor<16xi32>
  %result = hivm.hir.atomic_cas ins(%src0, %src1 : tensor<16xi32>, tensor<16xi32>)
                                outs(%dst : tensor<16xi32>) -> tensor<16xi32>
  return %result : tensor<16xi32>
}
```

## hir.atomic_xchg

### 概述

原子交换（Exchange）操作。该操作读取内存位置的当前值，写入新值，并返回旧值。整个过程是原子的。

### TableGen 定义

来源：[HIVMDMAOps.td:L445-L481](https://gitcode.com/Ascend/AscendNPU-IR/tree/master/bishengir/include/bishengir/Dialect/HIVM/IR/HIVMDMAOps.td#L445-L481)

```tablegen
def AtomicXchgOp : HIVM_Op<"atomic_xchg",
    [NoLibraryFunctionTrait,
     OperElemTypeConstraints<[0], [I8, I16, I32, F16, F32, I64]>
    ]> {
  let summary = "Atomic Exchange Op";
  let description = [{
    Atomic exchange is an atomic operation that consists of three steps:
    1. Read the current value of the specified memory address
    2. Write the new value to the memory address
    3. Return the old value read previously
    The whole process is atomic, that is, it will not be interrupted by other threads during the operation.

    Constraints:
      1. The input memref and output memref must have the same rank
         and the same element type.

    Arguments:
      * `src`: new value
      * `dst`: memory location in GM

    Examples:
    ```mlir
    hivm.hir.atomic_xchg ins(%src : memref<?xf32>) outs(%dst : memref<?xf32>)
    %result = hivm.hir.atomic_xchg ins(%src : tensor<?xf32>) outs(%dst : tensor<?xf32>) -> tensor<?xf32>
    ```
  }];
  let arguments = (ins Variadic<AnyType>:$src,
                       TensorOrMemref:$dst
  );
  let results = (outs Optional<AnyRankedTensor>:$result_tensor);
  let assemblyFormat = [{
    attr-dict
    `ins` `(` $src `:` type($src) `)`
    `outs` `(` $dst `:` type($dst) `)`
    (`->` type($result_tensor)^)?
  }];
}
```

### MLIR 语法

```mlir
hivm.hir.atomic_xchg ins(%src : memref<?xf32>)
                     outs(%dst : memref<?xf32>)

%result = hivm.hir.atomic_xchg ins(%src : tensor<?xf32>)
                               outs(%dst : tensor<?xf32>) -> tensor<?xf32>
```

### 参数说明

| 参数 | 类型 | 必选 | 说明 | 约束 |
|------|------|------|------|------|
| `src` | Variadic<AnyType> | 是 | 源操作数（1 个） | 新值 |
| `dst` | TensorOrMemref | 是 | 目标内存位置 | GM 地址空间 |

### 结果说明

| 结果 | 类型 | 说明 |
|------|------|------|
| `result_tensor` | Optional<AnyRankedTensor> | 返回 V 的原始值（Tensor 语义） |

### 数据类型约束

与 `atomic_cas` 相同：`i8`, `i16`, `i32`, `i64`, `f16`, `f32`

### IR 示例

```mlir
func.func @test_atomic_xchg_memref() {
  %src = memref.alloc() : memref<16xf32, #hivm.address_space<ub>>
  %dst = memref.alloc() : memref<16xf32, #hivm.address_space<gm>>
  hivm.hir.atomic_xchg ins(%src : memref<16xf32, #hivm.address_space<ub>>)
                       outs(%dst : memref<16xf32, #hivm.address_space<gm>>)
  return
}
```

## IR 层约束与验证

### atomic_cas 约束

1. **操作数数量**：`src` 必须包含恰好 2 个操作数（期望旧值和新值）
2. **元素类型一致性**：输入和输出的元素类型必须相同
3. **Rank 一致性**：输入和输出的 rank 必须相同
4. **NoLibraryFunctionTrait**：不生成库函数调用，直接映射到硬件指令

### atomic_xchg 约束

1. **操作数数量**：`src` 必须包含恰好 1 个操作数（新值）
2. **元素类型一致性**：输入和输出的元素类型必须相同
3. **Rank 一致性**：输入和输出的 rank 必须相同
4. **NoLibraryFunctionTrait**：不生成库函数调用，直接映射到硬件指令

## 与 store 原子操作的区别

| 特性 | `hir.store` + atomic | `hir.atomic_cas` | `hir.atomic_xchg` |
|------|---------------------|-------------------|-------------------|
| 操作类型 | ADD/MAX/MIN/AND/OR/XOR | CAS | XCHG |
| 依赖 set_atomic | 是 | 否 | 否 |
| 返回旧值 | 否 | 是 | 是 |
| 实现方式 | 硬件/软件 | 硬件指令 | 硬件指令 |
| 数据类型 | I8~F8E5M2 | I8/I16/I32/I64/F16/F32 | I8/I16/I32/I64/F16/F32 |

## 与其他 IR 操作的关系

### 从 Triton 到 HIVM

```
tt.atomic_cas  -->  hivm.hir.atomic_cas
tt.atomic_xchg -->  hivm.hir.atomic_xchg
```

### 与 set_atomic 的关系

`hir.atomic_cas` 和 `hir.atomic_xchg` 是独立的原子操作，不需要通过 `hir.set_atomic` 设置全局原子模式。它们直接映射到硬件的原子指令。

## 常见问题

### Q: atomic_cas 和 store 的 CAS 原子模式有什么区别？

A: `hir.store` 的 CAS 模式需要先通过 `hir.set_atomic` 设置全局原子模式，且 CAS 属于软件原子实现。`hir.atomic_cas` 是独立的操作，直接映射到硬件指令，性能更高。

### Q: atomic_cas 的返回值有什么用？

A: `atomic_cas` 返回内存位置的原始值，可以用于判断 CAS 是否成功（如果返回值等于期望旧值，说明更新成功；否则说明有并发修改）。

### Q: 为什么 atomic_cas/xchg 不支持 BF16 和 F8 类型？

A: 这是硬件约束。原子操作的数据类型支持由硬件决定，当前 Ascend NPU 的原子指令不支持 BF16 和 F8 类型。

## 相关文档

- Python API：[docs_triton_ascend/02-Core-API/01-memory-ops.md](../../../docs_triton_ascend/02-Core-API/01-memory-ops.md)
- DMA 操作总览：[00-overview.md](00-overview.md)
- hir.store：[02-store.md](02-store.md)
- 源码参考：
  - [HIVMDMAOps.td](https://gitcode.com/Ascend/AscendNPU-IR/tree/master/bishengir/include/bishengir/Dialect/HIVM/IR/HIVMDMAOps.td) - TableGen 定义
  - [HIVMAttrs.td](https://gitcode.com/Ascend/AscendNPU-IR/tree/master/bishengir/include/bishengir/Dialect/HIVM/IR/HIVMAttrs.td) - AtomicKind 枚举
