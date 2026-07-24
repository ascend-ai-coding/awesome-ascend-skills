# hir.gather_load / hir.scatter_store

> 关键词：Gather、Scatter、稀疏内存访问、间接索引、Mask

## 概述

`hir.gather_load` 和 `hir.scatter_store` 是 HIVM 方言中的稀疏内存访问操作，用于按索引张量从全局内存中收集数据或向全局内存散列写数据。这两个操作支持掩码（Mask）机制，可以条件化地控制哪些元素被加载或存储。

与 `hir.load`/`hir.store` 的连续内存访问不同，Gather/Scatter 操作支持非连续的、按索引的内存访问模式，适用于稀疏数据、嵌入查找等场景。

> Python API 对应：tl.load (带 index) / tl.store (带 index) -- 详见 [docs_triton_ascend/02-Core-API/01-memory-ops.md](../../../docs_triton_ascend/02-Core-API/01-memory-ops.md)

## hir.gather_load

### 概述

稀疏内存加载操作。从源内存缓冲区中按索引张量指定的偏移位置收集元素，生成输出张量。支持掩码和回退值。

### TableGen 定义

来源：[HIVMOps.td:L389-L426](https://gitcode.com/Ascend/AscendNPU-IR/tree/master/bishengir/include/bishengir/Dialect/HIVM/IR/HIVMOps.td#L389-L426)

```tablegen
def GatherLoadOp : HIVM_Op<"gather_load", [
    DeclareOpInterfaceMethods<MemoryEffectsOpInterface>,
    SameVariadicOperandSize,
]> {
  let summary = [{
    Perform sparse memory loading with optional mask and other
  }];

  let description = [{
    This operation takes a source memory buffer and a tensor of offsets,
    and produces an output tensor by gathering elements from the source
    at the specified offset positions. The operation supports masking
    and provides fallback values for masked-out positions.
  }];
  let arguments = (ins AnyMemRef:$base,
                       RankedTensorOf<[I32, I64]>:$indices,
                       AnyTypeOf<[I32, I64]>:$burst_len,
                       Optional<RankedTensorOf<[I1]>>:$mask,
                       Optional<AnyTypeOf<[AnyInteger, AnyFloat]>>:$other,

                       OptionalAttr<DenseI32ArrayAttr>:$boundaryCheck,
                       OptionalAttr<HIVM_PaddingOptionAttr>:$padding,
                       OptionalAttr<HIVM_CacheModifierAttr>:$cache,
                       OptionalAttr<HIVM_EvictionPolicyAttr>:$evict,
                       OptionalAttr<BoolAttr>:$isVolatile
  );
  let results = (outs AnyRankedTensor:$result);

  let assemblyFormat = [{
    `ins` `(` $base `:` type($base) `,` $indices `:` type($indices) `,`
    $burst_len `:` type($burst_len)
    (`,` $mask `:` type($mask)^)? (`,` $other `:` type($other)^)? `)`
    attr-dict
    `->` type($result)
  }];

  let hasVerifier = 1;
}
```

### MLIR 语法

```mlir
%result = hivm.hir.gather_load ins(%base : memref<?xf32>, %indices : tensor<16xi32>, %burst_len : i32)
                               -> tensor<16xf32>

%result = hivm.hir.gather_load ins(%base : memref<?xf32>, %indices : tensor<16xi32>, %burst_len : i32, %mask : tensor<16xi1>, %other : f32)
                               -> tensor<16xf32>
```

### 参数说明

| 参数 | 类型 | 必选 | 说明 | 约束 |
|------|------|------|------|------|
| `base` | AnyMemRef | 是 | 源内存缓冲区基地址 | GM 地址空间 |
| `indices` | RankedTensorOf<[I32, I64]> | 是 | 偏移索引张量 | I32 或 I64 |
| `burst_len` | AnyTypeOf<[I32, I64]> | 是 | 突发传输长度 | I32 或 I64 |
| `mask` | RankedTensorOf<[I1]> | 否 | 掩码张量 | 布尔类型 |
| `other` | AnyTypeOf<[AnyInteger, AnyFloat]> | 否 | 掩码位置的回退值 | 与结果元素类型相同 |

### 结果说明

| 结果 | 类型 | 说明 |
|------|------|------|
| `result` | AnyRankedTensor | 收集到的输出张量 |

### 属性说明

| 属性 | 类型 | 默认值 | 说明 | 可选值 |
|------|------|--------|------|--------|
| `boundaryCheck` | DenseI32ArrayAttr | 无 | 边界检查维度 | 维度索引数组 |
| `padding` | HIVM_PaddingOptionAttr | 无 | 填充选项 | `PAD_ZERO`(1) / `PAD_NAN`(2) |
| `cache` | HIVM_CacheModifierAttr | 无 | 缓存修改器 | NONE/CA/CG/WB/CS/WT/CV |
| `evict` | HIVM_EvictionPolicyAttr | 无 | 驱逐策略 | EvictFirst / EvictLast |
| `isVolatile` | BoolAttr | 无 | 是否为 volatile 访问 | true / false |

### PaddingOption 枚举

定义于 [HIVMAttrs.td:L920-L928](https://gitcode.com/Ascend/AscendNPU-IR/tree/master/bishengir/include/bishengir/Dialect/HIVM/IR/HIVMAttrs.td#L920-L928)

| 枚举值 | 数值 | IR 字面量 | 说明 |
|--------|------|----------|------|
| `PAD_ZERO` | 1 | `zero` | 用零填充 |
| `PAD_NAN` | 2 | `nan` | 用 NaN 填充 |

### CacheModifier 枚举

定义于 [HIVMAttrs.td:L930-L942](https://gitcode.com/Ascend/AscendNPU-IR/tree/master/bishengir/include/bishengir/Dialect/HIVM/IR/HIVMAttrs.td#L930-L942)

| 枚举值 | 数值 | IR 字面量 | 说明 |
|--------|------|----------|------|
| `NONE` | 1 | `none` | 无缓存修改 |
| `CA` | 2 | `ca` | Cache All |
| `CG` | 3 | `cg` | Cache Global |
| `WB` | 4 | `wb` | Write Back |
| `CS` | 5 | `cs` | Cache Streaming |
| `WT` | 6 | `wt` | Write Through |
| `CV` | 7 | `cv` | Cache Volatile |

### IR 示例

#### 基础 Gather Load

```mlir
func.func @test_gather_load(%base: memref<?xf32>, %indices: tensor<16xi32>, %burst_len: i32) -> tensor<16xf32> {
  %result = hivm.hir.gather_load ins(%base : memref<?xf32>, %indices : tensor<16xi32>, %burst_len : i32)
                                 -> tensor<16xf32>
  return %result : tensor<16xf32>
}
```

#### 带 Mask 和 Other 的 Gather Load

```mlir
func.func @test_gather_load_masked(%base: memref<?xf32>, %indices: tensor<16xi32>, %burst_len: i32, %mask: tensor<16xi1>, %other: f32) -> tensor<16xf32> {
  %result = hivm.hir.gather_load ins(%base : memref<?xf32>, %indices : tensor<16xi32>, %burst_len : i32, %mask : tensor<16xi1>, %other : f32)
                                 -> tensor<16xf32>
  return %result : tensor<16xf32>
}
```

## hir.scatter_store

### 概述

稀疏内存存储操作。将源张量中的元素按索引张量指定的偏移位置写入目标 GM 缓冲区。支持掩码机制。

### TableGen 定义

来源：[HIVMOps.td:L459-L492](https://gitcode.com/Ascend/AscendNPU-IR/tree/master/bishengir/include/bishengir/Dialect/HIVM/IR/HIVMOps.td#L459-L492)

```tablegen
def ScatterStoreOp : HIVM_Op<"scatter_store", [
    DeclareOpInterfaceMethods<MemoryEffectsOpInterface>,
    SameVariadicOperandSize,
]> {
  let summary = [{
    Performs sparse memory storing with optional mask
  }];
 
  let description = [{
    This operation takes a source tensor and a tensor of offsets from UB,
    and stores elements from the source into the destination GM buffer
    at the specified offset positions. The operation 
    supports masking to conditionally control which elements are stored.
  }];
  let arguments = (ins AnyMemRef:$base,
                       RankedTensorOf<[I32, I64]>:$indices,
                       AnyRankedTensor:$data,
                       AnyTypeOf<[I32, I64]>:$burst_len,
                       Optional<RankedTensorOf<[I1]>>:$mask,
 
                       OptionalAttr<DenseI32ArrayAttr>:$boundaryCheck,
                       OptionalAttr<HIVM_CacheModifierAttr>:$cache,
                       OptionalAttr<HIVM_EvictionPolicyAttr>:$evict
  );
 
  let assemblyFormat = [{
    `ins` `(` $base `:` type($base) `,` $indices `:` type($indices) `,`
    $data `:` type($data) `,` $burst_len `:` type($burst_len)
    (`,` $mask `:` type($mask)^)? `)`
    attr-dict
  }];
 
  let hasVerifier = 1;
}
```

### MLIR 语法

```mlir
hivm.hir.scatter_store ins(%base : memref<?xf32>, %indices : tensor<16xi32>, %data : tensor<16xf32>, %burst_len : i32)

hivm.hir.scatter_store ins(%base : memref<?xf32>, %indices : tensor<16xi32>, %data : tensor<16xf32>, %burst_len : i32, %mask : tensor<16xi1>)
```

### 参数说明

| 参数 | 类型 | 必选 | 说明 | 约束 |
|------|------|------|------|------|
| `base` | AnyMemRef | 是 | 目标内存缓冲区基地址 | GM 地址空间 |
| `indices` | RankedTensorOf<[I32, I64]> | 是 | 偏移索引张量 | I32 或 I64 |
| `data` | AnyRankedTensor | 是 | 源数据张量 | UB 中的数据 |
| `burst_len` | AnyTypeOf<[I32, I64]> | 是 | 突发传输长度 | I32 或 I64 |
| `mask` | RankedTensorOf<[I1]> | 否 | 掩码张量 | 布尔类型 |

### 属性说明

| 属性 | 类型 | 默认值 | 说明 | 可选值 |
|------|------|--------|------|--------|
| `boundaryCheck` | DenseI32ArrayAttr | 无 | 边界检查维度 | 维度索引数组 |
| `cache` | HIVM_CacheModifierAttr | 无 | 缓存修改器 | 同 gather_load |
| `evict` | HIVM_EvictionPolicyAttr | 无 | 驱逐策略 | EvictFirst / EvictLast |

### IR 示例

#### 基础 Scatter Store

```mlir
func.func @test_scatter_store(%base: memref<?xf32>, %indices: tensor<16xi32>, %data: tensor<16xf32>, %burst_len: i32) {
  hivm.hir.scatter_store ins(%base : memref<?xf32>, %indices : tensor<16xi32>, %data : tensor<16xf32>, %burst_len : i32)
  return
}
```

#### 带 Mask 的 Scatter Store

```mlir
func.func @test_scatter_store_masked(%base: memref<?xf32>, %indices: tensor<16xi32>, %data: tensor<16xf32>, %burst_len: i32, %mask: tensor<16xi1>) {
  hivm.hir.scatter_store ins(%base : memref<?xf32>, %indices : tensor<16xi32>, %data : tensor<16xf32>, %burst_len : i32, %mask : tensor<16xi1>)
  return
}
```

## IR 层约束与验证

### gather_load 约束

1. **索引类型**：`indices` 必须为 I32 或 I64 类型的张量
2. **掩码类型**：`mask` 必须为 I1 类型的张量
3. **回退值类型**：`other` 的类型必须与输出张量的元素类型相同
4. **SameVariadicOperandSize**：可变操作数的大小必须一致

### scatter_store 约束

1. **索引类型**：`indices` 必须为 I32 或 I64 类型的张量
2. **掩码类型**：`mask` 必须为 I1 类型的张量
3. **SameVariadicOperandSize**：可变操作数的大小必须一致

## 与其他 IR 操作的关系

### Gather/Scatter vs Load/Store

| 特性 | hir.load / hir.store | hir.gather_load / hir.scatter_store |
|------|---------------------|-------------------------------------|
| 访问模式 | 连续 | 按索引非连续 |
| 索引方式 | 隐式（连续偏移） | 显式（索引张量） |
| 掩码支持 | 通过 Padding | 通过 Mask 张量 |
| Pipeline | MTE2/MTE3 | 无固定 Pipeline |
| 适用场景 | 密集张量操作 | 稀疏数据、嵌入查找 |

### 与 indirect_load/indirect_store 的区别

| 特性 | gather_load / scatter_store | indirect_load / indirect_store |
|------|---------------------------|-------------------------------|
| 数据通路 | GM <-> UB | GM <-> UB |
| 索引方式 | 偏移量 | 偏移量 |
| Pipeline | 无固定 | PIPE_V |
| 接口风格 | ins/outs 分离 | ins/outs 统一 |
| 掩码 | 可选 mask 张量 | 可选 mask 张量 |

## 常见问题

### Q: gather_load 和 indirect_load 有什么区别？

A: `gather_load` 是更通用的稀疏加载操作，支持缓存修改器和边界检查等属性。`indirect_load` 是更结构化的间接加载操作，使用 Destination Style 接口，属于 Vector Pipeline。在编译流程中，`gather_load` 可能被分解为 `indirect_load`。

### Q: burst_len 参数的作用是什么？

A: `burst_len` 指定每次内存访问的突发传输长度，影响 DMA 引擎的传输效率。较大的突发长度可以提高带宽利用率，但需要更多的缓冲区空间。

### Q: 什么时候应该使用 gather/scatter 而不是 load/store？

A: 当数据访问模式是非连续的（如按索引访问数组元素、嵌入表查找）时，应使用 gather/scatter。当数据访问是连续的时，使用 load/store 更高效。

## 相关文档

- Python API：[docs_triton_ascend/02-Core-API/01-memory-ops.md](../../../docs_triton_ascend/02-Core-API/01-memory-ops.md)
- DMA 操作总览：[00-overview.md](00-overview.md)
- 间接访问：[09-indirect-access.md](09-indirect-access.md)
- 源码参考：
  - [HIVMOps.td](https://gitcode.com/Ascend/AscendNPU-IR/tree/master/bishengir/include/bishengir/Dialect/HIVM/IR/HIVMOps.td) - TableGen 定义
  - [HIVMAttrs.td](https://gitcode.com/Ascend/AscendNPU-IR/tree/master/bishengir/include/bishengir/Dialect/HIVM/IR/HIVMAttrs.td) - 属性枚举
