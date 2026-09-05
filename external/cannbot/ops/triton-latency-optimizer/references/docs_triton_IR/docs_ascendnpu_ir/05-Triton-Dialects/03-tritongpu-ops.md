# TritonGPU 操作详解

本文档详细描述 TritonGPU 方言（`ttg`）中所有操作的定义、签名和语义。所有操作签名均从 TableGen 源码精确提取。

源码参考：[TritonGPUOps.td](https://gitcode.com/Ascend/AscendNPU-IR/tree/master/triton/include/triton/Dialect/TritonGPU/IR/TritonGPUOps.td)

## 1. 布局转换操作

### 1.1 ttg.convert_layout

将张量从一种布局编码转换为另一种。这是 TritonGPU 中最核心的操作之一，实际的数据移动（如共享内存读写）在此操作中发生。

| 项目 | 内容 |
|------|------|
| 操作名 | `ttg.convert_layout` |
| 输入 | `$src`: `TT_Tensor` |
| 输出 | `$result`: `TT_Tensor` |
| Traits | `SameOperandsAndResultShape`, `SameOperandsAndResultElementType`, `Pure` |

```mlir
%result = ttg.convert_layout %src : tensor<128xf32, #blocked> -> tensor<128xf32, #dot_op>
```

## 2. 异步拷贝操作

### 2.1 ttg.async_copy_global_to_local

将数据从全局内存异步拷贝到共享内存（local memory）。类似于 `tt.load`，但数据写入共享内存描述符而非分布式张量。

| 项目 | 内容 |
|------|------|
| 操作名 | `ttg.async_copy_global_to_local` |
| 输入 | `$src`: `TT_PtrTensor` (MemRead<GlobalMemory>), `$result`: `TTG_MemDescType` (MemWrite<SharedMemory>), `$mask`: `I1Tensor` (可选), `$other`: `TT_Type` (可选) |
| 属性 | `$cache`: `TT_CacheModifierAttr` (默认 `NONE`), `$evict`: `TT_EvictionPolicyAttr` (默认 `NORMAL`), `$isVolatile`: `BoolAttr` (默认 `false`) |
| 输出 | `$token`: `TTG_AsyncToken` |
| Traits | `AttrSizedOperandSegments` |

支持的加载字节数（按计算能力）：

| 计算能力 | 有效加载字节数 |
|----------|---------------|
| >= 80 | 4, 8, 16 |

```mlir
%token = ttg.async_copy_global_to_local %src, %dst mask %mask other %other cacheModifier = #tt<cache ca> : tensor<128x!tt.ptr<f32>> -> !ttg.memdesc<128xf32, #shared>
```

### 2.2 ttg.async_wait

等待异步拷贝操作完成。

| 项目 | 内容 |
|------|------|
| 操作名 | `ttg.async_wait` |
| 输入 | `$asyncToken`: `Variadic<TTG_AsyncToken>` |
| 属性 | `$num`: `I32Attr` (等待的组数) |
| 输出 | `$retToken`: `TTG_AsyncToken` |

计算能力要求：>= 80

```mlir
%token2 = ttg.async_wait %token1 {num = 0 : i32}
```

### 2.3 ttg.async_commit_group

将当前未提交的异步操作标记为一个组。

| 项目 | 内容 |
|------|------|
| 操作名 | `ttg.async_commit_group` |
| 输入 | `$inputTokens`: `Variadic<TTG_AsyncToken>` |
| 输出 | `$asyncToken`: `TTG_AsyncToken` |

计算能力要求：>= 80

```mlir
%token = ttg.async_commit_group tokens %token1, %token2
```

## 3. 共享内存操作

### 3.1 ttg.local_alloc

在共享内存中分配缓冲区，返回内存描述符。

| 项目 | 内容 |
|------|------|
| 操作名 | `ttg.local_alloc` |
| 输入 | `$src`: `TT_Tensor` (可选，初始化值) |
| 属性 | `$alignment`: `I32Attr` (可选) |
| 输出 | `$result`: `TTG_MemDescType` |

构建器：

| 构建器 | 参数 |
|--------|------|
| 基础 | `(Type result)` |
| 带初始化 | `(Type result, Value src)` |
| 带对齐 | `(Type result, Value src, int32_t alignment)` |

额外方法：
- `isSharedMemoryAlloc()`：检查是否为共享内存分配
- `getAlignmentOrDefault()`：获取对齐值或默认值

```mlir
%buf = ttg.local_alloc %src : () -> !ttg.memdesc<128xf32, #shared>
%buf2 = ttg.local_alloc : () -> !ttg.memdesc<128xf32, #shared>
```

### 3.2 ttg.local_dealloc

释放共享内存缓冲区。可选操作，未显式释放的缓冲区在所有使用后的第一个后支配点被隐式释放。

| 项目 | 内容 |
|------|------|
| 操作名 | `ttg.local_dealloc` |
| 输入 | `$src`: `TTG_MemDescType` (MemFree<SharedMemory>) |

```mlir
ttg.local_dealloc %buf : !ttg.memdesc<128xf32, #shared>
```

### 3.3 ttg.local_load

从共享内存描述符加载到分布式张量。

| 项目 | 内容 |
|------|------|
| 操作名 | `ttg.local_load` |
| 输入 | `$src`: `TTG_MemDescType` (MemRead<SharedMemory>), `$token`: `TTG_AsyncToken` (可选) |
| 输出 | `$result`: `TT_Tensor` |
| Traits | `LocalLoadTrait` |

```mlir
%result = ttg.local_load %buf token %async_token : !ttg.memdesc<128xf32, #shared> -> tensor<128xf32, #blocked>
```

### 3.4 ttg.local_store

将分布式张量存储到共享内存。

| 项目 | 内容 |
|------|------|
| 操作名 | `ttg.local_store` |
| 输入 | `$src`: `TT_Tensor`, `$dst`: `TTG_MemDescType` (MemWrite<SharedMemory>) |

```mlir
ttg.local_store %tensor, %buf : tensor<128xf32, #blocked> -> !ttg.memdesc<128xf32, #shared>
```

## 4. 内存描述符视图操作

### 4.1 ttg.memdesc_index

取内存描述符沿第 0 维第 i 个元素的子视图。不影响底层内存。

| 项目 | 内容 |
|------|------|
| 操作名 | `ttg.memdesc_index` |
| 输入 | `$src`: `TTG_MemDescType`, `$index`: `I32` |
| 输出 | `$result`: `TTG_MemDescType` |
| Traits | `Pure`, `MemDescViewTrait` |

例如：输入 `2x4x16xf16`，index=1 → 输出 `4x16xf16`

```mlir
%sub = ttg.memdesc_index %src[%index] : !ttg.memdesc<2x4x16xf16, #shared> -> !ttg.memdesc<4x16xf16, #shared>
```

### 4.2 ttg.memdesc_subslice

取内存描述符的子视图，指定各维偏移。不影响底层内存。

| 项目 | 内容 |
|------|------|
| 操作名 | `ttg.memdesc_subslice` |
| 输入 | `$src`: `TTG_MemDescType` |
| 属性 | `$offsets`: `DenseI32ArrayAttr` |
| 输出 | `$result`: `TTG_MemDescType` |
| Traits | `Pure`, `MemDescViewTrait` |

例如：输入 `32x16xf16`，offsets=[2,1] → 输出 `8x16xf16`，覆盖 `input[2:10, 1:17]`

```mlir
%sub = ttg.memdesc_subslice %src[0, 0] : !ttg.memdesc<32x16xf16, #shared> -> !ttg.memdesc<8x16xf16, #shared>
```

### 4.3 ttg.memdesc_trans

转置内存描述符的视图。不影响底层内存。

| 项目 | 内容 |
|------|------|
| 操作名 | `ttg.memdesc_trans` |
| 输入 | `$src`: `TTG_MemDescType` |
| 属性 | `$order`: `DenseI32ArrayAttr` |
| 输出 | `$result`: `TTG_MemDescType` |
| Traits | `Pure`, `MemDescViewTrait`, `TransposeOpInterface`, `InferTypeOpWithLayoutEquivalence`, `SameOperandsAndResultElementType` |

```mlir
%transposed = ttg.memdesc_trans %src {order = [1, 0]} : !ttg.memdesc<8x16xf16, #shared> -> !ttg.memdesc<16x8xf16, #shared>
```

### 4.4 ttg.memdesc_reshape

创建不同形状的内存描述符视图。不影响底层内存。

| 项目 | 内容 |
|------|------|
| 操作名 | `ttg.memdesc_reshape` |
| 输入 | `$src`: `TTG_MemDescType` |
| 输出 | `$result`: `TTG_MemDescType` |
| Traits | `Pure`, `MemDescViewTrait`, `SameOperandsAndResultElementType` |

```mlir
%reshaped = ttg.memdesc_reshape %src : !ttg.memdesc<128xf16, #shared> -> !ttg.memdesc<8x16xf16, #shared>
```

### 4.5 ttg.memdesc_reinterpret

将内存描述符重新解释为不同类型和形状。要求原始描述符是连续的。

| 项目 | 内容 |
|------|------|
| 操作名 | `ttg.memdesc_reinterpret` |
| 输入 | `$src`: `TTG_MemDescType` |
| 输出 | `$result`: `TTG_MemDescType` |
| Traits | `Pure`, `MemDescViewTrait` |

```mlir
%reinterpreted = ttg.memdesc_reinterpret %src : !ttg.memdesc<128xf16, #shared> -> !ttg.memdesc<64xf32, #shared>
```

## 5. 流水线操作

### 5.1 ttg.predicate_stage

流水线阶段谓词，用于软件流水线。

| 项目 | 内容 |
|------|------|
| 操作名 | `ttg.predicate_stage` |
| 输入 | `$iv`: `AnySignlessIntegerOrIndex`, `$ub`: `AnySignlessIntegerOrIndex`, `$step`: `AnySignlessIntegerOrIndex` |
| 属性 | `$maxStage`: `I32Attr`, `$stage`: `I32Attr` |
| 输出 | `$result`: `I1` |
| Traits | `Pure`, `AllTypesMatch<["iv", "ub", "step"]>` |

```mlir
%pred = ttg.predicate_stage %iv, %ub, %step maxStage 2 stage 0 : index -> i1
```

### 5.2 ttg.mask

流水线掩码操作，包含一个区域。

| 项目 | 内容 |
|------|------|
| 操作名 | `ttg.mask` |
| 输入 | `$pred`: `I1` |
| 输出 | `$result`: `Variadic<AnyType>` |
| 区域 | `$region`: `SizedRegion<1>` |
| Traits | `SingleBlock` |

### 5.3 ttg.mask.return

`ttg.mask` 区域的终止操作。

| 项目 | 内容 |
|------|------|
| 操作名 | `ttg.mask.return` |
| 输入 | `$result`: `Variadic<AnyType>` |
| Traits | `HasParent<"MaskOp">`, `Pure`, `Terminator`, `ReturnLike` |

## 6. 类型转换操作

### 6.1 ttg.fp4_to_fp

将打包为 i8 的 FP4 (E2M1) 数据上转换为浮点类型。每个 i8 的低 4 位表示第一个 FP4 元素，高 4 位表示第二个。

| 项目 | 内容 |
|------|------|
| 操作名 | `ttg.fp4_to_fp` |
| 输入 | `$src`: `RankedTensorOf<[I8]>` |
| 属性 | `$axis`: `I32Attr` (FP4 元素打包的轴) |
| 输出 | `$result`: `TT_FloatTensor` |
| Traits | `Pure` |
| 验证器 | `hasVerifier = 1` |

```mlir
%result = ttg.fp4_to_fp %src {axis = 1 : i32} : tensor<16x8xi8> -> tensor<16x16xf32>
```

## 7. 全局内存操作

### 7.1 ttg.global_scratch_alloc

在全局内存中分配当前程序私有的缓冲区。

| 项目 | 内容 |
|------|------|
| 操作名 | `ttg.global_scratch_alloc` |
| 属性 | `$nbytes`: `I32Attr`, `$alignment`: `I32Attr` |
| 输出 | `$result`: `TT_Ptr` (MemAlloc<GlobalMemory>) |

```mlir
%ptr = ttg.global_scratch_alloc {nbytes = 1024 : i32, alignment = 16 : i32} : !tt.ptr<i8>
```

## 8. Warp 特化操作

### 8.1 ttg.warp_specialize

在不同 warp 组上异步执行不同代码。默认区域可隐式捕获，分区区域与上方隔离。

| 项目 | 内容 |
|------|------|
| 操作名 | `ttg.warp_specialize` |
| 输入 | `$explicitCaptures`: `Variadic<AnyType>` |
| 属性 | `$partitionNumWarps`: `DenseI32ArrayAttr`, `$warpGroupStartIds`: `DenseI32ArrayAttr` (可选), `$requestedRegisters`: `DenseI32ArrayAttr` (可选), `$actualRegisters`: `DenseI32ArrayAttr` (可选) |
| 输出 | `$defaultPassthrough`: `Variadic<AnyType>` |
| 区域 | `$defaultRegion`: `MinSizedRegion<1>`, `$partitionOpHolder`: `SizedRegion<1>` |
| Traits | `RecursiveMemoryEffects`, `RecursivelySpeculatable`, `AsyncRegions`, `RegionBranchOpInterface` |

额外方法：
- `getPartitionRegions()`：获取分区区域
- `getCaptureSizeAlign()`：获取捕获列表的大小和对齐
- `getTotalPartitionWarps()`：获取额外 warp 总数

```mlir
%0 = ttg.warp_specialize(%a, %b)
  default {
    %out = some_operation(%a)
    ttg.warp_yield %out : i32
  }
  partition0(%arg0: i32, %arg1: i32) num_warps(8) {
    some_async_dispatch(%arg0, %arg1)
    ttg.warp_return
  }
: (i32, i32) -> i32
```

### 8.2 ttg.warp_specialize.partitions

包含 `ttg.warp_specialize` 的隔离分区区域的容器操作。

| 项目 | 内容 |
|------|------|
| 操作名 | `ttg.warp_specialize.partitions` |
| 区域 | `$partitionRegions`: `VariadicRegion<MinSizedRegion<1>>` |
| Traits | `IsolatedFromAbove`, `RecursiveMemoryEffects`, `RecursivelySpeculatable`, `Terminator`, `HasParent<"WarpSpecializeOp">` |

### 8.3 ttg.warp_yield

`ttg.warp_specialize` 默认区域的终止操作。操作数作为 `ttg.warp_specialize` 的 SSA 结果传递。

| 项目 | 内容 |
|------|------|
| 操作名 | `ttg.warp_yield` |
| 输入 | `$values`: `Variadic<AnyType>` |
| Traits | `Pure`, `Terminator`, `ReturnLike`, `HasParent<"WarpSpecializeOp">`, `RegionBranchTerminatorOpInterface` |

```mlir
ttg.warp_yield %a, %b : i32, tensor<32xbf16, #blocked>
```

### 8.4 ttg.warp_return

`ttg.warp_specialize` 分区区域的隐式终止操作。无操作数，因为分区区域不能返回任何值。

| 项目 | 内容 |
|------|------|
| 操作名 | `ttg.warp_return` |
| Traits | `Pure`, `Terminator`, `ReturnLike`, `HasParent<"WarpSpecializePartitionsOp">` |

## 9. 操作分类速查表

| 类别 | 操作 | 关键字 |
|------|------|--------|
| 布局转换 | `ttg.convert_layout` | 数据移动 |
| 异步拷贝 | `ttg.async_copy_global_to_local` | 全局→共享 |
| 异步同步 | `ttg.async_wait`, `ttg.async_commit_group` | 异步令牌 |
| 共享内存分配 | `ttg.local_alloc`, `ttg.local_dealloc` | 缓冲区管理 |
| 共享内存读写 | `ttg.local_load`, `ttg.local_store` | 共享↔寄存器 |
| 内存描述符视图 | `ttg.memdesc_index`, `ttg.memdesc_subslice`, `ttg.memdesc_trans`, `ttg.memdesc_reshape`, `ttg.memdesc_reinterpret` | 子视图操作 |
| 流水线 | `ttg.predicate_stage`, `ttg.mask`, `ttg.mask.return` | 软件流水线 |
| 类型转换 | `ttg.fp4_to_fp` | FP4→FP |
| 全局内存 | `ttg.global_scratch_alloc` | Scratch 分配 |
| Warp 特化 | `ttg.warp_specialize`, `ttg.warp_specialize.partitions`, `ttg.warp_yield`, `ttg.warp_return` | 多 warp 组 |
