# 内存操作

## 1. 概述

HFusion 方言提供丰富的内存操作，包括稀疏加载/存储、间接访存、聚集/散射和原子操作。这些操作支持 GM（Global Memory）与 UB（Unified Buffer）之间的数据搬移，以及原子同步语义。

> 源码参考：[HFusionOps.td](https://gitcode.com/Ascend/AscendNPU-IR/tree/master/bishengir/include/bishengir/Dialect/HFusion/IR/HFusionOps.td#L329-L928)

## 2. gather_load

### 2.1 功能

从源内存缓冲区按偏移量聚集加载元素到输出张量，支持掩码和回退值。

### 2.2 操作签名

| 操作数 | 类型 | 说明 |
|--------|------|------|
| `base` | `AnyMemRef` | 源内存缓冲区 |
| `indices` | `RankedTensorOf<[I32, I64]>` | 偏移量张量 |
| `burst_len` | `AnyTypeOf<[I32, I64]>` | 突发长度 |
| `mask` | `Optional<RankedTensorOf<[I1]>>` | 掩码 |
| `other` | `Optional<AnyTypeOf<[AnyInteger, AnyFloat]>>` | 掩码位置的回退值 |
| `boundaryCheck` | `OptionalAttr<DenseI32ArrayAttr>` | 边界检查 |
| `padding` | `OptionalAttr<HFusion_PaddingOptionAttr>` | 填充选项 |
| `cache` | `OptionalAttr<HFusion_CacheModifierAttr>` | 缓存修饰符 |
| `evict` | `OptionalAttr<HFusion_EvictionPolicyAttr>` | 驱逐策略 |
| `isVolatile` | `OptionalAttr<BoolAttr>` | 是否 volatile |
| `result` | `AnyRankedTensor` | 输出张量 |

### 2.3 MLIR 示例

```mlir
%result = hfusion.gather_load
  ins(%base, %indices, %burst_len, %mask, %other :
    memref<?xf32>, tensor<128xi32>, i32, tensor<128xi1>, f32)
  -> tensor<128xf32>
```

## 3. scatter_store

### 3.1 功能

将源张量中的元素按偏移量散射存储到目标内存缓冲区，支持掩码。

### 3.2 操作签名

| 操作数 | 类型 | 说明 |
|--------|------|------|
| `base` | `AnyMemRef` | 目标内存缓冲区 |
| `indices` | `RankedTensorOf<[I32, I64]>` | 偏移量张量 |
| `data` | `AnyRankedTensor` | 源数据张量 |
| `burst_len` | `AnyTypeOf<[I32, I64]>` | 突发长度 |
| `mask` | `Optional<RankedTensorOf<[I1]>>` | 掩码 |
| `boundaryCheck` | `OptionalAttr<DenseI32ArrayAttr>` | 边界检查 |
| `cache` | `OptionalAttr<HFusion_CacheModifierAttr>` | 缓存修饰符 |
| `evict` | `OptionalAttr<HFusion_EvictionPolicyAttr>` | 驱逐策略 |

### 3.3 MLIR 示例

```mlir
hfusion.scatter_store
  ins(%base, %indices, %data, %burst_len, %mask :
    memref<?xf32>, tensor<128xi32>, tensor<128xf32>, i32, tensor<128xi1>)
```

## 4. indirect_load

### 4.1 功能

间接内存加载，支持 1D-5D 掩码和回退值。

### 4.2 操作签名

| 操作数 | 类型 | 说明 |
|--------|------|------|
| `src` | `AnyMemRef` | 源内存缓冲区 |
| `offsets` | `TensorOf<[I32, I64]>` | 偏移量张量 |
| `dst` | `AnyRankedTensor` | 目标张量（决定输出形状和类型） |
| `mask` | `TensorOf<[I1, I8]>` | 掩码 |
| `other` | `TensorOf<[AnyInteger, AnyFloat]>` | 回退值 |
| `result` | `Optional<AnyRankedTensor>` | 输出 |

### 4.3 语义

```
dst[i] = mask[i] ? src[offsets[i]] : other[i]
```

## 5. indirect_store

### 5.1 功能

间接内存存储，支持 1D-5D 掩码，使用 SIMT 模板。

### 5.2 操作签名

| 操作数 | 类型 | 说明 |
|--------|------|------|
| `dst` | `AnyMemRef` | 目标内存缓冲区 |
| `offsets` | `TensorOf<[I32, I64]>` | 偏移量张量 |
| `src` | `AnyRankedTensor` | 源数据张量 |
| `mask` | `Optional<TensorOf<[I1, I8]>>` | 掩码 |

### 5.3 语义

```
if (mask[i]) dst[offsets[i]] = src[i]
```

## 6. gatherT

### 6.1 功能

SIMT 模式下的 Gather 操作，沿指定轴从源 GM 缓冲区按索引张量聚集元素。

### 6.2 操作签名

| 操作数 | 类型 | 说明 |
|--------|------|------|
| `src` | `AnyMemRef` | 源 GM 缓冲区 |
| `index` | `AnyRankedTensor` | 索引张量 |
| `dst` | `AnyRankedTensor` | 目标张量 |
| `bound` | `AnyTypeOf<[I32, I64]>` | gather 维大小 |
| `dim` | `AnyTypeOf<[I32, I64]>` | gather 维度 |
| `src_stride` | `Variadic<AnyTypeOf<[I32, I64]>>` | 源张量步长 |
| `index_shape` | `Variadic<AnyTypeOf<[I32, I64]>>` | 索引张量形状 |
| `offsets` | `Variadic<AnyTypeOf<[I32, I64]>>` | 偏移量 |

## 7. index_put

### 7.1 功能

SIMT 模式下的 IndexPut 操作，按索引将值写入目标张量的指定位置。

### 7.2 操作签名

| 操作数 | 类型 | 说明 |
|--------|------|------|
| `dst` | `AnyMemRef` | 目标 GM 缓冲区 |
| `index` | `AnyRankedTensor` | 索引张量 |
| `value` | `AnyRankedTensor` | 值张量 |
| `scatter_dim` | `AnyTypeOf<[I32, I64]>` | 散射维度 |
| `bound` | `AnyTypeOf<[I32, I64]>` | 索引上界 |
| `end_offset` | `Variadic<AnyTypeOf<[I32, I64]>>` | 结束偏移 |
| `start_offset` | `Variadic<AnyTypeOf<[I32, I64]>>` | 起始偏移 |
| `dst_stride` | `Variadic<AnyTypeOf<[I32, I64]>>` | 目标步长 |

## 8. scatterT

### 8.1 功能

SIMT 模式下的 Scatter 操作，按索引将值写入目标张量的指定位置。

### 8.2 操作签名

| 操作数 | 类型 | 说明 |
|--------|------|------|
| `dst` | `AnyMemRef` | 目标 GM 缓冲区 |
| `value` | `AnyRankedTensor` | 值张量 |
| `index_tile` | `AnyRankedTensor` | 索引张量 |
| `index_boundary` | `AnyTypeOf<[I32, I64]>` | 索引上界 |
| `dim` | `AnyTypeOf<[I32, I64]>` | 维度 |
| `dst_stride` | `Variadic<AnyTypeOf<[I32, I64]>>` | 目标步长 |
| `index_shape` | `Variadic<AnyTypeOf<[I32, I64]>>` | 索引形状 |
| `offsets` | `Variadic<AnyTypeOf<[I32, I64]>>` | 偏移量 |

## 9. atomic_cas

### 9.1 功能

原子比较并交换（Compare-And-Swap）操作。

### 9.2 操作签名

| 操作数 | 类型 | 说明 |
|--------|------|------|
| `input` | `Variadic<TensorOrMemref>` | 期望旧值(src0)和新值(src1) |
| `dst` | `TensorOrMemref` | 内存位置 |
| `output` | `Variadic<TensorOrMemref>` | 原始值 |

### 9.3 语义

```
if (V == A) V = B; return old_V;
```

### 9.4 MLIR 示例

```mlir
hfusion.atomic_cas ins(%src0, %src1 : memref<?xf32>, memref<?xf32>)
  outs(%dst : memref<?xf32>)

%result = hfusion.atomic_cas ins(%src0, %src1 : tensor<?xf32>, tensor<?xf32>)
  outs(%dst : tensor<?xf32>) -> tensor<?xf32>
```

## 10. atomic_xchg

### 10.1 功能

原子交换操作，将新值写入内存地址并返回旧值。

### 10.2 操作签名

| 操作数 | 类型 | 说明 |
|--------|------|------|
| `input` | `Variadic<TensorOrMemref>` | 新值(src) |
| `dst` | `TensorOrMemref` | 内存位置 |
| `output` | `Variadic<TensorOrMemref>` | 旧值 |

### 10.3 MLIR 示例

```mlir
hfusion.atomic_xchg ins(%src : memref<?xf32>) outs(%dst : memref<?xf32>)

%result = hfusion.atomic_xchg ins(%src : tensor<?xf32>)
  outs(%dst : tensor<?xf32>) -> tensor<?xf32>
```
