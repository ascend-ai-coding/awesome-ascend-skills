# 特殊操作

## 1. 概述

HFusion 方言的特殊操作包括调试操作、排序、直方图、嵌入聚集、数值检查和扩展乘法等。

> 源码参考：[HFusionOps.td](https://gitcode.com/Ascend/AscendNPU-IR/tree/master/bishengir/include/bishengir/Dialect/HFusion/IR/HFusionOps.td#L48-L269)

## 2. print

### 2.1 功能

Device 端打印操作，用于调试。

### 2.2 操作签名

| 操作数 | 类型 | 说明 |
|--------|------|------|
| `prefix` | `StrAttr` | 前缀字符串 |
| `hex` | `BoolAttr` | 是否以十六进制打印 |
| `arg` | `AnyTypeOf<[AnyInteger, AnyFloat, AnyRankedTensor]>` | 待打印的值 |

### 2.3 MLIR 示例

```mlir
hfusion.print "value:" hex = true %val : tensor<128xf32>
```

## 3. assert

### 3.1 功能

Device 端断言操作，用于调试。

### 3.2 操作签名

| 操作数 | 类型 | 说明 |
|--------|------|------|
| `msg` | `StrAttr` | 断言消息 |
| `cond` | `AnyTypeOf<[AnyInteger, AnyRankedTensor]>` | 条件值 |

### 3.3 MLIR 示例

```mlir
hfusion.assert "index out of range" %cond : i32
```

## 4. barrier

### 4.1 功能

同步一个 Core 上所有 Pipeline 的执行。

### 4.2 操作签名

无操作数，无结果。

### 4.3 MLIR 示例

```mlir
hfusion.barrier
```

## 5. sort

### 5.1 功能

沿指定轴对张量排序，输出排序后的值和对应索引。

### 5.2 操作签名

| 操作数 | 类型 | 说明 |
|--------|------|------|
| `src` | `TensorOrMemref` | 待排序张量 |
| `descending` | `BoolAttr` (默认: false) | 是否降序 |
| `sort_axis` | `I64Attr` | 排序轴 |
| `result` | `Variadic<AnyRankedTensor>` | 排序结果 |

### 5.3 约束

- 输入和输出必须具有相同的 rank
- 当前仅支持尾轴排序

### 5.4 MLIR 示例

```mlir
%result = hfusion.sort ins(%src : tensor<?xf32>)
  descending = true sort_axis = 0 -> tensor<?xf32>
```

## 6. histogram

### 6.1 功能

计算整数张量的直方图，支持可选掩码。

### 6.2 操作签名

| 操作数 | 类型 | 说明 |
|--------|------|------|
| `input` | `RankedTensorOf<[I8, UI8, I16, UI16, I32, UI32, I64, UI64]>` | 输入张量 |
| `num_bins` | `I64Attr` | 直方图桶数 |
| `mask` | `Optional<RankedTensorOf<[I1]>>` | 掩码 |
| `output` | `RankedTensorOf<[I8, I16, I32, I64]>` | 输出直方图 |

### 6.3 语义

对输入张量的每个元素，递增输出直方图中对应的桶。如果提供掩码，仅统计 mask[i]=true 的元素。输出必须为 1D 张量，长度等于 num_bins。

### 6.4 MLIR 示例

```mlir
%hist = hfusion.histogram %input, 256, %mask : tensor<1024xi32>, tensor<1024xi1> -> tensor<256xi64>
```

## 7. embedding_gather

### 7.1 功能

使用 Gather 语义执行嵌入查找操作。

### 7.2 操作签名

| 操作数 | 类型 | 说明 |
|--------|------|------|
| `src` | `AnyMemRef` | 2D 嵌入表（GM） |
| `index` | `AnyRankedTensor` | 1D/2D 索引张量 |
| `dst` | `AnyRankedTensor` | 目标张量 |
| `bound` | `AnyTypeOf<[I32, I64]>` | 词汇表大小（边界检查） |
| `offsets` | `Variadic<AnyTypeOf<[I32, I64]>>` | 偏移量 |
| `numels` | `Variadic<AnyTypeOf<[I32, I64]>>` | 元素数量 |
| `result` | `Optional<AnyRankedTensor>` | 输出 |

### 7.3 语义

```
result[b][i][d] = src[index[b][i]][d]
```

### 7.4 MLIR 示例

```mlir
%result = hfusion.embedding_gather
  ins(%src, %index, %bound, [], [] :
    memref<10000x768xf16>, tensor<128xi32>, i32)
  outs(%dst : tensor<128x768xf16>)
  -> tensor<128x768xf16>
```

## 8. is_inf

### 8.1 功能

判断浮点张量元素是否为正无穷或负无穷。

### 8.2 操作签名

| 操作数 | 类型 | 说明 |
|--------|------|------|
| `input` | `RankedTensorOf<[BF16, F16, F32]>` | 输入张量 |
| `output` | `RankedTensorOf<[I1]>` | 输出布尔张量 |

### 8.3 MLIR 示例

```mlir
%result = hfusion.isinf %input : tensor<128xf32> -> tensor<128xi1>
```

## 9. is_nan

### 9.1 功能

判断浮点张量元素是否为 NaN。

### 9.2 操作签名

| 操作数 | 类型 | 说明 |
|--------|------|------|
| `input` | `RankedTensorOf<[BF16, F16, F32]>` | 输入张量 |
| `output` | `RankedTensorOf<[I1]>` | 输出布尔张量 |

### 9.3 MLIR 示例

```mlir
%result = hfusion.isnan %input : tensor<128xf32> -> tensor<128xi1>
```

## 10. is_finite

### 10.1 功能

判断浮点张量元素是否为有限值（非 NaN 且非无穷）。

### 10.2 操作签名

| 操作数 | 类型 | 说明 |
|--------|------|------|
| `input` | `RankedTensorOf<[BF16, F16, F32]>` | 输入张量 |
| `output` | `RankedTensorOf<[I1]>` | 输出布尔张量 |

### 10.3 Traits

- `BiShengIRAggregatedOpInterface`（支持 decomposeOperation）

### 10.4 MLIR 示例

```mlir
%result = hfusion.isfinite %input : tensor<128xf32> -> tensor<128xi1>
```

## 11. mulext

### 11.1 功能

扩展有符号整数乘法，返回 2N 位乘积的低半部分和高半部分。

### 11.2 操作签名

| 操作数 | 类型 | 说明 |
|--------|------|------|
| `lhs` | `SignlessIntegerLike` | 左操作数 |
| `rhs` | `SignlessIntegerLike` | 右操作数 |
| `low` | `SignlessIntegerLike` | 乘积低半部分 |
| `high` | `SignlessIntegerLike` | 乘积高半部分 |

### 11.3 Traits

- `Pure`, `Commutative`
- `AllTypesMatch<["lhs", "rhs", "low", "high"]>`

### 11.4 MLIR 示例

```mlir
%low, %high = hfusion.mulext %a, %b : i32
```

## 12. symbolic_dim

### 12.1 功能

引用符号维度并返回 index 类型值。

### 12.2 操作签名

| 操作数 | 类型 | 说明 |
|--------|------|------|
| `symbolName` | `SymbolRefAttr` | 符号名称 |
| `result` | `Index` | 结果 |

### 12.3 MLIR 示例

```mlir
%0 = hfusion.symbolic_dim @SymName : index
```
