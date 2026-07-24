# 归约操作

## 1. 概述

HFusion 方言提供多种归约操作，包括带索引的归约（reduce_with_index）和累积操作（cumsum/cumprod）。归约操作沿指定维度对张量进行聚合计算。

> 源码参考：[HFusionStructuredOps.td](https://gitcode.com/Ascend/AscendNPU-IR/tree/master/bishengir/include/bishengir/Dialect/HFusion/IR/HFusionStructuredOps.td#L60-L142)、[HFusionOps.td](https://gitcode.com/Ascend/AscendNPU-IR/tree/master/bishengir/include/bishengir/Dialect/HFusion/IR/HFusionOps.td#L275-L323)

## 2. reduce_with_index

### 2.1 功能

使用 max/min 对 AnyShaped 执行归约操作，同时返回归约值和对应的索引。支持两种模式：(1) 接受输入和索引，产生归约值和索引；(2) 仅接受输入，产生归约值和索引。

### 2.2 操作签名

| 操作数/结果 | 类型 | 说明 |
|-------------|------|------|
| `inputs` | `Variadic<AnyShaped>` | 输入张量 |
| `inits` | `Variadic<AnyShaped>` | 初始值 |
| `reduce_kind` | `HFusion_ReduceWithIndexOpAttr` | 归约类型（min/max） |
| `unsigned_src` | `BoolAttr` | 源是否为无符号整数类型 |
| `tie_break_left` | `OptionalAttr<BoolAttr>` | 平局时取最左索引 |
| `dimensions` | `DenseI64ArrayAttr` | 归约维度 |
| `result` | `Variadic<AnyTensor>` | 归约结果 |

### 2.3 归约类型

| ReduceWithIndexKind | 助记符 | 说明 |
|---------------------|--------|------|
| `MIN` | `min` | 最小值归约 |
| `MAX` | `max` | 最大值归约 |

### 2.4 Traits

- `AttrSizedOperandSegments`
- `ResultOnlyIfTensor`
- `DestinationStyleOpInterface`
- `LinalgStructuredInterface`
- `ReifyRankedShapedTypeOpInterface`

### 2.5 MLIR 示例

```mlir
%val, %idx = hfusion.reduce_with_index
  ins(%input, %val_init, %idx_init : tensor<128x256xf32>, tensor<128xf32>, tensor<128xi64>)
  outs(%val_out, %idx_out : tensor<128xf32>, tensor<128xi64>)
  reduce_with_index_kind = <max>
  unsigned_src = false
  dimensions = [1]
```

## 3. cumsum

### 3.1 功能

沿指定维度计算输入张量的累积和。

### 3.2 操作签名

| 操作数/结果 | 类型 | 说明 |
|-------------|------|------|
| `input` | `RankedTensorOf<[BF16, F16, F32, I8, I16, I32, I64, F8E4M3FN, F8E5M2]>` | 输入张量 |
| `cum_dims` | `DenseI64ArrayAttr` | 累积维度（严格递增排序） |
| `reverse` | `BoolAttr` | 是否反向累积 |
| `output` | `RankedTensorOf<[BF16, F16, F32, I8, I16, I32, I64, F8E4M3FN, F8E5M2]>` | 输出张量 |

### 3.3 约束

- 当前仅支持单个累积维度
- `cum_dims` 必须严格递增排序

### 3.4 MLIR 示例

```mlir
%result = hfusion.cumsum %input cum_dims = [1] reverse = false
  : tensor<128x256xf32> -> tensor<128x256xf32>
```

## 4. cumprod

### 4.1 功能

沿指定维度计算输入张量的累积积。

### 4.2 操作签名

| 操作数/结果 | 类型 | 说明 |
|-------------|------|------|
| `input` | `RankedTensorOf<[BF16, F16, F32, I8, I16, I32, I64]>` | 输入张量 |
| `cum_dims` | `DenseI64ArrayAttr` | 累积维度（严格递增排序） |
| `reverse` | `BoolAttr` | 是否反向累积 |
| `output` | `RankedTensorOf<[BF16, F16, F32, I8, I16, I32, I64]>` | 输出张量 |

### 4.3 约束

- 当前仅支持单个累积维度
- `cum_dims` 必须严格递增排序

### 4.4 MLIR 示例

```mlir
%result = hfusion.cumprod %input cum_dims = [0] reverse = true
  : tensor<128x256xf32> -> tensor<128x256xf32>
```

## 5. CumOpType 枚举

| 枚举值 | 助记符 | 说明 |
|--------|--------|------|
| `UNDEFINED` | `undefined` | 未定义 |
| `CUMSUM` | `cumsum` | 累积和 |
| `CUMPROD` | `cumprod` | 累积积 |

## 6. 语义说明

### 6.1 reduce_with_index 语义

对于 max 归约：
```
result_value[i] = max(input[i][0], input[i][1], ..., input[i][N-1])
result_index[i] = argmax(input[i][0], input[i][1], ..., input[i][N-1])
```

当 `tie_break_left = true` 时，如果多个元素具有相同的最大值，返回最左边的索引。

### 6.2 cumsum 语义

正向累积（reverse = false）：
```
output[i] = sum(input[0], input[1], ..., input[i])
```

反向累积（reverse = true）：
```
output[i] = sum(input[i], input[i+1], ..., input[N-1])
```

### 6.3 cumprod 语义

正向累积（reverse = false）：
```
output[i] = product(input[0], input[1], ..., input[i])
```

反向累积（reverse = true）：
```
output[i] = product(input[i], input[i+1], ..., input[N-1])
```
