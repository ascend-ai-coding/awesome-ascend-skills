# 数据搬移操作

## 1. 概述

HFusion 方言提供丰富的数据搬移操作，包括广播、转置、拼接、填充、交织、翻转和聚集等。这些操作用于在不同形状和布局的张量间进行数据重排。

> 源码参考：[HFusionOps.td](https://gitcode.com/Ascend/AscendNPU-IR/tree/master/bishengir/include/bishengir/Dialect/HFusion/IR/HFusionOps.td#L128-L213)、[HFusionStructuredOps.td](https://gitcode.com/Ascend/AscendNPU-IR/tree/master/bishengir/include/bishengir/Dialect/HFusion/IR/HFusionStructuredOps.td#L196-L260)

## 2. interleave

### 2.1 功能

将 n 个输入张量沿最后一维交织，构造一个输出张量。当前仅支持 n=2。

### 2.2 操作签名

| 操作数/结果 | 类型 | 说明 |
|-------------|------|------|
| `input` | `Variadic<AnyRankedTensor>` | 输入张量（2 个，形状相同） |
| `output` | `AnyRankedTensor` | 输出张量 |

### 2.3 Traits

- `Pure`, `Commutative`
- `SameOperandsAndResultRank`
- `ReifyRankedShapedTypeOpInterface`

### 2.4 MLIR 示例

```mlir
%output = hfusion.interleave %a, %b : tensor<128x256xf32>, tensor<128x256xf32> -> tensor<128x512xf32>
```

## 3. deinterleave

### 3.1 功能

将一个输入张量沿最后一维反交织为两个张量。偶数索引元素进入第一个输出，奇数索引元素进入第二个输出。

### 3.2 操作签名

| 操作数/结果 | 类型 | 说明 |
|-------------|------|------|
| `input` | `AnyRankedTensor` | 输入张量 |
| `channelIndex` | `I64Attr` | 通道选择：-1=全部, 0=偶数, 1=奇数 |
| `output` | `Variadic<AnyRankedTensor>` | 输出张量 |

### 3.3 channelIndex 行为

| 值 | 行为 |
|----|------|
| -1 | 输出两个张量（偶数索引 + 奇数索引） |
| 0 | 仅输出偶数索引通道 |
| 1 | 仅输出奇数索引通道 |

### 3.4 约束

- 输入张量最后一维大小必须是 2 的倍数

### 3.5 MLIR 示例

```mlir
%even, %odd = hfusion.deinterleave %input channel_index = -1
  : tensor<128x512xf32> -> tensor<128x256xf32>, tensor<128x256xf32>

%even_only = hfusion.deinterleave %input channel_index = 0
  : tensor<128x512xf32> -> tensor<128x256xf32>
```

## 4. flip

### 4.1 功能

沿指定维度翻转张量。当前仅支持最后一维。

### 4.2 操作签名

| 操作数/结果 | 类型 | 说明 |
|-------------|------|------|
| `input` | `AnyRankedTensor` | 输入张量 |
| `flip_axis` | `I64Attr` | 翻转轴 |
| `output` | `AnyRankedTensor` | 输出张量 |

### 4.3 MLIR 示例

```mlir
%output = hfusion.flip %input flip_axis = 1
  : tensor<128x256xf32> -> tensor<128x256xf32>
```

## 5. gather

### 5.1 功能

沿指定轴从源张量中聚集元素。对应 `triton.language.gather` 语义。

### 5.2 操作签名

| 操作数/结果 | 类型 | 说明 |
|-------------|------|------|
| `src` | `AnyShaped` | 源张量 |
| `index` | `AnyShaped` | 索引张量 |
| `init` | `AnyShaped` | 初始值 |
| `axis` | `I64Attr` | 聚集轴 |
| `result` | `Variadic<AnyTensor>` | 输出 |

### 5.3 Traits

- `AllRanksMatch<["src", "index", "init"]>`
- `AllShapesMatch<["index", "init"]>`
- `AllElementTypesMatch<["src", "init"]>`
- `BiShengIRAggregatedOpInterface`（支持 decomposeOperation）

### 5.4 语义

给定 src:tensor<16x16> 和 index:tensor<16x4>，axis=1：
```
for i in 0 to 16:
  for j in 0 to 4:
    for k in 0 to 16:
      output[i][j] = (index[i][j] == k) ? src[i][k] : output[i][j]
```

### 5.5 MLIR 示例

```mlir
%result = hfusion.gather ins(%src, %index, %init : tensor<16x16xf32>, tensor<16x4xi32>, tensor<16x4xf32>)
  axis = 1 -> tensor<16x4xf32>
```

## 6. arange

### 6.1 功能

生成等差数列张量，支持偏移和多维步长。

### 6.2 操作签名

| 操作数/结果 | 类型 | 说明 |
|-------------|------|------|
| `offset` | `Optional<Index>` | 偏移量（默认 0） |
| `strides` | `Variadic<Index>` | 各维步长 |
| `init` | `AnyShaped` | 初始张量（决定输出形状） |
| `result_tensor` | `Optional<AnyShaped>` | 输出张量 |

### 6.3 语义

3D arange：`arange[i, j, k] = offset + stride[0] * i + stride[1] * j + stride[2] * k`

## 7. 其他数据搬移操作

以下操作由 Linalg 命名操作或 HFusion 结构化操作覆盖，通过 OpDSL 或 Linalg 通用操作实现：

| 操作 | 说明 | 实现方式 |
|------|------|----------|
| `broadcast` | 广播张量到目标形状 | Linalg 广播语义 |
| `transpose` | 转置张量维度 | Linalg 转置语义 |
| `concat` | 沿指定维度拼接张量 | Linalg 拼接语义 |
| `pad` | 填充张量边界 | Linalg 填充语义 |
