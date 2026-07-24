# 矩阵乘操作

## 1. 概述

HFusion 方言提供矩阵乘操作，包括微缩放格式矩阵乘（matmul_mx）和分组矩阵乘（group_matmul）。这些操作是 NPU Cube 计算单元的核心负载。

> 源码参考：[HFusionOps.td](https://gitcode.com/Ascend/AscendNPU-IR/tree/master/bishengir/include/bishengir/Dialect/HFusion/IR/HFusionOps.td#L964-L1021)、[HFusionNamedStructuredOps.yaml](https://gitcode.com/Ascend/AscendNPU-IR/tree/master/bishengir/include/bishengir/Dialect/HFusion/IR/HFusionNamedStructuredOps.yaml#L386-L424)

## 2. matmul_mx

### 2.1 功能

执行微缩放（Microscaling）格式的矩阵乘法，输入通过缩放因子隐式缩放。常用于 FP8/FP4 等量化数据类型。计算公式：`C = (A * scale_a) dot (B * scale_b)`。

微缩放格式遵循 OCP Microscaling Formats (MX) 规范：https://www.opencompute.org/documents/ocp-microscaling-formats-mx-v1-0-spec-final-pdf

### 2.2 操作签名

| 操作数/结果 | 类型 | 说明 |
|-------------|------|------|
| `inputA` | `ShapedTypeOf<[F8E4M3FN, F8E5M2]>` | 左矩阵 |
| `inputB` | `ShapedTypeOf<[F8E4M3FN, F8E5M2]>` | 右矩阵 |
| `scaleA` | `ShapedTypeOf<[UI8, I8]>` | 左矩阵缩放因子 |
| `scaleB` | `ShapedTypeOf<[UI8, I8]>` | 右矩阵缩放因子 |
| `acc` | `ShapedTypeOf<[AnyFloat]>` | 累加器（DPS init） |
| `result` | `ShapedTypeOf<[AnyFloat]>` | 输出结果 |

### 2.3 Traits

- `Pure`
- `DestinationStyleOpInterface`
- `BiShengIRAggregatedOpInterface`（支持 decomposeOperation）

### 2.4 MLIR 示例

```mlir
%result = hfusion.matmul_mx
  ins(%a, %b, %sa, %sb :
    tensor<16x32xf8E4M3FN>, tensor<32x64xf8E4M3FN>,
    tensor<2x2xui8>, tensor<2x4xui8>)
  outs(%acc : tensor<16x64xf32>)
  -> tensor<16x64xf32>
```

### 2.5 Builder

```tablegen
OpBuilder<(ins "Value":$inputA, "Value":$inputB,
               "Value":$scaleA, "Value":$scaleB, "Value":$acc)>
```

## 3. group_matmul

### 3.1 功能

执行分组矩阵乘法，用于 MoE（Mixture of Experts）等场景。每个 Expert 的权重矩阵与其分配的 Token 进行矩阵乘。

### 3.2 操作签名

| 操作数/结果 | 类型 | 角色 |
|-------------|------|------|
| `w1` | input_tensor (type_var: T1) | Expert 权重矩阵，shape_map: `(d0, d1, d2) -> (d2, d1, d0)` |
| `tokens` | input_tensor (type_var: T2) | Token 嵌入，shape_map: `(d0, d1, d2) -> (d1, d2)` |
| `tokens_per_expert` | input_tensor (type_var: T3) | 每 Expert 的 Token 数，shape_map: `(d0, d1, d2) -> (1)` |
| `output` | output_tensor (type_var: T4) | 输出，shape_map: `(d0, d1, d2) -> (d0, d0)` |

### 3.3 Indexing Maps

```mlir
affine_map<(d0, d1, d2) -> (d2, d1, d0)>   // w1
affine_map<(d0, d1) -> (d0, d1)>            // tokens
affine_map<(d0) -> (0)>                      // tokens_per_expert
affine_map<(d0, d1, d2) -> (d0, d0)>         // output
```

### 3.4 Iterator Types

```
["parallel", "parallel", "reduction"]
```

### 3.5 MLIR 示例

```mlir
%result = hfusion.group_matmul
  ins(%w1, %tokens, %tokens_per_expert :
    tensor<?x?x?xf32>, tensor<?x?xf32>, tensor<1xi64>)
  outs(%output : tensor<?x?xf32>)
```

## 4. MmMapMode 枚举

| 枚举值 | 助记符 | 说明 |
|--------|--------|------|
| `CoreOp` | `core_op` | 核心操作模式 |
| `MacroInstr` | `macro_instr` | 宏指令模式 |

## 5. 数据格式说明

### 5.1 FP8 格式

| 类型 | 说明 |
|------|------|
| `F8E4M3FN` | 4 位指数、3 位尾数，无无穷，用于前向传播 |
| `F8E5M2` | 5 位指数、2 位尾数，支持无穷，用于反向传播 |

### 5.2 微缩放格式

微缩放格式将一组元素共享一个缩放因子，典型配置为每 32 个元素共享 1 个 UI8 缩放因子。这种格式在保持精度的同时大幅减少了存储和计算开销。
