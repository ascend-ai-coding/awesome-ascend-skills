# HFusion 方言总览

## 1. 简介

HFusion（Hybrid Fusion，混合融合）方言是 AscendNPU-IR 中的张量级融合操作方言，提供丰富的张量操作原语，包括逐元操作、归约、矩阵乘、数据搬移、内存操作和特殊操作。HFusion 操作基于 Linalg 结构化操作范式，支持通过 OpDSL 声明式定义，并可通过变换 Pass 进行自动融合和向量化。

- **方言名称**：`hfusion`
- **C++ 命名空间**：`::mlir::hfusion`
- **方言具有 Canonicalizer**：是

## 2. 依赖方言

| 依赖方言 | 说明 |
|----------|------|
| `hacc::HACCDialect` | 异构计算调用 |
| `hmap::HMAPDialect` | 混合 Mesh 感知并行 |
| `linalg::LinalgDialect` | 结构化操作 |
| `mathExt::MathExtDialect` | 扩展数学操作 |
| `mesh::MeshDialect` | Mesh 并行 |
| `symbol::SymbolDialect` | 符号化形状 |

## 3. 操作分类

| 类别 | 操作 | 数量 |
|------|------|------|
| 逐元操作 | elemwise_unary, elemwise_binary, compare, select, cast, bitcast | 6 |
| 归约操作 | reduce_with_index, cumsum, cumprod | 3 |
| 矩阵乘操作 | matmul_mx, group_matmul | 2 |
| 数据搬移 | load, store, broadcast, transpose, concat, pad, interleave, deinterleave, flip, gather, arange | 11 |
| 内存操作 | gather_load, scatter_store, indirect_load, indirect_store, gatherT, index_put, scatterT, atomic_cas, atomic_xchg | 9 |
| 特殊操作 | print, assert, barrier, sort, histogram, embedding_gather, is_inf, is_nan, is_finite, mulext, symbolic_dim | 11 |

## 4. 源码位置

| 文件 | 路径 | 说明 |
|------|------|------|
| 方言基类 | [HFusionBase.td](https://gitcode.com/Ascend/AscendNPU-IR/tree/master/bishengir/include/bishengir/Dialect/HFusion/IR/HFusionBase.td) | 方言定义与属性枚举 |
| 枚举定义 | [HFusionEnums.td](https://gitcode.com/Ascend/AscendNPU-IR/tree/master/bishengir/include/bishengir/Dialect/HFusion/IR/HFusionEnums.td) | 函数属性枚举 |
| 属性定义 | [HFusionAttrs.td](https://gitcode.com/Ascend/AscendNPU-IR/tree/master/bishengir/include/bishengir/Dialect/HFusion/IR/HFusionAttrs.td) | 方言属性 |
| 操作定义 | [HFusionOps.td](https://gitcode.com/Ascend/AscendNPU-IR/tree/master/bishengir/include/bishengir/Dialect/HFusion/IR/HFusionOps.td) | 非结构化操作 |
| 结构化操作 | [HFusionStructuredOps.td](https://gitcode.com/Ascend/AscendNPU-IR/tree/master/bishengir/include/bishengir/Dialect/HFusion/IR/HFusionStructuredOps.td) | 结构化操作 |
| OpDSL 定义 | [HFusionNamedStructuredOps.yaml](https://gitcode.com/Ascend/AscendNPU-IR/tree/master/bishengir/include/bishengir/Dialect/HFusion/IR/HFusionNamedStructuredOps.yaml) | YAML 声明式操作定义 |
| 变换 Pass | [Passes.td](https://gitcode.com/Ascend/AscendNPU-IR/tree/master/bishengir/include/bishengir/Dialect/HFusion/Transforms/Passes.td) | 变换 Pass 定义 |

## 5. 类型系统

| 类型 | 定义 | 说明 |
|------|------|------|
| `TensorOrMemref` | `AnyTypeOf<[AnyMemRef, AnyRankedTensor]>` | 张量或 MemRef |
| `ShapedTypeOf<allowedTypes>` | 自定义 | 元素类型受限的 ShapedType |

## 6. 操作基类

| 基类 | 说明 |
|------|------|
| `HFusion_Op<mnemonic, traits>` | 非结构化操作基类 |
| `HFusionStructuredBase_Op<mnemonic, props>` | 结构化操作基类，实现 LinalgStructuredInterface 和 DestinationStyleOpInterface |

## 7. 典型 IR 示例

```mlir
%result = hfusion.elemwise_unary ins(%input : tensor<128x256xf32>)
  outs(%init : tensor<128x256xf32>)
  unary_fn = <sqrt>

%reduced = hfusion.reduce_with_index ins(%input, %init : tensor<128x256xf32>, tensor<128xf32>)
  outs(%out_init, %idx_init : tensor<128xf32>, tensor<128xi64>)
  reduce_with_index_kind = <max>

%matmul_result = hfusion.matmul_mx ins(%a, %b, %sa, %sb : tensor<16x32xf8E4M3FN>, tensor<32x64xf8E4M3FN>, tensor<2x2xui8>, tensor<2x4xui8>)
  outs(%acc : tensor<16x64xf32>) -> tensor<16x64xf32>
```

## 8. 文档索引

| 文档 | 内容 |
|------|------|
| [01-elementwise-ops.md](01-elementwise-ops.md) | 逐元操作 |
| [02-reduction-ops.md](02-reduction-ops.md) | 归约操作 |
| [03-matmul-ops.md](03-matmul-ops.md) | 矩阵乘操作 |
| [04-data-movement-ops.md](04-data-movement-ops.md) | 数据搬移操作 |
| [05-memory-ops.md](05-memory-ops.md) | 内存操作 |
| [06-special-ops.md](06-special-ops.md) | 特殊操作 |
| [07-attributes-enums.md](07-attributes-enums.md) | 属性与枚举速查 |
| [08-transforms.md](08-transforms.md) | HFusion 变换 Pass 总览 |
