# HIVM 一元向量运算

> 关键词：HIVM, Unary, vexp, vabs, vln, vrelu, vrsqrt, vsqrt, vtanh, vsin, vcos, verf, vrec, vnot

## 概述

HIVM 一元向量运算继承自 `HIVM_ElementwiseUnaryOp`，对输入向量的每个元素独立执行运算，产生一个相同形状的结果向量。所有一元运算满足 `ElementwiseNaryOpTrait<1>`，即 1 个输入操作数、1 个输出结果。

一元运算可分为以下类别：
- **数学函数**：vexp, vln, vsqrt, vrsqrt, vrec, vsin, vcos, vtanh, verf
- **数值操作**：vabs, vrelu
- **位操作**：vnot

> Python API 对应：`tl.math.exp()`, `tl.math.log()`, `tl.abs()`, `tl.math.rsqrt()`, `tl.math.sqrt()`, `tl.math.tanh()`, `tl.math.sin()`, `tl.math.cos()`, `tl.math.erf()`, `tl.math.reciprocal()` 等。

## IR 操作定义

### 基类：HIVM_ElementwiseUnaryOp

```tablegen
class HIVM_ElementwiseUnaryOp<string mnemonic, list<Trait> traits = []> :
  HIVM_ElementwiseNaryOp<mnemonic,
                         !listconcat([ElementwiseNaryOpTrait<1>], traits)>;
```

源码参考：[HIVMVectorOps.td#L101-L103](https://gitcode.com/Ascend/AscendNPU-IR/tree/master/bishengir/include/bishengir/Dialect/HIVM/IR/HIVMVectorOps.td#L101-L103)

---

### hir.vexp — 逐元指数运算

#### TableGen 定义

```tablegen
def VExpOp : HIVM_ElementwiseUnaryOp<"vexp",
    [SameOperandsElementType, StaticMaxRankTrait<3>,
     VectorOnlyTrait<0>,
     OperElemTypeConstraints<[0, 1], [F16, F32]>,
     DeclareOpInterfaceMethods<ExtraBufferOpInterface,
       ["getExtraBufferSize"]>,
     BroadcastableOTF
    ]> {
  let summary = "Elementwise Vector Exponential Op";
  let description = baseClassDescription # [{
    Additional constraints:
      1. The input/init operands and result have the same element type.
  }];
  let arguments = (ins Variadic<AnyShaped>:$src,
                       Variadic<AnyShaped>:$dst,
                       Optional<AnyMemRef>:$temp_buffer,
                       DefaultValuedAttr<DenseI64ArrayAttr, "{}">:$transpose,
                       DefaultValuedAttr<DenseI64ArrayAttr, "{}">:$broadcast
  );
}
```

源码参考：[HIVMVectorOps.td#L105-L136](https://gitcode.com/Ascend/AscendNPU-IR/tree/master/bishengir/include/bishengir/Dialect/HIVM/IR/HIVMVectorOps.td#L105-L136)

#### 参数说明

| 参数 | 类型 | 必选 | 说明 | 约束 |
|------|------|------|------|------|
| $src | Variadic\<AnyShaped\> | 是 | 输入向量 | VectorOnly, 元素类型 F16/F32 |
| $dst | Variadic\<AnyShaped\> | 是 | 输出向量（DestinationStyle） | 与 src 相同元素类型 |
| $temp_buffer | Optional\<AnyMemRef\> | 否 | 临时缓冲区 | ExtraBufferOpInterface |
| $transpose | DenseI64ArrayAttr (默认 {}) | 否 | OTF 转置维度 | TransposableOTF 约束 |
| $broadcast | DenseI64ArrayAttr (默认 {}) | 否 | OTF 广播维度 | BroadcastableOTF 约束 |
| $result | Variadic\<AnyRankedTensor\> | 是（tensor 语义） | 结果 | 与 src 相同元素类型 |

#### 数据类型约束

| 操作 | 支持的元素类型 | 约束来源 |
|------|--------------|----------|
| vexp | F16, F32 | OperElemTypeConstraints<[0, 1], [F16, F32]> |

#### IR 示例

```mlir
%result = hivm.hir.vexp ins(%src : tensor<5x?x10xf32>) outs(%dst : tensor<5x?x10xf32>) -> tensor<5x?x10xf32>

hivm.hir.vexp ins(%src : memref<5x?x10xf32>) outs(%dst : memref<5x?x10xf32>)

%result = hivm.hir.vexp ins(%src : tensor<5x?x10xf32>) outs(%dst : tensor<5x?x10xf32>) broadcast = [0] -> tensor<5x?x10xf32>
```

---

### hir.vabs — 逐元绝对值

#### TableGen 定义

```tablegen
def VAbsOp : HIVM_ElementwiseUnaryOp<"vabs",
    [SameOperandsElementType, StaticMaxRankTrait<3>,
     VectorOnlyTrait<0>,
     OperElemTypeConstraints<[0], [F16, F32, I8, I16, I32, I64]>,
     DeclareOpInterfaceMethods<ImplByScalarOpInterface>,
     DeclareOpInterfaceMethods<ExtraBufferOpInterface,
       ["getExtraBufferSize"]>,
     DeclareOpInterfaceMethods<VectorizableOpInterface>,
     BroadcastableOTF
    ]>
```

源码参考：[HIVMVectorOps.td#L138-L171](https://gitcode.com/Ascend/AscendNPU-IR/tree/master/bishengir/include/bishengir/Dialect/HIVM/IR/HIVMVectorOps.td#L138-L171)

#### 参数说明

| 参数 | 类型 | 必选 | 说明 | 约束 |
|------|------|------|------|------|
| $src | Variadic\<AnyShaped\> | 是 | 输入向量 | VectorOnly, 元素类型 F16/F32/I8/I16/I32/I64 |
| $dst | Variadic\<AnyShaped\> | 是 | 输出向量 | 与 src 相同元素类型 |
| $temp_buffer | Optional\<AnyMemRef\> | 否 | 临时缓冲区 | - |
| $transpose | DenseI64ArrayAttr (默认 {}) | 否 | OTF 转置维度 | - |
| $broadcast | DenseI64ArrayAttr (默认 {}) | 否 | OTF 广播维度 | - |

#### 数据类型约束

| 操作 | 支持的元素类型 | 约束来源 |
|------|--------------|----------|
| vabs | F16, F32, I8, I16, I32, I64 | OperElemTypeConstraints<[0], [F16, F32, I8, I16, I32, I64]> |

#### 特殊接口

- **ImplByScalarOpInterface**：支持标量实现路径
- **VectorizableOpInterface**：支持向量化

#### IR 示例

```mlir
%result = hivm.hir.vabs ins(%src : tensor<1x?x10xf32>) outs(%dst : tensor<5x?x10xf32>) broadcast = [0] -> tensor<5x?x10xf32>

hivm.hir.vabs ins(%src : memref<1x?x10xf32>) outs(%dst : memref<5x?x10xf32>) broadcast = [0]
```

---

### hir.vln — 逐元自然对数

#### TableGen 定义

```tablegen
def VLnOp : HIVM_ElementwiseUnaryOp<"vln",
    [SameOperandsElementType, StaticMaxRankTrait<3>,
     VectorOnlyTrait<0>,
     OperElemTypeConstraints<[0], [F16, F32]>,
     DeclareOpInterfaceMethods<ExtraBufferOpInterface,
       ["getExtraBufferSize"]>,
     BroadcastableOTF
    ]>
```

源码参考：[HIVMVectorOps.td#L173-L204](https://gitcode.com/Ascend/AscendNPU-IR/tree/master/bishengir/include/bishengir/Dialect/HIVM/IR/HIVMVectorOps.td#L173-L204)

#### 数据类型约束

| 操作 | 支持的元素类型 | 约束来源 |
|------|--------------|----------|
| vln | F16, F32 | OperElemTypeConstraints<[0], [F16, F32]> |

#### IR 示例

```mlir
%result = hivm.hir.vln ins(%src : tensor<5x?x10xf32>) outs(%dst : tensor<5x?x10xf32>) -> tensor<5x?x10xf32>

hivm.hir.vln ins(%src : memref<5x?x10xf32>) outs(%dst : memref<5x?x10xf32>)
```

---

### hir.vrelu — 逐元 ReLU

#### TableGen 定义

```tablegen
def VReluOp : HIVM_ElementwiseUnaryOp<"vrelu",
    [SameOperandsElementType, StaticMaxRankTrait<3>,
     VectorOnlyTrait<0>,
     OperElemTypeConstraints<[0], [F16, F32, I32]>,
     DeclareOpInterfaceMethods<ExtraBufferOpInterface,
       ["getExtraBufferSize"]>,
     BroadcastableOTF
    ]>
```

源码参考：[HIVMVectorOps.td#L206-L237](https://gitcode.com/Ascend/AscendNPU-IR/tree/master/bishengir/include/bishengir/Dialect/HIVM/IR/HIVMVectorOps.td#L206-L237)

#### 数据类型约束

| 操作 | 支持的元素类型 | 约束来源 |
|------|--------------|----------|
| vrelu | F16, F32, I32 | OperElemTypeConstraints<[0], [F16, F32, I32]> |

#### IR 示例

```mlir
%result = hivm.hir.vrelu ins(%src : tensor<?x5x10xf32>) outs(%dst : tensor<5x?x10xf32>) transpose = [1, 0, 2] -> tensor<5x?x10xf32>

hivm.hir.vrelu ins(%src : memref<5x1x10xi32>) outs(%dst : memref<5x?x10xi32>) broadcast = [1]
```

降级到 HFusion：`hfusion.elemwise_unary {fun = #hfusion.unary_fn<relu>}`

---

### hir.vrsqrt — 逐元倒数平方根

#### TableGen 定义

```tablegen
def VRsqrtOp : HIVM_ElementwiseUnaryOp<"vrsqrt",
    [SameOperandsElementType, StaticMaxRankTrait<3>,
     VectorOnlyTrait<0>,
     OperElemTypeConstraints<[0], [F16, F32]>,
     DeclareOpInterfaceMethods<ExtraBufferOpInterface,
       ["getExtraBufferSize"]>,
     BroadcastableOTF
    ]>
```

源码参考：[HIVMVectorOps.td#L239-L270](https://gitcode.com/Ascend/AscendNPU-IR/tree/master/bishengir/include/bishengir/Dialect/HIVM/IR/HIVMVectorOps.td#L239-L270)

#### 数据类型约束

| 操作 | 支持的元素类型 | 约束来源 |
|------|--------------|----------|
| vrsqrt | F16, F32 | OperElemTypeConstraints<[0], [F16, F32]> |

#### IR 示例

```mlir
%result = hivm.hir.vrsqrt ins(%src : tensor<5x?x10xf32>) outs(%dst : tensor<5x?x10xf32>) -> tensor<5x?x10xf32>

hivm.hir.vrsqrt ins(%src : memref<5x?x10xf32>) outs(%dst : memref<5x?x10xf32>)
```

---

### hir.vsqrt — 逐元平方根

#### TableGen 定义

```tablegen
def VSqrtOp : HIVM_ElementwiseUnaryOp<"vsqrt",
    [SameOperandsElementType, StaticMaxRankTrait<3>,
     VectorOnlyTrait<0>,
     OperElemTypeConstraints<[0], [F16, F32]>,
     DeclareOpInterfaceMethods<ExtraBufferOpInterface,
       ["getExtraBufferSize"]>,
     BroadcastableOTF
    ]>
```

源码参考：[HIVMVectorOps.td#L272-L303](https://gitcode.com/Ascend/AscendNPU-IR/tree/master/bishengir/include/bishengir/Dialect/HIVM/IR/HIVMVectorOps.td#L272-L303)

#### 数据类型约束

| 操作 | 支持的元素类型 | 约束来源 |
|------|--------------|----------|
| vsqrt | F16, F32 | OperElemTypeConstraints<[0], [F16, F32]> |

#### IR 示例

```mlir
%result = hivm.hir.vsqrt ins(%src : tensor<5x?x10xf32>) outs(%dst : tensor<5x?x10xf32>) -> tensor<5x?x10xf32>
```

---

### hir.vtanh — 逐元双曲正切

#### TableGen 定义

```tablegen
def VTanhOp : HIVM_ElementwiseUnaryOp<"vtanh",
    [SameOperandsElementType,
     VectorOnlyTrait<0>, NoLibraryFunctionTrait
    ]>
```

源码参考：[HIVMVectorOps.td#L305-L315](https://gitcode.com/Ascend/AscendNPU-IR/tree/master/bishengir/include/bishengir/Dialect/HIVM/IR/HIVMVectorOps.td#L305-L315)

#### 数据类型约束

| 操作 | 支持的元素类型 | 约束来源 |
|------|--------------|----------|
| vtanh | 浮点类型 | SameOperandsElementType（无显式 OperElemTypeConstraints） |

注意：vtanh 使用 `NoLibraryFunctionTrait`，表示不通过库函数实现，而是使用内联实现路径。

#### IR 示例

```mlir
%result = hivm.hir.vtanh ins(%src : tensor<5x?x10xf32>) outs(%dst : tensor<5x?x10xf32>) -> tensor<5x?x10xf32>

hivm.hir.vtanh ins(%src : memref<5x?x10xf32>) outs(%dst : memref<5x?x10xf32>)
```

---

### hir.vsin — 逐元正弦

#### TableGen 定义

```tablegen
def VSinOp : HIVM_ElementwiseUnaryOp<"vsin",
    [SameOperandsElementType,
     VectorOnlyTrait<0>, NoLibraryFunctionTrait
    ]>
```

源码参考：[HIVMVectorOps.td#L317-L327](https://gitcode.com/Ascend/AscendNPU-IR/tree/master/bishengir/include/bishengir/Dialect/HIVM/IR/HIVMVectorOps.td#L317-L327)

#### 数据类型约束

| 操作 | 支持的元素类型 | 约束来源 |
|------|--------------|----------|
| vsin | 浮点类型 | SameOperandsElementType（无显式 OperElemTypeConstraints） |

#### IR 示例

```mlir
%result = hivm.hir.vsin ins(%src : tensor<5x?x10xf32>) outs(%dst : tensor<5x?x10xf32>) -> tensor<5x?x10xf32>
```

---

### hir.vcos — 逐元余弦

#### TableGen 定义

```tablegen
def VCosOp : HIVM_ElementwiseUnaryOp<"vcos",
    [SameOperandsElementType,
     VectorOnlyTrait<0>, NoLibraryFunctionTrait
    ]>
```

源码参考：[HIVMVectorOps.td#L329-L339](https://gitcode.com/Ascend/AscendNPU-IR/tree/master/bishengir/include/bishengir/Dialect/HIVM/IR/HIVMVectorOps.td#L329-L339)

#### 数据类型约束

| 操作 | 支持的元素类型 | 约束来源 |
|------|--------------|----------|
| vcos | 浮点类型 | SameOperandsElementType（无显式 OperElemTypeConstraints） |

#### IR 示例

```mlir
%result = hivm.hir.vcos ins(%src : tensor<5x?x10xf32>) outs(%dst : tensor<5x?x10xf32>) -> tensor<5x?x10xf32>
```

---

### hir.verf — 逐元误差函数

#### TableGen 定义

```tablegen
def VErfOp : HIVM_ElementwiseUnaryOp<"verf",
    [SameOperandsElementType,
     VectorOnlyTrait<0>, NoLibraryFunctionTrait
    ]>
```

源码参考：[HIVMVectorOps.td#L341-L351](https://gitcode.com/Ascend/AscendNPU-IR/tree/master/bishengir/include/bishengir/Dialect/HIVM/IR/HIVMVectorOps.td#L341-L351)

#### 数据类型约束

| 操作 | 支持的元素类型 | 约束来源 |
|------|--------------|----------|
| verf | 浮点类型 | SameOperandsElementType（无显式 OperElemTypeConstraints） |

#### IR 示例

```mlir
%result = hivm.hir.verf ins(%src : tensor<5x?x10xf32>) outs(%dst : tensor<5x?x10xf32>) -> tensor<5x?x10xf32>

hivm.hir.verf ins(%src : memref<5x?x10xf32>) outs(%dst : memref<5x?x10xf32>)
```

---

### hir.vrec — 逐元倒数

#### TableGen 定义

```tablegen
def VRecOp : HIVM_ElementwiseUnaryOp<"vrec",
    [SameOperandsElementType,
     VectorOnlyTrait<0>, StaticMaxRankTrait<3>,
     OperElemTypeConstraints<[0], [F16, F32]>,
     DeclareOpInterfaceMethods<ExtraBufferOpInterface,
       ["getExtraBufferSize"]>,
     BroadcastableOTF
    ]>
```

源码参考：[HIVMVectorOps.td#L353-L384](https://gitcode.com/Ascend/AscendNPU-IR/tree/master/bishengir/include/bishengir/Dialect/HIVM/IR/HIVMVectorOps.td#L353-L384)

#### 数据类型约束

| 操作 | 支持的元素类型 | 约束来源 |
|------|--------------|----------|
| vrec | F16, F32 | OperElemTypeConstraints<[0], [F16, F32]> |

#### IR 示例

```mlir
%result = hivm.hir.vrec ins(%src : tensor<5x?x10xf32>) outs(%dst : tensor<5x?x10xf32>) -> tensor<5x?x10xf32>

hivm.hir.vrec ins(%src : memref<5x?x10xf32>) outs(%dst : memref<5x?x10xf32>)
```

---

### hir.vnot — 逐元按位取反

#### TableGen 定义

```tablegen
def VNotOp : HIVM_ElementwiseUnaryOp<"vnot",
    [SameOperandsElementType,
     VectorOnlyTrait<0>, StaticMaxRankTrait<3>,
     OperElemTypeConstraints<[0],
       [I1, I8, UI8, I16, UI16, I32, UI32, I64, UI64, F16, BF16, F32]>,
     DeclareOpInterfaceMethods<ExtraBufferOpInterface,
       ["getExtraBufferSize"]>,
     BroadcastableOTF
    ]>
```

源码参考：[HIVMVectorOps.td#L386-L418](https://gitcode.com/Ascend/AscendNPU-IR/tree/master/bishengir/include/bishengir/Dialect/HIVM/IR/HIVMVectorOps.td#L386-L418)

#### 数据类型约束

| 操作 | 支持的元素类型 | 约束来源 |
|------|--------------|----------|
| vnot | I1, I8, UI8, I16, UI16, I32, UI32, I64, UI64, F16, BF16, F32 | OperElemTypeConstraints<[0], [I1, I8, UI8, I16, UI16, I32, UI32, I64, UI64, F16, BF16, F32]> |

注意：vnot 支持浮点类型的按位取反（即对浮点数的位模式取反），这在某些数值算法中有特殊用途。

#### IR 示例

```mlir
hivm.hir.vnot ins(%src : memref<5x1x10xi32>) outs(%dst : memref<5x?x10xi32>) broadcast = [1]
```

降级到 HFusion：`hfusion.elemwise_unary {fun = #hfusion.unary_fn<vnot>}`

## 数据类型约束汇总

| 操作 | 支持的元素类型 | 最大 Rank | OTF 广播 | OTF 转置 | Extra Buffer | 标量实现 | 向量化 |
|------|--------------|-----------|---------|---------|-------------|---------|--------|
| vexp | F16, F32 | 3 | 是 | 是 | 是 | - | - |
| vabs | F16, F32, I8, I16, I32, I64 | 3 | 是 | 是 | 是 | 是 | 是 |
| vln | F16, F32 | 3 | 是 | 是 | 是 | - | - |
| vrelu | F16, F32, I32 | 3 | 是 | 是 | 是 | - | - |
| vrsqrt | F16, F32 | 3 | 是 | 是 | 是 | - | - |
| vsqrt | F16, F32 | 3 | 是 | 是 | 是 | - | - |
| vtanh | 浮点 | - | - | - | - | - | - |
| vsin | 浮点 | - | - | - | - | - | - |
| vcos | 浮点 | - | - | - | - | - | - |
| verf | 浮点 | - | - | - | - | - | - |
| vrec | F16, F32 | 3 | 是 | 是 | 是 | - | - |
| vnot | I1, I8, UI8, I16, UI16, I32, UI32, I64, UI64, F16, BF16, F32 | 3 | 是 | 是 | 是 | - | - |

## IR 层约束与验证

1. **元素类型一致性**：所有一元操作要求输入和输出具有相同的元素类型（`SameOperandsElementType`）
2. **Rank 一致性**：输入和输出必须具有相同的 rank（`HIVMOpSameOperandsAndResultRank`）
3. **VectorOnly 约束**：第一个输入操作数（$src[0]）必须为向量类型
4. **OTF 广播约束**：
   - broadcast 数组中的维度必须唯一
   - 所有广播维度 d 满足 `0 <= d < rank(dst)`
   - 广播维度上 src 的 size 为 1 或等于 dst 的 size
5. **OTF 转置约束**：
   - transpose 必须是 `range(rank(dst))` 的排列
   - `transpose[rank(dst) - 1] = rank(dst) - 1`（最后一维不可转置）
6. **temp_buffer**：需要 ExtraBufferOpInterface 的操作在硬件执行时需要额外的临时存储空间

## 与其他 IR 操作的关系

| HIVM 操作 | 上游 linalg 降级 | HFusion 降级 |
|-----------|-----------------|-------------|
| vexp | linalg.exp | - |
| vabs | linalg.abs | - |
| vln | linalg.log | - |
| vrelu | - | hfusion.elemwise_unary {fun = relu} |
| vrsqrt | linalg.rsqrt | - |
| vsqrt | linalg.sqrt | - |
| vtanh | linalg.tanh | - |
| vrec | linalg.reciprocal | - |
| verf | linalg.erf | - |
| vnot | - | hfusion.elemwise_unary {fun = vnot} |

## 常见问题

**Q: vtanh, vsin, vcos, verf 为什么没有显式的 OperElemTypeConstraints？**
A: 这些操作使用 `NoLibraryFunctionTrait`，表示不通过预编译库函数实现。它们的类型约束由 `SameOperandsElementType` 隐式保证（输入输出同类型），具体支持的类型由硬件实现决定。

**Q: vnot 为什么支持浮点类型？**
A: vnot 执行的是按位取反操作，对浮点数的位模式取反在某些数值算法中有特殊用途（如快速生成 NaN 掩码等）。

**Q: 什么是一元操作的 temp_buffer？**
A: 部分一元操作（如 vexp, vln, vrsqrt 等）在硬件执行时需要额外的临时存储空间来保存中间计算结果。temp_buffer 参数用于提供这个空间。

## 相关文档

- Python API：docs_triton_ascend 中的 `tl.math` 模块文档
- 源码参考：
  - [HIVMVectorOps.td - Unary Ops](https://gitcode.com/Ascend/AscendNPU-IR/tree/master/bishengir/include/bishengir/Dialect/HIVM/IR/HIVMVectorOps.td#L105-L418)
  - [HIVMTraits.td](https://gitcode.com/Ascend/AscendNPU-IR/tree/master/bishengir/include/bishengir/Dialect/HIVM/IR/HIVMTraits.td)
  - [convert-hivm-to-upstream.mlir](https://gitcode.com/Ascend/AscendNPU-IR/tree/master/bishengir/test/ExecutionEngine/convert-hivm-to-upstream.mlir)
