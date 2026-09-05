# HIVM 数据搬移操作

> 关键词：HIVM, vbrc, vtranspose, vinterleave, vdeinterleave, vflip, vpad, vconcat, vgather

## 概述

HIVM 数据搬移操作负责在向量级别进行数据的重排、广播、拼接、收集等操作。这些操作不执行计算，而是改变数据的布局和形状，是向量计算的重要辅助操作。

> Python API 对应：`tl.broadcast_to()`, `tl.trans()`, `tl.flip()`, `tl.pad()`, `tl.concat()`, `tl.gather()` 等。

## hir.vbrc — 向量广播

### TableGen 定义

```tablegen
def VBrcOp : HIVM_VectorOp<"vbrc",
    [SameOperandsElementType,
     OperElemTypeConstraints<
         [0], [I8, UI8, I16, F16, UI16, I32, F8E4M3FN, F8E5M2, F32, UI32, BF16, I64, UI64, I1]>,
     DeclareOpInterfaceMethods<ExtraBufferOpInterface>,
     DeclareOpInterfaceMethods<HIVMStructuredOpInterface,
                               ["getIteratorTypesArray", "getIndexingMaps"]>,
     DeclareOpInterfaceMethods<HIVMInferCoreTypeInterface, ["inferCoreType"]>,
     UniformReassociationFlattenTrait,
     CollapsibleConsecutiveTargetDimsTrait,
     DeclareOpInterfaceMethods<FlattenInterface,
       ["getLimitedAxes", "adjustTargetDimensions"]>,
     InferMaxRankTrait,
     DeclareOpInterfaceMethods<LibraryFunctionOpInterface,
       ["inferOpLibraryMaxRank"]>,
     DeclareOpInterfaceMethods<BiShengIRAggregatedOpInterface,
       ["decomposeOperation","getDecomposePhase"]>,
    ], []> {
  let summary = "Vector Broadcast Op";
  let arguments = (ins AnyType:$src,
                       TensorOrMemref:$dst,
                       Optional<AnyMemRef>:$temp_buffer,
                       DefaultValuedAttr<DenseI64ArrayAttr, "{}">:$broadcast_dims
  );
}
```

源码参考：[HIVMVectorOps.td#L1056-L1123](https://gitcode.com/Ascend/AscendNPU-IR/tree/master/bishengir/include/bishengir/Dialect/HIVM/IR/HIVMVectorOps.td#L1056-L1123)

### 参数说明

| 参数 | 类型 | 必选 | 说明 | 约束 |
|------|------|------|------|------|
| $src | AnyType | 是 | 输入（标量或向量） | 元素类型见下方约束表 |
| $dst | TensorOrMemref | 是 | 输出向量 | 与 src 相同元素类型 |
| $temp_buffer | Optional\<AnyMemRef\> | 否 | 临时缓冲区 | - |
| $broadcast_dims | DenseI64ArrayAttr (默认 {}) | 否 | 广播维度数组 | 标量输入时必须为空 |

### 数据类型约束

| 操作 | 支持的元素类型 | 约束来源 |
|------|--------------|----------|
| vbrc | I8, UI8, I16, F16, UI16, I32, F8E4M3FN, F8E5M2, F32, UI32, BF16, I64, UI64, I1 | OperElemTypeConstraints<[0], [...]> |

### 约束规则

1. 输入和输出必须具有相同的 rank 和元素类型
2. 对于输入向量，被广播维度的 size 必须为 1
3. 向量输入时 broadcast_dims 不能为空
4. 标量输入时 broadcast_dims 必须为空
5. broadcast_dims 中的索引必须在 `[0, rank(src))` 范围内
6. I1 类型的输出尾轴需要对齐到 16

### IR 示例

```mlir
%result = hivm.hir.vbrc ins(%scalar : i32) outs(%dst : tensor<23x77xi32>) -> tensor<23x77xi32>

%result = hivm.hir.vbrc ins(%src : tensor<1xi32>) outs(%dst : tensor<?xi32>) broadcast_dims = [0] -> tensor<?xi32>

hivm.hir.vbrc ins(%scalar : f32) outs(%dst : memref<?x?xf32>)
```

---

## hir.vtranspose — 维度转置

### TableGen 定义

```tablegen
def VTransposeOp : HIVM_VectorOp<"vtranspose",
    [OperElemTypeConstraints<[0], [AnyI8, AnyI16, AnyI32, F16, BF16, F32, I64, UI64, F8E4M3FN, F8E5M2]>,
     DeclareOpInterfaceMethods<HIVMStructuredOpInterface,
       ["getIteratorTypesArray", "setIteratorTypesArray"]>,
     InferMaxRankTrait,
     UniformReassociationFlattenTrait,
     DeclareOpInterfaceMethods<FlattenInterface,
       ["getLimitedAxes", "adjustTargetDimensions"]>,
     DeclareOpInterfaceMethods<ExtraBufferOpInterface>,
    ]> {
  let summary = "Vector Transpose Op";
  let arguments = (ins TensorOrMemref:$src,
                       TensorOrMemref:$dst,
                       Optional<AnyMemRef>:$temp_buffer,
                       DefaultValuedAttr<DenseI64ArrayAttr, "{}">:$permutation
  );
}
```

源码参考：[HIVMVectorOps.td#L1222-L1272](https://gitcode.com/Ascend/AscendNPU-IR/tree/master/bishengir/include/bishengir/Dialect/HIVM/IR/HIVMVectorOps.td#L1222-L1272)

### 参数说明

| 参数 | 类型 | 必选 | 说明 | 约束 |
|------|------|------|------|------|
| $src | TensorOrMemref | 是 | 输入向量 | 元素类型见下方约束表 |
| $dst | TensorOrMemref | 是 | 输出向量 | 与 src 相同 rank 和元素类型 |
| $temp_buffer | Optional\<AnyMemRef\> | 否 | 临时缓冲区 | - |
| $permutation | DenseI64ArrayAttr (默认 {}) | 否 | 维度排列 | 必须是 range(rank) 的排列 |

### 语义

```
dim(dst, i) = dim(src, permutation[i])
```

### 数据类型约束

| 操作 | 支持的元素类型 | 约束来源 |
|------|--------------|----------|
| vtranspose | AnyI8, AnyI16, AnyI32, F16, BF16, F32, I64, UI64, F8E4M3FN, F8E5M2 | OperElemTypeConstraints<[0], [...]> |

### IR 示例

```mlir
%result = hivm.hir.vtranspose ins(%src : tensor<32x8xf32>) outs(%dst : tensor<8x32xf32>) permutation = [1, 0] -> tensor<8x32xf32>

hivm.hir.vtranspose ins(%src : memref<32x8xf32>) outs(%dst : memref<8x32xf32>) permutation = [1, 0]

%result = hivm.hir.vtranspose ins(%src : tensor<?x5x10xf32>) outs(%dst : tensor<5x?x10xf32>) permutation = [1, 0, 2] -> tensor<5x?x10xf32>
```

---

## hir.vinterleave — 交错合并

### TableGen 定义

```tablegen
def VInterleaveOp : HIVM_VectorOp<"vinterleave",
    [SameOperandsElementType, AttrSizedOperandSegments,
     HIVMOpSameOperandsAndResultRank, StaticMaxRankTrait<1>,
     OperElemTypeConstraints<[0],
       [I16, F16, UI16, I32, F32, UI32, BF16, I64, UI64, F8E4M3FN, F8E5M2]>,
     DeclareOpInterfaceMethods<ExtraBufferOpInterface>,
     DeclareOpInterfaceMethods<HIVMStructuredOpInterface,
       ["getIteratorTypesArray"]>,
     DeclareOpInterfaceMethods<ImplByScalarOpInterface>,
     UniformReassociationFlattenTrait,
     DeclareOpInterfaceMethods<FlattenInterface,
       ["getLimitedAxes"]>,
    ]> {
  let summary = "Vetor Interleave Op";
  let arguments = (ins Variadic<AnyType>:$src,
                       TensorOrMemref:$dst,
                       Optional<AnyMemRef>:$temp_buffer,
                       DefaultValuedAttr<I64Attr, "2">:$interleave_channel_nums
  );
}
```

源码参考：[HIVMVectorOps.td#L1336-L1377](https://gitcode.com/Ascend/AscendNPU-IR/tree/master/bishengir/include/bishengir/Dialect/HIVM/IR/HIVMVectorOps.td#L1336-L1377)

### 参数说明

| 参数 | 类型 | 必选 | 默认值 | 说明 | 约束 |
|------|------|------|--------|------|------|
| $src | Variadic\<AnyType\> | 是 | - | 输入向量列表 | 所有向量形状相同 |
| $dst | TensorOrMemref | 是 | - | 输出向量 | 最后一维 = src 最后一维 * channel_nums |
| $temp_buffer | Optional\<AnyMemRef\> | 否 | - | 临时缓冲区 | - |
| $interleave_channel_nums | I64Attr | 否 | 2 | 交错通道数 | 必须等于 $src 的数量 |

### 语义

将 N 个张量沿最后一维交错合并。例如，两个张量 `[a0, a1, a2]` 和 `[b0, b1, b2]` 交错后为 `[a0, b0, a1, b1, a2, b2]`。

### IR 示例

```mlir
%result = hivm.hir.vinterleave ins(%a, %b : tensor<2x16xf32>, tensor<2x16xf32>) outs(%c : tensor<2x32xf32>) interleave_channel_nums = 2 -> tensor<2x32xf32>
```

降级到 HFusion：`hfusion.interleave`

---

## hir.vdeinterleave — 交错分离

### TableGen 定义

```tablegen
def VDeinterleaveOp : HIVM_VectorOp<"vdeinterleave",
    [SameOperandsElementType, HIVMOpSameOperandsAndResultRank,
     OperElemTypeConstraints<[0],
       [I8, I16, F16, UI16, I32, F32, UI32, BF16, I64, UI64]>,
     DeclareOpInterfaceMethods<HIVMStructuredOpInterface,
       ["getIteratorTypesArray"]>,
     InferMaxRankTrait,
     DeclareOpInterfaceMethods<BiShengIRAggregatedOpInterface,
       ["decomposeOperation","getDecomposePhase"]>,
     DeclareOpInterfaceMethods<ImplByScalarOpInterface>,
     UniformReassociationFlattenTrait,
     DeclareOpInterfaceMethods<FlattenInterface,
       ["getLimitedAxes"]>,
    ]> {
  let arguments = (ins TensorOrMemref:$src,
                       Variadic<TensorOrMemref>:$dst,
                       DefaultValuedOptionalAttr<I64Attr, "2">:$channel_num,
                       DefaultValuedOptionalAttr<HIVM_DeinterleaveModeAttr,
                         "DeinterleaveMode::ALL_CHANNELS">:$index_mode
  );
}
```

源码参考：[HIVMVectorOps.td#L1383-L1423](https://gitcode.com/Ascend/AscendNPU-IR/tree/master/bishengir/include/bishengir/Dialect/HIVM/IR/HIVMVectorOps.td#L1383-L1423)

### 参数说明

| 参数 | 类型 | 必选 | 默认值 | 说明 |
|------|------|------|--------|------|
| $src | TensorOrMemref | 是 | - | 输入向量 |
| $dst | Variadic\<TensorOrMemref\> | 是 | - | 输出向量列表 |
| $channel_num | I64Attr | 否 | 2 | 通道数 |
| $index_mode | HIVM_DeinterleaveModeAttr | 否 | ALL_CHANNELS | 分离模式 |

### index_mode 属性

| 枚举值 | 值 | 说明 |
|--------|---|------|
| CHANNEL_0 | 0 | 仅提取通道 0 |
| CHANNEL_1 | 1 | 仅提取通道 1 |
| ALL_CHANNELS | 999 | 提取所有通道 |

### IR 示例

```mlir
%result = hivm.hir.vdeinterleave ins(%src : tensor<32xf32>) outs(%dst : tensor<16xf32>) index_mode = <CHANNEL_0> -> tensor<16xf32>
```

降级到 HFusion：`hfusion.deinterleave %src channel<0>`

---

## hir.vflip — 维度翻转

### TableGen 定义

```tablegen
def VFlipOp : HIVM_VectorOp<"vflip",
    [SameOperandsElementType, StaticMaxRankTrait<1>,
     OperElemTypeConstraints<[0, 1], [I8, UI8, I16, I32, UI16, UI32, I64, UI64, F16, F32, BF16]>,
     DeclareOpInterfaceMethods<HIVMStructuredOpInterface,
       ["getIteratorTypesArray"]>,
     UniformReassociationFlattenTrait,
     DeclareOpInterfaceMethods<FlattenInterface,
                               ["getLimitedAxes", "adjustTargetDimensions"]>
    ]> {
  let arguments = (ins TensorOrMemref:$src,
                       TensorOrMemref:$dst,
                       I64Attr:$flip_axis
  );
}
```

源码参考：[HIVMVectorOps.td#L1429-L1457](https://gitcode.com/Ascend/AscendNPU-IR/tree/master/bishengir/include/bishengir/Dialect/HIVM/IR/HIVMVectorOps.td#L1429-L1457)

### 参数说明

| 参数 | 类型 | 必选 | 说明 |
|------|------|------|------|
| $src | TensorOrMemref | 是 | 输入向量 |
| $dst | TensorOrMemref | 是 | 输出向量 |
| $flip_axis | I64Attr | 是 | 翻转的轴 |

### IR 示例

```mlir
%result = hivm.hir.vflip ins(%src : tensor<10xf32>) outs(%dst : tensor<10xf32>) flip_axis = 0 -> tensor<10xf32>
```

---

## hir.vpad — 填充

### TableGen 定义

```tablegen
def VPadOp : HIVM_VectorOp<"vpad",
    [HIVMOpSameOperandsAndResultRank,
     AttrSizedOperandSegments, NoLibraryFunctionTrait,
     DeclareOpInterfaceMethods<HIVMStructuredOpInterface,
       ["getIteratorTypesArray"]>,
     UniformReassociationFlattenTrait,
     DeclareOpInterfaceMethods<FlattenInterface,
       ["adjustTargetDimensions", "getLimitedAxes"]>,
     DeclareOpInterfaceMethods<BiShengIRAggregatedOpInterface,
       ["decomposeOperation", "getDecomposePhase"]>]> {
  let arguments = (ins TensorOrMemref:$src,
                       TensorOrMemref:$dst,
                       AnyType:$pad_value,
                       Variadic<Index>:$low,
                       Variadic<Index>:$high,
                       DenseI64ArrayAttr:$static_low,
                       DenseI64ArrayAttr:$static_high
  );
}
```

源码参考：[HIVMVectorOps.td#L1502-L1559](https://gitcode.com/Ascend/AscendNPU-IR/tree/master/bishengir/include/bishengir/Dialect/HIVM/IR/HIVMVectorOps.td#L1502-L1559)

### 参数说明

| 参数 | 类型 | 必选 | 说明 |
|------|------|------|------|
| $src | TensorOrMemref | 是 | 输入张量 |
| $dst | TensorOrMemref | 是 | 输出张量（bufferization） |
| $pad_value | AnyType | 是 | 填充值 |
| $low | Variadic\<Index\> | 是 | 每维起始方向的填充长度 |
| $high | Variadic\<Index\> | 是 | 每维末尾方向的填充长度 |
| $static_low | DenseI64ArrayAttr | 是 | 静态起始填充长度 |
| $static_high | DenseI64ArrayAttr | 是 | 静态末尾填充长度 |

### IR 示例

```mlir
hivm.hir.vpad ins(%src : tensor<2x16xf32>) outs(%dst : tensor<?x16xf32>)
              low[%first_dim_low, 0] high[%first_dim_high, 0]
              pad_value %pad_value : f32
              -> tensor<?x16xf32>
```

---

## hir.vconcat — 拼接

### TableGen 定义

```tablegen
def VConcatOp : HIVM_VectorOp<"vconcat",
    [SameOperandsElementType, NoLibraryFunctionTrait,
     DeclareOpInterfaceMethods<HIVMStructuredOpInterface,
       ["getIteratorTypesArray"]>,
     DeclareOpInterfaceMethods<BiShengIRAggregatedOpInterface,
       ["decomposeOperation","getDecomposePhase"]>,
     UniformReassociationFlattenTrait,
     DeclareOpInterfaceMethods<FlattenInterface,
       ["adjustTargetDimensions", "getLimitedAxes"]>
    ]> {
  let arguments = (ins I64Attr:$dim,
                       Variadic<AnyType>:$src,
                       TensorOrMemref:$dst
  );
}
```

源码参考：[HIVMVectorOps.td#L1565-L1604](https://gitcode.com/Ascend/AscendNPU-IR/tree/master/bishengir/include/bishengir/Dialect/HIVM/IR/HIVMVectorOps.td#L1565-L1604)

### 参数说明

| 参数 | 类型 | 必选 | 说明 |
|------|------|------|------|
| $dim | I64Attr | 是 | 拼接维度 |
| $src | Variadic\<AnyType\> | 是 | 输入张量列表 |
| $dst | TensorOrMemref | 是 | 输出张量 |

### 语义

沿指定维度拼接多个张量。拼接维度的大小等于所有输入在该维度大小之和，其他维度大小必须相同。

### IR 示例

```mlir
%result = hivm.hir.vconcat dim(0) ins(%a, %b : tensor<5x?x10xf32>, tensor<?x?x10xf32>) outs(%c : tensor<?x?x10xf32>) -> tensor<?x?x10xf32>

hivm.hir.vconcat dim(1) ins(%0, %1 : tensor<136x2048xf32>, tensor<136x2048xf32>) outs(%2 : tensor<136x4096xf32>) -> tensor<136x4096xf32>
```

降级到上游：`tensor.concat`

---

## hir.vgather — 按索引收集

### TableGen 定义

```tablegen
def VGatherOp : HIVM_VectorOp<"vgather",
    [HIVMOpSameOperandsAndResultRank,
     StaticMaxRankTrait<1>,
     OperElemTypeConstraints<[0], [I1, I8, I16, UI16, I32, UI32, F16, BF16, F32, F8E4M3FN, F8E5M2]>,
     OperElemTypeConstraints<[1], [I32]>,
     DeclareOpInterfaceMethods<ExtraBufferOpInterface>,
     DeclareOpInterfaceMethods<HIVMStructuredOpInterface,
       ["getIteratorTypesArray"]>,
     UniformReassociationFlattenTrait,
     DeclareOpInterfaceMethods<FlattenInterface,
       ["getLimitedAxes"]>
    ]> {
  let arguments = (ins TensorOrMemref:$src,
                       TensorOrMemref:$indices,
                       TensorOrMemref:$dst,
                       Optional<AnyMemRef>:$temp_buffer
  );
}
```

源码参考：[HIVMVectorOps.td#L1610-L1654](https://gitcode.com/Ascend/AscendNPU-IR/tree/master/bishengir/include/bishengir/Dialect/HIVM/IR/HIVMVectorOps.td#L1610-L1654)

### 参数说明

| 参数 | 类型 | 必选 | 说明 | 约束 |
|------|------|------|------|------|
| $src | TensorOrMemref | 是 | 源数据 | 元素类型见上方约束 |
| $indices | TensorOrMemref | 是 | 索引向量 | 元素类型 I32 |
| $dst | TensorOrMemref | 是 | 输出向量 | 与 src 相同元素类型 |
| $temp_buffer | Optional\<AnyMemRef\> | 否 | 临时缓冲区 | - |

### 语义

根据索引向量从源数据中收集元素，存储到输出向量中。收集轴为最后一维。

```
dst[i] = src[indices[i]]
```

### IR 示例

```mlir
%result = hivm.hir.vgather ins(%src : tensor<100xf32>) indices(%idx : tensor<10xi32>) outs(%dst : tensor<10xf32>) -> tensor<10xf32>
```

## 数据类型约束汇总

| 操作 | 支持的元素类型 | 最大 Rank |
|------|--------------|-----------|
| vbrc | I8, UI8, I16, F16, UI16, I32, F8E4M3FN, F8E5M2, F32, UI32, BF16, I64, UI64, I1 | 推断 |
| vtranspose | AnyI8, AnyI16, AnyI32, F16, BF16, F32, I64, UI64, F8E4M3FN, F8E5M2 | 推断 |
| vinterleave | I16, F16, UI16, I32, F32, UI32, BF16, I64, UI64, F8E4M3FN, F8E5M2 | 1 |
| vdeinterleave | I8, I16, F16, UI16, I32, F32, UI32, BF16, I64, UI64 | 推断 |
| vflip | I8, UI8, I16, I32, UI16, UI32, I64, UI64, F16, F32, BF16 | 1 |
| vpad | 无显式约束（由 pad_value 决定） | - |
| vconcat | SameOperandsElementType | - |
| vgather | 数据: I1, I8, I16, UI16, I32, UI32, F16, BF16, F32, F8E4M3FN, F8E5M2; 索引: I32 | 1 |

## 常见问题

**Q: vbrc 和 Elementwise 操作的 broadcast 属性有什么区别？**
A: vbrc 是独立的广播操作，生成新的张量。Elementwise 操作的 broadcast 属性是 OTF（On-The-Fly）广播，在计算的同时进行广播，不需要额外的广播步骤。

**Q: vinterleave 和 vconcat 有什么区别？**
A: vinterleave 是沿最后一维交错合并多个张量（如 `[a0,b0,a1,b1,...]`），而 vconcat 是沿指定维度简单拼接（如 `[a0,a1,...,b0,b1,...]`）。

**Q: vgather 的索引范围有约束吗？**
A: 索引值必须在源数据的最后一维大小范围内，即 `0 <= indices[i] < dim(src, last_dim)`。越界访问的行为未定义。

## 相关文档

- Python API：docs_triton_ascend 中的数据操作文档
- 源码参考：
  - [HIVMVectorOps.td - Data Movement Ops](https://gitcode.com/Ascend/AscendNPU-IR/tree/master/bishengir/include/bishengir/Dialect/HIVM/IR/HIVMVectorOps.td#L1056-L1654)
  - [convert-hivm-to-upstream.mlir](https://gitcode.com/Ascend/AscendNPU-IR/tree/master/bishengir/test/ExecutionEngine/convert-hivm-to-upstream.mlir)
