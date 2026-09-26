# HIVM 特殊操作

> 关键词：HIVM, varange, vmulextended, vmulext, arange, mul_extended

## 概述

HIVM 特殊操作包括范围序列生成（`hir.varange`）、扩展乘法（`hir.vmulextended`）和乘法高32位（`hir.vmulext`）。这些操作不属于标准的逐元运算分类，但在特定场景中不可或缺。

> Python API 对应：`tl.arange()`, 以及 Triton 中的扩展精度运算。

## hir.varange — 范围序列生成

### TableGen 定义

```tablegen
def VArangeOp
    : HIVM_VectorOp<
          "varange", [AttrSizedOperandSegments, StaticMaxRankTrait<3>,
                      OperElemTypeConstraints<[0], [I16, I32, F16, F32, I64]>,
                      DeclareOpInterfaceMethods<HIVMStructuredOpInterface,
                                                ["getIteratorTypesArray"]>,
                      DeclareOpInterfaceMethods<FlattenInterface,
                                                ["getLimitedAxes"]>]> {
  let summary = "Vector Arange Op";
  let description = [{
    Fill a vector with range 0,1,2... based on strides and offset.
    e.g. offset = 1, strides = [1, 2], tensor/memref shape = [2x4xi32],
    the result is [[1, 3, 5, 7,
                    2, 4, 6, 8]].

    Constraints:
      1. Must have at least one stride.
      2. Default offset is 0.

    Examples:
    ```mlir
    hivm.hir.varange offset[%o] strides[%s0, %s1] outs(%dst : memref<32xf32>)
    %result = hivm.hir.varange offset[%o] strides[%s0, %s1] outs(%dst : tensor<32xf32>)
                                -> tensor<32xf32>
    ```
  }];
  let arguments = (ins TensorOrMemref:$dst,
                       Optional<Index>:$offset,
                       Variadic<Index>:$strides
  );
  let results = (outs Optional<AnyRankedTensor>:$result);
  let assemblyFormat = [{
    attr-dict
    (`offset` `[` $offset^ `]`)?
    `strides` `[` $strides `]`
    `outs` `(` $dst `:` type($dst) `)`
    (`->` type($result)^)?
  }];
}
```

源码参考：[HIVMVectorOps.td#L1278-L1330](https://gitcode.com/Ascend/AscendNPU-IR/tree/master/bishengir/include/bishengir/Dialect/HIVM/IR/HIVMVectorOps.td#L1278-L1330)

### 参数说明

| 参数 | 类型 | 必选 | 说明 | 约束 |
|------|------|------|------|------|
| $dst | TensorOrMemref | 是 | 输出向量（同时指定形状） | 元素类型 I16/I32/F16/F32/I64 |
| $offset | Optional\<Index\> | 否 | 起始偏移量 | 默认为 0 |
| $strides | Variadic\<Index\> | 是 | 每维步长 | 至少一个步长 |

### 数据类型约束

| 操作 | 支持的元素类型 | 约束来源 |
|------|--------------|----------|
| varange | I16, I32, F16, F32, I64 | OperElemTypeConstraints<[0], [I16, I32, F16, F32, I64]> |

### 语义

varange 根据偏移量和步长生成一个范围序列，填充到输出张量中。

对于多维张量，每个维度有一个步长。计算公式：

```
value[i0, i1, ..., in] = offset + i0 * strides[0] + i1 * strides[1] + ... + in * strides[n]
```

示例（来自 TableGen 描述）：
- offset = 1, strides = [1, 2], shape = [2x4]
- 结果：[[1, 3, 5, 7], [2, 4, 6, 8]]

### IR 示例

```mlir
%result = hivm.hir.varange offset[] strides[%c0, %c3, %c2] outs(%dst : tensor<5x?x10xi64>) -> tensor<5x?x10xi64>

hivm.hir.varange offset[%c3] strides[%c1, %c1, %c1] outs(%dst : memref<5x?x10xi32>)

%result = hivm.hir.varange strides[%c1] outs(%dst : tensor<1011xi32>) -> tensor<1011xi32>
```

降级到 HFusion：`hfusion.arange`

---

## hir.vmulextended — 扩展乘法

### TableGen 定义

```tablegen
def VMulextendedOp : HIVM_VectorOp<"vmulextended",
    [AttrSizedOperandSegments, HIVMOpSameOperandsAndResultRank,
     StaticMaxRankTrait<1>, OperElemTypeConstraints<[0], [I16]>,
     DeclareOpInterfaceMethods<ExtraBufferOpInterface>,
     DeclareOpInterfaceMethods<HIVMStructuredOpInterface,
       ["getIteratorTypesArray",
       ]>,
     UniformReassociationFlattenTrait,
     DeclareOpInterfaceMethods<FlattenInterface,
                               ["getLimitedAxes"]>
     ]> {
  let summary = "Vector Mulextended Op";
  let description = [{
    Do vmul on two tensors. Get both high and low 16-bits.
  }];
  let arguments = (ins Variadic<TensorOrMemref>:$src,
                       Variadic<TensorOrMemref>:$dst,
                       Optional<AnyMemRef>:$temp_buffer
  );
}
```

源码参考：[HIVMVectorOps.td#L1463-L1496](https://gitcode.com/Ascend/AscendNPU-IR/tree/master/bishengir/include/bishengir/Dialect/HIVM/IR/HIVMVectorOps.td#L1463-L1496)

### 参数说明

| 参数 | 类型 | 必选 | 说明 | 约束 |
|------|------|------|------|------|
| $src | Variadic\<TensorOrMemref\> | 是 | 两个输入向量 | 元素类型 I16 |
| $dst | Variadic\<TensorOrMemref\> | 是 | 两个输出向量（高位和低位） | - |
| $temp_buffer | Optional\<AnyMemRef\> | 否 | 临时缓冲区 | - |

### 数据类型约束

| 操作 | 支持的元素类型 | 约束来源 |
|------|--------------|----------|
| vmulextended | I16 | OperElemTypeConstraints<[0], [I16]> |

### 语义

vmulextended 对两个 I16 向量执行乘法，产生两个 I16 输出：乘法结果的高 16 位和低 16 位。

```
product = src0[i] * src1[i]   // 32-bit result
dst_high[i] = (product >> 16) & 0xFFFF  // 高 16 位
dst_low[i]  = product & 0xFFFF          // 低 16 位
```

### IR 示例

```mlir
%high, %low = hivm.hir.vmulextended ins(%a, %b : tensor<32xi16>, tensor<32xi16>) outs(%dh, %dl : tensor<32xi16>, tensor<32xi16>) -> tensor<32xi16>, tensor<32xi16>
```

---

## hir.vmulext — 乘法高32位

### TableGen 定义

```tablegen
def VMulExtOp : HIVM_ElementwiseBinaryOp<"vmulext",
    [SameOperandsElementType, StaticMaxRankTrait<3>,
     OperElemTypeConstraints<[0, 1], [I32]>,
     VectorOnlyTrait<0>,
    DeclareOpInterfaceMethods<ImplByScalarOpInterface>
    ]> {
  let summary = [{
    Elementwise Binary Vector Multiplication that Calculates
    the Most Significant 32-bits.
  }];
  let description = baseClassDescription # [{
    Additional constraints:
      1. The input/init operands and result have the same element type.
      2. Support Vector-Vector operation.
  }];
}
```

源码参考：[HIVMVectorOps.td#L578-L594](https://gitcode.com/Ascend/AscendNPU-IR/tree/master/bishengir/include/bishengir/Dialect/HIVM/IR/HIVMVectorOps.td#L578-L594)

### 参数说明

| 参数 | 类型 | 必选 | 说明 | 约束 |
|------|------|------|------|------|
| $src[0] | AnyType | 是 | 第一个输入向量 | VectorOnly, 元素类型 I32 |
| $src[1] | AnyType | 是 | 第二个输入向量/标量 | 元素类型 I32 |
| $dst | Variadic\<AnyShaped\> | 是 | 输出向量 | 元素类型 I32 |

注意：vmulext 使用默认的 ElementwiseNaryOp 参数（无 temp_buffer, 无 OTF 广播/转置）。

### 数据类型约束

| 操作 | 支持的元素类型 | 约束来源 |
|------|--------------|----------|
| vmulext | I32 | OperElemTypeConstraints<[0, 1], [I32]> |

### 语义

vmulext 计算两个 I32 值乘积的最高有效 32 位。这等价于 64 位乘法结果的高 32 位。

```
product_64 = src0[i] * src1[i]   // 64-bit result
dst[i] = (product_64 >> 32) & 0xFFFFFFFF  // 高 32 位
```

### IR 示例

```mlir
%result = hivm.hir.vmulext ins(%a, %b : tensor<32xi32>, tensor<32xi32>) outs(%dst : tensor<32xi32>) -> tensor<32xi32>
```

## vmulextended 与 vmulext 的区别

| 特性 | vmulextended | vmulext |
|------|-------------|---------|
| 输入类型 | I16 | I32 |
| 输出数量 | 2（高位 + 低位） | 1（仅高位） |
| 操作类别 | HIVM_VectorOp（独立操作） | HIVM_ElementwiseBinaryOp |
| 最大 Rank | 1 | 3 |
| 乘法宽度 | I16 * I16 → I32 | I32 * I32 → I64 |
| 输出内容 | 高 16 位 + 低 16 位 | 高 32 位 |
| 标量支持 | 否 | 是（ImplByScalarOpInterface） |

## 数据类型约束汇总

| 操作 | 支持的元素类型 | 最大 Rank | 输出数量 |
|------|--------------|-----------|---------|
| varange | I16, I32, F16, F32, I64 | 3 | 1 |
| vmulextended | I16 | 1 | 2 |
| vmulext | I32 | 3 | 1 |

## IR 层约束与验证

### varange

1. **至少一个步长**：$strides 不能为空
2. **步长数量与 rank**：步长数量应与输出张量的 rank 匹配
3. **默认偏移**：offset 可选，默认为 0
4. **hasVerifier = 1**：包含自定义验证器

### vmulextended

1. **仅支持 I16**：输入和输出元素类型均为 I16
2. **两个输出**：高位结果和低位结果
3. **最大 Rank 1**：仅支持 1 维

### vmulext

1. **仅支持 I32**：输入和输出元素类型均为 I32
2. **VectorOnly**：第一个输入必须为向量
3. **ImplByScalarOpInterface**：第二个输入可以是标量

## 与其他 IR 操作的关系

| HIVM 操作 | 上游降级 | HFusion 降级 | 说明 |
|-----------|---------|-------------|------|
| varange | - | hfusion.arange | 范围序列 |
| vmulextended | - | - | 扩展乘法（无直接上游对应） |
| vmulext | - | - | 乘法高32位（无直接上游对应） |

## 常见问题

**Q: varange 的步长数量必须等于张量的 rank 吗？**
A: 是的。每个维度对应一个步长值，因此步长数量应该等于输出张量的 rank。

**Q: varange 支持浮点步长吗？**
A: varange 的步长参数类型为 Index（整数），但输出可以是浮点类型。步长值在计算时会被转换为输出类型。

**Q: vmulextended 和 vmulext 的典型应用场景是什么？**
A: 这些操作用于实现扩展精度运算。当标准乘法的位宽不够时，可以使用 vmulextended 获取完整的乘法结果（高位+低位），或使用 vmulext 获取高位部分。这在定点数运算和量化场景中特别有用。

**Q: vmulext 为什么继承自 ElementwiseBinaryOp 而不是 VectorOp？**
A: vmulext 的语义与二元逐元操作一致（两个输入，一个输出），且支持标量操作数。将其归类为 ElementwiseBinaryOp 可以复用逐元操作的通用基础设施。

## 相关文档

- Python API：docs_triton_ascend 中的 `tl.arange()` 文档
- 源码参考：
  - [HIVMVectorOps.td - VArangeOp](https://gitcode.com/Ascend/AscendNPU-IR/tree/master/bishengir/include/bishengir/Dialect/HIVM/IR/HIVMVectorOps.td#L1278-L1330)
  - [HIVMVectorOps.td - VMulextendedOp](https://gitcode.com/Ascend/AscendNPU-IR/tree/master/bishengir/include/bishengir/Dialect/HIVM/IR/HIVMVectorOps.td#L1463-L1496)
  - [HIVMVectorOps.td - VMulExtOp](https://gitcode.com/Ascend/AscendNPU-IR/tree/master/bishengir/include/bishengir/Dialect/HIVM/IR/HIVMVectorOps.td#L578-L594)
  - [convert-hivm-to-upstream.mlir](https://gitcode.com/Ascend/AscendNPU-IR/tree/master/bishengir/test/ExecutionEngine/convert-hivm-to-upstream.mlir)
