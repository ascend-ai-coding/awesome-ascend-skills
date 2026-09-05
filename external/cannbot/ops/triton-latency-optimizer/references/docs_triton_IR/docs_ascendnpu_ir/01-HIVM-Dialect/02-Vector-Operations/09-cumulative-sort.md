# HIVM 累积与排序操作

> 关键词：HIVM, vcumsum, vcumprod, vsort, cumulative, sort

## 概述

HIVM 累积与排序操作包括累积求和（`hir.vcumsum`）、累积求积（`hir.vcumprod`）和排序（`hir.vsort`）。累积操作沿指定维度计算从起始到当前位置的累积值，排序操作沿指定维度对元素进行排序。

> Python API 对应：`tl.cumsum()`, `tl.cumprod()`, `tl.sort()` 等。

## hir.vcumsum — 累积求和

### TableGen 定义

```tablegen
def VCumsumOp : HIVM_VectorOp<"vcumsum",
    [SameOperandsElementType,
     StaticMaxRankTrait<2>,
     OperElemTypeConstraints<[0], [I1, I8, I16, I32, I64, F16, F32, BF16]>,
     DeclareOpInterfaceMethods<HIVMStructuredOpInterface,
       ["getIteratorTypesArray"]>,
     UniformReassociationFlattenTrait,
     DeclareOpInterfaceMethods<FlattenInterface,
       ["getLimitedAxes", "adjustTargetDimensions"]>,
     DeclareOpInterfaceMethods<ImplByScalarOpInterface>
    ]> {
  let summary = "Vector Cumsum Op";
  let description = [{
    Calculate the cumulative sum of each element along the specified axis of
    `src`. Each element along the specified axis in the output of cumsum
    contains the sum of all elements from the first element to the current
    position in the original `src`.

    Constraints:
      1. The input vector and output vector must have the same rank
         and the same element type.

    Arguments:
      * `src`: the tensor/memref from which to calculate the cumulative sum
      * `dst`: the tensor/memref to store elements
      * `cum_dims`: specifies the dimension along which to calculate the
                    cumulative sum.

    Examples:
    ```mlir
    hivm.hir.vcumsum ins(%src : memref<?xf32>) outs(%dst : memref<?xf32>) cum_dims : [0]
    %result = hivm.hir.vcumsum ins(%src : tensor<?xf32>) outs(%dst : tensor<?xf32>) cum_dims : [0] -> tensor<?xf32>
    ```
  }];
  let arguments = (ins TensorOrMemref:$src,
                       TensorOrMemref:$dst,
                       ConfinedAttr<DenseI64ArrayAttr,
                       [DenseArrayStrictlySorted<DenseI64ArrayAttr>]>:$cum_dims,
                       BoolAttr:$reverse
  );
  let results = (outs Variadic<AnyRankedTensor>:$result);
  let assemblyFormat = [{
    attr-dict `ins` `(` $src `:` type($src) `)`
    `outs` `(` $dst `:` type($dst) `)`
    `cum_dims` `=` $cum_dims
    `reverse` `=` $reverse
    (`->` type($result)^)?
  }];
  let hasCanonicalizer = 1;
  let hasVerifier = 1;
}
```

源码参考：[HIVMVectorOps.td#L1719-L1772](https://gitcode.com/Ascend/AscendNPU-IR/tree/master/bishengir/include/bishengir/Dialect/HIVM/IR/HIVMVectorOps.td#L1719-L1772)

### 参数说明

| 参数 | 类型 | 必选 | 说明 | 约束 |
|------|------|------|------|------|
| $src | TensorOrMemref | 是 | 输入向量 | 元素类型 I1/I8/I16/I32/I64/F16/F32/BF16 |
| $dst | TensorOrMemref | 是 | 输出向量 | 与 src 相同 rank 和元素类型 |
| $cum_dims | DenseI64ArrayAttr | 是 | 累积维度 | 严格排序，Confined 约束 |
| $reverse | BoolAttr | 是 | 是否反向累积 | - |

### 数据类型约束

| 操作 | 支持的元素类型 | 约束来源 |
|------|--------------|----------|
| vcumsum | I1, I8, I16, I32, I64, F16, F32, BF16 | OperElemTypeConstraints<[0], [I1, I8, I16, I32, I64, F16, F32, BF16]> |

### 语义

正向累积（reverse = false）：
```
dst[i] = sum(src[0:i+1])  // 沿 cum_dims 维度
```

反向累积（reverse = true）：
```
dst[i] = sum(src[i:end])  // 沿 cum_dims 维度
```

### IR 示例

```mlir
hivm.hir.vcumsum ins(%src : memref<5x?x10xi32>) outs(%dst : memref<5x?x10xi32>) cum_dims = [1] reverse = false

%result = hivm.hir.vcumsum ins(%src : tensor<5x?x10xf32>) outs(%dst : tensor<5x?x10xf32>) cum_dims = [0] reverse = false -> tensor<5x?x10xf32>
```

降级到上游：`linalg.generic` 包含 `arith.addi`（整数）或 `arith.addf`（浮点）

---

## hir.vcumprod — 累积求积

### TableGen 定义

```tablegen
def VCumprodOp : HIVM_VectorOp<"vcumprod",
    [SameOperandsElementType,
     StaticMaxRankTrait<1>,
     OperElemTypeConstraints<[0], [I1, I8, I16, I32, I64, F16, F32, BF16]>,
     DeclareOpInterfaceMethods<HIVMStructuredOpInterface,
       ["getIteratorTypesArray"]>,
     UniformReassociationFlattenTrait,
     DeclareOpInterfaceMethods<FlattenInterface,
       ["getLimitedAxes", "adjustTargetDimensions"]>,
     DeclareOpInterfaceMethods<ImplByScalarOpInterface>
    ]> {
  let summary = "Vector Cumprod Op";
  let description = [{
    Calculate the cumulative product of each element along the specified axis
    of `src`. Each element along the specified axis in the output of cumprod
    contains the product of all elements from the first element to the current
    position in the original `src`.
  }];
  let arguments = (ins TensorOrMemref:$src,
                       TensorOrMemref:$dst,
                       ConfinedAttr<DenseI64ArrayAttr,
                       [DenseArrayStrictlySorted<DenseI64ArrayAttr>]>:$cum_dims,
                       BoolAttr:$reverse
  );
}
```

源码参考：[HIVMVectorOps.td#L1660-L1713](https://gitcode.com/Ascend/AscendNPU-IR/tree/master/bishengir/include/bishengir/Dialect/HIVM/IR/HIVMVectorOps.td#L1660-L1713)

### 参数说明

| 参数 | 类型 | 必选 | 说明 | 约束 |
|------|------|------|------|------|
| $src | TensorOrMemref | 是 | 输入向量 | 元素类型 I1/I8/I16/I32/I64/F16/F32/BF16 |
| $dst | TensorOrMemref | 是 | 输出向量 | 与 src 相同 rank 和元素类型 |
| $cum_dims | DenseI64ArrayAttr | 是 | 累积维度 | 严格排序，Confined 约束 |
| $reverse | BoolAttr | 是 | 是否反向累积 | - |

### 数据类型约束

| 操作 | 支持的元素类型 | 约束来源 |
|------|--------------|----------|
| vcumprod | I1, I8, I16, I32, I64, F16, F32, BF16 | OperElemTypeConstraints<[0], [I1, I8, I16, I32, I64, F16, F32, BF16]> |

### 语义

正向累积（reverse = false）：
```
dst[i] = prod(src[0:i+1])  // 沿 cum_dims 维度
```

反向累积（reverse = true）：
```
dst[i] = prod(src[i:end])  // 沿 cum_dims 维度
```

### IR 示例

```mlir
%result = hivm.hir.vcumprod ins(%src : tensor<5x?x10xf32>) outs(%dst : tensor<5x?x10xf32>) cum_dims = [0] reverse = false -> tensor<5x?x10xf32>

hivm.hir.vcumprod ins(%src : memref<5x?x10xi32>) outs(%dst : memref<5x?x10xi32>) cum_dims = [1] reverse = false
```

降级到上游：`linalg.generic` 包含 `arith.muli`（整数）或 `arith.mulf`（浮点）

---

## hir.vsort — 排序

### TableGen 定义

```tablegen
def VSortOp : HIVM_VectorOp<"vsort",
    [StaticMaxRankTrait<1>,
     AttrSizedOperandSegments,
     OperElemTypeConstraints<[0, 1], [F16, F32, I32, I64]>,
     DeclareOpInterfaceMethods<ExtraBufferOpInterface,
       ["getExtraBufferSize", "shouldAllocExtraBufferForScalarOrOTFBrc"]>
    ]> {
  let summary = "Vector Sort Op";
  let description = [{
    Sort the sorting axis of `src` in ascending or descending order, and output
    the sorted value and the index corresponding to the value.

    Constraints:
      1. The input vector and output vector must have the same rank.
      2. Currently only tail axis sorting is supported.

    Arguments:
      * `src`: the tensor/memref from which to be sorted
      * `dst_value`: the tensor/memref to store the sorted value
      * `dst_index`: the tensor/memref to store the index corresponding to dst_value
      * `descending`: determines whether to sort in ascending or descending
                      order. The default is false, which means ascending order
      * `sort_axis`: Axis to be sorted
  }];
  let arguments = (ins TensorOrMemref:$src,
                       Variadic<TensorOrMemref>:$dst,
                       Optional<AnyMemRef>:$temp_buffer,
                       DefaultValuedOptionalAttr<BoolAttr, "false">:$descending,
                       DefaultValuedAttr<I64Attr, "-1">:$sort_axis
  );
}
```

源码参考：[HIVMVectorOps.td#L1778-L1839](https://gitcode.com/Ascend/AscendNPU-IR/tree/master/bishengir/include/bishengir/Dialect/HIVM/IR/HIVMVectorOps.td#L1778-L1839)

### 参数说明

| 参数 | 类型 | 必选 | 默认值 | 说明 | 约束 |
|------|------|------|--------|------|------|
| $src | TensorOrMemref | 是 | - | 输入向量 | 元素类型 F16/F32/I32/I64 |
| $dst | Variadic\<TensorOrMemref\> | 是 | - | 输出（排序值 + 索引） | 2 个输出 |
| $temp_buffer | Optional\<AnyMemRef\> | 否 | - | 临时缓冲区 | - |
| $descending | BoolAttr | 否 | false | 是否降序排序 | - |
| $sort_axis | I64Attr | 否 | -1 | 排序轴 | 目前仅支持尾轴 |

### 数据类型约束

| 操作数位置 | 语义 | 支持的元素类型 | 约束来源 |
|-----------|------|--------------|----------|
| $src (idx=0) | 输入 | F16, F32, I32, I64 | OperElemTypeConstraints<[0, 1], [F16, F32, I32, I64]> |
| $dst (idx=1) | 排序值输出 | 与 src 相同 | OperElemTypeConstraints<[0, 1], [...]> |

注意：索引输出的元素类型为 I32。

### 语义

```
dst_value[i] = src[sorted_indices[i]]
dst_index[i] = sorted_indices[i]
```

- `descending = false`：升序排序（默认）
- `descending = true`：降序排序

### IR 示例

```mlir
hivm.hir.vsort ins(%src : memref<?xf32>) outs(%dst : memref<?xf32>) descending = true sort_axis = 0

%result = hivm.hir.vsort ins(%src : tensor<?xf32>) outs(%dst : tensor<?xf32>) descending = true sort_axis = 0 -> tensor<?xf32>
```

### 特殊接口

```cpp
Value getDstValue();     // 获取排序值输出
Value getDstIndex();     // 获取排序索引输出
int64_t getSignedSortAxis();  // 获取有符号排序轴
```

## 数据类型约束汇总

| 操作 | 支持的元素类型 | 最大 Rank | 输出数量 | reverse 属性 | descending 属性 |
|------|--------------|-----------|---------|-------------|----------------|
| vcumsum | I1, I8, I16, I32, I64, F16, F32, BF16 | 2 | 1 | 是 | - |
| vcumprod | I1, I8, I16, I32, I64, F16, F32, BF16 | 1 | 1 | 是 | - |
| vsort | F16, F32, I32, I64 | 1 | 2 | - | 是 |

## IR 层约束与验证

### 通用约束

1. **Rank 一致性**：输入和输出必须具有相同的 rank
2. **元素类型一致性**：输入和输出必须具有相同的元素类型（vsort 索引输出除外）
3. **cum_dims 约束**：必须严格排序（`DenseArrayStrictlySorted`）

### vsort 特有约束

1. **仅支持尾轴排序**：当前硬件实现仅支持沿最后一个轴排序
2. **两个输出**：排序值和排序索引
3. **sort_axis 默认值 -1**：表示最后一个轴
4. **hasVerifier = 1**：包含自定义验证器

## 与其他 IR 操作的关系

| HIVM 操作 | 上游降级 | 说明 |
|-----------|---------|------|
| vcumsum | linalg.generic { arith.addi/addf } | 累积求和 |
| vcumprod | linalg.generic { arith.muli/mulf } | 累积求积 |
| vsort | - | 无直接上游对应 |

## 常见问题

**Q: vcumsum 和 vreduce\<sum\> 有什么区别？**
A: vcumsum 保持输出与输入相同的形状，每个位置存储从起始到当前位置的累积和。vreduce\<sum\> 将归约维度压缩为 1，输出形状与输入不同。

**Q: vsort 为什么只支持尾轴排序？**
A: 这是当前硬件实现的限制。硬件排序指令仅支持沿最后一维排序。如果需要沿其他维度排序，需要先转置再排序再转置回来。

**Q: cum_dims 的 DenseArrayStrictlySorted 约束是什么意思？**
A: cum_dims 数组中的索引必须严格递增排列，不允许重复。例如 [0, 2] 合法，[2, 0] 或 [0, 0] 不合法。

**Q: reverse 属性在累积操作中的实际用途是什么？**
A: reverse = true 时，累积从数组末尾向起始方向计算。例如，对于序列 [1, 2, 3]，正向累积和为 [1, 3, 6]，反向累积和为 [6, 5, 3]。

## 相关文档

- Python API：docs_triton_ascend 中的 `tl.cumsum()`, `tl.sort()` 文档
- 源码参考：
  - [HIVMVectorOps.td - VCumsumOp](https://gitcode.com/Ascend/AscendNPU-IR/tree/master/bishengir/include/bishengir/Dialect/HIVM/IR/HIVMVectorOps.td#L1719-L1772)
  - [HIVMVectorOps.td - VCumprodOp](https://gitcode.com/Ascend/AscendNPU-IR/tree/master/bishengir/include/bishengir/Dialect/HIVM/IR/HIVMVectorOps.td#L1660-L1713)
  - [HIVMVectorOps.td - VSortOp](https://gitcode.com/Ascend/AscendNPU-IR/tree/master/bishengir/include/bishengir/Dialect/HIVM/IR/HIVMVectorOps.td#L1778-L1839)
  - [convert-hivm-to-upstream.mlir](https://gitcode.com/Ascend/AscendNPU-IR/tree/master/bishengir/test/ExecutionEngine/convert-hivm-to-upstream.mlir)
