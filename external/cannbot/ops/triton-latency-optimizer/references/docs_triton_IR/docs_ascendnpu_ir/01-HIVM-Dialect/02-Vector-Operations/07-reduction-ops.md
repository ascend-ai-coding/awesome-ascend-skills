# HIVM 归约运算

> 关键词：HIVM, vreduce, reduction, sum, prod, max, min, reduce_dims

## 概述

`hir.vreduce` 是 HIVM 方言的向量归约操作，沿指定维度对输入向量执行归约运算。它支持 12 种归约操作，从基本的求和/求积到带索引的最大/最小值归约。vreduce 是 HIVM 中功能最丰富的向量操作之一，直接映射到硬件的向量归约指令。

> Python API 对应：`tl.reduce()`, `tl.sum()`, `tl.max()`, `tl.min()`, `tl.argmax()`, `tl.argmin()` 等。

## IR 操作定义

### hir.vreduce — 向量归约

#### TableGen 定义

```tablegen
def VReduceOp : HIVM_VectorOp<"vreduce",
    [AttrSizedOperandSegments,
     OperElemTypeConstraints<[0, 1], [I1, I8, UI8, I16, UI16, I32, UI32, I64, UI64, F16, F32]>,
     DeclareOpInterfaceMethods<ExtraBufferOpInterface>,
     DeclareOpInterfaceMethods<HIVMStructuredOpInterface,
       ["getIteratorTypesArray", "getIndexingMaps"]>,
     InferMaxRankTrait, DeclareOpInterfaceMethods<LibraryFunctionOpInterface,
       ["inferOpLibraryMaxRank"]>,
     UniformReassociationFlattenTrait,
     CollapsibleConsecutiveTargetDimsTrait,
     DeclareOpInterfaceMethods<FlattenInterface,
       ["getLimitedAxes", "adjustTargetDimensions"]>,
     DeclareOpInterfaceMethods<ImplByScalarOpInterface>,
     DeclareOpInterfaceMethods<VectorizableOpInterface>,
     DeclareOpInterfaceMethods<BiShengIRAggregatedOpInterface,
       ["decomposeOperation", "getDecomposePhase"]>
    ]> {
  let summary = "Vector Reduction Op";
  let description = [{
    Recuce one or more axes of the source vector according to
    the reduction axes array, starting from an init value.

    Constraints:
      1. The input vector and output vector must have the same rank
         and the same element type.
      2. For the output operand, the size of the reduced axis must be 1.
      3. The reduction indices array can not be empty,
         nor can be larger than the ranks of the input vector.
      4. The reduced indices must be in `[0, RankOfDstVec)`.

    Examples:
    ```mlir
    hivm.hir.vreduce <add> ins(%src : memref<?xf32>) outs(%dst : memref<1xf32>) reduce_dims : [1]
    %result = hivm.hir.vreduce <max> ins(%src : tensor<?xf32>) outs(%dst : tensor<1xf32>) reduce_dims : [0] -> tensor<1xf32>
    ```
  }];
  let arguments = (ins TensorOrMemref:$src,
                       Variadic<TensorOrMemref>:$dst,
                       Optional<AnyMemRef>:$temp_buffer,
                       HIVM_ReduceOpAttr:$arith,
                       BoolAttr:$unsigned_src,
                       OptionalAttr<BoolAttr>:$tie_break_left,
                       DenseI64ArrayAttr:$reduce_dims
  );
  let results = (outs Variadic<AnyRankedTensor>:$result);
  let assemblyFormat = [{
    attr-dict $arith `ins` `(` $src `:` type($src) `)`
    `outs` `(` $dst `:` type($dst) `)`
    (`temp_buffer` `(` $temp_buffer^ `:` type($temp_buffer) `)`)?
    `unsigned_src` `=` $unsigned_src
    (`tie_break_left` `=` $tie_break_left^)?
    `reduce_dims` `=` $reduce_dims
    (`->` type($result)^)?
  }];
}
```

源码参考：[HIVMVectorOps.td#L1129-L1216](https://gitcode.com/Ascend/AscendNPU-IR/tree/master/bishengir/include/bishengir/Dialect/HIVM/IR/HIVMVectorOps.td#L1129-L1216)

#### 参数说明

| 参数 | 类型 | 必选 | 说明 | 约束 |
|------|------|------|------|------|
| $src | TensorOrMemref | 是 | 输入向量 | 元素类型 I1/I8/UI8/I16/UI16/I32/UI32/I64/UI64/F16/F32 |
| $dst | Variadic\<TensorOrMemref\> | 是 | 输出向量（归约值） | 与 src 相同 rank 和元素类型 |
| $temp_buffer | Optional\<AnyMemRef\> | 否 | 临时缓冲区 | ExtraBufferOpInterface |
| $arith | HIVM_ReduceOpAttr | 是 | 归约操作类型 | 见下方枚举表 |
| $unsigned_src | BoolAttr | 是 | 源操作数是否为无符号 | - |
| $tie_break_left | OptionalAttr\<BoolAttr\> | 否 | 归约冲突时偏向左侧 | 仅 max_with_index/min_with_index |
| $reduce_dims | DenseI64ArrayAttr | 是 | 归约维度数组 | 不可为空，索引在 [0, rank) 内 |

## reduce_operation 属性（$arith）

`$arith` 属性指定归约操作的类型。定义在 [HIVMAttrs.td#L596-L631](https://gitcode.com/Ascend/AscendNPU-IR/tree/master/bishengir/include/bishengir/Dialect/HIVM/IR/HIVMAttrs.td#L596-L631)。

| 枚举值 | 值 | 字符串表示 | 语义 | 输出数量 |
|--------|---|-----------|------|---------|
| sum | 1 | \<sum\> | 求和 | 1 |
| prod | 2 | \<prod\> | 求积 | 1 |
| max | 3 | \<max\> | 最大值 | 1 |
| min | 4 | \<min\> | 最小值 | 1 |
| max_with_index | 5 | \<max_with_index\> | 最大值及其索引 | 2 |
| min_with_index | 6 | \<min_with_index\> | 最小值及其索引 | 2 |
| any | 7 | \<any\> | 任意为真 | 1 |
| all | 8 | \<all\> | 全部为真 | 1 |
| xori | 9 | \<xori\> | 异或归约 | 1 |
| ori | 10 | \<ori\> | 或归约 | 1 |
| andi | 11 | \<andi\> | 与归约 | 1 |
| none | 0 | \<none\> | 无归约（占位符） | - |

### 归约操作分类

| 类别 | 操作 | 说明 |
|------|------|------|
| 算术归约 | sum, prod | 数值累加/累乘 |
| 极值归约 | max, min | 最大/最小值 |
| 带索引极值 | max_with_index, min_with_index | 最大/最小值及其位置索引 |
| 逻辑归约 | any, all | 逻辑或/与归约 |
| 位归约 | xori, ori, andi | 位运算归约 |

## unsigned_src 属性

`unsigned_src` 指示源操作数是否应被视为无符号数：

| 值 | 说明 |
|---|------|
| true | 源操作数视为无符号数 |
| false | 源操作数视为有符号数 |

这对整数类型的归约操作（特别是 max/min）有影响，因为无符号和有符号整数的比较语义不同。

## tie_break_left 属性

`tie_break_left` 仅对 `max_with_index` 和 `min_with_index` 归约有效。当多个元素具有相同的最大/最小值时：

| 值 | 说明 |
|---|------|
| true | 返回最左侧（最小索引）的位置 |
| false | 返回最右侧（最大索引）的位置 |
| 未设置 | 使用默认行为 |

## reduce_dims 参数

`reduce_dims` 指定沿哪些维度进行归约：

- 必须为非空数组
- 索引必须在 `[0, rank(dst))` 范围内
- 被归约维度在输出中的大小必须为 1
- 未被归约维度在输入和输出中的大小必须相同

## 数据类型约束

| 操作数位置 | 语义 | 支持的元素类型 | 约束来源 |
|-----------|------|--------------|----------|
| $src (idx=0) | 输入 | I1, I8, UI8, I16, UI16, I32, UI32, I64, UI64, F16, F32 | OperElemTypeConstraints<[0, 1], [...]> |
| $dst (idx=1) | 输出 | 与 src 相同 | OperElemTypeConstraints<[0, 1], [...]> |

## IR 示例

### 基本求和归约

```mlir
hivm.hir.vreduce <sum> ins(%src : memref<?xf32>) outs(%dst : memref<1xf32>) unsigned_src = false reduce_dims = [1]
```

### 最大值归约（tensor 语义）

```mlir
%result = hivm.hir.vreduce <max> ins(%src : tensor<?xf32>) outs(%dst : tensor<1xf32>) unsigned_src = false reduce_dims = [0] -> tensor<1xf32>
```

### 带索引的最大值归约

```mlir
%val, %idx = hivm.hir.vreduce <max_with_index> ins(%src : tensor<23x77xf32>) outs(%dst_val, %dst_idx : tensor<23x1xf32>, tensor<23x1xi32>) unsigned_src = false tie_break_left = true reduce_dims = [1] -> tensor<23x1xf32>, tensor<23x1xi32>
```

### 逻辑归约

```mlir
%result = hivm.hir.vreduce <any> ins(%src : tensor<Nxi1>) outs(%dst : tensor<1xi1>) unsigned_src = false reduce_dims = [0] -> tensor<1xi1>
```

## IR 层约束与验证

1. **Rank 一致性**：输入和输出向量必须具有相同的 rank
2. **元素类型一致性**：输入和输出必须具有相同的元素类型
3. **归约维度约束**：输出在归约维度上的大小必须为 1
4. **reduce_dims 非空**：归约维度数组不能为空
5. **reduce_dims 范围**：索引必须在 `[0, rank(dst))` 范围内
6. **max_with_index/min_with_index 输出**：需要两个输出操作数（值和索引）
7. **hasVerifier = 1**：包含自定义验证器
8. **hasCanonicalizer = 1**：包含自定义规范化器

## 与其他 IR 操作的关系

| HIVM 操作 | $arith | HFusion 降级 | 说明 |
|-----------|--------|-------------|------|
| vreduce | sum | hfusion.reduce \<sum\> | 求和 |
| vreduce | max | hfusion.reduce \<max\> | 最大值 |
| vreduce | min | hfusion.reduce \<min\> | 最小值 |
| vreduce | max_with_index | hfusion.reduce_with_index \<max\> | 带索引最大值 |
| vreduce | min_with_index | hfusion.reduce_with_index \<min\> | 带索引最小值 |

降级示例：
```mlir
%val, %idx = hivm.hir.vreduce <max_with_index> ins(%src : tensor<23x77xf32>)
           outs(%v, %i : tensor<23x1xf32>, tensor<23x1xi32>)
           unsigned_src = false tie_break_left = true reduce_dims = [1]
           -> tensor<23x1xf32>, tensor<23x1xi32>

%val, %idx = hfusion.reduce_with_index {tie_break_left = true, unsigned_src = false} <max>
           %src : tensor<23x77xf32> -> tensor<23x1xf32>, tensor<23x1xi32>
```

## 常见问题

**Q: vreduce 的输出为什么与输入具有相同的 rank？**
A: 这是 DestinationStyleOpInterface 的要求。归约维度在输出中的大小为 1，但 rank 保持不变。这与 NumPy 的 `keepdims=True` 行为类似。

**Q: max_with_index 的索引输出类型是什么？**
A: 索引输出的元素类型为 I32，与输入的数据类型无关。

**Q: unsigned_src 对浮点类型有影响吗？**
A: 对浮点类型没有影响，unsigned_src 仅对整数类型的归约操作有意义。

**Q: 可以同时对多个维度进行归约吗？**
A: 可以。reduce_dims 是一个数组，可以指定多个归约维度。例如 `reduce_dims = [0, 2]` 同时归约第 0 维和第 2 维。

**Q: vreduce 和 vcumsum/vcumprod 有什么区别？**
A: vreduce 将归约维度压缩为大小 1，输出 rank 不变但维度减小。vcumsum/vcumprod 保持所有维度不变，输出与输入形状相同，只是每个元素变为从起始到当前位置的累积值。

## 相关文档

- Python API：docs_triton_ascend 中的 `tl.reduce()`, `tl.sum()`, `tl.max()` 等文档
- 源码参考：
  - [HIVMVectorOps.td - VReduceOp](https://gitcode.com/Ascend/AscendNPU-IR/tree/master/bishengir/include/bishengir/Dialect/HIVM/IR/HIVMVectorOps.td#L1129-L1216)
  - [HIVMAttrs.td - ReduceOp 枚举](https://gitcode.com/Ascend/AscendNPU-IR/tree/master/bishengir/include/bishengir/Dialect/HIVM/IR/HIVMAttrs.td#L596-L631)
  - [convert-hivm-to-upstream.mlir](https://gitcode.com/Ascend/AscendNPU-IR/tree/master/bishengir/test/ExecutionEngine/convert-hivm-to-upstream.mlir)
