# HIVM 比较运算

> 关键词：HIVM, vcmp, compare, EQ, NE, LT, GT, GE, LE

## 概述

`hir.vcmp` 是 HIVM 方言的逐元比较操作，对两个输入向量的对应元素执行比较运算，产生一个布尔类型的结果向量。比较模式通过 `compare_mode` 属性指定，支持六种比较关系。

> Python API 对应：Triton 的比较运算符 `==`, `!=`, `<`, `>`, `<=`, `>=`，以及 `tl.math` 中的相关函数。

## IR 操作定义

### hir.vcmp — 逐元比较

#### TableGen 定义

```tablegen
def VCmpOp : HIVM_ElementwiseBinaryOp<"vcmp",
    [StaticMaxRankTrait<1>,
     OperElemTypeConstraints<[0, 1], [F16, F32, I8, I16, I32, I64]>,
     OperElemTypeConstraints<[/*dstIdx=*/2], [I1, I8]>,
     VectorOnlyTrait<0>,
     DeclareOpInterfaceMethods<ImplByScalarOpInterface>
    ]> {
  let summary = "Elementwise Binary Vector Comparison Op";
  let description = baseClassDescription # [{
    Compare elements from two source vector. If the comparison result is true,
    the corresponding bit of `dst` is 1 or 8.

    Additional constraints:
      1. The input vectors and output vector must have the same ranks
      2. The element type of `dst` must be bool
      3. The input is vector-only.
      4. Supports the following data type:

        |    compare mode   |       element type      |
        |-------------------|-------------------------|
        | GE/GT/LE/LT/NE/EQ | f16, f32, i16, i32, i64 |
  }];
  let arguments = (ins Variadic<AnyType>:$src,
                       Variadic<AnyShaped>:$dst,
                       DefaultValuedAttr<HIVM_CmpModeAttr, "CompareMode::EQ">:$compare_mode,
                       DefaultValuedAttr<DenseI64ArrayAttr, "{}">:$transpose,
                       DefaultValuedAttr<DenseI64ArrayAttr, "{}">:$broadcast
  );
  let assemblyFormat = [{
    attr-dict `ins` `(` $src `:` type($src) `)`
    `outs` `(` $dst  `:` type($dst) `)`
    (`compare_mode` `=` $compare_mode^)?
    (`->` type($result)^)?
  }];
}
```

源码参考：[HIVMVectorOps.td#L944-L979](https://gitcode.com/Ascend/AscendNPU-IR/tree/master/bishengir/include/bishengir/Dialect/HIVM/IR/HIVMVectorOps.td#L944-L979)

#### 参数说明

| 参数 | 类型 | 必选 | 默认值 | 说明 | 约束 |
|------|------|------|--------|------|------|
| $src[0] | AnyType | 是 | - | 第一个输入向量 | VectorOnly, 元素类型 F16/F32/I8/I16/I32/I64 |
| $src[1] | AnyType | 是 | - | 第二个输入向量/标量 | 元素类型 F16/F32/I8/I16/I32/I64 |
| $dst | Variadic\<AnyShaped\> | 是 | - | 输出向量 | 元素类型 I1 或 I8 |
| $compare_mode | HIVM_CmpModeAttr | 否 | EQ | 比较模式 | 见下方枚举表 |
| $transpose | DenseI64ArrayAttr | 否 | {} | OTF 转置维度 | - |
| $broadcast | DenseI64ArrayAttr | 否 | {} | OTF 广播维度 | - |

注意：vcmp 没有 `temp_buffer` 参数，也没有 `SameOperandsElementType` 约束（因为输入和输出的元素类型不同）。

## compare_mode 属性

`compare_mode` 控制 vcmp 的比较语义。定义在 [HIVMAttrs.td#L452-L473](https://gitcode.com/Ascend/AscendNPU-IR/tree/master/bishengir/include/bishengir/Dialect/HIVM/IR/HIVMAttrs.td#L452-L473)。

| 枚举值 | 值 | 字符串表示 | 语义 | Python 等价 |
|--------|---|-----------|------|------------|
| EQ | 0 | eq | 等于 | `a == b` |
| NE | 1 | ne | 不等于 | `a != b` |
| LT | 2 | lt | 小于 | `a < b` |
| GT | 3 | gt | 大于 | `a > b` |
| GE | 4 | ge | 大于等于 | `a >= b` |
| LE | 5 | le | 小于等于 | `a <= b` |

## 数据类型约束

vcmp 使用两组独立的 OperElemTypeConstraints：

| 操作数位置 | 语义 | 支持的元素类型 | 约束来源 |
|-----------|------|--------------|----------|
| $src[0], $src[1] | 输入 | F16, F32, I8, I16, I32, I64 | OperElemTypeConstraints<[0, 1], [F16, F32, I8, I16, I32, I64]> |
| $dst (idx=2) | 输出 | I1, I8 | OperElemTypeConstraints<[2], [I1, I8]> |

### 比较模式与数据类型对应

| 比较模式 | 支持的输入元素类型 |
|---------|------------------|
| EQ, NE, LT, GT, GE, LE | F16, F32, I16, I32, I64 |

注意：虽然 OperElemTypeConstraints 允许 I8 输入，但 TableGen 描述中的约束表仅列出 f16, f32, i16, i32, i64。I8 的支持可能取决于具体硬件版本。

## IR 示例

### 等于比较

```mlir
%eq = hivm.hir.vcmp
        ins(%a, %b : tensor<4xf32>, tensor<4xf32>)
        outs(%init : tensor<4xi1>)
        compare_mode = #hivm.compare_mode<eq> -> tensor<4xi1>
```

### 不等于比较

```mlir
%ne = hivm.hir.vcmp
        ins(%a, %b : tensor<4xf32>, tensor<4xf32>)
        outs(%init : tensor<4xi1>)
        compare_mode = #hivm.compare_mode<ne> -> tensor<4xi1>
```

### 小于比较

```mlir
%lt = hivm.hir.vcmp
        ins(%a, %b : tensor<4xf32>, tensor<4xf32>)
        outs(%init : tensor<4xi1>)
        compare_mode = #hivm.compare_mode<lt> -> tensor<4xi1>
```

### 向量-标量比较

```mlir
%result = hivm.hir.vcmp ins(%vec, %scalar : tensor<23x77xf32>, f32)
           outs(%dst : tensor<23x77xi1>) compare_mode = <ne> -> tensor<23x77xi1>
```

### Memref 语义

```mlir
hivm.hir.vcmp ins(%a, %b : memref<4xf32>, memref<4xf32>)
              outs(%dst : memref<4xi1>) compare_mode = #hivm.compare_mode<ge>
```

### 与 vsel 配合使用的典型模式

```mlir
%cond = hivm.hir.vcmp ins(%a, %zero : tensor<Nxf32>, f32)
         outs(%init : tensor<Nxi1>) compare_mode = <ne> -> tensor<Nxi1>
%result = hivm.hir.vsel ins(%cond, %val_true, %val_false : tensor<Nxi1>, f32, tensor<Nxf32>)
           outs(%dst : tensor<Nxf32>) -> tensor<Nxf32>
```

## IR 层约束与验证

1. **输出类型约束**：输出向量的元素类型必须为 I1 或 I8
2. **最大 Rank 限制**：StaticMaxRankTrait<1>，仅支持 1 维
3. **VectorOnly 约束**：第一个输入操作数必须为向量类型
4. **ImplByScalarOpInterface**：第二个输入可以是标量，支持向量-标量比较。此外，整数类型的 vcmp 在特定条件下会被降级为标量循环（详见下方「标量降级」章节）
5. **Rank 一致性**：输入向量和输出向量必须具有相同的 rank
6. **无 SameOperandsElementType**：输入和输出的元素类型不同（输入为数值类型，输出为布尔类型）

## 与其他 IR 操作的关系

| HIVM 操作 | compare_mode | HFusion 降级 | 说明 |
|-----------|-------------|-------------|------|
| vcmp | eq | hfusion.compare {compare_fn = veq} | 等于 |
| vcmp | ne | hfusion.compare {compare_fn = vne} | 不等于 |
| vcmp | lt | hfusion.compare {compare_fn = vlt} | 小于 |
| vcmp | le | hfusion.compare {compare_fn = vle} | 小于等于 |
| vcmp | gt | hfusion.compare {compare_fn = vgt} | 大于 |
| vcmp | ge | hfusion.compare {compare_fn = vge} | 大于等于 |

vcmp 的典型使用模式是与 vsel 配合，形成条件选择：

```
vcmp (compare_mode) → vsel (条件选择)
```

## 标量降级

vcmp 实现了 `ImplByScalarOpInterface`，在特定条件下会被降级为标量循环（`scf.for` + `arith.CmpIOp`），而非使用硬件向量比较指令。

### 降级条件

vcmp 在以下条件**全部满足**时降级为标量循环：

1. 操作具有纯 buffer 语义（`hasPureBufferSemantics()` 为 true）
2. 第一个操作数为 MemRefType 或 TensorType
3. 元素类型为整数类型
4. **且**满足以下任一条件：
   - 元素类型不是 i32（如 i8, i16, i64）
   - 元素类型是 i32，但比较模式不是 EQ 且不是 NE

### 降级判断矩阵

| 元素类型 | EQ | NE | LT | GT | LE | GE |
|---------|----|----|----|----|----|-----|
| f16 | 向量 ✅ | 向量 ✅ | 向量 ✅ | 向量 ✅ | 向量 ✅ | 向量 ✅ |
| f32 | 向量 ✅ | 向量 ✅ | 向量 ✅ | 向量 ✅ | 向量 ✅ | 向量 ✅ |
| i8 | **标量** ⚠️ | **标量** ⚠️ | **标量** ⚠️ | **标量** ⚠️ | **标量** ⚠️ | **标量** ⚠️ |
| i16 | **标量** ⚠️ | **标量** ⚠️ | **标量** ⚠️ | **标量** ⚠️ | **标量** ⚠️ | **标量** ⚠️ |
| i32 | 向量 ✅ | 向量 ✅ | **标量** ⚠️ | **标量** ⚠️ | **标量** ⚠️ | **标量** ⚠️ |
| i64 | **标量** ⚠️ | **标量** ⚠️ | **标量** ⚠️ | **标量** ⚠️ | **标量** ⚠️ | **标量** ⚠️ |

### 根本原因

硬件向量比较指令对整数类型的支持有限：
- 浮点类型（f16/f32）：全部 6 种比较模式均有向量指令支持
- i32 的 EQ/NE：硬件向量指令支持相等/不等比较
- i32 的 LT/GT/LE/GE 及其他整数宽度：无对应向量指令，只能退化为标量循环

### 降级实现细节

标量降级时，vcmp 被分解为嵌套 `scf.for` 循环，循环体内逐元素执行 `arith.CmpIOp`（有符号比较谓词：slt/sgt/sle/sge/eq/ne）。由于 store 操作不支持 i1 类型，比较结果（i1）会先通过 `arith.ExtUIOp` 零扩展为 i8 再存储。

此外，在 `HIVMDecomposeOp` Pass 中有预处理步骤：将 vcmp 的输出从 i1 转为 i8 临时 buffer，之后用 `VCastOp` 转回 i1。

### 优化建议

- **优先使用浮点比较**：f16/f32 的所有比较模式都走向量路径，性能最优
- **整数相等/不等比较使用 i32**：i32 的 EQ/NE 是唯一能走向量路径的整数比较
- **避免整数大小比较**：i32 的 LT/GT/LE/GE 以及 i8/i16/i64 的所有比较都会标量降级
- **替代策略**：如果业务逻辑允许，将整数比较转为浮点比较（先 cast 再 compare），可避免标量降级

> 完整的标量降级文档：[11-scalar-lowering.md](../../../docs_ascendnpu_ir/01-HIVM-Dialect/02-Vector-Operations/11-scalar-lowering.md)

## 常见问题

**Q: vcmp 的输出为什么支持 I8 而不仅仅是 I1？**
A: 硬件实现中，比较结果可以存储为 I8 类型（非零表示真，零表示假），这在某些后续操作中更高效。I1 是最紧凑的表示，但 I8 在内存对齐方面更有优势。

**Q: vcmp 支持浮点比较的 NaN 处理吗？**
A: HIVM 遵循 IEEE 754 标准的浮点比较语义。NaN 与任何值（包括自身）的比较结果均为 false（EQ 和 NE 除外：NaN != NaN 为 true）。

**Q: 为什么 vcmp 没有 temp_buffer？**
A: 比较操作是简单的逐元操作，不需要额外的临时存储空间。

**Q: vcmp 的默认 compare_mode 为什么是 EQ？**
A: 等于比较是最常用的比较模式，作为默认值可以简化最常见的使用场景。

## 相关文档

- 标量降级详解：[11-scalar-lowering.md](../../../docs_ascendnpu_ir/01-HIVM-Dialect/02-Vector-Operations/11-scalar-lowering.md)
- Python API：docs_triton_ascend 中的比较运算符文档
- 源码参考：
  - [HIVMVectorOps.td - VCmpOp](https://gitcode.com/Ascend/AscendNPU-IR/tree/master/bishengir/include/bishengir/Dialect/HIVM/IR/HIVMVectorOps.td#L944-L979)
  - [HIVMAttrs.td - CmpMode 枚举](https://gitcode.com/Ascend/AscendNPU-IR/tree/master/bishengir/include/bishengir/Dialect/HIVM/IR/HIVMAttrs.td#L452-L473)
  - [convert-hivm-to-upstream.mlir](https://gitcode.com/Ascend/AscendNPU-IR/tree/master/bishengir/test/ExecutionEngine/convert-hivm-to-upstream.mlir)
