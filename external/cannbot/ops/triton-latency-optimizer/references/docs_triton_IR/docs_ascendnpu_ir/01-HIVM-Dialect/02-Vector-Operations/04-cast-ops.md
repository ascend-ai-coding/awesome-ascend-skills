# HIVM 类型转换操作

> 关键词：HIVM, vcast, round_mode, cast, bitcast, type conversion

## 概述

`hir.vcast` 是 HIVM 方言的逐元类型转换操作，支持在不同数据类型之间进行转换。它是 HIVM 中最复杂的向量操作之一，提供了丰富的舍入模式（round_mode）和转换方式（cast）属性来控制转换行为。

> Python API 对应：`tl.cast()`, `tl.to()`, 以及隐式类型转换。

## IR 操作定义

### hir.vcast — 逐元类型转换

#### TableGen 定义

```tablegen
def VCastOp : HIVM_ElementwiseUnaryOp<"vcast",
    [StaticMaxRankTrait<2>,
     VectorOnlyTrait<0>, HIVMOpSameOperandsAndResultRank,
     DeclareOpInterfaceMethods<ExtraBufferOpInterface>,
     BroadcastableOTF
    ]> {
  let summary = "Elementwise Vector Type Conversion Op";
  let description = baseClassDescription # [{
     Additional constraints:
       1. Supports the following conversions:

        | src  | dst  | roundingmode                                      |
        |------|------|---------------------------------------------------|
        | f32  | f32  | round, rint, floor, ceil, trunc                   |
        | f32  | f16  | round, rint, floor, ceil, trunc, odd              |
        | f32  | i64  | round, rint, floor, ceil, trunc                   |
        | f32  | i32  | round, rint, floor, ceil, trunc                   |
        | f32  | i16  | round, rint, floor, ceil, trunc                   |
        | f32  | s64  | round, rint, floor, ceil, trunc                   |
        | f32  | bf16 | round, rint, floor, ceil, trunc                   |
        | f16  | f32  | rint                                              |
        | f16  | i32  | round, rint, floor, ceil, trunc                   |
        | f16  | i16  | round, rint, floor, ceil, trunc                   |
        | f16  | i8   | round, rint, floor, ceil, trunc                   |
        | f16  | ui8  | round, rint, floor, ceil, trunc                   |
        | f16  | i4   | round, rint, floor, ceil, trunc                   |
        | bf16 | f32  | rint                                              |
        | bf16 | i32  | round, rint, floor, ceil, trunc                   |
        | ui8  | f16  | rint                                              |
        | i8   | f16  | rint                                              |
        | i8   | i1   | rint                                              |
        | i16  | f16  | round, rint, floor, ceil, trunc                   |
        | i16  | f32  | rint                                              |
        | i32  | f32  | round, rint, floor, ceil, trunc                   |
        | i32  | i64  | rint                                              |
        | i32  | i16  | rint                                              |
        | i64  | i32  | rint                                              |
        | i64  | f32  | round, rint, floor, ceil, trunc                   |
        | i4   | f16  | rint                                              |
        | i1   | f16  | rint                                              |
        | i1   | f32  | rint                                              |
  }];
  let arguments = (ins Variadic<AnyShaped>:$src,
                       Variadic<AnyShaped>:$dst,
                       Optional<AnyMemRef>:$temp_buffer,
                       DefaultValuedAttr<HIVM_RoundModeAttr, "RoundMode::RINT">:$round_mode,
                       DefaultValuedAttr<HIVM_TypeFnAttr, "TypeFn::cast_signed">:$cast,
                       DefaultValuedAttr<DenseI64ArrayAttr, "{}">:$transpose,
                       DefaultValuedAttr<DenseI64ArrayAttr, "{}">:$broadcast
  );
  let hasVerifier = 1;
}
```

源码参考：[HIVMVectorOps.td#L420-L500](https://gitcode.com/Ascend/AscendNPU-IR/tree/master/bishengir/include/bishengir/Dialect/HIVM/IR/HIVMVectorOps.td#L420-L500)

#### 参数说明

| 参数 | 类型 | 必选 | 默认值 | 说明 | 约束 |
|------|------|------|--------|------|------|
| $src | Variadic\<AnyShaped\> | 是 | - | 输入向量 | VectorOnly |
| $dst | Variadic\<AnyShaped\> | 是 | - | 输出向量 | 与 src 相同 rank |
| $temp_buffer | Optional\<AnyMemRef\> | 否 | - | 临时缓冲区 | - |
| $round_mode | HIVM_RoundModeAttr | 否 | RINT | 舍入模式 | 见下方枚举表 |
| $cast | HIVM_TypeFnAttr | 否 | cast_signed | 转换方式 | 见下方枚举表 |
| $transpose | DenseI64ArrayAttr | 否 | {} | OTF 转置维度 | - |
| $broadcast | DenseI64ArrayAttr | 否 | {} | OTF 广播维度 | - |

## round_mode 属性

`round_mode` 控制 vcast 在精度降低时的舍入行为。定义在 [HIVMAttrs.td#L378-L406](https://gitcode.com/Ascend/AscendNPU-IR/tree/master/bishengir/include/bishengir/Dialect/HIVM/IR/HIVMAttrs.td#L378-L406)。

| 枚举值 | 字符串表示 | 说明 | C 语言等价 |
|--------|-----------|------|-----------|
| RINT (0) | rint | 向最近整数舍入，偶数优先 | `rint()` |
| ROUND (1) | round | 向最近整数舍入，远离零优先 | `round()` |
| FLOOR (2) | floor | 向负无穷方向舍入 | `floor()` |
| CEIL (3) | ceil | 向正无穷方向舍入 | `ceil()` |
| TRUNC (4) | trunc | 向零方向舍入 | `trunc()` |
| ODD (5) | odd | 向奇数舍入（Von Neumann 舍入） | - |
| TRUNCWITHOVERFLOW (6) | truncwithoverflow | 截断并允许溢出 | - |

### 舍入模式选择指南

| 场景 | 推荐模式 | 说明 |
|------|---------|------|
| 默认转换 | rint | 最常用，IEEE 754 标准舍入 |
| 精确向下取整 | floor | 用于计算索引 |
| 精确向上取整 | ceil | 用于分配大小 |
| F32 → F16 降精度 | odd | 保持最大精度，避免舍入偏差 |
| 截断小数部分 | trunc | 直接丢弃小数部分 |

## cast 属性

`cast` 属性控制整数类型转换时的符号处理方式。定义在 [HIVMAttrs.td#L431-L446](https://gitcode.com/Ascend/AscendNPU-IR/tree/master/bishengir/include/bishengir/Dialect/HIVM/IR/HIVMAttrs.td#L431-L446)。

| 枚举值 | 值 | 说明 |
|--------|---|------|
| cast_signed | 0 | 有符号转换（默认） |
| cast_unsigned | 1 | 无符号转换 |
| bitcast | 2 | 位转换（不改变位模式） |

### cast 模式说明

- **cast_signed**：将源整数视为有符号数进行转换。例如 `i16 → f32` 时，负数会被正确解释。
- **cast_unsigned**：将源整数视为无符号数进行转换。例如 `ui8 → f16` 时，所有值被视为正数。
- **bitcast**：不改变底层位模式，仅重新解释类型。例如 `f32 → i32` 的 bitcast 保持 32 位不变。

## unsigned_mode 属性（隐含）

虽然 vcast 本身使用 `cast` 属性，但 HIVM 还定义了 `UnsignedMode` 枚举，用于更细粒度的有符号/无符号转换控制：

| 枚举值 | 字符串表示 | 说明 |
|--------|-----------|------|
| SI2SI (0) | si2si | 有符号 → 有符号 |
| SI2UI (1) | si2ui | 有符号 → 无符号 |
| UI2SI (2) | ui2si | 无符号 → 有符号 |
| UI2UI (3) | ui2ui | 无符号 → 无符号 |

源码参考：[HIVMAttrs.td#L408-L425](https://gitcode.com/Ascend/AscendNPU-IR/tree/master/bishengir/include/bishengir/Dialect/HIVM/IR/HIVMAttrs.td#L408-L425)

## 支持的转换路径

### 浮点 → 浮点

| 源类型 | 目标类型 | 支持的 round_mode |
|--------|---------|------------------|
| f32 | f32 | round, rint, floor, ceil, trunc |
| f32 | f16 | round, rint, floor, ceil, trunc, odd |
| f32 | bf16 | round, rint, floor, ceil, trunc |
| f16 | f32 | rint |
| bf16 | f32 | rint |

### 浮点 → 整数

| 源类型 | 目标类型 | 支持的 round_mode |
|--------|---------|------------------|
| f32 | i64 | round, rint, floor, ceil, trunc |
| f32 | i32 | round, rint, floor, ceil, trunc |
| f32 | i16 | round, rint, floor, ceil, trunc |
| f16 | i32 | round, rint, floor, ceil, trunc |
| f16 | i16 | round, rint, floor, ceil, trunc |
| f16 | i8 | round, rint, floor, ceil, trunc |
| f16 | ui8 | round, rint, floor, ceil, trunc |
| f16 | i4 | round, rint, floor, ceil, trunc |
| bf16 | i32 | round, rint, floor, ceil, trunc |

### 整数 → 浮点

| 源类型 | 目标类型 | 支持的 round_mode |
|--------|---------|------------------|
| i8 | f16 | rint |
| ui8 | f16 | rint |
| i16 | f16 | round, rint, floor, ceil, trunc |
| i16 | f32 | rint |
| i32 | f32 | round, rint, floor, ceil, trunc |
| i64 | f32 | round, rint, floor, ceil, trunc |
| i4 | f16 | rint |
| i1 | f16 | rint |
| i1 | f32 | rint |

### 整数 → 整数

| 源类型 | 目标类型 | 支持的 round_mode |
|--------|---------|------------------|
| i8 | i1 | rint |
| i32 | i64 | rint |
| i32 | i16 | rint |
| i64 | i32 | rint |

## IR 示例

### 基本类型转换

```mlir
hivm.hir.vcast ins(%src : memref<2x16xbf16>) outs(%dst : memref<2x16xf32>)
```

### 指定舍入模式

```mlir
hivm.hir.vcast ins(%src : memref<2x16xbf16>) outs(%dst : memref<2x16xf32>)
               round_mode = #hivm.round_mode<rint>

hivm.hir.vcast ins(%src : memref<2x16xbf16>) outs(%dst : memref<2x16xi32>)
               round_mode = #hivm.round_mode<round>

hivm.hir.vcast ins(%src : memref<2x16xbf16>) outs(%dst : memref<2x16xi32>)
               round_mode = #hivm.round_mode<ceil>

hivm.hir.vcast ins(%src : memref<2x16xbf16>) outs(%dst : memref<2x16xi32>)
               round_mode = #hivm.round_mode<floor>
```

### Tensor 语义

```mlir
%result = hivm.hir.vcast ins(%src : tensor<23x77xi32>) outs(%dst : tensor<23x77xf32>) -> tensor<23x77xf32>
```

### 指定转换方式

```mlir
hivm.hir.vcast ins(%src : memref<2x16xi32>) outs(%dst : memref<2x16xf32>)
               cast = #hivm.cast<cast_signed>

hivm.hir.vcast ins(%src : memref<2x16xi32>) outs(%dst : memref<2x16xf32>)
               cast = #hivm.cast<cast_unsigned>
```

## IR 层约束与验证

1. **Rank 一致性**：输入和输出必须具有相同的 rank（`HIVMOpSameOperandsAndResultRank`）
2. **最大 Rank 限制**：StaticMaxRankTrait<2>，最多支持 2 维
3. **转换路径验证**：`hasVerifier = 1`，操作包含自定义验证器，根据硬件版本验证参数合法性
4. **round_mode 与转换路径兼容性**：不是所有 round_mode 都适用于所有转换路径，详见上方"支持的转换路径"表
5. **元素类型可以不同**：vcast 是唯一一个不要求 SameOperandsElementType 的 ElementwiseUnaryOp

## 与其他 IR 操作的关系

| HIVM 操作 | 上游降级 | HFusion 降级 | 说明 |
|-----------|---------|-------------|------|
| vcast | - | hfusion.cast {round_mode = ...} | round_mode 属性直接映射 |

降级示例：
```mlir
hivm.hir.vcast ins(%src : memref<2x16xbf16>) outs(%dst : memref<2x16xf32>)
               round_mode = #hivm.round_mode<rint>

%result = hfusion.cast {round_mode = #hfusion.round_mode<rint>} %src : memref<2x16xbf16> -> memref<2x16xf32>
```

## 常见问题

**Q: vcast 和 hivm.hir.bitcast 有什么区别？**
A: `hivm.hir.bitcast` 是独立的位转换操作，不改变位模式仅重新解释类型，不需要 round_mode。`hir.vcast` 使用 `cast = bitcast` 时语义类似，但 vcast 还支持需要舍入的类型转换。

**Q: F32 → F16 为什么支持 odd 模式而其他转换不支持？**
A: F32 → F16 是最常见的降精度场景，odd 舍入（Von Neumann 舍入）可以避免统计偏差，在量化场景中特别重要。硬件仅为此路径实现了 odd 模式。

**Q: TRUNCWITHOVERFLOW 和 TRUNC 有什么区别？**
A: TRUNC 在溢出时行为未定义或饱和，而 TRUNCWITHOVERFLOW 允许溢出发生而不做饱和处理。后者在需要检测溢出的场景中使用。

**Q: 为什么有些转换路径只支持 rint？**
A: 从低精度到高精度的转换（如 f16 → f32, i16 → f32）是精确的，不需要舍入，因此只使用默认的 rint 模式。从高精度到低精度的转换才需要指定舍入策略。

## 相关文档

- Python API：docs_triton_ascend 中的 `tl.cast()` 文档
- 源码参考：
  - [HIVMVectorOps.td - VCastOp](https://gitcode.com/Ascend/AscendNPU-IR/tree/master/bishengir/include/bishengir/Dialect/HIVM/IR/HIVMVectorOps.td#L420-L500)
  - [HIVMAttrs.td - RoundMode 枚举](https://gitcode.com/Ascend/AscendNPU-IR/tree/master/bishengir/include/bishengir/Dialect/HIVM/IR/HIVMAttrs.td#L378-L406)
  - [HIVMAttrs.td - TypeFn 枚举](https://gitcode.com/Ascend/AscendNPU-IR/tree/master/bishengir/include/bishengir/Dialect/HIVM/IR/HIVMAttrs.td#L431-L446)
  - [HIVMAttrs.td - UnsignedMode 枚举](https://gitcode.com/Ascend/AscendNPU-IR/tree/master/bishengir/include/bishengir/Dialect/HIVM/IR/HIVMAttrs.td#L408-L425)
  - [convert-hivm-to-upstream.mlir](https://gitcode.com/Ascend/AscendNPU-IR/tree/master/bishengir/test/ExecutionEngine/convert-hivm-to-upstream.mlir)
