# HIVM 移位运算

> 关键词：HIVM, vshl, vshr, shift, round

## 概述

HIVM 移位运算包含左移（`hir.vshl`）和右移（`hir.vshr`）两个操作。两者均属于 `HIVM_ElementwiseBinaryOp`，但具有特殊的操作数约束：第一个操作数为向量，第二个操作数为标量，即仅支持 Vector-Scalar 模式。

> Python API 对应：Triton 的 `<<` 和 `>>` 运算符。

## IR 操作定义

### hir.vshl — 逐元左移

#### TableGen 定义

```tablegen
def VShLOp : HIVM_ElementwiseBinaryOp<"vshl",
   [SameOperandsElementType, StaticMaxRankTrait<3>,
    OperElemTypeConstraints<[0, 1], [I16, I32, I64]>,
    VectorOnlyTrait<0>, ScalarOnlyTrait<1>,
    DeclareOpInterfaceMethods<ExtraBufferOpInterface,
      ["getExtraBufferSize"]>,
    DeclareOpInterfaceMethods<ImplByScalarOpInterface>,
    BroadcastableOTF
   ]> {
  let summary = "Elementwise Binary Vector Shift Left Op";
  let description = baseClassDescription # [{
    Additional constraints:
      1. The input vector and result have the same element type.
      2. Support only Vector - Scalar operation.
  }];
  let arguments = (ins Variadic<AnyType>:$src, Variadic<AnyShaped>:$dst,
      Optional<AnyMemRef>:$temp_buffer,
      DefaultValuedAttr<DenseI64ArrayAttr, "{}">:$transpose,
      DefaultValuedAttr<DenseI64ArrayAttr, "{}">:$broadcast);
}
```

源码参考：[HIVMVectorOps.td#L873-L904](https://gitcode.com/Ascend/AscendNPU-IR/tree/master/bishengir/include/bishengir/Dialect/HIVM/IR/HIVMVectorOps.td#L873-L904)

#### 参数说明

| 参数 | 类型 | 必选 | 说明 | 约束 |
|------|------|------|------|------|
| $src[0] | AnyType | 是 | 输入向量 | VectorOnly, 元素类型 I16/I32/I64 |
| $src[1] | AnyType | 是 | 移位量（标量） | ScalarOnly, 元素类型 I16/I32/I64 |
| $dst | Variadic\<AnyShaped\> | 是 | 输出向量 | 与 src 相同元素类型 |
| $temp_buffer | Optional\<AnyMemRef\> | 否 | 临时缓冲区 | - |
| $transpose | DenseI64ArrayAttr (默认 {}) | 否 | OTF 转置维度 | - |
| $broadcast | DenseI64ArrayAttr (默认 {}) | 否 | OTF 广播维度 | - |

#### 数据类型约束

| 操作 | 支持的元素类型 | 约束来源 |
|------|--------------|----------|
| vshl | I16, I32, I64 | OperElemTypeConstraints<[0, 1], [I16, I32, I64]> |

#### 特殊约束

- **VectorOnlyTrait<0>**：第一个操作数（被移位向量）必须为向量
- **ScalarOnlyTrait<1>**：第二个操作数（移位量）必须为标量
- 仅支持 Vector-Scalar 操作模式

#### IR 示例

```mlir
%result = hivm.hir.vshl ins(%vec, %shift : tensor<32xi32>, i32) outs(%dst : tensor<32xi32>) -> tensor<32xi32>

hivm.hir.vshl ins(%vec, %shift : memref<32xi32>, i32) outs(%dst : memref<32xi32>)
```

---

### hir.vshr — 逐元右移

#### TableGen 定义

```tablegen
def VShROp : HIVM_ElementwiseBinaryOp<"vshr",
    [SameOperandsElementType, StaticMaxRankTrait<3>,
     OperElemTypeConstraints<[0, 1], [I16, I32, I64]>,
     VectorOnlyTrait<0>,
     DeclareOpInterfaceMethods<ExtraBufferOpInterface,
       ["getExtraBufferSize"]>,
     DeclareOpInterfaceMethods<ImplByScalarOpInterface>,
     BroadcastableOTF
    ]> {
  let summary = "Elementwise Binary Vector Shift Right Op";
  let description = baseClassDescription # [{
    Additional constraints:
      1. The input vector and result have the same element type.
      2. Support only Vector - Scalar operation.
      3. If `round` is set to true, rounding is applied during arithmetic
         shift right.
  }];
  let arguments = (ins Variadic<AnyType>:$src,
                       Variadic<AnyShaped>:$dst,
                       Optional<AnyMemRef>:$temp_buffer,
                       DefaultValuedOptionalAttr<BoolAttr, "true">:$round,
                       DefaultValuedAttr<DenseI64ArrayAttr, "{}">:$transpose,
                       DefaultValuedAttr<DenseI64ArrayAttr, "{}">:$broadcast
  );
  let assemblyFormat = [{
    attr-dict `ins` `(` $src `:` type($src) `)`
      `outs` `(` $dst  `:` type($dst) `)`
      (`temp_buffer` `(` $temp_buffer^ `:` type($temp_buffer) `)`)?
      (`round` `:` $round^ )?
      (`->` type($result)^)?
  }];
}
```

源码参考：[HIVMVectorOps.td#L906-L942](https://gitcode.com/Ascend/AscendNPU-IR/tree/master/bishengir/include/bishengir/Dialect/HIVM/IR/HIVMVectorOps.td#L906-L942)

#### 参数说明

| 参数 | 类型 | 必选 | 默认值 | 说明 | 约束 |
|------|------|------|--------|------|------|
| $src[0] | AnyType | 是 | - | 输入向量 | VectorOnly, 元素类型 I16/I32/I64 |
| $src[1] | AnyType | 是 | - | 移位量（标量） | 元素类型 I16/I32/I64 |
| $dst | Variadic\<AnyShaped\> | 是 | - | 输出向量 | 与 src 相同元素类型 |
| $temp_buffer | Optional\<AnyMemRef\> | 否 | - | 临时缓冲区 | - |
| $round | BoolAttr | 否 | true | 是否在算术右移时进行舍入 | - |
| $transpose | DenseI64ArrayAttr | 否 | {} | OTF 转置维度 | - |
| $broadcast | DenseI64ArrayAttr | 否 | {} | OTF 广播维度 | - |

#### 数据类型约束

| 操作 | 支持的元素类型 | 约束来源 |
|------|--------------|----------|
| vshr | I16, I32, I64 | OperElemTypeConstraints<[0, 1], [I16, I32, I64]> |

#### round 属性说明

`round` 属性是 vshr 独有的，控制算术右移时的舍入行为：

- **round = true（默认）**：在算术右移时进行舍入。如果移出的最高位为 1，则结果加 1。这相当于 `(src + (1 << (shift - 1))) >> shift`，实现更精确的除以 2^n 运算。
- **round = false**：不进行舍入，直接截断。相当于标准的算术右移。

#### 特殊约束

- **VectorOnlyTrait<0>**：第一个操作数必须为向量
- 注意：vshr 没有 `ScalarOnlyTrait<1>`，但描述中说明仅支持 Vector-Scalar 操作
- **ImplByScalarOpInterface**：支持标量实现路径

#### IR 示例

```mlir
%result = hivm.hir.vshr ins(%vec, %shift : tensor<32xi32>, i32) outs(%dst : tensor<32xi32>) -> tensor<32xi32>

hivm.hir.vshr ins(%vec, %shift : memref<32xi32>, i32) outs(%dst : memref<32xi32>)

%result = hivm.hir.vshr ins(%vec, %shift : tensor<32xi32>, i32) outs(%dst : tensor<32xi32>) round : true -> tensor<32xi32>

%result = hivm.hir.vshr ins(%vec, %shift : tensor<32xi32>, i32) outs(%dst : tensor<32xi32>) round : false -> tensor<32xi32>
```

## 数据类型约束汇总

| 操作 | 支持的元素类型 | 最大 Rank | 操作模式 | OTF 广播 | Extra Buffer | round 属性 |
|------|--------------|-----------|---------|---------|-------------|-----------|
| vshl | I16, I32, I64 | 3 | Vector-Scalar | 是 | 是 | 无 |
| vshr | I16, I32, I64 | 3 | Vector-Scalar | 是 | 是 | 是（默认 true） |

## IR 层约束与验证

1. **元素类型一致性**：输入向量和输出向量必须具有相同的元素类型
2. **VectorOnly 约束**：第一个输入操作数必须为向量类型
3. **ScalarOnly 约束（vshl）**：第二个输入操作数必须为标量类型
4. **仅支持整数类型**：I16, I32, I64，不支持浮点类型
5. **移位量语义**：移位量为标量，所有元素使用相同的移位量

## 与其他 IR 操作的关系

| HIVM 操作 | 上游降级 | 说明 |
|-----------|---------|------|
| vshl | arith.shli | 逻辑左移 |
| vshr (round=false) | arith.shrsi | 算术右移（有符号） |
| vshr (round=true) | 需要额外的加法和移位组合 | 舍入右移 |

## 常见问题

**Q: vshl 和 vshr 为什么只支持 Vector-Scalar 模式？**
A: 硬件向量移位指令的设计是所有元素使用相同的移位量。如果需要逐元素不同移位量，需要通过循环或其他方式实现。

**Q: vshr 的 round 属性有什么实际用途？**
A: round 属性在将右移用作除以 2^n 的近似时特别有用。例如，`x >> 1` 等价于 `x / 2`（截断），而 `round: true` 的 `(x + 1) >> 1` 更接近数学上的四舍五入除法。

**Q: vshr 是算术右移还是逻辑右移？**
A: vshr 执行算术右移（保留符号位）。逻辑右移（无符号右移）需要先将数据视为无符号类型。

**Q: vshl 为什么有 ScalarOnlyTrait 而 vshr 没有？**
A: 这是一个实现细节差异。vshr 虽然没有显式的 ScalarOnlyTrait<1>，但其描述中明确说明仅支持 Vector-Scalar 操作。ImplByScalarOpInterface 也暗示了标量操作数的支持。

## 相关文档

- Python API：docs_triton_ascend 中的位运算文档
- 源码参考：
  - [HIVMVectorOps.td - VShLOp](https://gitcode.com/Ascend/AscendNPU-IR/tree/master/bishengir/include/bishengir/Dialect/HIVM/IR/HIVMVectorOps.td#L873-L904)
  - [HIVMVectorOps.td - VShROp](https://gitcode.com/Ascend/AscendNPU-IR/tree/master/bishengir/include/bishengir/Dialect/HIVM/IR/HIVMVectorOps.td#L906-L942)
