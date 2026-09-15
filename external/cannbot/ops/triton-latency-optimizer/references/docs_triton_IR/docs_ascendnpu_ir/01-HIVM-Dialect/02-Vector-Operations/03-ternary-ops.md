# HIVM 三元向量运算

> 关键词：HIVM, Ternary, vsel, select, conditional

## 概述

HIVM 三元向量运算继承自 `HIVM_ElementwiseTernaryOp`，对三个输入操作数执行逐元条件选择运算。目前仅有一个操作 `hir.vsel`，它根据条件向量的值从两个数据源向量中选择元素。

> Python API 对应：`tl.where(condition, x, y)` — 条件选择操作。

## IR 操作定义

### 基类：HIVM_ElementwiseTernaryOp

```tablegen
class HIVM_ElementwiseTernaryOp<string mnemonic, list<Trait> traits = []> :
  HIVM_ElementwiseNaryOp<mnemonic,
                         !listconcat([ElementwiseNaryOpTrait<3>], traits)>;
```

源码参考：[HIVMVectorOps.td#L1016-L1018](https://gitcode.com/Ascend/AscendNPU-IR/tree/master/bishengir/include/bishengir/Dialect/HIVM/IR/HIVMVectorOps.td#L1016-L1018)

---

### hir.vsel — 逐元条件选择

#### TableGen 定义

```tablegen
def VSelOp : HIVM_ElementwiseTernaryOp<"vsel",
    [StaticMaxRankTrait<1>,
     OperElemTypeConstraints<[/*condition=*/0], [I1,I8]>,
     OperElemTypeConstraints<[/*src0=*/1, /*src0=*/2], [I1, AnyI8, AnyI16, F16, BF16, AnyI32, F32, I64]>,
     DeclareOpInterfaceMethods<ExtraBufferOpInterface, ["getExtraBufferSize"]>,
     BroadcastableOTF]> {
  let summary = "Elementwise Vector Selection Op";
  let description = baseClassDescription # [{
    Select elements from two source vector according to the binary `condition` vector.
    If the corresponding bit of the indicator is 1, select `src0`. Otherwise,
    select `src1`.

    Additional constraints:
      1. The input vectors and output vector must have the same ranks.
      2. The element type of indicator vector must be bool.
  }];
  let arguments = (ins Variadic<AnyType>:$src,
                       Variadic<AnyShaped>:$dst,
                       Optional<AnyMemRef>:$temp_buffer,
                       DefaultValuedAttr<DenseI64ArrayAttr, "{}">:$transpose,
                       DefaultValuedAttr<DenseI64ArrayAttr, "{}">:$broadcast
  );
  let assemblyFormat = [{
    attr-dict `ins` `(` $src `:` type($src) `)`
      `outs` `(` $dst  `:` type($dst) `)`
    (`temp_buffer` `(` $temp_buffer^ `:` type($temp_buffer) `)`)?
    (`->` type($result)^)?
  }];
}
```

源码参考：[HIVMVectorOps.td#L1020-L1050](https://gitcode.com/Ascend/AscendNPU-IR/tree/master/bishengir/include/bishengir/Dialect/HIVM/IR/HIVMVectorOps.td#L1020-L1050)

#### 参数说明

| 参数 | 类型 | 必选 | 说明 | 约束 |
|------|------|------|------|------|
| $src[0] | AnyType | 是 | 条件向量（condition/indicator） | 元素类型 I1 或 I8 |
| $src[1] | AnyType | 是 | 数据源 src0（条件为真时选择） | 元素类型 I1/AnyI8/AnyI16/F16/BF16/AnyI32/F32/I64 |
| $src[2] | AnyType | 是 | 数据源 src1（条件为假时选择） | 与 src0 相同元素类型 |
| $dst | Variadic\<AnyShaped\> | 是 | 输出向量 | 与 src0/src1 相同元素类型 |
| $temp_buffer | Optional\<AnyMemRef\> | 否 | 临时缓冲区 | ExtraBufferOpInterface |
| $transpose | DenseI64ArrayAttr (默认 {}) | 否 | OTF 转置维度 | - |
| $broadcast | DenseI64ArrayAttr (默认 {}) | 否 | OTF 广播维度 | - |

#### 数据类型约束

| 操作数位置 | 语义 | 支持的元素类型 | 约束来源 |
|-----------|------|--------------|----------|
| $src[0] | 条件 | I1, I8 | OperElemTypeConstraints<[0], [I1, I8]> |
| $src[1] | 数据源0 | I1, AnyI8, AnyI16, F16, BF16, AnyI32, F32, I64 | OperElemTypeConstraints<[1, 2], [I1, AnyI8, AnyI16, F16, BF16, AnyI32, F32, I64]> |
| $src[2] | 数据源1 | I1, AnyI8, AnyI16, F16, BF16, AnyI32, F32, I64 | OperElemTypeConstraints<[1, 2], ...> |

#### 语义说明

```
dst[i] = condition[i] ? src0[i] : src1[i]
```

当条件向量的对应位为 1 时，选择 src0 的元素；否则选择 src1 的元素。

#### IR 示例

```mlir
%result = hivm.hir.vsel ins(%cond, %src0, %src1 : tensor<23x77xi1>, f32, tensor<23x77xf32>) outs(%dst : tensor<23x77xf32>) -> tensor<23x77xf32>

%result = hivm.hir.vsel ins(%cond, %a, %b : i1, tensor<5x?x10xf32>, tensor<5x?x10xf32>) outs(%dst : tensor<5x?x10xf32>) -> tensor<5x?x10xf32>
```

完整使用示例（来自测试文件）：

```mlir
%cond = hivm.hir.vcmp ins(%a, %b : tensor<23x77xf32>, f32)
         outs(%init : tensor<23x77xi1>) compare_mode = <ne> -> tensor<23x77xi1>
%result = hivm.hir.vsel ins(%cond, %val_true, %val_false : tensor<23x77xi1>, f32, tensor<23x77xf32>)
           outs(%dst : tensor<23x77xf32>) -> tensor<23x77xf32>
```

## IR 层约束与验证

1. **条件类型约束**：条件向量（$src[0]）的元素类型必须为 I1 或 I8
2. **数据源类型一致性**：src0 和 src1 必须具有相同的元素类型
3. **Rank 一致性**：所有输入向量和输出向量必须具有相同的 rank
4. **最大 Rank 限制**：StaticMaxRankTrait<1>，仅支持 1 维
5. **BroadcastableOTF**：支持 OTF 广播，允许条件或数据源在指定维度上进行广播

## 与其他 IR 操作的关系

| HIVM 操作 | 上游降级 | HFusion 降级 | 说明 |
|-----------|---------|-------------|------|
| vsel | linalg.select | - | 语义等价于 linalg.select |

vsel 通常与 vcmp 配合使用，形成条件选择模式：

```
vcmp → vsel  (比较后选择)
```

降级示例：
```mlir
%cond = hivm.hir.vcmp ... compare_mode = <ne> -> tensor<Nxi1>
%result = hivm.hir.vsel ins(%cond, %a, %b : ...) -> tensor<Nxf32>

%result = linalg.select ins(%cond, %a, %b : tensor<Nxi1>, tensor<Nxf32>, tensor<Nxf32>) -> tensor<Nxf32>
```

## 常见问题

**Q: vsel 的条件向量为什么支持 I8 而不仅仅是 I1？**
A: 硬件实现中，条件判断可以基于 I8 类型的非零值，不仅仅是布尔值。这提供了更大的灵活性，允许直接使用比较结果或掩码向量。

**Q: vsel 支持标量条件吗？**
A: 是的。从 IR 示例可以看到，$src 支持 AnyType（而非仅 AnyShaped），因此条件可以是标量 `i1` 值。当条件为标量时，所有元素使用相同的条件值。

**Q: vsel 的最大 Rank 为什么只有 1？**
A: 这是当前硬件实现的限制。对于多维条件选择，需要先展平为 1 维操作，或在编译阶段通过循环分解处理。

## 相关文档

- Python API：docs_triton_ascend 中的 `tl.where()` 文档
- 源码参考：
  - [HIVMVectorOps.td - VSelOp](https://gitcode.com/Ascend/AscendNPU-IR/tree/master/bishengir/include/bishengir/Dialect/HIVM/IR/HIVMVectorOps.td#L1020-L1050)
  - [convert-hivm-to-upstream.mlir](https://gitcode.com/Ascend/AscendNPU-IR/tree/master/bishengir/test/ExecutionEngine/convert-hivm-to-upstream.mlir)
