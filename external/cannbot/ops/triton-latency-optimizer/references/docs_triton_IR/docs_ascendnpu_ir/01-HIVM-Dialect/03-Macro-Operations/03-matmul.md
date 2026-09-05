# hir.matmul — 全局矩阵乘（GM→GM）

> 关键词：matmul, Matrix Multiply, Global Memory, Descale, Swizzle, Tiling

## 概述

`hir.matmul` 是 HIVM 方言中的全局矩阵乘操作，直接从全局内存（GM）读取输入矩阵并执行矩阵乘法，结果写回 GM。计算语义为 `C = A * B`（无 bias/descale 时）或 `C = descale * (A * B + bias)`（有 bias/descale 时）。

该操作涉及 MTE2（GM 数据加载）和 MTE3（GM 数据写回）两个 Pipeline，由编译器自动管理 L1/L0 层次的数据搬运和同步。用户只需提供 GM 地址和 Tiling 参数，无需手动管理片上存储。

matmul 支持反量化（descale）、bias、转置、Swizzle 优化等特性，适用于大矩阵乘法场景。

> Python API 对应：Triton 的 `tl.dot` 在非 Split-K 场景下通常被映射为 matmul。

## IR 操作定义

从 [HIVMMacroOps.td](https://gitcode.com/Ascend/AscendNPU-IR/tree/master/bishengir/include/bishengir/Dialect/HIVM/IR/HIVMMacroOps.td#L236-L318) 提取：

```
def MatmulOp : HIVM_GlobalMmadOp<"matmul"> {
  let summary = "HIVM Matrix Multiply Op with inputs from global memory";
  let arguments = (ins AnyShaped:$a,
                       AnyShaped:$b,
                       Optional<AnyShaped>:$tilingParams,
                       Optional<AnyShaped>:$bias,
                       Optional<AnyShaped>:$descale,
                       OptionalAttr<UnitAttr>:$aTranspose,
                       OptionalAttr<UnitAttr>:$bTranspose,
                       OptionalAttr<HIVM_DescaleModeAttr>:$descaleMode,
                       Variadic<I64>:$blockSizes,
                       Variadic<I64>:$processSizes,
                       Optional<I64>:$swizzleOffset,
                       Optional<I64>:$swizzleDirection,
                       Optional<I64>:$epiloguePTiles,
                       AnyShaped:$c);
}
```

## 参数说明

### 输入操作数（ins）

| 参数 | 类型 | 必选 | 说明 |
|------|------|------|------|
| `$a` | AnyShaped | 是 | 矩阵 A，`m x k` |
| `$b` | AnyShaped | 是 | 矩阵 B，`k x n` |
| `$tilingParams` | AnyShaped | 否 | Tiling 参数 |
| `$bias` | AnyShaped | 否 | Bias 向量，形状为 `[n]` |
| `$descale` | AnyShaped | 否 | 反量化缩放因子，形状取决于 descaleMode |
| `$c` | AnyShaped | 是 | 矩阵 C（输出），`m x n` |

### 输出操作数（outs）

| 参数 | 类型 | 说明 |
|------|------|------|
| `$result` | Variadic\<AnyRankedTensor\> | 结果 Tensor |

### 属性

| 属性 | 类型 | 必选 | 说明 |
|------|------|------|------|
| `$aTranspose` | UnitAttr | 否 | 矩阵 A 转置加载 |
| `$bTranspose` | UnitAttr | 否 | 矩阵 B 转置加载 |
| `$descaleMode` | HIVM_DescaleModeAttr | 否 | 反量化模式 |

### I64 操作数

| 参数 | 类型 | 必选 | 说明 |
|------|------|------|------|
| `$blockSizes` | Variadic\<I64\> | 否 | M/N/K 维度在 L1 层次处理的数据块大小 |
| `$processSizes` | Variadic\<I64\> | 否 | M/N/K 维度在 L0 层次处理的数据块大小 |
| `$swizzleOffset` | I64 | 否 | Swizzle 调度的连续块编号 |
| `$swizzleDirection` | I64 | 否 | Swizzle 调度的块方向 |
| `$epiloguePTiles` | I64 | 否 | Epilogue 阶段一次处理的 P tile 数量 |

### DescaleMode 说明

| 模式 | 值 | descale 形状 | 说明 |
|------|---|-------------|------|
| DescaleNull | 0 | 无 | 不使用反量化 |
| DescalePerChannel | 1 | `[n]` | 按 Channel 反量化，形状等于 N |
| DescalePerTensor | 2 | `[1]` | 按 Tensor 反量化，形状为 1 |

### 额外类方法

| 方法 | 返回类型 | 说明 |
|------|---------|------|
| `getOpName()` | StringRef | 返回 `"matmul"` |
| `getDpsInitsMutable()` | MutableOperandRange | DestinationStyleOpInterface 所需 |

## IR 示例

### 基本矩阵乘

```mlir
func.func @test_matmul_basic(%A_gm : memref<16x16xf16, #hivm.address_space<gm>>,
                             %B_gm : memref<16x16xf16, #hivm.address_space<gm>>,
                             %res_gm : memref<16x16xf16, #hivm.address_space<gm>>) {
  hivm.hir.matmul
     ins(%A_gm, %B_gm:
         memref<16x16xf16, #hivm.address_space<gm>>, memref<16x16xf16, #hivm.address_space<gm>>)
     outs(%res_gm : memref<16x16xf16, #hivm.address_space<gm>>)
     descale_mode = #hivm.descale_mode<DescaleNull>
  return
}
```

### 带 Per-Channel Descale 和 Bias

```mlir
hivm.hir.matmul
     ins(%A_gm, %B_gm:
         memref<16x16xf16, #hivm.address_space<gm>>, memref<16x16xf16, #hivm.address_space<gm>>)
     outs(%res_gm : memref<16x16xf16, #hivm.address_space<gm>>)
     bias = %bias_gm : memref<16xf16, #hivm.address_space<gm>>
     descale = %descale_perchannel_gm : memref<16xf16, #hivm.address_space<gm>>
     descale_mode = #hivm.descale_mode<DescalePerChannel>
```

### 带 Per-Tensor Descale

```mlir
hivm.hir.matmul
     ins(%A_gm, %B_gm:
         memref<16x16xf16, #hivm.address_space<gm>>, memref<16x16xf16, #hivm.address_space<gm>>)
     outs(%res_gm : memref<16x16xf16, #hivm.address_space<gm>>)
     bias = %bias_gm : memref<16xf16, #hivm.address_space<gm>>
     descale = %descale_pertensor_gm : memref<1xf16, #hivm.address_space<gm>>
     descale_mode = #hivm.descale_mode<DescalePerTensor>
```

## IR 层约束与验证

1. **Core Type**：操作在 Cube Core 上执行，通过 `HIVMInferCoreTypeInterface` 推断。
2. **Pipeline**：涉及 MTE2 和 MTE3 两个 Pipeline。
3. **NoMaxRankTrait**：不限制操作数的最大 rank。
4. **Address Space**：输入/输出操作数通常需要 `#hivm.address_space<gm>` 标记。
5. **Descale 一致性**：当提供 descale 操作数时，descaleMode 必须与 descale 形状一致。
6. **Bias 形状**：bias 的形状必须为 `[n]`，与矩阵 B 的列维度匹配。
7. **blockSizes / processSizes**：通常为 3 个 I64 值，分别对应 M、N、K 维度的块大小。

## 常见问题

**Q: matmul 和 mmadL1 的主要区别？**
A: matmul 直接从 GM 读写数据，编译器自动管理 L1/L0 搬运和同步；mmadL1 需要用户手动管理 L1 数据搬运。matmul 适合端到端矩阵乘法，mmadL1 适合需要精细控制数据流的 Split-K 场景。

**Q: Swizzle 参数的作用？**
A: Swizzle 用于优化 GM 访存的 bank conflict，通过改变数据块的读取顺序来避免冲突。`swizzleOffset` 指定起始块编号，`swizzleDirection` 指定遍历方向。

**Q: descale 的计算公式？**
A: `C = descale * (A * B + bias)`，其中 descale 是反量化缩放因子，用于量化推理场景。

## 相关文档

- 源码参考：[HIVMMacroOps.td](https://gitcode.com/Ascend/AscendNPU-IR/tree/master/bishengir/include/bishengir/Dialect/HIVM/IR/HIVMMacroOps.td#L236-L318)
- 测试用例：[ops.mlir](https://gitcode.com/Ascend/AscendNPU-IR/tree/master/bishengir/test/Dialect/HIVM/IR/ops.mlir)
- DescaleMode 枚举：[06-Attributes-Types/01-enumerations.md](../06-Attributes-Types/01-enumerations.md)
