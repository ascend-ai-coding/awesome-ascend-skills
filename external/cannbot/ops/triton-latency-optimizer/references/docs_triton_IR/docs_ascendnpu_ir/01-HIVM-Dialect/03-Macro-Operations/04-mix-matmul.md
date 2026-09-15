# hir.mix_matmul — 混合 Cube+Vector 矩阵乘

> 关键词：mix_matmul, Mix Matrix Multiply, Post-Vector Function, Workspace, Communication

## 概述

`hir.mix_matmul` 是 HIVM 方言中的混合矩阵乘操作，在 Cube Core 执行矩阵乘法后，支持在 Vector Core 上执行后处理函数（post-vector function）。这种 Cube+Vector 的混合模式允许在 tile 级别融合后处理操作（如激活函数、类型转换等），避免额外的 GM 读写开销。

计算语义与 matmul 相同：`C = descale * (A * B + bias)`，但额外支持：
- `post_vector_func_ins`：Vector 后处理函数的输入参数
- `workspace_ins`：工作空间缓冲区
- `comm_params`：通信参数（用于融合通信操作，如 AllReduce）

> Python API 对应：Triton 的 `tl.dot` + 后续 elementwise 操作在编译时可能被融合为 mix_matmul。

## IR 操作定义

从 [HIVMMacroOps.td](https://gitcode.com/Ascend/AscendNPU-IR/tree/master/bishengir/include/bishengir/Dialect/HIVM/IR/HIVMMacroOps.td#L320-L422) 提取：

```
def MixMatmulOp : HIVM_GlobalMmadOp<"mix_matmul"> {
  let summary = "HIVM (Mix) Matrix Multiply Op with inputs from global memory";
  let arguments = (ins AnyShaped:$a,
                       AnyShaped:$b,
                       Variadic<AnyShaped>:$postVecFuncIns,
                       Variadic<AnyShaped>:$workspaceIns,
                       Optional<AnyShaped>:$tilingParams,
                       Optional<AnyShaped>:$commParams,
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
| `$postVecFuncIns` | Variadic\<AnyShaped\> | 否 | Vector 后处理函数的输入参数 |
| `$workspaceIns` | Variadic\<AnyShaped\> | 否 | 工作空间缓冲区输入 |
| `$tilingParams` | AnyShaped | 否 | Tiling 参数 |
| `$commParams` | AnyShaped | 否 | 通信相关参数（拓扑、通信器、group 等） |
| `$bias` | AnyShaped | 否 | Bias 向量 |
| `$descale` | AnyShaped | 否 | 反量化缩放因子 |
| `$c` | AnyShaped | 是 | 矩阵 C（输出） |

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
| `post_vector_func` | StrAttr | 否 | Vector 后处理函数名称（通过属性指定） |

### I64 操作数

| 参数 | 类型 | 必选 | 说明 |
|------|------|------|------|
| `$blockSizes` | Variadic\<I64\> | 否 | L1 层次 M/N/K 块大小 |
| `$processSizes` | Variadic\<I64\> | 否 | L0 层次 M/N/K 块大小 |
| `$swizzleOffset` | I64 | 否 | Swizzle 起始块编号 |
| `$swizzleDirection` | I64 | 否 | Swizzle 方向 |
| `$epiloguePTiles` | I64 | 否 | Epilogue P tile 数量 |

### 与 matmul 的差异

| 特性 | matmul | mix_matmul |
|------|--------|------------|
| 后处理融合 | 不支持 | 支持 `post_vector_func_ins` |
| 工作空间 | 不支持 | 支持 `workspace_ins` |
| 通信参数 | 不支持 | 支持 `comm_params` |
| OpName | `"matmul"` | `"mix_matmul"` |

## IR 示例

### 基本用法

```mlir
hivm.hir.mix_matmul
   ins(%A_gm, %B_gm:
       memref<16x16xf16, #hivm.address_space<gm>>, memref<16x16xf16, #hivm.address_space<gm>>)
   outs(%res_gm : memref<16x16xf16, #hivm.address_space<gm>>)
   tiling_params = %tiling_params_gm : memref<16xf16, #hivm.address_space<gm>>
   comm_params = %comm_params_gm : memref<16xi64, #hivm.address_space<gm>>
```

### 带 Post-Vector Function 和 Workspace

```mlir
hivm.hir.set_ffts_base_addr %arg0
hivm.hir.mix_matmul {post_vector_func = "bishengir_gen_vector_epilogue_func"}
  ins(%arg1, %arg2 :
      memref<1024x1024xf16, #hivm.address_space<gm>>, memref<1024x1024xf16, #hivm.address_space<gm>>)
  post_vector_func_ins(%arg3 : memref<1024x1024xf16, #hivm.address_space<gm>>)
  workspace_ins(%arg4 : memref<1024x1024xf16, #hivm.address_space<gm>>)
  outs(%arg5 : memref<1024x1024xf16, #hivm.address_space<gm>>)
  block_sizes(%c128_i64, %c256_i64, %c256_i64 : i64, i64, i64)
  process_sizes(%c128_i64, %c256_i64, %c64_i64 : i64, i64, i64)
  swizzle_offset = %c1_i64 : i64
  swizzle_direction = %c0_i64 : i64
  epilogue_p_tiles = %c4_i64 : i64
```

## IR 层约束与验证

1. **Core Type**：通过 `HIVMInferCoreTypeInterface` 推断，通常为 Cube Core 执行矩阵乘法部分。
2. **Pipeline**：涉及 MTE2 和 MTE3 两个 Pipeline。
3. **Post-Vector Function**：`post_vector_func` 属性指定 Vector 后处理函数名称，`postVecFuncIns` 提供其输入参数。
4. **Workspace**：`workspaceIns` 提供工作空间缓冲区，用于中间计算结果存储。
5. **Communication**：`commParams` 用于融合通信操作（如 AllReduce），包含拓扑、通信器等信息。
6. **MIX Kernel**：使用 mix_matmul 的函数通常标记为 `hivm.func_core_type = #hivm.func_core_type<AIV>` 或 `MIX`。

## 常见问题

**Q: mix_matmul 的 post_vector_func 是什么？**
A: 它是 Vector Core 上执行的后处理函数，可以对矩阵乘结果进行激活函数、类型转换等操作。函数名通过 `post_vector_func` 属性指定，输入通过 `postVecFuncIns` 传入。

**Q: workspace_ins 的用途？**
A: 工作空间缓冲区用于存储中间计算结果，例如在融合 AllReduce 时需要临时存储部分和。

**Q: 什么时候应该用 mix_matmul 而不是 matmul？**
A: 当需要在矩阵乘后立即执行 Vector 后处理（如激活、量化）或需要融合通信操作时，使用 mix_matmul 可以避免额外的 GM 读写。

## 相关文档

- 源码参考：[HIVMMacroOps.td](https://gitcode.com/Ascend/AscendNPU-IR/tree/master/bishengir/include/bishengir/Dialect/HIVM/IR/HIVMMacroOps.td#L320-L422)
- 测试用例：[ops.mlir](https://gitcode.com/Ascend/AscendNPU-IR/tree/master/bishengir/test/Dialect/HIVM/IR/ops.mlir)
- matmul 详解：[03-matmul.md](03-matmul.md)
