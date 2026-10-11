# hir.mix_group_matmul — 分组矩阵乘（MoE 场景）

> 关键词：mix_group_matmul, Group Matmul, MoE, tokens_per_expert, Post-Vector Function

## 概述

`hir.mix_group_matmul` 是 HIVM 方言中的分组矩阵乘操作，专为 Mixture-of-Experts（MoE）场景设计。在 MoE 中，不同的 token 被路由到不同的 expert（权重矩阵），`mix_group_matmul` 通过 `tokens_per_expert` 参数指定每个 expert 处理的 token 数量，实现高效的分组矩阵乘法。

计算语义与 matmul 相同：`C = descale * (A * B + bias)`，但额外支持：
- `tokens_per_expert`：1D 向量，指定每个 expert 的 token 数量
- `post_vector_func_ins` / `post_vector_func_outs`：Vector 后处理函数的输入/输出
- `workspace_ins`：工作空间缓冲区
- `comm_params`：通信参数

> Python API 对应：MoE 模型中的分组矩阵乘操作。

## IR 操作定义

从 [HIVMMacroOps.td](https://gitcode.com/Ascend/AscendNPU-IR/tree/master/bishengir/include/bishengir/Dialect/HIVM/IR/HIVMMacroOps.td#L424-L535) 提取：

```
def MixGroupMatmulOp : HIVM_GlobalMmadOp<"mix_group_matmul"> {
  let summary = "HIVM (Mix) Matrix Group Multiply Op with inputs from global memory";
  let arguments = (ins AnyShaped:$a,           // weight, 3D
                       AnyShaped:$b,           // tokens, 2D
                       AnyShaped:$tokens_per_expert, // tokens_per_expert, 1D
                       Variadic<AnyShaped>:$postVecFuncIns,
                       Variadic<AnyShaped>:$postVecFuncOuts,
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
| `$a` | AnyShaped | 是 | 权重矩阵，3D 形状 `[num_experts, K, N]` |
| `$b` | AnyShaped | 是 | Token 矩阵，2D 形状 `[total_tokens, K]` |
| `$tokens_per_expert` | AnyShaped | 是 | 每个 expert 的 token 数量，1D 形状 `[num_experts]` |
| `$postVecFuncIns` | Variadic\<AnyShaped\> | 否 | Vector 后处理函数输入 |
| `$postVecFuncOuts` | Variadic\<AnyShaped\> | 否 | Vector 后处理函数输出 |
| `$workspaceIns` | Variadic\<AnyShaped\> | 否 | 工作空间缓冲区 |
| `$tilingParams` | AnyShaped | 否 | Tiling 参数 |
| `$commParams` | AnyShaped | 否 | 通信参数 |
| `$bias` | AnyShaped | 否 | Bias 向量 |
| `$descale` | AnyShaped | 否 | 反量化缩放因子 |
| `$c` | AnyShaped | 是 | 输出矩阵 C |

### 输出操作数（outs）

| 参数 | 类型 | 说明 |
|------|------|------|
| `$result` | Variadic\<AnyRankedTensor\> | 结果 Tensor |

### 属性

| 属性 | 类型 | 必选 | 说明 |
|------|------|------|------|
| `$aTranspose` | UnitAttr | 否 | 权重矩阵转置加载 |
| `$bTranspose` | UnitAttr | 否 | Token 矩阵转置加载 |
| `$descaleMode` | HIVM_DescaleModeAttr | 否 | 反量化模式 |

### I64 操作数

| 参数 | 类型 | 必选 | 说明 |
|------|------|------|------|
| `$blockSizes` | Variadic\<I64\> | 否 | L1 层次 M/N/K 块大小 |
| `$processSizes` | Variadic\<I64\> | 否 | L0 层次 M/N/K 块大小 |
| `$swizzleOffset` | I64 | 否 | Swizzle 起始块编号 |
| `$swizzleDirection` | I64 | 否 | Swizzle 方向 |
| `$epiloguePTiles` | I64 | 否 | Epilogue P tile 数量 |

### 与 mix_matmul 的差异

| 特性 | mix_matmul | mix_group_matmul |
|------|-----------|-----------------|
| 权重矩阵 A | 2D `[M, K]` | 3D `[num_experts, K, N]` |
| Token 矩阵 B | 2D `[K, N]` | 2D `[total_tokens, K]` |
| tokens_per_expert | 不支持 | 必需，1D `[num_experts]` |
| post_vector_func_outs | 不支持 | 支持 |
| OpName | `"mix_matmul"` | `"mix_group_matmul"` |

## IR 示例

### 基本用法

```mlir
hivm.hir.mix_group_matmul
   ins(%A_gm, %B_gm, %tokens_per_expert_gm:
       memref<16x16x16xf16, #hivm.address_space<gm>>,
       memref<16x16xf16, #hivm.address_space<gm>>,
       memref<16xi64, #hivm.address_space<gm>>)
   outs(%res_gm : memref<16x16xf16, #hivm.address_space<gm>>)
```

### 带 Post-Vector Function 和通信参数

```mlir
hivm.hir.mix_group_matmul
   ins(%A_gm, %B_gm, %tokens_per_expert_gm:
       memref<16x16x16xf16, #hivm.address_space<gm>>,
       memref<16x16xf16, #hivm.address_space<gm>>,
       memref<16xi64, #hivm.address_space<gm>>)
   post_vector_func_ins(%post_vector_func_ins : memref<1024x1024xf16, #hivm.address_space<gm>>)
   post_vector_func_outs(%post_vector_func_outs : memref<1024x1024xf16, #hivm.address_space<gm>>)
   outs(%res_gm : memref<16x16xf16, #hivm.address_space<gm>>)
   tiling_params = %tiling_params_gm : memref<16xf16, #hivm.address_space<gm>>
   comm_params = %comm_params_gm : memref<16xi64, #hivm.address_space<gm>>
```

## IR 层约束与验证

1. **Core Type**：通过 `HIVMInferCoreTypeInterface` 推断。
2. **Pipeline**：涉及 MTE2 和 MTE3 两个 Pipeline。
3. **权重矩阵维度**：A 必须为 3D，形状 `[num_experts, K, N]`。
4. **tokens_per_expert**：必须为 1D，长度等于 num_experts，所有元素之和应等于 total_tokens。
5. **Post-Vector Function**：同时支持输入和输出参数，允许后处理函数产生额外输出。
6. **NoMaxRankTrait**：不限制操作数的最大 rank。

## 常见问题

**Q: tokens_per_expert 的含义？**
A: 它是一个 1D 向量，`tokens_per_expert[i]` 表示第 i 个 expert 需要处理的 token 数量。编译器据此将 token 分配到不同的 expert 进行分组矩阵乘。

**Q: 为什么权重矩阵 A 是 3D 的？**
A: 在 MoE 场景中，每个 expert 有独立的权重矩阵。A 的第 0 维是 expert 数量，后两维是每个 expert 的权重矩阵。

**Q: mix_group_matmul 和多次 mix_matmul 的区别？**
A: mix_group_matmul 在单个操作中处理所有 expert 的分组计算，编译器可以优化数据搬运和同步。多次 mix_matmul 需要循环展开，可能产生更多开销。

## 相关文档

- 源码参考：[HIVMMacroOps.td](https://gitcode.com/Ascend/AscendNPU-IR/tree/master/bishengir/include/bishengir/Dialect/HIVM/IR/HIVMMacroOps.td#L424-L535)
- 测试用例：[ops.mlir](https://gitcode.com/Ascend/AscendNPU-IR/tree/master/bishengir/test/Dialect/HIVM/IR/ops.mlir)
- mix_matmul 详解：[04-mix-matmul.md](04-mix-matmul.md)
