# hir.batchMmadL1 — 批量本地矩阵乘加

> 关键词：batchMmadL1, Batch Matrix Multiply and Add, L1, L0C, Cube Core

## 概述

`hir.batchMmadL1` 是 HIVM 方言中的批量本地矩阵乘加操作，在 Cube Core 上执行。与 `mmadL1` 类似，但支持批量维度，适用于批量矩阵乘法场景。计算语义为 `C[i] = C[i] + A[i] x B[i] + (optional) channel_bias`。

该操作从 L1 存储层次读取批量矩阵 A 和 B，在 L0C 中执行乘加运算。矩阵 A、B、C 的 rank 必须为 3，其中第 0 维为批量维度。

> Python API 对应：Triton 的 `tl.dot` 在 batch 维度场景下可能被映射为 batchMmadL1。

## IR 操作定义

从 [HIVMMacroOps.td](https://gitcode.com/Ascend/AscendNPU-IR/tree/master/bishengir/include/bishengir/Dialect/HIVM/IR/HIVMMacroOps.td#L174-L184) 提取：

```
def BatchMmadL1Op : HIVM_LocalMmadOp<"batchMmadL1",
  [NoLibraryFunctionTrait]> {
  let summary = [{
    Batch Matrix Multiply and Add Op with inputs from L1 memory hierarchy.
  }];
  let description = localMmadBaseDes # [{
    Note: the rank of A, B, and C Matrix must be three, where the 0-th dimension
    being the batch dimension.
  }];
}
```

## 参数说明

### 输入操作数（ins）

| 参数 | 类型 | 必选 | 说明 |
|------|------|------|------|
| `$a` | TensorOrMemref | 是 | 矩阵 A，形状为 `[Batch, M, K]`，rank 必须为 3 |
| `$b` | TensorOrMemref | 是 | 矩阵 B，形状为 `[Batch, K, N]`，rank 必须为 3 |
| `$init_condition` | I1 | 是 | L0C 数据清零条件 |
| `$real_m` | Index | 是 | M 维度的实际数据大小 |
| `$real_k` | Index | 是 | K 维度的实际数据大小 |
| `$real_n` | Index | 是 | N 维度的实际数据大小 |
| `$c` | TensorOrMemref | 是 | 矩阵 C（输出/累加），形状为 `[Batch, M, N]` |
| `$per_channel_bias` | TensorOrMemref | 否 | Per-channel bias |
| `$sync_related_args` | Variadic\<I64\> | 否 | 同步相关参数，由 InjectSync Pass 管理 |
| `$unit_flag_cond` | Variadic\<I1\> | 否 | UnitFlag 启用条件 |

### 输出操作数（outs）

| 参数 | 类型 | 说明 |
|------|------|------|
| `$result_tensors` | Variadic\<AnyRankedTensor\> | 结果 Tensor |

### 属性

| 属性 | 类型 | 必选 | 说明 |
|------|------|------|------|
| `$a_transpose` | UnitAttr | 否 | 矩阵 A 在加载前转置 |
| `$b_transpose` | UnitAttr | 否 | 矩阵 B 在加载前转置 |
| `$enable_HF32` | UnitAttr | 否 | 启用 HF32 模式 |
| `$unit_flag_mode` | UnitFlagArrayAttr | 否 | 每个输出 Tensor 的 UnitFlag 模式 |

### 与 mmadL1 的差异

| 特性 | mmadL1 | batchMmadL1 |
|------|--------|-------------|
| 矩阵 rank | 2 | 3（第 0 维为 batch） |
| A 形状 | `[M, K]` | `[Batch, M, K]` |
| B 形状 | `[K, N]` | `[Batch, K, N]` |
| C 形状 | `[M, N]` | `[Batch, M, N]` |
| OpLayoutInterface | 实现 `getOperandsTargetFractalLayout` | 未实现 |
| LibraryFunctionTrait | 默认支持 | `NoLibraryFunctionTrait` |
| OpName | `"mma_tile"` | 无（使用默认） |

## IR 示例

### 基本用法

```mlir
%ma = memref.alloc() : memref<2x256x128xf16>
%mb = memref.alloc() : memref<2x128x256xf16>
%mc = memref.alloc() : memref<2x256x256xf32>
%c256 = arith.constant 256 : index
%c128 = arith.constant 128 : index
%init = arith.constant 1 : i1
hivm.hir.batchMmadL1 ins(%ma, %mb, %init, %c256, %c128, %c256 :
                        memref<2x256x128xf16>, memref<2x128x256xf16>, i1, index, index, index)
                  outs(%mc : memref<2x256x256xf32>)
```

## IR 层约束与验证

1. **Rank 约束**：矩阵 A、B、C 的 rank 必须为 3，第 0 维为批量维度。
2. **Core Type**：操作必须在 Cube Core 上执行（`CubeCoreTypeTrait`）。
3. **Pipeline**：涉及 MTE1 和 M 两个 Pipeline。
4. **NoLibraryFunctionTrait**：该操作没有预定义的库函数实现，不支持直接 lowering 到标准库调用。
5. **批量维度一致性**：A、B、C 的批量维度大小应一致。

## 常见问题

**Q: batchMmadL1 和多次 mmadL1 有什么区别？**
A: batchMmadL1 在单个操作中处理整个批量，硬件可以利用批量间的数据局部性。多次 mmadL1 需要循环展开，每次处理一个批量元素，可能产生更多的同步开销。

**Q: 为什么 batchMmadL1 没有 OpLayoutInterface？**
A: 当前实现中 batchMmadL1 未实现 `getOperandsTargetFractalLayout`，其 Fractal Layout 推导使用默认逻辑。

## 相关文档

- 源码参考：[HIVMMacroOps.td](https://gitcode.com/Ascend/AscendNPU-IR/tree/master/bishengir/include/bishengir/Dialect/HIVM/IR/HIVMMacroOps.td#L174-L184)
- mmadL1 详解：[01-mmad-l1.md](01-mmad-l1.md)
- 测试用例：[ops.mlir](https://gitcode.com/Ascend/AscendNPU-IR/tree/master/bishengir/test/Dialect/HIVM/IR/ops.mlir)
