# hir.mmadL1 — 本地矩阵乘加（L1→L0C）

> 关键词：mmadL1, Matrix Multiply and Add, L1, L0C, Cube Core, UnitFlag, Fractal Layout

## 概述

`hir.mmadL1` 是 HIVM 方言中的本地矩阵乘加操作，在 Cube Core 上执行。该操作从 L1 存储层次读取矩阵 A 和 B，在 L0C 中执行乘加运算，结果写回 L0C。计算语义为 `C = C + A x B + (optional) channel_bias`。

该操作是 HIVM 宏操作体系中最基础的矩阵计算单元，对应硬件上的 mma_tile 指令。它涉及 MTE1（L1 数据搬运）和 M（Cube 矩阵计算）两个 Pipeline，需要跨 Pipe 同步。

mmadL1 支持转置加载（a_transpose/b_transpose）、HF32 加速模式、per-channel bias、以及 UnitFlag 同步条件等高级特性。

> Python API 对应：Triton 的 `tl.dot` 操作在 Split-K 场景下可能被映射为 mmadL1。

## IR 操作定义

从 [HIVMMacroOps.td](https://gitcode.com/Ascend/AscendNPU-IR/tree/master/bishengir/include/bishengir/Dialect/HIVM/IR/HIVMMacroOps.td#L152-L172) 提取：

```
def MmadL1Op : HIVM_LocalMmadOp<"mmadL1", [
  NoMaxRankTrait,
  DeclareOpInterfaceMethods<OpLayoutInterface, ["getOperandsTargetFractalLayout"]>,
  DeclareOpInterfaceMethods<HIVMStructuredOpInterface, ["getIndexingMaps"]>
]>
```

基类 `HIVM_LocalMmadOp` 定义（[HIVMMacroOps.td](https://gitcode.com/Ascend/AscendNPU-IR/tree/master/bishengir/include/bishengir/Dialect/HIVM/IR/HIVMMacroOps.td#L49-L150)）：

```
class HIVM_LocalMmadOp<string mnemonic, list<Trait> traits = []> :
  HIVM_MacroOp<mnemonic, !listconcat(
    [AttrSizedOperandSegments,
     CubeCoreTypeTrait,
     HIVMUnitFlagEnabledInterface,
     MacroOpPipeTrait<"PIPE::PIPE_MTE1, PIPE::PIPE_M">,
    ], traits)>
```

## 参数说明

### 输入操作数（ins）

| 参数 | 类型 | 必选 | 说明 |
|------|------|------|------|
| `$a` | TensorOrMemref | 是 | 矩阵 A，形状为 `[M, K]`，rank 必须为 2 |
| `$b` | TensorOrMemref | 是 | 矩阵 B，形状为 `[K, N]`，rank 必须为 2 |
| `$init_condition` | I1 | 是 | L0C 数据清零条件：为 true 时清零 L0C 后再使用 |
| `$real_m` | Index | 是 | M 维度的实际数据大小 |
| `$real_k` | Index | 是 | K 维度的实际数据大小 |
| `$real_n` | Index | 是 | N 维度的实际数据大小 |
| `$c` | TensorOrMemref | 是 | 矩阵 C（输出/累加），形状为 `[M, N]` |
| `$per_channel_bias` | TensorOrMemref | 否 | Per-channel bias，形状为 `[N]` |
| `$sync_related_args` | Variadic\<I64\> | 否 | 同步相关参数，由 InjectSync Pass 自动管理 |
| `$unit_flag_cond` | Variadic\<I1\> | 否 | UnitFlag 启用条件，用于循环依赖场景 |

### 输出操作数（outs）

| 参数 | 类型 | 说明 |
|------|------|------|
| `$result_tensors` | Variadic\<AnyRankedTensor\> | 结果 Tensor |

### 属性

| 属性 | 类型 | 必选 | 说明 |
|------|------|------|------|
| `$a_transpose` | UnitAttr | 否 | 矩阵 A 在加载前转置 |
| `$b_transpose` | UnitAttr | 否 | 矩阵 B 在加载前转置 |
| `$enable_HF32` | UnitAttr | 否 | 启用 HF32 模式：FP32 数据在 CUBE 计算前舍入为 HF32，性能翻倍但精度降低 |
| `$unit_flag_mode` | UnitFlagArrayAttr | 否 | 每个输出 Tensor 的 UnitFlag 模式 |

### 额外类方法

| 方法 | 返回类型 | 说明 |
|------|---------|------|
| `getOpName()` | StringRef | 返回 `"mma_tile"` |
| `isInitConstant(opt<bool>)` | bool | 查询/设置 init_condition 是否为常量 |
| `setInitCondition(Value)` | void | 设置 init 条件值 |
| `getMatmulBiasMode()` | MatmulBiasMode | 获取 bias 模式 |
| `shouldDecomposeBiasByElementAdd()` | bool | 判断是否应将 bias 分解为逐元素加法 |
| `getNumSyncRelatedArgs()` | int | 获取同步参数数量 |
| `getInputOperands(bool)` | SmallVector\<Value\> | 获取输入操作数 |
| `getOperandALayout()` | FailureOr\<DataLayoutAttr\> | 获取 A 的 Fractal Layout |
| `getOperandBLayout()` | FailureOr\<DataLayoutAttr\> | 获取 B 的 Fractal Layout |
| `getOperandCLayout()` | FailureOr\<DataLayoutAttr\> | 获取 C 的 Fractal Layout |
| `getOperandBiasLayout()` | FailureOr\<DataLayoutAttr\> | 获取 Bias 的 Fractal Layout |

## IR 示例

### 基本用法

```mlir
%ma = memref.alloc() : memref<256x128xf16>
%mb = memref.alloc() : memref<128x256xf16>
%mc = memref.alloc() : memref<256x256xf32>
%c256 = arith.constant 256 : index
%c128 = arith.constant 128 : index
%init = arith.constant 1 : i1
hivm.hir.mmadL1 ins(%ma, %mb, %init, %c256, %c128, %c256 :
                      memref<256x128xf16>, memref<128x256xf16>, i1, index, index, index)
                outs(%mc : memref<256x256xf32>)
```

### 带转置

```mlir
%ma_t = memref.alloc() : memref<128x256xf16>
hivm.hir.mmadL1 {a_transpose}
               ins(%ma_t, %mb, %init, %c256, %c128, %c256 :
                     memref<128x256xf16>, memref<128x256xf16>, i1, index, index, index)
               outs(%mc : memref<256x256xf32>)
```

### Split-K 循环中的条件初始化

```mlir
%mc = memref.alloc() : memref<256x256xf32>
%start = arith.constant 0 : index
%end = arith.constant 1024 : index
%step = arith.constant 128 : index
scf.for %arg0 = %start to %end step %step {
  %ma = memref.alloc() : memref<256x128xf16>
  %mb = memref.alloc() : memref<128x256xf16>
  %init_condition = arith.cmpi eq, %arg0, %start : index
  hivm.hir.mmadL1 ins(%ma, %mb, %init_condition, %c256, %c128, %c256 :
                        memref<256x128xf16>, memref<128x256xf16>, i1, index, index, index)
                  outs(%mc : memref<256x256xf32>)
}
```

### Tensor 语义

```mlir
%mc = tensor.empty() : tensor<256x256xf32>
%res = hivm.hir.mmadL1 ins(%ma, %mb, %init_condition, %c256, %c128, %c256 :
                             tensor<256x128xf16>, tensor<128x256xf16>, i1, index, index, index)
                       outs(%mC_iter : tensor<256x256xf32>) -> tensor<256x256xf32>
```

## IR 层约束与验证

1. **Rank 约束**：矩阵 A、B、C 的 rank 必须为 2（batchMmadL1 为 3）。
2. **Core Type**：操作必须在 Cube Core 上执行（`CubeCoreTypeTrait`）。
3. **Pipeline**：操作涉及 MTE1 和 M 两个 Pipeline（`MacroOpPipeTrait<"PIPE::PIPE_MTE1, PIPE::PIPE_M">`）。
4. **init_condition**：在 Split-K 循环中，首次迭代应设置 init_condition 为 true 以清零 L0C，后续迭代为 false 以累加。
5. **a_transpose / b_transpose**：当设置转置时，输入矩阵的逻辑形状不变，但数据在加载时按转置方式读取。
6. **enable_HF32**：仅对 FP32 数据有效，将 FP32 舍入为 HF32 后计算，精度降低但性能提升。
7. **per_channel_bias**：如果提供，形状必须为 `[N]`，与矩阵 B 的列维度匹配。
8. **Fractal Layout**：mmadL1 实现了 `OpLayoutInterface` 的 `getOperandsTargetFractalLayout` 方法，用于确定操作数的 Fractal 布局。

## 常见问题

**Q: mmadL1 和 matmul 的区别是什么？**
A: mmadL1 是本地操作，数据从 L1 读取到 L0C 计算；matmul 是全局操作，数据直接从 GM 读取。mmadL1 需要用户手动管理 L1 数据搬运和同步，matmul 由编译器自动处理。

**Q: init_condition 什么时候设为 true？**
A: 当需要清零 L0C 累加器时设为 true，通常在 Split-K 循环的第一次迭代。后续迭代设为 false 以累加部分和。

**Q: HF32 模式适用于什么场景？**
A: HF32 将 FP32 舍入为 19-bit（10-bit 尾数），适用于对精度不敏感但需要 FP32 吞吐量的场景。

## 相关文档

- 源码参考：[HIVMMacroOps.td](https://gitcode.com/Ascend/AscendNPU-IR/tree/master/bishengir/include/bishengir/Dialect/HIVM/IR/HIVMMacroOps.td#L49-L172)
- 测试用例：[ops.mlir](https://gitcode.com/Ascend/AscendNPU-IR/tree/master/bishengir/test/Dialect/HIVM/IR/ops.mlir)
- UnitFlag 详解：[04-Synchronization/03-unit-flag.md](../04-Synchronization/03-unit-flag.md)
- Fractal Layout：[06-Attributes-Types/02-parameterized-attrs.md](../06-Attributes-Types/02-parameterized-attrs.md)
