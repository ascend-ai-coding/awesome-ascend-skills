# HIVM Trait 定义

> 关键词：Trait, MacroOpTrait, MacroOpPipeTrait, CoreTypeTrait, ElementwiseNaryOpTrait, BroadcastableOTF, TransposableOTF

## 概述

HIVM Trait 是附加在操作上的行为约束标记，用于编译器 Pass 中的操作分类和验证。Trait 从 [HIVMTraits.td](https://gitcode.com/Ascend/AscendNPU-IR/tree/master/bishengir/include/bishengir/Dialect/HIVM/IR/HIVMTraits.td) 定义。

## Rank 相关 Trait

### HIVMOpSameOperandsAndResultRank

验证所有操作数和结果类型（除临时缓冲区外）的 rank 相同。

- **依赖**：HIVMStructuredOpInterface
- **适用操作**：大多数 Elementwise 操作

### OpLibraryMaxRankTrait\<MaxRank\>

指定库函数支持的最大 rank。

| 参数 | 说明 |
|------|------|
| MaxRank > 0 | 静态已知最大 rank |
| MaxRank = 0 | 静态未知最大 rank（需推断） |
| MaxRank = -1 | 无 rank 限制 |

### StaticMaxRankTrait\<MaxRank\>

继承自 `OpLibraryMaxRankTrait<MaxRank>`，用于静态已知最大 rank 的操作。

| 操作 | MaxRank | 说明 |
|------|---------|------|
| debug | 4 | 调试操作最多 4D |
| embedding_gather | 3 | 嵌入查找最多 3D |
| indirect_load / indirect_store | 5 | 间接加载/存储最多 5D |
| gatherT / scatterT / index_put | 5 | 收集/散射最多 5D |

### InferMaxRankTrait

继承自 `OpLibraryMaxRankTrait<0>`，用于静态未知最大 rank 的操作。

### NoMaxRankTrait

继承自 `OpLibraryMaxRankTrait<-1>`，无 rank 限制。

| 操作 | 说明 |
|------|------|
| mmadL1 | 矩阵乘加，rank 由操作数决定 |
| matmul / mix_matmul / mix_group_matmul | 全局矩阵乘，无 rank 限制 |

## Elementwise Trait

### ElementwiseNaryOpTrait\<N\>

N 元素逐元素操作 Trait。

**语义**：对 N 个输入操作数执行逐元素操作，产生单个结果。

**约束**：
1. 实现 DestinationStyleOpInterface
2. 输入操作数数量为 N
3. 所有 shaped 操作数和结果的 rank 相同
4. Buffer 语义时，最终维度的 stride 相等

**依赖**：HIVMStructuredOpInterface, HIVMOpSameOperandsAndResultRank

## 广播与转置 Trait

### BroadcastableOTF

支持内联广播的操作 Trait。

**语义**：
```
for i                              // <- broadcast dim
  for j
    dst[i, j] = some_op(src1[0, j], src2[0, j], ..., srcN[0, j])
```

**约束**：
1. 实现 DestinationStyleOpInterface
2. 必须有 `DenseI64ArrayAttr` 名为 `"broadcast"`
3. broadcast 维度唯一
4. broadcast 维度在 `[0, rank(dst))` 范围内
5. broadcast 维度上 `dim(src_i, d) = 1 || dim(src_i, d) = dim(dst, d)`
6. 非 broadcast 维度上 `dim(src_i, d) = dim(dst, d)`

**依赖**：HIVMStructuredOpInterface, HIVMOpSameOperandsAndResultRank

### TransposableOTF

支持内联转置的操作 Trait。

**语义**：
```
for i                              // transpose = [1, 0]
  for j
    dst[i, j] = some_op(src1[j, i], src2[j, i], ..., srcN[j, i])
```

**约束**：
1. 实现 DestinationStyleOpInterface
2. 必须有 `DenseI64ArrayAttr` 名为 `"transpose"`
3. transpose 是 `range(rank(dst))` 的排列
4. `transpose[rank(dst) - 1] = rank(dst) - 1`
5. `dim(dst, d) = dim(src_i, transpose[d])`

**依赖**：HIVMStructuredOpInterface, HIVMOpSameOperandsAndResultRank

## Pipe 相关 Trait

### SinglePipeOpTrait

标识操作为单 Pipe 操作。

### OpPipeTrait\<Pipe\>

参数化 Trait，声明操作在单个 Pipeline 上执行。

| 操作 | Pipe | 说明 |
|------|------|------|
| embedding_gather | PIPE_V | Vector Pipe |
| indirect_load | PIPE_V | Vector Pipe |
| indirect_store | PIPE_V | Vector Pipe |
| gatherT | PIPE_V | Vector Pipe |
| scatterT | PIPE_V | Vector Pipe |
| index_put | PIPE_V | Vector Pipe |
| custom | 由属性指定 | 用户指定 |

**依赖**：SinglePipeOpTrait

### MacroOpTrait

标识操作为宏操作（跨 Pipeline）。

### MacroOpPipeTrait\<InOutPipes\>

参数化 Trait，声明宏操作的输入/输出 Pipeline。

| 操作 | InOutPipes | 说明 |
|------|-----------|------|
| mmadL1 / batchMmadL1 | PIPE::PIPE_MTE1, PIPE::PIPE_M | L1 加载 + Cube 计算 |
| matmul / mix_matmul / mix_group_matmul | PIPE::PIPE_MTE2, PIPE::PIPE_MTE3 | GM 加载 + GM 写回 |

**依赖**：MacroOpTrait

## Core Type Trait

### CoreTypeTrait\<CoreType\>

参数化 Trait，静态声明操作的 Core 类型。

| Trait | CoreType | 适用操作 |
|-------|----------|---------|
| VectorCoreTypeTrait | TCoreType::VECTOR | 大多数 Vector 操作 |
| CubeCoreTypeTrait | TCoreType::CUBE | mmadL1, batchMmadL1 |
| CubeVectorCoreTypeTrait | TCoreType::CUBE_OR_VECTOR | get_block_idx, pointer_cast, set_ffts_base_addr |

## 其他 Trait

### NoLibraryFunctionTrait

标识操作没有预定义的库函数实现。

| 操作 | 说明 |
|------|------|
| batchMmadL1 | 无库函数，不支持直接 lowering |

### CommutativeOpTrait

标识操作的输入操作数可交换。

### VectorOnlyTrait\<idx\>

指定操作数只支持 Vector（shaped）类型输入。

### ScalarOnlyTrait\<idx\>

指定操作数只支持标量类型输入。

### OperElemTypeConstraints\<indices, types\>

指定操作数的元素类型约束。

### UniformReassociationFlattenTrait

标识操作可以统一展平维度。

**依赖**：HIVMStructuredOpInterface

### CollapsibleConsecutiveTargetDimsTrait

标识操作在展平时必须保留目标维度的独立性和 rank。

**依赖**：HIVMStructuredOpInterface, UniformReassociationFlattenTrait

## Trait 依赖关系

```
HIVMStructuredOpInterface
├── HIVMOpSameOperandsAndResultRank
│   ├── ElementwiseNaryOpTrait<N>
│   ├── BroadcastableOTF
│   └── TransposableOTF
├── UniformReassociationFlattenTrait
│   └── CollapsibleConsecutiveTargetDimsTrait
└── OpLibraryMaxRankTrait<MaxRank>
    ├── StaticMaxRankTrait<MaxRank>
    ├── InferMaxRankTrait
    └── NoMaxRankTrait

SinglePipeOpTrait
└── OpPipeTrait<Pipe>

MacroOpTrait
└── MacroOpPipeTrait<InOutPipes>
```

## 相关文档

- 源码参考：[HIVMTraits.td](https://gitcode.com/Ascend/AscendNPU-IR/tree/master/bishengir/include/bishengir/Dialect/HIVM/IR/HIVMTraits.td)
