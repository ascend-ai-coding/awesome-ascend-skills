# HIVM 宏操作总览

> 关键词：MacroOp, MacroOpTrait, MacroOpPipeTrait, 跨 Pipeline, UnitFlag, 矩阵乘加

## 概述

HIVM 宏操作（Macro Operations）是 AscendNPU-IR 中表达跨 Pipeline 复合计算的核心抽象。与单 Pipe 操作不同，宏操作涉及多个硬件 Pipeline 之间的数据流动和同步协调，通常对应 NPU 上 Cube Core 与 Vector Core 之间的协作计算模式。

宏操作在 IR 层面通过 `MacroOpTrait` 和 `MacroOpPipeTrait` 两个关键 Trait 进行标记和约束：

- **MacroOpTrait**：标识该操作为跨 Pipeline 的宏操作，编译器在同步分析、Pipe 分配等 Pass 中会特殊处理此类操作。
- **MacroOpPipeTrait<InOutPipes>**：参数化 Trait，声明宏操作涉及的具体输入/输出 Pipeline 组合。例如 `MacroOpPipeTrait<"PIPE::PIPE_MTE1, PIPE::PIPE_M">` 表示该操作从 MTE1 Pipe 输入、从 M Pipe 输出。

宏操作体系在 HIVM 中分为两大类：

1. **本地矩阵乘加（Local MMAD）**：数据在片上存储层次（L1 → L0C）间流动，包括 `mmadL1` 和 `batchMmadL1`。
2. **全局矩阵乘（Global MMAD）**：数据从全局内存（GM）直接参与计算，包括 `matmul`、`mix_matmul`、`mix_group_matmul`。

> Python API 对应：Triton 的 `tl.dot` / `torch.matmul` 等操作在编译时可能被映射为 HIVM 宏操作。

## 宏操作类层次

```
HIVM_MacroOp (基类)
├── HIVM_LocalMmadOp (本地 MMAD 基类)
│   ├── MmadL1Op        -- hir.mmadL1
│   └── BatchMmadL1Op   -- hir.batchMmadL1
└── HIVM_GlobalMmadOp (全局 MMAD 基类)
    ├── MatmulOp          -- hir.matmul
    ├── MixMatmulOp       -- hir.mix_matmul
    └── MixGroupMatmulOp  -- hir.mix_group_matmul
```

## 核心 Trait 说明

### MacroOpTrait

标识操作为宏操作。编译器在以下场景中识别此 Trait：

- **同步注入（InjectSync）**：宏操作需要跨 Pipe 同步，InjectSync Pass 会自动在宏操作前后插入 `set_flag`/`wait_flag`。
- **Pipe 分配**：宏操作的 Pipe 信息由 `MacroOpPipeTrait` 提供，不同于单 Pipe 操作的 `OpPipeTrait`。
- **GraphSyncSolver**：基于图的同步求解器会将宏操作建模为多节点依赖图。

### MacroOpPipeTrait

参数化 Trait，格式为 `MacroOpPipeTrait<"PIPE::PIPE_IN, PIPE::PIPE_OUT">`，声明宏操作涉及的 Pipeline 组合：

| 操作 | MacroOpPipeTrait 参数 | 含义 |
|------|----------------------|------|
| mmadL1 / batchMmadL1 | `PIPE::PIPE_MTE1, PIPE::PIPE_M` | 数据从 L1 加载（MTE1），在 Cube Core 计算（M） |
| matmul | `PIPE::PIPE_MTE2, PIPE::PIPE_MTE3` | 数据从 GM 加载（MTE2），结果写回 GM（MTE3） |
| mix_matmul | `PIPE::PIPE_MTE2, PIPE::PIPE_MTE3` | 同 matmul，额外支持 Vector 后处理 |
| mix_group_matmul | `PIPE::PIPE_MTE2, PIPE::PIPE_MTE3` | 同 matmul，支持分组和 Vector 后处理 |

## UnitFlag 同步机制

宏操作支持 UnitFlag 同步模式，用于处理循环中"至少执行一次"的依赖场景。UnitFlag 有四种模式：

| 模式 | 值 | 说明 |
|------|---|------|
| DISABLED | 0 | 禁用 UnitFlag |
| RESERVED | 1 | 保留 |
| ENABLED_WITHOUT_UPDATE | 2 | 启用但不更新标志 |
| ENABLED_WITH_UPDATE | 3 | 启用并更新标志 |

在本地 MMAD 操作中，`unit_flag_cond` 参数提供可选的 i1 条件值，`unit_flag_mode` 属性指定每个输出 Tensor 的 UnitFlag 模式。

## 宏操作与单 Pipe 操作的对比

| 特性 | 单 Pipe 操作 | 宏操作 |
|------|------------|--------|
| Pipe 数量 | 单个 | 多个 |
| 标记 Trait | `SinglePipeOpTrait` + `OpPipeTrait` | `MacroOpTrait` + `MacroOpPipeTrait` |
| 同步需求 | Pipe 内同步 | 跨 Pipe 同步 |
| 典型操作 | load, store, vadd | mmadL1, matmul |
| DestinationStyleOpInterface | 支持 | 支持 |

## 操作列表

| 操作 | 助记符 | 说明 | 详细文档 |
|------|--------|------|---------|
| MmadL1Op | `hir.mmadL1` | 本地矩阵乘加（L1→L0C） | [01-mmad-l1.md](01-mmad-l1.md) |
| BatchMmadL1Op | `hir.batchMmadL1` | 批量本地矩阵乘加 | [02-batch-mmad-l1.md](02-batch-mmad-l1.md) |
| MatmulOp | `hir.matmul` | 全局矩阵乘（GM→GM） | [03-matmul.md](03-matmul.md) |
| MixMatmulOp | `hir.mix_matmul` | 混合 Cube+Vector 矩阵乘 | [04-mix-matmul.md](04-mix-matmul.md) |
| MixGroupMatmulOp | `hir.mix_group_matmul` | 分组矩阵乘（MoE） | [05-mix-group-matmul.md](05-mix-group-matmul.md) |

## 相关文档

- 源码参考：[HIVMMacroOps.td](https://gitcode.com/Ascend/AscendNPU-IR/tree/master/bishengir/include/bishengir/Dialect/HIVM/IR/HIVMMacroOps.td)
- 同步体系：[04-Synchronization/00-overview.md](../04-Synchronization/00-overview.md)
- 属性类型：[06-Attributes-Types/01-enumerations.md](../06-Attributes-Types/01-enumerations.md)
