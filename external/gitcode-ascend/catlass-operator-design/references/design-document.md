# Catlass 算子设计文档模板

设计文档用于向下游客观传递**「选了什么组件、为什么选、怎么组装」**，不要求可编译代码。代码细节由 `catlass-operator-code-gen` 完成。

---

## 1. 概述

| 项 | 内容 |
|----|------|
| 算子功能 | 一句话描述（如：矩阵乘 + GELU 融合） |
| 数学公式 | $D = \text{GELU}(A \times B)$ |
| 使用场景 | 说明适用的模型或场景 |

## 2. 输入输出信息表

| 变量名 | 数据类型 | Shape | 布局 | 描述 |
|--------|---------|-------|------|------|
| A | half | (m, k) | RowMajor | 输入矩阵 A |
| B | half | (k, n) | RowMajor | 输入矩阵 B |
| D | half | (m, n) | RowMajor | 输出矩阵 |

### 2.1 Gemm 与逻辑转置（涉及 A×B 时建议必写）

| 要点 | 说明 |
|------|------|
| 形状歧义 | 若 M、N、K 可取相同值，仅凭二维 shape **无法唯一**确定是否对 A/B 做了逻辑转置 |
| stride 与 Host | InferShape / Tiling **通常不可见** aclTensor stride，不应把「从 stride 推断布局」作为算子契约的一部分 |
| 推荐契约 | 用 **OpDef 布尔属性**（如 `transpose_a`、`transpose_b`）声明逻辑转置；设计文档给出 **属性 + 物理 shape → M、N、K** 的对照表；框架与 aclnn 调用方保证 **属性与数据布局一致** |

设计文档中应给出与 code-gen 一致的内维对齐表（示例，具体命名以本算子为准）：

| transpose_a | transpose_b | A 物理形状 | B 物理形状 | K 约束 |
|-------------|-------------|------------|------------|--------|
| false | false | (M, K) | (K, N) | A 列维 == B 行维 |
| true | false | (K, M) | (K, N) | A 行维 == B 行维 |
| … | … | … | … | … |

## 3. 核心组件选型（必写）

本节以**概念表格**描述各层组件的选型，不写代码。Agent 生成代码时按此表实例化模板参数。

### 3.1 硬件与架构

| 组件 | 选型 | 说明 |
|------|------|------|
| 目标芯片 | AtlasA2 | 对应 `Arch::AtlasA2` |

### 3.2 BlockMmad（块级矩阵乘）

| 组件 | 选型 | 说明 |
|------|------|------|
| DispatchPolicy | `MmadAtlasA2Pingpong` | 流水调度策略 |
| L1TileShape | `<128, 256, 256>` | L1 级分块 (M, N, K) |
| L0TileShape | `<128, 256, 64>` | L0 级分块 (M, N, K) |
| 输入类型 | AType: half, RowMajor | |
| 输入类型 | BType: half, RowMajor | |
| 输出类型 | CType: float, RowMajor | 中间计算用 float 保精度 |

> **代码映射提示**：code-gen 时将上表实例化为：
> `using DispatchPolicy = Gemm::MmadAtlasA2Pingpong<true>;`
> `using BlockMmad = Gemm::Block::BlockMmad<DispatchPolicy, L1TileShape, L0TileShape, AType, BType, CType>;`

### 3.3 BlockEpilogue（后处理）

如算子只有矩阵乘无后处理，写 `BlockEpilogue = void`。

否则列出后处理流水线中各环节：

| 顺序 | 环节 | 组件 | 说明 |
|------|------|------|------|
| 1 | 数据搬运入 | `TileCopy` | GM → UB |
| 2 | 计算 | `TileElemWiseGelu` | GELU 激活 |
| 3 | 数据搬运出 | `TileCopy` | UB → GM |

组合为 `BlockEpilogue<EpilogueDispatchPolicy, CType, DType, ...Tile组件>`。

> **代码映射提示**：上表中的组件按顺序填入 `BlockEpilogue` 模板参数列表。无后处理时直接写 `using BlockEpilogue = void;`。

#### 3.3.1 自定义 Tile Epilogue（catlass 无现成组件时）

在检索 [epilogue-components.md](./epilogue-components.md) 与 `catlass/include/catlass/epilogue/tile/` 后**仍无**匹配组件时，须按 [custom-epilogue.md](./custom-epilogue.md) 在设计文档中**单独**写明：自定义 Tile **名称**、**数学行为**、**DispatchPolicy**（NoSource / OneSource / TwoSource）、**computeLength 量级**、与 `TileCopy` 的组装顺序；不写可编译代码。下游 **code-gen** 在 `op_kernel/custom_epilogue/*.hpp` 实现该 Tile。

### 3.4 BlockScheduler

| 组件 | 选型 | 说明 |
|------|------|------|
| 调度器 | `GemmIdentityBlockSwizzle<3, 0>` | offset=3, direction=0 |

### 3.5 Kernel

| 组件 | 选型 | 说明 |
|------|------|------|
| Kernel 类型 | `Gemm::Kernel::BasicMatmul` | 纯矩阵乘 |
| | 或 `Gemm::Kernel::MatmulEpilogue` | 矩阵乘 + 后处理 |
| | 或 `Gemm::Kernel::MatmulActivation` | 矩阵乘 + 激活函数 |

组装关系：`Kernel = Kernel类型<BlockMmad, BlockEpilogue, BlockScheduler>`。

> **代码映射提示**：code-gen 时在 `TILING_KEY_IS` 分支内实例化：
> `using Kernel = Gemm::Kernel::BasicMatmul<BlockMmad, BlockEpilogue, BlockScheduler>;`
> `Kernel::Params params{...}; Kernel kernel; kernel(params);`

## 4. 参考 Example 与模板选型

| 项 | 内容 |
|----|------|
| 参考 example | `catlass/examples/00_basic_matmul` |
| 选型理由 | 功能最接近，可在此基础上增加 Epilogue |
| 模板类型 | Common 模板（M/N 方向分核） |

若无完全匹配，在此说明**变通方案**（如：基于 00_basic_matmul 增加 `TileElemWiseGelu` 后处理，参考 03_matmul_add 的 Epilogue 组装方式）。

## 5. TilingKey 分支设计

描述 Host 侧根据什么条件选择不同 TilingKey，Kernel 侧各 key 对应什么分支。

| TilingKey 值 | 条件 | Kernel 分支内容 |
|--------------|------|----------------|
| 0 | dtype=half, transA=N, transB=N | BasicMatmul + BlockMmad(Pingpong) |
| 1 | dtype=half, transA=N, transB=T | BasicMatmul + BlockMmad(转置B) |

每个 key 对应**一组模板实例化**（不同 dtype/转置/调度策略的组合）。

## 6. Workspace

| 项 | 内容 |
|----|------|
| 大小计算 | 从 `kernel/*.hpp` 的 `GetWorkspaceSize` 移植 |
| 用途 | 临时中间存储（如 SplitK 的 Reduce） |

## 7. 接口与使用

| 项 | 内容 |
|----|------|
| aclnn 接口 | 由框架自动生成 `aclnn<OpName>` |
| Host Tiling | 计算 TilingKey、填充 Tiling 数据、SaveToBuffer |
| Kernel 入口 | `Params{problemShape, gmA, layoutA, ...}; kernel(params);` |

## 8. 扩展性

列出后续可替换的组件（如更换激活函数、适配其他芯片等）。

## 9. 实现方案纲要

简要列出 Host 与 Kernel 各自做什么，引导 code-gen 而非写出完整代码：

**Host 侧**（op_host）：

- Tiling：解析 shape → 计算 TilingKey → SetTilingKey → SaveToBuffer
- InferShape：根据输入 shape 推导输出 shape
- OpDef：注册 Input/Output/DataType

**Kernel 侧**（op_kernel）：

- GET_TILING_DATA 取 tiling 数据
- TILING_KEY_IS 分支内实例化 Kernel 模板
- 调用 `kernel(params)` 执行

---

## 撰写原则

1. **用户层概念优先**：写清楚「输入是什么、输出是什么、选了什么模板、为什么选这个模板、TilingKey 怎么分」
2. **组件用表格描述**：用选型表格代替代码块，code-gen 按表格实例化
3. **不写可编译代码**：设计文档产出的是决策，代码由 code-gen 完成
4. **变通必须说明**：无完全匹配的 example 时，写明缺口和基于哪个 example 如何修改
