# 能力模型：五层决策栈 + 路线登记表（替代三维轴矩阵）

## Context

ADR-0001 第 3 条将能力声明定义为 chip × framework × op_type 三维轴矩阵。实施时发现该模型装不下实际能力域：

1. **MTE通信（MoE 路径）无法入轴**。main 已合入 MoE dispatch/combine 体系（`b4e8f727`），其通信路径是"HCCL window 资源分配 + 裸 Ascend C（MTE/DataCopyPad）搬运 + 状态位协议"——它不是"基础框架"（软件实体），`framework ∈ {shmem, apace}` 的取值集合无法表达。
2. **三轴概念层级混杂**。SHMEM 是通信库（仅通信），APACE 是模板框架（通信+计算+工程组织，且基于 Ascend C API 构建），MTE通信是通信路径/设计模式——三者不是同类事物，压成一维取值概念上不成立。
3. **真实决策过程是多层的**。需求分析层本来就把"通信路径选型"（comm-path-decision）与"编程抽象选型"（SHMEM 库 vs APACE）分开决策；且 L1（通信路径）与 L2（编程抽象）是多对多关系（UDMA 上可跑 blaze-shmem 或 apace 底座；apace 底座可跑 UDMA 或 HCCL windows），轴的笛卡尔积会推导出大量不存在的组合。
4. **调用形态与否定项需要登记位置**。apace 的 HCCL windows 变体仅限注册场景；"直调 × HCCL 高阶"这类常见误答需要明确否决依据，防止模型自由发挥。

## Decision

1. **能力模型定义为五层决策栈**：L0 需求层（op_type × chip × 调用形态）→ L1 通信路径层（UDMA(URMA) / MTE通信(AIV+UBMEM) / CCU / HCCL 高阶）→ L2 编程抽象层（blaze-shmem / apace / ascendc-api / HCCL 高阶+Matmul 高阶）→ L3 工程组织层 → L4 流水编排层。术语定义见 CONTEXT.md。
2. **能力声明语义为"路线登记表"**（`references/capability-declaration.md`）：每行 = 决策栈上一条一致路线 + status + reference_impl + 知识目录。不再是轴的笛卡尔积格子。
3. **调用形态进表当列**（直调/注册），为 apace × HCCL windows × 注册 等路径预留登记位置。
4. **登记表含否定行**：明确不支持的组合（直调 × HCCL 高阶、直调 × HCCL windows/CCU、dav-2201 × AIV+URMA）+ 原因，否决有据可查。
5. **路线（route）= 决策栈路径的标签，按底座命名**，是知识库组织单元；当前 supported 路线三条：blaze-shmem 路线、apace 路线、ascendc-api 路线。"MTE通信"为 L1 通信路径名（AIV+UBMEM），不再作路线名；ascendc-api 路线特指裸 Ascend C API 全自建，与"基于 Ascend C API 构建模板库"的 apace 路线不混。
6. **L2 编程抽象的实体命名为"编码底座"，知识库以 `references/foundations/{底座名}/` 组织**：`shmem/` 更名 `foundations/blaze-shmem/`（旧名把 SHMEM 库与"SHMEM 库+Blaze 组装"的路线混为一谈），`apace/` 移入 `foundations/apace/`，`moe-dispatch-combine/` 移入 `foundations/ascendc-api/`（ascendc-api 路线的底座即裸 Ascend C API）。底座目录直接承载该底座上的路线知识，当前 1:1，未来同底座多路线时以子目录区分。

## Considered Options

- **一维 route 路由标签（值即目录名）**：被否。承认通信路径×底座×工程组织不可分割，实用但概念混杂；且未来"同通信路径长出新底座"（如 mte-window 上的框架）时需要造复合值，扩展性差。
- **保留 framework 轴仅扩值（+mte-window）**：被否。MTE通信不是框架（无软件实体），撑破 CONTEXT.md"基础框架"定义；与 apace 的 HCCL windows 模式撞词。
- **两级四维矩阵（chip × mechanism × framework × op_type）**：部分采纳——决策栈吸收了其正交性洞察（L1/L2 分层），但矩阵形态改为路线登记表，避免稀疏矩阵的非法组合推导问题，且调用形态必须入表。

## Consequences

- ADR-0001 第 3 条（matrix.yaml 三维矩阵）被本条替代；其余各条（不建独立 plugin、流程归 plugin 领域归 skill、需求分析知识合入、Brownfield 增量）继续有效。
- 扩展语义清晰：新芯片 = 加/改行；新通信路径 = L1 加取值 + 加行；新编程抽象 = L2 加取值 + 加行 + `references/foundations/{新底座}/`；新算子类型 = L0 加取值 + 加行；注册形态路径 = 加行。
- `comm-path-decision.md` 对齐为 L1 通信路径层选项说明（补 MTE通信）。
- 需求文档模板 §6.2 通信引擎选项同步补 MTE通信 / CCU。
- 目录重构：`references/{shmem,apace,moe-dispatch-combine}/` → `references/foundations/{blaze-shmem,apace,ascendc-api/moe-dispatch-combine}/`，全量链接同步修复；能力登记表的"编程抽象"列值即底座目录名。
