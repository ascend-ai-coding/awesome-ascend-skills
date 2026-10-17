# MC2 算子生成能力规划

> 本文档是 MC2 算子生成能力的架构规划，记录通过 grilling + domain-modeling 会话达成的共识。
> 相关：[领域术语表](./CONTEXT.md) | [ADR-0001](./adr/0001-mc2-skill-not-plugin.md) | [ADR-0002](./adr/0002-decision-stack-path-registry.md)
> 2026-08-18 更新：能力模型由"三维轴矩阵"修订为"五层决策栈 + 路线登记表"（ADR-0002），落地路径明确为 PR-A（skill）→ PR-B（plugin）。

## 1. 能力域定义

MC2 算子生成能力按**五层决策栈**组织（详见 CONTEXT.md 与 ADR-0002）：

- **L0 需求层**：算子类型（collective-comm / moe）× 芯片（dav-3510 / dav-2201 …）× 调用形态（直调 / 注册）
- **L1 通信路径层**：UDMA（URMA）/ MTE通信（AIV+UBMEM）/ CCU（CCU+URMA，尚无直调参考工程）/ HCCL 高阶（仅注册）
- **L2 编程抽象层**：SHMEM 库 / APACE / 裸 Ascend C + compat 层 / HCCL 高阶 + Matmul 高阶（仅注册）。L1×L2 多对多，合法组合只能逐行登记
- **L3 工程组织层**：独立 CMake 工程 / 框架共享层 / 样例工程（被 L2 选定）
- **L4 流水编排层**：GET/PUT、flag 编排、tileCnt（部分被 L2 给定）

**路线（route）= 决策栈上的一条一致路线，按底座命名**，当前 supported 三条：blaze-shmem 路线（AIV+URMA × blaze-shmem 底座，3510）、apace 路线（AIV+URMA × apace 底座，3510）、ascendc-api 路线（HCCL window+MTE × ascendc-api 底座，3510+2201）。

两种工作模式：Greenfield（默认）、Brownfield（后续扩展，两个 plugin 各自的增量能力）。

## 2. 核心架构决策

### 2.1 不建独立 plugin（ADR-0001）

`ascendc-mc2-best-practice` 作为纯知识库 skill，被 ops-direct-invoke / ops-registry-invoke 两个 plugin 的 subagent 加载。MC2 横跨直调 + 注册两种形态，复用两个 plugin 的流程骨架。

### 2.2 流程归 plugin，领域知识归 skill（ADR-0001）

| 知识类型 | 归属 | 位置 |
|---------|------|------|
| 流程编排（何时拷问、门禁条件、阶段步骤、Step 映射） | plugin | AGENTS.md / task-prompts.md / workflow/SKILL.md |
| 领域知识（模板、API 用法、约束红线、设计模式、算子分类、拷问维度判据、自检标准） | skill | `ascendc-mc2-best-practice/references/` |

统一适用于需求分析、方案设计、开发、审查所有阶段。**SKILL.md 不出现 CANNBot Step 编号、route.json 等流程概念**（否则与未被 Step 1→7 约束的 plugin 不兼容）。

### 2.3 能力声明 = 路线登记表（ADR-0002）

`references/capability-declaration.md` 登记决策栈上的路径：每行 = chip × op_type × 调用形态 × 通信路径 × 编程抽象 + status + reference_impl + 知识目录。**调用形态当列**（为注册形态路径预留）；**含否定行**（unsupported + 原因，否决有据可查）。芯片知识复用 `npu-arch` skill。

### 2.4 需求分析知识合入 best-practice（单源）

PR #682（`ascendc-mc2-requirement-analysis`）的领域知识（REQUIREMENTS.md 模板、通信路径决策、算子分类速查、拷问 8 维度判据、自检 14 项清单）**单源**存放于 `references/requirement-analysis/`。plugin 侧不保留拷贝，task-prompts 逻辑引用 skill 路径（已有先例：developer agent 引用 `references/foundations/apace/development-guide.md`）。

### 2.5 Brownfield 是两个 plugin 各自的增量能力

代码分析规则（从代码推断路线坐标）作为领域知识放 skill（`references/codebase-analysis.md`，Stub），brownfield 流程编排放 plugin（后续扩展）。

## 3. Skill 内部结构（PR-A 落地后）

```
ops/ascendc-mc2-best-practice/
├── SKILL.md                          # 领域边界 + 触发信号 + 决策树 + 红线一句话版 + references 导航
├── docs/                             # 架构规划（本地，不入 git）
│   ├── CONTEXT.md                    # 领域术语表（决策栈模型）
│   ├── MC2-CAPABILITY-PLAN.md        # 能力规划（本文档）
│   └── adr/
│       ├── 0001-mc2-skill-not-plugin.md
│       └── 0002-decision-stack-path-registry.md
├── references/
│   ├── capability-declaration.md     # 路线登记表（含否定行）
│   ├── codebase-analysis.md          # brownfield：从代码推断路线坐标（Stub）
│   ├── requirement-analysis/         # 需求分析领域知识（单源）
│   │   ├── template.md               # REQUIREMENTS.md 模板（MC2 特有章节）
│   │   ├── comm-path-decision.md    # 通信路径选项（L1 层）
│   │   ├── classification.md         # 算子分类速查 + 可信源清单
│   │   ├── grill-protocol.md         # 需求拷问 8 维度判据
│   │   └── quality-checklist.md      # 文档自检 14 项 + 开发就绪闸门
│   ├── operators/                    # 算子类型跨底座共性设计模式
│   │   └── collective-comm.md        # 集合通信类：通信语义、轴切分、通算流水模式
│   ├── foundations/                  # 编码底座（L2 编程抽象的实体）
│   │   ├── blaze-shmem/              # blaze-shmem 路线知识（AIV+URMA × blaze-shmem 底座）
│   │   │   ├── workflow_integration.md   # Step 1→7 映射（ops-direct-invoke 场景）
│   │   │   ├── review-checklist.md       # 审查验收 R1-R7
│   │   │   ├── mc2_architecture.md / comm_shmem.md / matmul_blaze.md / codebase_map.md
│   │   │   └── all_to_all_matmul/        # 参考工程
│   │   ├── apace/                    # apace 路线知识（AIV+URMA × apace 底座）
│   │   │   ├── workflow_integration.md / review-checklist.md（R1-R8）
│   │   │   ├── architecture.md / compute.md / communication.md / fusion.md
│   │   │   ├── operator-anatomy.md / host-and-testing.md / development-guide.md
│   │   └── ascendc-api/              # ascendc-api 底座（裸 Ascend C 全自建）
│   │       └── moe-dispatch-combine/ # ascendc-api 路线知识（MTE通信）
│   │           ├── reading/ / samples/ / api-rules/ / tiling-scheme/
│   └── shared/                       # 跨底座共性（profiling_mc2.md、pipeline_tuning.md）
├── scripts/
│   └── fetch_apace.sh
└── evals/
    └── evals.json
```

## 4. 落地顺序

| 阶段 | 内容 | 状态 |
|------|------|------|
| PR-A | skill 重构（基于最新 main）：capability-declaration.md 路径登记模式 + requirement-analysis/ 合入（PR #682 领域知识）+ operators/collective-comm.md + review-checklist×2 + SKILL.md 重构（领域结构 + 红线一句话版 + 全量触发词）+ evals 补强 | 进行中 |
| PR-B | ops-direct-invoke 加 MC2 支持（greenfield）：subagent 加载 skill、Step 分支、多卡检查、task-prompts（逻辑引用 skill 单源的拷问协议，删除 plugin 内拷贝） | 依赖 PR-A；可改造自 PR #757 |
| C | ops-registry-invoke 加 MC2 支持（greenfield） | 依赖 PR-A |
| D | Brownfield 模式扩展（两个 plugin）+ codebase-analysis.md 填充 | 依赖 PR-B、C |
| E | 新芯片 / 新通信路径（CCU）/ 新编程抽象 / 新算子类型扩展；注册形态路线登记（如 apace × HCCL windows × 注册） | 按需 |

> PR #757（feat/mc2-capability-integration）基于 MTE 合入前的旧 base，其 SKILL.md 骨架化方案（按 Step 组织、route.json 流程红线、触发词缩减、双份拷问协议拷贝）与本规划冲突，已在 2026-08-18 设计讨论中否决；其 plugin 侧改动（agent 加载、多卡检查、task-prompts MC2 prompt）可作为 PR-B 的基础改造复用。

## 5. 扩展操作

| 扩展场景 | 操作 | 影响范围 |
|---------|------|---------|
| 新芯片验证通过 | 登记表加/改行 | 仅 capability-declaration.md |
| 新通信路径（如 CCU 直调落地） | L1 加取值 + 登记表加行 | capability-declaration.md + comm-path-decision.md |
| 新编程抽象（新框架/库） | L2 加取值 + 登记表加行 + 新建 `references/{新目录}/` | 登记表 + 一个知识目录 |
| 新算子类型 | L0 加取值 + 登记表加行 + `references/operators/{type}.md` | 登记表 + operators/ |
| 注册形态路径验证 | 登记表加行（调用形态=注册） | 仅 capability-declaration.md |
| 新工作模式（brownfield） | plugin 加流程分支；skill 填充 codebase-analysis.md | plugin + 一个 reference 文件 |
