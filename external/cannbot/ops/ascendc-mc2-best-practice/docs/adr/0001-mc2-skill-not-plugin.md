# MC2 算子生成能力：不建独立 plugin，知识归 skill，流程归 plugin

## Context

MC2 算子生成能力需要覆盖多芯片型号、多通信路径与编程抽象、多算子类型的组合（非正交）、两种工作模式（greenfield 新算子生成 + brownfield 已有算子拓展/优化）、两种工程形态（Kernel 直调 + 注册算子）。现有 `ascendc-mc2-best-practice` skill 已承载 SHMEM/apace 两条路线的设计开发知识。

> 注：本文的"三维轴矩阵"能力模型已被 ADR-0002 的"五层决策栈 + 路线登记表"替代（仅第 3 条受影响）。

## Decision

1. **不建独立 plugin。** `ascendc-mc2-best-practice` 作为纯知识库 skill，被 ops-direct-invoke（直调）和 ops-registry-invoke（注册）两个现有 plugin 的 subagent 加载。MC2 横跨两种工程形态，复用两个 plugin 的流程骨架。

2. **流程性知识归 plugin，领域知识归 skill。** 需求分析的拷问协议/自检清单/开发就绪闸门、方案设计的流程步骤/门禁条件——这些是 Agent 的决策职责，放 plugin 的 workflow/task-prompts。REQUIREMENTS.md 模板、DESIGN.md 模板、通信路径选项、框架约束、算子设计模式——这些是领域知识，放 skill 的 references。

3. **能力声明放 skill 内部。** ~~`references/matrix.yaml` 声明 chip×framework×op_type 的可用组合~~ **【已被 ADR-0002 替代】** `references/capability-declaration.md` 以路线登记表形式声明决策栈上的已验证/规划/不支持路径，Architect 加载 skill 后查询。芯片知识复用 `npu-arch` skill，不重复。

4. **需求分析知识合入 best-practice。** PR #682（`ascendc-mc2-requirement-analysis`）中的领域知识（模板、通信路径决策、算子分类速查）合入 `ascendc-mc2-best-practice/references/requirement-analysis/`；流程性知识（拷问协议、自检清单）留给 plugin。

5. **Brownfield 是两个 plugin 各自的增量能力。** 代码分析规则（从代码推断路线坐标）作为领域知识放 skill（`references/codebase-analysis.md`），brownfield 流程编排放 plugin。

## Considered Options

- **独立 plugin（ops-mc2-generator）**：被否。MC2 与通用直调/注册算子共享 7 步流程骨架，独立 plugin 会复制流程编排。MC2 的特殊性在领域知识层，不在流程层。
- **按轴拆分多个 skill（chip/framework/operator 各一个）**：被否。框架知识与算子设计模式高度耦合（如 apace 的通算流水编排不可拆分为"框架通用知识"+"算子通用知识"的简单叠加），拆分后交叉引用复杂。
- **需求分析独立 skill（PR #682 现状）**：被否。需求分析阶段的 MC2 领域知识（通信路径选项、算子分类）与开发阶段的领域知识有重叠和依赖，独立 skill 会导致知识漂移或跨 skill 引用。

## Consequences

- `ascendc-mc2-best-practice` 承载量增大（需求+设计+开发+审查+brownfield 全链路领域知识），通过 references 目录分层和渐进式披露控制。
- 两个 plugin 需各自增加 MC2 特化分支（task-prompts / workflow），流程编排维护在 plugin 侧。
- 扩展新路径时，领域知识改 skill，流程不变；扩展新工作模式时，流程改 plugin，领域知识不变。
