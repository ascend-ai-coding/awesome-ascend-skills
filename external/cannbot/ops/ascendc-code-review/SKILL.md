---
name: external-cannbot-ops-ascendc-code-review
description: Ascend C 代码检视技能。触发：检视代码、检视 PR、检查是否有问题、快速检视。支持文件检视、PR 检视、快速定向检视、设计一致性检查。自动识别代码侧别、提取适用条例、执行假设检验驱动的逐条检视。
permission:
  external_directory: allow
original-name: ascendc-code-review
synced-from: https://gitcode.com/cann/cannbot-skills
synced-date: '2026-11-14'
synced-commit: 750045df251d031bf708de9dc1614c2202eed6dd
license: UNKNOWN
---

# Ascend C 代码检视

## 工作流路由

根据用户意图和输入类型，选择对应 workflow 文件执行。

仅咨询检视原理、假设检验、置信度或证据要求时，读取 `core/methodology.md` 回答，无需启动代码检视工作流。说明判定方法时应先交代代码段识别及分析范围，再说明假设、证据和结论，避免脱离具体代码单元讨论风险评分。

| 触发条件 | 工作流文件 |
|---------|---------|
| 全量检视代码、审核代码、代码审查、帮我检视 xxx、设计实现一致性、设计一致性检查、对照 DESIGN.md | workflows/file-review.md |
| 检查是否有、有没有问题、快速检视、有什么风险、帮我看看有没有*问题、是否存在.*问题 | workflows/quick-review.md |
| 全量检视 PR、全面审核 PR、pr #、pull request、PR # | workflows/pr-review.md |

## 工作流概览

执行主体分工：主 Agent 负责编排（提取输入、派发子 Agent、传递上下文、直接执行 plan-design），取数/逐条检视由子 Agent 执行（quick-review 除外，主 Agent 直接执行）。

| 工作流 | 阶段链 |
|--------|--------|
| PR 检视 | 阶段0 从用户输入提取 PR 链接（解析 repo_url 与 pr_number，判断托管平台，仅支持 GitCode）→ **派发 code-fetch 子 Agent** 获取 diff 与完整源码（主 Agent 不直接执行脚本）→ plan-design（主 Agent 直接执行：**侧别识别**（`op_kernel/`→Kernel，`op_host/`/`op_graph/`/`op_api/`→Host，输出 Kernel/Host/混合）+ **文档级检视分组**）→ 阶段1 单波派发通用检视子 agent（每文件一个，全量并行）→ 阶段2 行号校对 → 阶段3 撰写报告 → 阶段4 可选行内评论 |
| 文件检视 | plan-design（主 Agent 直接执行：侧别识别 + 文档级检视分组）→ 单波逐条检视 → 行号校对 → 撰写报告（无 code-fetch 阶段，直接检视本地代码） |
| 快速检视 | 主 Agent 直接执行定向检查，无阶段、无子 Agent |

## 路径约定

`{skill_base}` = 本 skill 的根目录绝对路径（即本 SKILL.md 所在目录）。workflows/ 与 steps/ 中所有 `{skill_base}/...` 引用均以此展开。

子 Agent 的工作目录是被检代码仓而非 skill 目录，故**派发子 Agent 时必须在 prompt 的上下文参数中显式传入 `{skill_base}` 的实际路径**，不得依赖相对路径解析。

## 交付必列项

回答或报告中，涉及以下主题时必须逐项覆盖，漏列即视为缺项：

- **错误码**：引用依据写明 cann-error-code-standard 的 SKILL.md、references/development-standard.md、references/documented-error-codes.json 三件；替换建议先声明选路（单参数 / 跨参数 / expected-reason）与版本可见性前提，再给示例。
- **MC2 专家索引**：给出上下游校验判断条件——上游校验是否覆盖同一个 ID、使用前是否改值；材料不足时列为待核实缺口，不推定已完成。

## 执行规则

1. Read 对应 workflow 文件，获取编排定义
2. 严格按 workflow 的阶段顺序执行，禁止跳步
3. 每个阶段开始时 Read 对应 steps/ 文件，执行完成后再 Read 下一个
4. 禁止提前 Read 未执行阶段的 step 文件（上下文隔离）

## 资源索引

| 资源 | 路径 | 说明 |
|------|------|------|
| 工作流编排 | workflows/ | 定义阶段顺序、上下文传递、任务追踪 |
| 执行步骤 | steps/ | 每步的完整操作指令 |
| 检视方法论 | core/methodology.md | 假设检验流程、置信度标准、红线问题 |
| 编码规范 | references/*.md | 安全编码、API 最佳实践、性能、TOPK 等规则文档 |
| 规则集总览 | references/README.md | 全部规则文件的分类目录（仅供人读，机器路由不依赖） |
| 错误码检测 | references/ascendc-error-code.md | ERR-1~4 默认纳入 C/C++ 检视；引用相邻 cann-error-code-standard 的规范与码表，检查废弃码、注册契约、首报与失败返回 |
| 废弃码扫描 | scripts/clause.check_error_codes.py | 只读扫描被检代码中的废弃码及旧宏，作为 ERR-1 的检测证据 |
| Skill 自身规范 | docs/architecture.md | 目录结构、各类文件格式规范、扩展方式（维护本 skill 时读） |
| 规则集写作规范 | docs/rule-writing-standard.md | 新增/修改规则文件的逐条写作约束与提交前自查清单（维护规则集时读） |
| 常用算子仓 | ops-transformer / ops-math / ops-nn / ops-cv | gitcode.com/cann/{repo}，PR 检视时优先从完整 URL 推断 repo 名 |
