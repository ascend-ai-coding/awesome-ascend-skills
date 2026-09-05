---
name: external-cannbot-infra-gitcode-issue-handler
description: GitCode Issue 端到端处理工具。支持显式 Issue URL 的单 Issue 处理，也支持获取、分类和批量处理当前仓库 Issue；建立首响与解决时钟，在复现前核对
  CANN/源码/SoC 环境，把算子类 Issue 指派给已配置或由用户补充的算子责任人， 持续跟踪维护者首响后等待提出者补充或等待责任人处理、以及任一方再次回复的
  Issue， 检索本地处理知识，稳定复现并确认根因后实施最小修复， 运行相关测试和 NPU 质量门禁后创建 PR 或直接推送。触发：用户要求处理仓库 Issue、
  跟进 Issue、自动修复 Issue、从 Issue 创建 PR、批量分诊 Issue，或仅回复/答疑/不改代码。 每次主处理运行结束时仅对本轮实际处理的 Issue
  生成精简结果报告。
license: CANN-2.0
original-name: gitcode-issue-handler
synced-from: https://gitcode.com/cann/cannbot-skills
synced-date: '2026-09-05'
synced-commit: a426ec91e1038f233066724d63235c719a46a10d
---

# GitCode Issue Handler

把本 Skill 作为 Issue 处理状态机。`SKILL.md` 只维护入口、跨阶段门禁和阶段路由；进入阶段
前完整读取对应的一级 reference，以 reference 为该阶段的单一事实源。

## 目标与入口

流程服务于解决率 `> 90%`、平均解决时长 `< 7 个自然日`、`1 个工作日`内有效响应；不得
把等待、转交或未合入 PR 虚报为解决。指标口径见
[references/delivery-reporting.md](references/delivery-reporting.md)。

首先判断请求是否只在询问规则、流程、模式差异或门禁。这类 `policy_query`
直接依本文已有规则回答；不初始化运行状态，不读取执行阶段 reference，不检查
Token、Git、临时目录、git author 或 CANN 环境，也不访问 Issue/API。只有用户
明确要求对真实 Issue 执行获取、诊断、回评、修复或发布时，才进入下述运行模式。

| 模式 | 触发与边界 |
| --- | --- |
| `single` | 显式给出一个 GitCode Issue URL；只处理该 Issue，不受批量时间窗、作者和 `no_attention` 过滤 |
| `batch` | 处理或分诊当前仓库 Issue；以启动目录为仓库根，只推进 `need_attention` |
| `auto-close-stale` | 明确要求检查长期无响应的已答复咨询 Issue；走独立维护路径，默认 dry-run |

用户明确要求“只回复 / 答疑 / 评论回复 / 不改代码”时，把 `single` 限制为文字诊断与回评，
不创建分支、commit、push 或 PR。

## 核心门禁

1. `single` 的显式目标必须进入诊断；`batch` 只推进 `need_attention`。自动闭环只按
   [references/maintenance-stale-close.md](references/maintenance-stale-close.md) 执行。
2. 明确算子 Issue 先有效首响再转交责任人；责任人未知时收集 owner 或当前 Issue 的
   `direct` 决定，不得静默自行处理。等待提出者或责任人时，经批准将 GitCode 自定义状态
   改为`挂起`并建立对应 watch；任一方再回复后恢复`进行中`并重新处理。
3. 修改源码前必须通过目标环境一致性门禁、稳定复现并确认最终根因和方案；只实施与根因
   直接相关的最小修改，并完成相关质量门禁或明确记录 `degraded_validation`。
4. 每个修复组只在 manifest 管理的独立 worktree 中工作。禁止覆盖用户改动、破坏性 Git
   恢复、force push、自动合并 PR、强制删除 worktree，以及 `git add -A` / `git add .`。
5. `single` 和 `batch` 均从 `interactive` 开始。任何 GitCode 写入、暂存、commit、push、
   PR 或首次 CI 前，必须基于实际 diff 和验证结果完成一次统一执行确认。预览获批后，
   `batch` 才从 `interactive` 切换为 `approved_batch`；`single` 保持 `interactive` 并使用
   该检查点证据。direct push 在 commit 形成后另行确认。
6. 按 `runtime-state.md` 维护唯一运行状态；每次分类、诊断、复现、修改、验证、授权、
   发布、回评或状态转换后立即追加必要结果和证据，不在步骤 9 凭记忆重建过程。
7. 单个 Issue、PR 或非安全卡点不得暂停其他独立工作；按
   [references/policy-error-handling.md](references/policy-error-handling.md) 记录等待、
   降级和硬卡点。
8. 状态、指标和外部写入必须有可核验证据；外部评论不得包含 Token、绝对路径、维护侧
   环境、CI/基础设施或内部重试。每次主流程结束都生成仅含本轮实际处理 Issue 的报告。
9. 一旦已确定后续需要认证 GitCode 写操作，而用户消息和 `GITCODE_TOKEN` 均未提供
   Token，立即保存 `waiting_for_input` 断点，只汇总询问一次并停止本轮。等待期间不得
   继续 API 探测、真实 Issue 拉取或测试框架读取；用户补充 Token 后从断点继续，不重复询问。

## Reference 读取路由

| 时机 | 必须完整读取 |
| --- | --- |
| `policy_query` | 无；仅使用本文的入口、核心门禁和阶段路由 |
| 执行步骤 -1、0 | [runtime-setup.md](references/runtime-setup.md)、[runtime-state.md](references/runtime-state.md)、[runtime-capability-checks.md](references/runtime-capability-checks.md) 和 [authorization-contract.md](references/authorization-contract.md) |
| 步骤 0a | [runtime-knowledge.md](references/runtime-knowledge.md) |
| 步骤 1、2a、2b | [issue-intake.md](references/issue-intake.md)；开始步骤 2 前再读跨阶段 [policy-error-handling.md](references/policy-error-handling.md)，持续遵守到步骤 9 |
| 步骤 2c–2e | [issue-routing.md](references/issue-routing.md)；形成评论草案时读 [issue-comment-workflow.md](references/issue-comment-workflow.md)；进入等待或处理再次回复时再读 [issue-followup.md](references/issue-followup.md) |
| 分组完成后 | [code-worktree.md](references/code-worktree.md) |
| 步骤 3–4 | [code-root-cause.md](references/code-root-cause.md)；需要精确追溯时再读 [code-git-history.md](references/code-git-history.md) |
| 步骤 5–6 | [code-validation.md](references/code-validation.md) |
| 步骤 6f | [delivery-confirmation.md](references/delivery-confirmation.md) |
| 步骤 7–8 | [delivery-publish.md](references/delivery-publish.md) |
| 步骤 9 | [delivery-reporting.md](references/delivery-reporting.md) |

GitCode Token、API、评论 POST/GET、PR 和 CI 接口语义以同级 `gitcode-toolkit` 为真源；
评论业务编排与聚合授权只以本 Skill 的一级 references 为真源。安装、依赖和配置见
[docs/installation-guide.md](docs/installation-guide.md)。确定性操作优先使用 `scripts/`。

## 工作流程

```text
- [ ] 步骤 -1：区分 policy_query 与真实执行；仅后者初始化运行状态
- [ ] 步骤 0：解析目标，在首次 API/Git/tmp 操作前分别执行所需能力检查
- [ ] 步骤 0a：刷新运行时历史证据；失败时按可信快照降级
- [ ] 步骤 1：获取 Issue 并建立服务时钟
- [ ] 步骤 2：分类、首响预览、诊断、拟方案和并行调度
- [ ] 步骤 3：环境一致性门禁与稳定复现
- [ ] 步骤 4：确认最终根因和修改方案
- [ ] 步骤 5：实施最小代码变更
- [ ] 步骤 6：运行质量门禁
- [ ] 步骤 6f：展示全部待执行操作并取得分析后统一执行确认
- [ ] 步骤 7：按统一执行确认提交变更
- [ ] 步骤 8：创建 PR，或单独确认后直接推送
- [ ] 步骤 9：汇报、回评和清理
```

### 阶段 A：初始化、同步与知识刷新

非 `policy_query` 才按 `runtime-state.md` 初始化唯一运行状态，再按
`runtime-setup.md` 解析目标，并按 `runtime-capability-checks.md` 调用本 Skill 的
`scripts/preflight.sh`，在第一个真实 API、Git 或临时落盘操作紧前执行对应能力检查。
只更新远程跟踪引用，不切换或修改用户原始工作区。随后按
`runtime-knowledge.md` 执行步骤 0a。知识刷新失败不阻断主流程，只能使用上次校验
通过的快照或受审卡片。对应 API/Git 能力就绪并记录状态后才执行相应远程操作。
API 能力因缺 Token 进入 `waiting_for_input` 时，本阶段立即结束本轮，不得继续知识刷新、
Issue 获取、代码或测试分析；恢复规则以 `runtime-capability-checks.md` 为准。

### 阶段 B：获取与固定规则分类

按 `issue-intake.md` 合并 open 新建窗口、全状态 `updated_at` 增量和 follow-up
watchlist，建立首响、解决与再回复时钟，并按 SLA 排序。无匹配项时进入报告；`batch` 的
`no_attention` 不再推进，`single` 的目标必须进入阶段 C，并先排除已有解决证据和重复修改。

### 阶段 C：诊断、分派与分组

按 `issue-routing.md` 先检索知识库，再形成证据化根因假设和处置类型；等待与再次回复协议
按 `issue-followup.md` 执行。确认前只准备
评论、指派、状态迁移和交付操作，不执行外部写入。责任人缺失时先继续独立诊断，再于统一
预览前一次收齐 owner 或 `direct`。等待与 follow-up 状态严格按 reference 回查和落盘。

所有代码修改 Issue 都进入明确分组。按 `code-worktree.md` 规划冲突波次并创建独立
worktree；预计路径或独占资源冲突的组不得并行。

### 阶段 D：复现与根因确认

按 `code-root-cause.md` 在组 worktree 中核对目标环境、稳定复现并确认最终根因。
未稳定复现或未形成最终方案时不得改代码；预计修改路径变化时重新执行冲突检查。

### 阶段 E：实施与验证

按 `code-validation.md` 实施最小、可丢弃的未提交修改，使用目标仓库原生入口验证。
记录命令、结果和验证边界；失败时在本阶段迭代，不降低正确断言。

### 阶段 F：统一确认、提交与发布

按 `delivery-confirmation.md` 用最终 diff、测试结果和所有实际适用操作生成完整预览。用户
批准后才按 `delivery-publish.md` 执行精确 operation IDs；清单实质变化时使未执行批准
失效并重新确认。PR 只创建不合并；direct push 始终在 commit 后独立确认 exact 目标和 SHA。

### 阶段 G：报告与清理

按 `delivery-reporting.md` 只汇总本轮实际处理的 Issue，回查写操作并安全清理终态
worktree。调用 `scripts/generate_summary_report.py --strict` 生成历史报告和 `latest.md`，
回读检查内容后，在最终回复中提供可点击报告路径。

## 自动闭环维护路径

用户明确要求运行或部署咨询 Issue 自动闭环时，完整读取
[references/maintenance-stale-close.md](references/maintenance-stale-close.md)，使用
`scripts/auto_close_stale_issues.py`。该路径不进入代码修复阶段，默认 dry-run；交互运行先
展示逐 Issue 的完整评论和关闭操作并取得确认，随后才使用 `--apply`。初始“自动处理”请求
不能代替这次确认；关联 PR 或关联扫描不完整时禁止关闭。

## 验证

修改本 Skill、references 或脚本后至少运行：

```bash
python3 -m pytest -q infra/gitcode-issue-handler/tests infra/gitcode-toolkit/tests
python3 tests/lib/skill_validator.py validate-skill infra/gitcode-issue-handler/SKILL.md
```

若修改环境门禁，还必须覆盖纯 `policy_query` 不预检、handler 各 `--checks` 只探测所选
能力、失败仅阻断依赖操作，以及 toolkit 无参数旧调用兼容。缺少外部服务或 NPU 时记录验证边界，
不得声称完整上板验证通过。

## 完成条件

只有在所有实际处理 Issue 均有真实终态或明确等待对象和下一步、所有执行操作可追溯到有效
授权、代码修改具有复现/根因/验证/发布证据、受管 worktree 已安全处置，并且严格报告生成
成功后，运行才算完成。逐项口径、过程日志、指标和清理证据以
`delivery-reporting.md` 为准。
