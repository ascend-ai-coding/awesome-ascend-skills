# Issue 流程：文字诊断、方案分派与合并分组

> 首响及自提/历史自提豁免以 `issue-intake.md` 为准；已有文字回复不重发，已有负责人不覆盖。PR 作者补分配属于 response 阶段，按 `automation.md` 完成。

目录：

- [读取时机](#读取时机)
- [步骤-2c形成根因假设](#步骤-2c形成根因假设)
- [步骤 2d：先回复，再方案分派](#步骤-2d先回复再方案分派)
- [步骤-2e合并分组](#步骤-2e合并分组)
- [输出](#输出)

## 读取时机

对 `need_attention` Issue 执行步骤 2c 至 2e 前完整读取本文件。`needs_pr_owner_handoff` 已有首响或免首响时，只核对 Issue、PR、当前负责人及作者选择，保存简短分析和分配稿，按 automation 完成分配；无需重新分析根因、查询核心贡献候选或进入代码修复。

## 步骤 2c：形成根因假设

按 [runtime-knowledge.md](runtime-knowledge.md) 查询已有受审知识和目标仓库历史快照；不要求本轮已执行 refresh，无缓存也继续当前 Issue 的定向调查和首响：

```bash
python3 "$ISSUE_HANDLER_SKILL_ROOT/scripts/knowledge_query.py" \
  --repository-root "$ISSUE_HANDLER_REPOSITORY_ROOT" preflight \
  --task "<Issue 标题 + 现象/错误短语 + 算子/模块 + 平台/版本>"
```

先读取 `read_first` 中的受审规则卡和案例，再按需查看 `runtime_candidates`。运行时候选固定为 `provisional/low`，只用于提出调查路径，不能证明当前根因；两层均未命中时记录“知识库未命中”，继续根据当前 Issue 和代码分析。

具体算子仍有维护动作、且无已确认 owner/有效 assignee、也没有可承接的有效关联 PR 时，在起草首响时同步按 [operator-owner-candidates.md](operator-owner-candidates.md) 核查核心提交及 GitCode 身份，候选写入 summary；调查不延迟已获授权的首响。

本阶段只形成文字层假设，不修改代码、不运行复现命令。回答：

- 用户报告的现象以及期望与实际行为是什么？
- 候选原因有哪些，按什么证据排序？
- 属于代码变更、评论答疑、仓库范围外，还是信息不足？
- Issue 是否明确点名具体算子？责任人是谁，来自配置、当前用户输入还是尚未确定？

输出：

```yaml
root_cause_hypothesis:
proposed_solution_type: code_change | comment_explain | out_of_scope | need_more_info
evidence: []
required_environment:
  cann:
  source_revision:
  soc:
  architecture:
  tools: []
operator_name:
operator_owner:
operator_owner_source: config | user | none
operator_handling_decision: delegate | direct | pending
assignment_status: not_started | verified | failed | not_applicable
```

所有假设必须由 Issue 原文、评论或图片支撑。只有原文或明确文件路径能确定具体算子才设置 `operator_name`；不能仅凭常见函数名猜测，公共工程问题也不因出现函数名转交。

## 步骤 2d：先回复，再方案分派

协调者对每个本轮需负责的 Issue，先完整读取 `issue-comment-workflow.md` 并完成回复门禁；新首响满足 `response_status: verified | reused | waived_by_user` 后进入分派；已核实的自提自修豁免项以 `exempt_self_authored_pr` 记录依据，无负责人时仍完成 PR 作者分配；已有实质回复复用或跟进。纯答疑将答案直接融合到这一条回复；非自提且无首响的已有 PR 项在回复中简述 PR 做法、覆盖与状态，再完成缺失的作者分配；已有回复或自提豁免时不为了分配凑一条评论。
owner 未知时照常回应已知事实与下一步，不等责任人输入；失败项保留检查点，继续其他 Issue。

若 `proposed_solution_type: comment_explain` 且本轮答案已完整回应、无维护侧剩余动作，直接记录答疑结论并结束本项，不为算子名称额外收集 owner、指派或挂起。不能确答时走信息不足或待处理分支，不假装答疑完成。

### 算子责任人转交

需要转交具体算子时读取 [operator-handoff.md](operator-handoff.md)：复用首响证据、查/写已确认 owner、候选线下确认、指派回查、挂起与 watch。未知 owner 不阻断首响，也不允许自动自行修复；纯答疑已经完整结束时不进入转交。

### 常规分派

| 类型 | 动作 |
| --- | --- |
| `code_change` | 进入步骤 3 |
| `comment_explain` | 答案已融合本轮回复；记录证据和结论，不重复发答疑 |
| `out_of_scope` | 按 responsibility 路由为 `list-only` 或 `ignore`，不另发范围说明评论 |
| `need_more_info` | 本轮回复已提出最小补充请求；随后获授权迁移`挂起`并回查，写 watch 和 `waiting_context` |

### 等待提出者、责任人与再次回复

需要进入 `awaiting_reporter`、`awaiting_assignee`，补录历史等待，或处理 `reporter_followup` / `assignee_followup` 时，完整读取 `issue-followup.md`。本文件只决定处置类型和等待对象，不重复维护状态迁移、watch 和再次回复协议。

## 步骤 2e：合并分组

完成全部 Issue 分派后，只对 `code_change` 分组。以下情况可视为同类：

- 修改同一文件、模块或路径。
- 同类工程问题，例如文档、lint、依赖升级或配置。
- 同一根因的不同表现。

修复方向相斥、互相覆盖、存在冲突或依赖顺序不明确时不得合并。单 Issue 自动形成单成员组。

每组记录：

```yaml
group_id:
members: []
theme:
proposed_branch:
planned_paths: []
exclusive_resources: []
```

`planned_paths` 必须使用仓库相对文件或目录，来自当前根因假设；无法界定时留空，调度器会保守地把该组与所有其他组串行。`exclusive_resources` 记录不能共享的执行资源，例如 `npu:0`、特定开发服务器端口或非隔离构建缓存。`direct-push` 模式下还要为相同目标分支加入 `delivery:<remote>/<branch>`，使这些组串行。

分支命名：

- 单 Issue：`fix/issue-<IID>-<slug>-<run-id>`
- 多 Issue：`fix/issues-<IID1>-<IID2>-<theme>-<run-id>` 或 `fix/batch-<theme>-<run-id>`

分支名必须包含完整 `run_id`，避免安全清理保留本地分支后，后续运行因同名分支而阻塞。
不得通过删除旧分支或复用来源不明的分支来绕过重名。

`single` 和 `batch/interactive` 在本步骤先生成分组和操作草案，然后都可把代码组交给 `code-worktree.md` 创建受管 worktree，以完成本地复现、未提交实现和验证。交付确认前禁止在原始工作区改代码，也禁止暂存、commit 或产生未授权远端写入；最终 diff 和测试结果形成后再按 `delivery-confirmation.md` 生成精确统一预览。

后续步骤 3–8 按组执行；步骤 3–4 仍逐 Issue 验证，步骤 4b 再校验是否应拆组及更新 `planned_paths`。分组完成后进入 `code-worktree.md` 生成执行波次并创建 worktree。

## 输出

每个 Issue 已分派为终止状态或 `code_change`，所有 `code_change` 都属于一个明确分组。
完成后进入 `code-root-cause.md`。
