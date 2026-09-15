# Issue 流程：文字诊断、方案分派与合并分组

目录：

- [读取时机](#读取时机)
- [步骤-2c形成根因假设](#步骤-2c形成根因假设)
- [步骤-2d方案分派](#步骤-2d方案分派)
- [步骤-2e合并分组](#步骤-2e合并分组)
- [输出](#输出)

## 读取时机

对 `need_attention` Issue 执行步骤 2c 至 2e 前完整读取本文件。

## 步骤 2c：形成根因假设

先确认步骤 0a 已记录本轮 `knowledge_refresh_status`；未记录时返回
`runtime-knowledge.md` 完成刷新或可信降级。随后检索本 Skill 内嵌的受审知识和目标仓库
运行时历史候选：

```bash
python3 "$ISSUE_HANDLER_SKILL_ROOT/scripts/knowledge_query.py" \
  --repository-root "$ISSUE_HANDLER_REPOSITORY_ROOT" preflight \
  --task "<Issue 标题 + 现象/错误短语 + 算子/模块 + 平台/版本>"
```

先读取 `read_first` 中的受审规则卡和案例，再按需查看 `runtime_candidates`。运行时候选
固定为 `provisional/low`，只用于提出调查路径，不能证明当前根因；两层均未命中时记录
“知识库未命中”，继续根据当前 Issue 和代码分析。

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

所有假设必须由 Issue 原文、评论或图片支撑。只有 Issue 明确点名算子才设置
`operator_name`；公共工程、文档和仅由函数名推测的情况不算命中。

## 步骤 2d：方案分派

### 算子责任人转交

单 Issue 和批量模式使用同一规则。明确涉及具体算子的 Issue 默认交给算子责任人，不能
因为映射为空、未命中或当前是单 Issue 就由 Agent 自行修复。

按以下顺序执行：

1. **先准备有效首响**：检查现有评论；没有维护侧有效响应时，按 `gitcode-toolkit` 的
   Issue 评论工作流准备“已受理，正在确认 `<算子名>` 算子责任人，确认后将及时转交”
   这类简短评论。此时只把 Issue 目标和完整正文加入统一执行预览，禁止发送。分析后统一
   确认批准该操作时才发送并 GET 回查。纯 `/assign`、系统消息或仅 `@owner` 不算有效
   首响，预览也不算。
2. **解析已有决定**：先读取当前用户消息和本会话中对该算子的明确决定，再执行：

   ```bash
   python3 "$ISSUE_HANDLER_SKILL_ROOT/scripts/operator_owner_config.py" \
     --config "$PWD/.cannbot/gitcode-issue-handler/config/operator_owners.yaml" \
     lookup --operator "<算子名>"
   ```

   当前用户提供的登录名优先于旧配置；显式 `direct` 只授权 Agent 处理当前 Issue，不是
   责任人名称，也不能写入配置。
3. **缺失时先入队、继续诊断**：设置 `operator_handling_decision: pending`、
   `handling_status: pending_operator_owner`，把算子、相关 Issue IID、首响草稿和恢复
   检查点追加到 `run.deferred_operator_owner_requests`。同一算子只保留一项并合并 IID。
   不为这些 Issue 创建代码修复组或修改源码；跳过它们，继续其他 Issue 的诊断、分组、
   以及不依赖该输入的只读分析。此时不要发送首响、指派或修改目标源码。
4. **统一预览前一次询问**：所有不依赖责任人输入的 Issue 均完成诊断、拟方案和验证规划
   后，把 `operator_owner_request_status` 设为 `ready`。若队列非空，在生成统一执行预览前
   只发起一次方案输入交互，覆盖所有缺失算子。优先调用当前客户端的结构化用户交互能力；
   不可用时用普通问题并暂停。问题必须给出两个清楚选项，并说明该选择不授权执行：

   ```text
   以下算子在统一配置目录的 `operator_owners.yaml` 中没有责任人。请逐项选择：
   1. 配置并指派（推荐）：回复 `<算子名>: <GitCode 登录名>`；
   2. 由 Agent 直接处理某个 Issue：回复 `#<IID>: direct`。
   涉及 Issue：<算子名 -> Issue IID 列表>
   ```

   owner 是算子级配置，应用到该算子的全部待决 Issue；`direct` 是 Issue 级授权，同一算子
   有多个 Issue 时必须逐个列出选择 `direct` 的 IID，不能用 `<算子名>: direct` 批量授权。

   发起交互时把聚合项写入 `run.pending_user_inputs` 并设置
   `operator_owner_request_status: requested`。若用户没有回答，保持
   `pending_operator_owner`，不能默认选择 `direct`；其他 Issue 的已完成结果不得回滚或
   重做，也不能生成包含未知指派目标的批准范围。
5. **持久化用户提供的责任人**：去掉可选的 `@` 前缀，用以下命令写入目标仓库根目录的
   `.cannbot/gitcode-issue-handler/config/operator_owners.yaml`：

   ```bash
   python3 "$ISSUE_HANDLER_SKILL_ROOT/scripts/operator_owner_config.py" \
     --config "$PWD/.cannbot/gitcode-issue-handler/config/operator_owners.yaml" set \
     --operator "<算子名>" --owner "<GitCode 登录名>"
   ```

   命令失败时保留 `pending_operator_owner` 并报告配置错误，不手工拼接 YAML，不覆盖其他
   算子映射。写入成功后设置 `operator_owner_source: user`；配置命中则使用 `config`。
6. **把真正指派与等待状态纳入预览**：为每个待指派 Issue 列出问题摘要评论的完整正文、
   owner、独立的 `/assign @<owner>`、`<当前状态> -> 挂起` 和 assignee watch 操作。只有
   这些 operation IDs 已通过分析后统一执行确认，才
   按 `gitcode-toolkit` 的算子转交流程依次发送并 GET 回查 Issue，确认 assignee 的 GitCode
   login 与 `<owner>` 大小写无关地一致。仅在 assignee 回查成功后执行并回查`挂起`迁移，
   再写入 `waiting_on: assignee` 的 watch；全部成功后设置 `assignment_status: verified`、
   `handling_status: delegated` 和 `resolution_status: resolution_pending`。
7. **处理失败和 direct**：指派评论或 assignee 回查失败时设置
   `assignment_status: failed`、`handling_status: assignment_failed`，记录可重试动作并继续
   其他 Issue，禁止自动改为代码修复。用户选择 `direct` 时设置
   `operator_owner_source: none`、`operator_handling_decision: direct`、
   `assignment_status: not_applicable`，只让明确写出 IID 的 Issue 进入下方常规分派；不要把
   `direct` 写入配置，也不要扩展为同算子其他 Issue 的授权。

收到完整回复后，从每个待决 Issue 的步骤 2d 检查点恢复：owner 分支只确定配置写入、
   转交和回查计划；`direct` 分支才进入常规分派并形成必要的代码计划。不得重新执行已通过
   的能力检查、Issue 获取、已完成的算子识别或其他 Issue 的处理。全部聚合项都有确定方案后清空
`run.pending_user_inputs`，设置 `operator_owner_request_status: resolved`，把所有适用动作
纳入 `delivery-confirmation.md` 的统一预览；尚未批准前不得执行评论、指派、暂存、commit
或发布。明确选择 `direct` 的代码 Issue 可先在受管 worktree 中完成未提交实现和验证，
再进入统一预览。

已验证指派后记录 Issue、算子、owner、首响评论、assignee 与`挂起`回查证据，标记
`resolution_pending`，写 assignee watch 后跳过该 Issue 的代码处理并继续下一项。转交和
`挂起`都不等于 Issue 已解决。

### 常规分派

| 类型 | 动作 |
| --- | --- |
| `code_change` | 进入步骤 3 |
| `comment_explain` | 用可复核证据准备答疑并加入统一预览，批准后发布并回查 |
| `out_of_scope` | 准备范围和证据说明并加入统一预览，批准后发布并回查 |
| `need_more_info` | 准备最小上下文请求、`挂起`状态迁移并加入统一预览，批准后发布、回查并标记 `waiting_context` |

### 等待提出者、责任人与再次回复

需要进入 `awaiting_reporter`、`awaiting_assignee`，补录历史等待，或处理
`reporter_followup` / `assignee_followup` 时，完整读取 `issue-followup.md`。本文件只决定
处置类型和等待对象，不重复维护状态迁移、watch 和再次回复协议。

## 步骤 2e：合并分组

完成全部 Issue 分派后，只对 `code_change` 分组。以下情况可视为同类：

- 修改同一文件、模块或路径。
- 同类工程问题，例如文档、lint、依赖升级或配置。
- 同一根因的不同表现。

修复方向相斥、互相覆盖、存在冲突或依赖顺序不明确时不得合并。单 Issue 自动形成
单成员组。

每组记录：

```yaml
group_id:
members: []
theme:
proposed_branch:
planned_paths: []
exclusive_resources: []
```

`planned_paths` 必须使用仓库相对文件或目录，来自当前根因假设；无法界定时留空，调度器
会保守地把该组与所有其他组串行。`exclusive_resources` 记录不能共享的执行资源，例如
`npu:0`、特定开发服务器端口或非隔离构建缓存。`direct-push` 模式下还要为相同目标分支
加入 `delivery:<remote>/<branch>`，使这些组串行。

分支命名：

- 单 Issue：`fix/issue-<IID>-<slug>-<run-id>`
- 多 Issue：`fix/issues-<IID1>-<IID2>-<theme>-<run-id>` 或
  `fix/batch-<theme>-<run-id>`

分支名必须包含完整 `run_id`，避免安全清理保留本地分支后，后续运行因同名分支而阻塞。
不得通过删除旧分支或复用来源不明的分支来绕过重名。

`single` 和 `batch/interactive` 在本步骤先生成分组和操作草案，然后都可把代码组交给
`code-worktree.md` 创建受管 worktree，以完成本地复现、未提交实现和验证。确认前
禁止在原始工作区改代码，也禁止暂存、commit 或产生远端写入；最终 diff 和测试结果形成
后再按 `delivery-confirmation.md` 生成精确统一预览。

后续步骤 3–8 按组执行；步骤 3–4 仍逐 Issue 验证，步骤 4b 再校验是否应拆组及更新
`planned_paths`。分组完成后进入 `code-worktree.md` 生成执行波次并创建 worktree。

## 输出

每个 Issue 已分派为终止状态或 `code_change`，所有 `code_change` 都属于一个明确分组。
完成后进入 `code-root-cause.md`。
