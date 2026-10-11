# Issue Handler 聚合授权契约

## 读取时机与边界

初始化状态时读取；生成统一执行预览、复用批准证据或执行自动闭环前再次核对。本文只定义 Issue Handler 业务授权；GitCode 写入通用边界仍以同级 `gitcode-toolkit/references/authorization-contract.md` 为准。

能力检查、运行平台工具审批和仅提供责任人 login 的方案选择都不是业务写入授权。用户明确要求对准确 Issue 执行具体指派等操作时，保存并复用该操作授权。

首响、安全门禁通过的 PR 关联、明确 owner/有效 PR 作者转交、候选临时指派按 [automation.md](automation.md) 的配置策略执行：当前处理请求限定目标，开启的开关限定操作；记录 `authorization_source: config_policy`、配置快照、Issue 和 operation IDs，保持 `interactive`。其余操作仍遵循下述批准流程。

## 模式与批准

| 模式 | 范围 | 行为 |
| --- | --- | --- |
| `interactive` | 单 Issue，或未获分析后批准的批处理 | 分析和预览；已有覆盖当前 operation 的明确授权即可执行并回查 |
| `approved_batch` | 用户批准精确 Issue 清单和完整操作预览的批处理 | 只执行批准清单内 operation，不逐项重复确认 |

`single`、`batch` 都从 `interactive` 开始；初始“处理/自动处理”或配置声明不能切换模式；两开关授权的首响/指派无需切换。只有基于实际 Issue、最终 diff、验证结果和完整预览的明确批准，`batch` 才切换为 `approved_batch`；`single` 保持 interactive 并复用该检查点证据。批准必须记录 `authorization_source: explicit_user_approval`、`execution_confirmation_source: post_analysis_user_approval`、仓库、Issue 范围、operation IDs、交付模式、预览摘要和会话证据，缺任一项回退 interactive。Issue 集合、文件、正文、状态目标、commit、分支、PR 内容或交付模式变化会使未执行部分失效。

模式未知、批准证据缺失、目标或操作超出 scope 时，停止尚未执行写入并回到统一预览；`approved_batch` 不适用于单 Issue。授权记录至少包括 `mode`、`source`、`scope.repository`、`scope.issue_iids`、`scope.operation_ids`、`scope.delivery_mode`、`approved_at`、检查点和预览摘要。

### 子流程授权上下文

传给子流程的结构如下；从同一运行状态的对应回复/交付检查点派生，不另建平行授权口径。
`mode/source/scope` 对应 `run.authorization_mode/authorization_source/authorization_scope`；批准时间和证据取实际覆盖该 operation 的检查点，回复批准不能冒充交付批准。

```yaml
authorization:
  mode: interactive | approved_batch
  source: default | explicit_user_approval | config_policy
  scope:
    repository: owner/repo
    issue_iids: []
    operation_ids: []
    delivery_mode: pr | direct-push | none
  approved_at:
  evidence:
    checkpoint:
    preview_digest:
```

统一预览可合并已就绪操作，但每项仍是独立 operation：评论（准确 Issue 与完整正文）、指派（Issue/login/操作）、状态迁移（当前与目标状态）及交付（文件、commit、分支、PR 与首次 CI）。评论批准不隐含指派、状态、commit、push 或 PR 批准。用户要求不评论或缩小范围时删除 operation 并同步缩小 scope。正文、目标或依赖实质变化须重新确认尚未执行部分。

回复检查点按 [delivery-confirmation.md](delivery-confirmation.md) 执行：文字诊断后先完成必要回复，不等待 owner 或代码验证；批次可合并当前就绪回复，不能为交付预览拖延首响。已有合法授权不重问。

## Direct-push 与自动闭环

`direct-push` 不属于统一执行确认或 `approved_batch`。commit 形成后单独展示 remote URL、目标分支、SHA、保护/共享分支提示和非快进检查，确认前保留本地 commit，不自动改走 PR。

`auto-close-stale` 独立于 `approved_batch`：交互运行逐 Issue 展示固定评论和关闭操作，确认后才 `--apply`；无 `--apply` 保持 dry-run。部署模式仅接受独立记录、覆盖精确仓库和闭环策略的授权，不继承普通处理授权。

## 写后回查

评论、指派、状态变更后 GET 回查；push 后 `git ls-remote` 回查；PR 创建后回查源/目标分支及 opened 状态。回查失败不得标记完成；结果未知的非幂等写入只做回查，未查清不得重试。每项外部写入写入状态的 `external_operations`，包括 operation ID、授权、执行和回查证据。

## 阶段约束

首响、安全 PR 关联与临时指派按 [automation.md](automation.md) 的两开关执行，默认均关闭；开启配置与当前处理请求共同授权对应操作，明确会话指令可单次覆盖。复用准确授权，不重复询问；Token、能力检查不是授权。
   `single`/`batch` 从 `interactive` 开始；回复与交付分开确认，只有精确批次交付批准才进入 `approved_batch`。direct push 在 commit 后按确切目标和 SHA 单独确认。

## 流水线操作登记与回查

在首次可能写入前登记稳定操作标识：

```bash
python3 "$ISSUE_HANDLER_SKILL_ROOT/scripts/issue_pipeline.py" record-operation \
  --operation-id comment-296-first-response --iid 296
```

登记会保存 unknown 恢复点，不产生写入授权。回查后保存审核凭据 JSON（`operation_id`、`verified: true`、非空 `evidence`，其中引用真实执行器结果和 GET 证据），用 `verify-operation --operation-id ... --result-file ...` 登记。它仅登记核验结论，不执行 GET；未知结果恢复必须先沿原执行器回查，不能手写 verified 跳过核验。损坏恢复的 `audit-recovery --result-file ...` 同样需要实际审计保留记录的证据，不以状态文件可读代替操作核查。

发布或外部状态更新后 `resume --refresh`，从真实远端重新分类；未决操作独立保留，即使 Issue 已变成 no_attention 也不能静默丢弃。代码、提交、PR、CI 等后续阶段继续现有受管 worktree 与交付门禁。
