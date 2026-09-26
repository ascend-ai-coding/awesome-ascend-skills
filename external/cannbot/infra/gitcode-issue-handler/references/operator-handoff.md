# 算子责任人转交

单 Issue 和批量模式使用同一规则。明确涉及具体算子的 Issue 默认交给算子责任人，不能因为映射为空、未命中或当前是单 Issue 就由 Agent 自行修复。

已有有效关联 PR 时优先按 [automation.md](automation.md) 在 response 阶段选择和分配 PR 作者，由 auto-response 控制；自提免首响，不进入下列未知 owner 确认流程。没有有效 PR 时，明确 owner 可在首响后按该文档直接转交。
owner 不明且启用 `auto-assign` 或用户单次明确要求自动选择候选时，走该文档的临时指派路径：首响回查后选择一人，不等确认、不写正式 owner 配置、不自动挂起。
其余情况按以下正常确认流程执行：

1. **核对已完成回复**：读取当前 Issue 的回复门禁证据；没有则先按评论工作流执行。
   把准确目标和完整正文加入回复执行预览，禁止发送未经授权的正文；已有批准不重复确认。
   纯 `/assign`、系统消息或仅 `@owner` 不算有效首响，预览也不算。
2. **解析已有决定**：先读取当前用户消息和本会话中对该算子的明确决定，再执行：

   ```bash
   python3 "$ISSUE_HANDLER_SKILL_ROOT/scripts/operator_owner_config.py" \
     --config "$PWD/.cannbot/gitcode-issue-handler/config/operator_owners.yaml" \
     lookup --operator "<算子名>"
   ```

   当前用户提供的登录名优先于旧配置；显式 `direct` 只授权 Agent 处理当前 Issue，不是责任人名称，也不能写入配置。
3. **缺失时先入队、继续诊断**：设置 `operator_handling_decision: pending`、 `handling_status: pending_operator_owner`，把算子、相关 Issue IID、首响回查证据和恢复检查点追加到 `run.deferred_operator_owner_requests`。同一算子只保留一项并合并 IID。
   不为这些 Issue 创建代码修复组或修改源码；跳过它们，继续其他 Issue 的诊断、分组、以及不依赖该输入的只读分析。首响已完成；此时不要执行指派或修改目标源码。
4. **交付候选供线下确认**：按 `operator-owner-candidates.md` 优先委派轻量子 agent，完成每个算子最多 5 人的核心贡献与身份核查，逐算子将账号和简述写入 summary；完整证据及排除理由留在状态/附件。设置 `operator_owner_request_status: awaiting_offline_confirmation`，保留队列并继续其他工作，不强制本轮即时收齐账号。用户确认具体 owner 或当前 Issue 的 `direct` 后恢复；候选排名不等于责任人确认，不写 owner 配置、不发送候选 @ 或 `/assign`。

5. **持久化用户提供的责任人**：去掉可选的 `@` 前缀，用以下命令写入目标仓库根目录的 `.cannbot/gitcode-issue-handler/config/operator_owners.yaml`：

   ```bash
   python3 "$ISSUE_HANDLER_SKILL_ROOT/scripts/operator_owner_config.py" \
     --config "$PWD/.cannbot/gitcode-issue-handler/config/operator_owners.yaml" set \
     --operator "<算子名>" --owner "<GitCode 登录名>"
   ```

   命令失败时保留 `pending_operator_owner` 并报告配置错误，不手工拼接 YAML，不覆盖其他算子映射。写入成功后设置 `operator_owner_source: user`；配置命中则使用 `config`。
6. **把真正指派与等待状态纳入预览**：为每个待指派 Issue 引用已回查的回复，只有新增信息才列补充评论的完整正文、 owner、独立的 `/assign @<owner>`、`<当前状态> -> 挂起` 和 assignee watch 操作。只有这些 operation IDs 已通过当前回复/交付预览确认，或具有当前会话覆盖准确操作的授权证据，才按 toolkit 评论 API 单次 POST `{"body":"/assign @<owner>"}`，再 GET 回查 Issue，确认 assignee 的 GitCode login 与 `<owner>` 大小写无关地一致。仅在 assignee 回查成功后执行并回查`挂起`迁移，再写入 `waiting_on: assignee` 的 watch；全部成功后设置 `assignment_status: verified`、 `handling_status: delegated` 和 `resolution_status: resolution_pending`。
   指派前先 GET，目标已是该 owner 则复用。`/assign` 是平台指令，不使用逐字正文回查的 `post_issue_comment.py`：平台可能将 mention 改写为 Markdown，成功依据是 assignee login。
   POST 结果未知时只回查 assignee 和评论，不重发；回查不足则保留未确认状态。
   若仓库未执行 `/assign` 指令，且当前授权覆盖该 Issue 的负责人变更，可使用 toolkit 原生更新接口 PATCH `{"assignee":"<owner>"}` 并 GET 核验；不添加评论、不改变目标账号。
   仅授权发送评论时不能扩大为原生指派。
7. **处理失败和 direct**：指派评论或 assignee 回查失败时设置 `assignment_status: failed`、`handling_status: assignment_failed`，记录可重试动作并继续其他 Issue，禁止自动改为代码修复。用户选择 `direct` 时设置 `operator_owner_source: none`、`operator_handling_decision: direct`、 `assignment_status: not_applicable`，只让明确写出 IID 的 Issue 进入下方常规分派；不要把 `direct` 写入配置，也不要扩展为同算子其他 Issue 的授权。

收到完整回复后，从每个待决 Issue 的步骤 2d 检查点恢复：owner 分支只确定配置写入、转交和回查计划；`direct` 分支才进入常规分派并形成必要的代码计划。不得重新执行已通过的能力检查、Issue 获取、已完成的算子识别或其他 Issue 的处理。全部 owner 聚合项都有确定方案后仅移除对应 owner 输入项，不清空其他待输入项；更新 `run.pending_user_inputs`，设置 `operator_owner_request_status: resolved`，把所有适用动作纳入 `delivery-confirmation.md` 的统一预览；尚未批准前不得执行评论、指派、暂存、commit 或发布。明确选择 `direct` 的代码 Issue 可先在受管 worktree 中完成未提交实现和验证，再进入统一预览。

已验证指派后记录 Issue、算子、owner、首响评论、assignee 与`挂起`回查证据，标记 `resolution_pending`，写 assignee watch 后跳过该 Issue 的代码处理并继续下一项。转交和 `挂起`都不等于 Issue 已解决。

## 阶段约束

算子问题默认转交已确认 owner，未知时不能自行修复。仍有维护动作且无已确认 owner/有效 assignee、也无有效关联 PR 时，起草首响并按涉及的每个算子分析核心贡献候选，每算子最多 5 人；summary 只写可指派账号及简述，供用户线下确认。
   候选不等于 owner，不公开候选列表 @，不延迟首响；用户对当前 Issue 明确 `direct` 才自行处理。
   纯答疑已完整解决且无剩余维护动作时不额外收集 owner。
