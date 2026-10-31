# 运行状态完整 schema

本文件承载 [runtime-state.md](runtime-state.md) 的完整字段、枚举和历史兼容说明。仅在实现状态读写、报告生成、恢复或测试需要精确字段时读取；普通执行按需读取相关段落。

```yaml
run:
  run_id:
  mode: single | batch
  started_at:
  completed_at:
  overall_status: running | waiting_for_input | completed | partial | blocked | no_issues
  authorization_mode: interactive | approved_batch
  authorization_source: default | explicit_user_approval | config_policy
  authorization_scope: {}
  authorization_evidence: {}
  execution_confirmation_status: not_required | pending | approved | rejected | invalidated
  execution_preview_path:
  execution_preview_digest:
  execution_approved_at:
  execution_confirmation_source: post_analysis_user_approval
  response_confirmation_status: not_required | pending | approved | rejected | invalidated
  response_preview_path:
  response_preview_digest:
  direct_push_confirmation_status: not_required | pending | approved | rejected
  capability_checks:
    api: not_started | ready | waiting_for_input | blocked | not_required
    git: not_started | ready | blocked | not_required
    tmp: not_started | ready | blocked | not_required
    author: not_started | ready | blocked | not_required
  pending_user_inputs:
    - input_id: gitcode_token
      capability: api
      reason: authenticated_gitcode_write
      status: requested | resolved
      request_count: 1
      requested_at:
      resume_from:
  deferred_operator_owner_requests: []
  operator_owner_request_status: collecting | ready | requested | awaiting_offline_confirmation | resolved | not_needed
  sync_completed: false
  # 查询已有快照不要求刷新；仅实际维护刷新后更新 refresh 字段，详见 runtime-knowledge.md。
  knowledge_refresh_status: not_started | fresh | refreshed | stale_fallback | unavailable
  knowledge_refresh_mode: none | skip | full | incremental
  knowledge_snapshot_usable: false
  knowledge_corpus_path:
  repository:
  repository_root:
  base_branch: master
  base_ref: origin/master
  base_commit:
  delivery_mode: pr | direct-push
  target_remote_branch: origin/master
  remote_branches: []
  worktree_root:
  worktree_manifest:
  time_scope:
  issues_scanned_total: 0
  issues_total: 0
  issues_listed_total: 0
  report_path:
  report_generated: false

issues:
  - iid:
    handled_in_run: true
    url:
    title:
    author:
    bucket:
    category:
    reason:
    problem_summary:
    responsibility: handle | list-only | ignore
    responsibility_evidence: []
    responsibility_summary:
    response_status: pending | prepared | verified | reused | exempt_self_authored_pr | not_applicable_existing | waived_by_user | failed
    response_operation_id:
    response_comment_id:
    response_comment_at:
    response_review: {}
    related_code: [] # 已核实关联代码：每项含 path、url、kind（directory/file）、revision；独立模块优先目录链接
    related_code_note: # 无法定位或无代码关联时说明原因
    response_artifacts: {} # analysis.md、适用的 reply.md/assign.md 的路径与 digest
    auto_action: {} # 沿用分类器的 response 阶段指派计划；不构成外部写入授权
    response_evidence: []
    issue_age_days:
    first_response_sla:
    conversation_state: awaiting_maintainer | maintainer_replied | awaiting_reporter | awaiting_assignee | reporter_followup | assignee_followup | reopened_followup
    waiting_on: maintainer | reporter | assignee
    latest_reporter_comment_id:
    latest_maintainer_comment_id:
    followup_pending_since:
    followup_sla: pending | at_risk | breached | unknown
    reopen_required: false
    activate_required: false
    resolution_status:
    resolution_metric_reason:
    signals: []
    required_environment: {}
    environment_check: {}
    root_cause_hypothesis:
    proposed_solution_type:
    evidence: []
    operator_name:
    owner_candidate_analysis: {}
    operator_owner:
    operator_owner_source: config | user | none
    operator_handling_decision: delegate | direct | pending
    assignment_status: not_started | verified | failed | not_applicable
    assignment_source: operator_owner | linked_pr | core_candidate | fallback_user
    assignment_provisional: false # 候选、PR 作者或兜底接收人的临时指派为 true
    assigned_candidate: # 临时指派的账号；fallback_user 不加入候选表
    operator_owner_request_status: awaiting_offline_confirmation | resolved
    assignment_evidence: []
    reproduction_status:
    final_root_cause:
    solution_plan:
    operation_authorizations: []
    group_id:
    handling_status:
    result_summary:
    process_log: []
    reproduction_attempts: []
    comments: []
    blockers: []
    remaining_risks: []
    next_action:

listed_issues:
  - iid:
    url:
    responsibility: list-only
    responsibility_summary:

groups:
  - group_id:
    members: []
    theme:
    branch:
    planned_paths: []
    exclusive_resources: []
    conflicts_with: []
    execution_wave:
    worktree_path:
    lifecycle_status: planned | active | blocked | published | no_changes | cancelled_clean | cleaned
    changed_files: []
    tests: []
    validation_status:
    commit_sha:
    pr_url:
    ci_status:

external_operations:
  - operation_id:
    kind: issue_comment | issue_assignment | issue_state_change | prepared_source_change | commit | branch_push | pr_create | first_ci | direct_push
    issue_iids: []
    target:
    summary:
    body:
    planned_files: []
    depends_on: []
    status: prepared | planned | approved | executed | skipped | failed
    authorization_evidence:

metrics: {}
internal_blockers: []
validation_boundaries: []
cleanup: {}
artifacts: {}
```

`listed_issues` 重生成时必须保留；每项只要求 iid、URL、`responsibility: list-only` 和非空一行摘要，不要求 `process_log`。`issues_total` 仅计实际处理，`issues_listed_total == len(listed_issues)`； ignore 不逐项展示。旧历史中的 `in_scope | out_of_scope | unknown` 兼容读取，不回写删除。

阶段对象的细节按需读取，不能因这里使用 `{}`/`[]` 就丢弃已记录的字段：

| 对象 | 字段与写入规则 |
|---|---|
| `owner_candidate_analysis` | [候选责任人](operator-owner-candidates.md)；候选不构成 owner 确认 |
| `required_environment`、`environment_check`、复现/方案 | [根因与环境](code-root-cause.md)、[验证](code-validation.md) |
| `process_log`、指标与报告 | [报告契约](delivery-reporting.md) |
| 授权、预览与 `external_operations` | [授权](authorization-contract.md)、[确认](delivery-confirmation.md) |
| 会话、watch 与恢复 | [跟进](issue-followup.md) |
| 代码组与 worktree 生命周期 | [worktree](code-worktree.md)、[发布](delivery-publish.md) |
