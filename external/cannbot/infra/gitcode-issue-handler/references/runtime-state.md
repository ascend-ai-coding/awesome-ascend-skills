# 运行时：授权与状态契约

## 读取时机

仅真实执行请求在步骤 -1 与 `runtime-setup.md` 一起完整读取本文件。`policy_query` 不建立
运行状态。执行模式开始时建立唯一运行状态，各阶段 reference 只更新自己负责的字段，
不另建平行口径；条件字段不适用时可以省略，但不得用虚假值补齐。

## 初始化授权模式

在首次真实操作前记录授权模式，但不要把任一能力检查结果当成写操作授权：

- `single` 和 `batch` 始终先设置 `authorization_mode: interactive`。用户最初要求
  “处理 Issue”“批量自动执行/auto apply”或类似目标只授权分析，不得在初始化时设置
  `approved_batch`。
- 完成实际 Issue 的诊断，并在受管 worktree 中形成稳定复现、未提交 diff 和本地验证
  结果后，按 `delivery-confirmation.md` 展示当前仓库、Issue 清单和全部实际适用操作。
  只有用户基于该精确预览明确批准，`batch` 才切换为 `approved_batch`；`single` 保持
  `interactive` 并记录统一检查点证据。
- `approved_batch` 的状态必须同时记录 `authorization_source: explicit_user_approval`、
  `execution_confirmation_source: post_analysis_user_approval`、仓库、Issue 范围、operation
  IDs、交付模式、预览摘要和当前会话批准证据。缺任一项都回退到 `interactive`。
- `direct-push` 不得写入批次授权范围；它始终在 commit 形成后单独确认。

完整字段和检查点复用规则见本 Skill 的
[authorization-contract.md](authorization-contract.md)。

## 状态结构

```yaml
run:
  run_id:
  mode: single | batch
  started_at:
  completed_at:
  overall_status: running | waiting_for_input | completed | partial | blocked | no_issues
  authorization_mode: interactive | approved_batch
  authorization_source: default | explicit_user_approval
  authorization_scope: {}
  authorization_evidence: {}
  execution_confirmation_status: not_required | pending | approved | rejected | invalidated
  execution_preview_path:
  execution_preview_digest:
  execution_approved_at:
  execution_confirmation_source: post_analysis_user_approval
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
  operator_owner_request_status: collecting | ready | requested | resolved | not_needed
  sync_completed: false
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
  worktree_root:
  worktree_manifest:
  time_scope:
  issues_scanned_total: 0
  issues_total: 0
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
    operator_owner:
    operator_owner_source: config | user | none
    operator_handling_decision: delegate | direct | pending
    assignment_status: not_started | verified | failed | not_applicable
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

## 更新规则

每次分类、诊断、复现、修改、验证、授权、发布、回评或状态转换后立即追加必要结果和证据，
不得在步骤 9 凭记忆重建过程。字段的阶段性写入规则以对应 reference 为准。

能力状态只在对应真实操作紧前更新。未进入该路径时保持 `not_started` 或在运行终态记为
`not_required`，不得为了补齐状态而执行环境探测。API Token 只保存在当前会话，不写入状态。

缺 Token 时只保存元数据，不保存 Token 明文：把 `overall_status` 和
`capability_checks.api` 设为 `waiting_for_input`，追加或复用唯一
`input_id: gitcode_token`，并把下一条尚未执行的操作写入 `resume_from`。同一未解决输入的
`request_count` 固定为 `1`；它存在时不得再次询问。用户补充 Token 后把该项标为
`resolved`，API 检查通过后恢复 `overall_status: running`、`capability_checks.api: ready`，
从 `resume_from` 继续，不重新初始化、不重复已完成阶段。
