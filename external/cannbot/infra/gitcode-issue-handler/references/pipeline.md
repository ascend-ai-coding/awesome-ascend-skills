# 批量入口：持久队列与逐项推进

每次用户新发起批量处理，在目标仓根执行：

```bash
python3 "$ISSUE_HANDLER_SKILL_ROOT/scripts/issue_pipeline.py" resume --new-run --repository-root .
```

续跑同一轮次用普通 `resume`，轮内刷新用 `resume --refresh`；两者沿用原报告目录。新轮次独立创建目录，同时保留有效队列和范围缓存。

报告目录统一为 `reports/YYYYMMDD-HHMMSS-P0800/`（北京时间），同秒冲突追加 `-02` 起的序号。`run.json` 保存本轮内部 ID、开始时间、模式、仓库及全部内部扫描 ID；扫描 UUID 不作为报告目录名。`summary.md` 为用户入口，当前稿在 `issues/issue-IID/`，修订前的稿件在其 `history/` 下；分类、工作任务、校验、队列和运行状态在 `_internal/`。新一轮复用的有效材料也会复制到本轮目录，旧 UUID 目录仅兼容读取。

入口只读 GitCode、写本地运行目录；返回 `next_action` 和当前需要的 `required_references`。配置校验、能力检查、旧记录迁移、范围缓存和分类由脚本管理。缺 Token 时保存队列并停止；配置错误按返回字段修复，重跑原命令重验。

## 下一步协议

| next_action | 动作 |
| --- | --- |
| configure_repository / fix_configuration | 按错误和 [configuration-setup.md](configuration-setup.md) 配置，复用已明确选择 |
| claim_task | `claim --worker NAME` |
| work_task | 读取 required_references、task_file，依 allowed_values 填写 result_file |
| correct_result_file / complete_response_materials | 按 errors 修正原 result_file，重试原命令，保留当前 claim |
| review_results | 逐项审核 submitted 的证据并 accept；不等全批 |
| classify_ready | 在线 `resume`，推进范围已就绪项 |
| refresh_evidence | `resume --refresh`；扫描未完整不能报告全批完成 |
| await_task_results | 有单项交回即接手；确认执行者仍工作才 renew |
| resume_waiting_work | 用 status 查询 waiting_issues / tasks.waiting，按实际等待对象继续；缺新证据不重新派工 |
| verify_operations | 按返回文档和 operation_files 回查精确操作目标，未知结果不重发 |
| complete | 队列无待推进动作，不等于所有 open Issue 已解决 |

默认输出计数、有限条目和完整文件路径。`pages` 给出 total、returned、next_offset；查更多用 `--section NAME --offset N --limit N`，`--details` 返回完整回执。submit/accept 只回传本项和聚合数，其余待办用 `status`；`status --iid IID` 查看该项结论、等待和材料路径；需要原始记录时加 --details。

`status_file` 保留完整队列视图，唯一状态为 `.cannbot/gitcode-issue-handler/data/pipeline-state.json`。首次及定期全量扫描，增量扫描仍定点回查未完成旧项；列表缺席不能推出关闭。恢复排错才读 [runtime-state.md](runtime-state.md)。同会话复用扫描，新会话自动检查变更；明确新一轮用 `resume --refresh`。

`--offline` 不访问 API；`resume --input FETCH_JSON --offline` 用于隔离回放，缺扫描完整性字段仍保留不完整状态。特殊时间/编号子集走 [issue-intake.md](issue-intake.md)，不替代完整仓库基线。

## 领取与核查

claim 返回只读 `claim_file`、`task_file` 和可编辑 `result_file`。后续命令使用 claim 文件加载任务身份与结果路径：

```bash
python3 "$ISSUE_HANDLER_SKILL_ROOT/scripts/issue_pipeline.py" claim --worker coordinator
# 填写返回的 result_file
python3 "$ISSUE_HANDLER_SKILL_ROOT/scripts/issue_pipeline.py" submit --claim-file CLAIM &&
python3 "$ISSUE_HANDLER_SKILL_ROOT/scripts/issue_pipeline.py" accept --claim-file CLAIM
```

scope 按 [responsibility-scope.md](responsibility-scope.md) 填 decision、summary、evidence、source_mode。正文或固定提交用 fixed，依赖当前分支内容用 moving；任务提供旧结论和失效原因，只补必要调查。范围判断不代表已确认根因。

简单任务由主会话处理；复杂独立调查需要委派时读 [batch-analysis.md](batch-analysis.md)。先领取再启动执行者，空闲补位、单项审核，不等全批。默认最多三个运行任务，每执行者一个；三个 submitted 积压先审核。租约 300 秒；`renew --claim-file CLAIM` 续租，外部等待用 `hold --claim-file CLAIM --reason REASON` 释放名额，证据到达再 requeue。无子 agent 能力时顺序执行。

## 响应材料

需要源码结论时先采集固定版本证据：

```bash
python3 "$ISSUE_HANDLER_SKILL_ROOT/scripts/issue_pipeline.py" inspect --claim-file CLAIM --path FILE
# 可重复 --path；关键实现按行展开：
python3 "$ISSUE_HANDLER_SKILL_ROOT/scripts/issue_pipeline.py" inspect --claim-file CLAIM --path FILE --lines 161:280
```

inspect 自动把完整 `evidence_file` 加入 result_file.inspection_files。输出摘录默认最多 2400 字符，截断显式标记；按行每次最多 160 行。完整文件事实、Markdown 相对链接目标和历史保存在证据文件，按 pages 查询更多列表，使用返回的 detail_command 复用已有证据，不重复采集。新行范围使用原 claim 和证据中的固定 --revision。目录只能定位文件，结论需核查具体源码；未读内容不当证据，文件存在不能推出远端页面状态，报告版本不同须说明。

按任务契约填写 analysis、reply、source_verdict、owner_review、response_review、next_action。需要责任人时读 [operator-handoff.md](operator-handoff.md) 和 [operator-owner-candidates.md](operator-owner-candidates.md)，使用 `inspect --identities` 取得 GitCode 账号依据；历史作者只是候选线索，不从姓名或邮箱猜账号。仅需分配的任务按分类候选填写 assignment，无需额外首响。

```bash
python3 "$ISSUE_HANDLER_SKILL_ROOT/scripts/issue_pipeline.py" check --claim-file CLAIM &&
python3 "$ISSUE_HANDLER_SKILL_ROOT/scripts/issue_pipeline.py" submit --claim-file CLAIM &&
python3 "$ISSUE_HANDLER_SKILL_ROOT/scripts/issue_pipeline.py" accept --claim-file CLAIM
```

单步执行时，成功回执的 next_command 给出后续命令参数；失败按 errors 修正原文件后重试。check/submit/accept 校验材料、身份、链接事实和证据完整性；协调者仍需审核语义和事实。材料不足按 errors 修正原文件；证据不确定用 needs_evidence/needs_escalation 保留已知结论和等待对象，不勉强 prepared。

## 审核、报告和恢复

resume/accept 自动生成适用的 analysis.md、reply.md、assign.md 及本轮 summary.md（delivery_report）；报告失败用 `report` 恢复，不手写大份 run_state。回读报告，最终材料清单依据 prepared_items 中实际存在的文件，确认真实结论、代码链接、适用草稿和下一步。内部队列摘要/status 不替代严格交付报告。

错稿用 `revise --iid IID --reason REASON` 产生新版本任务，旧稿保留历史，新 attempt 重新核查。accept 不授权发布；配置关闭自动响应时保留材料等待审核，不为 complete 伪造结果。执行实际发布时读 [authorization-contract.md](authorization-contract.md)、[automation.md](automation.md) 和 [issue-comment-workflow.md](issue-comment-workflow.md)；之后按 [delivery-reporting.md](delivery-reporting.md) 补实际交付记录。

错误沿返回 next_action 处理：busy 等当前命令结束再试；stale 重新读取当前任务；`requeue_corrected_result` 先 requeue 再 claim，按实际领取的新任务提交。网络、分页或磁盘失败保留检查点，修复依赖再 resume，不宣布 all_clear。恢复审计按 [runtime-state.md](runtime-state.md)。
