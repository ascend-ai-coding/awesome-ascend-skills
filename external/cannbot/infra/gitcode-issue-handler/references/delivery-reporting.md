# 交付：报告与完成条件

## 读取时机与产物

普通 batch 的准备材料和严格报告由 pipeline 的 resume/accept 自动生成，使用返回的 delivery_report；失败时执行 report 恢复。无需读取本篇或重新手写 run_state。

以下用于 single 及后续实际发布、代码交付：所有 Issue/分组到达终止或等待状态后、步骤 9 前读取，生成内容相同的：

- 历史：`.cannbot/gitcode-issue-handler/reports/<run_id>/summary.md`（不可覆盖其他 run）
- 最新：`.cannbot/gitcode-issue-handler/reports/latest.md`

即使无匹配、启动/同步失败、只有 `no_attention`、全部等待或发布失败也要生成，并说明结束原因。缺 Token 暂停时只用已有状态记录等待原因，不为报告继续 API、诊断或加载测试。顺序固定：

1. 按 `responsibility` 汇总：`handle` 进入逐项 `issues`/group；`list-only` 中按正常规则仍为 `need_attention` 的项规范化到顶层 `listed_issues`，其余只留聚合计数；`ignore` 不进入逐项报告；纯获取/分类只留聚合计数。
2. 回查回复及已批准执行的外部操作，记录授权、写入和 GET 结果；未批准只保留预览。
3. 按 `code-worktree.md` 安全清理并回写结果。
4. 未知 owner 按 `operator-owner-candidates.md` 逐算子给候选账号与贡献（每算子最多 5 人，整单可超过）；无候选写“待确认”，可留 `awaiting_offline_confirmation`，候选不等于已确认负责人/解决。临时指派成功时另写“已临时指派 @账号，请确认真正负责人”，保留逐算子候选，不冒充正式 owner。兜底指派按 `assignment_source: fallback_user` 标明“无候选、兜底接收人”，不将其加入候选表。
5. 按 [逐项响应材料](delivery-reporting.md#逐项响应材料) 核对本轮每条 `need_attention` 的分析、适用草稿和 `response_artifacts`；关闭自动响应不免除落盘，受阻缺稿须有具体原因，未完成子任务须有恢复点。设置 `completed_at`/`overall_status`，写 `_internal/run_state.json`。
6. 在目标仓根执行：
   ```bash
   python "$ISSUE_HANDLER_SKILL_ROOT/scripts/generate_summary_report.py" \
     --state ".cannbot/gitcode-issue-handler/reports/<run_id>/_internal/run_state.json" --strict
   ```
7. 回读 `summary.md`：逐项只含 `handle`，数量等于 `issues_total`；`list-only` 只在“仅列举”中一行展示编号链接和 `responsibility_summary`，不展示候选、过程或外部操作；`ignore`、仅观察到的 `self_assigned` 和既有 `/assign` 不出现；本轮 response 阶段实际补分配的 `needs_pr_owner_handoff` 必须保留。失败先修状态重试，不用对话摘要替代报告，也不因单 Issue 失败跳过整批。
   同时按 [关联代码链接](delivery-reporting.md#关联代码链接) 核对逐项分析和汇总是否展示关联目录/文件链接；无链接时须说明原因。核对每项结果和下一步是否具体、候选是否实际渲染；`--strict` 通过不代表内容合格，不能用全批相同的“已回复、维护侧承接”替代调查结论。
8. 写 `report_generated: true`、`report_path`；最终答复给简短结论和历史报告路径。

`run_id` 由脚本统一创建为 `YYYYMMDD-HHMMSS-P0800`，同秒冲突追加序号；`run.json` 是用户可读元信息，内部 UUID 与目录名独立。当前分析、回复、分配稿在 `issues/issue-IID/`，历史修订归入其 `history/`，分类与执行器结果等记录放 `_internal/`。旧 ID/路径保持读取兼容，新增轮次采用上述结构。

## 逐项契约与字段

`issues` 只保存本轮实际处理且新路由为 `responsibility: handle` 的 Issue。实际处理包括：进入 `need_attention` 后诊断/答疑/索要上下文/转交/等待落盘；处理新增评论或状态迁移回查；复现、根因、修改、测试、提交、推送、PR、CI；或发送非 `/assign` 有效评论并成功回查；也包括按本轮 response 计划准备、执行和回查 PR 作者分配，哪怕没有文字首响。仅获取/分类、发现已有负责人/PR/评论/跟进、仅观察到的 `self_assigned`、既有纯 `/assign` 都不算。本轮仅分配项保留处理前 `needs_pr_owner_handoff` 分类，记录 `handled_in_run: true`、分配结果、简短分析与稿件路径；自提豁免不计首响成功，分配不计问题解决。`list-only` 即使 `handled_in_run: true` 也移至 `listed_issues`；`ignore` 完全不入逐项。按 iid（缺失则 URL）去重；旧 `in_scope | out_of_scope | unknown` 字段兼容读取，不能删除历史。

`run.issues_scanned_total` 是扫描规模；`run.issues_total == len(issues)` 只计实际处理；`run.issues_listed_total == len(listed_issues)` 单列仅列举。其他无需处理最多留 category 聚合计数，不留标题、作者、assign、PR 逐项明细。`--strict` 对 `handle` 用完整契约，对 `list-only` 只校验 `iid`、`url`、非空 `responsibility_summary` 和 `responsibility: list-only`；缺摘要必须报错。

每个 `handle` 至少保留：iid、标题、URL、作者；`handling_status`、`resolution_status`、`result_summary`、`next_action`；实际动作和证据。实际等待/再回复才记录对话状态、双方最近 comment ID/时间和 follow-up SLA；有复现/变更/测试/发布/卡点才记录对应字段。`process_log` 格式：

```json
{"time":"2026-08-11T12:00:00+08:00","stage":"triage | diagnose | reproduce | implement | validate | authorize | deliver | comment","action":"执行了什么","result":"结果与状态变化","evidence":["命令、相对路径、commit/PR/Issue URL 或关键观测"]}
```

时间取不到写 `unknown`，不得删记录。创建未合入 PR、等待上下文、环境不匹配、转交和 CI 失败保持 `resolution_pending`/`unresolved`；不得为好看标 `resolved`。

## 对外摘要、指标和清理

生成器报告只含：运行概览（仓库、时间、总体状态、扫描/处理/列举数）；实际处理 Issue 的结果、状态、下一步和链接（根因、评论、变更、测试、PR 仅真实存在时追加）；仅列举一行摘要；有代码组才有变更与交付；仅被阻塞/部分完成或与实际 Issue 相关才有卡点/边界。禁止空章节、例行轨迹、空环境/测试/清理、`unknown` 表格和逐项 `no_attention`。Markdown 可含内部 blocker，禁止 Token、Authorization、密码等 secret；命令和产物优先相对路径，脚本仍会脱敏。

- 有效响应是维护者/责任人/授权助手的受理、判断、最小信息请求、明确转交或解决结论；系统消息和纯 `/assign` 不计。
- 解决时长是 `created_at` 到可核验 `resolved_at` 的自然日；解决率仅为本仓 `handle` 已解决/`handle` 总数。`list-only` 不计，`ignore` 不入集合；有证据的重复、非本仓、无效项可排除并注明理由。
- 未合入 PR、`delegated`、`waiting_*`、`intermittent_waiting`、`ci_blocked`、`comment_failed` 都不算解决；缺失数据用 `unknown` 或 `resolution_pending`，禁止推测。
- 内部卡点保留 `waiting_environment`（目标/当前 CANN、源码 revision、SoC、架构对比）、`ci_blocked`（PR、阶段、分类、轮次、建议）、权限/基础设施/工具缺失和验证边界，均不写外部评论。
- 报告列已清理 group/worktree；因 active、blocked、不干净或 manifest 失败而保留的项及下一步；不得 force 或直接删目录。清理不删分支、远程分支、commit、PR、证据或 manifest。附件结束后仅可删除本流程生成的 `.cannbot/gitcode-issue-handler/tmp/downloaded-attachments`。

完成条件：每个 `need_attention` 有真实状态、责任人或下一步；发布可核验；历史 summary、run_state、latest 均可读；`--strict` 成功；`issues`/`listed_issues` 数量与计数一致、覆盖所有 `handle`，且无 `ignore`、仅观察到的 `self_assigned`、既有纯 `/assign` 或观察到的 `no_attention` 明细。若实际处理为 0，终端和最终回复只写“本次未实际处理任何 Issue”及报告链接，不把扫描范围冒充整体指标。

## 逐项响应材料

每个 `need_attention` Issue 在 `.cannbot/gitcode-issue-handler/reports/<run_id>/issues/issue-<iid>/` 单独保存 `analysis.md` 和适用的 `reply.md`、`assign.md`。进入响应准备时先保存已知分析，随调查更新；分析简写实际判定依据、相关 PR 与选择理由、计划动作和阻塞。`reply.md` 是完整可发布草稿；只需分配时分析两三行即可，`assign.md` 只写 `/assign @login`，不生成无必要的文字首响。回复和分配都需要时保存两份稿件。

### 关联代码链接

单个和批量 Issue 的 `analysis.md` 都须列出与问题关联、实际核查过的代码链接，并同步到该 Issue 的 `related_code`，供汇总报告展示：

- 问题位于相对独立的算子、样例 Story 或其他模块目录时，优先附该目录的仓库网页链接，标签使用仓库相对路径；跨多个独立模块时分别列出，不用仓库根目录代替。
- 无合适独立目录时附具体文件链接；目录链接不足以支撑关键结论时可另附关键文件及已核实的行号链接。
- 链接对应实际核查的仓库与版本，优先固定 commit；使用分支或 tag 时标明版本。采用平台返回或实际核实的网页 URL，不把本地绝对路径、猜测路径或未经确认的行号当远端链接。目录与文件必须在该版本存在；本地未发布代码只能说明本地路径和未发布状态。
- 尚未定位关联代码、缺少版本或纯流程问题无代码关联时，在分析和 `related_code_note` 写明具体原因，不编造链接，也不为补链接越过已有能力门禁。

`related_code` 为对象列表，每项含 `path`、`url`、`kind`（`directory` 或 `file`）和 `revision`。调查者交回这些链接与依据，协调者在报告前核对；`list-only` 和 `ignore` 的原有精简规则不变，不为它们额外展开代码调查。

自动响应开启时先落盘再执行，关闭时同样保存材料并列出待用户指令的动作。受阻项也须保存分析和能确定的草稿；尚不能形成稿件时记录具体缺口，不编造回复或分配对象。调查笔记不替代最终材料，由协调者审核并归入上述目录，在本项 `response_artifacts` 记录可读文件路径。

将评论和分配的 `--result-file` 放在本轮 `_internal/issues/issue-<iid>/`。日期时间目录下，执行器自动将用户 Markdown 写入 `issues/issue-<iid>/`，JSON 索引留在 `_internal/`。旧目录兼容以下规则：两个执行器预览时就会保存分析与草稿，`response-artifacts.json` 汇总路径及正文摘要；若结果文件放在运行根，则自动写到其旁的 `issues/issue-<iid>/`。Agent 在 `analysis.md` 补充实际分析，将分配结果的 artifacts 或 `response-artifacts.json` 的 files 映射到本项 `response_artifacts`，其他执行状态继续写唯一 run_state。分配实际使用原生 PATCH，`assign.md` 是用户可读的分配稿，不重复发布为评论。

response 阶段将首响和 PR 作者分配作为独立 operation，一次准备并按依赖完成；首响已完成而分配失败时只恢复分配，不等下轮重新分类才补动作。写前刷新必要评论、PR 和负责人证据，变化后重算剩余动作；写入结果未知先回查。只需分配的项目在分配回查成功后才完成本阶段，不要求文字回复文件。


指标目标：解决率 >90%、平均解决时长 <7 个自然日、1 个工作日内有效响应；只据真实证据计算。
