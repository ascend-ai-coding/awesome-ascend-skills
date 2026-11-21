# 交付：报告与完成条件

## 读取时机与产物

普通 batch 的准备材料和严格报告由 pipeline 的 resume/accept 自动生成，使用返回的 delivery_report；失败时执行 report 恢复。无需重走下面的手工报告步骤或重新手写 run_state；准备回复、发布和受阻收尾时仍按本篇“逐项响应材料”核对落盘契约。

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
7. 回读 `summary.md`：顶部结果汇总表只含 `handle`，每条一行，数量等于 `issues_total`；`list-only` 只在“仅列举”中一行展示编号链接和 `responsibility_summary`，不展示候选、过程或外部操作；`ignore`、仅观察到的 `self_assigned` 和既有 `/assign` 不出现；本轮 response 阶段实际补分配的 `needs_pr_owner_handoff` 必须保留。失败先修状态重试，不用对话摘要替代报告，也不因单 Issue 失败跳过整批。
   同时按 [关联代码链接](delivery-reporting.md#关联代码链接) 核对逐项分析和下方处理详情中的关联目录/文件链接；无链接时须说明原因。核对每项最终结果和下一步是否具体、候选是否在详情中渲染；`--strict` 通过不代表内容合格，不能用全批相同的“已回复、维护侧承接”替代调查结论。
8. 写 `report_generated: true`、`report_path`；最终答复按下述“对话结果汇总”直接列出处理结果，并附历史报告链接。

`run_id` 由脚本统一创建为 `YYYYMMDD-HHMMSS-P0800`，同秒冲突追加序号；`run.json` 是用户可读元信息，内部 UUID 与目录名独立。当前分析、回复、分配稿在 `issues/issue-IID/`，历史修订归入其 `history/`，分类与执行器结果等记录放 `_internal/`。旧 ID/路径保持读取兼容，新增轮次采用上述结构。

## 对话结果汇总

处理结束或停在等待状态时，先简述本轮实际处理数量与总体结果，再在当前对话 session 中列出简短结果列表，让用户无需打开报告即可了解结果。每条实际处理的 Issue 一行，格式为：

- `Issue 编号（附链接）：最终处理结果；状态：平台状态；责任人：@login 或待确认；首响 SLA：达标/临近超时/已超时/未知/不适用。` 未完成项在同一行补充下一步或等待对象。

列表依据回读的报告与唯一运行状态，平台状态和 SLA 沿用汇总表的证据口径；已创建 PR、已转交、已回复或等待中不得概括为已解决。不要在对话中重复核心动作、命令、测试过程或大段卡点；末尾附历史 `summary.md` 的可点击链接。仅列举项与实际处理分开说明，不将忽略或观察项加入处理结果列表。无实际处理时沿用下文的零处理回复规则。

## 逐项契约与字段

`issues` 只保存本轮实际处理且新路由为 `responsibility: handle` 的 Issue。实际处理包括：进入 `need_attention` 后诊断/答疑/索要上下文/转交/等待落盘；处理新增评论或状态迁移回查；复现、根因、修改、测试、提交、推送、PR、CI；或发送非 `/assign` 有效评论并成功回查；也包括按本轮 response 计划准备、执行和回查 PR 作者分配，哪怕没有文字首响。仅获取/分类、发现已有负责人/PR/评论/跟进、仅观察到的 `self_assigned`、既有纯 `/assign` 都不算。本轮仅分配项保留处理前 `needs_pr_owner_handoff` 分类，记录 `handled_in_run: true`、分配结果、简短分析与稿件路径；自提豁免不计首响成功，非自提缺有效回复时保持首响待处理；分配不计问题解决。`list-only` 即使 `handled_in_run: true` 也移至 `listed_issues`；`ignore` 完全不入逐项。按 iid（缺失则 URL）去重；旧 `in_scope | out_of_scope | unknown` 字段兼容读取，不能删除历史。

`run.issues_scanned_total` 是扫描规模；`run.issues_total == len(issues)` 只计实际处理；`run.issues_listed_total == len(listed_issues)` 单列仅列举。其他无需处理最多留 category 聚合计数，不留标题、作者、assign、PR 逐项明细。`--strict` 对 `handle` 用完整契约，对 `list-only` 只校验 `iid`、`url`、非空 `responsibility_summary` 和 `responsibility: list-only`；缺摘要必须报错。

每个 `handle` 至少保留：iid、标题、URL、作者；`handling_status`、`resolution_status`、`result_summary`、`next_action`；实际动作和证据。实际等待/再回复才记录对话状态、双方最近 comment ID/时间和 follow-up SLA；有复现/变更/测试/发布/卡点才记录对应字段。`process_log` 格式：

```json
{"time":"2026-08-11T12:00:00+08:00","stage":"triage | diagnose | reproduce | implement | validate | authorize | deliver | comment","action":"执行了什么","result":"结果与状态变化","evidence":["命令、相对路径、commit/PR/Issue URL 或关键观测"]}
```

时间取不到写 `unknown`，不得删记录。创建未合入 PR、等待上下文、环境不匹配、转交和 CI 失败保持 `resolution_pending`/`unresolved`；不得为好看标 `resolved`。

## 对外摘要、指标和清理

报告先给简短概览，再给“处理结果汇总”表，每条实际处理的 Issue 一行：

| Issue | 最终处理结果 | Issue 状态 | 责任人 | 首响 SLA | 下一步 |
| --- | --- | --- | --- | --- | --- |
| 编号链接与标题 | 一句话说明最终结论及未完成事项 | 平台当前状态 | GET 确认的 assignee 或待指派/回查 | 达标 / 临近超时 / 已超时 / 未知 | 具体待办或等待对象 |

`result_summary` 只写最终结论，不罗列中间步骤、命令或验证过程。Issue 状态优先使用已获取/回查的 `issue_state`，缺失时用核心 `state`，均缺失则写“未知”；不得用本地 `handling_status` 或 `resolution_status` 冒充平台状态。首响 SLA 使用有证据的 `first_response_sla`：`met`、`at_risk`、`breached` 分别显示达标、临近超时、已超时；缺失或 `unknown` 显示未知。`exempt_self_authored_pr` 显示不适用，不计首响成功；不能凭已回复或已指派推断 SLA 达标。

“仅列举”仍只展示一行摘要。详细处理过程放在汇总表和仅列举之后，使用默认折叠的 `<details>`，标题为“处理详情（按需展开）”；逐项分析、关联代码、候选责任人、响应材料、核心动作、变更与交付、卡点与验证边界均放在其中，按实际存在的内容展示。影响结果的未完成事项须同时简写在汇总表的最终结果或下一步中，不能只留在折叠详情里。禁止空章节、空环境/测试/清理和逐项 `no_attention`。Markdown 禁止 Token、Authorization、密码等 secret；命令和产物优先相对路径，脚本仍会脱敏。

- 有效响应是维护者/责任人/授权助手的受理、判断、最小信息请求、明确转交或解决结论；系统消息和纯 `/assign` 不计。
- 解决时长是 `created_at` 到可核验 `resolved_at` 的自然日；解决率仅为本仓 `handle` 已解决/`handle` 总数。`list-only` 不计，`ignore` 不入集合；有证据的重复、非本仓、无效项可排除并注明理由。
- 未合入 PR、`delegated`、`waiting_*`、`intermittent_waiting`、`ci_blocked`、`comment_failed` 都不算解决；缺失数据用 `unknown` 或 `resolution_pending`，禁止推测。
- 内部卡点保留 `waiting_environment`（目标/当前 CANN、源码 revision、SoC、架构对比）、`ci_blocked`（PR、阶段、分类、轮次、建议）、权限/基础设施/工具缺失和验证边界，均不写外部评论。
- 报告列已清理 group/worktree；因 active、blocked、不干净或 manifest 失败而保留的项及下一步；不得 force 或直接删目录。清理不删分支、远程分支、commit、PR、证据或 manifest。附件结束后仅可删除本流程生成的 `.cannbot/gitcode-issue-handler/tmp/downloaded-attachments`。

## 首响与指派的逐项核对

收尾前以本轮 `handle/need_attention` 清单逐项比对，不用“已回复”代替“已指派”：每项记录 `response_status`、`assignment_status` 及各自结果。回复成功或复用须有 `response_comment_id`（或已核验评论记录）；指派成功须有 GET 回查的 `assignee`，并与预定 owner 一致，指派后确认标签仍按 automation 独立核验。

非自提实际处理项均须有效首响和平台负责人；已有结果可复用，纯答疑和 direct 不另行豁免。自提项免首响，记录同账号或有效同作者 PR 证据，仍需核验平台负责人。未指派或指派失败时填写 `assignment_reason` 和具体 `next_action`；`not_applicable`、未经核验的豁免、已准备草稿或仅 @ 都不能满足完成标准。自动响应关闭则保存首响/分配稿并等待授权，不能为通过校验擅自发布或捏造已完成。缺少 owner 时记录待确认与恢复点，技术调查不延迟已获授权的首响或已明确 owner 的转交。

最终收尾用 `generate_summary_report.py --strict --classification <本轮完整分类.json>` 同时核对本轮 `handle/need_attention` 清单是否全部进入报告；子集分类须先合并，不能用只含已完成项的子集代替。普通 pipeline 中间报告继续自动生成，最终人工发布/等待收尾时补这次全量核对。

已有有效首响且只缺负责人时，pipeline 只准备责任材料：`assignment` 写准确 `login`、`reason`；基于责任规则时另写 `source: config | user | fallback_user` 和非空 `evidence`。该来源是经审核的当前依据，不是解析自然语言配置的猜测；未知接单人使用 `needs_evidence` 保留等待。无需重复 `reply.md`、重查 kernel 或贡献历史。材料不是发布授权，执行后仍须 GET 核验。

`generate_summary_report.py --strict` 校验上述双项记录；待发送/待指派或回查失败会在汇总表直接标出，不能仅留在折叠详情，不能把未完成项标作已解决或转交完成。

完成条件：每条实际处理项有可核验有效首响或自提豁免依据，以及 GET 确认的 assignee；整批 `overall_status: completed` 须通过 `--strict --classification` 全量覆盖校验，任何缺项只能设置等待/部分完成；发布可核验；历史 summary、run_state、latest 均可读；`--strict` 成功；`issues`/`listed_issues` 数量与计数一致、覆盖所有 `handle`，且无 `ignore`、仅观察到的 `self_assigned`、既有纯 `/assign` 或观察到的 `no_attention` 明细。若实际处理为 0，终端和最终回复只写“本次未实际处理任何 Issue”及报告链接，不把扫描范围冒充整体指标。

## 逐项响应材料

每个 `need_attention` Issue 在 `.cannbot/gitcode-issue-handler/reports/<run_id>/issues/issue-<iid>/` 单独保存 `analysis.md` 和适用的 `reply.md`、`assign.md`。进入响应准备时先保存已知分析，随调查更新；分析简写实际判定依据、相关 PR 与选择理由、计划动作和阻塞。`reply.md` 是完整可发布草稿；只需分配时分析两三行即可，`assign.md` 只写 `/assign @login`，不生成无必要的文字首响。回复和分配都需要时保存两份稿件。

### 关联代码链接

单个和批量 Issue 的 `analysis.md` 都须列出与问题关联、实际核查过的代码链接，并同步到该 Issue 的 `related_code`，供汇总报告下方的处理详情展示：

- 问题位于相对独立的算子、样例 Story 或其他模块目录时，优先附该目录的仓库网页链接，标签使用仓库相对路径；跨多个独立模块时分别列出，不用仓库根目录代替。
- 无合适独立目录时附具体文件链接；目录链接不足以支撑关键结论时可另附关键文件及已核实的行号链接。
- 链接对应实际核查的仓库与版本，优先固定 commit；使用分支或 tag 时标明版本。采用平台返回或实际核实的网页 URL，不把本地绝对路径、猜测路径或未经确认的行号当远端链接。目录与文件必须在该版本存在；本地未发布代码只能说明本地路径和未发布状态。
- 尚未定位关联代码、缺少版本或纯流程问题无代码关联时，在分析和 `related_code_note` 写明具体原因，不编造链接，也不为补链接越过已有能力门禁。

`related_code` 为对象列表，每项含 `path`、`url`、`kind`（`directory` 或 `file`）和 `revision`。调查者交回这些链接与依据，协调者在报告前核对；`list-only` 和 `ignore` 的原有精简规则不变，不为它们额外展开代码调查。

自动响应开启时，发布文字回复前先保存本项 `analysis.md` 与已审核的完整 `reply.md`；仅分配项保存分析与 `assign.md`，不补无必要的文字首响。关闭自动响应时同样保存材料并列出待用户指令的动作，不以执行器 JSON 结果路径或调查笔记替代完整稿件。

### 受阻收尾与状态归属

实际执行、解释或模拟受阻收尾时，都逐项说明材料保存与写入者，不能仅汇总“受阻项保持待处理”：

1. **独立就绪项**：先保存本项分析和已审核稿件，按已有授权发布并 GET 回查，不等其他受阻项。
2. **证据不足项**：将已知事实、具体缺口和恢复点保存到本项 `analysis.md`；能确定的完整回复/分配稿照常保存，无法成稿时记录缺稿原因，不编造作者、回复或分配对象。
3. **调查子任务失败或未返回项**：同样保存本项 `analysis.md`，包括已有分析、调用失败原因、缺失证据和重派/主会话接续的恢复点；尚无调查结果时明确该缺口，不以失败代替“未找到”，也不省略材料落盘。
4. **写入归属与核对**：共享运行状态只由协调者更新，调查者只写各自结果目录，不改共享输入、状态或配置。协调者审核后将材料归入上述目录，在本项 `response_artifacts` 记录最终可读文件路径；报告前按完整 `need_attention` 清单核对每项分析、适用草稿或缺稿原因及恢复点。未完成调查保留待处理，不把全批记为完成。

将评论和分配的 `--result-file` 放在本轮 `_internal/issues/issue-<iid>/`。日期时间目录下，执行器自动将用户 Markdown 写入 `issues/issue-<iid>/`，JSON 索引留在 `_internal/`。旧目录兼容以下规则：两个执行器预览时就会保存分析与草稿，`response-artifacts.json` 汇总路径及正文摘要；若结果文件放在运行根，则自动写到其旁的 `issues/issue-<iid>/`。Agent 在 `analysis.md` 补充实际分析，将分配结果的 artifacts 或 `response-artifacts.json` 的 files 映射到本项 `response_artifacts`，其他执行状态继续写唯一 run_state。分配实际使用原生 PATCH，`assign.md` 是用户可读的分配稿，不重复发布为评论。

response 阶段将首响和 PR 作者分配作为独立 operation，一次准备并按依赖完成；首响已完成而分配失败时只恢复分配，不等下轮重新分类才补动作。写前刷新必要评论、PR 和负责人证据，变化后重算剩余动作；写入结果未知先回查。只需分配的项目在分配回查成功后才完成本阶段，不要求文字回复文件。


指标目标：解决率 >90%、平均解决时长 <7 个自然日、1 个工作日内有效响应；只据真实证据计算。
