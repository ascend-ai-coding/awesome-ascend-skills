# 首次响应与责任人指派开关

真实处理先读取 `classify_config.yaml`；默认值来自模板。配置错误在外部操作前报错。

| auto-response | auto-assign | 用户要求处理 Issue 时 |
|---|---|---|
| false | false | 保存完整回复稿，用户审核后发送；候选照常调查并写报告 |
| true | false | response 阶段完成所需首响及 PR 作者分配；自提免首响，已有回复不重发；无 PR 且责任不明时仅列候选 |
| true | true | 先按上行处理；责任仍不明时，从候选中选择一人临时指派；无候选时按 fallback 配置处理，均提示确认 |
| false | true | 报错：启用自动分配需要先启用自动响应；不执行写入 |

配置只控制本次请求目标内的首响、安全 PR 关联与负责人指派，不切换为 `approved_batch`，不授权后续追问、关闭/挂起、源码修改、commit、push 或 PR。当前会话的明确限制优先于配置。
用户明确要求直接回复，可覆盖 `auto-response: false`；明确要求“自动分配最可能的候选”可在两项均 false 时独立执行，但仍先取得有效首响证据。不要为了单次请求改写配置。
无效的 true/false 组合仍需先纠正，手动覆盖不用于掩盖配置错误。

## 首响

先按 `issue-intake.md` 的五种组合判断是否需要首响：负责人或任一有效关联 PR 作者与提出者同账号即免首响；有效同作者 PR 缺负责人时只补指派。自提豁免不是一条已发布评论，也不要求为满足执行器而补发文字。已有负责人不重复指派；新追问独立跟进。

先完成范围、分类和内容检查，每项材料按 [批量调查](delivery-reporting.md#逐项响应材料) 持久化；只分配时只需简短分析和 `assign.md`，不凑文字首响。
`post_triage_comment.py --config <配置> ... --apply` 根据配置执行；关闭自动响应时，用户审核通过或明确要求发送后加 `--reply-approved`，并把会话依据写入运行状态。
仅“处理一下”不等于关闭开关时的回复批准。无 `--apply` 仅预览；未获批准继续独立调查，回复状态记 `pending_approval`。非自提项须有本轮成功或适用的历史首响回查证据才能临时指派；有效自提 PR 免首响，直接分配并回查。

## 已明确负责人

`auto-response: true` 时，首响回查后可直接转交当前用户指定或 owner 配置中的负责人，不受 `auto-assign` 限制。沿用已有 assignee；不同账号的已有指派不自动覆盖。
当前用户明确决定优先于配置；已有关联修复 PR 且当前无 assignee 时按下节选择 PR 作者；不因候选 owner 配置不同退回 auto-assign。
这些是明确负责人转交，不标为候选临时指派。

## 有效修复 PR 的自动关联与作者临时指派

`auto-response: true` 时，有效修复 PR 分两类处理：

1. 处理前已经关联：在 response 阶段一次完成首响和缺失的作者指派。非自提无回复时先首响并回查；已有回复时复用；自提免首响。已有 assignee 则只处理必要回复，不覆盖指派。
2. Issue/PR 正文或评论已有明确互引，但尚未原生关联：首响回查后，先用 `associate_issue_pr.py` 关联并双向 GET 回查，再把执行器生成的 `linked-pr-file` 交给 `assign_issue.py` 临时指派 PR 作者。日常响应不额外搜索潜在修复 PR；用户明确要求查找时发现的候选仍须满足下列门禁，不能仅凭相似就关联或指派。

这两类都不受 `auto-assign: false` 限制，也不要求把 PR 作者证明为算子核心贡献候选；PR 作者只因正在承接该修复而成为本 Issue 的临时负责人，不因此写入长期 owner 配置。执行器写入前重新读取真实 Issue、PR 作者/状态/base/head 和现有关联；本地快照不能替代核验。只有 open/opened 或 merged 且与本问题相关的 PR 有效；closed 未合入、已核实失效和无关 PR 不适用。已有关联不按目标分支排除；新发现 PR 的自动关联仍须满足下面的基线与完整覆盖门禁。

自动关联必须同时满足以下安全门禁，并在证据文件中逐项记录：

- Issue 与 PR 属于同一仓库，PR 目标 base 与当前 Issue 的处理基线兼容，审阅后的 head 未变化；
- 已读实际 diff，`coverage_verified: true`，能说明改动如何完整覆盖本 Issue，而非仅标题相似；
- `additional_risk_reviewed: true`，变更范围、兼容性和副作用已检查，`unresolved_risks` 为空；
- `safe_to_associate: true`，没有冲突的关联 PR、不同修复方向或需要提出者/owner 决策的工程取舍；
- changed files 与验证证据非空。仅相似实现、部分覆盖、跨版本未确认、head 已变化、多个作者方向冲突或任何未解决风险，都只作为参考，不自动关联、不据此指派。

关联是独立外部 operation，依赖有效首响；POST 前 GET 防重，POST 后从 PR→Issue 与 Issue→PR 两侧回查。非幂等 POST 结果未知时只回查，不重试。关联失败或未知时停止作者指派，不把评论中的 PR 链接冒充平台关联。已有不同 assignee 时可保留安全关联，但不自动覆盖负责人。
GitCode 当前原生限制一个 PR 只能关联一个 Issue。若 PR 已关联另一个 Issue，默认阻断且不替换旧关系；仅当两条 Issue 已核验为同一问题、现有首响评论已公开同时链接原 Issue 与 PR、且其余安全门禁全部通过时，保留原生关系并记录 `duplicate_cross_reference` 降级关联，再临时指派 PR 作者。该降级必须实时回查原生关系和首响正文；不同问题、部分重复或缺少公开串联时不适用。不得为满足当前 Issue 而删除原 Issue 的关联。
有效关联 PR 中只要有一条与 Issue 同作者，就走自提路径：免首次响应，无负责人时优先分配给该作者。所有有效 PR 都非自提时，Agent 阅读描述和实际 diff，选择覆盖本 Issue 诉求最多的 PR 作者；覆盖相当时优先匹配问题版本，仍相当按 PR 编号升序稳定选择，不因多个作者直接转人工。对每个有效 PR 用同一组诉求标识记录 `coverage_review.covered_issue_points`、非空 `evidence`、`target_version_match` 和已读的 `head_sha`；执行器据此选择并校验所选 head。覆盖面指实际问题覆盖，不是代码行数或文件数。作者或覆盖证据缺失时先补查，无法取得才记录阻塞。
此例外不授权创建新 PR，也不因本轮新建 PR 触发自动指派。自动关联与作者指派均记为临时，直到维护侧确认真正责任归属。

## 临时指派

复用已确认的 owner/已有 assignee；没有时按 `operator-owner-candidates.md` 查每个实际算子。
Agent 按相关核心贡献与身份核验选择整条 Issue 最合适的一人，记录选择理由；多算子仍保留每个算子的候选表。自动推断责任人的准确性有限，请谨慎使用。
调查后整条 Issue 没有任何候选时，`auto-assign-fallback-user` 非空且 `auto-assign: true` 才临时指派兜底接收人；默认留空则只记待确认，不阻止首响。填写 GitCode login，如 `songkai111`，执行器先通过用户 API 验证账号，再指派并回查。账号无效或核验失败时不指派，报告原因。
有候选但无法排序、候选证据不合格、调查未完成或已知 owner/PR 作者存在冲突，均不能转走兜底。
兜底接收人不是核心贡献候选，不加入候选表或 owner 配置；仍提醒用户确认真正负责人。
候选推选不能只凭关联 PR 作者或测试文件提交人；已关联修复 PR 的转交走上一节独立规则。

使用 `assign_issue.py`，输入准确 Issue URL、候选文件、选定 login、已回查首响结果和结果文件。
默认预览；配置开启自动分配才 `--apply`。单次用户明确要求自动选择并指派时附 `--assignment-approved`；覆盖依据留状态，不能为绕过关闭开关自行添加。
脚本核查依赖、保留已有 assignee、原生 PATCH 并 GET 验证；同一目标已指派则复用，失败/结果未知不能记成功或盲目重试。

回查成功后记录 `assignment_provisional: true`、`assigned_candidate`、 `operator_owner_request_status: awaiting_offline_confirmation`、`assignment_status: verified`。
候选不能写成已确认 `operator_owner`，不写 `operator_owners.yaml`，不自动挂起或关闭。
summary 明确写“已临时指派 @账号，请确认真正负责人”，并保留逐算子候选简表。
真正 owner 经用户确认后才沿正常转交路径更新配置或换人。

## 执行器输入

```bash
python3 "$ISSUE_HANDLER_SKILL_ROOT/scripts/assign_issue.py" \
  --config <classify_config.yaml> --issue-url <准确Issue-URL> \
  --candidate-file <候选.json> --owner <选定login> \
  --response-result <首响结果.json> --result-file <指派结果.json> --apply
```

候选文件复用 `operator-owner-candidates.md` 的输出；选择理由留证据。已配置 owner 时，候选条目只需 `login` 和 `operators`，执行器从分类配置同目录的 `operator_owners.yaml` 核实映射，不要求额外核心贡献调查。单次明确指派命令按上文附 `--assignment-approved`。

无候选分支仍用 `--candidate-file`，省略 `--owner`，执行器读取 fallback 配置。
文件保存 `target_issue_url`、`status: insufficient_evidence`、`candidates: []`、全部已调查的 `operators` 及调查缺口；只有逐项调查完成才可声明空结果。空配置或关闭 auto-assign 时返回 `skipped`，单次 `--assignment-approved` 不启用配置中的 fallback。其余首响、已有 assignee、失败恢复门禁保持一致。结果 `assignment_source: fallback_user` 须传入运行状态，summary 写“无候选，已临时指派兜底接收人 @账号，请确认真正负责人”；不要把兜底人写进候选简表。

已有 PR 分支把 `--candidate-file` 换为 `--linked-pr-file`，可省略 `--owner` 由执行器按自提优先/覆盖审阅选择。快照保留全部有效关联 PR，不只传入想分配的那一个；`covers_issue: true` 在已有原生关联时表示实际关联且覆盖本问题的至少一项诉求，新增自动关联仍须完整覆盖。自提免首响时省略 `--response-result`；非自提仅预览时也可暂缺，真正执行时必须提供 `verified/reused` 结果。保存实际核验后的处理前快照：

```json
{
  "target_issue_url": "https://gitcode.com/owner/repo/issues/42",
  "issue_author": "reporter-login",
  "linked_prs": [{
    "pr_url": "https://gitcode.com/owner/repo/merge_requests/7",
    "pr_author": "author-login",
    "state": "open",
    "preexisting_association": true,
    "covers_issue": true
  }]
}
```

处理前已关联使用 `preexisting_association: true`；本轮由关联执行器创建并回查的关系使用 `association_verified: true` 和 `association_source: current_run`，且 `assign_issue.py` 会再次查询真实 PR 关联列表。布尔字段记录实际关联与覆盖核验，不为通过门禁随意置 true。首响结果使用发送器保存的 `verified` 结果；复用既有首响时保存相同目标的 `reused` 结果与原评论回查证据。
重复 Issue 降级由执行器写入 `public_cross_reference_verified: true`、 `association_mode: duplicate_cross_reference` 和原生关联 Issue；不得手工伪造这些字段。

新发现 PR 的关联命令（默认预览）：

```bash
python3 "$ISSUE_HANDLER_SKILL_ROOT/scripts/associate_issue_pr.py" \
  --config <classify_config.yaml> --issue-url <准确Issue-URL> --pr-url <准确PR-URL> \
  --evidence-file <关联安全审阅.json> --response-result <首响结果.json> \
  --result-file <关联结果.json> --linked-pr-file <已核验关联证据.json> --apply
```

关闭自动首响时，只有用户明确批准准确 Issue/PR 关联操作才加 `--association-approved`。
