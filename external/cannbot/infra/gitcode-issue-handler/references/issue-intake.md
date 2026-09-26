# Issue 流程：获取、分类与优先级

目录：

- [读取时机](#读取时机)
- [输入](#输入)
- [步骤-1获取-issue](#步骤-1获取-issue)
- [步骤-2a固定规则分类](#步骤-2a固定规则分类)
- [首响检查与排序](#首响检查与排序)
- [步骤-2b补齐详情](#步骤-2b补齐详情)
- [并行边界](#并行边界)
- [输出](#输出)

## 读取时机

single、兼容旧 CLI 获取分类或需要解释分类规则时读取；已配置 batch 先读 [pipeline.md](pipeline.md)，新一轮执行 `issue_pipeline.py resume --new-run --repository-root .`，恢复同一轮使用普通 `resume`，按 `next_action` 按需加载规则。范围缓存、未完成队列及必要重分类由脚本维护，不要求先手工首次分类再调查全批。只做 API 答疑不以 Git 同步为前提。

## 输入

- 已解析的目标仓库；只做 API 答疑不要求提前 fetch/同步源码，依赖代码版本时再同步。
- 仓库 URL 或单 Issue URL。
- `GITCODE_TOKEN`。
- 批量模式下的 `.cannbot/gitcode-issue-handler/config/classify_config.yaml` 或 `--repo owner/repo`；缺失配置会自动初始化，目标确认后保存 `repo`。

single 每次新触发先按 [runtime-state.md](runtime-state.md#初始化) 创建日期时间报告目录，再获取该 Issue；中断续跑使用原目录和状态。

## 步骤 1：获取 Issue

以下 CLI 继续支持 single、诊断及兼容场景；正常 batch 优先走 pipeline，避免另行维护同一批输入与任务状态。入口变化不绕过以下过滤、责任或授权规则。

正常 `single`/`batch` 只处理核心 open Issue。先获取元数据并过滤 closed，再做责任范围核查、PR 关联及评论获取；显式 single、增量更新和 watchlist 均不能绕过此过滤。closed 只保留跳过结果，不跟进、不自动 reopen。

批量合并常规 open/创建时间窗口、open 的 `updated_at` 增量以及 follow-up watchlist 定点刷新。watch 项先查核心状态，closed 即停止该项后续查询。通过入口与责任门禁后，只为可能改变分类结果的 Issue 获取评论和 PR 证据：

```bash
# 批量模式：先落盘，避免流式上游失败使整轮结果作废
python3 "$ISSUE_HANDLER_SKILL_ROOT/scripts/fetch_issues.py" \
  --url "https://gitcode.com/<owner>/<repo>" --since YYYY-MM-DD \
  > .cannbot/gitcode-issue-handler/data/issues.json

# 单 Issue 模式
python3 "$ISSUE_HANDLER_SKILL_ROOT/scripts/fetch_issues.py" \
  --issue https://gitcode.com/<owner>/<repo>/issues/<iid>
```

批量模式可使用 `--today`、`--since YYYY-MM-DD`、`--until YYYY-MM-DD` 和 `--state opened`；正常获取 CLI 不接受 `closed|all`，全状态历史读取由独立知识维护脚本完成。完整参数以脚本 `--help` 为准，不在本文复制。
CLI 参数优先于环境变量；`--url` 与 `--issue` 互斥。没有匹配 Issue 时报告并结束。
批量获取时将 `<owner>/<repo>` 替换为步骤 0 已解析的目标仓库，并显式传入 `--url`；获取器优先使用已确认并保存的 `repo`；传入 URL 必须与配置一致。为空时自动推导并保存，歧义或冲突返回 `needs_selection`，按 `runtime-setup.md` 问卷确认后继续。

常规列表按创建时间倒序分页。指定 `--today` 或 `--since` 时，获取器到达时间下界后必须停止继续翻页；但该创建时间窗口不得过滤 `updated` 或 `watchlist` 来源，否则旧 Issue 的新回复会丢失。open 增量按更新时间倒序读取，首次默认回看 30 天，之后使用 `.cannbot/gitcode-issue-handler/data/followup-watch.json` 的游标；只有扫描完整时才能推进游标。
达到页数上限或请求失败时保留旧游标，下轮重试。watchlist 不受创建时间窗影响，逐项 GET 后仍执行核心 closed 过滤。`--no-follow-up` 只用于明确的诊断/兼容场景，不得用于日常批量工作流。
获取器会自动读取存在的统一 `classify_config.yaml`；显式 `--config` 和 follow-up CLI 参数优先，可覆盖 watch 文件、首次回看天数和扫描页数。

评论和原生 PR 关联结果逐项写入 `.cannbot/gitcode-issue-handler/cache/issues/`； Issue/PR 未更新时直接复用，失败后重跑从未完成项续跑。HTTP 429 必须按响应中的重试窗口原地等待，不得从头重抓。`--with-comments` 仅用于确需全量评论快照的场景。

fetch 与 classify 的真实 HTTP attempt 共用 `.cannbot/gitcode-issue-handler/cache/gitcode-rate-limit/` 中的滚动窗口状态，默认 45 次/60 秒、突发 1；重试同样计入。输出中的 `transport` 记录 attempt、主动等待次数/ 秒数和 429 次数，`comment_fetch.comment_pages` 记录实际评论页数。

每个 Issue 至少记录：

- `iid`、`title`、`description`、`labels`、核心 `state`、自定义 `issue_state`、`created_at`
- `comments` 和 `comments_fetch`（获取、缓存、跳过或失败）
- `fetch_sources`、`followup_watch`、`conversation_state`
- `issue_age_days`
- `first_effective_response_at`
- `first_response_sla`：`met` / `at_risk` / `breached` / `unknown`
- `resolution_status`：`resolved` / `resolution_pending` / `unresolved` / `excluded`
- `resolution_duration_days`
- `followup_pending_since`、`followup_sla`、`reopen_required`、`activate_required`

不要只凭 `updated_at` 判断有效响应或解决状态。工作日默认使用 Asia/Shanghai、排除周六和周日；无法获取节假日日历时在报告中注明。

记录处理模式：

- `single`：使用 `--issue`；明确涉及具体算子时仍执行责任人解析和转交，除非用户对当前 Issue 明确选择 `direct`。
- `batch`：使用仓库 URL 和列表过滤；明确涉及具体算子时按责任人映射转交，缺失映射按步骤 2c 在起草首响时分析核心贡献候选，写入 summary 并保留线下确认队列；确认后再准备指派。

## 责任范围

pipeline 优先复用仍有效的范围证据，仅为缺失或失效项生成核查任务；`pending / responsibility_review_required` 不是发送清单。简单核查由主会话批量完成，复杂独立调查按 [batch-analysis.md](batch-analysis.md) 委派，逐项提交和审核，不等所有范围核查结束。调查者按 [responsibility-scope.md](responsibility-scope.md) 核查实际实现与配置；兼容旧 CLI 时由协调者写回 `responsibility_review` 后对子集重分类。仅调查范围的子 agent 无需加载本文的获取、游标和排序细节。

## 步骤 2a：固定规则分类

同批需要重分类时，可传 `--pr-snapshot <本轮独立目录/pr-snapshot.json>`：首次抓取后复用该批 PR 列表；跨运行必须使用新路径。`--no-cache` 禁用快照；目标或扫描上限不匹配、快照未覆盖本次时间窗，以及上次扫描不完整时重新抓取。对子集重分类同时传 `--no-update-last-check` 和同批快照，合并保留其他项结果；子集完成不能推进全批游标或宣称全批完成。

含关联 PR 的分类必须刷新，不能假定 PR 状态变化一定会更新 Issue 时间。其余分类缓存仅在 batch 复用已稳定的 `no_attention` 项：核心 open、已有 assignee 和维护侧实质回复，且 Issue 元数据、本地 watch、责任配置等匹配时保留原分类及状态字段。single、`--refresh-comments`、`--no-cache` 均绕过；新评论或 owner 变化必须重新分析。缓存是既有评论证据的复用，不能仅凭评论数或 assignee 推断首响，也不构成外部写操作授权。

```bash
python3 "$ISSUE_HANDLER_SKILL_ROOT/scripts/classify_issues.py" \
  --config .cannbot/gitcode-issue-handler/config/classify_config.yaml \
  --input .cannbot/gitcode-issue-handler/data/issues.json \
  --authorization-mode interactive
```

单 Issue 输入已包含 `filters.mode=single` 和 `filters.repository`，可直接运行：

```bash
python3 "$ISSUE_HANDLER_SKILL_ROOT/scripts/classify_issues.py" \
  --input .cannbot/gitcode-issue-handler/data/issues.json
```

输入带有 `filters.since` 时，该窗口是唯一时间口径，分类器不再叠加 `last_check.json`。不带显式窗口时才使用增量游标；全量处理当前输入使用 `--ignore-last-check`。本次 intake 固定传 `interactive`，即使存在 Token 或初始请求写了“auto apply”也只分类，不进行指派。`--authorization-mode` 和 `--no-auto-assign` 保留兼容；分类器不执行外部指派；首响回查后的 owner/有效 PR 作者/候选分支按 [automation.md](automation.md) 独立执行。

分类器输出：

- `bucket`：`need_attention` / `no_attention`
- `category`、`reason`、`linked_prs`、`comments_count`、`auto_action`
- 汇总字段：`total`、`by_bucket`、`since`、`all_clear`、`authorization_mode`、`dry_run`
- `association_scan`：PR 页数、原生关联补查次数、预算耗尽和请求失败诊断
- `conversation_state`、`waiting_on`：下一行动者是维护者、提出者还是 assignee，以及最近评论 ID/时间
- `followup_sla`：提出者追加评论后的 1 个工作日响应时钟

核查显式 PR 关系时先在 PR 标题和正文中解析 `#N` 与 `/issues/N`；默认仅对含目标 Issue 歧义编号的 PR 查询原生关联，并复用缓存。随后按每个待核查 Issue 补查原生关联 PR 列表，以及 Issue 正文/评论中指向本仓的明确 PR 链接；这些定向读取不受近期 PR 更新时间窗口限制，不消耗歧义消解预算，复用共享限流。关联列表最多读取 `pr_fetch_pages` 页，读取失败、返回数据无效或达到分页上限时，只阻塞受影响 Issue，不把缺失证据当作无关联 PR。仅用户明确要求穷举关联时使用 `--full-pr-linkage-scan`。无显式关系时，不按算子名、路径或现象搜索潜在修复 PR，也不把相似 PR 加入首响；基于 Issue 和必要源码给潜在处理方向。关闭且未合入的 PR 不作为有效 PR；已合入仍有效。过期须有明确失效证据，不按静默天数推断。分类和分配共用 `pr_response_policy.py`；历史关联不删除，不把它们混入有效 PR 作者选择。

规则：

- 先判定响应对象。已确认的纯路线图、开发规划或工作汇总在批量处理中整体归 `planning_record/no_attention`，不因新增评论、缺少负责人/PR、责任范围待核查或评论获取失败而进入处理；用户明确点名的单项任务仍按 single 入口执行。不能仅因标题/正文出现 roadmap 一词就跳过具体缺陷或提问；正文报告当前故障或请求答复时须按实际诉求分类，规划中的“计划修复错误”本身不构成问题报告。
- 需求与缺陷统一按账号关系判断自提，不再以“需求建议 + 任意 assignee”豁免。assignee 与 Issue 作者为同一 GitCode 账号即 `self_assigned/no_attention`，即使 PR 由他人提交也无须另发首响；assignee 不同账号时再核对 PR 是否有同作者证据。新追问继续走 follow-up。
- `batch/no_attention`：不进入本轮首响和分配；已有未闭环处理记录、等待状态和 watch 继续保留，新追问仍走跟进。步骤 9 不把纯观察项计作本轮处理。
- `list-only` 复用同一 attention 判定但始终不进入处理流程：仅当其原始判定为 `need_attention` 时进入 `listed_issues`；原始判定为 `no_attention` 时只保留责任范围聚合计数。不得只凭 assignee 或评论数隐藏，仍须核实实质回复、有效 PR 和新追问。
- `single`：分类器保留原始结果为 `classification_bucket`，并输出 `must_handle: true`（仅核心 open）。责任范围为 `handle` 的 open 显式目标进入诊断；`pending/list-only/ignore` 不因 single 绕过范围门槛。
  已有责任人、回复或 PR 是避免重复动作的证据，不是静默结束本次请求的条件。单 Issue 分类器不自动发送 `/assign`。
- 对责任范围内尚无他人实质响应、且不满足任一自提条件的 Issue，已有有效关联修复 PR 仍进入 `need_attention / needs_first_response_with_pr`；先回复并回查，再沿用已有 PR/负责人跟进策略，不在分类时自动 `/assign` 绕过首响。关联 PR 不能成为跳过评论获取的理由；先完整读取评论，扫描失败保留待重试，避免漏响或重复首响。
- 自提身份比较使用非空 GitCode login，忽略大小写和首尾空格；显示名或两个缺失账号不能作为同一人证据。当前 assignee 与 Issue 作者同账号、或任一有效关联 PR 同作者，任一条件足够；多 PR 只要有一条同作者且确实覆盖本 Issue 即走自提路径，无负责人时优先分配该作者。无关或仅相似 PR 不算。已经由 assignee 同账号确立豁免时，其他 PR 作者缺失或关联扫描不全不产生新的首响义务；评论证据不全和真实新追问仍按各自门禁处理。未实际处理的自提项不写入逐项报告。
- 已有他人实质响应、责任人已落实且无新追问/责任人新回复时，归 `no_attention`，不重复首响或调查。历史未切挂起、未建 watch、PR 尚未合入均不单独触发本次响应；不为状态补录把它加入待处理列表或报告卡点。新追问仍跟进，未闭环不等于待首响。
- `need_attention`：继续首响检查和 2b。
- 临时指派由独立执行器在首响回查后执行并 GET 核对 assignee；分类器不执行外部写入。
- Issue 提出者自己的评论是补充材料，不计为维护侧有效回复；评论作者未知时不得据此推断 Issue 已解决。
- 有效回复排除纯 `/assign`、系统消息和仅 `@owner`。优先使用 GitCode 明确的系统消息字段；字段缺失时，只有“已知平台机器人身份 + 已识别的固定系统模板”同时命中才按系统消息排除，不能仅凭机器人身份或相似文案过滤真实回复。
- 评论按 `created_at` 排序；提出者在最新维护侧实质回复后追加评论时，无论已有 assignee、关联 PR 或创建时间，只要核心仍为 open，就必须进入 follow-up 分类。评论获取不完整时不得根据缓存旧顺序执行状态写入。

`replied_no_owner` 需要一次算子路由复核，不能仅因已有实质回复就永久跳过：用标题、正文、报错栈和明确文件路径做轻量检查；若证据明确点名具体算子且 Issue 仍无 assignee，将其提升为 `need_attention / operator_routing_required`，保留已有评论作为首响证据并进入步骤 2b–2d。
非算子 Issue 仍保持 `no_attention`。不要仅凭常见函数名猜测算子，也不要把该复核扩展成代码诊断。

先应用上述内容类型和新跟进规则，再按下表判断首响与分配（纯 `/assign` 不算文字首响）。这五种组合不把旧文字回复当作新追问的已答复证据：

| 有效 PR | 负责人 | 回复/作者关系 | 分类及 response 阶段动作 |
|---|---|---|---|
| 无 | 无 | 无首响 | `needs_first_look` / `needs_only_assign_cmd`，首响，可参与 auto-assign |
| 无 | 有 | 负责人与 Issue 同作者 | `self_assigned`，no_attention，免首响，无需再指派 |
| 无 | 有 | 不同作者、无首响 | `our_team_needs_work` / `our_team_only_assign_cmd`，只首响 |
| 有 | 无 | 任意 PR 与 Issue 同作者 | `needs_pr_owner_handoff`，need_attention，只分配该作者 |
| 有 | 无 | 全部非自提、无首响 | `needs_first_response_with_pr`，首响概述 PR 方案并分配覆盖面最大的 PR 作者 |
| 有 | 无 | 全部非自提、已有首响 | `needs_pr_owner_handoff`，need_attention，只补 PR 作者分配 |
| 有 | 有 | 负责人或任意有效 PR 与 Issue 同作者、无首响 | `self_assigned`，no_attention |
| 有 | 有 | 负责人及全部有效 PR 均非同作者、无首响 | `needs_first_response_with_pr`，首响概述 PR，保留负责人 |
| 有 | 有 | 已有首响 | `our_team_done_with_pr`（自提保留 `self_assigned`），no_attention |

账号关系冲突时仍用“任一同作者即自提”：现任 assignee 就是 Issue 作者、但 PR 全部由他人提交，也免首响；不以 PR 作者关系覆盖已成立的负责人同账号条件。

多个 PR 先过滤失效项，再应用自提优先和问题覆盖面选择；分配属于 auto-response，在 response 阶段完成，不受 auto-assign 限制。分类器只给出 `auto_action`，不会调用外部写入。已有负责人不覆盖；等待状态不能吞掉缺失的 PR 作者分配。

所有 PR 已失效且已有负责人时：已有文字首响的归 `our_team_replied/no_attention`；历史同作者 PR 已核实的保留 `self_assigned/no_attention`，即使此前只有 `/assign` 也不补首响。两者均继续原有未闭环跟踪，不记已解决。没有负责人时不根据失效 PR 新增分配；没有首响也没有历史自提豁免时仍补首响。已核验历史豁免仅在没有有效 PR 时适用，不能覆盖其他有效非自提 PR 的首响要求。新追问保持 follow-up 优先。

抓取或责任证据不完整的原分类保留 need_attention，但只列具体阻塞和下一步，不发送猜测回复。自动响应关闭时同样完成逐项分析和草稿，列出准确的回复/分配动作等待指令。首次采用新规则做全量复核时，获取全部 open 输入，并用 `--ignore-last-check` 分类；日常仍按用户指定时间窗/增量获取。

需要解释具体 `category` 时查 [classification-categories.md](classification-categories.md)，正常执行直接消费脚本输出。

## 首响检查与排序

保留既有 category、bucket 和业务状态，先按已核验评论及 owner 证据分流，再决定调查深度：新追问走 follow-up；无实质回复走首响；已有回复且 owner 明确时复用处理记录；已有回复但缺 owner 的算子项只做必要路由复核。评论数和 assignee 只能作查询线索，不能推断已首响；证据不完整时补取相应评论，不重做无关调查。

对原策略筛出的新首响，按 [response-writing.md](response-writing.md) 形成有依据的回复，再按 `issue-comment-workflow.md` 核对授权并发布回查，进入后续处理；纯答疑直接给答案。SLA 决定顺序，预览不算已首响。

排序优先级：

1. `reporter_followup` / `reopened_followup` / `assignee_followup`，其中 `followup_sla` 已 breached 或 at_risk 优先
2. `first_response_sla` 为 `breached` 或 `at_risk`
3. 未解决且 `issue_age_days >= 7`
4. 未解决且 `issue_age_days >= 5`
5. 其他 `need_attention`

持久任务队列在同一优先级内先处理预计工作量较大的任务，再按创建时间排序；旧分类列表保持创建时间从早到晚。中间状态仍需记录责任人、阻塞原因、下一步和更新时间。

## 步骤 2b：补齐详情

处理 `batch/need_attention` 和显式 `single` 目标。pipeline 按就绪任务逐项推进，复杂独立调查与拟稿按 [batch-analysis.md](batch-analysis.md) 分工，并为每条 `need_attention` 先保存已知分析，补齐适用草稿和阻塞原因；自动响应关闭也执行。步骤 1 数据已足够时直接复用；需要刷新时按 `gitcode-toolkit` 的 Issue API 规则重新获取。

获取脚本的完整正文在 `description` 字段（原生 API 可能为 `body`）。先读完整正文及代码块，不能仅以标题或截断列表做范围判定、起草或索要信息。

从标题、正文、评论和附件提取：

- 报错和日志短语
- 复现命令、输入样例、期望与实际行为
- 文件、函数、类或算子名
- CANN/框架版本、源码 commit/分支、SoC、CPU 架构、OS、Python 和工具路径

禁止脑补。环境敏感问题缺少必要字段时，准备索要最小字段的完整评论并加入统一预览，标记 `waiting_context`，继续其他 Issue；批准前不得回评或虚报评论已发送。

图片先下载到 `.cannbot/gitcode-issue-handler/images/issue-<iid>-<index>-<filename>`，再使用当前工具的图片查看能力分析。提取界面表现、错误日志、期望效果和操作步骤。

## 并行边界

范围待核查和分类后响应准备按动态队列推进，简单项主会话处理、复杂项按 [batch-analysis.md](batch-analysis.md) 分工；逐项 `claim/submit/accept`，空闲补位，不等全批返回。同一项拟稿等待必要证据，外部等待用 `hold` 释放名额，不阻塞其他独立任务。
单项回复证据和授权就绪即发布回查，不等全批或 owner 查询；2e 只等待当前代码组的依赖和冲突核查。调查任务不得修改代码或运行复现命令。

## 输出

每个 Issue 至少更新：

```yaml
mode: single | batch
bucket: need_attention | no_attention
category:
reason:
issue_age_days:
first_response_sla:
resolution_status:
signals: []
required_environment: {}
```

其中只有进入实际处理流程的 Issue 才写入最终 `issues` 状态；`batch` 的 `no_attention` 仅进入分类器输出和聚合计数，禁止为生成报告补齐大段“不适用”字段。

仅需处理的范围内项目进入 `issue-routing.md`；其余按责任级别汇总。

## 批量内容筛选

纯路线图/规划汇总不进入响应。现任负责人或任一有效关联 PR 作者与 Issue 作者同账号，即按自提处理；需求与缺陷使用相同判定。已有实质回复和责任人、无新跟进时也不纳入本轮，不为历史状态/watch 补录重新激活。
