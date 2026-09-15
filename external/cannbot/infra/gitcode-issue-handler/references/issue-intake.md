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

步骤 0 成功后、执行步骤 1 至步骤 2b 前完整读取本文件。

## 输入

- 已完成同步的目标仓库。
- 仓库 URL 或单 Issue URL。
- `GITCODE_TOKEN`。
- 批量模式下的 `.cannbot/gitcode-issue-handler/config/classify_config.yaml` 或
  `--repo owner/repo`；单 Issue 不需要配置文件。

## 步骤 1：获取 Issue

批量模式先获取 Issue 元数据。默认合并三类来源：常规 open/创建时间窗口、核心状态为
all 的 `updated_at` 增量、以及 follow-up watchlist 中每个 Issue 的定点刷新。分类器完成
PR 文本关联后，再只为可能改变分类结果的 Issue 获取评论：

```bash
# 批量模式：先落盘，避免流式上游失败使整轮结果作废
python3 "$ISSUE_HANDLER_SKILL_ROOT/scripts/fetch_issues.py" --since YYYY-MM-DD \
  > .cannbot/gitcode-issue-handler/data/issues.json

# 单 Issue 模式
python3 "$ISSUE_HANDLER_SKILL_ROOT/scripts/fetch_issues.py" \
  --issue https://gitcode.com/<owner>/<repo>/issues/<iid>
```

批量模式可使用 `--today`、`--since YYYY-MM-DD`、`--until YYYY-MM-DD` 和
`--state opened|closed|all`。完整参数以脚本 `--help` 为准，不在本文复制。
CLI 参数优先于环境变量；`--url` 与 `--issue` 互斥。没有匹配 Issue 时报告并结束。

常规列表按创建时间倒序分页。指定 `--today` 或 `--since` 时，获取器到达时间下界后必须
停止继续翻页；但该创建时间窗口不得过滤 `updated` 或 `watchlist` 来源，否则旧 Issue 的
新回复会丢失。全状态增量按更新时间倒序读取，首次默认回看 30 天，之后使用
`.cannbot/gitcode-issue-handler/data/followup-watch.json` 的游标；只有扫描完整时才能推进游标。
达到页数上限或请求失败时保留旧游标，下轮重试。watchlist 不受核心 open/closed 和时间窗
影响，逐项 GET。`--no-follow-up` 只用于明确的诊断/兼容场景，不得用于日常批量工作流。
获取器会自动读取存在的统一 `classify_config.yaml`；显式 `--config` 和 follow-up CLI 参数
优先，可覆盖 watch 文件、首次回看天数和扫描页数。

评论和原生 PR 关联结果逐项写入 `.cannbot/gitcode-issue-handler/cache/issues/`；
Issue/PR 未更新时直接
复用，失败后重跑从未完成项续跑。HTTP 429 必须按响应中的重试窗口原地等待，不得从头
重抓。`--with-comments` 仅用于确需全量评论快照的场景。

fetch 与 classify 的真实 HTTP attempt 共用
`.cannbot/gitcode-issue-handler/cache/gitcode-rate-limit/` 中的滚动窗口状态，默认
45 次/60 秒、突发 1；重试同样计入。输出中的 `transport` 记录 attempt、主动等待次数/
秒数和 429 次数，`comment_fetch.comment_pages` 记录实际评论页数。

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

不要只凭 `updated_at` 判断有效响应或解决状态。工作日默认使用 Asia/Shanghai、
排除周六和周日；无法获取节假日日历时在报告中注明。

记录处理模式：

- `single`：使用 `--issue`；明确涉及具体算子时仍执行责任人解析和转交，除非用户对
  当前 Issue 明确选择 `direct`。
- `batch`：使用仓库 URL 和列表过滤；明确涉及具体算子时按责任人映射转交，缺失映射
  按步骤 2c 加入方案输入队列，在统一执行预览生成前一次合并询问。

## 步骤 2a：固定规则分类

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

输入带有 `filters.since` 时，该窗口是唯一时间口径，分类器不再叠加
`last_check.json`。不带显式窗口时才使用增量游标；全量处理当前输入使用
`--ignore-last-check`。本次 intake 固定传 `interactive`，即使存在 Token 或初始请求写了
“auto apply”也只预览自动指派。分析后统一执行确认尚未发生，禁止在分类阶段传
`--authorization-mode approved_batch`；`--no-auto-assign` 保留为强制 dry-run 覆盖开关。

分类器输出：

- `bucket`：`need_attention` / `no_attention`
- `category`、`reason`、`linked_prs`、`comments_count`、`auto_action`
- 汇总字段：`total`、`by_bucket`、`since`、`all_clear`、`authorization_mode`、`dry_run`
- `association_scan`：PR 页数、原生关联补查次数、预算耗尽和请求失败诊断
- `conversation_state`、`waiting_on`：下一行动者是维护者、提出者还是 assignee，以及最近评论 ID/时间
- `followup_sla`：提出者追加评论后的 1 个工作日响应时钟

关联 PR 时先在标题和正文中本地解析 `#N` 与 `/issues/N`。默认仅在标题、正文或
分支出现目标 Issue 裸编号时调用原生接口消歧，并复用缓存；确需穷举时显式使用
`--full-pr-linkage-scan`。扫描失败或预算耗尽只阻塞受影响 Issue 的外部动作。关闭且未
合入的 PR 不作为 Issue 已处理证据。

规则：

- `batch/no_attention`：不得进入后续处理；步骤 9 只保留聚合计数，不写入逐 Issue 报告。
- `single`：分类器保留原始结果为 `classification_bucket`，并输出
  `must_handle: true`。显式目标始终进入诊断；已有责任人、回复或 PR 是避免重复动作的证据，
  不是静默结束本次请求的条件。单 Issue 分类器不自动发送 `/assign`。
- Issue 作者与关联 PR 作者相同时视为自提自处理，不发送 `/assign`，也不记录到最终报告。
- `need_attention`：继续首响检查和 2b。
- `approved_batch` 自动 `/assign` 后必须 GET 回查 assignee；POST 失败或回查不一致都降级为
  `need_attention / auto_assign_failed`。
- Issue 提出者自己的评论是补充材料，不计为维护侧有效回复；评论作者未知时不得据此
  推断 Issue 已解决。
- 有效回复排除纯 `/assign`、系统消息和仅 `@owner`。优先使用 GitCode 明确的系统消息字段；
  字段缺失时，只有“已知平台机器人身份 + 已识别的固定系统模板”同时命中才按系统消息
  排除，不能仅凭机器人身份或相似文案过滤真实回复。
- 评论按 `created_at` 排序；提出者在最新维护侧实质回复后追加评论时，无论已有 assignee、
  关联 PR、核心 state 或创建时间，都必须进入 follow-up 分类。评论获取不完整时不得根据
  缓存旧顺序执行状态写入。

`replied_no_owner` 需要一次算子路由复核，不能仅因已有实质回复就永久跳过：用标题、正文、
报错栈和明确文件路径做轻量检查；若证据明确点名具体算子且 Issue 仍无 assignee，将其提升
为 `need_attention / operator_routing_required`，保留已有评论作为首响证据并进入步骤 2b–2d。
非算子 Issue 仍保持 `no_attention`。不要仅凭常见函数名猜测算子，也不要把该复核扩展成
代码诊断。

常见分类：

| category | bucket | 含义 |
| --- | --- | --- |
| `self_assigned` | no_attention | 提出者已自行负责 |
| `auto_assign_via_pr` | no_attention | 已根据关联 PR 自动指派 |
| `auto_assign_failed` | need_attention | 自动指派失败 |
| `association_scan_incomplete` | need_attention | PR 列表扫描失败，等待自动重试且不执行外部动作 |
| `comment_scan_incomplete` | need_attention | 评论获取失败，等待从缓存断点续跑 |
| `reporter_followup` | need_attention | 提出者在维护者回复后追加评论，需优先跟进并恢复`进行中` |
| `reopened_followup` | need_attention | 提出者在关闭/终态 Issue 追加评论，需 reopen、恢复`进行中`后跟进 |
| `assignee_followup` | need_attention | 被等待责任人在挂起后新增回复，需恢复`进行中`并判断是否解决 |
| `awaiting_reporter_setup` | need_attention | 等待提出者的 watch 存在但状态偏离`挂起`，需修复状态 |
| `awaiting_assignee_setup` | need_attention | 已有首响且明确等待责任人，需切`挂起`并建立 assignee watch |
| `awaiting_reporter` | no_attention | 已明确请求补充，当前保持`挂起`且由 watchlist 定点刷新 |
| `awaiting_assignee` | no_attention | 已转交责任人且尚未解决，当前保持`挂起`并定点刷新，不参与静默关闭 |
| `needs_manual_no_pr_author` | need_attention | 有 PR 但无法确定作者 |
| `needs_first_look` | need_attention | 无负责人、PR 和有效评论 |
| `needs_only_assign_cmd` | need_attention | 只有 `/assign` 评论 |
| `replied_no_owner` | no_attention | 无负责人但已有实质回复 |
| `operator_routing_required` | need_attention | 已回复但无 assignee，且明确涉及具体算子，需解析责任人 |
| `our_team_done_with_pr` | no_attention | 已有负责人和 PR |
| `our_team_only_assign_cmd` | need_attention | 团队负责但只有指派命令 |
| `our_team_replied` | no_attention | 团队已实质回复 |
| `our_team_needs_work` | need_attention | 团队负责但尚无响应 |

## 首响检查与排序

分类后立即检查全部 `need_attention` Issue。进入耗时分析或构建前，先为已经达到或
即将达到 1 个工作日且无有效响应的 Issue 准备简短受理评论，把目标和完整正文排入统一
执行预览；确认前禁止提交。批准后实际提交并 GET 回查。评论只写当前判断或真正缺失的
最小信息，不披露维护侧环境、CI、权限和内部状态。

算子转交还有更严格的首响门禁：步骤 2c 一旦确认是具体算子 Issue，若仍没有维护侧有效
响应，统一预览必须把简短受理评论排在指派前；批准后先发送并 GET 回查成功，才能执行
指派。预览不算已首响。纯 `/assign`、系统消息和仅 `@owner` 不计为有效首响。

排序优先级：

1. `reporter_followup` / `reopened_followup` / `assignee_followup`，其中 `followup_sla` 已 breached 或 at_risk 优先
2. `first_response_sla` 为 `breached` 或 `at_risk`
3. 未解决且 `issue_age_days >= 7`
4. 未解决且 `issue_age_days >= 5`
5. 其他 `need_attention`

同级按创建时间从早到晚。中间状态仍需记录责任人、阻塞原因、下一步和更新时间。

## 步骤 2b：补齐详情

处理 `batch/need_attention` 和显式 `single` 目标。步骤 1 数据已足够时直接复用；需要刷新时按
`gitcode-toolkit` 的 Issue API 规则重新获取。

从标题、正文、评论和附件提取：

- 报错和日志短语
- 复现命令、输入样例、期望与实际行为
- 文件、函数、类或算子名
- CANN/框架版本、源码 commit/分支、SoC、CPU 架构、OS、Python 和工具路径

禁止脑补。环境敏感问题缺少必要字段时，准备索要最小字段的完整评论并加入统一预览，
标记 `waiting_context`，继续其他 Issue；批准前不得回评或虚报评论已发送。

图片先下载到
`.cannbot/gitcode-issue-handler/images/issue-<iid>-<index>-<filename>`，再使用
当前工具的图片查看能力分析。提取界面表现、错误日志、期望效果和操作步骤。

## 并行边界

多个 `need_attention` Issue 的 2b、2c、2d 可以按 Issue 独立并行，但 2e 必须等待
全部分析结束。并行任务不得修改代码或运行复现命令。

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

其中只有进入实际处理流程的 Issue 才写入最终 `issues` 状态；`batch` 的
`no_attention` 仅进入分类器输出和聚合计数，禁止为生成报告补齐大段“不适用”字段。

完成后进入 `issue-routing.md`。
