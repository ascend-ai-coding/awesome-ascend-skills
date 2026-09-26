# 运行时：最小授权与状态契约

## 读取时机

真实执行请求在步骤 -1 初始化唯一运行状态时读取本文件；`policy_query` 不建立状态，也不读取执行阶段 reference。完整字段、枚举和兼容字段见 [runtime-state-schema.md](runtime-state-schema.md)，只有当前阶段需要的字段才按需读取。

## 初始化

- `single`、`batch` 均从 `authorization_mode: interactive` 开始。用户说“处理 Issue”“自动执行”不产生批次交付批准；首响、关联与指派可按 [automation.md](automation.md) 的配置及会话授权执行。
- 批量新轮次由 `resume --new-run` 建立；单条处理在确认目标配置后执行 `python3 "$ISSUE_HANDLER_SKILL_ROOT/scripts/report_runs.py" start --repository-root . --mode single --iid <编号>`。使用返回的 `run_id`、`report_directory`、`state_file`，续跑复用原值。目录固定为北京时间 `YYYYMMDD-HHMMSS-P0800`，同秒冲突追加 `-02` 起的序号；本轮元信息和内部 ID 保存到 `run.json`。
- 在返回的 `_internal/run_state.json` 记录 `mode`、时间、`overall_status: running`、`capability_checks`、目标仓库和后续阶段待填字段；不为补齐 schema 执行环境探测；不得用虚假值补齐条件字段。
- 真实操作按阶段更新同一状态，不另建平行口径；条件字段不适用时省略。每次分类、诊断、复现、修改、验证、授权、发布、回查或状态转换后立即写入结果和证据。
- 各阶段只更新自己负责的字段。未进入的能力保持 `not_started`，终态可记 `not_required`；能力状态仅在对应真实操作紧前更新，能力就绪不等于业务写入授权。外部写入必须记录稳定 `operation_id`、授权证据、结果和回查证据。

## 授权与回复

授权模型、回复检查点、批次批准失效条件和 direct-push 独立确认统一以 [authorization-contract.md](authorization-contract.md) 为准；本文件不重复其操作表。

`response_status` 独立于首响 SLA 和解决状态。新增追问会使本轮响应重新待处理，旧评论不能自动通过门禁；`verified/reused` 必须有本轮适用的 GET 证据，`waived_by_user` 留用户原话。`exempt_self_authored_pr` 只用于已核验同作者 PR 的免首响路径，不计首响成功；由 assignee 与作者同账号确立的 `self_assigned/no_attention` 仅保留分类证据，不伪造评论结果或本轮处理记录。历史失效自提且已有负责人时保留豁免及未闭环状态。
每个后续 operation 的 `depends_on` 引用该回复或适用的历史/豁免证据。

## 能力失败与恢复

能力检查的选择、顺序和失败路由以 [runtime-capability-checks.md](runtime-capability-checks.md) 为准。一般失败只阻断依赖该能力的操作；缺少 Token 且后续已确定需要认证写操作时，按以下契约暂停整轮：

1. 保存元数据，将 `overall_status` 和 `capability_checks.api` 设为 `waiting_for_input`，创建或复用唯一 `pending_user_inputs` 项 `input_id: gitcode_token`，记录 `request_count: 1` 与准确 `resume_from`，不保存 Token 明文。
2. 只询问一次并停止 API、Issue 拉取、知识刷新、代码诊断和测试读取。未解决项存在时不重复询问或重新初始化。
3. 用户补充 Token 后标记输入 `resolved`，重跑 `api` 检查；通过后恢复 `running`/`ready`，从 `resume_from` 继续，不重复已完成阶段。

## 状态骨架

初始化至少包含 `run`、`issues`、`groups`、`external_operations`、`internal_blockers`；按需填充 schema 中字段。实际处理项在 `issues` 中标记 `handled_in_run`，仅列举项放入顶层 `listed_issues`，报告生成不得丢失后者。

## 阶段约束

每个阶段把结论和证据追加到唯一运行状态，不凭记忆重建。评论/指派/状态回查失败不记成功；停止依赖该操作的后续动作，继续其他独立 Issue。外部回复不披露维护侧环境、CI/权限故障、内部重试或敏感信息。转交、等待、PR 未合入和 CI 阻塞不算解决。

## 批量恢复与完整性

唯一队列：`.cannbot/gitcode-issue-handler/data/pipeline-state.json`。脚本用仓库级文件锁和原子替换维护；上一代保留为 `pipeline-state.previous.json`。损坏或不支持的状态原样保留为 `pipeline-state.invalid-*.json`，恢复上一代并要求完整扫描及未决操作审计。仓库不匹配直接报错，不覆盖。

- 首次、旧记录迁移、恢复异常与每隔 24 小时执行完整 open 扫描；不以默认 7 天窗口排除旧 open Issue。
- 同会话复用扫描检查点，新会话自动检查增量；同会话明确开始新一轮扫描用 `resume --refresh`。使用宿主的 CODEX_THREAD_ID/CODEX_SESSION_ID 识别会话，宿主不提供时退回 5 分钟检查间隔。增量游标回退 5 分钟读取并去重。
- 拉取使用 `--defer-cursor`，队列和流水线独立游标写入同一个原子检查点；不提前修改旧 follow-up 游标。
- 扫描失败保留旧游标；收到的部分结果仍可存入队列。不完整不等于无匹配。
- 未完成、等待、需要独立刷新 PR 的旧项持续保留；不在本次列表中时定点 GET，不能根据缺席推断关闭。
- 只有明确核心 closed 才停止该项；未知状态继续刷新。重新 open 后重新进入判断。
- 导入旧 reviewed JSON 的范围证据及旧运行中未决外部操作，不把旧报告的“完成”当作最新远端事实。

本入口是当前配置责任范围内的正常批量巡检。若用户限定特殊时间/Issue 子集，按 issue-intake 的显式范围兼容路径执行，不能把子集记录当作完整仓库基线。
