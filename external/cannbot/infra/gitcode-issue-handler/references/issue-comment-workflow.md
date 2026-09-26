# Issue Handler 评论编排

## 读取与边界

协调者在步骤 2c–2e 复用、预览、发布评论或步骤 9 发布最终回评时读取本文；只起草正文读 [response-writing.md](response-writing.md)。GitCode 目标解析、评论 POST/GET、幂等、Token 安全、限流和通用错误处理遵循 [gitcode-toolkit 评论工作流](../../gitcode-toolkit/references/issue-comment-workflow.md)。本文只规定 Issue Handler 的业务内容、授权和后续状态编排。

首响是否需要发布以 `issue-intake.md` 的账号关系和决策表为准。纯规划汇总不响应；assignee 或任一有效关联 PR 作者与 Issue 作者同账号即自提免首响，需求与缺陷一致，不因另有他人 PR 而取消豁免。自提无负责人只补必要指派；非自提无实质回复时补首响并简述适用 PR 方案。`/assign`、仅 @ 提醒和提出者自己的评论不算维护侧回复，自提豁免也不记为已发布首响。已有负责人且所有 PR 已失效时，按 intake 保留已核验的历史自提豁免。已有准确回复且没有新问题时不得重发；责任人已落实则不为补状态/watch 重新纳入批量响应。需要复用首响执行后续动作时才 GET 核对并记为 `reused`；提出者或责任人有新评论时按跟进顺序回应。用户明确要求回复、删除或重发时，仅执行准确范围。`no_attention` 不批量刷评论。

首响是否直接发送先按 [automation.md](automation.md) 读取两开关；关闭自动首响时保存草稿，审核或明确发送授权后再执行。自动首响不扩展为后续追问的自动回复。

## 正文与审核

起草、复用或审核正文按 [response-writing.md](response-writing.md) 执行；保留完整 `response_review`，质量不足或正式评分再读取其链接的评分表。调查者仅拟稿时无需加载本文的发布流程。

## 每轮门禁与依赖

1. 对 `single` 或 `batch/need_attention` 的本轮负责项读取完整问题、最新相关评论和证据，判断未回答诉求；豁免已有响应必须记录 `not_applicable_existing` 及具体依据。
2. 为每个需新回复的 Issue 写一条与问题相称的完整正文；纯答疑不拆成“已收到”再答复。显式关联 PR 先读描述、状态和相关 diff，对外只概述相关方案、进展与必要适用边界；测试缺口、实现不足等评审意见保留内部。没有显式关联时不搜索、展示潜在修复 PR，给出大致处理方向。正文/评论已明确互引但尚未原生关联的 PR，只有满足 `automation.md` 门禁后才能准备关联；标题相似或历史其他 Issue 的链接本身不够。
3. 按 [delivery-confirmation.md](delivery-confirmation.md) 回复检查点核对当前会话的准确授权。已授权则复用，不再请求；缺失时展示准确 URL、完整正文、operation ID 和依赖，只确认当前就绪评论，不等待未知 owner、PR、commit、修复或测试。可并行准备独立 Issue。
4. 首响和新追问回复取得授权后，使用本 Skill 的 `post_triage_comment.py`（`--analysis-file` 可传本项简短分析），由它核对分类结果并复用 toolkit POST/GET；最终结果回评、用户明确指定的额外评论仍用 toolkit `post_issue_comment.py`。核对目标、正文、comment ID 和时间。只有实际成功记 `verified`；预览、POST 成功码、纯 `/assign`、系统消息和仅 @owner 不算完成。失败记 `comment_failed`，保留恢复点并停止该 Issue 的依赖操作，其他独立 Issue 继续。
5. 非自提指派和其他首响依赖操作须有 `verified` 或当前适用证据充分的 `reused`。有效自提 PR 的补分配例外：无需评论结果，由 `assign_issue.py` 实时核验同作者和关联后直接分配；豁免记 `exempt_self_authored_pr`，不伪造 verified/reused。PR 关联必须先于基于该 PR 作者的临时指派，且两项分别回查。创建前再次 GET；非幂等 POST 结果未知时先回查，禁止盲目重发。回复失败或本轮必需回复失败时，停止该 Issue 的代码/PR 流程。

首响/新追问发送命令（不带 `--apply` 仅预览）：

```bash
python3 "$ISSUE_HANDLER_SKILL_ROOT/scripts/post_triage_comment.py" \
  --config <classify_config.yaml> --classification <二次分类输出.json> \
  --issue-url <准确Issue-URL> \
  --body-file <本条回复.md> --result-file <本条发送结果.json> --apply
```

关闭自动首响或回复后续追问时，用户明确批准后附加 `--reply-approved`；`--apply` 本身不表示批准。

门禁失败回到 intake 补证据、重跑分类；不得改用通用执行器绕过。该检查不替代授权和内容核实。

用户明确不评论时记录 `waived_by_user` 和原话，按明确范围缩小授权；不能从缺 Token、失败、owner 未知或已有 PR 推导豁免。正常 single/batch 发现核心 closed 即停止该项，不为评论 reopen；核心 open 的状态迁移顺序沿用 `issue-followup.md`。

## 场景状态顺序

- **算子转交**：有效首响回查后才 `/assign`；回查 assignee login 与目标 owner 一致才算成功。assignee 回查只证明转交，不证明解决。等待责任人时将 `<当前状态> -> 挂起` 与 assignee watch 纳入预览；责任人新评论先恢复`进行中`再跟进，等待期间不静默关闭。
- **索要上下文**：评论 GET 成功后才切`挂起`并写 reporter watch；普通受理、进展同步或正在排查不挂起。提出者新评论时，即使已有维护侧回复也在核心仍为 open 时先获授权恢复`进行中`并回查，再处理新内容；新维护响应和下一状态形成后才更新/删除旧 watch。
- **根因/答疑**：答疑直接回答；根因评论写现象、根因、可确认引入点和下一步，但不冒充最终解决。

## 授权、记录与顺序

每条评论是独立 operation，进入回复或交付预览，并必须展示目标 Issue 与完整正文：`single` 使用覆盖当前 operation 的 `interactive` 证据；`approved_batch` 只执行精确 Issue 清单和批准的 operation IDs；`batch/interactive` 缺授权只预览，回复检查点已授权可 POST/GET，无需切换 `approved_batch`。正文或目标实质变化使未执行批准失效；用户不评论则删除该 operation 并缩小范围。

后续操作用 `depends_on` 引用已回查回复或 `reused`/`waived_by_user` 证据；自提仅分配引用已核验的自提豁免。评论回查→指派/状态/watch 回查；责任人或提出者再回复沿同样顺序；commit→push→功能分支回查→PR→首次 CI。依赖失败标记 `skipped`，不得改走未预览替代动作。正文文件和结果文件放本轮运行目录并在恢复时复用；日志/报告记录 Issue、operation ID、comment ID、时间、状态和脱敏摘要，不记录 Token、完整敏感正文或维护侧环境。

Handler API attempt 共用目标仓运行缓存的滚动窗口，默认 45 次/60 秒、突发 1；按 API host 和 Token 不可逆摘要隔离，只记录请求时间与 429 cooldown，fetch/classify/写入复用同一缓存。
