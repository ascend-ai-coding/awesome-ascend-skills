# Issue Handler 评论编排

## 读取与边界

协调者在步骤 2c–2e 复用、预览、发布评论或步骤 9 发布最终回评时读取本文；只起草正文读 [response-writing.md](response-writing.md)。GitCode 目标解析、评论 POST/GET、幂等、Token 安全、限流和通用错误处理遵循 [gitcode-toolkit 评论工作流](../../gitcode-toolkit/references/issue-comment-workflow.md)。本文只规定 Issue Handler 的业务内容、授权和后续状态编排。

首响以有效评论证据为准：非自提处理项须有可核验回复；自提免首响，但不豁免缺失负责人或后续追问。提出者与负责人同账号，或提出者与有效关联 PR 作者同账号，均按自提处理。纯规划汇总按 intake 不进入批量处理。`/assign`、系统通知、仅 @ 提醒和提出者自己的补充不算维护侧首响。已有准确回复且无新问题时复用，不重发；已有负责人不覆盖。需要复用首响执行后续动作时 GET 核对并记录 `reused`；提出者或责任人有新评论时按跟进顺序回应。`no_attention` 不批量刷评论。

首响是否直接发送先按 [automation.md](automation.md) 读取两开关；关闭自动首响时保存草稿，审核或明确发送授权后再执行。自动首响不扩展为后续追问的自动回复。

## 正文与审核

起草、复用或审核正文按 [response-writing.md](response-writing.md) 执行；保留完整 `response_review`；咨询拟稿（含模拟）按其链接的评分表记录咨询档位、已满足事实、缺项与理由，其他类型质量不足或正式评分时再展开评分表。调查者仅拟稿时无需加载本文的发布流程。

## 每轮门禁与依赖

5. 指派和其他首响依赖操作须有 `verified` 或当前适用证据充分的 `reused`，同作者 PR 可使用已核验的自提免首响依据，其余项须提供回复结果。PR 关联必须先于基于该 PR 作者的指派，分别回查。创建前再次 GET，非幂等 POST 结果未知时先回查，不盲目重发。回复失败停止依赖该回复的动作，记录待办并继续其他独立 Issue。用户明确禁止评论时保留等待/受限状态，不虚报已完成。

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

后续首响依赖操作用 `depends_on` 引用已回查的 `verified`/`reused` 回复证据；自提项可使用经核验的免首响依据；`waived_by_user` 仅记录授权限制，不能当作完成首响。评论回查→指派/状态/watch 回查；责任人或提出者再回复沿同样顺序；commit→push→功能分支回查→PR→首次 CI。依赖失败标记 `skipped`，不得改走未预览替代动作。正文文件和结果文件放本轮运行目录并在恢复时复用；日志/报告记录 Issue、operation ID、comment ID、时间、状态和脱敏摘要，不记录 Token、完整敏感正文或维护侧环境。

Handler API attempt 共用目标仓运行缓存的滚动窗口，默认 45 次/60 秒、突发 1；按 API host 和 Token 不可逆摘要隔离，只记录请求时间与 429 cooldown，fetch/classify/写入复用同一缓存。
