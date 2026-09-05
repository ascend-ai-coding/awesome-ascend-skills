# Issue 流程：等待与再次回复

## 读取时机

步骤 2d 将 Issue 判定为等待提出者、等待责任人，或步骤 1 检测到任一方再次回复时，完整
读取本文件。普通答疑、当前处理人继续排查和没有等待对象的 Issue 不需要读取。

## 进入等待

只有评论正文明确提出可行动的补充请求、且下一步确实依赖提出者时，才能进入
`awaiting_reporter`。单纯“已受理”或处理人自己“正在排查”仍由当前处理人行动，保持
`进行中`，不得用`挂起`隐藏首响后的处理时钟。只有 assignee 已回查成功，且评论明确表示
已联系、转交或正在等待该责任人处理时，才能进入 `awaiting_assignee`。

为 `need_more_info` 建立两个相互依赖的外部 operation：完整评论和自定义状态切换
`issue_state_change: <当前状态> -> 挂起`。统一预览批准后严格按以下顺序执行：

1. POST 评论并 GET 回查，取得真实 comment ID 和 `created_at`；失败时停止，不改状态、不写 watch。
2. 先 dry-run 核对动态状态目录，再携带该 operation 的 `authorization_evidence` 执行并回查：

   ```bash
   python3 "$ISSUE_HANDLER_SKILL_ROOT/scripts/followup_state.py" transition \
     --issue "<Issue URL>" --status-name "挂起"
   python3 "$ISSUE_HANDLER_SKILL_ROOT/scripts/followup_state.py" transition \
     --issue "<Issue URL>" --status-name "挂起" --apply \
     --authorization-evidence "<已批准 operation 的证据引用>"
   ```

   脚本按状态名称实时查询仓库目录，不得把当前观察到的 ID 写死。PUT 成功但 GET 回查不一致
   视为失败，记录 `issue_state_change_failed`，不写 watch。
3. 评论与`挂起`均回查成功后，写入提出者、维护者 comment ID/时间和等待起点：

   ```bash
   python3 "$ISSUE_HANDLER_SKILL_ROOT/scripts/followup_state.py" \
     --state-file "<follow_up.state_file>" watch \
     --repo "<owner/repo>" --issue "<iid>" --reporter "<login>" \
     --issue-url "<Issue URL>" --maintainer-comment-id "<comment-id>" \
     --maintainer-comment-at "<created_at>" --waiting-on reporter
   ```

责任人转交按“有效首响/摘要评论回查 → `/assign` 与 assignee 回查 → `挂起`迁移回查 →
assignee watch 落盘”执行，任一步失败都停止其依赖操作。watch 命令为：

```bash
python3 "$ISSUE_HANDLER_SKILL_ROOT/scripts/followup_state.py" \
  --state-file "<follow_up.state_file>" watch \
  --repo "<owner/repo>" --issue "<iid>" --reporter "<reporter-login>" \
  --assignee "<owner-login>" --waiting-on assignee \
  --issue-url "<Issue URL>" --maintainer-comment-id "<handoff-comment-id>" \
  --maintainer-comment-at "<created_at>"
```

## 历史补录

只有当前存在非提出者 assignee、没有解决证据，且最新有效评论明确表达“已联系/转交/等待
责任人处理”时，才分类为 `awaiting_assignee_setup`。把状态迁移和 watch 纳入统一预览，
不重复发布已有首响。普通受理、答疑结论或“正在排查”不能仅因已有 assignee 就推断为等待
责任人。

## 再次回复

`reporter_followup` 的基线必须是维护侧实质回复。纯 `/assign`、系统消息和仅 `@owner`
不建立基线；GitCode 未提供明确系统字段时，只排除“已知平台机器人身份 + 已识别固定模板”
同时命中的评论，避免把普通用户或维护者的相似文案误过滤。

检测到 `reporter_followup` 时，把恢复`进行中`加入统一预览；若核心 `state=closed`，同一
操作还必须声明 `--reopen`。获批后执行
`transition --status-name "进行中" [--reopen] --apply` 并回查，再优先处理提出者的新评论。
不要仅因自定义状态已经是`已完成`而忽略新评论。

检测到 `assignee_followup` 时，同样把恢复`进行中`加入统一预览并回查，然后判断责任人的
新评论属于解决结论、继续处理、再次转交还是索要提出者信息；不得因“责任人已回复”直接
标记 resolved。等待责任人的 watch 没有静默自动关闭期限。

旧 watch 不能在“发现回复”时立即删除：只有新的维护侧响应成功回查并形成下一状态后才
处理。若新响应再次索要信息，用新 comment ID/时间覆盖为 reporter watch 并保持`挂起`；
若再次转交，用新的责任人和基线覆盖 assignee watch；若已答复、转入代码处理或到达真实
终态，执行
`followup_state.py --state-file <follow_up.state_file> resolve --repo ... --issue ...`。
状态恢复或评论失败时保留 watch，供下轮定点刷新和重试。

## 证据与授权

答疑或索要上下文前，用更精确的问题再次执行知识预检，优先读取 `comment_evidence`。
知识卡不能替代当前代码和版本证据。需要历史行为佐证时，按需读取
`code-git-history.md`。

所有评论遵循本 Skill 的 [issue-comment-workflow.md](issue-comment-workflow.md) 和
[authorization-contract.md](authorization-contract.md)。调用方名称和初始处理请求不构成
授权：按 `delivery-confirmation.md` 逐条展示目标与完整正文，一次确认后为对应 operation
IDs 派生证据；POST 后 GET 回查不重复询问。用户明确要求不评论时从预览中删除评论操作并
缩小授权范围。

## 输出

更新 `conversation_state`、`waiting_on`、双方最近 comment ID、等待起点、follow-up SLA、
状态迁移结果和 watch 证据。进入真实终态、代码处理或新的等待状态后再处理旧 watch。
