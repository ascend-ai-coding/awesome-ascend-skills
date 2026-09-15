---
schema_version: okf.v1
kind: guide
type: programming_guide
source_family: community
title: Issue 挂起原因与再次回复跟踪
description: 对等待提出者或责任人的未解决 Issue 建立分类 watch，并让任一方的新回复重新进入处理。
tags: [issue_followup, awaiting_reporter, awaiting_assignee, reporter_reply, issue_state]
resource: https://gitcode.com/cann/ops-math/issues/2617
sources:
  - role: primary
    url: https://gitcode.com/cann/ops-math/issues/2617
  - role: example
    url: https://gitcode.com/cann/ops-math/issues/2535
status: verified
confidence: verified
created_at: '2026-08-13T00:00:00Z'
updated_at: '2026-08-14T00:00:00Z'
---

# Issue 挂起原因与再次回复跟踪

## 可观察事实

一个代表案例中，维护侧答复后 Issue 已进入完成状态，提出者数日后继续评论。另一个案例
已有有效首响和 assignee，但回复只是“已联系算子责任人，请稍等”，尚无问题结论。前者在
等待提出者，后者在等待责任人；两者都未解决，但恢复处理的触发方和自动闭环边界不同。

## 稳定规则

1. 维护侧明确索要继续定位所需的信息时，评论回查成功后把自定义状态切为`挂起`并写
   `awaiting_reporter` watch。
2. 有效首响后已验证指派责任人、仍无解决证据且下一行动者是该责任人时，把自定义状态切为
   `挂起`并写 `awaiting_assignee` watch。普通受理或当前处理人自己排查不挂起。
3. 日常批量获取同时使用常规 open 范围、全状态更新时间增量和 watchlist 定点刷新；创建
   时间窗口不能过滤后两种来源。
4. 提出者在最新维护侧实质回复后追加评论时重新进入 `need_attention`。已有负责人、PR、
   旧回复、核心 closed 或自定义终态都不能覆盖这一信号。
5. 被等待责任人追加评论时也重新进入 `need_attention`，恢复`进行中`后判断是否解决、继续
   等待或转为等待提出者；不能把任意责任人回复直接当作解决。
6. 提出者再回复且核心 state 已关闭时同时 reopen。状态操作与评论一样需要精确授权和
   写后回查。
7. 旧 watch 只在新的响应成功并形成下一状态后更新或删除，避免中途失败造成漏跟。

## 边界

- 第三方评论不能在不知道身份关系时自动视为维护侧结论。
- 评论证据不完整时保守重试，不执行状态迁移。
- `awaiting_assignee` 没有用户静默自动关闭期限；维护侧仍欠处理结论。
- `挂起`是服务状态，不是解决证据；任一等待和 reopened follow-up 都不能计为已解决。

<!-- okf:related:start -->

# 相关

- [Issue 代码变更与评论路径判定](mode_routing.md) — 再次回复后重新选择处置路径。
- [Issue 评论答复的证据要求](comment_evidence.md) — 对外回复的证据和授权边界。

<!-- okf:related:end -->
