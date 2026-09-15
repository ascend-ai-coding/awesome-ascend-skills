---
schema_version: okf.v1
kind: guide
type: programming_guide
source_family: curated
title: Issue 评论答复的证据要求
description: 评论路径要用代码、文档、版本历史或明确的未知边界支撑结论，并在发布前回查正文。
tags: [issue_comment, evidence, git_history, boundary]
resource: https://gitcode.com/cann/cannbot-skills/blob/9e349d56a7af729be6f2c66ba8b74b295c77ecbf/infra/gitcode-issue-handler/references/comment-path.md
sources:
  - role: primary
    url: https://gitcode.com/cann/cannbot-skills/blob/9e349d56a7af729be6f2c66ba8b74b295c77ecbf/infra/gitcode-issue-handler/references/comment-path.md
status: verified
confidence: verified
created_at: '2026-08-03T00:00:00Z'
updated_at: '2026-08-03T00:00:00Z'
---

# Issue 评论答复的证据要求

评论答疑不是跳过调查。先定位 Issue 涉及的代码、文档或接口，再按提问层次组织证据：

- “为什么这样实现”：结合 `git log`、`git blame`、`git show` 说明演进和取舍。
- “是否支持”：核实 dtype、shape、format、SoC、版本等边界。
- “如何使用”：给最小可执行示例和必要前置，不复制整段 README。
- 现有材料没有规约：明确写未知范围和建议确认对象，不编造结论。

答复至少包含一句话结论、可定位依据、适用边界和下一步。若调查发现需要改代码，停止起草 patch 式评论，转代码变更路径。

发布评论属于外部写操作，应先展示正文、获得授权，再 POST；提交后 GET 回查内容，不能只依据 HTTP 成功码认定写入完整。

<!-- okf:related:start -->

# 相关

- [Issue 代码变更与评论路径判定](mode_routing.md) — 何时进入或退出 Comment 路径。
- [修改前先复现并确认根因](../../runbooks/diagnosis/reproduce_before_change.md) — 调查发现缺陷后的代码路径门禁。

<!-- okf:related:end -->
