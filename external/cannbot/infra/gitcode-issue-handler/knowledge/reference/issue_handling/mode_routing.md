---
schema_version: okf.v1
kind: guide
type: programming_guide
source_family: curated
title: Issue 代码变更与评论路径判定
description: 先读 Issue 的显式诉求与证据，再选择 code_change、comment_explain、need_more_info 或 out_of_scope。
tags: [issue_triage, route, code_change, comment]
resource: https://gitcode.com/cann/cannbot-skills/blob/9e349d56a7af729be6f2c66ba8b74b295c77ecbf/infra/gitcode-issue-handler/references/mode-detection.md
sources:
  - role: primary
    url: https://gitcode.com/cann/cannbot-skills/blob/9e349d56a7af729be6f2c66ba8b74b295c77ecbf/infra/gitcode-issue-handler/references/mode-detection.md
status: verified
confidence: verified
created_at: '2026-08-03T00:00:00Z'
updated_at: '2026-08-03T00:00:00Z'
---

# Issue 代码变更与评论路径判定

## 决策顺序

1. 先尊重用户显式边界，例如“只回复”“不改代码”或“修复并提 PR”。
2. 再从 Issue 的标题、正文、标签和评论提取事实：现象、期望、错误日志、复现、明确诉求。
3. 根据信号选择路径，不用是否提供 fork、是否关闭或标题关键词代替内容判断。

## 路由

| 路径 | 必要信号 | 下一步 |
|---|---|---|
| `code_change` | 可验证缺陷、明确功能诉求或文档/代码事实错误 | 先复现或核实事实，再确认根因和最小修改 |
| `comment_explain` | 用法、支持范围、设计原因或预期行为咨询 | 读代码/文档/历史后给有定位的答复 |
| `need_more_info` | 缺少判断所需的版本、环境、输入、日志或复现 | 只索要最小缺失信息，不猜根因 |
| `out_of_scope` | 证据指向其他仓库、环境、服务或责任域 | 给出可验证理由和下一步去向 |

模式可以在调查中切换：评论路径发现真实缺陷时转 `code_change`；修复调查证明是误用时转 `comment_explain`。切换时保留证据，不强行完成原路径。

## 禁止推断

- Issue 已关闭 ≠ 已修复。
- 有负责人 ≠ 已处理完成。
- 有评论 ≠ 已给出有效答复。
- 相似历史案例 ≠ 当前根因相同。

<!-- okf:related:start -->

# 相关

- [Issue 评论答复的证据要求](comment_evidence.md) — Comment 路径的证据门槛。
- [证据优先的 Issue 初判](../../runbooks/triage/evidence_first_triage.md) — 路由前的证据分层。
- [历史 Issue 经验摄入规则](../../runbooks/curation/historical_issue_ingestion.md) — 相似案例如何进入知识库。
- [ops-math 非自提 Issue 处理模式](../../runbooks/field_notes/ops_math/non_self_issue_patterns_2026_08_03.md) — 路由切换和弱信号反例。

<!-- okf:related:end -->
