---
schema_version: okf.v1
kind: guide
type: programming_guide
source_family: curated
title: 证据优先的 Issue 初判
description: 在根因假设前组合 Issue 原文、评论、历史知识和当前代码，并把强弱证据分层。
tags: [issue_triage, evidence, knowledge_query, hypothesis]
resource: https://gitcode.com/cann/cannbot-skills/blob/9e349d56a7af729be6f2c66ba8b74b295c77ecbf/infra/gitcode-issue-handler/SKILL.md
sources:
  - role: primary
    url: https://gitcode.com/cann/cannbot-skills/blob/9e349d56a7af729be6f2c66ba8b74b295c77ecbf/infra/gitcode-issue-handler/SKILL.md
status: verified
confidence: verified
created_at: '2026-08-03T00:00:00Z'
updated_at: '2026-08-03T00:00:00Z'
---

# 证据优先的 Issue 初判

## 最小证据包

初判前至少记录：

- 期望行为与实际行为；
- 原始错误、日志、命令或输入；
- 版本、平台和涉及模块；
- 评论中的补充、维护者结论和 PR 链接；
- 当前代码或文档是否仍存在所述行为。

随后用 `scripts/knowledge_query.py preflight` 检索稳定规则和相似案例。历史案例的作用是提出检查项，例如“先核对文档路径是否真实存在”或“先确认平台支持矩阵”，不是直接复制历史结论。

## 证据等级

- 强：当前版本稳定复现、当前代码/文档、关联 PR 的实际 diff。
- 中：维护者明确评论、固定 commit 历史、两个独立材料互相支撑。
- 弱：标题关键词、关闭状态、单个相似案例、没有上下文的结论句。

根因和修改方案必须由强证据支撑；中证据可形成待验证假设；弱证据只用于安排下一步调查。

<!-- okf:related:start -->

# 相关

- [Issue 代码变更与评论路径判定](../../reference/issue_handling/mode_routing.md) — 根据证据选择处理路径。
- [修改前先复现并确认根因](../diagnosis/reproduce_before_change.md) — 把初判假设升级为代码级结论。
- [历史 Issue 经验摄入规则](../curation/historical_issue_ingestion.md) — 历史案例的证据边界。
- [ops-math 非自提 Issue 处理模式](../field_notes/ops_math/non_self_issue_patterns_2026_08_03.md) — 可转化为检查项的代表案例。

<!-- okf:related:end -->
