---
schema_version: okf.v1
kind: guide
type: programming_guide
source_family: curated
title: 历史 Issue 经验摄入规则
description: 首次全量、日常增量并周期校准运行时 Issue 证据，再人工复核并提升为少量 runbook 或 field note。
tags: [history_ingestion, issue_corpus, curation, provenance]
resource: https://gitcode.com/cann/ops-math/issues
sources:
  - role: primary
    url: https://gitcode.com/cann/ops-math/issues
status: verified
confidence: verified
created_at: '2026-08-03T00:00:00Z'
updated_at: '2026-08-13T00:00:00Z'
---

# 历史 Issue 经验摄入规则

运行 `scripts/refresh_issue_knowledge.py` 管理历史 corpus：首次全量 bootstrap，日常按
`updated_at` 游标增量刷新，并周期全量校准删除、状态变化和评论编辑。排除提出者与负责人
相同的自提 Issue，并读取评论与 PR 描述中的关联。生成器只输出显式信号：关联变更、评论
声称修复、答疑、索要信息、转交、无文本证据等。

原始语料默认写入 `.cannbot/gitcode-issue-handler/data/issue-history.json`，统计报告默认
写入 `.cannbot/gitcode-issue-handler/reports/knowledge-corpus.md`，两者都不提交。显式
`--output` / `--report` 可覆盖单次路径。提升知识时：

1. 按类型和证据强度选候选，不按标题热词直接归因；
2. 打开 Issue、评论和关联 PR 复核实际处理过程；
3. 跨多个案例成立的规律合并到 runbook；
4. 只有单案例但调查链完整时写 field note；
5. 写明适用边界、失败路径和不可推断项。

每轮增量只更新运行时 corpus，不自动新增或修改知识卡。需要把候选提升为受审卡时，人工
复核公开来源、同步逐层 index 并走代码评审。无公开证据的关闭 Issue 只保留统计，不提升为
“成功修复案例”。

<!-- okf:related:start -->

# 相关

- [证据优先的 Issue 初判](../triage/evidence_first_triage.md) — 历史信号如何用于当前 Issue。
- [Issue 代码变更与评论路径判定](../../reference/issue_handling/mode_routing.md) — 历史处理结果的路由语义。
- [ops-math 非自提 Issue 处理模式](../field_notes/ops_math/non_self_issue_patterns_2026_08_03.md) — 本规则的首个全量快照实践。

<!-- okf:related:end -->
