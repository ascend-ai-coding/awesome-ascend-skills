---
schema_version: okf.v1
kind: guide
type: programming_guide
source_family: curated
title: 修改前先复现并确认根因
description: 对代码缺陷先稳定复现，再用代码、对照实验和 Git 历史确认引入点，避免症状式修补。
tags: [reproduction, root_cause, git_pickaxe, code_change]
resource: https://gitcode.com/cann/cannbot-skills/blob/9e349d56a7af729be6f2c66ba8b74b295c77ecbf/infra/gitcode-issue-handler/SKILL.md
sources:
  - role: primary
    url: https://gitcode.com/cann/cannbot-skills/blob/9e349d56a7af729be6f2c66ba8b74b295c77ecbf/infra/gitcode-issue-handler/SKILL.md
status: verified
confidence: verified
created_at: '2026-08-03T00:00:00Z'
updated_at: '2026-08-03T00:00:00Z'
---

# 修改前先复现并确认根因

按现有测试、最小脚本、手动路径的顺序复现，记录命令、输入、期望、实际、关键日志和稳定性。无法稳定复现时不改源代码：先排除预期行为，再索要最小缺失上下文。

稳定复现后，从首个仓内堆栈帧、边界输入、日志打点和对照分支定位。用户指出具体标识符时，用 `git log --all -S` 粗筛引入历史，再逐个 `git show -- <path>` 验证；merge、squash、孤儿提交和整体重写会造成 pickaxe 假阳性。

进入修改前必须能回答：精确根因、修改位置、最小策略、兼容风险和验证方式。禁止吞异常、放宽阈值或修改测试断言来掩盖问题。

<!-- okf:related:start -->

# 相关

- [证据优先的 Issue 初判](../triage/evidence_first_triage.md) — 复现前的假设与证据分层。
- [Issue 评论答复的证据要求](../../reference/issue_handling/comment_evidence.md) — 证明是预期行为后的答复路径。

<!-- okf:related:end -->
