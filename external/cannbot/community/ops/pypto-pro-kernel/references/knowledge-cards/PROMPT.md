---
type: "Prompt"
title: "从调优会话记录提炼知识卡片"
description: "把算子开发与性能调优的会话内容提炼成 PyPTO-Pro 性能知识卡片的参考示例提示词。"
status: "stable"
tags: ["pypto-pro", "performance", "knowledge-card", "contribution", "prompt"]
---

# 从调优会话记录提炼知识卡片

把这段提示词连同会话记录一起交给仓库内 Agent，让它先抽取候选知识点，经确认后再生成卡片。
贡献流程与事实包要求见[贡献指南](CONTRIBUTING.md)，字段与状态规则见[OKF 格式约定](PROFILE.md)。

本提示词是**参考示例**：会话内容、素材齐备程度和目标卡片各不相同，贡献者应按实际情况增删步骤、
调整粒度或改写措辞，不必照搬。

```text
请把本次算子开发与性能调优会话的内容提炼成 PyPTO-Pro 性能知识卡片，并按贡献流程提交 PR。

开始前完整阅读：
1. cannbot-skills/ops/pypto-pro-kernel/references/knowledge-cards/CONTRIBUTING.md — 贡献流程与事实包要求。
2. 同目录 PROFILE.md、CARD_TEMPLATE.md、index.md 与 examples/annotated-card.md。
3. index.md 中与本素材主题相近的 2-3 张既有卡片全文，对齐语气与详略。

抽取候选（只做提取，不做创作）：
- 只提取会话内容直接支持的诊断现象、优化动作、before/after、适用与失效条件和实测数据。
- 区分事实与会话中的推测，推测不得写成结论。
- 一个清晰优化主题对应一个候选；一次会话可产出 0~N 个，宁缺毋滥；
  素材达不到卡片粒度时如实说明并停止。
- 每个候选按 CONTRIBUTING.md 的事实包格式整理；会话未覆盖的字段写“未知/待验证”，不用通用知识补全。

去重判定：
- 对照 index.md 全部分组、../workflows/optimization.md 与 ../../examples/ 中的完整样例。
- 输出判定表：每个候选 → 投稿 / 不投稿（原因：必要材料缺失、与现有卡片重复等）。
- 先给出判定表，经贡献者确认后再继续。

生成卡片：
- 按 CONTRIBUTING.md“让 Agent 生成卡片”一节的要求逐条执行。
- 会话中的性能数字注明测试条件；非当前环境复现的一律标注“未在当前算子复现”。
- 新卡文件名 <类别>/<item_id>-<短名>.md，status=stable，同步登记 index.md Active 表；
  提交前重新读取 index.md，确认编号未被占用。
```
