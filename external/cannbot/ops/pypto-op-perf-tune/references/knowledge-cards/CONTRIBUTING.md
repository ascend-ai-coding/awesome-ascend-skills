---
type: "Reference"
title: "贡献 PyPTO 性能知识卡片"
description: "用事实包和 Agent 整理知识卡片，交由专业人员评审的快速入口。"
status: "stable"
tags: ["pypto", "performance", "knowledge-card", "contribution"]
---

# 贡献知识卡片

知识卡片记录 PyPTO 性能优化方法及其适用条件。Agent 负责整理结构、核对引用和暴露缺口；
贡献者负责技术事实与证据，专业人员在合入前 review。Agent 默认以 Active 提交新卡。

## 1. 准备事实包

先给 Agent 尽可能多的原始材料；不知道的内容明确写“未知”或“待验证”，不要让 Agent 补猜：

- **问题与诊断**：当前代码/生成物/profiler 中可直接核对的现象，以及它为什么值得优化；
- **优化方法**：before、after，各步骤的关系，以及减少或重排了什么工作；
- **适用边界**：何时适用、何时不适用，以及容量、dtype、shape、layout、tail、同步等约束；
- **目标能力**：方法依赖的设备、公开 API 与已知限制；
- **验证方法与已有材料**：正确性覆盖、性能指标、测试条件和机制证据；已有用例、数据或 trace 一并提供，未实测如实说明；
- **参考资料（可选）**：有帮助的文档链接、代码路径或实验材料。

只有口头经验、旧版本数字或未定位的代码片段也可以投稿；Agent 应保留其不确定性，供评审核对。

可直接复制下面的事实包，按已有材料填写；没有的项目写“未知”：

```text
问题与诊断：
候选优化方法：
before：
after：
适用条件：
不适用条件与失败案例：
适用设备、公开 API 与已知限制：
正确性、性能和机制验证材料：
参考资料（可选）：
仍然未知的内容：
```

## 2. 让 Agent 生成卡片

把事实包附在下面提示词之后交给仓库内 Agent。若只需要草稿，把“直接修改仓库”改为“仅输出
候选内容”。

```text
请把下方事实包整理成一张 PyPTO 性能优化知识卡片，并直接修改仓库。

开始前完整阅读：
1. cannbot-skills/ops/pypto-op-perf-tune/references/knowledge-cards/CONTRIBUTING.md — 贡献流程与事实包要求。
2. cannbot-skills/ops/pypto-op-perf-tune/references/knowledge-cards/PROFILE.md — 字段、来源与状态规则。
3. cannbot-skills/ops/pypto-op-perf-tune/references/knowledge-cards/CARD_TEMPLATE.md — 新卡结构与默认字段。
4. cannbot-skills/ops/pypto-op-perf-tune/references/knowledge-cards/index.md — 已有卡片清单、分类编号与候选登记。

要求：
- 建议围绕清晰的优化主题组织内容，说明相关步骤和分支。
- 按 index.md 的分类与编号规则确定目录和候选 ID；新卡默认登记到 Active 表。
- 默认使用 status=stable，登记到 Active 表；保持标题、状态和适用边界一致。
- 不臆造 API、版本、数值、来源、评审记录或适用范围。缺失内容明确标为待验证，并说明补证办法。
- 事实与推断分开写；外部或历史数据注明未在当前算子复现，不能承诺当前收益。
- API 未核验时只写“能力门控伪码”，写清缺口与补证方法，不得伪装成可运行代码。
- 参考资料按需附实际读取过的链接或代码路径；没有时省略，不补造来源。
- 完整保留事实包中的有效语义、限制条件和失败案例。
- 先看索引；与已有方法重复时说明重合点和建议补充内容，否则仅新增卡片和 Active 索引行。
- 按卡片模板自查，合入前交专业人员 review。

回复中给出：候选 ID/文件名、变更摘要、未解决的证据缺口和待 review 要点。

事实包：
（粘贴问题、before/after、适用条件、验证材料和可选参考资料）
```

Agent 输出待评审的 Active 卡片。事实包不足时，[带注释示例](examples/annotated-card.md)展示了如何
保留语义并显式暴露缺口。

## 3. 提交前检查

1. 按[卡片模板](CARD_TEMPLATE.md)整理内容；字段用法见[OKF 格式约定](PROFILE.md)。
2. 人工核对机制、适用条件、改法和验证方法是否清楚，代码等级是否准确，未知项是否如实保留。
3. 确认类别目录、文件名、`item_id` 和 Active 索引行一致；提交前重新读取 index，避免并行
   贡献占用了同一编号。

## 4. 提交与人工评审

将卡片和索引变更一并提交 PR。专业人员在合入前 review 机制、适用边界、来源、能力缺口和验证方法，
贡献者按评审意见修改。

Active 仅表示调优候选。实际应用时的适用性判断、实验和结果记录按[本 Skill](../../SKILL.md)执行。
