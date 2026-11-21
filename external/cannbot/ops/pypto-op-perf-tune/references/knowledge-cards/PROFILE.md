---
type: "Reference"
title: "PyPTO 知识卡片格式约定"
description: "使用 OKF v0.2 组织卡片、来源和状态；内容质量由技术人员 review。"
status: "stable"
tags: ["pypto", "performance", "knowledge-card", "okf"]
sources: [{"id": "okf-spec", "resource": "https://github.com/GoogleCloudPlatform/open-knowledge-format/blob/main/SPEC.md", "title": "Open Knowledge Format v0.2 specification"}]
---

# PyPTO 知识卡片格式约定

本目录采用 OKF v0.2 的 Markdown 与 YAML 表达方式，组织方法、适用条件和验证方式。[^okf-spec]
技术内容由合入前 review 把关，实际效果由当前算子的实验验证。

## 1. 文件组织

- 包根使用 OKF 保留名 [index.md](index.md)，声明 `okf_version: "0.2"`。
- 其余 Markdown 使用 YAML frontmatter 和非空 `type`；卡片类型为
  `PyPTO Performance Optimization Card`，指南、模板和示例使用各自类型。
- frontmatter 使用合法 YAML，可参考[模板](CARD_TEMPLATE.md)中便于复制的单行、flow 写法。
- 卡片按类别放入子目录，目录与稳定 ID 遵循 [index.md](index.md) 的约定。
- [index.md](index.md) 是候选清单的唯一入口。模板、指南、示例和未登记文件不会因存在于目录中而被采用。

## 2. 卡片字段

沿用模板中的字段，帮助人和 Agent 快速理解卡片；未知内容如实说明，不为填满字段而补猜。

| 字段 | 用途 |
|---|---|
| `type`、`title`、`description`、`tags` | 类型、标题、摘要和分类 |
| `status` | 卡片的唯一状态字段，索引分组按下表显示 |
| `sources` | 可选参考资料；填写时提供 `resource`，需逐项引用时配 `id` 和同名脚注 |
| `item_id` | 稳定编号，如 `vec-01`，对应文件名前缀与索引 |
| `bound_hint` | 预期主要影响的计算、访存、标量或调度瓶颈 |
| `applicability` | 方法的使用场景和技术条件 |
| `target_api_gate` | 方法依赖的设备、公开 API 能力及已知限制 |

参考资料按需附普通链接或代码路径。未实测、外部历史数据和不确定内容如实标注，
不能写成当前算子的验证结论。

## 3. 卡片状态与人工 review

| 索引分组 | `status` | 含义 |
|---|---|---|
| Draft | `draft` | 暂存的草稿，不进入调优候选 |
| Active | `stable` | 调优候选，新卡的默认状态 |
| Retired | `deprecated` | 不再采用，保留 ID、历史和原因 |

Agent 默认以 Active 提交卡片并同步索引，技术人员在合入前 review 机制、来源、适用边界、示例和
验证方法。评审与实验结论按实际情况记录。

## 4. 使用边界

Active 仅表示调优候选，不代表当前算子一定适用、示例可直接运行或已有收益。
实际应用时的适用性判断、实验和结果记录按[本 Skill](../../SKILL.md)执行。

[^okf-spec]: [Open Knowledge Format v0.2 specification](https://github.com/GoogleCloudPlatform/open-knowledge-format/blob/main/SPEC.md)。
