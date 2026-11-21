---
okf_version: "0.2"
---

# PyPTO 性能优化知识卡片库

本目录是一个采用 [Open Knowledge Format（OKF）v0.2](https://github.com/GoogleCloudPlatform/open-knowledge-format/blob/main/SPEC.md)
组织的 PyPTO（非 pypto_pro 前端）性能优化方法库，供本 Skill 调优时查阅和实验。
卡片记录优化方法、适用条件、改法、技术限制与验证方式，内容质量在合入前由技术人员把关。

本文件是卡片清单的**唯一入口**，调优时只从 Active 表获取候选，不通过扫描目录自动采用卡片。
实际卡片按稳定类别放入子目录。首批内置 1 张 shape 卡片，位于 `shape/`，编号 `shape-01`，
以 `status=stable` 登记到 Active 表。

贡献流程、卡片字段与模板约定见本目录的 [贡献指南](CONTRIBUTING.md)、
[OKF 格式约定](PROFILE.md)与[卡片模板](CARD_TEMPLATE.md)
（事实包 → Agent 整理 → 专业人员评审，卡片统一面向 PyPTO 前端，即 `pypto.*` API、`@pypto.frontend.jit`）。

## Active items

新卡默认登记到本表，合入前由技术人员 review。调优时按本表列出候选，
由使用者结合卡片描述、当前算子和目标能力判断适用性。

| item_id | 卡片与锚点 | status | bound_hint | applicability | target/api_gate |
|---|---|---|---|---|---|
| `shape-01` | [整块分支重设动态 valid_shape 为静态 shape 常量](shape/shape-01-full-tile-static-validshape.md) | `stable` | `mixed` | varlen/动态边界切出的视图 shape 为常量而 valid_shape 是数据依赖标量，流入以 concat/cat 等对动态属性敏感的操作为主的公共计算体，且存在可由循环结构静态证明 valid==shape 的分支（如非 loop-end 整块分支） | pypto（非 pypto_pro）前端公开 `pypto.view` 的 shape/offsets/valid_shape 参数；设备支持以算子 README 为准；具体受限 pass 未指明，采用前须在当前工具链复现动态 valid_shape 退化并复核 |

## ID 与状态规则

- 按方法主题选择类别，优先沿用已有分类；新增类别使用简短英文名（如 `shape`）。
- `item_id` 使用 `<类别>-<两位递增序号>`，文件名使用 `<item_id>-<短名>.md`，放入类别目录。
  编号从已登记条目中该类别的最大编号递增，新类别从 `01` 开始；已登记 ID 保持不变且不复用。
- 修改卡片时同步索引中的 ID、状态、适用条件和 API 限制，方便查阅和 review。
- 索引按 `status` 分组：`stable` 为 Active，`draft` 为 Draft，`deprecated` 为 Retired；
  有条目时列出对应分组。退役卡保留稳定 ID、历史链接和退役原因，编号不复用。
- 调优时按现有账本记录卡片 ID、文件位置和本轮结论；Draft、Retired 不作为候选。

字段、来源和状态说明见 [OKF 格式约定](PROFILE.md)。

## 贡献入口

第一次贡献先按[贡献指南](CONTRIBUTING.md)准备事实包，
并使用其中的提示词让 Agent 生成卡片，默认以 Active 提交。采用卡片调优的执行流程见
[本 Skill](../../SKILL.md)。

## Bundle resources

- [贡献指南](CONTRIBUTING.md) - 事实包要求与 Agent 生成卡片的流程。
- [OKF 格式约定](PROFILE.md) - 字段、来源与状态规则。
- [卡片模板](CARD_TEMPLATE.md) - 新卡结构与默认字段。
