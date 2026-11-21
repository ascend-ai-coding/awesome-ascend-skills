---
type: "PyPTO Performance Optimization Card Template"
title: "PyPTO-Pro 性能优化知识卡片模板"
description: "整理 PyPTO-Pro 性能优化方法、适用条件和验证方式的模板。"
status: "stable"
tags: ["pypto-pro", "performance", "knowledge-card", "template"]
---

# 使用说明

1. 第一次贡献先阅读[贡献指南](CONTRIBUTING.md)，准备事实包并用其中的提示词让 Agent 生成
   卡片；字段用法见[OKF 格式约定](PROFILE.md)。
2. 按 [index.md](index.md) 的分类与编号规则，创建
   `<类别>/<item_id>-<短名>.md`。
3. 将下面的“卡片内容”复制到新文件。frontmatter 使用 YAML，可保留示例的 flow 写法或展开为多行。
   默认使用下列 `status=stable`，登记到 [index.md](index.md) 的 Active 表。
4. 将卡片和索引变更一并提交 PR，合入前由专业人员 review。

标题、适用条件和改法描述可复用的计算流程与优化机制；算子专属变量、固定参数和实测数字
放在明确标注的验证案例或来源记录中。参数化示意需写清成立条件，不将实例选择推广为固定最优值。

本文件顶部 frontmatter 描述模板文档自身，不属于新卡片；只复制下方 fenced block 的内容：

```markdown
---
type: "PyPTO Performance Optimization Card"
title: "<优化方法名称>"
description: "<在什么条件下，把什么结构改成什么结构>"
status: "stable"
tags: ["pypto-pro", "<方法主题>"]
item_id: "<类别>-<两位递增序号>"
bound_hint: "<MTE2|MTE3|MTE1|FIXPIPE|SCALAR|MAC|VEC|L2Cache|iCache|PIPELINE>"
applicability: "<方法适用的场景、数据特征和技术条件>"
target_api_gate: "<所需设备、公开 API 及已知限制>"
---

# 技术卡片 <item_id>：<优化方法名称>

## 何时用（诊断特征）

列出可从当前源码、生成物或 profiler 直接核对的特征；ratio 和历史经验只能作为线索。

## 何时不适用

列出方法不成立的技术条件和容易误判的相邻情形。

## 原理

先用短句说明性能根因，再说明改写如何改变依赖或数据流。首次出现时解释专业术语。
解释改动减少或重排的工作、搬运、依赖、冲突或固定开销；涉及多个步骤或分支时，说明它们的关系。

## 怎么改（before / after）

用最小 before/after 说明主要改动；API 已核验时优先给出 PyPTO-Pro 片段。代码 fence 标为以下一种：

- `结构示意`：只展示数据流，明确缺少哪些上下文且不可直接编译或交付；
- `嵌入片段`：使用已核验的公开 `pl.*` / `vf.*` API，并说明嵌入的 kernel/VF 上下文、
  dtype、shape、tail 等必要前提；
- `能力门控伪码`：API 尚未确认或存在已知缺口，禁止直接复制实现；写清缺口和补证方法，
  不因卡片 Active 而视为可用。核心动作与可选增强分别标明能力边界。

不得臆造 PyPTO-Pro API，也不得把生成 C++ 或 AscendC 写法伪装成 Python API。

## 性能与验证指标

写明预期变化的指标、正确性覆盖及已有性能和机制证据。外部或历史数字注明“未在当前算子
复现”；未实测时不填写收益结论。

## 技术限制与风险

说明本方法涉及的精度、dtype/layout、容量、tail、同步等技术限制，以及额外开销或实现风险。

## 参考资料（可选）

附有帮助的 API 文档、代码路径或实验资料。
没有资料时删除本节；需要逐项归因时按 PROFILE 的 sources 与脚注写法填写。
```

## 可选的 kernel 知识关联

与 kernel skill 的现有契约或文档有关联时，按 [接入协议](KERNEL_INTEGRATION.md)向卡片 frontmatter 添加 `knowledge_links`。先核对真实目标，再选择关系；不相关的卡片省略整个字段。下例表达技术契约引用，实际 `scope`、`boundary` 必须按本卡填写。

```yaml
knowledge_links:
  schema: "pypto-knowledge-links/1"
  relations:
    - kind: uses_contract
      target:
        skill: pypto-pro-kernel
        id: "contract:ub-l1"
      scope: "本卡改法需要核对的 UB→L1 物理表示、有效范围和消费同步。"
      boundary: "关联不证明本卡的实际布局和工具链已经验证，也不把整个目标章节视为同一优化。"
```
