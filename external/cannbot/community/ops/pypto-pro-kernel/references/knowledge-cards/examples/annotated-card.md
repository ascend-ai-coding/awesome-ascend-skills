---
type: "Example"
title: "带注释的知识卡片示例"
description: "展示 Agent 默认提交 Active 卡片时，如何保留事实、限制和待确认项。"
status: "stable"
tags: ["pypto-pro", "performance", "knowledge-card", "example"]
---

# 带注释的知识卡片示例

本页演示“事实不足时怎么填”，空白结构见[卡片模板](../CARD_TEMPLATE.md)。它是教学材料，
不是可登记或执行的性能卡片。事实为虚构材料，`sample-01` 只用于演示；真实贡献从
[index.md](../index.md) 分配 ID，并使用自己的技术材料。

## 输入给 Agent 的事实包

- profiler 显示一个 VF 的内层循环反复读取同一份只读数据；具体报告尚未归档；
- 候选方向是把不变量读取移到循环外并在循环内复用；
- 目标 SoC、公开 API、Tile 容量、tail 正确性和性能数据都未知；
- 没有可附的参考资料。

这个事实包足以表达一个候选机制，但不足以证明它可实现、正确或有收益。Agent 不应补写设备、
API 或数字，而应在卡片中如实保留待确认项。

## Agent 生成的待评审卡片

````markdown
---
type: "PyPTO Performance Optimization Card"
title: "在 VF 内将循环不变量读取移出内层循环（教学示例）"
description: "当同一只读数据被 VF 内层循环重复读取时，在容量允许的前提下读取一次并复用。"
status: "stable"
tags: ["pypto-pro", "vec", "example"]
item_id: "sample-01"
bound_hint: "memory"
applicability: "VF 内层循环重复读取同一只读不变量，且外提后的 live set 可被目标容量容纳；当前读取热点及容量材料待补充"
target_api_gate: "目标 SoC 和公开 API 支持均待核验"
---

# 技术卡片 sample-01：在 VF 内将循环不变量读取移出内层循环（教学示例）

## 何时用（诊断特征）

适用于内层循环重复读取同一份只读不变量，且这些读取构成热点的场景。外提后的 live set 须能被
目标容量容纳。当前事实包尚未提供可定位报告，读取热点、设备容量和公开 API 仍待核验。

## 何时不适用

数据会被循环体修改、每次迭代读取范围不同、外提会超出 Tile/寄存器容量，或重复读取不是
热点时不适用。改写需保持 shape、layout、tail 和同步语义等价。

## 原理

候选动作把循环内的重复读取改为一次读取和多次复用，目标是减少重复搬运及其固定开销。它不会
自动减少消费者计算，也不保证端到端性能提升。

## 怎么改（before / after）

以下是**能力门控伪码**，只表达数据流；公开 PyPTO-Pro API、设备支持和容量尚未核验，禁止直接
复制实现。

```text
before: for each iteration: read invariant -> consume
after:  read invariant once -> for each iteration: consume cached value
```

## 性能与验证指标

关注生成物或 profiler 中的重复读取及 spill 变化，并核对设备容量与 API 支持。正确性需覆盖正常
shape、边界 shape、tail、dtype/layout 和同步相关用例；性能材料注明 baseline/candidate 的测试
条件。当前没有可引用数据，不能填写收益结论。

## 技术限制与风险

复用要求值在循环内不变，并保持数据流、同步/并发所有权和 wrapper 合同等价；Tile/寄存器容量
限制了可保留数据的规模。额外占用可能引入 spill，须结合实际材料说明其性能影响。

````

## 示例要点

- 它保留了事实包中的主要机制与限制，没有把“怀疑重复读取”改写成已证实结论；
- 未知的设备、API、容量和性能仍明确未知，没有生成貌似可信的值；
- 按默认 Active 状态提交，代码仍明确标为能力门控伪码，采用前核验所需能力；
- 它给出了下一步补证方法，供专业人员在合入前 review。

真实贡献可复制[卡片模板](../CARD_TEMPLATE.md)，字段用法见[OKF 格式约定](../PROFILE.md)。
