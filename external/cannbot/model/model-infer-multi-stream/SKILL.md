---
name: external-cannbot-model-model-infer-multi-stream
description: NPU 多流技术知识技能。提供整网模块 / 算子 DAG 拆解、模块间与模块内并行性判断、多流候选编排派生、TorchAir(Ascend
  IR/GE) 与 npugraph_ex/aclgraph 的多流 API 路由、切流 / 同步 / 控核实现，以及假并行（overlap_pct）排查与 Profile
  验证等技术规则。供两类工作引用：形成多流优化候选 plan，以及实施 / review 多流改造。触发场景包括：多流、双流、stream overlap、控核、limit_core_num、整网
  DAG、模块拆解、npu_stream_switch、npu_wait_tensor、TorchAir 多流、record_stream、多流假并行排查。本技能只提供多流技术规则，不做
  Plan / round 编排。
user-invocable: true
original-name: model-infer-multi-stream
synced-from: https://gitcode.com/cann/cannbot-skills
synced-date: '2026-07-24'
synced-commit: bff73845607ac78808e2af3e0014d7eb72094ef3
license: UNKNOWN
---

# NPU 多流技术知识

本技能是多流优化的**技术知识包**，不负责任务编排——它不管 Plan / round / 状态机，也不负责性能数据的采集调度。候选的取舍、轮次推进、Dashboard 记账都交给调用方（例如上层优化编排流程）。本技能只回答一件事：多流这件事**怎么分析、怎么实现、怎么验证**。

它支持两类工作：

- **形成 plan**：给定一个已经 profiling 过的模型，判断哪里能做多流，并为每个并行点提出若干种不同的编排候选。
- **实施 / review**：把某个多流方案落到代码（选执行路径、切流、加同步、控核），并验证它在物理执行层面是不是真的并行了。

## 怎么用这个技能

| 你在做的事 | 先读 | 要点 |
| --- | --- | --- |
| 形成多流候选 plan | 下文"分析"一节 + [`references/module-decomposition.md`](references/module-decomposition.md) + [`examples/README.md`](examples/README.md) + [`references/ge-reorder-design-check.md`](references/ge-reorder-design-check.md) + [`references/plan-detail-fragment.md`](references/plan-detail-fragment.md) | 拆模块 / 算子 → 判并行性 → 每个并行点派生 ≥2 种编排 |
| 实施多流方案 | 下文"实现"一节 + [`references/api-routing.md`](references/api-routing.md) + `resources/` 官方 doc | 先定执行路径，再切流 / 同步 / 控核，始终保留开关和回退路径 |
| review / 调试多流方案 | 下文"调试与验证"一节 + [`references/timeline-overlap-check.md`](references/timeline-overlap-check.md) + [`references/kernel-fields-lookup.md`](references/kernel-fields-lookup.md) + [`references/plan-detail-fragment.md`](references/plan-detail-fragment.md) | 用 overlap_pct 判真假并行，按五类问题分诊 |

被调用方（如优化编排流程）消费时，本技能产出两类内容，结构都由本技能定义、调用方不解析：**方向级分析产物**（module / op DAG + 并行性判断，一个模型一份，被所有候选 Plan 共享，结构见 [`references/module-decomposition.md`](references/module-decomposition.md)）和**每个 Plan 的方案细节**（流分组 / GE 风险 / overlap_pct 等，结构见 [`references/plan-detail-fragment.md`](references/plan-detail-fragment.md)）。

## 核心原则

- **Stream ID 切对 ≠ 物理并行成立（最重要）**：`torchair.scope.npu_stream_switch` 给副流贴的是逻辑标签，GE 编译期会重排算子——尤其会把"消费副流输出的轻量 precompute"（如 Cast / Swish / Sigmoid）拉到主流，导致主流被 barrier 卡住、副流跑的时候主流 idle，物理执行仍是串行。任何"副流已落到 side stream"的结论都**不能**直接当方案成立，必须按 [`references/timeline-overlap-check.md`](references/timeline-overlap-check.md) 用 `Start Time(us)` + `Duration(us)` 算 `overlap_pct`。"Stream ID 分裂但 wall 没降"的现象，**必须先排除假并行**，再去谈资源争抢 / host overhead。
- **先整网后局部**：先回答整网主路径和模块间并行性，再进入模块内算子，不要一上来就抓某个局部热点函数。
- **模块级目标明确**：模块拆解的目的是判断模块与模块之间哪些可以并行；decoder layer 不能当作单个模块，必须再往下拆。
- **每个模块都要下钻**：第二层要对每个候选模块补齐算子清单和算子 DAG，把可行编排沉淀成候选。
- **开关必须可关闭**：多流路径要保留 enable 开关和原始回退路径，方便调试、review 和淘汰时干净隔离。
- **先证明正确，再追性能**：先验证依赖、功能和精度，再看 overlap 和时延收益。
- **overlap 不等于收益**：出现拖尾、资源争抢、host bound、shape 劣化时，overlap 真实成立也可能不降 wall，要继续评估控核和图模式限制。
- **不要混抄案例**：先确定当前模型走哪条执行模式，再选一套主 API 路径，不要把 eager 和 graph 风格混着套。
- **编排不唯一，要多试**：同一个并行点的分流编排往往不止一种，调试时多尝试不同编排，选物理上真并行且 wall 下降最大的那个。
- **官方约束优先看 `resources/`**：先读对应模式的 multi_stream 文档；只有 overlap 已成立但出现资源争抢、拖尾或卡死风险时，再读对应模式的 limit_cores 文档。
- **开发与调试穿插**：改造一处 → 看物理布局 → 再改下一处或回滚，是多轮反复的过程；关注算子 / 模块是否真落在目标流上、cube / vector 的利用率与占核情况。

## 执行路径定界

动手前先确认当前代码走哪条多流路径，两条路径的 API 与约束不能混用。

> **术语辨析**：上游 torchair 文档里「Ascend IR」和「GE 图模式」指同一条路径，只是分别从 IR 侧和执行侧描述——通过 `torchair.CompilerConfig(mode="max-autotune")` 把 PyTorch FX 图转成 Ascend IR，再由 GE（Graph Engine）编译执行。所以「multi_stream（Ascend IR）」文档里写「仅适用于 GE 图模式」并不矛盾。

- `torchair.CompilerConfig()` + `torchair.get_npu_backend()` + GE 图模式：走 **Ascend IR** 路径。先看 [`resources/ascend_ir_multi_stream.md`](resources/ascend_ir_multi_stream.md)，需要控核时再看 [`resources/ascend_ir_limit_cores.md`](resources/ascend_ir_limit_cores.md)。
- `torch.compile(..., backend="npugraph_ex")`：走 **npugraph_ex / aclgraph** 路径。先看 [`resources/npugraph_ex_multi_stream.md`](resources/npugraph_ex_multi_stream.md)，需要控核时再看 [`resources/npugraph_ex_limit_cores.md`](resources/npugraph_ex_limit_cores.md)。
- 不要把 Ascend IR 的图内 API 和 aclgraph 的显式 stream API 混写进同一套实现。
- GE 图模式通常通过 `CompilerConfig` + `get_npu_backend()` 定界，不要把 `torch.compile(mode=...)` 当成当前主入口写法。
- `npugraph_ex` 路径本质是显式 stream / event / 生命周期管理；如果代码已经是这套风格，顺着现有实现继续，不要强改成 TorchAir 图内表达。

> **`npu_stream_switch` 的 with-block 不是硬边界**：`torchair.scope.npu_stream_switch(stream_tag, stream_priority=0, enable_inner_parallel=True)` 默认 `enable_inner_parallel=True`，GE 仍可在 block 内 / 外做调度——可能把 block 内的小算子（Cast / Reshape / Swish 等）挪到主流，也可能把 block 外但被 block 内 tensor 间接依赖的预计算（典型 `silu(z) = Swish(Cast(z))`）拉到主流。**源码层级 ≠ 物理层级**：切流之后必须用 profiling 的 `Stream ID` + `Start Time(us)` + `Duration(us)` 验证物理布局，绝不能凭 Stream ID 切对就认为方案落地。

## 分析：拆解与并行性判断（形成 plan）

详细规则见 [`references/module-decomposition.md`](references/module-decomposition.md)，主线固定为三步：

1. **拆整网模块、画模块级 DAG、判模块间并行**：从整网 decoder layer 的执行路径出发，沿子网络 / 通信 / 状态 / 同步 / 资源边界把每层拆成语义完整、输入输出明确、可独立讨论调度的模块；标注依赖类型（`data` / `state` / `event` / `collective_order`），据此把模块对判成 `serial` / `parallel_candidate` / `parallel_pending_validation`。
2. **对每个候选模块拆算子、画算子 DAG、判模块内并行**：仍沿计算 / 通信 / 同步 / 状态边界拆，识别模块内部哪些算子可并行、哪些依赖边不能打破。
3. **为每个并行点派生候选编排**：把共享同一输入、彼此无 `data` 依赖的模块 / 算子看成一个集合，按集合划分思考分流。**每个并行点至少给 2 种不同编排**——并行对象、流分组、汇合点、tag 粒度或跨阶段 overlap 不同，就是不同的候选方案。

选案例先读 [`examples/README.md`](examples/README.md) 的快速选型表，找到与当前模型结构最接近的模式再派生候选。

**派生维度**（每一种取值变化都是一个独立候选）：算子粒度（是否合并 / 拆分 / 重排）、集合划分（哪些算子同流串行、哪些放另一条流）、汇合点（早汇合还是晚汇合）、流数量（双流还是三流）、tag 粒度（同一副流串行下发还是多个独立副流）、跨阶段 overlap（仅本模块内，还是跨层 / 跨阶段掩盖 dominator）。

**实现强度是一个复杂度阶梯**，不是不同的并行方案：同一个编排可以从 **C1 最小切流** → **C2 切流 + 控核** → **C3 切流 + 控核 + 手动同步点控制** 逐级加码，先用 C1 验证核心编排是否真并行，再决定要不要上更高强度。改变并行对象 / 流分组 / 汇合点则属于另一个候选，不要塞进同一方案的强度递进里。

**设计期必查**：任何一个把 op 切到副流的候选，落地前都要按 [`references/ge-reorder-design-check.md`](references/ge-reorder-design-check.md) 过一遍 GE auto-reorder 风险清单——副流输出有没有轻量 precompute 下游消费者会被 GE 拉回主流形成 barrier。这是假并行最常见的根因，在设计阶段就要预判。

## 实现（实施方案）

动手前先读 [`references/api-routing.md`](references/api-routing.md)，确认当前执行路径对应的官方文档入口、推荐 API、控核约束和已知坑点。

1. **按执行路径选 API**：
   - Ascend IR / GE 图模式：切流用 `torchair.scope.npu_stream_switch(stream_tag, ...)`，时序用 `torchair.scope.npu_wait_tensor`，已有 tagged event 风格时沿用 `torchair.ops.npu_record_tagged_stream` / `torchair.ops.npu_tagged_event_record` / `torchair.ops.npu_tagged_event_wait`（注意：切流 / 时序在 `torchair.scope`，而 record / wait / tagged event 在 `torchair.ops`，是两个不同命名空间）。
   - npugraph_ex / aclgraph：切流用 `torch.npu.Stream()` + `with torch.npu.stream(stream)`，时序用 `torch.npu.Event()` + `Event.record()` + `Event.wait(stream)`，或 Stream 侧的 `record_event` / `wait_event` / `wait_stream`。在 `torch.compile(fullgraph=True, dynamic=True)` 下 dynamo 拦截 Stream / Event 对象（缺 `as_proxy()`）时，改用 `torch.npu.npugraph_ex.scope.npu_stream_switch(tag)` 或 `torch.npu.npugraph_ex.scope.npu_tagged_event_record` / `torch.npu.npugraph_ex.scope.npu_tagged_event_wait` 这套 string-tag / tagged-event 接口（npugraph_ex 路径下切流与 tagged event 都在 `torch.npu.npugraph_ex.scope` 命名空间，与 GE 路径 tagged event 落在 `torchair.ops` 不同）。
2. **显式 stream 路径检查内存生命周期**：短生命周期 tensor 若被其他 stream 继续消费，补 `record_stream()`；模型权重、常驻 cache 这类长生命周期对象一般不需要。
3. **先做最小可验证 overlap**：只改一个明确的并行点，先补同步再扩大并行窗口，始终保留原路径和 enable 开关。
4. **overlap 成立后再评估控核**：出现明显拖尾时，Ascend IR 用算子级 `torchair.scope.limit_core_num`（优先级高于全局 `config.ge_config.aicore_num`），npugraph_ex 用 Stream 级 `torch.npu.npugraph_ex.scope.limit_core_num`。
5. **放大改动前先做一次 warm-run 验证**：对比单流 / 多流，用 profiler 确认真出现了 overlap，而不是逻辑分流但执行仍串行（profile 至少跑 5 次，丢掉前 2-3 次 cold-start 取中位数）。

## 调试与验证（实施后自检 / review）

按下面五类问题分开排查，不要混在一起。

1. **依赖错误 / 同步错误**：重点查事件记录、等待顺序、跨流汇合点、共享状态写入次序；典型现象是读到未完成结果、死等、结果偶发错误。
2. **精度 / 功能异常**：先对比优化前基线，再按 `prefill/decode → 模块 → 算子` 缩小范围，判断是不是多流引入的状态时序问题，而不是算子本身的问题。
3. **性能无收益**——必须按 3a → 3b → 3c 顺序排查，**不允许跳过 3a/3b 直接走资源 / host 假设**：

   - **3a 副流根本没真跑（逻辑切流失败）**：现象是副流 ops 的 `stream_id` 仍落在主流、或全退到 `sid=1`。检查 `npu_stream_switch` 的 with-block 是否在 `is_prefill / enable_multi_stream` 条件下真的走到、`_TORCHAIR_AVAILABLE` 兜底有没有被命中、源码顺序有没有让 GE 把 ops 挪到 block 外。
   - **3b 副流跑了但和主流串行（假并行 / GE auto-reorder barrier）**：现象是 Stream ID 已分裂但候选模块 wall 与 baseline 持平、并行度 ≤ 1.05×。**必须**按 [`references/timeline-overlap-check.md`](references/timeline-overlap-check.md) 重建甘特图算 `overlap_pct`，`≤ 0.05` 即判假并行。典型根因是 GE 把消费副流输出的轻量 precompute（Cast / Swish / Sigmoid / Reshape）拉到主流提前下发，卡住主流 dominator。解法是把 barrier op 显式钉在 with-block 内（自己写出 precompute，不依赖 GE 自动放置），或重新设计 op 集合让副流有真正的 dominator 可吃——这两种都是新的编排候选。
   - **3c 真重叠（overlap_pct ≥ 0.5）但 wall 不变**：仅在 3a/3b 排除后才走这步。看 shape 变化、task 数增加、host bound、带宽争抢、流间资源抢占、拖尾。判"主 / 副是否争 cube"用 `kernel_details.csv` 的 `Block Dim` 加和 vs 设备 `ai_core_cnt`，**不要**用"算子类型是 MatMul / Vector"二元判断（MatMul 也能 blk=1 占很少核）。跨流 tensor 维度偏大时先查 `TransData` / `BroadcastTo` / `MemSet` 是否长尾。
4. **图模式 / runtime 限制**：重点看 graph break、图模式 API 约束、stream 语义差异、运行时不支持；graph 场景按 TorchAir 路径排查，不要回退成 eager 思路硬套。
5. **aclgraph 生命周期错误**：重点查短生命周期 tensor 是否跨流继续使用、是否遗漏 `record_stream()`；典型现象是结果偶发错误、数据踩踏、段错误或异常内存问题。

**现象 → 下一步技术方向**（不是完整规则集，要结合 profiler、`kernel_details.csv`、代码结构和模型语义判断，可基于表外现象提出新方向）：

| 现象 | 技术方向 |
| --- | --- |
| 改造点之外的 TransData / BroadcastTo / MemSet wall 显著上升，或单算子 max duration > 5x | 可能触发 GE 全局重排或算子上提；缩小副流、换集合划分，或先做算子合并让分流单元更独立 |
| 副流已落 side stream，但 layer wall 几乎不变 | 先算 `overlap_pct`：≤5% 是假并行，把 GE 拉过来的 precompute 显式钉进 with-block；0.05~0.5 是部分并行，调 op 集合让副流更长 / 换 dominator；≥0.5 才走资源争抢 / shape / host 假设 |
| 主流出现新空洞 / 拖尾，副流已提前结束 | 同步点编排不对；调整汇合点，或让副流接力到下游非 dominator 算子 |
| 主 / 副流 cube 利用率叠加高于单流，但 wall 没降 | 资源争抢；换资源对偶性更好的集合划分，让一边偏 cube、一边偏 vector / memory |
| 多流后某个原本未关注的 Cast / Reshape / DynamicQuant 位置变化 | 隐含 data 依赖或新瓶颈被暴露；把该算子显式纳入候选集，重做集合划分 |

**选定方案前做 Profile 双重验证**（第 1、2 步都不可选）：

1. **Stream ID 落点检查**：副流算子的 `stream_id` 确实落到目标 stream、没被 GE 挪回主流；不对就回 3a。
2. **overlap_pct 时间线验证**：按 [`references/timeline-overlap-check.md`](references/timeline-overlap-check.md) 算 `overlap_pct`。判方案成立要 `≥ 0.5`；判方案无效要先确认是 `≤ 0.05`（假并行）还是 `≥ 0.5`（真并行但 wall 没降），两者的改进方向完全不同。
3. 控核场景下副流算子 `Block Dim` 不超过配置值，主 / 副 `Block Dim` 加和不超过设备核数。
4. 汇合点后没有新的 `EVENT_WAIT` 空洞或明显 gap。
5. 跨流 tensor 名称和维度与设计一致，没把大 tensor 搬运变成新瓶颈。

**功能 / 同步层面的 review 要点**：enable 前后输出一致、开关关闭后原路径正常；汇合点前后无缺失等待，共享状态 / KVCache / 通信结果没有读写乱序。

## 实施检查清单

- 已确认当前实现走 Ascend IR / GE 还是 npugraph_ex。
- 已确认存在真实可并行分支，而不是把串行链路硬拆成多流。
- 每个并行点已派生 ≥2 种不同编排候选。
- 跨流依赖已显式表达，而不是依赖隐式同步；跨流 tensor 维度尽量 ≤ 数十 element / token，汇合点尽量只 1 个。
- 显式 stream 路径里，短生命周期 tensor 的跨流使用已检查 `record_stream()`。
- overlap 已用 `overlap_pct` 证明真实成立后，才继续评估控核 / stream limit / 预取。
- 已通过 warm-run baseline 和 profiler 确认优化方向成立（丢前 2-3 次 cold-start）。

## 参考文档索引

- **整网模块 / 算子拆解与候选派生（方向级分析产物结构）**：[`references/module-decomposition.md`](references/module-decomposition.md)
- **多流 Plan 方案细节片段（Plan 自由块结构）**：[`references/plan-detail-fragment.md`](references/plan-detail-fragment.md)
- **多流 / 控核 API 路由**：[`references/api-routing.md`](references/api-routing.md)
- **overlap 时间线复盘（判真 / 假并行）**：[`references/timeline-overlap-check.md`](references/timeline-overlap-check.md)
- **kernel_details 字段查法（设计 / post-mortem 两套字段集）**：[`references/kernel-fields-lookup.md`](references/kernel-fields-lookup.md)
- **GE auto-reorder 风险设计清单**：[`references/ge-reorder-design-check.md`](references/ge-reorder-design-check.md)
- **案例库与选型表**：[`examples/README.md`](examples/README.md)
- **官方多流 / 控核文档（torchair docs/zh 离线副本）**：
  - Ascend IR / GE 图模式：[`resources/ascend_ir_multi_stream.md`](resources/ascend_ir_multi_stream.md)、[`resources/ascend_ir_limit_cores.md`](resources/ascend_ir_limit_cores.md)
  - npugraph_ex / aclgraph：[`resources/npugraph_ex_multi_stream.md`](resources/npugraph_ex_multi_stream.md)、[`resources/npugraph_ex_limit_cores.md`](resources/npugraph_ex_limit_cores.md)
