# 多流 Plan 方案细节片段

用于填写编排报告里某个多流 Plan 的「方案细节」自由块。**形成 plan 时**给出设计部分，**review / 落地后**回填实测部分。编排层（orchestrator）不解析这块，只有多流方向的 implementer / reviewer 读写它。

一个多流 Plan 的方案细节包含：

- **并行对象与流分组**：并行什么；各算子 / 模块分到哪条流（主 / 副1 / 副2）；tag 粒度（同一副流串行还是多个独立副流）。
- **汇合点**：在哪里汇合，用什么同步（Ascend IR 用 `npu_wait_tensor` / tagged event；aclgraph 用 `Event` / `wait_stream`）。
- **方案 DAG**：Mermaid，标清主 / 副流归属和 event 边（`-.->`）。
- **强度**：当前档位 + 计划（C1 最小切流 → C2 + 控核 → C3 + 手动同步），见 SKILL 正文"分析"一节。
- **GE auto-reorder 风险**（Ascend IR 路径设计期必填）：按 [`ge-reorder-design-check.md`](ge-reorder-design-check.md) 过一遍——副流输出 → 主流消费者 → 是否有 cheap precompute → 是否成 barrier → 应对（是否在 with-block 内显式钉住）。
- **overlap_pct 实测**（review / 落地后填，按 [`timeline-overlap-check.md`](timeline-overlap-check.md)，**不允许"是 / 否"二元**）：副流时间窗 `[start,end] us` / 主流 dominator 时间窗 `[start,end] us` / `overlap_pct` 数值 / 判定（真并行 ≥0.5 / 部分 0.05~0.5 / 假并行 ≤0.05）。
- **enable 开关与回退路径**：开关名、关掉后走哪条原路径。
- **风险与副作用**：`TransData` / `BroadcastTo` / `MemSet` 长尾、主副争核（`Block Dim` 加和）、shape 劣化等。

**上浮到编排报告信封的「证据摘要」**（这是 orchestrator 裁决要看的）：一句话给出 `overlap_pct` 判定 + wall Δ%（以性能分析报告为准）+ 关键副作用；通过 / 淘汰的状态建议交给 reviewer 和主 agent。其余细节都留在本片段里。

> 方向级的 module / op DAG 与并行性分析不写在这里——那是被多个 Plan 共享的方向级分析产物，按 [`module-decomposition.md`](module-decomposition.md) 的结构单独成文，本 Plan 引用它即可。
