# GE auto-reorder 风险设计清单

GE 图模式编译期会重排算子，最常见的事故是把"消费副流输出的轻量 precompute"拉到主流、卡住主流 dominator，让切流变成假并行（原理见 [`timeline-overlap-check.md`](timeline-overlap-check.md)）。这个风险在**设计阶段就能预判**，不必等到测出来才发现。任何一个把 op 切到副流的候选，落地前都过一遍这份清单。

> 本清单只适用于 Ascend IR / GE 图模式（`npu_stream_switch` 路径）。npugraph_ex / aclgraph 是显式 stream，没有 GE 自动重排，但仍要关注跨流 tensor 生命周期（`record_stream`）。

## 设计期逐条过

针对当前候选，回答下面每一条，不允许留空：

1. **副流的输出有哪些下游消费者在主流？** 列出 `tensor 名 → 消费者`。
2. **这些消费者里有没有 cheap precompute？** 典型是 `Cast` / `Reshape` / `Swish` / `Sigmoid` / `Mul` 这类轻量算子（例如 `silu(z) = Swish(Cast(z))`）。
3. **GE 会把这些 precompute 提前下发到哪条流？** 设计期先推断，落地后用 profiling 的 `Stream ID` + `Start Time` 验证。
4. **这些 precompute 会不会在主流 FIFO 顺序里排在主流 dominator 之前，形成 barrier？** 一旦排在前面，主流 dominator 就得等它，而它又得等副流输出 → 物理串行。
5. **应对**：要不要在副流 `with npu_stream_switch(...)` 块内**显式预计算**这些 op，把它们 pin 在副流上，而不是依赖 GE 自动放置？

如果第 2、4 条命中而第 5 条没处理，这个候选大概率会落成假并行——要么在设计里就把 precompute 钉进副流，要么换一种集合划分让副流有真正的 dominator 可吃。

## overlap 结论的纪律

判断这个候选有没有真并行，**必须用 `overlap_pct` 数值，不允许"是 / 否"二元结论**。记录时给全三项：

- 副流时间窗：`[start, end] us`
- 主流 dominator 时间窗：`[start, end] us`
- `overlap_pct` 数值 + 判定：`真并行 (≥0.5)` / `部分并行 (0.05~0.5)` / `假并行 (≤0.05)`

算法见 [`timeline-overlap-check.md`](timeline-overlap-check.md)。"Stream ID 已切到副流"不能作为并行成立的结论。
