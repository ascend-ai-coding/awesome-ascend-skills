---
title: Pipeline 与 Buffer
purpose: 把 queue buffering 与 handoff 作为兼容材料解释清楚，用于和 RegBase 主路径对照。
read_when:
  - 需要解释或比较 MemBase-like queued pipeline。
  - fallback 路径仍使用 `TQue` / `TBuf` 语义，需要明确限制。
not_for:
  - 默认 RegBase shell 语义。
  - VF compute-body 设计。
keywords:
  - pipeline
  - buffer
  - TQue
  - compatibility
next_reads:
  - datacopy_best_practices.md
  - regbase_api_sync.md
  - ../pitfalls/regbase_vs_membase_confusions.md
depth: intermediate
topic_type: api
---

# Pipeline 与 Buffer

本卡片把 UB buffering 与 queue-style handoff 作为 RegBase direct-invoke 的兼容材料记录。需要对比 MemBase-style pipeline 与 RegBase-authoritative path，或记录 fallback 实现时使用。

## Buffer 选择

记录 compatibility path 时：

- 当 buffer 同时参与 movement 和 compute handoff，使用 `TQue`。
- 当 buffer 只是该兼容路径中的 vector computation scratch space，使用 `TBuf`。

不要因为看到 `TQue`、`TBuf` 或 `LocalTensor` 就判定不是 RegBase；关键要看 compute core 是否进入 VF / register compute。反过来，也不要让 queue pipeline 成为 RegBase 设计的中心而遮住 VF body。

## Queue Depth 与 Double Buffer

不要把 queue depth 和 double buffering 混为一谈：

- `TQue<..., depth>` 控制 queue abstraction。
- `InitBuffer(que, num, size)` 控制 provision 的 physical buffer 数量。
- `num = 2` 才是为该 queue 开启 double buffering 的关键。

实践中，对比或 fallback 到 queued model 时，template depth 保持小而清晰，让 `InitBuffer` 的 count 表达真实 buffer 数量。

## Pipeline Sync Model

兼容路径中的典型流程是：

1. allocate a queue buffer
2. copy into it
3. `EnQue` it
4. `DeQue` it in the next stage
5. compute
6. enqueue the output
7. `DeQue` the output when it is ready to leave UB

这个模型保证 movement 和 compute 可以安全 overlap。它是 UB pipeline 层的 stage handoff，不是 VF lane control，也不是跨核同步。

## `PipeBarrier` 何时使用

只有在兼容路径里需要粗粒度 fence，或正在诊断 sync defect 时才使用 barrier。

- 适合调试 bug 是否由 ordering 引起。
- 不适合作为默认性能策略。
- 不是 proper queue handoff 的替代品。

## Batch Movement Pattern

row-oriented 工作中常见模式：

- batch copy 多行进入 UB。
- 在 UB 内逐行处理。
- batch copy 结果回 GM。

这通常比 queue-based fallback 中每行发一次 copy call 更好。

## 非可编译结构示例

```cpp
// queued compatibility path 的层级示意：
auto tile = inQue.AllocTensor<T>();
// DataCopy(tile, gmIn, copyLen);
inQue.EnQue(tile);

auto ready = inQue.DeQue<T>();
// compute on ready, or pass UB address into VF body if this is RegBase-compatible shell
outQue.EnQue(ready);
```

设计记录要说明：queue 数量、`InitBuffer` 的 `num` 与 `size`、是否 double buffer、`CopyIn -> Compute -> CopyOut` ownership 如何转移，以及 VF body 从哪个 UB address load。

## 常见错误

- 把 `AllocTensor` 当成也会等待数据 ready。
- 以为 `depth` 本身开启 double buffering。
- movement path 需要 queue 语义，却误用 `TBuf`。
- 用 barrier 串行化 pipeline，而 queue handoff 原本可以保持 stage overlap。

## 相关文档

- [[regbase_api_sync]]
- [[datacopy_best_practices]]
- [[arithmetic_and_reduce]]
- [[../pitfalls/precision_guide]]
