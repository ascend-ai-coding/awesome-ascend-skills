# 多流 overlap 时间线复盘

本页回答一个问题：**副流上的算子是不是真的和主流并行执行了？** 这是判断多流方案在物理执行层面是否成立的唯一证据。

## 背景：为什么"Stream ID 切对了"不等于"并行成立"

`torchair.scope.npu_stream_switch` 给副流贴的是**逻辑标签**。GE 图模式编译期还会做几件事：

- **算子提前 / 延后下发**：基于数据依赖图的拓扑序，而不是源码顺序。
- **被动拉取**：副流上某个 tensor 被主流消费时，GE 可能把"消费前必须做的小预计算"（如 `Cast` / `Swish` / `Sigmoid`）自动 pin 到消费侧的流。
- **流内 FIFO 排队**：同一 stream id 上的 op 严格按下发顺序串行执行。

最常见的错位：副流的输出 `T` 有一个轻量级 precompute `f(T)`（例如 `silu(T) = Swish(Cast(T))`），下游消费者在主流。GE 把 `f(T)` 拉到主流提前下发，而主流被它占住、必须等副流的 `T` 就绪才能继续。结果是副流跑的时候主流 idle，副流跑完后主流再串行做 `f(T)` 和原本想跑的 dominator。这种"假并行"在只看 Stream ID 的报告里看不出来，必须看 `Start Time`。

## 何时必须做时间线复盘

下面任意一种情况都必须算 `overlap_pct`，不能跳过：

1. 切流改造后，候选模块 wall 与改造前持平或仅微变（< 5%），但 Stream ID 已经按设计切到副流。
2. 候选模块的并行度（`kernel_sum / wall`）≤ 1.05×。
3. 准备判某个方案**无效**之前——必须先排除"假并行"再下结论，否则改进方向会判错。
4. 准备判某个方案**成立**之前——确认收益不是噪声、并行点确实在物理上落地了。

## 输入

从 profiling 的 `kernel_details.csv`（或等价的 per-op 明细）取每个 op 的 `Stream ID` / `Start Time(us)` / `Duration(us)`，并按方案设计区分：哪些 op 应该在副流、哪些是主流上被掩盖的 dominator。字段查法见 [`kernel-fields-lookup.md`](kernel-fields-lookup.md)。

## 5 步重建甘特图

### Step 1：选窗口

按方案描述选两个 op 区间：

- **副流时间窗** = 方案切到副流的那段 ops。
- **主流期望被掩盖的 dominator 时间窗** = 方案假设这段时间副流能躲在它背后跑的主流 op（通常是方案里"主路径 dominator"那一项，常见为 conv1d / matmul / FA）。

两个区间都要拿到每个 op 的 `Start Time(us)` 和 `Start Time(us) + Duration(us)`。

### Step 2：取每个 op 的 `(Stream ID, Start Time, Duration)`

只下钻这几个字段的必要几条 op，不要把整份明细灌进上下文。字段集和命令见 [`kernel-fields-lookup.md`](kernel-fields-lookup.md) 的 "post-mortem" 字段集。

### Step 3：算两组区间

```text
side_window  = [min(side_op.start), max(side_op.start + side_op.duration)]
main_window  = [main_dominator.start, main_dominator.start + main_dominator.duration]
overlap      = max(0, min(side_window.end, main_window.end) - max(side_window.start, main_window.start))
side_len     = side_window.end - side_window.start
main_len     = main_window.end - main_window.start
overlap_pct  = overlap / min(side_len, main_len)
```

### Step 4：判定

- **真并行**：`overlap_pct ≥ 0.5`，至少 50% 的副流时间能藏在主流 dominator 后面。
- **部分并行**：`0.05 < overlap_pct < 0.5`，主副流有重叠但 dominator 没把副流完全吃下；通常说明 op 集合划分要再调（让副流更长 / 让 dominator 更长 / 换 dominator）。
- **假并行（serial）**：`overlap_pct ≤ 0.05`，Stream ID 切对了但物理串行——主流大概率被某个 GE 拉过来的 precompute 卡住。

### Step 5：找 barrier op（仅在判定为假并行时）

- 把主流 stream id 上的 op 按 `Start Time(us)` 排序。
- 找到主流上**第一个**输入来自副流的 op X（多半是 `Cast` / `Reshape` / 小算子）。
- 检查 X 在主流 FIFO 顺序里是否排在主流 dominator 之前。
- 如果是，X 就是 barrier：主流 dominator 必须等 X 开始才能跑，而 X 必须等副流输出就绪。
- 解法：派生一个新的编排候选，把这个 op **显式钉到副流**——在 `with npu_stream_switch(...)` 块内显式写出 `f(T)` 的计算，而不是依赖 GE 自动放置。设计期怎么提前预判见 [`ge-reorder-design-check.md`](ge-reorder-design-check.md)。

## 怎么记录结论

只记关键摘要，不贴整张 op 表。例：

```text
side_window [1615.25, 1634.75] us、main_dominator window [1638.00, 1660.75] us（in_proj_qkv），
overlap=0us / overlap_pct=0%，判为假并行；barrier = ops 55-56 (Cast+Swish, silu(z))，被 GE 拉到主流；
下一候选：把 silu(z) 显式预计算钉在 with-block 内。
```

## 该做、不该做

- ✅ overlap 复盘是判断方案成败的核心证据，必须自己做，不能因为"Stream ID 已切到副流"就直接判通过。
- ✅ 判通过和判淘汰前都要算 `overlap_pct`，并把结论连同 barrier（如有）记下来。
- ❌ 不要把整份 per-op 明细贴进对话，只下钻必要的几条 op。
- ❌ 不要把"host overhead 吃掉收益"当默认解释，先证明 overlap 真实成立。

## 关键约束

- `Start Time(us)` 在不同采集之间不可比（基准时间不同），`overlap_pct` 只能在**同一次采集内**计算。
- `Duration(us)` 在不同采集之间通常稳定，除非 shape 变化。
- 跨采集对比 wall / 并行度要用性能分析报告（同口径指标），对比物理布局用源码 + 方案差异，而不是直接比时间戳。
