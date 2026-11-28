# 深化分析（按需）

## 何时用 / 边界

- 用于主 `model-recommend-analysis` 已经完成数据质量、总账、Host、算子、gap、图结构定位之后。
- 只在以下情况加载：
  - 需要判断优化天花板（Roofline）；
  - 需要确认某 op 是否在关键路径（消融/反事实）；
  - 动态 shape/tiling、并发/吞吐、精度/dtype/格式、内存/搬运、图规范化成为候选瓶颈；
  - 用户要求跨模型/跨版本比较。
- 只读分析：不采集、不实施、不验证、不做循环/收口。
- 结论仍由主 analysis 汇总；本 skill 只补充证据与方法，不单独输出优化建议文件。

## 输入

| 输入 | 用途 |
| --- | --- |
| Profiling：`op_statistic` / `op_summary` / `task_time` / `api_statistic` | 耗时、wait、gap、host 开销 |
| `dump_graph` 各阶段 | 图结构、消费者、布局/格式 |
| 模型/算子 shape、dtype、FLOPs、访存量 | Roofline 与消融 |
| 基线、配置、约束 | 判断实验条件与目标 |
| 主 analysis 的瓶颈分类与候选 | 决定加载哪个深化方法 |

缺输入时停止并列出缺项，不臆测。

## 1 实验设计与统计判定

用于判断"结论来自优化还是环境/测量偏差"；实验由工作流执行，本 skill 给设计与判定标准。

- 基线必须与实验组同模型、同 die、同 threads/cores、同 dtype、同输入、同 Pass 配置。
- 每场景至少 3 次，报告中位值并保留单次结果；仅报告平均值不足以判断稳定性。
- 优化项与基线交替或相邻执行，降低共享设备负载漂移影响；长序列测试分组重复。
- 对照实验只改一个变量：Pass/core/thread/dtype/shape/stream 不同时比较。
- 先估计噪声：同配置 3 次中段结果的标准差或极差；收益必须显著大于噪声，否则写"未观测到稳定收益"。
- 相对效应：`ΔQPS/QPS`、`Δavg/avg`、`ΔTP99/TP99`，标注样本数与离群点处理方式。
- 不用单次最优值代表收益，也不用单次最差值否定优化项。
- 首次运行含编译/内存分配/kernel 加载，单独记录；切换配置后的首轮先 warmup 或排除。
- 性能测试关闭不必要的输出、Profiling、plog、DumpData，避免额外开销。

## 2 深化分析

### 2.1 解析上界与 Roofline

```text
compute_bound ≈ FLOPs / 峰值算力
memory_bound  ≈ 访存量 / 峰值带宽
kernel_lower_bound ≈ max(compute_bound, memory_bound)
```

- 实际 kernel 时间与理论上界差距大，说明实现/并行/访存可能有问题。
- 小 M/K/N 的 MatMul 理论算力利用率通常偏低，区分"shape 限制"与"实现问题"。
- 逐元素算子关注访存与 UB 复用，不只比较 FLOPs。
- host-bound 场景的解析下界还应包含 `kernel_count × launch_cost` 与同步成本。

### 2.2 消融与反事实分析

- 消融：临时移除/替换一个算子或分支，测量迭代时间变化，确认是否在关键路径。
- 反事实：用轻量实现替换候选 op（或跳过非必要计算）观察上限；保持输出契约。
- 单变量消融：一次只改一个 shape/dtype/format/Pass。
- 无法移除的 op 可做"只读/空跑/固定输入"变体，估计下限成本。
- 消融用于验证"Profiling 耗时是否真的影响 wall"，避免只看 op 总时间。

### 2.3 动态 shape 与 tiling

- 统计 shape 分布：典型、最大/最小、动态维。
- 检查是否触发运行时 tiling、workspace 重分配、sync、kernel 选择变化。
- 对比 `all_dynamic` 与静态 shape 的 kernel 数、tiling 时间、sync 次数、QPS；区分"语义需要动态"与"测试配置导致动态"。
- 明确优化目标是"典型 shape 延迟"还是"最坏 shape 延迟"。
- 检查 tiling 缓存、workspace 复用、kernel 编译缓存是否命中。
- 覆盖退化输入（零尺寸/空序列、极小/极大 shape、边界值）；不用退化输入的性能数据作最终结论。

### 2.4 并发、吞吐与时延

- 先优化单实例时延，再做多实例并发；目标不同。
- 固定每实例 core 配额，再改 threads/instances；不同时改 core 与并发数。
- 记录每实例 avg/TP99 与全局吞吐；吞吐提升可能以尾时延为代价。
- Little 定律/排队视角：并发增加后单实例时延线性上升 → 资源饱和；吞吐不增 → host/同步/串行点未消除。
- 并发收益同时报告 QPS、avg、TP99/TP999 与长尾，不只看 QPS。

### 2.5 精度、dtype 与格式

- dtype 实验同时记录性能与数值差异，不能只看 QPS。
- 检查硬件支持：f64/int64 可能只能走 AI_CPU；fp16/bf16/fp32 的 kernel 选择不同。
- 布局/格式实验（ND/NZ、Transpose/Reshape）验证：是否减少搬运/提升 Cube 利用率、是否引入额外转换 kernel、是否改变输出 layout/dtype/接口。
- 记录 `HF32`、`opImplMode`、`keep_dtype` 等选项的实际生效值及精度影响；HF32 建议由主 analysis 输出。
- 任何 dtype/精度修改需产品/精度负责人确认；无 golden 时标注风险。

### 2.6 内存与数据搬运

- 区分 HBM 容量、workspace、中间张量、输入/输出拷贝、权重常驻。
- 检查 `aclrtMemcpy`、D2H/H2D、workspace 重分配是否在关键路径。
- memory-bound 场景分析 MTE2/带宽/访存模式，不只 kernel 时间。
- 减少中间张量、复用 workspace、权重常驻、减少 layout 转换通常优于单 kernel 微调。

### 2.7 图规范化与成本归属

- 图分析前统一节点命名、shape、dtype、格式，避免同名/重复节点干扰。
- 对重复子图先做单实例成本归属：哪些是实际 compute kernel、哪些是 shape/Reshape/Concat 轻量节点、哪些被多实例共享。
- 只对"重复出现 + 单实例有实际耗时 + 可融合/可消除"的 pattern 评估收益。
- 图前后对比使用同一阶段（如 `AfterInferShape`）与同一命名规则。

## 3 跨模型/跨版本比较

- 比较不同模型先比较结构：IO 数量、op 类型分布、shape/dtype、动态维、算子族数量、图规模。
- 不直接比较不同模型/子图范围的绝对 QPS；比较同类算子单位成本、结构比例、瓶颈类型。
- 同族/同规模模型可比较：新增/删除算子族、dtype/格式变化、动态 shape/V-pad/输出数量变化、相同优化项 effect 是否成立。
- 版本升级记录 CANN/驱动/算子包版本；同一 Pass 在不同版本可能 effect 不同。
- 外部 GMM/插件包与 CANN 版本匹配；构建成功不等于运行时选择正确，需检查实际 kernel/dtype。

## 输出与交接

- HF32 不在本 skill 输出建议：由主 `model-recommend-analysis` 提出，`model-recommend-core-limit-stress` 在限核 Step 1 执行；本 skill 只提供 Roofline/消融等证据。
- 将补充证据回传主 `model-recommend-analysis`：结论、数据、限制、置信度。
- 不单独生成 `optimization_suggestions.json`；建议由主 skill 统一输出。
- 若深化分析改变了瓶颈判断，明确标注"主 skill 需重新归类"。

## 自检

- 数据质量与统计口径是否与主 skill 一致？
- Roofline/消融是否标注假设与实测校准？
- 并发/动态 shape 是否区分目标口径（典型/最坏）？
- 跨模型/版本是否只比结构与单位成本，不直接比绝对 QPS？
- 是否把"实测/推断/假设"分开，并给出置信度？
