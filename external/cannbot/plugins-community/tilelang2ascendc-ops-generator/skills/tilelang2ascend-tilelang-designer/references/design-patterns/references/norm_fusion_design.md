# Norm 族 + 激活融合设计要点（Design 视角）

> 本文件只含**设计阶段决策**（可用 TileLang DSL 表达）。AscendC 实现细节
> （TQue 双缓冲、affine 预重排、两遍中心化结构、swish 简洁形式、S_CHUNK 分片）
> 见 translator references/ascendc_norm_fusion_patterns.md。

## 适用

`group_norm / layer_norm / rms_norm / batch_norm / instance_norm` 等 Norm 族，以及它们与
`swish / silu / gelu` 等激活的融合算子（如 `GroupNormSwish`、`LayerNorm`+affine）。

## 设计决策 1：组间独立归一化的并行范式

- 归一化统计量按「组」独立（GroupNorm 的 `N×G` 组、LayerNorm 的 `N` 行、InstanceNorm 的
  `N×C` 组），**组间无数据依赖** → 直接把「组」作为并行工作单元分核，无需跨核归约/同步。
- 展平成 `[M, L]`：`M` = 组数，`L` = 每组元素数（`CPerG × S`，S = 空间维乘积）。
- 单组内两遍：Pass1 归约求 mean/rstd，Pass2 归一化 + affine + 激活。

## 设计决策 2：方差算法路线（在 reduce_design §4b 之上细化）

- 先按输入动态范围选路线：小值域/已归一化 → 同趟 `sum+sumsq`；大动态范围 → 两趟/Welford
  （见 reduce_design §4b）。
- 选两趟时，落地结构用「**两遍中心化 + mean 修正**」：第一遍求 mean → 中心化 → 求 mean
  修正 → 用修正后 mean 再中心化 → `Mul(x²)` 求 var（细节见实现层 §3）。
- 跨 chunk 的部分和标量累加需考虑 Kahan 补偿（长归约 fp32 精度）。

## 设计决策 3：affine 融合（预计算 scale/bias）

- 归一化后接 per-channel affine 时，设计上把 `(x−mean)·rstd·γ+β` 合并成
  `x·(γ·rstd) + (β − γ·mean·rstd)`：Pass1 末尾每通道预计算 `scale_c`、`bias'_c`，
  Pass2 退化为一次乘加。
- 无 affine（纯 norm）则只合并标量因子 `rstd`，同样消除逐元素重复乘。

## 设计决策 4：双缓冲流水（两遍读 GM 的算子必选）

- 只要 Pass1/Pass2 两遍读同一输入，就必须在 tile 数据流里规划**双缓冲流水**
  （输入/输出各 depth 2），让搬运与计算重叠，否则两遍带宽全暴露在关键路径。
- 设计时明确「队列槽位数」与「哪些 buffer 进队列、哪些用 TBuf 临时」，避免用全管线
  barrier 打断流水。

## 设计决策 5：按 shape 规模分档（与归约/重排类同规律）

- 小/中规模按规模带少开核、少分片（避免 dispatch ramp 与分片固定开销）；大规模铺满核并
  按空间维分片。
- 见 shuffle_design.md 的「核数分档」一节（同类规律）。
