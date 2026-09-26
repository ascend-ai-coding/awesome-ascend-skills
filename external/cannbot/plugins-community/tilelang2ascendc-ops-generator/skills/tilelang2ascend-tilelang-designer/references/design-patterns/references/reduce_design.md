# 归约族算子设计要点（Design 视角）

> 本文件只含**设计阶段决策**（可用 TileLang DSL 表达）。AscendC 实现细节
> （rightPadding 参数、isReuseSource、AR/RA 模式 API 用法等）见
> translator references/ascendc_reduce_patterns.md。

## 适用

`sum / mean / max / min / prod` 等沿维（或全部）归约，以及 `layer_norm / rms_norm /
batch_norm / var / std` 等 forward 内含归约的统计量算子。

## 设计决策 1：(O,R,I) 三分解路由

任意维归约统一分解为 `(O, R, I)`：O = 归约维之前的外层维度积、R = 归约维长度、
I = 归约维之后的内层维度积。**不做物理转置**（转置是一次全量数据搬移，比归约本身更贵）。

| 条件 | 路径 | 归约模式 | 设计要点 |
|---|---|---|---|
| `I > 1`（非末轴） | A：跨行归约 | RA 树形归约 | 工作单元 = (o, iTile)；沿 I 分块 |
| `I == 1` 且 R 可装入 UB | B：末轴多行批归约 | AR | 一次调用产出多行结果；批行数受 UB 预算约束 |
| `I == 1` 且 R 超 UB | C：末轴分块两级树 | 矢量累加器逐块 += | 一行一核，块循环无标量回读 |

## 设计决策 2：沿 I 分块（非末轴，Path A）

- 归约方向是 R（"谁和谁加一起" = 同一列的所有行），但一次能处理多宽受 **I 的宽度** 限制
  （`[R, I]` 整块装不进 UB）
- 沿 **I** 切成 iTile（每块 tileA 列），块内完成跨 R 归约得到 `[tileA]` 部分和，块间累加
- 归约轴（R）与分块轴（I）解耦：tileA 由 UB 预算定上限、核填充定下限

## 设计决策 3：tile 内 pad + 补零语义

- 行尾对齐用 pad（对齐到硬件拷贝要求），设计时明确 **pad 区的填充语义**：
  `sum→0`、`max→-inf`、`min→+inf`、`prod→1`（由算子语义决定）
- pad 区不补零会把 UB 残留加进归约结果（正确性），务必在设计时规划

## 设计决策 4：累积精度

- fp16/bf16 输入先升 fp32 再累加/归约（精度保证；且部分归约模式仅支持 fp32）

## 设计决策 4b：方差/二阶矩算法路线

- `var / std` 等二阶矩计算先定**算法路线**：
  - 同趟法：`sum` 与 `sumsq` 同一次遍历累加（`var = E[x²] − mean²`），省一趟搬移——**小值域/已归一化数据**适用
  - 两趟法 / Welford：**大动态范围输入**用（避免 `E[x²] − mean²` 的 catastrophic cancellation，见上游 `alg-welford.md`）
- 与"fp32 累加"同属精度决策，按输入动态范围选，别只看性能选同趟法

## 设计决策 5：host 核数分档（与重排类同规律）

- 小/中规模按规模带少开核（避免 AIV dispatch ramp），大规模铺满核
- 见 shuffle_design.md 的"核数分档"一节（两类算子同规律）
