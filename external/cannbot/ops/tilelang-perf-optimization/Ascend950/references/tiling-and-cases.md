# Tiling 与 case 建模

## 先还原执行路径

对每个 case 从公开入口向下记录：dispatch 条件 → build 参数 → `@T.prim_func` → `T.Kernel` 任务映射 → tile 循环 → 执行域。相同算子名不代表相同执行路径；broadcast 轴、reduction 轴、dtype、对齐和尾块都可能改变分支。

建议建立下表：

| Case | 语义路径 | dtype/累加 | dispatch | 核数 | tile | 尾块 | 基线瓶颈 |
|---|---|---|---|---:|---|---|---|

只有执行路径和瓶颈均相近的 case 才合并建模。

## 通用模型

### 任务与核数

- `independent_tasks`：无需跨任务依赖即可并行的任务数。
- `used_cores <= independent_tasks`。
- `tasks_per_core = ceil(independent_tasks / used_cores)`。
- 同时记录最长核和最短核的任务量；平均值不能反映尾差。

核数来源及 vector/cube/mixed 选择必须由 `tilelang-performance-best-practices` 和当前仓库配置确认。

### tile 与循环

- `tiles_per_task = ceil(valid_extent / tile_extent)`。
- 每 tile 区分 `valid_extent` 与为 SIMD/DMA/layout 准备的 `padded_extent`。
- 估算固定开销时使用实际 tile 次数；估算 GM 字节时只统计合法读写；估算 UB 占用时使用完整 padded footprint。

对小 shape，增大 tile 或合并任务可能降低循环与调度开销；对大 shape，过大 tile 可能减少并行度或造成 buffer 压力。两者都必须由实测验证。

### buffer 预算

逐个 buffer 计算：

`bytes = product(padded_shape) * dtype_bytes * versions`

总预算还要加入常驻数据、临时结果、对齐 padding 和安全余量。`T.annotate_buffer_versions` 与 `T.Persistent(..., num_stages=N)` 的版本关系以当前 TileLang 实现和已验证样例为准，不自行假设。

### 数据流

为每个 tile 列出：

- GM 读取/写回字节。
- UB/L1/L0 中间结果生命周期。
- 同一数据的重复读取次数。
- 运算量与 reduction/GEMM 累加状态。
- 搬入、计算、写回是否有可验证的重叠机会。

## 参数输出

方案中的参数必须同时给出：当前值、候选值、推导依据、适用 case、容量/对齐门禁和回退条件。示例：

| 参数 | 当前 | 候选 | 依据 | 适用 case | 门禁 |
|---|---:|---:|---|---|---|
| `tile_n` | 128 | 256 | 小 case 循环开销主导 | group-S | UB 含 versions 与 padding 后不超预算 |
| `used_cores` | 20 | 8 | 独立任务仅 8 个 | group-tail | 不超过任务数，逐 case latency 不退化 |

没有经过当前版本 API、lowering 和容量核验的候选值只能标为待验证，不能写成最终配置。
