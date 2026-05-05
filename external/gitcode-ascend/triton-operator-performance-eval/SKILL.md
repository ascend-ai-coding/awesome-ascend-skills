---
name: external-gitcode-ascend-triton-operator-performance-eval
description: 评估 Ascend NPU 上 Triton 算子性能。使用 msprof/msprof op 采集性能数据，诊断 Memory-Bound/Compute-Bound
  瓶颈，测量硬件利用率，生成性能报告。
original-name: triton-operator-performance-eval
synced-from: https://gitcode.com/Ascend/agent-skills
synced-date: '2026-05-05'
synced-commit: 03b4bbd870d13fcd48044b285e84dd80e955214f
license: UNKNOWN
---

# Triton 算子性能评估（Ascend NPU）

## 核心原则

**只相信 msprof 数据，不凭直觉。**

**性能比（Ratio）定义**：`Ratio = torch_npu 耗时 / Triton 耗时`。Ratio > 1.0 表示 Triton 更快，Ratio < 1.0 表示更慢。**不是耗时的直接比值，而是耗时的倒数**。目标通常为 Ratio ≥ 1.0x。

**唯一可信采集方式：`msprof`（函数级）和 `msprof op`（算子级）**。其他方式（`time.time()`、`torch.npu.Event`、`do_bench`等）因包含 Host 开销且精度不足，**绝对不可用于性能评估**。

| 命令 | 用途 | 典型场景 |
|------|------|---------|
| `msprof --application="python x.py"` | 函数级：多算子对比、全链路分析 | "哪个算子最慢？" |
| `msprof op --kernel-name=K python x.py` | 算子级：硬件利用率、Bank Conflict | "这个 kernel 为什么慢？" |

**决策**：先 `msprof` 定位热点，再 `msprof op` 深度分析。

## 参考资源加载

| 任务 | 必须加载 | 不要加载 |
|------|----------|----------|
| 函数级性能对比 | [`msprof-function-level.md`](references/msprof-function-level.md) | msprof-op-level |
| 算子级硬件分析 | [`msprof-op-level.md`](references/msprof-op-level.md), [`performance-data-analysis.md`](references/performance-data-analysis.md) | msprof-function-level |
| 理解硬件术语 | [`ascend-terminology.md`](../triton-operator-shared/references/ascend-terminology.md) | — |

## 性能总结模板

完成评估后**必须**输出：

| 模块 | 内容 |
|------|------|
| 基本信息 | 算子名称、输入规模、硬件型号、测量方法 |
| 性能指标 | 执行耗时、内存带宽利用率、Cube/Vector 利用率、L2 Cache 命中率、Bank Conflict |
| 瓶颈诊断 | 瓶颈类型 + 判断依据 + CSV 数据证据 |
| 优化建议 | 按优先级排列，附预期收益 |

**原则**：所有结论必须有 Profiling 数据支撑。利用率 < 30% 标记为 **重点关注**。

## 瓶颈分类（仅用于诊断归类）

| 瓶颈 | 判断条件 | 详细优化手法 |
|------|----------|-------------|
| **Memory-Bound** | AI < 硬件平衡点 | → 交给 performance-optim skill |
| **Compute-Bound** | AI > 硬件平衡点 | → 交给 performance-optim skill |
| **Latency-Bound** | 带宽和计算利用率均低 | → 交给 performance-optim skill |

**边界**：本 skill 负责诊断瓶颈类型并输出数据。优化手法由 `triton-operator-performance-optim` 负责。

## 反模式清单（NEVER）

- ❌ 使用非 msprof 方式计时（time.time/Event/do_bench 等）
- ❌ Triton(NPU) vs PyTorch(CPU) 对比 — 必须 **同硬件** 对比
- ❌ 不预热就采集（首次含编译开销）
- ❌ 只测一次就下结论 / 只测大 shape
- ❌ 采集时含打印或日志（I/O 干扰）
- ❌ 混淆 `msprof` 和 `msprof op` 的用途

## 常见陷阱

| 陷阱 | 表现 | 正确做法 |
|------|------|----------|
| 用 `msprof op` 做对比 | 只看单个 kernel | 对比用 `msprof`，深度分析用 `msprof op` |
| `--kernel-name` 拼错 | 静默完成但无数据 | 确认名称与 Triton 函数定义一致 |
| 未区分编译和稳态 | 首次耗时异常高 | 至少 5 次预热后采集 |
| 忽略小 shape 性能 | 小 shape 差未发现 | 必须覆盖小/中/大 shape |
| 忽略 dtype 影响 | FP16/FP32 差异大 | 固定 dtype 对比，分别评估 |
| **函数级含 JIT 编译** | op_statistic 中 Triton avg 异常高（ms 级）| 对比用 `msprof op` 的 Task Duration，或函数级仅比较稳态调用 |
| **Triton vs torch_npu Core Type 不同** | Triton=AI_VECTOR_CORE，torch_npu=MIX_AIC | 注意：MIX_AIC 不一定使用 Cube 引擎，需检查 aic_cube_ratio 是否为 NA；真正差距在 MTE2 指令粒度 |

### 函数级 msprof 的 JIT 编译陷阱（重要）

`msprof` 函数级采集会包含 **JIT 编译时间**。Triton kernel 首次调用每种 (shape, dtype, mode) 组合时触发编译，耗时可达数秒。op_statistic.csv 中的 Total Time 和 Avg Time 会将这些编译时间算入。

**案例**：Triton kernel 的 op_statistic 显示 Avg=17791 us，但 `msprof op` 的 Task Duration 仅 1362 us。差距完全来自 11 种 unique config 的 JIT 编译。

**正确做法**：
1. 函数级对比仅看 **稳态调用**（忽略前 N 次或单独统计编译开销）
2. 或直接用 `msprof op` 的 Task Duration 作为 kernel 性能基准
3. 对比 torch_npu 时注意 Core Type：torch_npu=MIX_AIC 可用 Cube 引擎，Triton=AI_VECTOR_CORE 不行
