---
name: external-cannbot-ops-pypto-op-perf-tune-tune-swimlane
description: PyPTO 算子深度性能调优技能。通过泳道图分析及调优性能，包括 Stitch 调优、TileShape 深度调优、合图调优、调度策略调优等。当用户需要进行深度性能调优、泳道图分析、Stitch
  优化、合图优化时使用此技能。触发词：深度性能调优、泳道图分析、Stitch 调优、合图调优、调度优化。
original-name: tune-swimlane
synced-from: https://gitcode.com/cann/cannbot-skills
synced-date: '2026-09-05'
synced-commit: a426ec91e1038f233066724d63235c719a46a10d
license: UNKNOWN
---

# PyPTO 算子深度性能调优

## 概述

深度性能调优通过泳道图分析及调优性能，采用 man-in-loop 的方式，通过获取并分析当前算子性能数据，针对性调整各性能配置参数，经过迭代调优逐步逼近最佳性能。

> 资料获取统一使用 skill `pypto-docs-search`：按需搜索算子参考实现等文件/目录/内容。

## ⛔ 前置条件（强制门控）

1. **完成开箱性能调优**：先进行代码级优化
2. **精度校验通过**：确保算子计算正确
3. **已采集性能数据**：生成泳道图和气泡分析报告

**⛔ ⛔ ⛔ 独立采集数据（强制）：每次进入此阶段时，必须重新运行测试（带 debug_options）采集最新泳道图数据。禁止复用 FRONTEND 阶段或其他轮次的旧数据！修改代码后性能特征已变，旧数据无法反映当前状态，以此决策会导致错误结论。⛔ ⛔ ⛔**

## 泳道图分析

### 泳道图文件位置

泳道图数据文件位于 `output/output_*/` 目录：
- `merged_swimlane.json` - 泳道图数据文件
- `bubble_analysis.log` - 气泡分析报告

### 查看泳道图

1. 通过 PyPTO Toolkit 插件查看
2. 或在 https://ui.perfetto.dev/ 上传泳道图文件
3. 查看泳道图文件及日志信息

### 泳道图关键信息

- 任务的执行顺序和耗时信息
- 各核心的工作时间和等待时间
- 气泡（线程等待调度的时间）
- 任务依赖关系

## 调优方向

### 1. Stitch 调优

Stitch 配置决定了多少个 root function 被同时下发调度。

#### 1.1 配置方法

```python
@pypto.frontend.jit(
    runtime_options={"stitch_function_max_num": 128}
)

```

**参考资料**
- [stitch_function_max_num 参数设置说明](https://raw.gitcode.com/cann/pypto/raw/master/docs/zh/api/config/pypto-frontend-jit.md)

#### 1.2 参数影响

| 参数值 | 优点 | 缺点 |
|--------|------|------|
| 过小（如 1） | - | 每个任务需同步，调度开销大 |
| 适中（如 128） | 泳道图紧凑，调度开销低 | - |
| 过大（如 512） | 泳道图更紧凑 | 调度耗时增加，workspace 增加 |

#### 1.3 调优建议

在内存资源允许的前提下，逐步增大 Stitch 配置，结合泳道图和端到端总耗时数据调整参数。

### 2. TileShape 深度调优

> **⛔ 执行 TileShape 深度调优时，必须加载 [TileShape 深度调优](references/tileshape-deep-tuning.md) 获取完整指南。**

### 3. 核使用率分析与负载均衡（合图前置条件）

> **⛔ ⛔ 合图调优前，必须加载 [核使用率分析与负载均衡](references/core-usage-load-balancing.md) 完成核使用率分析。未完成本步骤禁止进入第 4 节合图。**

### 4. 合图调优

> **⛔ ⛔ ⛔ 配置任何合图参数前，必须加载 [合图调优](references/merge-optimization.md) 获取完整指南。合图配置错误是性能退化的最常见原因。**


### 5. 调度策略调优

当上下游子图之间依赖较为简单，或下游子图输入 Tensor 的 L2 命中率较为重要时，推荐使用 L2 亲和调度。

```python
@pypto.frontend.jit(runtime_options={"device_sched_mode": 1})

```

**调优建议**：
- 尝试不同的调度策略，值域范围是[0, 3]

**注意事项**：综合考虑 L2 复用与负载均衡的影响，不同场景的最佳配置策略不同。

**参考资料**
- [device_sched_mode 参数设置说明](https://raw.gitcode.com/cann/pypto/raw/master/docs/zh/api/config/pypto-frontend-jit.md)


### 6. Matmul 访存布局优化（L2 命中率优化）

> **在大 shape Matmul 场景下（M、N、K 全部较大），即使 TileShape 配置了推荐值，固定的分核布局可能导致 L2 命中率偏低、MTE2 带宽利用率不足，此时应加载 [Matmul 访存布局优化](references/matmul-l2-layout.md) 获取 L2 命中率优化方法。**

## 调优检查清单

**⛔ 必须按以下清单逐项执行。每项标记为 ✅已尝试 或 ❌已失败（附原因），禁止跳过。完整优化点信息参考 [shared/optimization_catalog.md](../shared/optimization_catalog.md)。**

**优化优先级**：
1. ⭐⭐⭐ **P0 - 核使用率分析 + 核填充** → 详见 [S-1][S-2]
2. ⭐⭐⭐ **P1 - 负载均衡 + TileShape 深度调优 + 访存布局优化** → 详见 [S-3][S-11][S-12][S-13]
3. ⭐⭐⭐ **P2 - 手动合图（sg_set_scope）** → 详见 [S-4]
4. ⭐⭐ **P3 - 自动合图** → 详见 [S-5][S-6][S-7][S-8]
5. ⭐ **P4 - Stitch + 调度策略** → 详见 [S-9][S-10]

**🔥 P0 - 核使用率分析 [S-1]**（合图前置条件）：
- [ ] [S-1] 是否运行 analyze_core_usage.py 统计每个 leafHash 核使用率
- [ ] [S-2] 每个 NOT FULL 子图是否已尝试 TileShape 调整（减小 L0/L1 增加任务数）
- [ ] 是否尝试完所有轴的 TileShape 调整后才判定"核无法再增"

**🔥 P1 - 负载均衡 [S-3]**（核填充后强制）：
- [ ] [S-3] 是否按 total(us) 降序排列所有子图，识别瓶颈子图
- [ ] 瓶颈差距是否已量化（>20% 必须优化）
- [ ] [S-11] 是否针对瓶颈子图尝试了 Cube TileShape 深度调优（每次只调一个子图）
- [ ] [S-12] 是否对 Vector 计算尝试了 Vector TileShape 深度调优（调整 TileShape 对齐上下游）
- [ ] [S-13] 大 shape Matmul 是否检查了 MTE2 带宽利用率并尝试了分核布局优化

**🔥 P2 - 手动合图 [S-4]**（最重要但最易跳过）：
- [ ] [S-4] 是否运行 analyze_aiv_dep_chains.py 分析 AIV 依赖链
- [ ] 是否检查了可合并的连续 Vector 操作（有直接数据依赖、同循环层级、无 Cube 夹杂）
- [ ] 是否对每个可合并链段尝试了 sg_set_scope
- [ ] 如果跳过此项，是否说明了具体原因（而非"觉得不适用"）

**P3 - 自动合图 [S-5~S-8]**：
- [ ] [S-5] 是否对短耗时（<10us）AIV 任务尝试了 vec_nbuffer_setting
- [ ] [S-6] 是否对核满的 AIC 子图尝试了 cube_l1_reuse_setting
- [ ] [S-7] 是否对核满的 AIC 子图尝试了 cube_nbuffer_setting
- [ ] [S-8] 如已配置 S-6/S-7，是否检查了协同使用是否过大

**P4 - Stitch [S-9] + 调度策略 [S-10]**：
- [ ] [S-9] 是否尝试了 stitch_function_max_num 调整
- [ ] [S-10] 是否尝试了 device_sched_mode 调整（1/2/3）


## 常见问题

> 遇到以下问题时，加载 [常见问题](references/faq.md) 获取详细解答：
> - Q1: 泳道图文件在哪里
> - Q2: 如何查看性能统计
> - Q3: 气泡是什么
> - Q4: 控制开销占比过高怎么办
> - Q5: 如何选择合适的 Tilesize


## 参考资料

- [性能调优文档](https://raw.gitcode.com/cann/pypto/raw/master/docs/zh/tutorials/debug/performance.md)
- [Matmul 高性能编程](https://raw.gitcode.com/cann/pypto/raw/master/docs/zh/tutorials/debug/matmul_performance_guide.md)
- 算子参考实现案例（attention 类等）：用 `pypto-docs-search` 搜索算子参考实现
- [性能优化案例](https://raw.gitcode.com/cann/pypto/raw/master/docs/zh/tutorials/debug/performance_case_quantindexerprolog.md)
