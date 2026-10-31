---
name: external-cannbot-community-model-model-recommend-performance-analysis
description: 模型性能分析（工作流条目 5）。端到端与算子级性能采集（run.sh --prof ge/msprof）、kernel 表/overlap
  机会表/融合模式表三表报告、bound 判定（AIC/AIV 利用率、带宽、kernel/墙钟占比）与优化建议。触发：性能分析、性能采集、profiling、瓶颈定位、算子耗时、TP99/QPS
  分析、bound 判定、AIC/AIV 利用率、融合命中分析。
original-name: model-recommend-performance-analysis
synced-from: https://gitcode.com/cann/cannbot-skills
synced-date: '2026-10-31'
synced-commit: 39919e2c995e1cd9e193f11ac2e3fcb3cbd1d26c
license: UNKNOWN
---

# 模型性能分析

## 流水位置（强制）

三重角色：
1. **基线与瓶颈分析**：采集性能数据、三表分析、bound 判定，驱动优化方向
2. **「改动校验双门禁」的性能侧执行者**：所有对模型执行的改动，落地前必须做**同口径前后对比**，主指标不得劣化超过噪声带（精度门禁先行，见 [precision-comparison](../model-recommend-precision-comparison/SKILL.md)；双门禁定义见 [ascend-env-baseline](../model-recommend-ascend-env-baseline/SKILL.md) 全局约束）
3. **优化验证**：每个优化项落地后复测，收益归因与回退判定

**深化衔接（强制）**：本 skill 完成三表 + bound 判定后，转入 [profiling-deep-analysis](../model-recommend-profiling-deep-analysis/SKILL.md)（条目 5 的四项专项）完成四项专项分析（算子执行时间 / 多流掩盖 / H2D-D2H / host 瓶颈）——本 skill 回答"瓶颈在哪一类"，四项专项回答"具体是什么、可优化空间多大"；循环场景下（[optimization-loop](../model-recommend-optimization-loop/SKILL.md)）两者每轮都执行。

## Prerequisites

- 可运行的基线（`run.sh <id>` 跑通，串行与目标并发口径已知）
- 采集场景定义（见 Step 1）
- msprof 工具链可用（CANN 9.1.0 自带，`which msprof`）
- 采集前确认目标卡空闲（`npu-smi info`；`sweep.py` 内置占用检测可复用）

## 目标

端到端与算子级性能采集、瓶颈定位（bound 分类）、优化前后对比，驱动 shape 策略（6）、算子适配（7）、并发（8）、访存（9）、融合规划（10）的迭代。

## 输入 / 输出契约

**输入**：可运行基线 + 采集场景定义（batch、`THREAD_NUM`、`CORE_NUM`、`CircleCount`）。

**输出**：性能报告——分场景时延/QPS 数据、算子级 kernel 表、overlap 机会表、融合模式表、bound 判定结论、优化建议。

## 工作流步骤

### Step 1: 定义采集场景

单并发时延基线与目标并发吞吐基线**分开采集**（瓶颈完全不同）：

- 单并发：看 kernel 串行执行、launch 开销、H2D 拷贝——shape 策略/融合收益主战场
- 目标并发：看资源争抢、排队、TP99 劣化——并发/限核调优主战场
- `CircleCount` 足够大（≥1000）保证 `[GE PROGRESS]` 打点稳定；`--debug-freq` 控制频率

### Step 2: 模型级采集

`bash run.sh <id> --prof ge|msprof`（**onnx 路径**；pytorch/HF 路径改用 `torch_npu.profiler` 导出 kernel 明细与 trace，口径对齐后同表分析）：产物落 `models/M<id>/.cache/profiling/PROF_*`；主数据源 `mindstudio_profiler_output/op_summary.csv`（算子名 + shape + 时长 + 流水指标）。

**归档（必做）**：完整采集目录拷贝到 `prof/{model_name}/{case}/`（case 命名含场景标识，如 `serial-b1` / `t4-b1`），并提取 kernel 汇总（kernel_name/count/total_us/avg_us/min_us/max_us/total_pct，按 total_us 降序）落同目录。

**已知问题处理**：若 op_summary/trace 解析失败，先检查 `libsqlite3.so` 软链（历史环境仅有 `libsqlite3.so.0`，需建软链并注入 `LD_LIBRARY_PATH`）。

### Step 3: 算子级深度采集

msprof 对热点 kernel 深度采集（aic-metrics 覆盖 Cube/Vector/MTE 各流水），用于 [kernel-generation](../model-recommend-kernel-generation/SKILL.md) 的 kernel 级优化归因与自定义算子 benchmark。

### Step 4: 三表分析

| 表 | 内容 | 阈值 |
|----|------|------|
| kernel 表 | 耗时 Top-N 及占比（op_summary 聚合） | 累计 ≥1% |
| overlap 机会表 | 多线程下 AIC/AIV 利用率、kernel 间隙、H2D 拷贝窗口、独立子链并行窗口 | 同上 |
| 融合模式表 | 对照 GE 融合结果（`models/M<id>/.cache/run/fusion_result.json` + `--dump_graph` 图结构）判定"应融合未融合"的小算子串/GM 往返链 | 同上（source-backed，不做模糊匹配） |

### Step 5: bound 判定与归因

- **compute-bound**：AIC/AIV 利用率 vs 本机实测锚定峰值（matmul TFLOPS）
- **memory-bound**：带宽利用率 vs 本机实测 copy 带宽；查表/Gather/大 GEMM 权重读取热点 → [memory-access-optimize](../model-recommend-memory-access-optimize/SKILL.md) 候选
- **host/launch-bound**：kernel 总时长 ÷ 墙钟时长——占比低说明调度/launch 主导 → 算子融合（[fusion-operator-adaptation](../model-recommend-fusion-operator-adaptation/SKILL.md) / [fusion-planning-estimation](../model-recommend-fusion-planning-estimation/SKILL.md)）优先
- **排队/争抢（并发场景）**：TP99 劣化但 avg 改善 → 限核或线程数再平衡（[multi-thread-optimization](../model-recommend-multi-thread-optimization/SKILL.md) / [concurrency-strategy-analysis](../model-recommend-concurrency-strategy-analysis/SKILL.md)）

### Step 6: 对比与回修

优化前后同口径对比（同卡/同 batch/同 THREAD_NUM/CORE_NUM/同统计策略）；结论回修 [concurrency-strategy-analysis](../model-recommend-concurrency-strategy-analysis/SKILL.md) 与 [fusion-planning-estimation](../model-recommend-fusion-planning-estimation/SKILL.md) 的估算模型。

**改动性能校验（双门禁性能侧，强制口径）**：

1. **噪声带测定**：对当前基准配置重复测量（≥3 次同口径运行，或复用 `sweep.py` 统计策略的重复数据），取主指标方差确定噪声带（典型 ±2%～±5%）；噪声带内差异视为无变化
2. **主指标判定**：单并发场景看 avg/TP99；吞吐场景看 QPS/TP99——**劣化超过噪声带即回退**；改善小于噪声带不得宣称提升
3. **取舍例外**：局部指标劣化换主指标显著改善，须显式声明、量化两者、经用户确认后记录
4. **算子级改动**必须补整图端到端对比（算子级加速比不等于端到端提升；融合 kernel 引入的调度/布局变化可能抵消算子收益）
5. **结论落账**：对比结果写入 `data/{model_name}/optimization_log/perf_ledger.json`（`perf_before`/`perf_after`/`delta_pct`/`reverted`）

## 昇腾 NPU 专属要点

- **AIC/AIV 双列利用率**：910B 系 AIC/AIV 分离（限核 `CORE_NUM="AIC|AIV"`），bound 判定须分别看两类核的占用，配比失衡是限核调优信号。
- **静态与动态 shape 策略的图分别采集**（编译产物不同，不可混口径）；换配置后首次运行含编译耗时，正式计时在框架 warmup 之后。
- **kernel/墙钟占比判定法**：kernel 总时长（op_summary 聚合）÷ `[GE PROGRESS]` 墙钟时长——是 host/launch-bound 判定的第一手证据。
- GE 编译产物（`kernel_meta` / `fusion_result.json`）同配置保留复用；分析时确认缓存命中状态避免口径污染。
- QPS/TP99/TP999 从 `[GE PROGRESS]` 提取（`iter/Qps/avg session run time/TP99/TP999` 字段）；`sweep.py` 的稳定段统计口径（去首去偏取均、中位数）可复用于自定义对比脚本。

## 产物目录约束

- **文档格式约束（GFM 删除线）**：Markdown 文档禁止裸单波浪线 `~`——同段落/同表格单元内成对出现会被 GFM 渲染为删除线；数值区间用全宽 `～`（`-15%～-30%`）或 `-`/`至`，约数用 `≈`；交付前 grep 确认文档无裸 `~`（详见 [ascend-env-baseline](../model-recommend-ascend-env-baseline/SKILL.md) 全局约束）
- **文档类交付件** → `docs/{model_name}/performance-analysis/`：性能分析报告（三表 + bound 结论 + 优化建议，含采集命令与场景记录）
- **数据类交付件** → `data/{model_name}/performance/`：performance.json、profiling 汇总数据（供回归与优化环节程序化引用）
- **profiling 归档** → `prof/{model_name}/{case}/`：完整 PROF_* 采集目录 + kernel 汇总 csv
- **中间产物** → `tmp/{model_name}/performance-analysis/`：profiler 采集工作目录原始数据、试验性分析脚本输出
- 全局约定与判定标准见 [ascend-env-baseline](../model-recommend-ascend-env-baseline/SKILL.md)；违反目录约定的产物视为未交付

## 验收标准

- 每次采集产出三表 + bound 结论
- profiling 归档完整（PROF_* 完整目录 + kernel 汇总，kernel 事件数 > 0）
- 优化项收益有前后对照数据（同 workload 同口径，含统计策略说明）
- 报告可复现（采集命令与场景记录在案）

## 依据

【框架】`run.sh --prof`、`.cache` 产物治理、`sweep.py` 统计口径与设备占用检测、`[GE PROGRESS]` 指标。
【实测】msprof 可用（CANN 9.1.0）；libsqlite3 历史问题排查路径。
【技能库经验】三表报告方法论、1% 阈值、kernel/墙钟占比判定、source-backed 融合模式分析。
