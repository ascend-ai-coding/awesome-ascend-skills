---
name: external-cannbot-community-model-model-recommend-concurrency-strategy-analysis
description: 单卡并发与限核策略分析（工作流条目 3）。基于算子构成（AIC/AIV 占比）与目标场景（高吞吐/低时延/均衡），推荐 THREAD_NUM
  / CORE_NUM（AIC|AIV）/ GRAPH_MAX_PARALLEL / batch_size 组合及定量依据，sweep.py 实测校准，只分析不改代码。触发：并发策略、线程数、限核、CORE_NUM、图实例并行、QPS
  优化、TP99 约束、吞吐时延权衡、batch 选择。
original-name: model-recommend-concurrency-strategy-analysis
synced-from: https://gitcode.com/cann/cannbot-skills
synced-date: '2026-10-31'
synced-commit: 39919e2c995e1cd9e193f11ac2e3fcb3cbd1d26c
license: UNKNOWN
---

# 单卡并发与限核策略分析

## Prerequisites

- [structure-analysis](../model-recommend-structure-analysis/SKILL.md) 的算子构成分析（AIC/AIV 占比）与显存估算
- [baseline-run](../model-recommend-baseline-run/SKILL.md) 基线数据（时延、HBM 占用）
- 目标场景（高吞吐 QPS / 低时延 TP99 / 均衡）、TP99 上限（若有）、目标 batch

## 目标

基于算子构成与基线数据，输出单卡并发配置推荐（`THREAD_NUM` / `CORE_NUM="AIC|AIV"` / `GRAPH_MAX_PARALLEL` / `batch_size`）及定量依据。**只分析不改代码**（实施由 [multi-thread-optimization](../model-recommend-multi-thread-optimization/SKILL.md) 承接）。

## 输入 / 输出契约

**输入**：算子构成分析；基线数据；目标场景与 TP99 上限；HBM 余量。

> **pytorch / HuggingFace 路径为实验性**：本 skill 目前只有 onnx / GE 侧实现（sweep 校准依赖 GE 侧 `sweep.py`），pytorch 侧需用户提供等价压测脚本。

**输出**：结构化并发配置推荐 + 方案文档（限核配比理由、图实例显存估算、线程收益上限、QPS/TP99 影响分析、sweep 实测校准数据）。

## 工作流步骤（五步强制，禁止跳步拍数）

### Step 1: 提取模型参数与算子构成

复用 [structure-analysis](../model-recommend-structure-analysis/SKILL.md) 产物：AIC/AIV 算子数与占比、单请求计算量（基线 avg 时延）、权重/workspace/激活显存。

### Step 2: 定性分类 → 候选并发形态

逐项回答：

- **图粒度**：单请求时延小（亚毫秒/毫秒级，推荐模型典型）→ 多线程并发提交收益大；时延大（大 batch 重计算）→ 少线程、关注单请求限核资源占用
- **限核配比方向**：AIC 主导（GEMM/Conv 密集）优先保 AIC 核；AIV/查表主导优先保 AIV 核；默认起点 `24|48`
- **batch 与并发的权衡**：大 batch 少线程（摊薄单请求开销、kernel 效率高）vs 小 batch 多线程（时延低、排队灵活）；多 shape 请求场景评估 shape 策略影响（见 [ge-graph-optimization](../model-recommend-ge-graph-optimization/SKILL.md)）
- **图实例并行**：`GRAPH_MAX_PARALLEL` 需覆盖 `THREAD_NUM`（为上限参考，真实并发由 `THREAD_NUM` 决定）；每实例 workspace/激活显存叠加

### Step 3: 定量估算

各候选取值的估算：

- **显存**：权重（共享，一份）+ `GRAPH_MAX_PARALLEL` × (workspace + 激活峰值)，对照 HBM 余量（留安全裕量）
- **线程收益上限**：单请求 kernel 并行度 × 核数上界；`THREAD_NUM` 超过有效并行度后 QPS 不再增长、TP99 劣化
- **限核影响**：压缩 AIC 或 AIV 核数 → 单请求时延上升、多请求并发能力上升；以 QPS×TP99 双指标评估
- **batch 影响**：batch 增大 → 单请求时延超线性/次线性变化（对照基线）、吞吐摊薄

### Step 4: 方案审查 → 输出推荐

- 单套配置兼顾目标场景（QPS 优先时接受 TP99 上浮；TP99 硬上限约束下取最大 QPS）
- 推荐配置必须可被 `run.sh --threads N --cores "A|V" --graph-max-parallel N --batch N` 直接表达
- 输出 2-3 个候选档位（保守/推荐/激进）及适用条件

### Step 5: sweep 实测校准（必做）

`python3 tools/sweep.py <id>` 三阶段扫描，回修估算：

| 阶段 | 内容 | 说明 |
|------|------|------|
| Phase1 限核寻优 | 固定主调核维度（当前实现固定 AIC 全量、黄金分割搜 AIV），`--core-budget` 限评估次数 | 找最优 `CORE_NUM` |
| Phase2 并发寻优 | 固定最优限核，枚举 `--threads`（默认 1..8，失败配置跳过） | 找最优 `THREAD_NUM` |
| Phase3 batch（可选） | `--batches` 给定时 | batch 维度扫描 |

统计口径（框架固定）：每配置重复 3 次；每次 `CircleCount=5000`/`DebugFreq=1000` 取 5 条打点，去首条、去偏差最大条后取平均；三次取中位数。最优判定最高 QPS，可配 TP99 上限。产物 `models/M<id>/sweep/{sweep_<ts>.csv, report_<ts>.md}`。

**重分析触发**：场景 / batch / TP99 上限 / CANN 版本任一变更，重做本分析。

## 昇腾 NPU 专属要点

- **单卡单设备**：`DEVICE_NUM=1` 恒定；`START_DEVICE` 选卡（`TOTAL_DIE` 取模）；`sweep.py --devices` 支持多卡轮询但每卡仍单设备执行（设备占用检测经 npu-smi 进程表）。
- **并发约束（框架强制）**：`THREAD_NUM>1` 仅压测——不做 dump/精度验证/输出落盘；精度验收在单并发做（配置一致即可迁移）。
- `GRAPH_MAX_PARALLEL`（默认 8）是图实例并行上限参考，**不等于真实并发**；超配浪费显存，欠配限制并发。
- 限核语义：`CORE_NUM="AIC|AIV"` 映射 GE `ge.aicoreNum`，控制图执行可用核预算；AIC/AIV 配比按算子构成定，最终以 sweep 实测为准。
- shape 策略联动：`all_static` 多 batch 档位 = 多份编译产物（编译时间与缓存权衡），策略变更须重跑 sweep。

## 产物目录约束

- **文档格式约束（GFM 删除线）**：Markdown 文档禁止裸单波浪线 `~`——同段落/同表格单元内成对出现会被 GFM 渲染为删除线；数值区间用全宽 `～`（`-15%～-30%`）或 `-`/`至`，约数用 `≈`；交付前 grep 确认文档无裸 `~`（详见 [ascend-env-baseline](../model-recommend-ascend-env-baseline/SKILL.md) 全局约束）
- **文档类交付件** → `docs/{model_name}/concurrency-strategy-analysis/`：并发配置方案文档（含每项理由与定量估算、sweep 校准数据）
- **数据类交付件** → `data/{model_name}/concurrency-strategy-analysis/`：结构化推荐配置（json，供实施环节程序化引用）；sweep 结果 csv 同步归档
- **中间产物** → `tmp/{model_name}/concurrency-strategy-analysis/`：候选方案估算草稿、试验性 sweep 日志
- 全局约定与判定标准见 [ascend-env-baseline](../model-recommend-ascend-env-baseline/SKILL.md)；违反目录约定的产物视为未交付

## 验收标准

- 每个参数取值有定量依据（显存装得下 + QPS/TP99 达标 + sweep 实测支撑）
- 估算与 sweep 实测对照回修（偏差说明）
- 推荐配置可被 `run.sh` 参数直接表达
- 档位（保守/推荐/激进）与适用条件明确

## 依据

【框架】`sweep.py` 三阶段与统计口径、`run.sh --threads/--cores/--graph-max-parallel/--batch`、并发运行约束、设备占用检测。
【实测】基线时延/HBM 数据、sweep 结果。
【技能库经验】四步分析范式（定性分类 → 定量估算 → 方案审查 → 实测校准，禁止跳步拍数）。
