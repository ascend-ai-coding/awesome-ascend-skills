---
name: external-cannbot-community-model-model-recommend-multi-thread-optimization
description: 多线程与图并行优化（工作流条目 8）。GE Session 多线程并发提交（RunGraphAsync × THREAD_NUM）、图实例并行（GRAPH_MAX_PARALLEL）与限核（CORE_NUM
  AIC|AIV）的实施与验收，sweep.py 三阶段寻优，提升 NPU 推理性能（QPS/TP99）。触发：多线程优化、并发压测、图并行、QPS 优化、TP99
  调优、限核实施、sweep 扫描、吞吐优化、性能调优。
original-name: model-recommend-multi-thread-optimization
synced-from: https://gitcode.com/cann/cannbot-skills
synced-date: '2026-10-31'
synced-commit: 39919e2c995e1cd9e193f11ac2e3fcb3cbd1d26c
license: UNKNOWN
---

# 多线程与图并行优化

## Prerequisites

- [concurrency-strategy-analysis](../model-recommend-concurrency-strategy-analysis/SKILL.md) 的推荐配置（THREAD_NUM / CORE_NUM / GRAPH_MAX_PARALLEL / batch）
- [performance-analysis](../model-recommend-performance-analysis/SKILL.md) 三表报告（overlap 机会表）
- 基线数据（单并发时延 + 目标场景 TP99 上限）

## 目标

按推荐配置实施并发/限核组合，用 sweep 与 profiling 验收 QPS/TP99 收益。

**重要性**：推荐模型单请求计算量通常不大，**QPS 主靠并发**：`THREAD_NUM`（多线程并发提交 `RunGraphAsync`）× `GRAPH_MAX_PARALLEL`（图实例并行上限）× `CORE_NUM`（限核）三者组合是吞吐优化主要手段。

## 输入 / 输出契约

**输入**：并发策略推荐 + overlap 机会表 + 基线数据。

**输出**：`docs/{model_name}/multi-thread-optimization/multi-thread-analysis.md` 分析报告 + 验收数据（QPS/TP99/TP999 开关前后对比）+ 最终部署配置。

## 工作流步骤

### Step 1: 整网分析（先分析后实施）

1. 请求间独立性确认：推荐模型请求天然独立（无 LLM 序列依赖），无需分析请求级依赖；关注图内资源画像
2. 图内 AIC/AIV 资源画像：kernel 表的 AIC/AIV 时长分布 → 限核配比依据（复用 [performance-analysis](../model-recommend-performance-analysis/SKILL.md)）
3. 结果统一落到 `docs/{model_name}/multi-thread-optimization/multi-thread-analysis.md`

### Step 2: 配置实施

- 按推荐档位（保守/推荐/激进）逐一验证：`run.sh <id> --threads N --cores "A|V" --graph-max-parallel N [--batch N]`
- 或全量扫描：`python3 tools/sweep.py <id>` 三阶段（限核黄金分割 → 并发 1..8 → 可选 batch）
- 显存校验：并发图实例叠加后 HBM 不越界（对照 [structure-analysis](../model-recommend-structure-analysis/SKILL.md) 估算与 npu-smi 实测）

### Step 3: 验收测量

- 每配置取 sweep 统计口径结果（repeat=3、去首去偏取均、中位数）
- 记录 QPS / TP99 / TP999 / avg；对比单并发基线与推荐档位
- 资源争抢信号识别：QPS 增长放缓但 TP99 陡增 → 线程数过头或限核失衡，回调后复测

### Step 4: 结果验证与收口

- 并发下统计稳定性（无异常尖刺、无失败请求；框架 fail-fast 下任一失败即终止，失败配置如实记录）
- 最终部署配置写入 `data/{model_name}/multi-thread-optimization/`（结构化，供回归引用）

## 重要原则

- **参数天然可回退**：`--threads 1 --cores "24|48" --graph-max-parallel 8` 即还原默认；不引入不可回退的改动
- **改动校验双门禁（强制）**：并发/限核/图并行参数是对模型执行的改动——落地前必须①**精度门禁**：该配置组合下串行 verify 全 pass（并发参数理论上不改数值，仍须实测确认一次）；②**性能门禁**：目标场景口径（QPS/TP99）同口径对比，**不劣化超过噪声带**（相对上一已验收配置；相对单并发基线允许 TP99 上浮但 QPS 必须有收益，取舍显式声明并经用户确认）；任一不通过即回退该配置；详见 [ascend-env-baseline](../model-recommend-ascend-env-baseline/SKILL.md) 全局约束
- **先正确再性能**：先单并发 verify 全 pass，再上调并发
- **并发 ≠ 收益**：出现资源争抢、拖尾、TP99 劣化时，回到限核与线程数再平衡，并回修 [concurrency-strategy-analysis](../model-recommend-concurrency-strategy-analysis/SKILL.md)
- **不混用口径**：单并发看时延、并发看吞吐，两套指标分别记录，不交叉对比

## 昇腾 NPU 专属要点

- **框架约束**：并发（`THREAD_NUM>1`）仅做性能压测——不做 dump、不做精度验证、输出不落盘；`--verify` 在并发下报错
- `GRAPH_MAX_PARALLEL`（默认 8）是图实例并行**上限参考**，真实并发由 `THREAD_NUM` 决定；图实例显存随并行度叠加
- 限核 `CORE_NUM="AIC|AIV"`（默认 `24|48`）按算子构成配比（AIC 主导保 AIC，AIV/查表主导保 AIV），sweep Phase1 黄金分割搜 AIV（当前实现固定 AIC 全量）
- `MaxQPS` 可做限速压测；`START_DEVICE` 选卡；设备占用检测避免与他任务冲突（npu-smi 进程表）
- 多线程 fail-fast：任一线程失败置 stop 标志全线程退出（框架行为），失败配置直接淘汰

## 产物目录约束

- **文档格式约束（GFM 删除线）**：Markdown 文档禁止裸单波浪线 `~`——同段落/同表格单元内成对出现会被 GFM 渲染为删除线；数值区间用全宽 `～`（`-15%～-30%`）或 `-`/`至`，约数用 `≈`；交付前 grep 确认文档无裸 `~`（详见 [ascend-env-baseline](../model-recommend-ascend-env-baseline/SKILL.md) 全局约束）
- **文档类交付件** → `docs/{model_name}/multi-thread-optimization/`：multi-thread-analysis 分析报告（含资源画像、配置对比、争抢识别过程）
- **数据类交付件** → `data/{model_name}/multi-thread-optimization/`：开/关性能验收对比数据、最终部署配置（json）
- **中间产物** → `tmp/{model_name}/multi-thread-optimization/`：sweep 原始日志（`models/M<id>/sweep/logs/` 同步）、试验性配置数据
- 全局约定与判定标准见 [ascend-env-baseline](../model-recommend-ascend-env-baseline/SKILL.md)；违反目录约定的产物视为未交付

## 验收标准

- 开/关对比 QPS 与 TP99 有收益数据（同口径、统计策略与噪声带在案）
- **双门禁通过**：配置组合串行 verify 全 pass；目标场景 QPS/TP99 不劣化超过噪声带（取舍项已显式声明并经确认）；通过的配置已写 perf_ledger
- 并发下结果统计稳定（无异常尖刺、无失败配置未记录）
- 显存不越界（并发图实例叠加有实测数据）
- 最终配置可被 `run.sh` 参数直接表达并已结构化归档
- 报告落盘 `docs/{model_name}/multi-thread-optimization/multi-thread-analysis.md`

## 依据

【框架】`run.sh --threads/--cores/--graph-max-parallel`、`sweep.py` 三阶段与统计口径、并发运行约束（不 dump/不验证/不落盘）、fail-fast、设备占用检测。
【实测】sweep 结果与 npu-smi HBM 数据。
【技能库经验】先正确再性能、overlap 不等于收益、两套口径分离。
