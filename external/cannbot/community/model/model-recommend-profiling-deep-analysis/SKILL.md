---
name: external-cannbot-community-model-model-recommend-profiling-deep-analysis
description: Profiling 深度分析（工作流条目 5 的四项专项）。基于 msprof 采集数据的四项专项分析：算子执行时间分析（类型/AIC-AIV
  分列/碎片化/单 kernel 效率）、多流掩盖分析（流时间线重叠度/串行化识别）、H2D-D2H 通信分析（InputCopy/OutputCopy 固定开销/关键路径占比）、host
  bound 时的 host 瓶颈分析（API 耗时/launch 开销/同步等待）。产出结构化优化点并入下一步优化动作表。触发：profiling 深度分析、多流分析、流重叠、H2D、D2H、拷贝开销、host
  瓶颈、launch 开销、API 分析、算子效率、AIC AIV 利用率。
original-name: model-recommend-profiling-deep-analysis
synced-from: https://gitcode.com/cann/cannbot-skills
synced-date: '2026-10-31'
synced-commit: 39919e2c995e1cd9e193f11ac2e3fcb3cbd1d26c
license: UNKNOWN
---

# Profiling 深度分析（四项专项）

## 流水位置

[performance-analysis](../model-recommend-performance-analysis/SKILL.md)（条目 5 三表 + bound 判定）的**深化环节**（条目 5 的四项专项）：三表回答"瓶颈在哪一类"，本 skill 回答"具体是什么、可优化空间多大"。是「持续优化循环」（[optimization-loop](../model-recommend-optimization-loop/SKILL.md)）每轮的固定分析动作，结果直接并入下一步优化动作表。

## Prerequisites

- 同口径 profiling 归档（`--prof msprof` 产出，`prof/{model_name}/{case}/`，含 `mindstudio_profiler_output/`）
- 实测锚定数据（`data/env-baseline/anchor.json`：TFLOPS / 带宽）
- 墙钟指标（`[GE PROGRESS]` 的 avg/TP99/QPS，同场景）
- 迭代编号（循环轮次 k；首轮基线记 iter0）

## 目标

四项专项分析，每项产出**结构化优化点**（发现 → 证据 → 建议 → 预期收益 → 对应工作流条目），供下一步优化动作表合并排序。

## 数据源映射（msprof mindstudio_profiler_output）

> onnx 路径用 msprof 的 `mindstudio_profiler_output`；pytorch/HF 路径用 `torch_npu.profiler` 导出的 `kernel_details` / `trace_view`，字段按下表同义映射（算子耗时、流、拷贝、host API）后再做四项专项。

| 数据 | 文件 | 关键列 |
|------|------|--------|
| 逐 kernel 事件 | `op_summary_*.csv` | Op Name / OP Type / **Stream ID** / **Task Type**（AI_CORE=Cube / AI_VECTOR_CORE=Vector）/ Task Start Time(us) / Task Duration(us) |
| host API 统计 | `api_statistic_*.csv` | API Name / Time(us) / Count / Avg / Min / Max |
| 融合结果 | `fusion_op_*.csv` | Fusion Op / Original Ops |
| 阶段分解 | `step_trace_*.csv` / `task_time_*.csv` | stage / 耗时 |
| 设备状态 | `npu-smi info`（采样） | CPU / AICore / HBM |

## 输入 / 输出契约

**输入**：profiling 归档 + 锚定数据 + 墙钟指标 + 轮次编号。

**输出**：
- `data/{model_name}/profiling-deep-analysis/iter{k}/findings.json`：结构化优化点（每项含 `id`/`category`/`finding`/`evidence`/`suggestion`/`expected_gain`/`confidence`/`workflow_entry`）
- `docs/{model_name}/profiling-deep-analysis/iter{k}/deep_analysis_report.md`：四项分析报告（含图表数据与结论）
- 优化点**必须**同步并入 `next_actions.json`（[regression-baseline-mgmt](../model-recommend-regression-baseline-mgmt/SKILL.md) 动作表来源④）

## 工作流步骤（四项专项，全部必做）

### Module A: 算子执行时间分析

1. **类型聚合**：按 OP Type 聚合（count / total_us / avg_us，单次推理口径 = 总量 ÷ 迭代数）
2. **AIC/AIV 分列**：按 Task Type 分列（AI_CORE / AI_VECTOR_CORE）——核型利用率配比与限核建议的直读依据
3. **Top-N 与碎片化**：Top-N kernel 占比（累计 ≥1% 阈值）；kernel 时长分布（<5us / 5-20us / ≥20us 计数与时长占比）；单次推理 kernel 总数与均值
4. **单 kernel 效率（Roofline）**：对 Top-N 大 kernel 用图 shape 推导 FLOPs 与搬运字节数，算有效 TFLOPS / 有效带宽，对照锚定（偏差大 = tiling/实现退化或固定开销主导）
5. **重复与冗余识别**：同型 kernel 大量重复（如 TransData/Cast/控制类）→ 融合或消减候选

### Module B: 多流掩盖分析

1. **流分布**：op_summary 按 Stream ID 聚合（kernel 数 / 时长）——**单流串行 vs 多流并行**是第一判定
2. **流时间线重叠度**：按 Task Start Time + Duration 构建各流时间线，计算：
   - 并行窗口占比（≥2 流同时有 kernel 的时间 / 墙钟）
   - 空闲间隙占比（所有流均无 kernel 的时间 / 墙钟）
   - 理论重叠收益：若 AIC 段与 AIV 段（或独立链）可分流，上限 = `sum(各段) - max(各段)`
3. **串行化链识别**：访存类 kernel（TransData）与计算 kernel 是否同流串行；独立子链（无数据依赖的分支）是否被排到同一流
4. **并发实例口径**（THREAD_NUM>1 采集时）：各图实例的流归属与争抢表现（TP99 劣化 vs avg 改善）
5. 优化方向映射：GE 图内分流（编译选项/图结构）、多实例并发（[multi-thread-optimization](../model-recommend-multi-thread-optimization/SKILL.md)）、算子融合消转换（[fusion-operator-adaptation](../model-recommend-fusion-operator-adaptation/SKILL.md)）

### Module C: H2D/D2H 通信分析

1. **拷贝 API 分解**：api_statistic 中 `InputCopy` / `OutputCopy` / `aclrtMemcpy`（及 MemcpySync 类）的 Count / Avg / Total，换算单次推理口径
2. **固定开销判定**：实际耗时 vs 理论搬运时间（字节量 ÷ 锚定带宽）——**比值远大于 1 判定固定开销主导**（小包同步拷贝典型）
3. **关键路径占比**：`H2D + kernels + D2H` 对账墙钟（三段和 ≈ ModelExecute 耗时即证串行在关键路径）；H2D/D2H 占比 = 拷贝耗时 / 墙钟
4. **重叠情况**：拷贝是否与 kernel 执行重叠（时间线对齐检查；未重叠 = 流水化机会）
5. 优化方向映射：输入常驻/批量/异步重叠（框架需求，移交）、减少拷贝次数（图结构）、并发下拷贝与计算的流水（[multi-thread-optimization](../model-recommend-multi-thread-optimization/SKILL.md)）

### Module D: host 瓶颈分析（条件触发：kernel/墙钟 < 80%）

> kernel/墙钟 ≥ 80% 时本模块仅做记录性检查（结论"非 host bound"），不展开。

1. **缺口量化**：墙钟 −（kernel 总时长 + 拷贝耗时）= host 侧空转时间；占比
2. **API 耗时分布**：api_statistic 按 Total 排序的 host API（剔除同步等待类的阻塞时间）——`launch` 次数 × 均值、`RunGraphAsync` 分解（API 本体 vs InputCopy）、SetDevice/MapMem 等杂项
3. **CPU 佐证**：npu-smi 的 CPU 占用采样（host-bound 时单核打满常见）
4. **同步等待分解**：StreamSynchronize/ModelExecute 阻塞时长 vs 真正 host 处理时长
5. 优化方向映射：减少 host 侧处理、拷贝/launch 异步化、并发提交掩盖（[multi-thread-optimization](../model-recommend-multi-thread-optimization/SKILL.md)）；GE 调度类问题记框架需求

### 汇总：优化点输出（强制）

每个发现按统一 schema 落 `findings.json`，并**逐项并入 next_actions**（去重：与既有动作重复的标注 `merged_into_existing`，不重复占位）：

```json
{
  "id": "PD-<iter>-<序号>",
  "category": "op-exec | stream-overlap | h2d-d2h | host-bottleneck",
  "finding": "具体发现（如：259 kernel 全部单流串行执行）",
  "evidence": {"metric": "值", "source": "数据文件/列"},
  "suggestion": "优化建议（具体可执行）",
  "expected_gain": "量化区间（换算到 avg/TP99/QPS 口径）+ 推导方法",
  "confidence": "高（实测外推）/ 中（理论估算）/ 低（结构推测）",
  "workflow_entry": "条目编号 + 承接 skill",
  "merged_into_next_actions": true
}
```

## 昇腾 NPU 专属要点

- **Task Type 直读核型**：AI_CORE（Cube）/ AI_VECTOR_CORE（Vector）分列是 910B 系 AIC/AIV 分离架构的利用率真相；两列时长比对照限核配比（`CORE_NUM="AIC|AIV"`）
- **GE 单流默认行为**：GE 图编译默认将整图 kernel 排到单一计算流（实例级并发由 `THREAD_NUM` 提供多流）；**单请求时延优化受单流串行约束**，流内收益只能来自减少 kernel 数/时长
- **kOnHost 输入的拷贝语义**：GE_SESSION_DEMO 输入以 host 张量传入，GE 内部执行 H2D（`InputCopy`）——每次请求一次同步拷贝；输出 D2H 同理（`OutputCopy`）
- 同步类 API（StreamSynchronize/ModelExecute）的耗时是**阻塞等待**，不计入 host 开销，须与真正 host 处理区分
- 多任务共享环境：npu-smi CPU/AICore 采样时注意排除他进程干扰（进程表核对）

## 产物目录约束

- **文档格式约束（GFM 删除线）**：Markdown 文档禁止裸单波浪线 `~`——数值区间用全宽 `～` 或 `-`/`至`，约数用 `≈`；交付前 grep 确认无裸 `~`（详见 [ascend-env-baseline](../model-recommend-ascend-env-baseline/SKILL.md) 全局约束）
- **文档类交付件** → `docs/{model_name}/profiling-deep-analysis/iter{k}/`：deep_analysis_report.md（四模块结论 + 优化点清单）
- **数据类交付件** → `data/{model_name}/profiling-deep-analysis/iter{k}/`：findings.json（供动作表程序化合并）
- **中间产物** → `tmp/{model_name}/profiling-deep-analysis/`：分析脚本与中间提取数据
- 全局约定见 [ascend-env-baseline](../model-recommend-ascend-env-baseline/SKILL.md)；违反目录约定的产物视为未交付

## 验收标准

- 四项模块全部执行（host-bound 未触发时给出判定依据而非跳过）
- 每个优化点有量化证据与预期收益区间，schema 字段齐全
- **findings.json 已并入 next_actions.json**（去重标注在案；无新增时显式说明）
- 报告含数据对账（kernels + H2D/D2H + host 缺口 ≈ 墙钟，误差说明）
- 分析可复现（脚本入 tmp/，数据文件与轮次在案）

## 依据

【实测】M100 iter0 验证：op_summary 的 Stream ID/Task Type 列可用（单流判定、AIC/AIV 分列）；api_statistic 的 InputCopy/OutputCopy/aclrtMemcpy/launch/RunGraphAsync 可量化（H2D 81us/次、D2H 64us/次、launch 18.65us×260）。
【框架】kOnHost 输入放置与 GE 内部 H2D 语义、`--prof msprof` 采集口径。
【技能库经验】Roofline 单 kernel 效率判定、流时间线重叠度方法、固定开销判定（实际/理论比值）。
