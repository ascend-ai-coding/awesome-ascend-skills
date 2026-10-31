---
name: external-cannbot-community-model-model-recommend-regression-baseline-mgmt
description: 优化回归验收与基线管理（工作流条目 14，收口环节）。统一基线标准化、优化项台账、逐项验收门禁（精度+性能+功能一致三关）、变更触发重分析、经验沉淀、perf_ledger、最终优化报告与下一步优化动作表（优化手段/预期收益/对应工作流条目）。触发：回归验收、基线管理、优化项台账、性能归因、收益归属、优化开关管理、经验沉淀、最终优化报告、下一步优化动作、下轮优化规划。
original-name: model-recommend-regression-baseline-mgmt
synced-from: https://gitcode.com/cann/cannbot-skills
synced-date: '2026-10-31'
synced-commit: 39919e2c995e1cd9e193f11ac2e3fcb3cbd1d26c
license: UNKNOWN
---

# 优化回归验收与基线管理

## Prerequisites

- 至少一个已采集的基线（[baseline-run](../model-recommend-baseline-run/SKILL.md) 产物）
- 优化项清单（shape 策略 / 编译选项 / onnx 改写 / 融合 / 并发限核 / 访存 / 自定义算子）

## 目标

建立统一基线与逐项验收机制，管理所有优化开关。**多优化项叠加时必须逐项可归因、可回退、可复现**，否则收益归属不清、故障无法二分。

## 机制设计

### 1. 基线标准化

- 每阶段产出 `data/{model_name}/baseline/baseline_metadata.json`（配置 + 输出摘要 + 性能数据）
- 单进程执行（单卡单设备），无多卡 rank 日志规则；基线采集命令与场景（batch/THREAD_NUM/CORE_NUM/CircleCount/统计策略）记录在案，保证可复现
- perf_ledger.json 持续维护（更新时机：基线首写、每个优化项追加、收口汇总）

### 2. 优化项台账

每个优化项登记：

| 字段 | 说明 |
|------|------|
| 开关名/参数 | 如 `shape_mode=all_static`、`THREAD_NUM=4`、`patched_model.onnx`、算子包部署 |
| 默认状态 | 启用/关闭 |
| 依赖前置项 | 如：算子替换依赖自定义算子包部署（`M<id>_env.sh`） |
| 验收数据 | 精度 + 性能 + 功能一致三关记录 |
| 回退路径 | 关闭参数/换回原始 onnx/卸载算子包后如何还原 |

**全局约定：所有优化必须可开关与可回退**（shape 策略可切回、改写 onnx 保留原始版本、并发/限核参数可还原、自定义算子可卸载）。

### 3. 逐项验收门禁（双门禁的执行与台账化）

**每笔对模型执行的改动必须同时过双门禁才可合入**（全局定义见 [ascend-env-baseline](../model-recommend-ascend-env-baseline/SKILL.md)；主文档 `workflows/recommend-infer-workflow.md`（在插件仓 recommend-infer-optimize 内）「全局强制门禁」）：

1. **精度门禁**（[precision-comparison](../model-recommend-precision-comparison/SKILL.md)：串行 verify 全 pass；精度相关选项附误差指标）
2. **性能门禁**（[performance-analysis](../model-recommend-performance-analysis/SKILL.md) 同口径对比：主指标不劣化超过噪声带；改善小于噪声带不得宣称提升；取舍项显式声明并经用户确认）
3. **功能一致**（输出 shape 与数值分布合理、`output_shapes.json` 与预期一致）

三关全过 → 合入并写 perf_ledger（`reverted=false`）；任一不过 → 回退并记录原因（`reverted=true`）。**台账中不存在"未过门禁但已合入"的项。**

### 4. 变更触发重分析

| 变更 | 动作 |
|------|------|
| 场景 / batch / TP99 上限 | 重做 [concurrency-strategy-analysis](../model-recommend-concurrency-strategy-analysis/SKILL.md) |
| CANN / 框架版本 | 重采全量基线 |
| onnx / 配置变更 | 重跑受影响优化项验收 |
| shape 策略切换 | 重跑精度门禁 + 相关性能对比（编译产物不同） |

### 5. 经验沉淀

新算子约束、onnx 改写等价性陷阱、排错案例按模板回写对应 skill（如 [fusion-operator-adaptation](../model-recommend-fusion-operator-adaptation/SKILL.md) 高频适配点表、[runtime-debug](../model-recommend-runtime-debug/SKILL.md) 直达路径表）。

### 6. 下一步优化动作表（收口必产出，强制交付件）

所有阶段执行完毕后，产出**下一步优化动作建议表**，作为下一轮优化的执行入口（表格式与字段要求见主文档 `workflows/recommend-infer-workflow.md`（在插件仓 recommend-infer-optimize 内）「全局强制输出要求」第 3 项）：

```
| 优先级 | 优化动作 | 优化手段类型 | 依据 | 预期收益（保守～乐观） | 置信度 | 对应工作流 | 前置依赖 |
|---|---|---|---|---|---|---|---|
```

- **字段纪律**：动作具体可执行（禁止空话）；预期收益量化到主指标口径（avg/TP99/QPS）并注明推导方法；不得与已实施项收益重复计算；对应工作流必须给出条目编号 + 承接 skill（下一轮从该条目直接进入，不重跑全流程）
- **动作来源六类（逐类核对，缺失须说明）**：
  1. [fusion-planning-estimation](../model-recommend-fusion-planning-estimation/SKILL.md) 候选清单中未进入 Top-N / 未实施的开发项
  2. perf_ledger 与台账中的回退项（附重试条件：精度修复后、依赖就绪后、环境变更后）
  3. 各条目分析识别但未实施的机会（未覆盖热点、sweep 未扫档位、门禁未达标项的瓶颈改进方向）
  4. **[profiling-deep-analysis](../model-recommend-profiling-deep-analysis/SKILL.md) 专项发现**（算子执行效率 / 多流掩盖 / H2D-D2H / host 瓶颈四类 findings，逐项按其 schema 并入）
  5. [fusion-operator-adaptation](../model-recommend-fusion-operator-adaptation/SKILL.md) 未适配项清单中未解决项
  6. 遗留风险与框架侧需求（标注"移交"，不入本工作流执行队列）
- **循环消费方**：进入持续优化循环（[optimization-loop](../model-recommend-optimization-loop/SKILL.md)）后，本表是每轮 L1 动作选择的唯一输入；每轮 L7 更新本表（完成项移除、新发现并入、回退项附重试条件），轮次状态由 `data/{model_name}/optimization_log/loop_state.json` 管理
- **优先级**：P0（下一轮立即执行）/ P1（下轮排队）/ P2（条件触发，注明触发条件）；排序依据 = 预期收益 × 置信度 ÷ 实施成本
- **质量要求**：每个动作可独立执行并过「改动校验双门禁」；空表必须给出理由（已知机会已穷尽 / 数据不足需先补采）；与报告"未实施项与遗留风险"章节交叉一致（不得漏项）
- **落盘**：报告章节 → `docs/{model_name}/final_optimization_report.md`；机器可读版本 → `data/{model_name}/regression-baseline-mgmt/next_actions.json`（字段与表一致，供下一轮程序化消费）

## 收口交付：最终优化报告与下一步优化动作表（强制）

条目 14 收口时产出 `docs/{model_name}/final_optimization_report.md`，格式与字段见主工作流文档（`workflows/recommend-infer-workflow.md`（在插件仓 recommend-infer-optimize 内））「全局强制输出要求」：优化手段总表、逐步性能记录表（perf_ledger 汇总）、累计性能收益（时延 avg/TP99/QPS/HBM 峰值）、各项贡献分解、精度总结、未实施项与遗留风险、环境信息；**并同章产出「下一步优化动作表」**（格式见机制设计第 6 项），机器可读版本落 `data/{model_name}/regression-baseline-mgmt/next_actions.json`。

**循环期间维护（与 [optimization-loop](../model-recommend-optimization-loop/SKILL.md) L7 联动）**：进入循环后，最终报告的「循环轨迹」表由每轮 L7 **增量追加**（轮次/动作/条目/before/after/收益/落地/计数/交付件指针）；收口时只补累计收益与结论——**禁止收口时事后批量补写轨迹**。

## 产物目录约束

- **文档格式约束（GFM 删除线）**：Markdown 文档禁止裸单波浪线 `~`——同段落/同表格单元内成对出现会被 GFM 渲染为删除线；数值区间用全宽 `～`（`-15%～-30%`）或 `-`/`至`，约数用 `≈`；交付前 grep 确认文档无裸 `~`（详见 [ascend-env-baseline](../model-recommend-ascend-env-baseline/SKILL.md) 全局约束）
- **文档类交付件** → `docs/{model_name}/regression-baseline-mgmt/`：优化项台账、逐项验收报告、`docs/{model_name}/final_optimization_report.md`（含下一步优化动作表章节）
- **数据类交付件** → `data/{model_name}/baseline/`（各阶段基线）、`data/{model_name}/optimization_log/perf_ledger.json`、`data/{model_name}/regression-baseline-mgmt/`（回归对比数据、**next_actions.json 下一步优化动作表机器可读版**）
- **中间产物** → `tmp/{model_name}/regression-baseline-mgmt/`：重跑临时日志、回归中间数据
- 全局约定与判定标准见 [ascend-env-baseline](../model-recommend-ascend-env-baseline/SKILL.md)；违反目录约定的产物视为未交付

## 验收标准

- 任意优化项可单独开启/关闭并复现其收益数据
- **每笔改动均有双门禁记录**（精度结论 + 同口径性能对比 + 噪声带 + 回退状态），台账无"未过门禁已合入"项
- 最终配置 = 台账中已验收项的组合
- 全量回归一次通过
- **perf_ledger.json 完整覆盖所有已执行条目；final_optimization_report.md 已产出**（含累计收益与贡献分解）
- **下一步优化动作表已产出**：每个动作有量化预期收益、置信度、对应工作流条目与 skill；六类来源逐类核对；空表有理由说明；`next_actions.json` 与报告表一致

## 依据

【框架】基线与产物结构、运行参数可回退性、串行/并发分离口径。
【技能库经验】台账/门禁/变更触发方法论、变更可归因思想、回归验证体系。
