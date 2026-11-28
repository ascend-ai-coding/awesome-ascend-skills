# 模型交付说明书（最终交付）

## 何时用 / 边界

- 用于优化收口后的最终交付；输入必须已包含适配、分析、优化记录、门禁与最终配置。
- 本 skill 是 `model_delivery_manual.md` 的唯一写者。
- 只写交付文档，不重新采集、不重新分析、不实施优化。
- 由 `model-recommend-optimization-loop` 在 Phase 8 加载；单独调用时，缺失产物需先补采或向用户确认。
- 不修改原始 onnx、demo 推理/运行逻辑与交付件。

## 输入

| 输入 | 用途 |
| --- | --- |
| `adaptation_status.json` / 适配报告 | 形态、边界、覆盖矩阵、IO、输入来源、功能验证 |
| `optimization_status.json` / `optimization_log.md` | 轮次、动作、门禁、回退、终止 |
| `optimization_suggestions.json` / 分析报告 | 建议、证据、收益上界、暂缓/越界 |
| `best_ledger.json` / `stress_rounds` | 绑核、CORE_NUM、THREAD_NUM、HF32 前置门禁 |
| baseline / profiling / dump_graph / golden | 性能与精度证据 |
| 最终配置与复现命令 | 环境变量、batch/threads/cores、精度模式 |

## 输出

- **唯一最终交付文件**：`models/M<id>/model_delivery_manual.md`（UTF-8 Markdown）。
- 过程产物不单独作为最终交付，作为附录引用。
- 每个结论必须可追溯到状态/台账/证据；禁止只写结论不写依据。

## 标准结构

### 必填章节

| 章节 | 必填 | 数据来源 |
| --- | --- | --- |
| 0 文档信息 | 是 | 人工 + status |
| 1 执行摘要 | 是 | status/log/ledger |
| 2 模型与环境 | 是 | 输入清单/环境记录 |
| 3 适配结论 | 是 | `adaptation_status.json` |
| 4 优化项清单 | 是 | suggestions/log/applied |
| 5 性能结果 | 是 | baseline/profiling/ledger |
| 6 逐项收益与门禁 | 是 | log/gate |
| 7 最终配置与复现 | 是 | 最终配置/命令 |
| 8 精度与风险 | 是 | golden/verify/风险记录 |
| 9 未实施/暂缓/越界 | 是 | deferred/out_of_scope |
| 10 附录与证据 | 是 | 产物清单 |

### 0 文档信息

- 标题、模型 ID、文档版本、状态（draft/final）、日期、编写者/负责人。
- 适用范围与约束（设备、batch、精度、不可改动项）。
- 修订记录（版本、日期、变更）。

### 1 执行摘要

- 最终指标 vs 基线：avg/TP99/TP999/QPS。
- 已实施/回退/暂缓项数量。
- 精度结论与发布建议（release / hold）。
- 一行复现命令与最终配置入口。

### 2 模型与环境

- 模型路径 + SHA256；权重路径 + SHA256；opset/IR 版本；输入 shape 模式。
- CANN、驱动/固件、OS、Python、芯片型号、die、HBM、GE/demo 版本。
- 设备共享情况与测试窗口。

### 3 适配结论

- 形态（全量/主图/子图/副本）与边界。
- 覆盖矩阵摘要：已覆盖/可替代/缺口/不需要。
- IO 契约与输入来源级别（real/staged/full-model/synthetic）。
- 功能验证结论：输出数量/顺序/形状/有限性、golden、容差。
- 缺口处理结果与承接方。

### 4 优化项清单

| id | 类型 | 改了什么 | 开关/默认值 | 产物路径 + SHA256 | 精度结果 | 性能结果 | 状态 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| OPT-xxxx | pass/plugin/custom-op/runtime_config |  |  |  | pass/fail/not-run | improved/no-change/regressed | applied/reverted/deferred |

- `runtime_config`（如 HF32）必须写实际值（如 `ENABLE_HF32=01`）、`matmul_ratio`、`avg_gain_pct`、开/关决策。
- 精度模式（fp16/bf16 等）如实施，单独列出并标注“最终可选”。

### 5 性能结果

- 测量口径：3 次中位数、2/4–3/4 窗口、首迭代处理、噪声带、离群点处理。
- 测试条件：设备、batch、threads、cores、CORE_NUM、HF32 状态、精度模式、Pass/插件/算子版本。
- 逐阶段表：

| 阶段 | 配置摘要 | avg(ms) | TP99(ms) | TP999(ms) | QPS | HBM | 相对基线 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 开箱基线 |  |  |  |  |  |  | — |
| 优化项 1 |  |  |  |  |  |  |  |
| … |  |  |  |  |  |  |  |
| 最终 |  |  |  |  |  |  |  |

- HF32 前置门禁：附 off/on A/B、`avg_gain_pct`、精度结果与最终开/关决策。
- 跨模型/版本对比只比结构比例与同类算子单位成本，并记录版本。

### 6 逐项收益与门禁

| id | before | after | delta | 精度门禁 | 性能门禁 | 噪声带 | 结论 | 回退 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
|  |  |  |  | pass/fail | improved/no-change/regressed |  | keep/revert | yes/no |
| HF32 门禁 | avg_off | avg_on | avg_gain_pct | pass/fail | >5% 且 >噪声带 |  | on/off | — |

### 7 最终配置与复现

- 环境变量：`ENABLE_HF32`、`CORE_NUM`、`THREAD_NUM`、`START_DEVICE`、`CircleCount`、`DebugFreq` 等实际值。
- batch/threads/cores/精度模式/绑核命令。
- 模型/权重 SHA256；配置/Pass/插件/算子包版本与 SHA256。
- 完整命令与产物路径。
- 复现清单：

```text
模型：路径 + SHA256；权重：路径 + SHA256
输入：数据路径/生成方式 + SHA256；shape 模式
环境：CANN、驱动/固件、OS、Python
设备：die、core、HBM、共享情况
配置：config.ini、model env、PARSER_*、KEEP_*、Pass/插件/算子包版本与 SHA256
命令：完整运行/采集命令、circle、threads、cores、profiling 方式
性能：3×N 单次结果、中段窗口、QPS/avg/TP99/TP999、离群点处理
图：dump_graph 阶段、关键 op 计数、Pass effect/match
精度：输出数量/顺序/shape/dtype、golden、容差、比对结果
```

### 8 精度与风险

- golden 来源、容差、验证命令与结论。
- HF32 精度：`verify`、`max_abs_err`、`max_rel_err`、必要时 `cosine`；失败或收益不足的回退条件。
- 精度模式（fp16/bf16 等）的实际影响与回退条件。
- NaN/Inf、分布偏移、长尾、稳定性风险。
- 版本/环境风险与规避方式。

### 9 未实施/暂缓/越界

| id | 类型 | 原因码 | 证据 | 决策人 | 后续条件 |
| --- | --- | --- | --- | --- | --- |
| DEF/OOS-xxxx | deferred/out_of_scope | noise/roi/… |  | 用户/负责人 |  |

### 10 附录与证据

- 产物清单：适配/分析/状态/优化日志/压测/Profiling/dump_graph/日志，含路径、采集时间、SHA256。
- 决策记录与用户确认记录。
- 可选的失败与回退记录。

## 交付验收清单

```text
[ ] 10 个必填章节齐全
[ ] 最终指标可追溯到 baseline/profiling/ledger
[ ] 每个 applied 项都有精度 + 性能门禁记录
[ ] reverted/deferred 项有原因与证据
[ ] HF32 有 matmul_ratio、--verify、off/on A/B、avg_gain_pct 与开/关决策
[ ] 精度模式（如实施）有实际值与精度结果
[ ] 最终配置可复现（命令 + 环境变量 + SHA256）
[ ] 未实施/越界项与风险已列
[ ] 用户确认记录齐全
```

## 禁止

- 只写结论不写证据。
- 混合不同环境/配置对比。
- 漏未实施项。
- 用单次最优值代表结果。
- 无 SHA256/命令的“可复现”声明。
