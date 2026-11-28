---
name: external-cannbot-community-model-model-recommend-adaptation
description: 推荐模型 onnx 推理的适配（从模型包到可运行、可验证）。含入库校验、形态识别、部署边界、算子覆盖矩阵、IO 契约、输入来源分级、跑通与功能验证、故障分层定位。已确认存在算子/内核缺口、需开发交付件时转
  GE 自定义交付件 skill（plugin/fusion-pass/custom-op，按缺口形态由工作流路由）。触发：模型接入、模型包入库、形态识别、主图/子图判断、部署边界、算子覆盖矩阵、IO
  契约、输入构造、解析失败、infershape 失败、跑通、功能验证。
original-name: model-recommend-adaptation
synced-from: https://gitcode.com/cann/cannbot-skills
synced-date: '2026-11-28'
synced-commit: 7a09cdfac4b2869dca6b6fd4531a0a5e539d6096
license: UNKNOWN
---

# 模型适配（从模型包到可运行、可验证）

## 何时用 / 边界

- 用于模型**首次接入**或适配返工，目标是把模型跑到能正常出结果，并留下可复核的适配结论。
- 只支持 **onnx + GE_SESSION_DEMO + 单卡**。子图拆分、CPU+NPU 异构部署、多卡不在本 skill 内；出现时标注越界并上报。
- 不做性能优化与优化点分析（转 `model-recommend-analysis`）；不做插件/Pass/算子开发（转 `model-recommend-plugin-dev` / `model-recommend-fusion-pass-dev` / `model-recommend-custom-op-dev`）。
- 不修改原始 onnx；适配缺口只记录，由对应 GE 自定义交付件 skill 实施。

## 适配与优化的边界

- 适配目标：**覆盖完整、功能正确、接口稳定**；优化目标：**在约束下提升指标**。
- 适配只解决"能否正确跑通"，不采集性能数据、不做性能结论；优化只解决"跑得快"。
- 适配未通过（含功能验证失败）不进入基线与优化；优化阶段冻结基线与输入，不同时改变适配配置。
- 适配阶段可记录"性能观察"作为线索，但不得作为优化收益依据。

## 输入 / 产出

| 类型 | 内容 |
| --- | --- |
| 必需 | `models/M<id>/model/model.onnx`；外挂权重时 `model_weights.bin` |
| 可询问 | 模型来源与版本、预期输入输出、边界执行器、是否允许重新拆分、精度要求 |
| 产出 | `models/M<id>/adaptation/adaptation_report.md`（人读）、`adaptation_status.json`（机读） |

- 输入来源与保真度必须标注；合成输入只能用于编译/执行 smoke，不作为正确性结论。

## 流程

| 步骤 | 做什么 | 退出条件 |
| --- | --- | --- |
| 0 入库校验 | 文件齐全、sha256、opset/IR 版本、外挂权重 | 文件与指纹可复现 |
| 1 形态识别 | 读节点/IO/权重归属/调用边界，以图结构判断全量与部分 | 能说清图的边界与调用关系 |
| 2 边界与执行器 | 明确各阶段执行器、边界张量、是否允许重新拆分 | 边界无歧义 |
| 3 覆盖评估 | 算子直方图 × 运行时能力，产出覆盖矩阵 | 已覆盖/可替代/缺口/不需要清晰 |
| 4 环境准备 | NPU/CANN/GE/工具链、demo 编译 | 最小样例可启动 |
| 5 IO 契约 | 名称、dtype、shape/符号维、顺序、数据放置 | 与运行时一致 |
| 6 输入来源 | 按保真度选择真实/分阶段/整模型/合成 | 来源与保真度明确 |
| 7 跑通 | parse/build/run；按故障分层逐层排错 | 无阻断性错误 |
| 8 功能验证 | 输出数量、完整性、有限性、形状；有 golden 才做数值比对 | 满足功能验收 |
| 9 交接 | 缺口交 `model-recommend-plugin-dev` / `model-recommend-fusion-pass-dev` / `model-recommend-custom-op-dev`（按工作流路由）；功能通过的交 `model-recommend-analysis` | 交接项明确 |

## 形态识别

- 以图为准：节点数、IO、权重归属、是否存在调用节点/子图边界。
- 不按文件名或时间戳推断版本先后；重复文件用 sha256 与语义比对确认。
- 明确运行顺序：前置处理 → 边界调用 → 子图执行 → 后置处理。

## 覆盖矩阵

模型算子（域/类型/数量）× 运行时能力，分四类：

| 分类 | 含义 | 处理 |
| --- | --- | --- |
| 已覆盖 | 内置可直接运行 | 记录版本与限制 |
| 可替代 | 等价改写/Pass/Plugin 可达成 | 交对应 GE 自定义交付件 skill（工作流路由） |
| 缺口 | 现有能力无法达成 | 评估自定义算子；不划算则记录越界 |
| 不需要 | 模型不涉及 | 不安装、不优化 |

评估维度：语义、dtype、shape/动态性、精度要求、性能收益、实现成本。

## IO 契约

- 记录：名称、dtype、静态/动态维、符号关系、顺序、数据放置。
- 区分声明 shape（可为符号/`-1`）与运行期 shape（由实际输入决定）。
- IO 契约是输入构造、联调与验收的共同依据。

## 输入来源分级

| 级别 | 来源 | 用途 |
| --- | --- | --- |
| 1 | 真实边界张量 | 功能与性能验证 |
| 2 | 真实请求 + 分阶段运行导出的边界张量 | 功能与性能验证 |
| 3 | 完整模型导出边界张量 | 功能与性能验证 |
| 4 | 合成输入 | 仅编译/执行 smoke |

- 需要正确性/精度结论时只用 1–3；合成输入固定随机种子与形状，并标注非正确性依据。

## 环境与资源准备

- 工具链、框架、运行时、依赖、编译验证；最小样例可启动。
- 共享设备先观测占用与进程再决定运行窗口；固定设备、线程、核心、流、缓存目录与插件/Pass 版本；存在其他进程占用或同配置波动 >3% 时隔离运行。
- 关注显存峰值与稳态、碎片、host/device 放置、权重/精度转换产生的重复副本。
- 失败/异常退出后检查残留进程与资源释放。

## 功能验证

- 检查输出数量、顺序、完整性、有限性（无 NaN/Inf）、形状。
- 有 golden 才做数值比对；golden 优先用户 reference，缺失时用 CPU onnxruntime；bf16/自定义算子需用户提供标杆。
- 容差口径：浮点默认 `rtol=atol=1e-3`；int/bool 精确匹配；自定义算子可经用户确认放宽，并记录决策。
- 精度维度：dtype 路径（fp32/fp16/bf16/整型）与累加精度；逐层/逐算子误差；分布偏移。
- 稳定性：重复运行验证一致性；长稳关注泄漏、碎片、抖动。
- **"能跑通"不等于"正确"**：功能、精度、性能是三个独立验收项。
- 覆盖典型输入与退化输入（零尺寸/空序列、极小/极大 shape、动态维、边界值）；退化输入不得用于最终性能结论。
- 区分声明 shape 与运行期 shape；边界张量 dtype/顺序/布局必须一致。

## 产物与度量

| 产物 | 内容 |
| --- | --- |
| 入库清单 | 文件、sha256、opset/IR、依赖 |
| 形态/拓扑说明 | 全量/主图/子图/副本、调用关系 |
| 边界与 IO 规格 | 执行器、张量名称、dtype、shape、顺序 |
| 算子覆盖矩阵 | 已覆盖/可替代/缺口/不需要 |
| 输入清单 | 来源、形状、dtype、保真度、生成方式 |
| 运行日志 | 命令、环境、关键错误、dump/debug 路径 |
| 功能验证报告 | 输出检查、golden、容差、结论 |

- 本 skill 不采集性能数据；功能与性能是独立验收项。性能指标（供后续阶段使用）：QPS/吞吐、avg、P50/P95/P99/P999、时延分布、波动（标准差/变异系数）；分解口径：wall、kernel_sum、non_kernel、gap、wait、host API 占比。
- 对外部包/插件/Pass 记录实际安装路径与 `.so` 哈希，不只记录源码路径。

## 故障分层定位

| 层 | 典型症状 | 证据 | 处理 |
| --- | --- | --- | --- |
| 模型语义 | 输出错误、精度异常 | 参考输出、golden | 对齐语义/版本 |
| 图解析 | parse 失败、算子/属性未知 | parser 日志、节点名 | 插件/Pass/改写 |
| shape/dtype | infershape 失败、广播/轴不一致 | 输入 shape、节点属性 | 修正输入/契约 |
| 资源环境 | OOM、超时、设备占用 | 设备监控、进程信息 | 等待/隔离/降载 |
| 工具链 | 编译失败、依赖缺失 | 编译日志、版本 | 修复环境/版本匹配 |

方法：最小复现 → 单变量修改 → 对比日志 → 记录结论。避免把输入 shape 问题误判为算子不支持。

## 完成标准 / 检查清单

- [ ] 版本/指纹/依赖齐全；全量、主图、子图、副本关系明确；权重与外部数据完整。
- [ ] 部署边界与执行器明确；覆盖矩阵完成；自定义算子/插件/Pass 版本匹配。
- [ ] IO 契约与输入来源明确；资源（显存/内存/设备）可用。
- [ ] 模型可 `prepare`/`compile`/`run`，退出码为 0；无阻断性错误。
- [ ] 功能验证：输出数量、顺序、形状、有限性正确；有 golden 时 `--verify` 通过。
- [ ] 缺口有承接方；工作流可读取 `adaptation_status.json`。
- [ ] 原始 onnx 未被修改；全部改动可回溯、可回滚。

## 机读状态

`adaptation_status.json`（本 skill 唯一写）：

```json
{
  "schema_version": "1.0",
  "model": "M<id>",
  "updated_at": "<ISO8601>",
  "input": {"onnx": "<path>", "weights": "<path|null>", "sha256": {"<file>": "<hash>"}},
  "form": {"kind": "full|main-graph|subgraph|replica", "call_relation": "", "stages": [], "note": ""},
  "boundary": {"executors": [], "tensors": []},
  "coverage": {"covered": [], "replaceable": [], "gaps": [], "not_needed": []},
  "io": [{"name": "", "dtype": "", "shape": [], "dynamic": false}],
  "input_source": {"level": "real|staged|full-model|synthetic", "path": "", "note": ""},
  "run": {"ok": false, "command": ""},
  "functional_check": {"output_count_ok": null, "shape_ok": null, "finite": null, "reference": "none|user|ort", "result": "pass|fail|not-run"},
  "handoff": {"gaps_to": ["model-recommend-plugin-dev", "model-recommend-fusion-pass-dev", "model-recommend-custom-op-dev"], "routing": "workflow", "analysis_to": "model-recommend-analysis", "missing": []}
}
```

## 反模式

- 用文件名/时间戳推断版本或执行顺序。
- 未确认边界就开始安装插件或改图。
- 只用合成输入验证，把"能跑"当作"正确"。
- 无差别安装全部插件/Pass，或优化模型不涉及的结构。
- 把输入 shape 问题误判为算子不支持。
- 在共享设备上不做资源观测与隔离。
- 修改模型/框架但不记录原因、范围与回滚方式。
- 适配未通过就进入基线与优化。
- 把适配阶段的性能观察当作优化收益依据。

## 参考

- 结构档案、算子构成与 drawio 图集细节：`references/structure.md`、`references/drawio-template.md`。
