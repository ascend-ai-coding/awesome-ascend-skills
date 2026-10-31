---
name: external-cannbot-community-model-model-recommend-ge-graph-optimization
description: GE 图模式与 shape 策略优化（工作流条目 6）。推荐模型以 GE Session v2 图模式为执行基座，本 skill 负责 shape
  策略选型（all_static/all_dynamic/custom）、GE 编译选项调优（限核/精度模式/融合开关）、dump_graph 图验证与重编译控制。触发：shape
  策略、动静静态、all_static、all_dynamic、batch 语义、INPUT_SHAPE、pass_input_shape、图编译选项、融合开关、dump_graph、重编译、编译缓存、kernel_meta。
original-name: model-recommend-ge-graph-optimization
synced-from: https://gitcode.com/cann/cannbot-skills
synced-date: '2026-10-31'
synced-commit: 39919e2c995e1cd9e193f11ac2e3fcb3cbd1d26c
license: UNKNOWN
---

# GE 图模式与 shape 策略优化

## 流水位置

基线跑通后的**第一优先优化项**：onnx 路径以 GE 图模式为唯一执行基座（无 eager 对照）；pytorch/HF 路径经 torchair 转 GE 后与 onnx 同一口径，shape 策略与编译选项直接决定 kernel 选择、融合质量与重编译行为。

## Prerequisites

- [baseline-run](../model-recommend-baseline-run/SKILL.md) 已跑通（GE Session v2 执行链路正常）
- [structure-analysis](../model-recommend-structure-analysis/SKILL.md) 的动态维分布与显存估算
- [performance-analysis](../model-recommend-performance-analysis/SKILL.md) 基线性能数据（可选，用于收益判定）

## 目标

为模型选定最优 shape 策略与 GE 编译选项组合，最大化执行性能并控制编译开销；用 `dump_graph` 验证图编译质量。

**重要性**：静态化 shape 让 GE 获得完整形状信息，kernel 选择与融合通常显著优于动态 shape；策略错误会导致反复重编译或 workspace 膨胀。

## 输入 / 输出契约

**输入**：跑通的模型工作区；动态维分布；性能基线。

**输出**：shape 策略与编译选项的部署配置 + 前后性能/编译时间对比 + `dump_graph` 图验证记录 + 重编译评估结论。

## 重要原则

- **图模式是基座不是选项**：不存在"是否上图"的决策，只有"图怎么编译得更好"
- **保持模型语义**：shape 策略只影响建图提示（`parser_shape`），不得改计算逻辑
- **每次只改一个变量**：mode / batch / 精度模式 / 融合开关逐一变更，性能与精度同时记录（精度验证走 [precision-comparison](../model-recommend-precision-comparison/SKILL.md)）
- **数据 shape 与建图 shape 是两回事**：`runtime_shape` 来自 `.npy` 实际数据；`parser_shape` 是给 GE 的建图提示（`all_static` 下 batch 维按 `batch_size` 静态化，数据自动沿第 0 维扩展/裁剪）
- **改动校验双门禁（强制）**：本 skill 的每次策略/选项变更都是对模型执行的改动——落地前必须**串行 verify 全 pass**（精度门禁）且**同口径主指标不劣化超过噪声带**（性能门禁，编译时间可作辅助指标但主指标为 avg/TP99/QPS）；切换策略若造成性能回退，即使精度通过也必须回退并在 perf_ledger 记录；详见 [ascend-env-baseline](../model-recommend-ascend-env-baseline/SKILL.md) 全局约束

## shape 策略决策表

| mode | parser_shape 生成规则 | 传 `INPUT_SHAPE` | 适用场景 |
|------|---------------------|-----------------|---------|
| `all_static`（默认） | 每输入取实际 shape；onnx 动态 batch 维按 `batch_size` 静态化 | 是 | 请求形状固定、内存可控，**首选**；kernel 选择与融合最优 |
| `all_dynamic` | 维持 `[-1,...]`（`default_parser_dynamic_value` 占位） | 否 | 形状多变、避免反复编译；workspace 可能很大 |
| `custom` | `overrides[name].dynamic` 指定维动态，其余取数据维 | 是 | 动静混合（如 batch 动态、特征维固定） |

关键语义（框架 §6）：

- **`all_static` 的 batch 语义**：`batch_size` 是权威——onnx 第 0 维为 `-1` 的输入，数据自动扩展/裁剪到 `batch_size`，parser_shape 变为 `[batch_size,...]`；`run.sh --batch N` 即测不同静态形状
- **`all_dynamic` 不随 `batch_size` 固化**：parser_shape 维持 `[-1,...]`，喂入的仍是一份具体形状数据
- **输出动态维无需手填**：`io_config.outputs[].shape` 保留 `-1`，运行时以 GE 返回的真实 shape 为准（写 `output_shapes.json`）；静态声明不一致会打 `[GE OUTPUT][WARN]` 并用真实值
- **mode 只在生成阶段生效**：改完必须重跑 `prepare_input.py`（parser_shape 烘焙进 `io_config.json`）；`run.sh` 只改运行参数不暴露 mode

## 工作流步骤

### Step 1: 策略选型分析

按请求形状分布选型：固定形状（在线打分典型）→ `all_static`；形状集合有限且离散 → `all_static` 逐档编译（评估编译时间与缓存）；形状连续多变 → `custom`（batch 动态）或 `all_dynamic`（免重编译，接受 workspace 与性能代价）。

### Step 2: 方案确认

输出设计方案（mode / batch 档位 / 需要的 `overrides` / 输出动态维处理）等待用户确认；有修改意见返回 Step 1。

### Step 3: 实施与验证（实际执行，记录真实结果）

1. **切换策略**：`bash prepare.sh <id> --mode X --force`（或 `--config-only` 后手改 `shape_config.json` 再 `--input-only`）。
   - 本步骤会改 `shape_config.json` / `io_config.json`，二者属**预期实验变量**：与基线对比前用 `python3 tools/check_provenance.py <M<id>> --declare shape_config_sha256,io_config_sha256` 声明变更（工具会输出 `declared_change` 记录），不得当成口径漂移或直接重建基线
2. **确认生效**：`bash run.sh <id> --show`（打印 `shape_mode` / `pass_input_shape`）；或直接看 `io_config.json` 的 `shape_mode` 与 `inputs[].parser_shape`（动态维为 `-1`）
3. **精度验证**：`bash run.sh <id> --verify`（仅串行）
4. **性能验证**：`run.sh`（单并发时延）与目标并发口径，与基线同口径对比
5. **编译行为验证**：记录冷启动编译耗时与 `kernel_meta` 复用情况；多档 batch 场景统计每档编译成本

### Step 4: 图质量验证（dump_graph）

`bash run.sh <id> --dump_graph` → `models/M<id>/.cache/dump_graph`（最多保留 3 份）：

- 查看编译各阶段图（解析图 → 优化图 → 执行图），验证预期 pass 生效（融合、转换消除、布局优化）
- 配合 `.cache/run/fusion_result.json` 量化融合命中，为 [fusion-operator-adaptation](../model-recommend-fusion-operator-adaptation/SKILL.md) 提供证据

### Step 5: 编译选项调优（逐项）

| 选项 | 入口 | 作用 | 权衡 |
|------|------|------|------|
| `CORE_NUM`（`ge.aicoreNum`，"AIC\|AIV"） | `run.sh --cores` | 限核 | 与 [concurrency-strategy-analysis](../model-recommend-concurrency-strategy-analysis/SKILL.md) 联动 |
| `FORCE_16` / `ENABLE_HF32` | `test.sh` 环境变量 | 精度模式 | 性能 vs 精度，逐项过 [precision-comparison](../model-recommend-precision-comparison/SKILL.md) |
| `ge.exec.precision_mode` / `ge.exec.op_precision_mode` | GE 选项（改 main.cc 或环境注入） | 全局/算子级精度路由 | 精细控制，变更须记录 |
| `ge.fusionSwitchFile` | GE 选项 | 融合开关文件 | 关闭/开启特定融合 pass，归因与回退用 |
| `GRAPH_MAX_PARALLEL` | `run.sh --graph-max-parallel` | 图实例并行上限 | 与 `THREAD_NUM` 联动 |

### Step 6: 重编译控制结论

- `all_static` 多档形状：`kernel_meta` 同配置复用（冷启动收益）；档位过多时评估编译时间占比，必要时收敛档位或转 `custom`
- `all_dynamic`：无重编译但单次执行性能通常劣于静态——以实测数据为决策依据，不做无依据推断

## 昇腾 NPU 专属要点

- **`pass_input_shape=false`（all_dynamic）时 GE 不收 `INPUT_SHAPE`/`OUT_NODES`**，融合与 kernel 选择信息变少——性能差距须实测，不要拍脑袋。
- **`hint_shape` 始终传**（`ge.inputHintShape`，运行时 shape），与 parser_shape 是两个通道。
- 精度模式收益是 shape 相关的：同一模式在静态/动态图下收益不同，分别测。
- 并发（`THREAD_NUM>1`）与 dump/verify 互斥（框架约束）：图验证与精度验证在串行做，并发只看性能。

## 产物目录约束

- **文档格式约束（GFM 删除线）**：Markdown 文档禁止裸单波浪线 `~`——同段落/同表格单元内成对出现会被 GFM 渲染为删除线；数值区间用全宽 `～`（`-15%～-30%`）或 `-`/`至`，约数用 `≈`；交付前 grep 确认文档无裸 `~`（详见 [ascend-env-baseline](../model-recommend-ascend-env-baseline/SKILL.md) 全局约束）
- **文档类交付件** → `docs/{model_name}/ge-graph-optimization/`：策略选型与选项调优方案、验证报告（--show/dump_graph 证据、性能/编译时间对比、重编译结论）
- **数据类交付件** → `data/{model_name}/ge-graph-optimization/`：配置对比结构化数据（mode/batch/选项 → 性能指标，供回归引用）
- **中间产物** → `tmp/{model_name}/ge-graph-optimization/`：试验性 shape_config、dump_graph 原始分析数据、调试日志
- 全局约定与判定标准见 [ascend-env-baseline](../model-recommend-ascend-env-baseline/SKILL.md)；违反目录约定的产物视为未交付

## 验收标准

- shape 策略与场景匹配有前后数据（时延/QPS/编译时间，同口径）
- `--show` 与 `io_config.json` 一致（策略真实生效）
- dump_graph 证明预期编译行为（融合命中/无异常转换残留）
- **双门禁通过**：串行 verify 全 pass 且主指标不劣化（或显式取舍经用户确认）；每笔变更已写 perf_ledger（精度结论 + 同口径性能对比 + 回退状态）
- 重编译行为有结论（编译耗时与缓存复用记录在案）

## 依据

【框架】shape 策略三模式语义与 batch 语义（`README.md` §6「配置项 → shape_config.json / 三种输入场景」）、`--show`/`--dump_graph`/`--batch`、`run.sh` 选项、`kernel_meta`/`fusion_result.json` 缓存治理、运行约束。
【实测】策略切换前后性能对比数据。
