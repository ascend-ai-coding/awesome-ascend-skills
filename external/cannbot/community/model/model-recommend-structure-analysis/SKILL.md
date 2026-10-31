---
name: external-cannbot-community-model-model-recommend-structure-analysis
description: 推荐模型结构与显存分析（工作流条目 1）。解析 models/M{id}/model/model.onnx，产出结构档案（输入输出清单/dtype/动态维）、算子构成分析（AIC/AIV
  占比、Top-N、embedding 大表）、模块链路清单、显存静态估算与单卡可执行判定，附 drawio 结构图集（可选）。触发：模型分析、onnx 解析、算子构成、AIC/AIV
  统计、embedding 统计、显存/内存估算、可执行性评估、新模型接入、模型结构图、drawio。
original-name: model-recommend-structure-analysis
synced-from: https://gitcode.com/cann/cannbot-skills
synced-date: '2026-10-31'
synced-commit: 39919e2c995e1cd9e193f11ac2e3fcb3cbd1d26c
license: UNKNOWN
---

# 模型结构分析：onnx 结构，算子构成，显存占用等

## Prerequisites

- 已通过 [ascend-env-baseline](../model-recommend-ascend-env-baseline/SKILL.md) 环境自检（NPU 可用、CANN/GE 组件完整、GE_SESSION_DEMO 编译通过）
- GE_SESSION_DEMO 模型工作区：`models/M<id>/model/model.onnx` 必需（外部权重须与 onnx 同目录）；`input_data/` 可选

> **pytorch / HuggingFace 路径为实验性**：本 skill 目前只有 onnx / GE 侧实现，pytorch 侧需用户提供等价脚本（结构导出 / 基线脚本 / 并发压测 / `torch_npu.profiler` 采集）。

## 目标

从 onnx 出发，产出「结构档案 + 算子构成分析 + 模块链路清单 + 显存静态估算 + 单卡可执行判定」，作为 [concurrency-strategy-analysis](../model-recommend-concurrency-strategy-analysis/SKILL.md)、[ge-graph-optimization](../model-recommend-ge-graph-optimization/SKILL.md)、[fusion-operator-adaptation](../model-recommend-fusion-operator-adaptation/SKILL.md)、[fusion-planning-estimation](../model-recommend-fusion-planning-estimation/SKILL.md) 的公共输入。

## 输入 / 输出契约

**输入**：`models/M<id>/` 工作区（onnx 模型与可选输入数据）。

**输出**：

1. **结构档案**：输入/输出清单（`index`/`name`/`dtype`/`elem_bytes`/`onnx_shape`/`dynamic_dims`/`symbolic`）、opset、initializer（权重）数量与总量、图规模（节点数/算子种类数）。
2. **算子构成分析**：AIC 类（MatMul/MatMulV2/BatchMatMul*/Gemm/Conv 族，与 `tools/sweep.py::analyze_ops` 的 AIC_OPS 集合同口径）与 AIV 类（其余）计数及占比、算子频次 Top-N、超大 initializer（embedding 表）单列、不支持 dtype 与非标 domain 节点预警清单。
3. **模块链路清单**：按 onnx 拓扑归纳「特征输入 → Embedding/查表 → 特征交互（cross/attention/interaction）→ DNN/MLP → 多任务输出」链路（按实际图裁剪），标注各算子归属（AIC/AIV/访存/控制）。
4. **显存估算（独立文档）**：`docs/{model_name}/model-analysis/memory_estimation.md`——权重（initializer 逐表字节数）+ 激活（按输入 shape 推导主要中间张量）+ GE workspace（动态 shape 策略偏大）+ 并发图实例叠加（`GRAPH_MAX_PARALLEL` × workspace）；配套数据 `data/{model_name}/model-analysis/memory_estimation.json`。
5. **batch / 动态 shape 可行性判定**：`data/{model_name}/model-analysis/feasibility.json`——从图上证据判断（TopK / NonZero / Where / If / 依赖第 0 维的 reshape/slice、第 0 维参与算术的后处理、动态维之间的耦合关系）给出 `batch_support: yes|no|unknown`、`dynamic_support: yes|no|unknown` 与 `evidence`（节点名/理由）；`unknown` 表示图上判不出，交由条目 2 Step 4 冒烟探测定案。判定为 `no` 时，条目 3 的 batch 候选与 sweep batch 阶段直接禁用。
6. **drawio 结构图集（可选）**：`docs/{model_name}/model-analysis/diagrams/{model_name}.drawio` 单文件多页，规范见 [references/drawio-template.md](references/drawio-template.md)。

## 工作流步骤

### Step 1: 结构参数提取

`bash prepare.sh <id> --config-only`：只生成 `config/model_config.json`（输入输出清单与 dtype）与默认 `shape_config.json`，**不生成 `io_config.json` / `batch_list.json`**（那需要输入数据，属条目 2）；`prepare.sh` 自带工作区与外部权重同目录校验。dtype 支持集合：`float32`/`float16`/`bfloat16`/`int32`/`int64`/`bool`，越界 dtype 在此阶段告警/报错。

### Step 2: 算子构成统计

- onnx 节点 `op_type` 频次 Counter；AIC 类集合判定（口径与 `tools/sweep.py` 一致：MatMul/MatMulV2/BatchMatMul/V2/Gemm/Conv/Conv2D/Conv3D/DepthwiseConv2D/Conv2DBackprop*）
- initializer 体积排序，超大 embedding 表（GB 级）单列并标注 dtype
- 兼容性预警逐项核对：
  - 非标 domain 节点（如 `custom.aiflow::SparseToDense`）→ [fusion-operator-adaptation](../model-recommend-fusion-operator-adaptation/SKILL.md)（非标 domain 折叠改写，工作副本）或 [custom-op-development](../model-recommend-custom-op-development/SKILL.md)
  - bool 输入的算子（ReduceSum/Min/Clip 等，NPU 无 bool engine）→ bool→int32 无损 Cast 可改写（工作副本）
  - 输出含动态维 → 需 `shape_config.output_overrides` 或运行时以 GE 返回 shape 为准

### Step 3: 模块链路拆解

按拓扑序归纳模块链路并标注归属（直接支撑条目 3 限核配比与条目 10 融合规划）：

| 归属 | 典型算子 |
|------|---------|
| AIC（Cube） | MatMul/Gemm/Conv 族 |
| AIV（Vector） | elementwise（Add/Mul/Relu/Sigmoid）、Reduce、Norm、Cast、Concat |
| 访存类 | Gather/GatherNd/Embedding 查表/ScatterND/SparseToDense |
| 控制类（小算子串，launch 开销候选） | Shape/Slice/Reshape/Transpose/Squeeze/Unsqueeze/Where/If/Loop |

### Step 4: 生成 drawio 结构图集（可选）

依据 Step 3 链路清单生成单文件多页 drawio（页清单按推荐模型裁剪：P1 主题结构 / P2 特征输入与 Embedding / P3 特征交互 / P4 DNN-MLP / P5 输出层；格式、图例与自校验见 [references/drawio-template.md](references/drawio-template.md)）。自校验必做：XML well-formed、页内节点 id 唯一、边引用有效；图与结构档案字段一致。

### Step 5: 显存静态估算（独立文档 memory_estimation.md）

- 权重：initializer 逐表字节数与合计（对照 onnx `location` 外部权重文件大小）
- 激活：按目标输入 shape 推导主要中间张量（大 Concat/交互矩阵单列）
- GE workspace：静态 shape 策略 vs 动态 shape 策略分别估（动态偏大）
- 并发叠加：`GRAPH_MAX_PARALLEL` × (workspace + 激活)
- 数值必须经脚本核验（脚本入 `tmp/{model_name}/model-analysis/`），文档与 `data/` 下 JSON 一致
- 对照 `npu-smi info` 实测 HBM（单卡 65536MB），输出最大可用 batch/并发图实例数

### Step 6: 单卡可执行判定

- 权重 + 单请求激活 + workspace 装得下 → 流转 [baseline-run](../model-recommend-baseline-run/SKILL.md)
- 放不下（embedding 超表等）→ 如实输出问题清单与结论（分片/稀疏化超出本工作流范围，需用户决策）；并发上限受 HBM 约束时在报告中给出安全 `GRAPH_MAX_PARALLEL` 建议值

## 昇腾 NPU 专属要点

- AIC/AIV 占比是单卡限核配比（`CORE_NUM="AIC|AIV"`）与融合规划的第一输入：AIC 主导优先保 AIC 核，AIV/查表主导优先保 AIV 核。
- 动态维分布决定 shape 策略：第 0 维（batch）动态为主 → `all_static + batch_size` 或 `custom`；多维动态 → 权衡 `all_dynamic`（免重编译 vs workspace 偏大），详见 [ge-graph-optimization](../model-recommend-ge-graph-optimization/SKILL.md)。
- bool 输入算子、非标 domain 是推荐模型（aiflow/sklearn 导出）高频适配点，本阶段必须逐一预警，不得留到运行期报错。
- fp16/bf16 权重模型注意 golden 生成约束（onnxruntime 对 bf16 不可用，需用户提供标杆）。

## 产物目录约束

- **文档格式约束（GFM 删除线）**：Markdown 文档禁止裸单波浪线 `~`——同段落/同表格单元内成对出现会被 GFM 渲染为删除线；数值区间用全宽 `～`（`-15%～-30%`）或 `-`/`至`，约数用 `≈`；交付前 grep 确认文档无裸 `~`（详见 [ascend-env-baseline](../model-recommend-ascend-env-baseline/SKILL.md) 全局约束）
- **文档类交付件** → `docs/{model_name}/model-analysis/`：结构档案（model_analysis_report.md）、显存估算（memory_estimation.md）、drawio 图集（`diagrams/{model_name}.drawio`）
- **数据类交付件** → `data/{model_name}/model-analysis/`：结构化档案（model_params.json，供下游 skill 程序化引用）、显存估算明细（memory_estimation.json）
- **中间产物** → `tmp/{model_name}/model-analysis/`：onnx 解析中间文件、估算脚本、试运行原始日志
- 全局约定与判定标准见 [ascend-env-baseline](../model-recommend-ascend-env-baseline/SKILL.md)；违反目录约定的产物视为未交付

## 验收标准

- 结构档案字段与 `model_config.json` 一致（输入输出清单、dtype、动态维）
- 算子统计与 onnx 逐节点可复核（AIC/AIV 计数口径与 `sweep.py` 一致）
- 显存估算与实跑 HBM 占用偏差 < 15%
- 兼容性预警覆盖全部非标 domain 节点、bool 输入算子、不支持 dtype 与动态输出
- **配置一致性验收不在本步骤**：`inspect_config.py` 需要 `io_config.json` 与 `batch_list.json`，而本步骤只跑 `prepare.sh --config-only`（不生成输入配置）。完整的 `prepare.sh` + `inspect_config.py` 验收在条目 2（[baseline-run](../model-recommend-baseline-run/SKILL.md) Step 3）执行
- drawio 图集（如产出）XML 校验通过且与结构档案字段一致

## 依据

【框架】`prepare.sh` / `prepare_model.py`（结构解析、外部权重同目录校验、dtype 集合）、`sweep.py::analyze_ops`（AIC_OPS 口径）、`inspect_config.py`（配置校验，在条目 2 的完整 `prepare.sh` 之后使用）。兼容性改写（domain 折叠、bool→int32 Cast）没有现成脚本，按需写一次性改写脚本、输出落 `tmp/`，原图不动。
【实测】npu-smi info（单卡 HBM 65536MB）。
【插件】插件内对应 skill 的适配要点与最小验证模型（底层实现以环境内 CANN 算子库为准）。
