---
name: external-cannbot-community-model-model-recommend-fusion-operator-adaptation
description: 既有算子与图优化适配（工作流条目 7）。分析模型算子组成，用不新增算子实现的手段落地：GE 图编译内置优化（布局/转换消除）、onnx 等价改写（domain
  折叠/bool cast/算子替换）、精度模式路由；图融合 pass 属一次性框架扩展（默认关闭）。触发：算子适配、融合命中分析、onnx 改写、domain
  折叠、bool 算子、算子替换、fusion_result、融合开关、算子耗时热点。
original-name: model-recommend-fusion-operator-adaptation
synced-from: https://gitcode.com/cann/cannbot-skills
synced-date: '2026-10-31'
synced-commit: 39919e2c995e1cd9e193f11ac2e3fcb3cbd1d26c
license: UNKNOWN
---

# 既有算子与图优化适配

## 强制约束（本阶段执行纪律）

1. **已有能力 + 功能语义可对齐 → 必须执行完整适配**，不涉及新算子开发的所有依赖项（onnx 形态/等价改写/GE 选项/参数组合）全部在本阶段完成，不得推迟
2. **功能不能对应或适配后有问题 → 必须输出明确未适配原因**（错误码/约束/语义差异），并归档为「未适配项清单」——性能替换型并入 [fusion-planning-estimation](../model-recommend-fusion-planning-estimation/SKILL.md) 统一排序，功能补齐型转 [custom-op-development](../model-recommend-custom-op-development/SKILL.md)；**本阶段禁止自行启动新算子开发，也不得把未适配项直接标记为 kernel 开发任务绕过规划**
3. **未适配项清单格式**（逐项）：模块/链路 → 失败原因分类（无算子 / 硬约束 / 语义不匹配 / 精度未达标）→ 所需算子能力描述 → 已排查的候选手段与参考 → 与热点链路的关系（是否已有 profiling 证据）
4. **算子能力查证源**：CANN 随包算子清单文档（随 Toolkit 提供，GE/ACL 支持的 onnx 算子与 aclnn 算子）+ 插件内适配要点（本 skill 与 [flash-attention-fusion](../model-recommend-flash-attention-fusion/SKILL.md) 的调用形态与语义参考）
5. **参数报错排查法**：先查 CANN 算子文档取参数语义与约束，再对照 GE parser 报错（`run.sh --info/--debug` 开 plog，日志落 `models/M<id>/report/run_debug_*.log`）定位真实错误
6. **改动校验双门禁（强制）**：每个适配项（onnx 改写/选项变更）都是对模型执行的改动——落地前必须**串行 verify 全 pass**（精度门禁，改写假设逐项核对）且**同口径端到端主指标不劣化超过噪声带**（性能门禁；融合收益是 shape/场景相关的，回退项如实记录）；任一不通过即回退该项并记录；详见 [ascend-env-baseline](../model-recommend-ascend-env-baseline/SKILL.md) 全局约束

## Prerequisites

- [structure-analysis](../model-recommend-structure-analysis/SKILL.md) 的算子构成与模块链路清单
- onnx 模型（含改写前的原始版本备份）
- 运行场景特征：shape 策略（[ge-graph-optimization](../model-recommend-ge-graph-optimization/SKILL.md) 产物）、batch、并发口径

## 目标

分析模型算子组成，把可替换/可改善的计算模式用**不新增算子实现**的手段落地，收益优先；图融合 pass 属一次性框架扩展，单独开关。

## 输入 / 输出契约

**输入**：算子构成分析 + onnx 模型 + 运行场景特征。

**输出**：逐模块适配方案（模块 → 手段 + 约束核对 + 未覆盖子链路清单）、适配前后 [performance-analysis](../model-recommend-performance-analysis/SKILL.md) 对比数据，以及**未适配项清单**（格式见强制约束 3）。

## 三类适配手段（GE 图模式）

### 手段一：GE 图编译内置优化（零改动）

- 融合 pass（自动算子融合）、布局优化、转换消除——GE 编译时自动执行
- **验证手段**：`run.sh --dump_graph`（编译各阶段图）+ `.cache/run/fusion_result.json`（融合统计）判定命中与遗漏
- **精细控制**：`ge.fusionSwitchFile` 融合开关文件，关闭/开启特定 pass（归因与回退用）

### 手段二：onnx 等价改写（工具层，框架已有先例）

| 工具/手段 | 适用 | 说明 |
|-----------|------|------|
| 非标 domain 折叠脚本（一次性，落 `tmp/`） | 非标 domain 节点（如 `custom.aiflow::SparseToDense`） | 节点 domain 置空按 op_type 解析；同时去重 opset_import；输出**工作副本**，原图不动 |
| bool 无损 Cast 脚本（一次性，落 `tmp/`） | bool 输入算子（ReduceSum/Min/Clip 等无 bool engine） | 输入插 Cast(bool→int32)、输出还原 bool：`ReduceSum(bool)`＝计数、`Min(bool,bool)`＝逻辑与；图输入输出名不变；输出**工作副本** |
| 算子替换 | 语义等价但 GE 支持/性能更好的算子形态 | 如 Scatter/Gather 系列索引语义变体、Cast 链压缩、常量折叠 |
| 子图重写 | 小算子串等价合并（Reshape/Transpose 链、重复 Cast） | 中间张量物化消除，减少 GM 往返 |

**所有改写必须过 [precision-comparison](../model-recommend-precision-comparison/SKILL.md) verify**；改写假设（如 bool 语义）不成立的场景禁止使用。

### 手段三：精度模式路由

`FORCE_16` / `ENABLE_HF32` / `ge.exec.precision_mode` / `ge.exec.op_precision_mode`：逐项评估收益与精度，与手段一/二组合时逐一归因。

## 工作流步骤

### Step 1: 模块拆解

复用 [structure-analysis](../model-recommend-structure-analysis/SKILL.md) 链路清单，展开到可替换链路级（查表链、交互链、MLP 链、输出链、控制小算子串）。

### Step 2: 逐链路匹配

每个链路**独立**匹配：GE 已有优化是否覆盖（dump_graph 证据）→ 等价改写是否可行 → 精度模式是否有收益。命中后作为「候选蓝图」，说明覆盖/未覆盖子链路。

### Step 3: 算子能力校验

所有候选查 CANN 算子清单文档，记录文档路径、参数约束（dtype/shape/layout）与当前模型是否满足；插件内适配要点与 CANN 算子库实现作语义参考，**不能替代文档校验**。

### Step 4: 实施开发（带开关保护，分阶段验证收益）

- 改写产物独立存放（`tmp/{model_name}/fusion-operator-adaptation/patched_model.onnx`），**原始 onnx 保留**，工作区替换可一键回退
- **模型替换属预期实验变量**：把改写模型放回工作区后，与基线对比前用 `python3 tools/check_provenance.py <M<id>> --declare model_sha256` 声明（必要时加 `shape_config_sha256`/`io_config_sha256`），工具输出 `declared_change` 记录粘进报告；`input_sha256`/`golden_sha256` 是 frozen 域，不得改动
- **逐项 enable 记录**：每个改写/选项单独记录，先逐项验证再组合
- **融合收益是 shape/并发相关的**：同一改写在静态大 batch 可获益、在动态 shape 或高并发下可能回退——单并发与目标并发分别测

### Step 5: 适配后验证与提升状态归档（双门禁执行点）

1. **精度门禁先行**：每个适配项串行 `run.sh <id> --verify` 全 pass（改写假设逐项核对）；不过 → 回退该项，不进性能测量
2. **性能门禁**：与基线相同口径（同卡/同 batch/同 THREAD_NUM/CORE_NUM/同统计策略）端到端计时，主指标不劣化超过噪声带（或显式取舍经用户确认）；同口径 profiling 归档 `prof/{model_name}/fused-ops/{case}/`
3. **算子提升状态归档** → `data/{model_name}/fusion-operator-adaptation/op_improvement_{case}.json`：逐算子 before/after diff（`status`: eliminated / added / changed + `delta_us`，按 delta 排序）+ kernel 总时长变化 + 端到端变化（含方差/噪声带说明）
4. **落账**：每个通过的适配项写 perf_ledger（`reverted=false`）；回退项 `reverted=true` 并记录原因

## 昇腾 NPU 专属要点（推荐模型高频适配点）

| 适配点 | 现象 | 处理 |
|--------|------|------|
| bool 输入算子 | GE 无法放置（ReduceSum/Min/Clip 收 bool） | bool→int32 无损 Cast 改写（工作副本，先验证语义假设） |
| 非标 domain | parser 找不到算子 | 非标 domain 折叠改写（工作副本）；不可折叠的转未适配项清单 |
| AIV 小算子串 | elementwise/reduce/shape 类 launch 开销占比高 | 优先确认 GE 融合是否已覆盖（dump_graph/fusion_result），未覆盖进融合模式表 |
| 查表/Gather 链 | memory-bound 热点 | 等价形态评估（索引语义变体），深度优化转 [memory-access-optimize](../model-recommend-memory-access-optimize/SKILL.md) / [fusion-planning-estimation](../model-recommend-fusion-planning-estimation/SKILL.md) |
| ScatterND 索引 | 语义差异（坐标/扁平索引） | 逐一核对文档语义，等价改写后必过 verify |

- 融合命中对 shape 敏感：静态 shape（[ge-graph-optimization](../model-recommend-ge-graph-optimization/SKILL.md)）下融合率通常更高，策略与适配联动评估。
- dump/验证须串行（框架约束）；改写后并发性能单独测。

## 产物目录约束

- **文档格式约束（GFM 删除线）**：Markdown 文档禁止裸单波浪线 `~`——同段落/同表格单元内成对出现会被 GFM 渲染为删除线；数值区间用全宽 `～`（`-15%～-30%`）或 `-`/`至`，约数用 `≈`；交付前 grep 确认文档无裸 `~`（详见 [ascend-env-baseline](../model-recommend-ascend-env-baseline/SKILL.md) 全局约束）
- **文档类交付件** → `docs/{model_name}/fusion-operator-adaptation/`：适配方案（adaptation_plan.md）+ 适配结果报告（含逐项启用策略与回退项）
- **数据类交付件** → `data/{model_name}/fusion-operator-adaptation/`：算子热点分析、**算子提升状态（op_improvement_{case}.json）**、适配后计时
- **profiling 归档** → `prof/{model_name}/fused-ops/{case}/`（与基线归档同口径可对比）
- **中间产物** → `tmp/{model_name}/fusion-operator-adaptation/`：改写后 onnx（含原始备份）、改写脚本、golden 对比中间数据
- 全局约定与判定标准见 [ascend-env-baseline](../model-recommend-ascend-env-baseline/SKILL.md)；违反目录约定的产物视为未交付

## 验收标准

- 每个适配项有文档/参考路径与约束核对记录
- **每个适配项有精度验证结论**（verify 全 pass；改写假设有核对记录）
- **双门禁通过**：逐项同口径端到端主指标不劣化（或显式取舍经确认）；分项隔离归因数据在案；按场景（shape/并发）的启用策略明确；回退项如实记录并已写 perf_ledger
- **适配后 profiling 已归档**且**算子提升状态已归档**（op_improvement json：逐算子 diff + kernel 总时长变化 + 端到端变化含噪声带说明）
- **未适配项清单完整**：每个失败项有原因分类/所需能力/候选手段，已移交 [fusion-planning-estimation](../model-recommend-fusion-planning-estimation/SKILL.md) 或 [custom-op-development](../model-recommend-custom-op-development/SKILL.md)（禁止遗留"直接开发"口头结论）

## 依据

【插件】等价改写两类先例（非标 domain 折叠、bool 无损 Cast）与语义假设；【框架】`--dump_graph` / `fusion_result.json`（融合证据）、`ge.fusionSwitchFile`（融合开关）、`--info/--debug`（plog 落盘）。
【框架】CANN 自带算子清单与 GE 文档（候选逐一校验入口）。
【插件】插件内对应 skill 的适配要点与最小验证模型（底层实现以环境内 CANN 算子库为准）。
【技能库经验】先查既有路径再谈新增、source-backed 融合模式分析、逐项归因。
