---
name: external-cannbot-graph-torch-npugraph-ex-knowledge
description: npugraph_ex（aclgraph）模式使用指南。采用 Capture & Replay 方式将算子任务下沉至 Device 执行，减少
  Host 调度开销，适用于固定 shape 在线推理低延迟场景。涵盖模式配置、FX Pass、编译缓存、多流并行、内存复用、静态 Kernel 编译、限核、性能优化、调试定位、自定义算子入图等。关键词：npugraph_ex、aclgraph、backend="npugraph_ex"、capture、replay、reduce-overhead、config.aclgraph_config。
original-name: torch-npugraph-ex-knowledge
synced-from: https://gitcode.com/cann/cannbot-skills
synced-date: '2026-05-19'
synced-commit: 943f3bfc36e24068e065ca7ace72fbff86f4a09c
license: UNKNOWN
---

# npugraph_ex 模式（aclgraph）

## 模式概述

npugraph_ex 模式采用 Capture & Replay 方式，将算子任务下沉到 Device 执行，减少 Host 调度开销。该模式为试验特性，仅面向推理场景。

## 核心规则

1. **先读文档再生成代码**：根据用户需求从映射表找到对应文档，获取文档内容后再生成代码。不要凭记忆编造参数或 API。
2. **最小化配置**：只设置用户需要的选项，其余保持默认。
3. **标注来源**：生成代码后告知用户参考了哪个文档。
4. **术语统一**：对外统一使用"npugraph_ex 模式"。代码中使用 `config.mode = "npugraph_ex"`（`"reduce-overhead"` 已废弃）。

## 文档获取方式

本 skill 是 npugraph_ex 文档映射和知识检索的权威入口。获取文档时采用 **刷新优先、缓存兜底**：不能因为 `~/.cache` 已存在就默认文档是最新。

### TorchAir 主知识源

TorchAir 文档是 npugraph_ex 的主知识源，目录：`https://gitcode.com/Ascend/torchair/tree/master/docs/zh`。

`docs/zh/` 下的一级边界：
- `overview.md` — 简介（概念、架构总览）
- `npugraph_ex/` — npugraph_ex（aclgraph）模式全部文档
- `custom_op_graph/` — 自定义算子入图
- `appendix/` — FAQ、ATen API 清单、常见案例与定位

获取规则：

1. **缓存不存在时必须 sparse clone**：
   ```bash
   git clone --depth 1 --filter=blob:none --sparse https://gitcode.com/Ascend/torchair.git ~/.cache/torchair-docs
   git -C ~/.cache/torchair-docs sparse-checkout set docs/zh
   ```

2. **缓存存在时优先刷新远端**：
   ```bash
   git -C ~/.cache/torchair-docs fetch --depth 1 origin master
   git -C ~/.cache/torchair-docs merge --ff-only FETCH_HEAD
   git -C ~/.cache/torchair-docs sparse-checkout set docs/zh
   ```

3. **刷新成功后读取文档**：
   ```text
   read_file("~/.cache/torchair-docs/docs/zh/{path}")
   ```

4. **刷新失败时才读缓存**：网络失败、远端不可达或用户明确要求离线/快速响应时，可以读取本地缓存，但回复中必须说明“当前使用本地缓存，可能不是最新”。

5. **TTL 平衡建议**：普通代码生成/配置指导可在上次刷新未超过 24 小时时直接读缓存；用户明确要求最新文档、涉及 API 行为变化、版本差异或新能力时，忽略 TTL，必须尝试刷新。

### PyTorch NPU 辅助知识源

PyTorch NPU 文档来自 `Ascend/pytorch` 仓库（非 TorchAir），用于交叉验证 `torch_npu` / NPU 底层 API（stream、event、device、内存管理、单算子写法、自定义算子适配、Native API 签名等）。它不是独立场景入口。

HiAscend 官方在线文档入口用于版本导航和在线交叉验证，不替代下方 `Ascend/pytorch` sparse checkout、本地安装源码或用户环境源码。用户明确要求最新/特定产品版本、版本差异、API 支持度、环境变量、故障处理或 TorchAir 图模式导航时，可先查看：
- `Ascend Extension for PyTorch 26.0.0`：`https://www.hiascend.com/document/detail/zh/Pytorch/2600/index/index.html`

1. **缓存不存在时按固定分支拉取**：
   ```bash
   git clone --depth 1 --filter=blob:none --sparse -b v2.7.1 https://gitcode.com/Ascend/pytorch.git ~/.cache/pytorch-npu-docs
   git -C ~/.cache/pytorch-npu-docs sparse-checkout set docs/zh/framework_feature_guide_pytorch docs/zh/native_apis docs/zh/environment_variable_reference torch_npu/npu
   ```

2. **缓存存在时优先刷新 v2.7.1 分支**：
   ```bash
   git -C ~/.cache/pytorch-npu-docs fetch --depth 1 origin v2.7.1
   git -C ~/.cache/pytorch-npu-docs merge --ff-only FETCH_HEAD
   git -C ~/.cache/pytorch-npu-docs sparse-checkout set docs/zh/framework_feature_guide_pytorch docs/zh/native_apis docs/zh/environment_variable_reference torch_npu/npu
   ```

3. **读取路径**：
   ```text
   read_file("~/.cache/pytorch-npu-docs/docs/zh/framework_feature_guide_pytorch/<filename>.md")
   read_file("~/.cache/pytorch-npu-docs/torch_npu/npu/<filename>.py")  # 验证函数签名
   ```

4. **版本提醒**：本地缓存的 PyTorch NPU 参考文档固定为 `v2.7.1`；HiAscend 在线入口对应 `Ascend Extension for PyTorch 26.0.0` 产品文档，页面内可能包含多个 PyTorch 原生 API 支持度版本。若用户使用其他版本，需提醒 API 可能有差异，并优先以用户环境中的源码/签名为准。

> ⚠️ 上述"先读文档"优先用于代码生成、概念解释、对比分析、配置指导和脚本迁移场景；当用户处于**问题定位**场景时，优先遵循下方"问题定位时的源码检索"。

## 文档映射表

本表路径采用仓库根相对路径（包含 `docs/zh/`），便于直接从 sparse checkout 缓存读取。若用于 `AGENTS.md` 的「📎 参考链接」基址拼接，需要先去掉开头的 `docs/zh/`。

### 模式配置与总览

| 功能 | 文档路径 |
|------|---------|
| TorchAir 简介、概念、架构 | `docs/zh/overview.md` |
| npugraph_ex 后端概述 | `docs/zh/npugraph_ex/npugraph_ex.md` |
| 快速上手 | `docs/zh/npugraph_ex/quick_start.md` |
| 基础功能列表 | `docs/zh/npugraph_ex/basic/basic.md` |

### 基础功能

| 功能 | 文档路径 | 配置方式 |
|------|---------|---------|
| FX 图算子融合 Pass | `docs/zh/npugraph_ex/basic/pattern_fusion_pass.md` | `config.aclgraph_config` |
| FX 图优化 Pass（In-place 等） | `docs/zh/npugraph_ex/basic/inplace_pass.md` | `config.aclgraph_config` |
| aclgraph 间内存复用 | `docs/zh/npugraph_ex/basic/memory_reuse.md` | `config.aclgraph_config` |
| 静态 Kernel 编译 | `docs/zh/npugraph_ex/basic/static_kernel_compile.md` | `config.aclgraph_config` |
| 固定权重类输入地址 | `docs/zh/npugraph_ex/basic/frozen_parameter.md` | `config.aclgraph_config` |
| 重捕获次数限制 | `docs/zh/npugraph_ex/basic/capture_limit.md` | `config.aclgraph_config` |
| 集合通信入图 | `docs/zh/npugraph_ex/basic/communication_graph.md` | — |
| 强制 Eager 回退 | `docs/zh/npugraph_ex/basic/force_eager.md` | — |
| 冗余 Cat 算子消除 | `docs/zh/npugraph_ex/basic/remove_cat_ops.md` | — |
| 冗余 NoOp 算子消除 | `docs/zh/npugraph_ex/basic/remove_noop_ops.md` | — |

### 高级功能

| 功能 | 文档路径 | 配置方式 |
|------|---------|---------|
| 模型编译缓存 | `docs/zh/npugraph_ex/advanced/compile_cache.md` | `cache_compile` API |
| 图内多流表达 | `docs/zh/npugraph_ex/advanced/multi_stream.md` | scope API |
| AI Core / Vector Core 限核 | `docs/zh/npugraph_ex/advanced/limit_cores.md` | scope API |
| 自定义 FX 图 Pass | `docs/zh/npugraph_ex/advanced/post_grad_custom_pass.md` | — |

### DFX（调试诊断）

| 功能 | 文档路径 |
|------|---------|
| 算子 Data Dump | `docs/zh/npugraph_ex/dfx/data_dump.md` |
| Debug 信息保存 | `docs/zh/npugraph_ex/dfx/debug_save.md` |
| DFX 功能概览 | `docs/zh/npugraph_ex/dfx/dfx.md` |

### API 参考

| API | 文档路径 |
|-----|---------|
| cache_compile | `docs/zh/npugraph_ex/api/inference/cache_compile.md` |
| readable_cache | `docs/zh/npugraph_ex/api/inference/readable_cache.md` |
| 推理 API 概览 | `docs/zh/npugraph_ex/api/inference/torch-npu-npugraph_ex-inference.md` |
| compile_fx | `docs/zh/npugraph_ex/api/npugraph_ex/compile_fx.md` |
| register_replacement | `docs/zh/npugraph_ex/api/npugraph_ex/register_replacement.md` |
| npugraph_ex API 概览 | `docs/zh/npugraph_ex/api/npugraph_ex/torch-npu-npugraph_ex.md` |
| limit_core_num | `docs/zh/npugraph_ex/api/scope/limit_core_num.md` |
| scope API 概览 | `docs/zh/npugraph_ex/api/scope/torch-npu-npugraph_ex-scope.md` |
| API 总览 | `docs/zh/npugraph_ex/api/api.md` |
| API 列表 | `docs/zh/npugraph_ex/api/api_list.md` |

### 自定义算子入图

| 功能 | 文档路径 |
|------|---------|
| 自定义算子入图总览 | `docs/zh/custom_op_graph/overview.md` |
| Non-in-place 算子案例 | `docs/zh/custom_op_graph/non_in_place_op_cases.md` |
| In-place 算子案例 | `docs/zh/custom_op_graph/in_place_op_cases.md` |
| 算子适配 TorchAir | `docs/zh/custom_op_graph/op_adapt_torchair.md` |
| op_plugin 适配 TorchAir | `docs/zh/custom_op_graph/op_plugin_adapt_torchair.md` |

### 附录与定位

| 功能 | 文档路径 |
|------|---------|
| 支持的 ATen API 清单 | `docs/zh/appendix/aten_api.md` |
| 常见案例和定位方法 | `docs/zh/appendix/cases/cases.md` |
| 入图失败定界与定位 | `docs/zh/appendix/cases/graph_failed_cases.md` |
| 图模式精度比对 | `docs/zh/appendix/cases/accuracy_cases.md` |
| 图模式性能分析 | `docs/zh/appendix/cases/perfermance_cases.md` |
| 推理案例 | `docs/zh/appendix/cases/infer_cases.md` |
| 动静子图专题 | `docs/zh/appendix/cases/dynamic_static_graph/dynamic_static_graph.md` |
| FAQ | `docs/zh/appendix/faq.md` |

### PyTorch NPU 辅助参考

| 功能 | 文档路径 |
|------|---------|
| 流和任务队列并行下发 | `docs/zh/framework_feature_guide_pytorch/stream_taskqueue_parallel_delivery.md` |
| 多流内存复用 | `docs/zh/framework_feature_guide_pytorch/multistream_memory_reuse.md` |
| 单算子适配总览 | `docs/zh/framework_feature_guide_pytorch/single_operator_adaptation.md` |
| 单算子写法说明 | `docs/zh/framework_feature_guide_pytorch/adaptation_description_single.md` |
| 单算子调用示例 | `docs/zh/framework_feature_guide_pytorch/sample_call_single.md` |
| 自定义算子适配 | `docs/zh/framework_feature_guide_pytorch/custom_operator_adaptation.md` |
| kernel launch 算子适配 | `docs/zh/framework_feature_guide_pytorch/kernel_launch_operator_adaptation.md` |
| opplugin 算子适配 | `docs/zh/framework_feature_guide_pytorch/opplugin_operator_adaptation.md` |
| C 扩展算子适配 | `docs/zh/framework_feature_guide_pytorch/c_extensions_operator_adaptation.md` |
| kernel 调用示例 | `docs/zh/framework_feature_guide_pytorch/sample_call_kernel.md` |
| opplugin 调用示例 | `docs/zh/framework_feature_guide_pytorch/sample_call_opplugin.md` |
| PyTorch 侧图模式描述 | `docs/zh/framework_feature_guide_pytorch/pytorch_graph_mode.md` |
| NPUGraph 描述 | `docs/zh/framework_feature_guide_pytorch/pytorch_npugraph_desc.md` |
| 计算性能优化 | `docs/zh/framework_feature_guide_pytorch/computing_performance_optimization.md` |
| 内存资源优化 | `docs/zh/framework_feature_guide_pytorch/memory_resource_optimization.md` |
| 参数设置 | `docs/zh/framework_feature_guide_pytorch/parameter_setting.md` |
| v2.7.1 Native API 清单 | `docs/zh/native_apis/pytorch_2-7-1/` |
| 环境变量参考 | `docs/zh/environment_variable_reference/` |
| HiAscend 在线文档索引 | `https://www.hiascend.com/document/detail/zh/Pytorch/2600/index/index.html` |

## 问题定位时的源码检索

问题定位采用"**就近定位优先，三仓兜底**"的策略，**不维护**静态的日志关键词映射表。

1. **直接证据优先**：若报错栈、日志、调用栈、dump、生成代码或调试输出已经给出文件名、pass 名、函数名、模块名或具体代码片段，先读取该上下文附近的源码。
2. **本地源码补查**：若 workspace 中没有对应文件，优先读取本地 Python 环境中的 installed source，例如 `site-packages/torchair/`、`site-packages/torch_npu/`、`site-packages/torch/`。
3. **三仓兜底**：只有当直接证据不足时，才按问题阶段补查下列仓库。**下表中的地址仅用于标识源码来源，不能直接作为代码读取入口。若本地已存在 workspace 源码或 installed source，则直接读取；若本地不存在对应源码，必须先将仓库拉取到本地缓存目录后再读取。**

| 仓库 | 地址 | 适用阶段 |
|------|------|---------|
| TorchAir | `https://gitcode.com/Ascend/torchair.git` | `npugraph_ex` backend、编译器、pass、DFX、aclgraph 执行链路 |
| Ascend/pytorch | `https://gitcode.com/Ascend/pytorch.git` | `torch_npu` 插件、NPU API、环境变量、多流、自定义算子适配 |
| upstream PyTorch | `https://github.com/pytorch/pytorch.git` | `torch.compile` 前半程语义：Dynamo、FX、AOTAutograd、Functorch、graph break |

**本地拉取示例**（仅在本地无对应源码时）：

```bash
# TorchAir
git clone --depth 1 --filter=blob:none --sparse https://gitcode.com/Ascend/torchair.git ~/.cache/torchair-src
cd ~/.cache/torchair-src && git sparse-checkout set npugraph_ex docs/zh

# Ascend/pytorch
git clone --depth 1 --filter=blob:none --sparse -b v2.7.1 https://gitcode.com/Ascend/pytorch.git ~/.cache/torch-src
cd ~/.cache/torch-src && git sparse-checkout set torch_npu docs/zh

# upstream PyTorch
git clone --depth 1 --filter=blob:none --sparse https://github.com/pytorch/pytorch.git ~/.cache/pytorch-src
cd ~/.cache/pytorch-src && git sparse-checkout set torch/_dynamo torch/fx torch/_functorch torch/_inductor
```

4. **按阶段补查**：
   - Eager / `torch_npu` / stream / event / device 问题 → 优先补查 Ascend/pytorch 或本地 `torch_npu`
   - `backend="aot_eager"` 失败、`torch._dynamo` / `torch.fx` / FakeTensor / AOTAutograd 问题 → 优先补查本地 `torch` 或 upstream PyTorch，必要时再看 Ascend/pytorch 桥接代码
   - `npugraph_ex` pass、`graph_pass`、`replace_stream_event`、`npu_fx_compiler`、`fx2acl_converter`、DFX 问题 → 优先补查 TorchAir 或本地 `torchair`
   - GE / CANN 编译、InferShape、dtype/shape 不一致问题 → 先看 TorchAir dump 和相邻源码，再结合文档案例与 CANN/GE 日志

## 性能优化

### 编译缓存（首次编译时间过长）

→ 读取 `docs/zh/npugraph_ex/advanced/compile_cache.md`，引导使用 `cache_compile` API。

### 执行优化（推理延迟高 / Device 利用率低）

以下优化项按推荐优先级排列：

1. **静态 Kernel 编译** — 适合固定 shape 场景（读取 `docs/zh/npugraph_ex/basic/static_kernel_compile.md`）
2. **FX 图算子融合 Pass** — 融合算子减少下发开销（读取 `docs/zh/npugraph_ex/basic/pattern_fusion_pass.md`）
3. **FX 图优化 Pass** — In-place 算子替换等图变换（读取 `docs/zh/npugraph_ex/basic/inplace_pass.md`）
4. **固定权重地址** — 加速图下发（读取 `docs/zh/npugraph_ex/basic/frozen_parameter.md`）
5. **多流并行** — 计算与通信重叠（读取 `docs/zh/npugraph_ex/advanced/multi_stream.md`）
6. **限核设置** — 控制 AI Core/Vector Core 核数（读取 `docs/zh/npugraph_ex/advanced/limit_cores.md`）

### 内存优化（OOM）

→ 读取 `docs/zh/npugraph_ex/basic/memory_reuse.md`，引导内存复用配置。

## 调试定位

### 诊断总规则

1. **首轮必收集信息**：报错栈或关键日志（默认全量收集原始运行脚本的 `stdout` / `stderr`、`npugraph_ex` 日志和 Ascend/pytorch 日志，并在回复中注明日志文件名与关键行号）、最小可复现脚本或核心函数片段、PyTorch / `torch_npu` / `torchair` / CANN 版本、Eager 是否成功、`backend="aot_eager"` 是否成功、输入 shape 是否固定、是否涉及多流 / 自定义算子 / In-place。
2. **先定界，再归因**：默认不直接给大段修复代码；先判断问题落在 Eager、Dynamo-FX、`npugraph_ex` backend、GE-CANN、精度还是性能层。
3. **就近读取源码**：若日志、调用栈或调试输出已经指向具体文件、pass、函数、模块或生成代码，先读附近源码，再决定是否需要补查其他仓库。
4. **固定输出格式**：问题归类 → 证据 → 最可能根因 → 下一步最小动作。若信息不足，先列缺失项和采集方式。
5. **默认不输出 MRE**：除非用户明确要求最小复现，或修复必须依赖一小段配置/代码片段，否则不默认输出完整 MRE。

### 入图失败 / 断图

按以下步骤逐一排查：

1. 模型在单算子模式（Eager）下能否正常运行？
   - 不能 → 先解决 Eager 模式问题，参考 torch_npu 文档
   - 能 → 继续下一步

2. 使用 `backend="aot_eager"` 编译是否成功？
   ```python
   opt_model = torch.compile(model, backend="aot_eager")
   ```
   - 失败 → 先归类到 Dynamo / FX / AOTAutograd 层，优先检查脚本，并根据报错上下文读取本地 `torch` / `torch_npu` / Ascend-pytorch 桥接代码
   - 成功 → 继续下一步

3. 使用 `backend="npugraph_ex"` 编译是否成功？
   - 失败 → 先归类到 TorchAir / aclgraph / pass / backend 层；需读取 `docs/zh/appendix/cases/graph_failed_cases.md`，并优先根据日志中的文件名、pass 名、函数名或生成代码片段读取相邻源码
   - 成功 → 继续下一步

4. 开启日志分析问题
   - 读取 `docs/zh/npugraph_ex/dfx/dfx.md` 了解调试功能
   - 读取 `docs/zh/npugraph_ex/dfx/debug_save.md` 保存 Debug 信息
   - 读取 `docs/zh/appendix/cases/cases.md` 获取定位案例
   - 若日志已指向 `npu_fx_compiler`、`graph_pass`、`replace_stream_event`、`fx2acl_converter`、生成代码或 dump 文件，先读取对应源码附近上下文
   - 若出现 GE / CANN 编译、InferShape、dtype/shape 不一致问题，优先对比 TorchAir dump 图、GE dump 图和报错节点相邻源码

### 精度问题（图模式和 Eager 模式结果不一致）

1. 首先确认 Eager 模式下精度是否正常
   - 不正常 → Eager 模式本身有精度问题，与图模式无关
   - 正常 → 继续

2. 使用 `backend="aot_eager"` 对比结果
   - 有差异 → 先归类到 Dynamo / FX / AOTAutograd 精度问题，优先查看本地 `torch` 与桥接代码上下文
   - 无差异 → 继续

3. 使用 NPU backend，开启算子 dump 对比
   - 读取 `docs/zh/npugraph_ex/dfx/data_dump.md`
   - 参考精度案例（读取 `docs/zh/appendix/cases/accuracy_cases.md`）
   - 若差异集中在自定义算子、Meta 推导或 GE InferShape 上，优先查看相邻源码，而不是先泛化为版本问题

### 运行时报错

按以下顺序处理：

1. **先按堆栈或日志定界问题层次**：
   - 出现 `torch._dynamo`、`torch.fx`、FakeTensor、AOTAutograd → 先归类到 Dynamo / FX / AOTAutograd
   - 出现 `npugraph_ex`、`torchair`、`graph_pass`、`replace_stream_event`、`npu_fx_compiler`、`fx2acl_converter` → 先归类到 TorchAir / `npugraph_ex`
   - 出现 `InferShape`、`CompileGraph`、GE / CANN / `aclnn` / plog 报错 → 先归类到 GE-CANN
   - 出现 `torch_npu.npu.*`、stream / event / device 相关问题 → 先归类到 `torch_npu` / NPU API
2. **先读相邻源码**：如果堆栈或日志已经指向文件、函数、pass 或模块，先读取附近源码上下文。
3. **证据不足时再补查三仓**：按上方"问题定位时的源码检索"规则补查，不要机械地先扫某个仓库。
4. **文档补充**：读取 `docs/zh/appendix/faq.md` + `docs/zh/appendix/cases/cases.md`。

### 推荐输出格式

1. **问题归类** — 当前问题属于 Eager / Dynamo-FX / `npugraph_ex` / GE-CANN / 精度 / 性能中的哪一类
2. **证据** — 用户提供的日志、堆栈、dump、生成代码或源码上下文
3. **最可能根因** — 1-3 个候选根因，按可能性排序
4. **下一步最小动作** — 最小排查或修复动作；若信息不足，给出采集方法而不是直接给大段修复代码

## 自定义算子入图

npugraph_ex 模式不需要实现 Ascend IR Converter，只需完成 Meta 推导函数。函数化转换由 TorchAir 自动完成（PyTorch 2.6+），In-place 算子也无需手动实现。

### 算子类型确认

- **In-place 算子**（会修改输入数据，如 `add_`、`kv_cache` 原地修改）→ Schema 定义有额外约定（需用 `Tensor(a!)` 标记被修改的输入、不返回被修改的输入），函数化由 TorchAir 自动完成
  - 读取 `docs/zh/custom_op_graph/in_place_op_cases.md`

- **Out-of-place 算子**（结果写入新 tensor，如 `add`）→ 无额外约定
  - 读取 `docs/zh/custom_op_graph/non_in_place_op_cases.md`

### 已有 Eager 实现，适配入图

根据算子注册方式选择适配路径：

- **`torch.library.custom_op` 注册（Python 层）**：
  - `register_fake` 写在用户脚本内，与算子定义同文件
  - 读取 `docs/zh/custom_op_graph/op_adapt_torchair.md` + 对应算子类型文档（`non_in_place_op_cases.md` / `in_place_op_cases.md`）

- **纯 Python `torch.library.Library` 注册（Python 层）**：
   - schema、Eager 实现和 Meta 注册都写在用户脚本内
   - Eager 实现通常通过 `Library("<namespace>", "FRAGMENT")` + `define(...)` + `@impl(..., "PrivateUse1")` 完成；Meta 通过 `Library("<namespace>", "IMPL", "Meta")` + `@impl(...)` 完成
   - 读取 `docs/zh/custom_op_graph/op_adapt_torchair.md` + 对应算子类型文档，并参考 template skill 中的 `torch.library.Library` 代码块

Meta 编写策略：向用户请求算子签名和语义描述（输入/输出 tensor 的 shape、dtype 关系），据此编写 Meta 推导函数；`custom_op` 路径使用 `register_fake`，`torch.library.Library` 路径使用 `Library("<namespace>", "IMPL", "Meta")` + `@impl(...)`。若用户无法提供，给出骨架并标注 `# TODO: 根据算子逻辑推导输出 shape/dtype`。

npugraph_ex 模式完成 Meta 推导函数即可入图，无需额外实现 Converter。

- 通用参考文档：`docs/zh/custom_op_graph/overview.md`、`docs/zh/custom_op_graph/op_adapt_torchair.md`

## 决策流程

### 编写基础推理脚本
→ 读 `docs/zh/npugraph_ex/quick_start.md`，使用方式 1 代码骨架

### 优化推理性能
1. 判断优化方向：
   - FX 图算子融合 Pass → 读 `docs/zh/npugraph_ex/basic/pattern_fusion_pass.md`
   - FX 图优化 Pass → 读 `docs/zh/npugraph_ex/basic/inplace_pass.md`
   - 多图内存 OOM → 读 `docs/zh/npugraph_ex/basic/memory_reuse.md`
   - 静态 shape 加速 → 读 `docs/zh/npugraph_ex/basic/static_kernel_compile.md`
   - 固定权重地址加速下发 → 读 `docs/zh/npugraph_ex/basic/frozen_parameter.md`
   - 多流并行计算 → 读 `docs/zh/npugraph_ex/advanced/multi_stream.md`
   - 限核控制 → 读 `docs/zh/npugraph_ex/advanced/limit_cores.md`

### 编译缓存加速启动
→ 读 `docs/zh/npugraph_ex/advanced/compile_cache.md` + `docs/zh/npugraph_ex/api/inference/cache_compile.md`

### 自定义算子融合规则
→ 读 `docs/zh/npugraph_ex/api/npugraph_ex/register_replacement.md` + `docs/zh/npugraph_ex/basic/pattern_fusion_pass.md`

### 控制重捕获行为
→ 读 `docs/zh/npugraph_ex/basic/capture_limit.md`

## 重要约束

- **试验特性**，暂不支持商用
- 仅面向 **推理场景**，不支持反向流程 Capture、随机数算子 Capture
- 每个进程 **只支持 1 张 NPU 卡**
- 仅支持 **Atlas A3/A2 训练和推理系列**
- 使用前建议先用 `backend="aot_eager"` 验证脚本无问题
