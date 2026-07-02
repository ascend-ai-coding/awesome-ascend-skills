---
name: external-cannbot-model-model-infer-migrator
description: 基于 PyTorch 框架的昇腾 NPU 模型推理适配与部署基线技能。支持两种部署模式：框架部署模式（接入 cann-recipes-infer
  的 executor/core/）和独立部署模式（自管 Runner 不依赖框架）。从 HF 链接或本地代码适配为可运行的标准模型目录，并采集性能基线。触发场景：新模型适配到昇腾
  NPU 推理框架、已有模型的部署基线采集、模型迁移和初始跑通验证。
original-name: model-infer-migrator
synced-from: https://gitcode.com/cann/cannbot-skills
synced-date: '2026-07-02'
synced-commit: 342679f37ec1b052a8b64d6c9fd59f67e0c89073
license: UNKNOWN
---

# 模型适配与部署基线技能

把 HF 模型或本地代码适配到 infer 仓的两种部署模式之一，产出符合规范的模型目录和标准化性能基线。

本技能聚焦**单卡基础适配 + 部署基线采集**（eager 模式、PyTorch 标准算子），不涉及并行化、KV 模式优化、融合算子、图模式等性能改造。这些由后续 skill（parallel-impl / kvcache / fusion / graph-mode）接手。

---

## 部署模式判定

**框架部署模式（推荐）**：模型声明契约接入 `cann-recipes-infer/executor/core/`，调度 / KV 管理 / 批组装 / 采样 / PD 分离 / MTP / profiler 等公共流程由框架负责。模型通过 `support_models.py` 注册。

**独立部署模式**：模型不接 `cann-recipes-infer/executor/core/`，自管 Runner / KV cache / 推理脚本。适用仓外项目复用 skill 或框架契约不适用的特殊算法。

### 判定优先级

| 优先级 | 判定来源 | 说明 |
|---|---|---|
| 1 | **dispatch prompt 显式指定** | prompt 含 `部署模式: 框架部署` 或 `部署模式: 独立部署` → 直接采用，不再自检 |
| 2 | **用户对话指定** | 仅独立调用（用户直接 invoke skill）时适用，subagent 派发场景不能交互 |
| 3 | **工作目录自检** | 按下表识别 |

| 自检信号 | 部署模式 |
|---|---|
| 模型注册到 `cann-recipes-infer/executor/core/support_models.py` | **框架部署模式**（推荐） |
| 模型目录有 `runner_*.py` 且不继承任何 framework 基类，runner 自管推理循环 | **独立部署模式** |
| 不确定 / 新模型 | 默认 **框架部署模式** |

> 编排层（如 model-infer-optimize）派发本 skill 时不能与用户交互，应在 dispatch prompt 中显式带 `部署模式: ...`；缺失时按工作目录自检，仍无法确定则默认框架部署。用户直接调用本 skill 且信号不明确时，可向用户确认。

判定后跳转：
- 框架部署模式 → 「## 框架部署模式」大节
- 独立部署模式 → 「## 独立部署模式」大节

---

## 覆盖场景（共通）

| 场景 | 输入状态 | 处理 |
|------|---------|------|
| A | 只有 HF 链接，无代码 | 下载 + 适配 + 生成标准文件 + 基线采集 |
| B | 有本地代码但跑不通 | 诊断修复（最多 5 轮）+ 基线采集 |
| C | 代码可运行但无标准化基线 | 直接基线采集 |

---

## 共通工作流

### Step 1: 环境检测 + 部署模式判定

确认 NPU 可用（`npu-smi info`）、torch 和 torch_npu 版本匹配。按上面「部署模式判定」表选择模式。

### Step 2: 场景判断

- 无 infer.sh → 场景 A → Step 3
- 有 infer.sh → 执行一次：成功则场景 C → Step 4，失败则场景 B → Step 3
- NPU 环境不可用 → 报告环境问题，结束

### Step 3: 代码准备（场景 A/B）

跳到对应大节执行：
- 框架部署模式 → 「## 框架部署模式」适配要点
- 独立部署模式 → 「## 独立部署模式」适配要点

### Step 4: 试运行 + 基线采集

执行 `bash infer.sh`，按结果分流：

- **跑通**：用 `dataset: "default"`（读 `cann-recipes-infer/dataset/default_prompt.json`）采集基线
  ```bash
  python3 scripts/collect_baseline.py \
      --log-file {log_path} \
      --output {output_dir}/{model_name}/agentic/baseline/baseline_metadata.json \
      --yaml-file {yaml_path} \
      --rank 1
  ```
  输出吐字正常（可读、不重复、长度合理），baseline_metadata.json 中的 output_text 需记录完整输出。多卡推理读 rank 1 的 log（rank 0 经 tee 可能截断）。
  - **补采子场景**：reviewer log 已有 + 代码未提交新 commit + YAML 未变 → 跳过 `bash infer.sh` 重跑，直接用既有 log 调脚本提取即可

- **OOM（显存不足）**：本阶段输出代码骨架并标记"需多卡"，由 optimize 编排进入并行化阶段；并行 + kvcache 等优化跑通后由编排层补采基线。产物：多卡占位 YAML（待 parallel-impl 接入）+ 空 `agentic/baseline/` 目录 + README 标注 OOM 预期与下游 skill 任务；单卡 YAML 跑不通时不必保留。

- **其他错误**：参考 `references/common_issues.md`；NPU 运行时错误 → model-infer-runtime-debug；精度异常 → model-infer-precision-debug。

日志路径：
- 框架部署模式：`${RES_PATH}/log_0.log`（由 launch 创建）
- 独立部署模式：`infer.py` 输出到 stdout / 自定义日志路径

---

## 产物文件结构

两模式产物**文件命名 100% 一致**，YAML schema 同 4 段式：

```
{output_dir}/{model_name}/
├── config/{model_name}.yaml
├── cann-recipes-infer/models/
│   ├── configuration_{model_name}.py
│   └── modeling_{model_name}.py
├── infer.py                      ★ 独立部署模式自带；框架部署模式不带（统一走 cann-recipes-infer/executor/offline/infer.py，不推荐自带）
├── infer.sh
├── runner_{model_name}.py        ★ 独立模式独有
├── requirements.txt
├── README.md
└── agentic/baseline/baseline_metadata.json
```

> 注意：`cann-recipes-infer/models/{model_name}/models/` 下**不要放 `__init__.py`**。

---

## 框架部署模式

接入 `cann-recipes-infer/executor/core/` 框架，模型走 `InferenceConfig + CommManager + cache_unit` 契约，启动通过 `cann-recipes-infer/executor/offline/infer.py + launch`。本阶段为最简部署基线，KV 先用 Legacy 模式跑通；PA 等进一步改造由 kvcache 优化阶段处理。

> 模型类作为框架扩展点与 ModelWorker / ExecutionEngine / Scheduler / InferenceConfig / CommManager / ForwardMetaData 协作；调度、KV 管理、批组装、采样由框架统一负责。

### 适配要点

1. **复制 HF 实现到 `cann-recipes-infer/models/{model_name}/models/`**：vendoring HF modeling 和 configuration（**禁止 `from transformers import 模型类`**）。删除训练专用代码，保留 PyTorch 标准算子。modeling 类基类改为 `nn.Module`。
   - **命名规则**：目录 / 文件 / 模块名用下划线
   - **HF 依赖剪裁**（通用替换）：
     - `Cache` / `DynamicCache` → 删，用 Legacy `self.k_cache` / `v_cache`
     - `create_causal_mask` → 删，用 `forward_metadata.attention_mask`
     - `PreTrainedModel` / `GradientCheckpointingLayer` / `ALL_ATTENTION_FUNCTIONS` → 删，改为 `nn.Module`
     - 纯函数（RMSNorm / rotate_half / RoPE 工具）→ 保留

2. **改造 ForCausalLM 构造签名**为 `(config, infer_config, comm_manager, prefix)`。读取参数从 `runner_settings.get(...)` 改为 `infer_config.parallel_config.attn_tp_size` 等点访问；通信组从 `comm_manager.get_group(...)` 取。

3. **改造 forward 签名**为 `(input_ids, position_ids, forward_metadata, slot_mapping=None, block_table=None, **kwargs)`。input_ids 是 packed `[TotalTokens]`，运行时元数据从 `forward_metadata` 读，返回 packed logits。

4. **KV 路径选择**（按场景二选一，互斥；框架按是否存在 `get_cache_info()` 分发）：
   - **路径 A — Legacy + HF SDPA**（默认推荐 — 单卡 + 等长 batch baseline，最简）
     - Attention 声明 `self.cache_unit = (num_kv_heads_per_rank * head_dim,)`，框架按 `(batch, seq, *cache_unit)` 预分配 `k_cache` / `v_cache`；模型 forward 内用标准 indexing 或 `torch_npu.scatter_update_` 写 cache
     - 顶层 forward 入口对 packed 1D `input_ids` 做 BSND reshape 回 `[B, S]`（gemma-4 BSND shim 套路；仅等长 batch 适用）
     - 不实现 `get_cache_info()` / 不声明 `cache_entries`
   - **路径 B — 最简 Paged + HF SDPA**（多卡 / 变长 batch / 用户明确指定 Paged 起步时使用）
     - 实现 `get_cache_info()` + `cache_entries`（`FullAttention`，按实际 KV head 维度，不引入 MLA latent 压缩）
     - 写 cache 用 `npu_scatter_nd_update_` 接 `forward_metadata.slot_mapping[attn_type]`；attention 读 cache 后用 HF 原版 SDPA / 标准 PyTorch（**不**上 FA 算子）
     - 顶层 forward 直接走 packed 1D，无需 BSND shim
   - **共通约定**：
     - 两路径互斥，选定后整套代码（cache 写入、attention 读法、forward 输入形态）保持一致
     - reference 已注册模型的 `get_cache_info()` 是 kvcache skill 升级后的产物（含 MLA absorb / FA / nope+rope 双 entry），**不要整段照搬**到 migrator 阶段
     - 模型自有 cache（非 KVCache，如 N-gram / hash buffer / speculative buffer）由 model class 自管，不上报 `cache_entries`
     - FA 算子接入 / MLA absorb / 复杂 attn_type（SlidingWindow / 混合）/ latent 压缩 由 kvcache skill 接手

5. **实现权重契约**：`load_weights(weights)` 接 HF checkpoint stream；`process_weights_after_loading()` 触发 `quant_method.process_weights_after_loading`（MoE / 量化模型必须实现）。

6. **注册到 `cann-recipes-infer/executor/core/support_models.py`**：
   ```python
   model_dict = {"{model-key}": ({ModelName}ForCausalLM, {ModelName}Config)}
   ```
   含 MTP 的注册为 3 元组：`("key", MainCausalLM, MTPClass, Config)`。

7. **写 YAML、infer.sh、requirements、README**：模板见 `references/framework_templates.md`。`requirements.txt` 优先对齐仓内已注册同架构模型（参考路由表），缺失的再按 HF 原仓补；避免引入与既有模型冲突的版本约束。

> migrator 输出框架契约骨架 + Legacy KV + HF 原版 attention 的 `world_size=1` 单卡代码。FA 算子改造、Paged 改造、复杂架构 KV、多卡部署 / 权重切分由 kvcache / parallel-impl skill 接手；本阶段不做。

### 模型类契约速查

| 钩子 | 签名 | 职责 |
|---|---|---|
| `__init__` | `(config, infer_config, comm_manager, prefix)` | 构造子模块 |
| `forward` | `(input_ids, position_ids, forward_metadata, slot_mapping=None, block_table=None, **kwargs)` | 走前向，返回每请求末位 token logits `[batch_size, 1, vocab_size]`（Prefill 取 `cu_seq_q - 1`，Decode 本就 1 token） |
| `load_weights` | `(weights: Iterable[(name, tensor)])` | 加载 HF 权重 |
| `process_weights_after_loading` | `(self)` | 触发量化/MoE 权重后处理（框架自动调用） |
| `check_model_settings` | `(self)` | 模型配置一致性校验（attn_tp_size 整除约束、num_heads 切分可行性等），框架自动调用，校验失败 raise RuntimeError |
| `cache_unit`（migrator 用此走 Legacy） | per-Attention 属性 `(num_kv_heads_per_rank * head_dim,)` | 框架据此为各层静态预分配 `k_cache` / `v_cache` |
| `get_cache_info`（kvcache 阶段改造为 Paged 时填） | `() -> ModelCacheInfo` | 声明 cache_entries 让框架走 Paged 模式 |

完整签名、ForwardMetaData 字段表、CommManager 通信组命名见 `references/framework_templates.md`。

### 参考模型路由表

已在 `cann-recipes-infer/executor/core/support_models.py` 注册的参考实现：

| 架构 | 参考模型 | 路径 |
|---|---|---|
| 标准 LLM（MHA/GQA） | `qwen3_8b` / `qwen25_7b_instruct` | `cann-recipes-infer/models/qwen/` |
| MoE（标准 attention） | `qwen3-moe` / `gpt-oss` | `cann-recipes-infer/models/qwen3_moe/`、`cann-recipes-infer/models/gpt_oss/` |
| MoE + MLA + MTP | `deepseek_r1` | `cann-recipes-infer/models/deepseek_r1/` |

---

## 独立部署模式

不依赖 cann-recipes-infer 框架，模型自管 Runner / 推理脚本。仍可复用 `cann-recipes-infer/module/linear.py` 的 ParallelLinear（独立加载）和 `cann-recipes-infer/executor/model_loader/weight_utils.py` 的权重加载工具。

### 适配要点

1. **复制 HF 实现到 `cann-recipes-infer/models/{model_name}/models/`**：vendoring HF modeling 和 configuration，删除训练专用代码。

2. **改造 modeling forward 接口**为自定义参数（不用 ForwardMetaData）：
   ```python
   def forward(self, input_ids, position_ids=None,
               past_key_values=None, kv_len=None,
               attention_mask=None, is_prefill=True)
   ```
   Attention 层接收 `past_key_value` + `kv_len`，用 `torch_npu.scatter_update_` 更新 KV。decode 分支 `attention_mask=None`。

3. **写自定义 Runner 骨架**（`runner_{model_name}.py`，不继承 ModelRunner）：
   - `__init__(yaml_file_path)`: 解析 yaml；加载 tokenizer / config / model；预创建 mask；预计算 RoPE
   - `init_static_kvcache`: 最简占位（每层一对 (k, v) tensor，BSH layout，简单 zeros 分配）
   - `model_generate(prompts, warm_up=False)`: tokenize batch → prefill → decode loop → detokenize；日志格式 `{model_name} inference time cost of {stage} is X ms` 与仓库对齐

4. **写 infer.py**：实例化 Runner（传 yaml 路径），先 `model_generate(prompts, warm_up=True)` 预热，再 `warm_up=False` 正式跑。

5. **写 infer.sh、requirements、README**：模板见 `references/standalone_templates.md`。

> migrator 输出 Runner 骨架 + 最简 KV 占位 + HF 原版 attention 的单卡代码。FA 算子接入、PA 启用、SlidingWindow / MLA absorb 等优化由 kvcache skill 接手；多卡 `dist.*` + ParallelLinear 替换由 parallel-impl skill 接手。

### Runner 契约速查

| 方法 | 职责 |
|---|---|
| `__init__(yaml_file_path)` | 解析 yaml（model_path / scheduler / data 段）；加载 tokenizer / config / model；预计算 mask、RoPE |
| `init_static_kvcache(batch, seq)` | 基础 KV 分配（默认连续缓存，list of (k, v) tuples） |
| `model_generate(prompts, warm_up=False)` | 批量推理：prefill + decode loop + detokenize；warmup 短路计时与输出；日志格式与仓库 framework 模式对齐 |
| `prefill(...)` / `decode(...)`（可选拆分） | 单步执行；decode 可 torch.compile |

---

## 权重管理（共通）

通过 YAML 的 `model_config.model_path` 指定权重路径。

```
权重已在本地？
  ├─ 是 → 填入 YAML model_config.model_path
  └─ 否 → 权重大小？
           ├─ < 5GB → 下载到本地（huggingface-cli / snapshot_download / modelscope）
           └─ >= 5GB → 提示用户自行下载并提供路径
```

> migrator 阶段单卡跑通即可；并行权重切分、在线/离线切分由 parallel-impl skill 处理。

---

## baseline_metadata.json 格式（共通）

```json
{
  "timestamp": "<ISO8601>",
  "yaml_path": "config/{model_name}.yaml",
  "environment": {
    "npu_model": "<npu_model>",
    "num_cards": <int>,
    "cann_version": "<x.y.z>",
    "pytorch_version": "<x.y.z>",
    "torch_npu_version": "<x.y.z>",
    "exe_mode": "<eager|ge_graph|npugraph_ex>"
  },
  "model_config": {
    "model_name": "<model_name>",
    "model_source": "<hf_url_or_local_path>"
  },
  "performance": {
    "prefill_ms": <float>,
    "decode_avg_ms": <float>,
    "output_text": "<sample_output>"
  },
  "verification": {
    "all_ranks_success": <bool>
  }
}
```

> `yaml_path` 指向运行时使用的 YAML，`parallel_config` / `scheduler_config` 等并行/调度配置不重复嵌入 JSON。`decode_avg_ms` 须剔除首步冷启（编译落地开销，约 30-50ms 偏高）。分析推算（非 runtime 实测）的性能字段加 `_estimated` 后缀以示区分。

---

## 结束条件

共通条目（必须）：

1. 标准文件结构完整（modeling + configuration + YAML + infer.sh + requirements + README）
2. modeling 框架契约骨架到位（构造签名 / forward 签名 / 权重契约；KV 按部署模式接入：框架部署用 Legacy `cache_unit`，独立部署用最简 list of (k, v)）

跑通态条目（按场景二选一）：

- **单卡可跑通**：`bash infer.sh` 跑通 + 输出吐字正常 + agentic/baseline/baseline_metadata.json 已生成
- **单卡 OOM**：标记"需多卡"，不必跑通；baseline 由 optimize 编排在并行化后补采

部署模式专属：

- **框架部署模式**：模型已注册到 `cann-recipes-infer/executor/core/support_models.py`
- **独立部署模式**：含 `runner_{model_name}.py` + 自带 `infer.py`

---

## 参考文档索引

| 文档 | 路径 | 适用模式 |
|------|------|---|
| 框架部署模式模板和契约速查 | `references/framework_templates.md` | 框架部署 |
| 独立部署模式模板和 Runner 速查 | `references/standalone_templates.md` | 独立部署 |
| 常见问题速查 | `references/common_issues.md` | 共通 |
| 基线采集脚本 | `scripts/collect_baseline.py` | 共通 |
| InferenceConfig 字段语义（仓库框架） | `cann-recipes-infer/docs/common/inference_config_guide.md` | 框架部署 |
| 框架架构总览（仓库框架设计） | `cann-recipes-infer/docs/design/executor_design.md` | 框架部署 |
| KV cache 设计（仓库框架的 paged 实现） | `cann-recipes-infer/docs/design/kv_cache_design.md` | 框架部署 |
| 在线推理 / PD 分离（仓库框架机制） | `cann-recipes-infer/docs/design/online_inference_design.md` | 框架部署 |
| MTP 投机采样（仓库框架流程） | `cann-recipes-infer/docs/design/mtp_design.md`、`cann-recipes-infer/docs/common/mtp_model_guide.md` | 框架部署 |
| 在线切分权重（仓库框架流程） | `cann-recipes-infer/docs/common/online_split_weight_guide.md` | 框架部署 |
