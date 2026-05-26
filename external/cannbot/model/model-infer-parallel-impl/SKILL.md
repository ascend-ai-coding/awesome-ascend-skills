---
name: external-cannbot-model-model-infer-parallel-impl
description: 基于 PyTorch 框架的昇腾 NPU 模型推理并行切分实施技能。根据已确认的 parallel_config，实施模型代码的并行化改造，包括并行线性层替换、MoE
  并行模式适配、通信组创建、Embedding/LMHead 并行、YAML 配置生成和权重转换。触发场景：model-infer-parallel-analysis
  完成后需要实施改造、现有模型需要支持新的并行配置。
original-name: model-infer-parallel-impl
synced-from: https://gitcode.com/cann/cannbot-skills
synced-date: '2026-05-26'
synced-commit: ac5bbd2b4cf427d011874e11f8d1e8b1bef66eda
license: UNKNOWN
---

# 模型并行切分实施

> **前置条件**：
> 1. parallel_config 已由 `/model-infer-parallel-analysis` 确定并经用户确认
> 2. 目标模型已有完整的单卡框架适配代码（modeling + Runner + infer.py + infer.sh + YAML），本 skill 在此基础上做并行化改造

---

## 适用范围

- 模型代码的并行层替换
- 通信组创建和管理
- YAML 配置文件生成
- 权重转换脚本编写

## 实施流程

```
第一步：确认输入 + 选择参考模型
    ↓
第二步：通信组创建
    ↓
第三步：逐模块并行层替换
    ↓
第四步：Embedding / LMHead 并行
    ↓
第五步：YAML 配置生成
    ↓
第六步：权重处理
    ↓
第七步：验证
```

**禁止**：跳过第二步直接替换层

---

## 第一步：确认输入 + 选择参考模型

### 1.1 确认 parallel_config

从编排层（主 agent prompt 或用户输入）获取已确认的配置：

```yaml
parallel_config:
  attn_tp_size: {value}       # attn_dp_size = world_size // attn_tp_size
  dense_tp_size: {value}
  moe_tp_size: {value}        # moe_ep_size = world_size // moe_tp_size
  embed_tp_size: {value}
  lmhead_tp_size: {value}
  o_proj_tp_size: {value}     # MLA 模型需要，非 MLA 可省略
  cp_size: {value}            # 长序列 Prefill 需要，Decode 可省略
  kvp_size: {value}           # 超长序列需要，普通场景可省略
```

### 1.2 选择参考模型

根据目标代码实现模式，选择仓库中最接近的已适配模型阅读其 modeling 代码：

| 实现模式 | 参考模型 | 关注点 |
|---------|---------|--------|
| 标准 GQA + 纯 TP | GPT-OSS | QKVParallelLinear 基础用法 |
| 标准 GQA + MoE EP | Qwen3-MoE | MoE routing + npu_swiglu 融合 |
| MLA + 模块差异化 TP + EP | DeepSeek-R1 / V3.2 | MLA 投影切分、oproj 独立 group、EP dispatch/combine |
| MLA + 差异化 TP + KVP/AFD | LongCat-Flash | KVP 的 oproj 对齐约束、模块间数据重排、AFD 通信 |
| MLA + CP + 差异化 TP | Kimi-K2 / GLM-5 | 多组 comm group 创建（参考 infer.py）、CP KV 分片 |

**必须**：读取参考模型的 `modeling_*.py`（并行层替换）和 `infer.py`（通信组创建），了解实际实现方式。
配置数值参考 model-infer-parallel-analysis skill 的 `references/config-index.md`。

### 完成标志

- [ ] parallel_config 已确认
- [ ] 参考模型已选定并阅读关键代码

---

## 配置参数 → 实施步骤速查

根据 parallel_config 中各参数值，确定需要执行哪些步骤：

| 参数条件 | 需要的通信组（第二步） | 需要的代码改造（第三/四步） |
|---------|---------------------|--------------------------|
| `attn_tp_size > 1` | `attn_tp_group` | Attention QKV/O 替换为 ParallelLinear |
| `attn_tp_size = 1` 且 `world_size > 1` | 无 Attention TP 组（走 DP） | Attention 不需要 TP 替换 |
| `dense_tp_size > 1` | `dense_tp_group` | Dense FFN Gate/Up/Down 替换 |
| `moe_tp_size > 1` | `moe_tp_group` | MoE 专家 FFN 做 TP 切分 |
| `moe_tp_size = 1` 且有 MoE | `moe_ep_group`（需 HCCL group name） | MoE EP dispatch/combine 实现 |
| `o_proj_tp_size ≠ attn_tp_size` | `oproj_tp_group`（独立组） | O_proj 使用独立通信组 |
| `embed_tp_size > 1` | `embed_tp_group` | Embedding 替换为 VocabParallelEmbedding |
| `lmhead_tp_size > 1` | `lmhead_tp_group` | LMHead 替换为 ColumnParallelLinear |
| 相邻模块 TP 度不同 | — | 模块间插入 AllGather/ReduceScatter 重排 |

> `tp_size = 1` 的模块不需要 TP 替换和对应通信组。先扫一遍 parallel_config，标注哪些模块需要改，再逐步执行。

---

## 第二步：通信组创建

并行层替换前，必须先在模型初始化中创建通信组。

### 2.1 通信组获取

**当前仓库**：所有模型通过 `**kwargs` 传入 `hccl_comm_dict`（普通 dict）

```python
class Model(nn.Module):
    def __init__(self, config, runner_settings, **kwargs):
        self.hccl_comm_dict = kwargs.get("hccl_comm_dict", None)
        self.attn_tp_group = self.hccl_comm_dict["attn_tp_group"]
        self.moe_ep_group = self.hccl_comm_dict["moe_ep_group"]
        self.embed_tp_size = runner_settings.get("embed_tp_size", 1)
```

**重构版**：通过 `CommManager` 对象封装通信组

```python
class Model(nn.Module):
    def __init__(self, config, infer_config, comm_manager):
        tp_rank = comm_manager.get_rank("attn_tp_group")
        ep_rank = comm_manager.get_rank("moe_ep_group")
```

> 根据目标仓库版本选择模式。当前仓库所有模型使用 hccl_comm_dict。

### 2.2 模块级差异化并行的通信组

当 attn_tp ≠ dense_tp ≠ moe_tp 时，各模块使用不同的通信组：

```python
# 示例：attn_tp=1, dense_tp=8, moe_tp=1
self.attn_tp_group = self.hccl_comm_dict.get("attn_tp_group", None)   # size=1
self.dense_tp_group = self.hccl_comm_dict.get("dense_tp_group", None) # size=8
self.moe_ep_group = self.hccl_comm_dict["moe_ep_group"]              # EP 通信组
```

**通信组创建**：在模型的 `init_parallel_comm_group()` 方法中完成。
参考：DeepSeek-R1、Kimi-K2 的 `modeling_deepseek.py` 中均有此方法。

### 2.3 DP 大小自动计算

```python
attn_dp_size = world_size // attn_tp_size
moe_dp_size = world_size // moe_tp_size
moe_ep_size = moe_dp_size  # EP size = DP size
embed_dp_size = world_size // embed_tp_size
```

### 完成标志

- [ ] 所有需要的通信组已在 infer.py 中创建
- [ ] 各组的 rank 划分正确
- [ ] DP size 自动计算逻辑正确

---

## 第三步：逐模块并行层替换

> 并行层替换同时支持了权重在线切分：ParallelLinear 内置 `weight_loader()`，按 `tp_rank` 自动加载切片。权重处理详见第六步。

### 3.1 Attention 层（当 `attn_tp_size > 1` 时）

QKV → `QKVParallelLinear`，O → `RowParallelLinear`，均使用 `attn_tp_group`。
MLA 模型 O_proj 可能使用独立的 `oproj_tp_group`。

### 3.2 Dense FFN 层（当 `dense_tp_size > 1` 时）

Gate/Up → `ColumnParallelLinear`，Down → `RowParallelLinear`，均使用 `dense_tp_group`。

### 3.3 MoE 层（有 MoE 时必须处理）

- `moe_tp_size > 1`：专家 FFN 做 TP 切分 + AllReduce
- `moe_tp_size = 1`：EP 模式，Prefill 用 re_routing + AllToAll，Decode 用 dispatch/combine + AllToAll

> MoE 并行与融合算子紧密耦合，详见 `{file:./references/moe-parallel.md}`（含完整代码和算子说明）

### 3.4 模块间数据重排（当相邻模块 TP 度不同时）

边界处需要 AllGather/ReduceScatter 做数据重排。

### 3.5 CP / KVP 实施

> TODO: CP 见 `cann-recipes-infer/models/deepseek-v3.2-exp/`，KVP 见 `cann-recipes-infer/models/longcat-flash/`。

> Attention/Dense FFN/Embed/LMHead 的代码示例和数据重排示例见 `{file:./references/code-examples.md}`

### 完成标志

- [ ] Attention 层 QKV/O 已替换为并行版本
- [ ] Dense FFN 层 Gate/Up/Down 已替换（如有差异化 tp）
- [ ] MoE 层已按选定模式实现（TP 或 EP Prefill/Decode）
- [ ] 模块间数据重排已正确插入（不同 TP 度边界的 AllGather/ReduceScatter）
- [ ] 各模块使用了正确的通信组

---

## 第四步：Embedding / LMHead 并行

当 `embed_tp_size > 1` 或 `lmhead_tp_size > 1` 时需要并行化。

### 4.1 Embedding 并行

```python
from module.linear import VocabParallelEmbedding

# 按词表维度切分（参数为 tp_size + tp_rank，无 tp_group）
self.embed_tokens = VocabParallelEmbedding(
    config.vocab_size,
    config.hidden_size,
    self.padding_idx,
    torch.bfloat16,
    tp_size=self.embed_tp_size,
    tp_rank=dist.get_rank(self.hccl_comm_dict["embed_tp_group"]) if self.embed_tp_size > 1 else 0,
)
```

### 4.2 LMHead 并行

```python
self.lm_head = ColumnParallelLinear(
    config.hidden_size,
    config.vocab_size,
    tp_size=self.lmhead_tp_size,
    tp_rank=dist.get_rank(self.hccl_comm_dict["lmhead_tp_group"]) if self.lmhead_tp_size > 1 else 0,
    # gather_output 由框架处理
)
```

### 完成标志

- [ ] Embedding 已按 embed_tp_size 并行化
- [ ] LMHead 已按 lmhead_tp_size 并行化
- [ ] 约束检查通过

---

## 第五步：YAML 配置生成

为每种部署场景生成独立的 YAML 配置文件。

### 5.1 配置模板

```yaml
model_name: "{model_name}"
model_path: "/path/to/weights"
exe_mode: "eager"                    # 初始用 eager，后续可切 ge_graph
world_size: {W}

model_config:
  enable_pa: True
  pa_block_size: 128
  enable_weight_nz: True
  with_ckpt: True
  enable_online_split_weight: True   # 推荐：运行时自动按 tp_rank 切分权重，改配置无需重新转换
  enable_multi_streams: False
  enable_profiler: False
  # MoE 专用
  moe_chunk_max_len: {1024 for Decode, 65536 for Prefill}
  perfect_eplb: False

data_config:
  dataset: "default"
  input_max_len: {根据场景}
  max_new_tokens: {根据场景}
  batch_size: {根据显存估算}

parallel_config:
  attn_tp_size: {value}
  dense_tp_size: {value}
  moe_tp_size: {value}
  embed_tp_size: {value}
  lmhead_tp_size: {value}
  o_proj_tp_size: {value}     # MLA 模型需要
  cp_size: {value}            # 长序列 Prefill 需要
  kvp_size: {value}           # 超长序列需要
```

### 5.2 命名规范

```
config/
├── {model_name}_rank_{W}_{W}ep_decode.yaml          # Decode 纯 EP
├── {model_name}_rank_{W}_{tp}tp_prefill.yaml        # Prefill 纯 TP
├── {model_name}_rank_{W}_densetp{n}_ep{m}.yaml      # 混合模式
└── ci/
    └── {model_name}_ci.yaml                          # CI 测试用
```

### 5.3 CI 配置（可选）

仓库有 `config/ci/` 目录存放精简配置用于自动化测试。如需 CI 覆盖，参考已有配置生成。

### 完成标志

- [ ] 每种部署场景有独立的 YAML 文件
- [ ] 配置文件命名符合规范

---

## 第六步：权重处理

并行层替换后，需要确保权重能正确加载到各卡。

### 6.1 在线权重切分（推荐）

启用 `enable_online_split_weight: True`，框架在运行时通过各模块的 `weight_loader()` 自动按 rank 加载对应切片。所有 rank 读同一份完整 checkpoint，各自只保留本卡需要的部分。

**TP 权重加载**：`ColumnParallelLinear` / `RowParallelLinear` 的 `weight_loader` 按 `tp_rank` 取对应列/行切片。

**EP 权重加载**（MoE 模型）：`FusedMoEGMM` 的 `weight_loader` 按 `ep_rank` 过滤专家——只保留 `[ep_rank * experts_per_rank, (ep_rank+1) * experts_per_rank)` 范围内的专家权重，丢弃其他。`load_weights()` 中需通过 `make_expert_params_mapping(num_experts=...)` 生成全局专家映射，逐个传入 `weight_loader(expert_id=...)`。

适配要点（详见 `docs/common/online_split_weight_guide.md`）：
- 模型类须实现 `load_weights()` 方法，遍历权重文件并匹配到各模块的 `weight_loader()`
- MoE 模型必须调用 `make_expert_params_mapping` 生成专家权重映射
- `MergedColumnParallelLinear`（如 gate+up 合并）需要特殊的 weight_loader 处理 slice 顺序
- `process_weights_after_loading()` 处理权重转置和 NZ 格式转换

### 6.2 离线权重转换（备选）

若模型未实现 online split，或需要预切权重用于离线部署：

```bash
bash utils/weight_convert.sh \
    --input_path /path/to/origin \
    --output_path /path/to/output \
    --world_size {W} \
    --quant_mode {w8a8/w8a8c8/...}
```

输出为 `rank_0/` ~ `rank_N/` 目录结构，每个 rank 只包含该卡需要的权重切片。

> 注意：offline 预切的权重与 parallel_config 绑定。改了配置必须重新转换。

参考实现：`cann-recipes-infer/models/deepseek_r1/utils/convert_model.py`

### 完成标志

- [ ] 权重加载方式已确定（online split / offline convert）
- [ ] 若 online split：`load_weights()` 和 `weight_loader()` 已实现
- [ ] 若 offline convert：转换脚本已编写或复用，输出目录结构正确

---

## 第七步：验证

### 7.1 配置校验（优先复用现有校验函数）

```python
# 当前仓库没有统一的 scripts/validate_config.py。
# 优先复用 executor/utils/common_utils.py 中的 check_common_parallel_settings，
# 再调用模型自己的 model_setting.py / infer.py 中的 check_parallel_settings / check_vars。
assert world_size % attn_tp_size == 0
assert world_size % moe_tp_size == 0
assert world_size % embed_tp_size == 0
assert world_size % lmhead_tp_size == 0
assert num_attention_heads % attn_tp_size == 0
assert num_key_value_heads % attn_tp_size == 0  # GQA
assert num_experts % ep_size == 0               # MoE
assert embed_tp_size >= attn_tp_size
assert embed_tp_size % attn_tp_size == 0
```

### 7.2 功能验证

1. 确认推理实际加载的是修改后的代码（检查模型注册表、import 路径、日志中的模块路径等，确保运行时代码路径与修改路径一致）
2. 修改 `infer.sh` 中 `YAML_FILE_NAME` 指向目标配置
3. 执行 `bash infer.sh`
3. 检查 Prefill + Decode 推理成功（无 crash）
4. 检查各 Rank 输出形状一致
5. 如加载权重：检查输出文本合理性

每种配置独立验证，通过后再验下一个。

### 7.3 权重加载验证

加载权重验证时，在 YAML 中设置 `with_ckpt: True` + `model_path`，确认各 rank 权重加载无报错。

### 完成标志

- [ ] 配置校验脚本通过
- [ ] 至少一种配置的 Prefill + Decode 验证通过
- [ ] 使用 `enable_profiler: True` 运行一次，生成 profiler 数据供后续策略校准
- [ ] 验证结果已输出

---

## 常见错误

| 错误模式 | 根因 | 预防 |
|---------|------|------|
| 跳过通信组创建直接替换层 | 运行时找不到 group | 第二步必须先于第三步完成 |
| 所有模块用同一个 tp_group | 未区分 attn_tp / dense_tp / moe_tp | 配置→步骤速查表逐项检查 group 来源 |
| EP 模式下 Prefill/Decode 用同一套代码 | 两阶段的 routing 算子不同 | 参考 moe-parallel.md 中 Prefill/Decode 分支 |
| 模块间 TP 度不同但缺少数据重排 | 相邻模块 tensor shape 不匹配 | 速查表"相邻模块 TP 度不同"行提示了此步骤 |
| embed_tp_size < attn_tp_size | 框架约束：embed 输出需能被 attn 消费 | 第七步校验脚本检查 |
| 权重加载 shape 不匹配 | 改了 parallel_config 但未重新处理权重 | 第六步确认权重处理方式 |

---

## 仓库参考实现索引

| 实现模式 | 参考文件 | 搜索关键词 |
|---------|---------|-----------|
| TP 线性层替换 | `cann-recipes-infer/models/gpt_oss/models/modeling_*.py` | `QKVParallelLinear` |
| 通信组创建 | `cann-recipes-infer/models/kimi-k2-thinking/models/modeling_*.py` | `init_parallel_comm_group` |
| 通信组校验 | `cann-recipes-infer/models/kimi-k2-thinking/infer.py` | `check_parallel_settings` |
| MoE EP Prefill | `cann-recipes-infer/models/deepseek_r1/models/modeling_*.py` | `npu_moe_re_routing` |
| MoE EP Decode | `cann-recipes-infer/models/deepseek_r1/models/modeling_*.py` | `npu_moe_distribute_dispatch` |
| Embed/LMHead 并行 | `cann-recipes-infer/models/kimi-k2-thinking/models/modeling_*.py` | `VocabParallelEmbedding` |
| oproj_tp 独立配置 | `cann-recipes-infer/models/longcat-flash/models/modeling_*.py` | `oproj_tp` |
| 权重转换脚本 | `cann-recipes-infer/models/deepseek_r1/utils/` | `weight_convert` |
| YAML 多场景配置 | `cann-recipes-infer/models/deepseek_r1/config/` | decode/prefill 分离 |
