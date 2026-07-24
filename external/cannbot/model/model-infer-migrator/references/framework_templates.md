# 框架契约模板和速查

本文件提供两类内容：
1. **模型类契约速查**：构造签名、forward 签名、ForwardMetaData 字段、CommManager 通信组、Packed/TND 约定
2. **产物模板**：YAML 配置、infer.sh、modeling 骨架

权威接口契约定义见 [cann-recipes-infer/docs/design/executor_design.md §6](../../../../docs/design/executor_design.md)。

替换 `{model_name}`（小写下划线）和 `{ModelName}`（驼峰）后使用。

---

## 1. 模型类契约

> **入参布局**：input_ids 由框架按 packed 布局拼装为一维 `[TotalTokens]`，prefill 各 prompt 串接，decode 每请求 1 token。返回 packed 输出 logits `[TotalTokens, vocab]`。

### 1.1 ForCausalLM 类骨架

```python
import torch
import torch.nn as nn
from typing import Iterable, Optional, Tuple

from executor.core.config import InferenceConfig, CommManager
from executor.utils.forward_metadata import ForwardMetaData

from .configuration_{model_name} import {ModelName}Config


class {ModelName}ForCausalLM(nn.Module):
    def __init__(
        self,
        config: {ModelName}Config,
        infer_config: InferenceConfig,
        comm_manager: CommManager = None,
        prefix: str = "",
    ):
        super().__init__()
        self.config = config
        self.infer_config = infer_config
        self.comm_manager = comm_manager

        # 读取并行配置
        self.attn_tp_size = infer_config.parallel_config.attn_tp_size
        self.lmhead_tp_size = infer_config.parallel_config.lmhead_tp_size
        # ... MoE 模型还有 moe_tp_size / moe_ep_size

        # 构造子模块
        self.model = {ModelName}Model(config, infer_config, comm_manager, prefix="model")
        self.lm_head = ...  # ColumnParallelLinear

    def forward(
        self,
        input_ids: torch.LongTensor,        # packed 一维 [TotalTokens]
        position_ids: Optional[torch.LongTensor] = None,
        forward_metadata: ForwardMetaData = None,
        **kwargs,                           # 框架透传 slot_mapping / block_table 等（Paged 模式才用到，Legacy 阶段忽略）
    ) -> torch.Tensor:
        # Prefill：每请求取末位 token（cu_seq_q - 1 索引）→ lm_head → [batch_size, 1, vocab_size]
        # Decode：input_ids 本就是每请求 1 token → lm_head → [batch_size, 1, vocab_size]
        ...
        return logits

    def load_weights(self, weights: Iterable[Tuple[str, torch.Tensor]]) -> set[str]:
        """从 HF checkpoint stream 加载权重，返回已加载参数名集合。"""
        params_dict = dict(self.named_parameters())
        loaded = set()
        for name, tensor in weights:
            # 按名称匹配 param，调用 weight_loader 或 default_weight_loader
            ...
        return loaded

    def process_weights_after_loading(self):
        """加载权重后触发量化/MoE 后处理（NZ 转换、scale 处理等）。框架自动调用。"""
        for _, module in self.named_modules():
            qm = getattr(module, "quant_method", None)
            if qm is not None and hasattr(qm, "process_weights_after_loading"):
                qm.process_weights_after_loading(module)

    def check_model_settings(self):
        """模型配置一致性校验（如 attn_tp_size 整除约束、num_heads 切分可行性等）。
        框架在加载完模型后自动调用；校验失败应 raise RuntimeError。可选实现，缺方法时框架跳过。"""
        pass
```

> Paged 模式接入（`get_cache_info()` 实现 + `cache_entries` 声明）由 model-infer-kvcache skill 改造时添加，migrator 阶段不写。

### 1.2 Attention 层（KV 接入）

按 SKILL.md 步骤 4 选定路径，二选一。

#### 路径 A — Legacy 模式（默认推荐，单卡 + 等长 batch baseline）

```python
class {ModelName}Attention(nn.Module):
    def __init__(self, config, infer_config, comm_manager, layer_idx, prefix=""):
        super().__init__()
        # ... QKV / O 投影、RoPE 等 ...

        # Legacy 模式：声明 cache_unit，框架按 (batch, seq, *cache_unit) 自动预分配
        self.k_cache = torch.Tensor([])
        self.v_cache = torch.Tensor([])
        self.cache_unit = (self.num_kv_heads_per_rank * self.head_dim,)

    def forward(self, hidden_states, position_embeddings, forward_metadata: ForwardMetaData, **kwargs):
        is_prefill = forward_metadata.is_prefill
        kv_len = forward_metadata.kv_len
        # decode 分支必须把 atten_mask 设为 None（详见 common_issues.md）
        atten_mask = forward_metadata.attention_mask if is_prefill else None

        # 计算 Q / K / V，写 KV 到 self.k_cache / self.v_cache
        # 用 torch_npu.scatter_update_(self.k_cache, kv_len, key_states, -2) 写入
        # 然后 SDPA / 标准 PyTorch attention 计算
        ...
```

> 顶层 forward 需对 packed 1D `input_ids` 做 BSND reshape 回 `[B, S]`（等长 batch 适用，参考 `cann-recipes-infer/models/gemma4_26b_a4b/models/modeling_gemma4.py` 的 BSND shim）。`ForCausalLM` 不实现 `get_cache_info()`。

#### 路径 B — 最简 Paged 模式（多卡 / 变长 batch / 用户指定 Paged 起步）

```python
class {ModelName}Attention(nn.Module):
    def __init__(self, config, infer_config, comm_manager, layer_idx, prefix=""):
        super().__init__()
        # ... QKV / O 投影、RoPE 等 ...

        self.attn_type = "FullAttention"
        self.k_cache = torch.Tensor([])   # 占位，框架通过 tensor_setter 注入
        self.v_cache = torch.Tensor([])
        cache_dtype = config.torch_dtype or torch.bfloat16
        self.cache_entries = [
            CacheEntry(
                cache_name="k_cache", attn_type=self.attn_type,
                dim=self.head_dim, num_head=self.num_kv_heads_per_rank,
                dtype=cache_dtype, needs_block=True,
                tensor_setter=lambda t, layer=self: setattr(layer, "k_cache", t),
            ),
            CacheEntry(
                cache_name="v_cache", attn_type=self.attn_type,
                dim=self.head_dim, num_head=self.num_kv_heads_per_rank,
                dtype=cache_dtype, needs_block=True,
                tensor_setter=lambda t, layer=self: setattr(layer, "v_cache", t),
            ),
        ]

    def forward(self, hidden_states, position_embeddings, forward_metadata: ForwardMetaData, **kwargs):
        slot_mapping = forward_metadata.slot_mapping[self.attn_type]
        block_table = forward_metadata.block_table[self.attn_type]
        # ... QKV 投影 / RoPE → key_states / value_states (packed [TotalTokens, ...]) ...

        # 写 cache：按 slot_mapping 写入
        torch_npu.npu_scatter_nd_update_(
            self.k_cache.view(-1, self.num_kv_heads_per_rank, self.head_dim),
            slot_mapping.view(-1, 1),
            key_states.view(-1, self.num_kv_heads_per_rank, self.head_dim),
        )
        torch_npu.npu_scatter_nd_update_(
            self.v_cache.view(-1, self.num_kv_heads_per_rank, self.head_dim),
            slot_mapping.view(-1, 1),
            value_states.view(-1, self.num_kv_heads_per_rank, self.head_dim),
        )

        # Attention：HF 原版 SDPA / 标准 PyTorch，不上 FA
        # - Prefill：直接用刚算的 key_states / value_states + causal mask
        # - Decode：用 block_table 从 cache gather 累计 KV，再走 SDPA
        # 变长 batch 下按 forward_metadata.actual_seq_lengths_cu_q 切 batch
        ...
```

> `ForCausalLM` 必须实现 `get_cache_info()`（5 行模板，遍历 `self.model.layers` 收集 `cache_entries`）；`__init__` 保存 `self.block_size = infer_config.scheduler_config.block_size`。完整字段表与 `get_cache_info()` 实现见 `kvcache skill` 的 `references/framework_kv_reference.md §1.1 / §3.1`。FA 算子接入 / MLA absorb / 复杂 attn_type 由 kvcache skill 接手。

### 1.3 ForwardMetaData 字段速查

来源：`cann-recipes-infer/executor/utils/forward_metadata.py`

| 字段 | 类型 | 说明 |
|------|------|------|
| `is_prefill` | `bool` | 当前是 prefill 还是 decode |
| `attention_mask` | `Optional[Tensor]` | prefill 用 causal mask；**decode 必须置 None** |
| `kv_len` | `Optional[Tensor]` | 每条请求当前 KV 实际长度（packed） |
| `actual_seq_lengths_kv` | `Optional[Tensor]` | KV 边的 cumulative seq lengths |
| `actual_seq_lengths_q` | `Optional[Tensor]` | Q 边的 cumulative seq lengths |
| `actual_seq_lengths_cu_kv` / `_cu_q` | `Optional[Tensor]` | cumulative 累加版本 |
| `actual_seq_lengths_cu_list_kv` / `_cu_list_q` | `Optional[list]` | list 版本 |
| `prompt_tokens` | `int` | 原始 prompt 长度（位置编码偏移用） |
| `block_table` | `Dict[str, Tensor]` | Paged 模式下每个 attn_type 的 block 索引（也作为 forward 参数透传） |
| `slot_mapping` | `Dict[str, Tensor]` | Paged 模式下每 token 的 slot 索引（也作为 forward 参数透传） |

> CommManager 通信组取用（`comm_manager.get_group(...)`）由 model-infer-parallel-impl skill 详细说明，migrator 阶段 `world_size=1` 不展开。

---

## 2. YAML 模板（offline）

`InferenceConfig` 共 5 个子配置：`DataConfig` / `ModelConfig` / `ParallelConfig` / `SchedulerConfig` / `DisaggConfig`。`DisaggConfig.disaggregation_mode` 默认 `NONE`（offline），无需在 YAML 配置；migrator 默认产物只覆盖前 4 段。online（PD 分离）需要的 disagg 字段见 [cann-recipes-infer/docs/design/online_inference_design.md](../../../../docs/design/online_inference_design.md)。


### 2.1 单卡 eager（最小启动配置）

```yaml
model_name: "{model-key}"          # 必须与 support_models.py 中 key 一致
world_size: 1

model_config:
  model_name: "{model-key}"
  model_path: "{absolute_or_relative_weights_path}"
  exe_mode: "eager"                # ["eager", "ge_graph", "npugraph_ex"]
  enable_profiler: False
  with_ckpt: True

data_config:
  dataset: "default"               # ["default", "LongBench"]
  input_truncated_len: 4096

parallel_config:
  world_size: 1
  attn_tp_size: 1                  # attn_dp_size = world_size // attn_tp_size（自动推导）
  moe_tp_size: 1                   # moe_ep_size = world_size // moe_tp_size（自动推导，仅 MoE）
  embed_tp_size: 1
  lmhead_tp_size: 1

scheduler_config:
  batch_size: 1
  max_new_tokens: 32
  max_prefill_tokens: 4096          # 单次 prefill 最大 packed token 数，决定调度切批策略
  # block_size 由 kvcache skill 改造为 Paged 模式时配置；migrator Legacy 模式不需要
```

> 多卡 YAML（`world_size > 1` + 各 `*_tp_size` 配置）由 model-infer-parallel-impl skill 提供。
>
> 布尔值统一用 `True` / `False`，不用 `true` / `false`（与仓内主流风格一致）。
>
> 命名维度：参考仓内已注册模型，按"模型_rank_N_拓扑_后端_场景"组合，禁止用非结构性差异的临时描述符（如 `_4k1k`、`_b8`）；新建 yaml 只针对结构性差异（拓扑 / 量化 / prefill/decode / 特性开关组合），运行时参数或 `exe_mode` 切换改字段即可。

---

## 3. infer.sh 模板

```bash
#!/bin/bash
SCRIPT_PATH=$(cd "$(dirname "${BASH_SOURCE[0]}")" &>/dev/null && pwd)
SET_ENV_ABS_PATH="${SCRIPT_PATH}/../../executor/scripts/set_env.sh"
FUNCTION_ABS_PATH="${SCRIPT_PATH}/../../executor/scripts/function.sh"
SET_ENV_ABS_PATH=$(realpath "${SET_ENV_ABS_PATH}")
FUNCTION_ABS_PATH=$(realpath "${FUNCTION_ABS_PATH}")

source ${SET_ENV_ABS_PATH}
source ${FUNCTION_ABS_PATH}

export MODEL_DIR=$(basename "$SCRIPT_PATH")
export YAML_PARENT_PATH="${SCRIPT_PATH}/config"

mode="$1"
pd_role="$2"

if [ "$mode" = "online" ]; then
    export PD_ROLE="$pd_role"
    export P_YAML="${YAML_PARENT_PATH}/{model_name}_pd.yaml"
    export D_YAML="${YAML_PARENT_PATH}/{model_name}_pd.yaml"
    echo "====================> launch online inference (${PD_ROLE:-auto})"
else
    export YAML="${YAML_PARENT_PATH}/{model_name}.yaml"
    echo "====================> launch offline inference"
fi

launch "$mode"
```

`../../executor/scripts/` 要求模型目录与仓库根目录之间有两层。`cann-recipes-infer/models/{model_name}/` 满足此条件。`launch` 函数会自动选用模型目录下的 `infer.py`，缺失时落回 `cann-recipes-infer/executor/offline/infer.py`。

---

## 4. README 模板

README 模板见共享文件 `references/readme_template.md`（框架部署与独立部署共用）。框架部署模式：保留 `<!-- 仅框架部署 -->` 标注的段（即"注册"段）；忽略 `<!-- 仅独立部署 -->` 标注的追加项。

---

## 5. 参考实现

仓内已注册模型已经历过 kvcache / parallel-impl 等优化阶段，**不是 migrator 阶段产物**：

- 标准 LLM (GQA) → `cann-recipes-infer/models/qwen/models/modeling_qwen.py`
- MoE → `cann-recipes-infer/models/qwen3_moe/models/modeling_qwen3_moe.py`、`cann-recipes-infer/models/gpt_oss/models/modeling_gpt_oss.py`
- MoE + MLA + MTP → `cann-recipes-infer/models/deepseek_r1/models/modeling_deepseek.py`

migrator 阶段参考其构造签名 / forward 签名 / `load_weights` / `cache_unit` 声明等骨架部分；`get_cache_info()` / `cache_entries` / FA 算子 / `ParallelLinear` / MLA absorb / MoE EP 算子链等是后续 skill 产物，不要带入 migrator 输出。
