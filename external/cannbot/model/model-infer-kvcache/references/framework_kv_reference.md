# 框架部署模式：KVCache 完整接入参考

按 `cann-recipes-infer/docs/design/kv_cache_design.md §5` 接入流程展开，覆盖 `CacheEntry` 字段、`get_cache_info` 实现、attention forward 接入、自定义 manager 等。

---

## §1 CacheEntry / ModelCacheInfo 字段详解

定义位置：`cann-recipes-infer/executor/core/kv_cache/cache_info.py`。

### 1.1 CacheEntry

| 参数 | 说明 |
|-----|------|
| `cache_name` | cache 名称，如 `"k_cache"` / `"v_cache"` / `"nope_cache"` / `"rope_cache"` |
| `attn_type` | attention 类型（决定走哪个 `SingleTypeKVCacheManager`），从「步骤 1」确定 |
| `dim` | 单个 attention head 的特征维度（注意 MLA 的压缩维度 = `kv_lora_rank`，与标准 head_dim 不同） |
| `num_head` | cache 包含的 head 数量。**TP 切分后必须用 `num_key_value_heads_per_rank = max(num_kv_heads // attn_tp_size, 1)`**；MLA 类 cache 通常 `num_head=1` |
| `dtype` | cache tensor 数据类型（`torch.bfloat16` / `torch.int8` 等） |
| `needs_block` | 是否参与 block 分配，通常 `True` |
| `tensor_setter` | 回调函数，用于将分配的 cache tensor 注册到 attention 类。固定写法 `lambda tensor, layer=self: setattr(layer, "<cache_name>", tensor)` |
| `sliding_window` | 仅 `attn_type="SlidingWindow"` 需要，设置窗口大小（来自 config） |

### 1.2 ModelCacheInfo

| 参数 | 说明 |
|-----|------|
| `num_layers` | 模型层数（含 MTP 层时按实际遍历总数填） |
| `block_size` | 每个 block 的 token 容量，从 `infer_config.scheduler_config.block_size` 取 |
| `layer_infos` | `LayerCacheInfo` 列表 |
| `is_mla_backend` | **MLA 模型必须设 True**（默认 False）。MLA 的 latent KV 在 TP 各 rank 间复制，PD 传输等场景需要据此挑单 rank 传输；非 MLA 模型保持默认 |

> attn_type 决定 cache 条目映射到哪一类 `SingleTypeKVCacheManager`。分组维度是 `attn_type`，因此本框架支持单层同时包含多个 cache 条目，也支持单层同时包含**多种 cache 类型**。

---

## §2 cache_entries 完整骨架

### 2.1 标准 GQA / MHA（FullAttention）

参考来源：`cann-recipes-infer/models/qwen/models/modeling_qwen.py`。

```python
class XxxAttention(nn.Module):
    def __init__(self, config, infer_config, comm_manager, layer_idx, prefix=""):
        # ...（QKV 投影、RoPE 等）
        self.attn_type = "FullAttention"
        self.k_cache = torch.Tensor([])      # 占位 tensor，由框架在初始化时通过 tensor_setter 注入实际 storage
        self.v_cache = torch.Tensor([])
        cache_dtype = torch.bfloat16 if config.torch_dtype is None else config.torch_dtype
        self.cache_entries = [
            CacheEntry(
                cache_name="k_cache",
                attn_type=self.attn_type,
                dim=self.head_dim,
                num_head=self.num_key_value_heads_per_rank,
                dtype=cache_dtype,
                needs_block=True,
                tensor_setter=lambda tensor, layer=self: setattr(layer, "k_cache", tensor),
            ),
            CacheEntry(
                cache_name="v_cache",
                attn_type=self.attn_type,
                dim=self.head_dim,
                num_head=self.num_key_value_heads_per_rank,
                dtype=cache_dtype,
                needs_block=True,
                tensor_setter=lambda tensor, layer=self: setattr(layer, "v_cache", tensor),
            ),
        ]
```

### 2.2 SlidingWindow（含 sliding_window 字段）

参考来源：`cann-recipes-infer/models/gpt_oss/models/modeling_gpt_oss.py`。仓库中 gpt-oss 部分层是 SlidingWindow（`config.sliding_window` 存在），其他层是 FullAttention（`sliding_window=None`）。两类层在 `__init__` 内通过条件设置 `attn_type` 和 `sliding_window` 字段。

```python
self.attn_type = "SlidingWindow" if self.sliding_window else "FullAttention"
self.cache_entries = [
    CacheEntry(
        cache_name="k_cache",
        attn_type=self.attn_type,
        dim=self.head_dim,
        num_head=self.num_key_value_heads_per_rank,
        dtype=cache_dtype,
        needs_block=True,
        tensor_setter=lambda tensor, layer=self: setattr(layer, "k_cache", tensor),
        sliding_window=self.sliding_window if self.sliding_window else None,
    ),
    CacheEntry(
        cache_name="v_cache",
        attn_type=self.attn_type,
        dim=self.head_dim,
        num_head=self.num_key_value_heads_per_rank,
        dtype=cache_dtype,
        needs_block=True,
        tensor_setter=lambda tensor, layer=self: setattr(layer, "v_cache", tensor),
        sliding_window=self.sliding_window if self.sliding_window else None,
    ),
]
```

> `attn_type="SlidingWindow"` 的 cache 由 `SlidingWindowManager` 管理，block 数量通过 `calculate_fixed_block_memory_bytes()` 按窗口长度固定预留（不随序列总长线性增长）。

### 2.3 MLA（DeepSeek 系列）

参考来源：`cann-recipes-infer/models/deepseek_r1/models/modeling_deepseek.py`。MLA 将 K/V 压缩为低维 latent，缓存 `nope_cache`（kv_lora_rank 维度）+ `rope_cache`（qk_rope_head_dim 维度）；FA absorb 调用时 key 和 value 传同一个 nope_cache。

```python
self.attn_type = "FullAttention"   # MLA 也归入 FullAttention 管理（按序列长度线性增长）
self.block_size = self.infer_config.scheduler_config.block_size
self.nope_cache = torch.Tensor([])
self.rope_cache = torch.Tensor([])
dtype_nope = torch.int8 if self.kv_cache_quant_mode == "int8" else self.config.torch_dtype
dtype_rope = self.config.torch_dtype
self.cache_entries = [
    CacheEntry(
        cache_name="nope_cache",
        attn_type=self.attn_type,
        dim=self.config.kv_lora_rank,        # 压缩维度
        num_head=1,                          # MLA 压缩到 1 个 head
        dtype=dtype_nope,
        needs_block=True,
        tensor_setter=lambda tensor, layer=self: setattr(layer, "nope_cache", tensor),
    ),
    CacheEntry(
        cache_name="rope_cache",
        attn_type=self.attn_type,
        dim=self.config.qk_rope_head_dim,    # RoPE 维度（必须 64）
        num_head=1,
        dtype=dtype_rope,
        needs_block=True,
        tensor_setter=lambda tensor, layer=self: setattr(layer, "rope_cache", tensor),
    ),
]
```

> 同层多 cache 条目（nope + rope）`attn_type` 相同，会被归入同一个 manager 统一管理；分配 block 时按总 token 字节合并算。

---

## §3 get_cache_info 完整实现

模型类（如 `XxxForCausalLM`）实现 `get_cache_info()` 方法返回 `ModelCacheInfo`，被 `ModelWorker._get_cache_info()` 调用。

### 3.1 标准模型

参考来源：`cann-recipes-infer/models/qwen/models/modeling_qwen.py`、`cann-recipes-infer/models/qwen3_moe/models/modeling_qwen3_moe.py`、`cann-recipes-infer/models/gpt_oss/models/modeling_gpt_oss.py`。

```python
def get_cache_info(self) -> ModelCacheInfo:
    layer_infos = []
    for layer_idx, layer in enumerate(self.model.layers):
        layer_infos.append(
            LayerCacheInfo(
                layer_idx=layer_idx,
                caches=list(layer.self_attn.cache_entries),
            )
        )
    return ModelCacheInfo(
        num_layers=len(layer_infos),
        block_size=self.block_size,
        layer_infos=layer_infos,
    )
```

### 3.2 含 MTP 的 MLA 模型（DeepSeek-R1）

参考来源：`cann-recipes-infer/models/deepseek_r1/models/modeling_deepseek.py`。MTP 模型 `self.model.layers` 是 `dict`（按 layer_idx 索引），需用 `.values()` 遍历；MLA 模型必须设 `is_mla_backend=True`。

```python
def get_cache_info(self) -> ModelCacheInfo:
    layers = self.model.layers if not self.is_mtp else self.model.layers.values()
    layer_infos = []
    for layer_idx, layer in enumerate(layers):
        layer_infos.append(
            LayerCacheInfo(
                layer_idx=layer_idx,
                caches=list(layer.self_attn.cache_entries),
            )
        )
    return ModelCacheInfo(
        num_layers=len(layer_infos),
        block_size=self.block_size,
        layer_infos=layer_infos,
        is_mla_backend=True,    # MLA 模型必须显式设 True
    )
```

### 3.3 含 dual / multi sublayer 的模型

每个 logical layer 含 N 个 attention sublayer（如 LongCat 系列）。遍历展开到 sublayer 粒度，`layer_infos` 数量与物理 sublayer 总数一致。

```python
def get_cache_info(self) -> ModelCacheInfo:
    layer_infos = []
    global_idx = 0
    for layer in self.model.layers:
        for sub in layer.self_attn:           # ModuleList of N sublayers
            layer_infos.append(
                LayerCacheInfo(
                    layer_idx=global_idx,
                    caches=list(sub.cache_entries),
                )
            )
            global_idx += 1
    return ModelCacheInfo(
        num_layers=len(layer_infos),
        block_size=self.block_size,
        layer_infos=layer_infos,
        is_mla_backend=True,    # 若为 MLA 架构
    )
```

### 3.4 必备前置：模型类持有 block_size

```python
class XxxForCausalLM(nn.Module):
    def __init__(self, config, infer_config, comm_manager, prefix=""):
        # ...
        self.block_size = infer_config.scheduler_config.block_size
```

---

## §4 attention forward 接入 block_table / slot_mapping

### 4.1 forward 顶层签名（透传）

模型顶层 forward 接收 `forward_metadata`，框架在 `ExecutionEngine._build_model_inputs` 内通过 `prepare_block_tables()` / `prepare_slot_mapping()` 构造好 dict 形式的 `block_table[attn_type]` / `slot_mapping[attn_type]`，模型类透传给各层。

> **packed sequence 协议（必须遵守）**：
> - `input_ids: [TotalTokens]` 是 packed 一维（变长 batch 串接，无 padding）
> - `position_ids: [TotalTokens]` 也是 packed 一维（modeling 内 `if position_ids.dim() != 1: raise RuntimeError("expects packed 1D position_ids.")`，参考 `cann-recipes-infer/models/qwen3_moe/models/modeling_qwen3_moe.py:104`）
> - `hidden_states` 全程保持 `[TotalTokens, hidden_size]` 二维形态，**不要 reshape 成 `[B, S, H]`**——变长 batch 不能简单 reshape，batch 边界由 FA 调用时的 `actual_seq_qlen` / `actual_seq_kvlen` 表达
> - 阶段分支判断**统一用 `forward_metadata.is_prefill`**

```python
def forward(self, input_ids, position_ids, forward_metadata, **kwargs):
    is_prefill = forward_metadata.is_prefill
    block_table = forward_metadata.block_table   # Dict[str, Tensor]
    slot_mapping = forward_metadata.slot_mapping
    hidden_states = self.embed_tokens(input_ids)
    for layer in self.model.layers:
        hidden_states = layer(
            hidden_states, position_ids, forward_metadata,
            slot_mapping=slot_mapping, block_table=block_table,
        )
    return self.lm_head(hidden_states)
```

### 4.2 attention 层按 attn_type 取出对应 tensor

参考来源：`cann-recipes-infer/models/gpt_oss/models/modeling_gpt_oss.py`。

**关键事实**：
- 写入算子用 `torch_npu.npu_scatter_nd_update_`（不是 `scatter_update_`）；操作 cache 时需先 reshape 为 `[total_slots, num_kv_heads, head_dim]`
- ForwardMetaData 字段：FA 的 `actual_seq_qlen` 取 `forward_metadata.actual_seq_lengths_cu_q`（cumulative 形态），`actual_seq_kvlen` 取 `forward_metadata.actual_seq_lengths_kv`（直接形态）
- Prefill 与 Decode 的 FA 调用不同：Prefill 不传 `block_table`，直接对 `key_states` / `value_states` 算注意力（再写 cache）；Decode 必须先写 cache，再以 `block_table=block_table[self.attn_type]` 读取

```python
class XxxAttention(nn.Module):
    def forward(self, hidden_states, position_ids, forward_metadata,
                slot_mapping=None, block_table=None, **kwargs):
        # ... QKV / RoPE ...

        actual_seq_kvlen = forward_metadata.actual_seq_lengths_kv
        actual_seq_qlen = forward_metadata.actual_seq_lengths_cu_q
        attention_mask = forward_metadata.attention_mask

        if forward_metadata.is_prefill:
            # Prefill：先 FA 再写 cache（直接读 key_states / value_states，不传 block_table）
            attn_output, _ = torch_npu.npu_fused_infer_attention_score_v2(
                query_states, key_states, value_states,
                num_query_heads=self.num_attention_heads_per_rank,
                num_key_value_heads=self.num_key_value_heads_per_rank,
                input_layout="TND",
                softmax_scale=self.scaling,
                sparse_mode=4 if self.sliding_window else 3,
                pre_tokens=self.sliding_window if self.sliding_window else torch.iinfo(torch.int32).max,
                next_tokens=0,
                actual_seq_qlen=actual_seq_qlen,
                actual_seq_kvlen=actual_seq_qlen,    # Prefill: KV 与 Q 同
                atten_mask=attention_mask,
            )
            self.update_cache(slot_mapping, key_states, value_states)
        else:
            # Decode：先写 cache 再读 cache（传 block_table[self.attn_type]）
            self.update_cache(slot_mapping, key_states, value_states)
            attn_output, _ = torch_npu.npu_fused_infer_attention_score_v2(
                query_states,
                self.k_cache.view(*self.k_cache.shape[:2], -1),   # [bn, bs, num_head*dim]
                self.v_cache.view(*self.v_cache.shape[:2], -1),
                num_query_heads=self.num_attention_heads_per_rank,
                num_key_value_heads=self.num_key_value_heads_per_rank,
                input_layout="TND",
                softmax_scale=self.scaling,
                sparse_mode=4 if self.sliding_window else 3,
                pre_tokens=self.sliding_window if self.sliding_window else torch.iinfo(torch.int32).max,
                next_tokens=0,
                actual_seq_qlen=actual_seq_qlen,
                actual_seq_kvlen=actual_seq_kvlen,
                atten_mask=attention_mask,
                block_table=block_table[self.attn_type],
                block_size=self.block_size,
            )

    def update_cache(self, slot_mapping, key_states, value_states):
        # 写入算子：npu_scatter_nd_update_，cache / states 都 view 为 [total, num_kv_heads, head_dim]
        torch_npu.npu_scatter_nd_update_(
            self.k_cache.view(-1, self.num_key_value_heads_per_rank, self.head_dim),
            slot_mapping[self.attn_type].view(-1, 1),
            key_states.view(-1, self.num_key_value_heads_per_rank, self.head_dim),
        )
        torch_npu.npu_scatter_nd_update_(
            self.v_cache.view(-1, self.num_key_value_heads_per_rank, self.head_dim),
            slot_mapping[self.attn_type].view(-1, 1),
            value_states.view(-1, self.num_key_value_heads_per_rank, self.head_dim),
        )
```

### 4.3 MLA 模型的写入：融合算子 npu_kv_rmsnorm_rope_cache

参考来源：`cann-recipes-infer/models/longcat-flash/models/modeling_longcat_flash.py`。MLA 的 KV 写入可以用融合算子一步完成 RMSNorm + RoPE + Cache 写入：

```python
_, _, k_rope, k_nope = torch_npu.npu_kv_rmsnorm_rope_cache(
    latent_cache, self.kv_a_layernorm.weight,
    cos, sin,
    slot_mapping[self.attn_type].view(-1),    # 写入位置
    self.rope_cache, self.nope_cache,         # 输出缓存
    epsilon=self.kv_a_layernorm.variance_epsilon,    # 用 RMSNorm 模块的 eps，不要硬编码
    cache_mode="PA_NZ",
    is_output_kv=True,
)
```

---

## §5 自定义 SingleTypeKVCacheManager（可选）

仅在现有 `ATTN_TYPE_MANAGER_MAP`（含 `FullAttention` / `SlidingWindow`）无法满足新 cache 类型需求时执行。

### 5.1 何时需要

- 新模型引入了独特的 cache 行为（如固定块数 ≠ SlidingWindow 的窗口规则）
- 现有 manager 的 `allocate_new_blocks` / `get_num_skipped_tokens` 逻辑不能复用

### 5.2 子类实现要点

定义位置：`cann-recipes-infer/executor/core/kv_cache/single_type_kv_cache_manager.py`。

| 函数 | 说明 |
|------|------|
| `get_num_blocks_to_allocate(request_id, num_tokens)` | 计算本次需要新增多少 block |
| `allocate_new_blocks(request_id, block_num_to_allocate, num_tokens)` | 执行真实 block 分配，更新 `req_to_blocks` |
| `get_num_skipped_tokens(num_computed_tokens)` | 计算可跳过分配的 token 数 |
| `remove_skipped_blocks(request_id, total_computed_tokens)` | （可选）特殊回收机制 |
| `validate_and_build_kwargs(group_entries)` | （静态方法）从 `CacheEntry` 提取和校验类型特定参数 |

参考实现：`single_type_kv_cache_manager.py::FullAttentionManager` / `::SlidingWindowManager`。

### 5.3 注册到 ATTN_TYPE_MANAGER_MAP

```python
ATTN_TYPE_MANAGER_MAP: Dict[str, Type[SingleTypeKVCacheManager]] = {
    "FullAttention": FullAttentionManager,
    "SlidingWindow": SlidingWindowManager,
    "NewAttentionType": NewAttentionManager,   # 新增
}
```

### 5.4 扩展 calculate_block_num（可选）

仅当新 attn_type 是「固定 block 数」类型（block 不随序列长度增长，如 SlidingWindow）时需要：

1. 加入 `FIXED_BLOCK_ATTN_TYPES` 列表
2. 实现 `calculate_fixed_block_memory_bytes()` 分支

详见 `cann-recipes-infer/docs/design/kv_cache_design.md §5.4`。

---

## §6 多 attn_type 混合模型

### 6.1 单层混合 vs 跨层混合

| 形态 | 示例 | 处理 |
|---|---|---|
| 单层多 cache 但同 attn_type（MLA：nope_cache + rope_cache） | DeepSeek-R1 | 同 manager 统一管理，按字节合并算 block |
| 跨层不同 attn_type（gpt_oss 部分层 SW + 部分层 FA） | gpt-oss | 各层在 `__init__` 内根据条件设置 `attn_type`，框架按 attn_type 分组各自分配 |
| 单层多 attn_type 混合 | （较少见） | 同 cache_entries 列表内不同 entry 设不同 `attn_type` 即可 |

### 6.2 forward 内按 attn_type 取 block_table

```python
# 错误：用同一个 block_table 处理所有层
block_table = forward_metadata.block_table   # 这是 Dict[str, Tensor]，不能直接传给 FA

# 正确：在每层 attention 内按 self.attn_type 取出对应 tensor
attn_output, _ = FA_OP(
    ..., block_table=block_table[self.attn_type], ...,
)
```

参考实现：`cann-recipes-infer/models/gpt_oss/models/modeling_gpt_oss.py`（搜 `block_table[self.attn_type]`）。

---

## §7 验证三链路

按 `cann-recipes-infer/docs/design/kv_cache_design.md §5.6`：

### 7.1 初始化链路验证

- 模型类 `__init__` 完成后，`get_cache_info()` 能返回正确的 `ModelCacheInfo`（`num_layers` / `block_size` / `layer_infos` 数量与配置一致）
- ModelWorker 调 `_get_cache_info()` → `KVCacheManager.allocate_cache_tensors()` 后，每层 attention 的 `k_cache` / `v_cache`（或 `nope_cache` / `rope_cache`）已被 `tensor_setter` 注入实际 storage（不再是 `torch.Tensor([])` 占位）

### 7.2 运行期链路验证

- forward 内 `forward_metadata.block_table[self.attn_type]` 与 `slot_mapping[self.attn_type]` 不为 None
- FA 调用前后 dtype / shape 与 `cache_entries` 声明一致
- Prefill 后第一个 Decode step 的 `actual_seq_lengths_kv` 等于 `prompt_length + 1`

### 7.3 释放链路验证

- 请求完成后 `KVCacheManager` 释放 block，下一请求复用 block 不出现脏数据
- 长跑无显存泄漏（连跑 N 请求显存占用稳定）
