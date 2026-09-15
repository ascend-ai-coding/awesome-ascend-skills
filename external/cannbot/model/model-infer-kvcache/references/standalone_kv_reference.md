# 独立部署模式：KVCache 自管完整参考

Runner 自管 KV 数据结构、`block_table` / `slot_mapping`、`forward_metadata` 构造，不接入 `cann-recipes-infer/executor/core/kv_cache/`。modeling 内 `scatter_update_` 写入 + FA 算子调用与框架部署完全一致。

本文档按 KV 模式分三类骨架：连续缓存（Legacy）、手动 Paged 模式、MLA 压缩缓存（可叠加在前两者之上）。

---

## §1 模式一：连续缓存（Legacy，最简）

KV 以连续 tensor 存储，`scatter_update_` 写入，FA 直接读取整个缓存。无 paging 概念，不需要 `block_table` / `slot_mapping`。

### 1.1 Runner 内 KV 分配

```python
# runner_{model_name}.py 内 _init_kvcache(...)
def _init_kvcache(self, num_layers, batch_size, max_seq_len,
                  num_kv_heads_per_rank, head_dim, dtype):
    self.kv_caches = []  # list of (k, v)
    for _ in range(num_layers):
        k = torch.zeros(batch_size, max_seq_len,
                        num_kv_heads_per_rank * head_dim,
                        dtype=dtype, device=self.device)   # BSH layout
        v = torch.zeros_like(k)
        self.kv_caches.append((k, v))
```

### 1.2 modeling 写入 + FA 调用

```python
# attention layer
class XxxAttention(nn.Module):
    def forward(self, hidden_states, kv_len, attention_mask, past_kv, ...):
        # ... QKV projection、RoPE ...
        past_key, past_value = past_kv  # 来自 Runner 注入

        # 写入（migrator 阶段 BSH legacy 骨架）：scatter_update_ 按 kv_len 写位置（BSH 时 axis=1）
        torch_npu.scatter_update_(past_key, kv_len, key_states, 1)
        torch_npu.scatter_update_(past_value, kv_len, value_states, 1)

        # FA 调用（FA v1 示例）
        attn_output, _ = torch.ops.npu.npu_fused_infer_attention_score(
            query_states, past_key, past_value,
            num_heads=num_heads, num_key_value_heads=num_kv_heads,
            input_layout="BSH",
            scale=1.0 / math.sqrt(head_dim),
            actual_seq_lengths_kv=actual_seq_lengths_kv,
            atten_mask=attention_mask,
            sparse_mode=0 if not is_prefill else 3,
        )
```

> 改造边界：只动 attention 计算（→ FA）和 cache 存储（→ `scatter_update_`），上游（QKV projection / RoPE）保持不变；layout 不匹配时在接缝处 transpose / reshape 适配。
>
> **写入算子选择**：`scatter_update_` 是 BSH Legacy 骨架专用（migrator 阶段最简）；进入 Paged 模式（模式二）后必须切到 `npu_scatter_nd_update_`（接 `slot_mapping`）或 `npu_kv_rmsnorm_rope_cache`（融合写入）。

该骨架形态由 model-infer-migrator skill 在独立部署阶段产出，按本节代码块对齐。FA 调用示例见 `references/fa-code-examples.md`。

---

## §2 模式二：手动 Paged 模式 + FA

KV 按固定大小 Block 存储，FA 通过 `block_table` 索引分块缓存。**Paged 模式必须配合 FA 使用，无法走标准 softmax**。

### 2.1 Runner 内 KV + block_table 静态预分配

```python
def _init_kvcache(self, num_layers, batch_size, max_seq_len,
                  num_kv_heads_per_rank, head_dim, dtype, block_size=128):
    self.block_size = block_size
    num_blocks_per_seq = max_seq_len // block_size
    total_blocks = batch_size * num_blocks_per_seq

    # 物理 cache：[total_blocks, block_size, num_kv_heads, head_dim]
    self.kv_caches = []
    for _ in range(num_layers):
        k = torch.zeros(total_blocks, block_size,
                        num_kv_heads_per_rank, head_dim,
                        dtype=dtype, device=self.device)
        v = torch.zeros_like(k)
        self.kv_caches.append((k, v))

    # block_table 静态预分配（推理全程不变）
    # block_table[b, i] = 第 b 个 batch 的第 i 个逻辑 block 对应的物理 block ID
    self.block_table = torch.arange(0, total_blocks).reshape(
        batch_size, -1).to(torch.int32).npu()

    # 预计算 kv_len_offset（slot_mapping 用）
    self.kv_len_offset = torch.arange(
        0, batch_size * max_seq_len, max_seq_len, device=self.device,
    ).view(-1, 1)
```

### 2.2 Runner 内 slot_mapping 动态计算

`slot_mapping` 是缓存**写入**位置索引。独立部署 Runner 通常走 block_table 静态预分配（arange）模式，slot 公式简化为：

```
slot(batch_idx, seq_pos) = batch_idx × max_seq_len + seq_pos    （顺序分配模式下等于展平后线性索引）
```

> 框架部署模式下 BlockPool 动态分配 block，slot 计算依赖 `block_table` 索引（`block_id × block_size + offset`），不适用以下静态预分配公式。详见 SKILL.md §2.3。

**Prefill** —— 每个 batch 写入多个 token：

```python
def _build_slot_mapping_prefill(self, kv_len, max_seq_len):
    # kv_len: [batch_size]，每条请求的 prompt 长度
    all_tensors = []
    for i, seq_len in enumerate(kv_len):
        all_tensors.append(torch.arange(
            max_seq_len * i, seq_len.item() + max_seq_len * i,
            dtype=torch.int32, device=self.device,
        ))
    return torch.cat(all_tensors)

# 示例：kv_len=[512, 256], max_seq_len=2048
#   batch 0: [0, 1, ..., 511]
#   batch 1: [2048, 2049, ..., 2303]
#   拼接 → shape=[768] 一维 tensor
```

**Decode** —— 每个 batch 写入 1 个 token：

```python
def _build_slot_mapping_decode(self, kv_len):
    # kv_len: [batch_size]，每条请求当前 KV 实际长度
    return kv_len.view(-1, 1) + self.kv_len_offset    # [batch_size, 1]

# 示例：kv_len=[522, 266], kv_len_offset=[[0], [2048]]
#   slot_mapping = [[522], [2314]]
```

### 2.3 actual_seq_lengths 构造

FA 算子需要每个 batch 的实际 KV/Q 长度，构造方式取决于 `input_layout`：

| | TND layout（多 batch token 拼一维） | BSH layout（各 batch 独立） |
|--|------|------|
| **Prefill KV** | `cumsum(kv_len)` → [512, 768] | `kv_len` → [512, 256] |
| **Prefill Q** | 同 KV | 同 KV |
| **Decode KV** | `kv_len` → [522, 266] | `kv_len` → [522, 266] |
| **Decode Q** | `cumsum([1,1])` → [1, 2] | `[1, 1]` |

```python
# TND Prefill 示例
actual_seq_lengths_kv = torch.cumsum(kv_len, dim=0)
actual_seq_lengths_q = actual_seq_lengths_kv.clone()
```

### 2.4 modeling 写入 + FA 调用

```python
class XxxAttention(nn.Module):
    def forward(self, hidden_states, slot_mapping, block_table, ...):
        # 写入：用 npu_scatter_nd_update_（slot_mapping 索引模式），cache / states 都 view 为 [total, num_kv_heads, head_dim]
        torch_npu.npu_scatter_nd_update_(
            self.k_cache.view(-1, num_kv_heads, head_dim),
            slot_mapping.view(-1, 1),
            key_states.view(-1, num_kv_heads, head_dim),
        )
        # 或融合写入（MLA 模型推荐，一步完成 RMSNorm + RoPE + Cache 写入）：
        # torch_npu.npu_kv_rmsnorm_rope_cache(..., slot_mapping.view(-1), ...)

        # FA 调用（FA v2 示例，传 block_table）
        attn_output, _ = torch_npu.npu_fused_infer_attention_score_v2(
            query_states, self.k_cache, self.v_cache,
            block_table=block_table,
            block_size=self.block_size,
            num_query_heads=num_heads,
            num_key_value_heads=num_kv_heads,
            actual_seq_kvlen=actual_seq_lengths_kv,
            actual_seq_qlen=actual_seq_lengths_q,
            input_layout="TND_NTD",
            sparse_mode=0 if not is_prefill else 3,
            softmax_scale=1.0 / math.sqrt(head_dim),
        )
```

> 算子调用方式可参考 `references/fa-code-examples.md`。

---

## §3 模式三：MLA 压缩缓存（叠加在模式一或模式二之上）

MLA 将 KV 压缩为低维 latent，只缓存压缩后的 `cache_nope`（非位置）和 `cache_rope`（位置）。

### 3.1 Runner 内 KV 分配

```python
def _init_kvcache(self, num_layers, batch_size, max_seq_len,
                  kv_lora_rank, qk_rope_head_dim, dtype, block_size=128):
    # MLA 压缩维度（kv_lora_rank=512）远小于完整 KV（num_heads * head_dim=16384）
    num_blocks_per_seq = max_seq_len // block_size
    total_blocks = batch_size * num_blocks_per_seq

    self.cache_nope = []  # 非位置部分
    self.cache_rope = []  # 位置编码部分
    for _ in range(num_layers):
        c_nope = torch.zeros(total_blocks, block_size, 1, kv_lora_rank,
                             dtype=dtype, device=self.device)
        c_rope = torch.zeros(total_blocks, block_size, 1, qk_rope_head_dim,
                             dtype=dtype, device=self.device)
        self.cache_nope.append(c_nope)
        self.cache_rope.append(c_rope)
```

### 3.2 modeling FA absorb 调用

```python
# key 和 value 传同一个 cache_nope（absorb 技术将 V 投影吸收到 O 投影中）
attn_output, _ = torch_npu.npu_fused_infer_attention_score_v2(
    q_nope, k_nope_cache, k_nope_cache,        # key = value = cache_nope
    query_rope=q_pe, key_rope=k_rope_cache,    # RoPE 单独传入
    block_table=block_table,
    block_size=self.block_size,
    ...
)
```

**重要约束**：
- `query_rope` 和 `key_rope` 必须同时传或同时不传
- rope D 必须为 64
- MLA query D 仅支持 512 或 128
- MLA D=512 时仅支持 `sparse_mode` 为 0、3、4

参考实现：`cann-recipes-infer/models/deepseek_r1/models/modeling_deepseek.py::forward_absorb()`。

---

## §4 完整 Runner 改造与 forward_metadata 自管

独立部署 Runner 在每步推理前需自管构造 `forward_metadata`：

```python
def _build_forward_metadata(self, kv_len, is_prefill, max_seq_len):
    # 直接复用 from executor.utils.forward_metadata import ForwardMetaData
    # 或参考字段在 runner 内 inline 一个轻量 dataclass

    if is_prefill:
        slot_mapping = self._build_slot_mapping_prefill(kv_len, max_seq_len)
        actual_seq_lengths_kv = torch.cumsum(kv_len, dim=0).to(torch.int32)
        actual_seq_lengths_q = actual_seq_lengths_kv.clone()
        # Prefill 用 [2048, 2048] bool 因果 mask（参考 executor.utils.common_utils.get_init_attn_mask）
        attention_mask = self.share_mask_tril
    else:
        slot_mapping = self._build_slot_mapping_decode(kv_len)
        actual_seq_lengths_kv = kv_len.to(torch.int32)
        actual_seq_lengths_q = torch.arange(1, kv_len.shape[0] + 1).to(torch.int32)
        attention_mask = None    # full-attention Decode 不需要 mask

    return ForwardMetaData(
        is_prefill=is_prefill,
        kv_len=kv_len,
        slot_mapping=slot_mapping,         # 单 attn_type 时传 tensor；多 attn_type 时按 dict 包装
        block_table=self.block_table,      # 同上
        actual_seq_lengths_kv=actual_seq_lengths_kv,
        actual_seq_lengths_q=actual_seq_lengths_q,
        attention_mask=attention_mask,
    )
```

### kv_len 生命周期

`kv_len` 是驱动所有入参计算的核心变量：

```python
# Prefill：从 attention_mask 计算
position_ids = attention_mask.long().cumsum(-1) - 1
kv_len = torch.max(position_ids, axis=1)[0] + 1   # 如 [512, 256]

# 每次 Decode 后递增（在 Runner 层，model.forward() 之外）
kv_len = kv_len + 1   # [513, 257], [514, 258], ...

# kv_len 驱动三个计算：
# 1. slot_mapping → 决定新 token 写到哪
# 2. actual_seq_lengths_kv → 告诉 FA 读多少 KV
# 3. position_ids → RoPE 位置编码
```

> **kv_len 是 Runner 层变量，各层只读不写。** Runner 计算一次 `kv_len`（及由它派生的 `slot_mapping`），所有 Transformer 层用同一份值写入各自缓存（模式一用 `scatter_update_` 接 `kv_len` / 模式二/三用 `npu_scatter_nd_update_` 或 `npu_kv_rmsnorm_rope_cache` 接 `slot_mapping`）。不要在 attention 内部递增 `kv_len`——否则 Prefill 阶段各层写入位置逐层偏移，精度损坏。
