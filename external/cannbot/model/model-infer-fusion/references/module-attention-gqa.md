# Attention 层：GQA / MHA 标准路径

**参考模型**：`cann-recipes-infer/models/qwen3_moe/`（MoE）、`cann-recipes-infer/models/gpt_oss/`（Dense）

**核心特征**：标准多头 / 分组查询注意力，KV Cache 存完整的 K/V。**Prefill 和 Decode 走不同 FA 参数**。

## Prefill 链路

```python
# ─── Pre-Norm ───
hidden_states, residual = npu_add_rms_norm(residual, hidden_states, weight, eps)
#   首层无 residual 时退化为 npu_rms_norm

# ─── QKV 投影 ───
q, k, v = qkv_proj(hidden_states).split(...)

# ─── QK Head Norm（部分模型有，如 qwen3-moe）───
q = npu_rms_norm(q, ...)
k = npu_rms_norm(k, ...)

# ─── RoPE ───
q, k = npu_apply_rotary_pos_emb(q, k, cos, sin, layout='BSH')

# ─── KV Cache 写入 ───
scatter_update_(past_key, kv_len, k, dim=-2)
scatter_update_(past_val, kv_len, v, dim=-2)

# ─── Flash Attention（用当前 batch 的 k/v）───
output = npu_fused_infer_attention_score(q, k, v, sparse_mode=3, ...)

# ─── O 投影 ───
output = o_proj(output)
```

## Decode 链路

```python
# ─── Pre-Norm ───
hidden_states, residual = npu_add_rms_norm(residual, hidden_states, weight, eps)

# ─── QKV 投影 ───
q, k, v = qkv_proj(hidden_states).split(...)

# ─── QK Head Norm ───
q = npu_rms_norm(q, ...)
k = npu_rms_norm(k, ...)

# ─── RoPE ───
q, k = npu_apply_rotary_pos_emb(q, k, cos, sin, layout='BSH')

# ─── KV Cache 写入 ───
scatter_update_(past_key, kv_len, k, dim=-2)
scatter_update_(past_val, kv_len, v, dim=-2)

# ─── Flash Attention（用完整 cache）───
output = npu_fused_infer_attention_score(q, past_key, past_val, actual_seq_lengths_kv=..., ...)

# ─── O 投影 ───
output = o_proj(output)
```

## Prefill vs Decode 关键差异

| 环节 | Prefill | Decode |
|------|---------|--------|
| FA 的 KV 输入 | 当前 batch 的 k/v（非 cache） | 完整 past_key/past_value（cache） |
| FA 参数 | `sparse_mode=3`（causal mask，推荐） | 无 `sparse_mode`，传 `actual_seq_lengths_kv` |
