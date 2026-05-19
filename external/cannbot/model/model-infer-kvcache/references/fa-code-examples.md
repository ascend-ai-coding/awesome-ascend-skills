# FA 调用完整代码示例

基于仓库中已有模型的实际调用，按模式分类。

## 模式一：连续缓存

### GPT-OSS（TND layout, FA v2, sliding window）

```python
# 参考: cann-recipes-infer/models/gpt_oss/models/modeling_gpt_oss.py
attn_output, _ = torch_npu.npu_fused_infer_attention_score_v2(
    query_states,
    past_key, past_value,
    num_query_heads=self.num_attention_heads_per_rank,
    num_key_value_heads=self.num_key_value_heads_per_rank,
    input_layout="TND",
    softmax_scale=self.scaling,
    sparse_mode=4 if self.sliding_window else 3,
    pre_tokens=self.sliding_window if self.sliding_window else torch.iinfo(torch.int32).max,
    next_tokens=0,
    actual_seq_qlen=actual_seq_qlen,
    actual_seq_kvlen=actual_seq_lengths_kv,
    atten_mask=attention_mask,
    learnable_sink=self.sinks,
)
```

### Qwen3-MoE（BSH layout, FA v1, Prefill/Decode 分离）

```python
# 参考: cann-recipes-infer/models/qwen3_moe/models/modeling_qwen3_moe.py
# Decode
attn_output, _ = torch.ops.npu.npu_fused_infer_attention_score(
    query_states, past_key_states, past_value_states,
    num_heads=self.num_heads_per_rank,
    num_key_value_heads=self.num_key_value_heads_per_rank,
    input_layout="BSH", atten_mask=attention_mask,
    scale=self.scale_fa, actual_seq_lengths_kv=actual_seq_lengths_kv,
)

# Prefill（注：sparse_mode=2 为仓库历史实现，推荐使用 sparse_mode=3）
attn_output, _ = torch.ops.npu.npu_fused_infer_attention_score(
    query_states, key_states, value_states,
    num_heads=self.num_heads_per_rank,
    num_key_value_heads=self.num_key_value_heads_per_rank,
    input_layout="BSH", atten_mask=attention_mask,
    sparse_mode=2, scale=self.scale_fa, next_tokens=0,
)
```

## 模式二 + 模式三：分页注意力 + MLA 压缩缓存

以下示例均使用 PA（block_table）+ MLA（key=value=cache_nope, query_rope/key_rope 分离）。

### DeepSeek-R1（TND_NTD layout, FA v2, MLA absorb）

```python
# 参考: cann-recipes-infer/models/deepseek_r1/models/modeling_deepseek.py
attn_output, _ = self.fa_ops.npu_fused_infer_attention_score_v2(
    q_nope, k_nope, k_nope,
    query_rope=q_pe, key_rope=k_rope,
    atten_mask=attention_mask,
    actual_seq_kvlen=actual_seq_lengths_kv,
    actual_seq_qlen=actual_seq_lengths_q,
    block_table=self.block_table,
    num_query_heads=self.num_heads_per_rank,
    num_key_value_heads=self.num_key_value_heads_per_rank,
    softmax_scale=self.softmax_scale,
    input_layout="TND_NTD",
    sparse_mode=0,
    block_size=self.block_size,
    query_quant_mode=0, key_quant_mode=0, value_quant_mode=0,
)
```

### Kimi-K2（TND_NTD, FA v1, Prefill/Decode 分离实例）

```python
# 参考: cann-recipes-infer/models/kimi-k2-thinking/models/modeling_deepseek.py
fa_input_kwargs = {
    "query": q_nope, "key": k_nope, "value": k_nope,
    "query_rope": q_pe, "key_rope": k_pe,
    "num_heads": self.num_heads_per_rank,
    "num_key_value_heads": self.num_key_value_heads_per_rank,
    "input_layout": "TND_NTD",
    "actual_seq_lengths": actual_seq_qlen,
    "actual_seq_lengths_kv": actual_seq_lengths_kv,
    "sparse_mode": 3, "atten_mask": attention_mask,
    "block_table": block_table, "block_size": self.block_size,
    "scale": self.softmax_scale,
}
if is_prefill:
    attn_output, _ = self.fa_ops_prefill.npu_fused_infer_attention_score(**fa_input_kwargs)
else:
    attn_output, _ = self.fa_ops_decode.npu_fused_infer_attention_score(**fa_input_kwargs)
```

### LongCat-Flash（BSND_NBSD, FA v1, KVP）

```python
# 参考: cann-recipes-infer/models/longcat-flash/models/modeling_longcat_flash.py
attn_partial, lse_partial = self.fa_ops.npu_fused_infer_attention_score(
    query_states[0], k_nope, k_nope,
    query_rope=query_states[1], key_rope=k_rope,
    num_heads=self.num_heads_per_rank,
    num_key_value_heads=self.num_key_value_heads_per_rank,
    input_layout="BSND_NBSD",
    block_table=self.block_table, block_size=self.block_size,
    atten_mask=attention_mask, actual_seq_lengths_kv=actual_seq_lengths_kv,
    scale=self.softmax_scale,
    sparse_mode=sparse_mode,
    softmax_lse_flag=self.kvp_size > 1,
)
```

## 缓存写入融合算子

```python
# 参考: cann-recipes-infer/models/longcat-flash/models/modeling_longcat_flash.py
_, _, k_rope, k_nope = torch_npu.npu_kv_rmsnorm_rope_cache(
    latent_cache, self.kv_a_layernorm.weight,
    cos, sin,
    slot_mapping.view(-1),          # 写入位置
    rope_cache, nope_cache,         # 输出缓存
    epsilon=1e-6,
    cache_mode="PA_NZ",
    is_output_kv=True,
)
```
