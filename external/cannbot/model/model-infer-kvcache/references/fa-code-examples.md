# FA 调用完整代码示例

按模型分组。部分模型 Prefill 直读 KV（不传 `block_table`）、Decode 走 Paged 模式（传 `block_table`），与独立部署模式一（无 paging、FA 直读整个 cache）不是同一形态。

## 标准 LLM TND / 滑窗 TND 路径

### GPT-OSS（TND layout, FA v2, 滑窗 + sink，Prefill 直读 KV）

```python
# 参考: cann-recipes-infer/models/gpt_oss/models/modeling_gpt_oss.py（Prefill 分支）
# sliding window 层必须保留 band 约束；短序列下与 full attention 结果相同不代表可改 sparse_mode=0。
# Prefill 直接传 key_states / value_states（不走 block_table），Decode 切到 block_table 路径见下方注释。
attn_output, _ = torch_npu.npu_fused_infer_attention_score_v2(
    query_states,
    key_states, value_states,
    num_query_heads=self.num_attention_heads_per_rank,
    num_key_value_heads=self.num_key_value_heads_per_rank,
    input_layout="TND",
    softmax_scale=self.scaling,
    sparse_mode=4 if self.sliding_window else 3,
    pre_tokens=self.sliding_window if self.sliding_window else torch.iinfo(torch.int32).max,
    next_tokens=0,
    actual_seq_qlen=actual_seq_qlen,
    actual_seq_kvlen=actual_seq_qlen,
    atten_mask=attention_mask,
    learnable_sink=self.sinks,
)
# Decode 分支在 Prefill 调用末尾 update_cache 后，额外传 block_table=block_table[self.attn_type] + block_size。
```

### Qwen3-MoE（TND layout, FA v1, Paged 模式标准 LLM 默认路径）

```python
# 参考: cann-recipes-infer/models/qwen3_moe/models/modeling_qwen3_moe.py 函数 exec_qkv
# Prefill + Decode 统一 sparse_mode=3 + TND + block_table
attn_output, _ = torch.ops.npu.npu_fused_infer_attention_score(
    query_states, self.k_cache, self.v_cache,
    num_heads=self.num_heads_per_rank,
    num_key_value_heads=self.num_key_value_heads_per_rank,
    input_layout="TND",
    sparse_mode=3,
    atten_mask=attention_mask,                  # [2048, 2048] bool causal
    actual_seq_qlen=actual_seq_qlen,            # cumulative 形态（取 forward_metadata.actual_seq_lengths_cu_q）
    actual_seq_lengths_kv=actual_seq_lengths_kv,
    scale=self.scale_fa,
    block_table=block_table[self.attn_type],
    block_size=self.block_size,
)
```

> migrator 阶段过渡形态（独立部署 BSH + 连续缓存 + 不传 block_table）参考 `references/standalone_kv_reference.md` §1。

## MLA 系列（TND_NTD layout + MLA absorb）

以下示例均使用 Paged 模式（传 `block_table`）+ MLA（key=value=cache_nope, query_rope/key_rope 分离）。

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

> `BSND_NBSD` layout 仅 KVP 长序列（KV 沿 head 切分多卡）场景使用，普通 Paged 模式仍走 TND / TND_NTD 默认推荐路径。

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

## MLA absorb 权重初始化范式

`kv_b_proj.weight` 拆为 W_uk / W_uv 给 FA absorb 路径；`nn.Parameter` wrap 必须在 `process_weights_after_loading` 内（依赖 ParallelLinear post-load transpose 完成），不能在 `__init__` 直接 wrap。

```python
# 参考: cann-recipes-infer/models/deepseek_r1/models/modeling_deepseek.py
# __init__：拆 + permute，停在 *_data buffer 形态
self.kv_b_proj_w_k_data, self.kv_b_proj_w_v_data = kv_b_proj_weight.split(...)
self.kv_b_proj_w_k_data = self.kv_b_proj_w_k_data.permute(1, 2, 0)    # [N_h, qk_nope, kv_lora]
self.kv_b_proj_w_v_data = self.kv_b_proj_w_v_data.transpose(0, 1)     # [N_h, kv_lora, v_head]

# process_weights_after_loading：转 nn.Parameter
def process_weights_after_loading(self):
    self.init_splited_kv_b_weight()      # *_data → nn.Parameter
```

多 sublayer 模型遍历用 `for layer in self.model.layers: for sub in layer.self_attn`，不要 `self.modules() + isinstance(MLA)`——模块注册顺序不保证，会与 `cache_entries` 的 layer_idx 错位。

## 缓存写入融合算子

```python
# 参考: cann-recipes-infer/models/longcat-flash/models/modeling_longcat_flash.py
_, _, k_rope, k_nope = torch_npu.npu_kv_rmsnorm_rope_cache(
    latent_cache, self.kv_a_layernorm.weight,
    cos, sin,
    slot_mapping.view(-1),                              # 写入位置
    rope_cache, nope_cache,                             # 输出缓存
    epsilon=self.kv_a_layernorm.variance_epsilon,       # 用 RMSNorm 模块的 eps，不要硬编码
    cache_mode="PA_NZ",
    is_output_kv=True,
)
```
