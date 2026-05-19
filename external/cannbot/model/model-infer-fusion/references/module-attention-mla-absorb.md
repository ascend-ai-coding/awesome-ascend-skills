# Attention 层：MLA Absorb 路径（无 Indexer）

**参考模型**：`cann-recipes-infer/models/deepseek_r1/`、`cann-recipes-infer/models/kimi-k2-thinking/`

**核心特征**：Q 吸收 W_uk 使维度降至 `kv_lora_rank`，KV Cache 只存 latent 向量。**Prefill 和 Decode 走不同算子路径**。

## Prefill 链路

参考 `deepseek-r1/forward_page_attention_normal`

```python
# ─── Pre-Norm ───
hidden_states, residual = npu_add_rms_norm(residual, hidden_states, weight, eps)

# ─── Q 投影 ───
q = q_b_proj(npu_rms_norm(q_a_proj(x)))
q_nope, q_pe = q.split([qk_nope_head_dim, qk_rope_head_dim])

# ─── KV 低秩投影 ───
latent_cache = kv_a_proj_with_mqa(x)

# ─── KV RMSNorm + RoPE + Cache Write（三合一融合）───
k_rope, k_nope = npu_kv_rmsnorm_rope_cache_v2(
    latent_cache, kv_a_layernorm.weight, cos, sin,
    slot_mapping, rope_cache, nope_cache,
    cache_mode="PA_NZ", is_output_kv=True)

# ─── Q RoPE ───
q_pe = npu_interleave_rope(q_pe, cos, sin)

# ─── 展开 K/V（从 latent 反投影到 full dim，非 absorb）───
k_full = matmul(k_nope, kv_b_proj_w_k)
v_full = matmul(k_nope, kv_b_proj_w_v)

# ─── Flash Attention（标准 v1）───
output = npu_fused_infer_attention_score(
    q_nope, k_full, v_full,
    query_rope=q_pe, key_rope=k_rope,
    input_layout="NTD_TND", sparse_mode=3)

# ─── O 投影 ───
output = o_proj(output)
```

## Decode 链路（推荐路径）

参考 `deepseek-r1/forward_page_attention_mla_prolog`

```python
# ─── Pre-Norm ───
hidden_states, residual = npu_add_rms_norm(residual, hidden_states, weight, eps)

# ─── 可选：npu_dynamic_quant 量化 hidden_states（W8A8 场景）───

# ─── 超级融合 prolog（QKV投影 + LayerNorm + RoPE + Q absorb + KV Cache Write）───
q_nope, q_pe, dequant_scale, _, _ = npu_mla_prolog_v3(
    token_x=hidden_states,
    weight_dq=q_a_proj.weight, weight_uq_qr=q_b_proj.weight,
    weight_uk=kv_b_proj_w_k,         # Q absorb：W_uk 吸收进 Q
    weight_dkv_kr=kv_a_proj.weight,
    rmsnorm_gamma_cq=..., rmsnorm_gamma_ckv=...,
    kv_cache=nope_cache, kr_cache=rope_cache,
    cache_index=slot_mapping,
    cache_mode="PA_NZ",
    weight_quant_mode=2,              # 0:无量化 1:仅qb 2:全int8
    kv_cache_quant_mode=1)            # 0:无 1:per-tensor

# ─── Flash Attention v2（absorb 后 key=value=latent cache）───
output = npu_fused_infer_attention_score_v2(
    q_nope, k_nope_cache, k_nope_cache,    # key 和 value 都是 latent
    query_rope=q_pe, key_rope=k_rope_cache,
    block_table=block_table,
    actual_seq_kvlen=actual_seq_lengths_kv,
    input_layout="TND_NTD", sparse_mode=0)

# ─── V absorb（FA 后用 W_v 反投影）───
output = npu_transpose_batchmatmul(output, kv_b_proj_w_v)

# ─── O 投影 ───
output = o_proj(output)
```

## Prefill vs Decode 关键差异

| 环节 | Prefill | Decode |
|------|---------|--------|
| QKV + Norm + RoPE + Cache | 分步：手动投影 + `npu_kv_rmsnorm_rope_cache_v2` | 超级融合：`npu_mla_prolog_v3` 一步完成 |
| K/V 形态 | 展开为 full dim（`matmul(latent, W_k/W_v)`） | 不展开，保持 latent（absorb） |
| FA 算子 | `npu_fused_infer_attention_score`（v1） | `npu_fused_infer_attention_score_v2` |
| V absorb | 无（Prefill 已展开 V） | `npu_transpose_batchmatmul(output, W_v)` |
