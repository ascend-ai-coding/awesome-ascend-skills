# Attention 层：MLA + Indexer 稀疏路径

**参考模型**：`cann-recipes-infer/models/deepseek-v3.2-exp/`、`cann-recipes-infer/models/glm-5/`

**核心特征**：在 MLA Absorb 基础上增加 Indexer（Top-K KV Block 选择），FA 替换为稀疏版本。**Prefill 和 Decode 共用 `forward_absorb` 路径**。

## Attention 链路（Prefill 和 Decode 共用）

```python
# ─── Pre-Norm ───
hidden_states, residual = npu_add_rms_norm(residual, hidden_states, weight, eps)

# ─── 超级融合 prolog（Prefill 和 Decode 都用 npu_mla_prolog_v3）───
q_nope, q_pe, _, qr, _ = npu_mla_prolog_v3(
    token_x=hidden_states,
    weight_dq=..., weight_uq_qr=..., weight_uk=kv_b_proj_w_k,
    weight_dkv_kr=...,
    kv_cache=nope_cache, kr_cache=rope_cache,
    cache_mode="PA_BSND",              # 注意：不是 PA_NZ
    cache_index=slot_mapping,          # Decode
    # Prefill 且 cp_size>1 时 cache_mode="BSND"，后续手动 scatter_update_ 写入 cache
    weight_quant_mode=1)               # 仅 qb 量化
    # C8 路径额外传：kv_cache_quant_mode=3, tile_size=128

# ─── Indexer：Top-K KV Block 选择 ───
#    内部流程：投影 → RoPE(npu_rotary_mul) → 量化(npu_dynamic_quant, C8时) → Top-K
topk_indices = npu_lightning_indexer(q, k, weights, ...)
#   C8 路径：npu_lightning_indexer_quant(...)
#   Indexer KV Cache: Prefill 用 scatter_update_，Decode 用 npu_scatter_nd_update_

# ─── 稀疏 Flash Attention（key=value=latent cache，通过 topk_indices 选择）───
#   FP16 路径：
output = npu_sparse_flash_attention(
    query=q_nope, key=k_latent, value=k_latent,
    query_rope=q_pe, key_rope=k_pe,
    sparse_indices=topk_indices,
    layout_query='TND', layout_kv='PA_BSND', sparse_mode=3)
#   C8 路径：
output = npu_sparse_flash_attention_antiquant(
    query=cat([q_nope, q_pe]), key=k_latent, value=k_latent,
    key_quant_mode=2, attention_mode=2, rope_head_dim=64, ...)

# ─── V absorb ───
output = matmul(output, kv_b_proj_w_v)

# ─── O 投影（支持 oproj_tp_size 额外并行维度）───
output = o_proj(output)
```

## Prefill vs Decode 关键差异

| 环节 | Prefill | Decode |
|------|---------|--------|
| mla_prolog cache_mode | `PA_BSND`（CP 时 `BSND` + 后续 `scatter_update_`） | `PA_BSND` |
| Indexer KV Cache 写入 | `scatter_update_` | `npu_scatter_nd_update_` |
| Offload（可选） | 无 | `npu_gather_selection_kv_cache` 聚集离散 KV Block |

## 与 MLA Absorb（无 Indexer）的核心差异

| 维度 | MLA Absorb（deepseek-r1） | MLA + Indexer（deepseek-v3.2） |
|------|----------------------|------------------------|
| Prefill Attention | 展开 K/V → 标准 FA（v1） | absorb → Indexer → 稀疏 FA |
| Decode Attention | absorb → FA v2（全量 KV） | absorb → Indexer → 稀疏 FA |
| FA 算子 | `npu_fused_infer_attention_score(_v2)` | `npu_sparse_flash_attention` |
| KV Cache layout | `PA_NZ` | `PA_BSND` |
| V absorb 算子 | `npu_transpose_batchmatmul` | `torch.matmul` |
