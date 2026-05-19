# 仓库参考配置索引

分析时应读取最接近的参考模型配置，对比推荐值与已验证配置的差异。设计文档路径：`cann-recipes-infer/docs/models/{model}/`。

---

## Decode 高吞吐（大 EP）

MoE 模型 Decode 阶段的主流模式：Attention DP + MoE 大 EP，embed/lmhead 独立设 TP。

| 参考模型 | 卡数 | attn_tp | dense_tp | moe_tp | embed_tp | lmhead_tp | oproj_tp | 量化 | 特殊配置 | 配置路径 |
|---------|------|---------|----------|--------|----------|-----------|----------|------|---------|---------|
| DeepSeek-R1 | 128 | 1 | 1 | 1 | 1 | — | — | W8A8 | moe_chunk=65536, prefill_multi_cycle | `decode_r1_rank_128_128ep_a8w8.yaml` |
| DeepSeek-R1 | 128 | 1 | 1 | 1 | 1 | — | — | W8A8C8 | MTP, perfect_eplb, SuperKernel | `decode_r1_rank_128_128ep_a8w8c8_mtp_benchmark.yaml` |
| DeepSeek-R1 | 16 | 1 | 1 | 1 | 16 | 16 | — | W8A8 | 16EP, micro_batch=0 | `decode_r1_rank_16_16ep_a8w8.yaml` |
| DeepSeek-V3.2-Exp | 128 | 1 | 4 | 1 | 16 | 16 | 8 | BF16 | cp_size=128 | `deepseek_v3.2_exp_rank_128_128ep_decode_benchmark.yaml` |
| DeepSeek-V3.2-Exp | 128 | 1 | 4 | 1 | 16 | 16 | 4 | W8A8C8 | MTP3, cp_size=128 | `deepseek_v3.2_exp_rank_128_128ep_w8a8c8_decode_benchmark.yaml` |
| GLM-5 | 128 | 1 | 4 | 1 | 16 | 16 | 4 | W8A8 | MTP3, cp_size=128 | `glm_5_rank_128_128ep_w8a8_decode_benchmark.yaml` |
| Kimi-K2 | 128 | 1 | 4 | 1 | 16 | 16 | 8 | — | cp_size=128 | `kimi_k2_thinking_rank_128_128ep_decode_benchmark.yaml` |
| LongCat-Flash | 128 | 1 | 8 | 1 | 1 | 16 | 8 | W8A8 | AFD, MTP2, perfect_eplb, SuperKernel, prefetch | `longcat_flash_densetp8_ep128_*.yaml` |
| Qwen3-MoE | 128 | 1 | 1 | 1 | 1 | 1 | — | — | 128EP 纯EP模式 | `qwen3_235b_128ep.yaml` |

**经验总结**：
- 128 卡 Decode 通用模式：attn_tp=1, moe_tp=1
- embed/lmhead_tp 取决于词表大小：V3.2/GLM-5/Kimi-K2（大词表）用 16，R1/Qwen3（小词表或全 MoE）用 1
- dense_tp 取决于模型结构：V3.2/GLM-5/Kimi-K2（有 Shared Expert）用 4，LongCat（Dense FFN 大）用 8，R1（全 MoE）用 1
- oproj_tp 在 MLA 模型中独立设值（4 或 8），非 MLA 模型不需要

---

## Decode 低时延（纯 TP）

Dense 模型或小规模 MoE 部署，全 TP 最小化通信延迟。

| 参考模型 | 卡数 | attn_tp | dense_tp | moe_tp | 量化 | 配置路径 |
|---------|------|---------|----------|--------|------|---------|
| DeepSeek-R1 | 16 | 16 | 16 | 16 | W8A8 | `decode_r1_rank_16_16tp_a8w8.yaml` |
| Qwen3-MoE | 16 | 16 | 16 | 16 | — | `qwen3_235b_16tp.yaml` |
| GPT-OSS 120B | 8 | 8 | — | 8 | — | `gpt_oss_120b_8tp.yaml` |
| GPT-OSS 20B | 1 | 1 | — | 1 | — | `gpt_oss_20b.yaml` |

**经验总结**：
- 纯 TP 适合 Dense 模型（GPT-OSS）或 MoE 小规模部署（≤16 卡）
- all tp = W 最简单，但通信量随 W 增长；跨节点时性能急剧下降
- Dense 模型没有 dense_tp（只有 attn_tp 和 moe_tp）

---

## Decode 混合（Attention DP + Dense TP + MoE EP）

MoE 模型的差异化并行：各模块按计算/访存特性独立设 TP。

| 参考模型 | 卡数 | attn_tp | dense_tp | moe_tp | oproj_tp | 量化 | 特殊配置 | 配置路径 |
|---------|------|---------|----------|--------|----------|------|---------|---------|
| LongCat-Flash | 32 | 1 | 8 | 1 | 1 | — | eager, moe_chunk=1024 | `longcat_flash_densetp8_ep32.yaml` |
| LongCat-Flash | 32 | 1 | 8 | 1 | 8 | — | ge_graph, MTP | `longcat_flash_densetp8_ep32_gegraph_mtp.yaml` |
| Qwen3-MoE | 32 | 4 | — | 1 | — | — | attn4tp+8dp | `qwen3_235b_attn4tp8dp.yaml` |

**经验总结**：
- 混合模式核心思路：Attention 用 DP（Decode 时 KV 访存为主），Dense FFN 用 TP（大矩阵），MoE 用 EP（专家分散）
- oproj_tp 在图模式下需要和 dense_tp 匹配（LongCat ge_graph 时 oproj=8）
- Qwen3 的 attn4tp 示例说明 Attention TP 可以取中间值（不一定是 1 或 W）

---

## Prefill 长序列（CP / 大 TP）

Prefill 阶段以 TTFT 为目标，长序列需要 CP 或大 TP 分摊计算和显存。

| 参考模型 | 卡数 | attn_tp | dense_tp | moe_tp | embed_tp | lmhead_tp | oproj_tp | cp_size | 序列长度 | 量化 | 配置路径 |
|---------|------|---------|----------|--------|----------|-----------|----------|---------|---------|------|---------|
| DeepSeek-R1 | 32 | 32 | 32 | 1 | 32 | 32 | — | — | 65536 | W8A8 | `prefill_k2_rank_32_32sp_32tp_32ep_a8w8.yaml` |
| DeepSeek-R1 | 32 | 1 | 1 | 1 | 32 | 32 | — | — | 4096 | W8A8 | `prefill_r1_rank_32_32dp_32ep_a8w8.yaml` |
| DeepSeek-V3.2-Exp | 64 | 1 | 1 | 1 | 16 | 16 | 8 | 64 | 65536 | BF16 | `*_rank_64_64ep_prefill_benchmark.yaml` |
| DeepSeek-V3.2-Exp | 64 | 1 | 1 | 1 | 16 | 16 | 1 | 64 | 65536 | W8A8C8 | `*_rank_64_64ep_w8a8c8_prefill_benchmark.yaml` |
| GLM-5 | 64 | 1 | 1 | 1 | 16 | 16 | 1 | 64 | 65536 | W8A8 | `glm_5_rank_64_64ep_w8a8_prefill_benchmark.yaml` |
| Kimi-K2 | 64 | 1 | 1 | 1 | 16 | 16 | 1 | 64 | 65536 | — | `kimi_k2_thinking_rank_64_64ep_prefill_benchmark.yaml` |

**经验总结**：
- **CP 是长序列 Prefill 的标准方案**：V3.2-Exp/GLM-5/Kimi-K2 都用 cp_size=卡数，attn_tp=1
- CP 模式下 Attention 不用 TP（attn_tp=1），靠 CP 沿序列维度切分
- DeepSeek-R1 65K 序列 Prefill 用的是大 TP（attn_tp=32）而不是 CP——R1 没有 sparse attention，TP 更直接
- R1 4K 短序列 Prefill 用纯 DP（attn_tp=1），此时序列短不需要分摊
- Prefill 的 oproj_tp 通常较小（1 或 8），因为 Prefill 阶段 o_proj 计算量相对不大
- embed/lmhead_tp 在 Prefill 通常设为节点内 TP（如 16），分摊词表显存

---

## 超长序列（KVP / Offload）

256K+ 序列的 KV Cache 超出单卡显存，需要 KVP 或 Offload。

| 参考模型 | 卡数 | kvp_size | oproj_tp | 序列长度 | 量化 | 特殊配置 | 配置路径 |
|---------|------|---------|----------|---------|------|---------|---------|
| LongCat-Flash | 32 | 8 | 8 | 131072 | W8A8 | decode_only, MLA_prolog | `longcat_flash_densetp8_ep32_kvp8_gegraph_w8a8.yaml` |
| DeepSeek-V3.2-Exp | 128 | — | 4 | 65536 | W8A8C8 | enable_offload=T | `*_w8a8c8_offload_benchmark.yaml` |
| DeepSeek-R1 | 128 | — | — | 65536 | W8A8C8 | dense_tp=4, lmhead_tp=16, moe_chunk=512 | `decode_r1_rank_128_128ep_a8w8c8_mtp_longseq.yaml` |

**经验总结**：
- **KVP**：KV Cache 按 head 维度切到多卡，每卡算部分 Attention 后聚合。约束：`oproj_tp = kvp_size`
- **KVCache Offload**：KV Cache 存主机内存，利用 TopK 局部性（~60% 命中率）减少 H2D 传输。需要 GatherSelectionKvCache 算子
- R1 长序列方案不同：不用 KVP/Offload，而是 dense_tp=4 + moe_chunk=512 控制显存
- 超长序列时 moe_chunk_max_len 要调小（512-1024），否则 MoE 激活显存爆

---

## oproj_tp_size 使用规则

`oproj_tp_size`（o_proj 输出投影的 TP 度）在 MLA 和复杂并行场景中独立于 attn_tp：

| 场景 | oproj_tp 取值 | 约束 | 参考 |
|------|-------------|------|------|
| 无 MLA / 简单 TP | 不需要（跟随 attn_tp） | — | GPT-OSS, R1 |
| MLA Decode 大 EP | 4-8 | 需整除 `num_heads * v_head_dim` | V3.2-Exp, GLM-5, Kimi-K2 |
| KVP 模式 | = kvp_size | 强制对齐 | LongCat-Flash |
| Prefill CP 模式 | 1 或 local_tp | 通常较小 | V3.2-Exp Prefill |

---

## 部署链路速查

一个模型通常需要 **Prefill + Decode 两套配置**，按需选择：

| 模型 | Prefill 配置 | Decode 配置 | 设计文档 |
|------|------------|-----------|---------|
| DeepSeek-R1 | 32 卡 TP32 或 DP32 | 128 卡 EP128 | `cann-recipes-infer/docs/models/deepseek-r1/` |
| DeepSeek-V3.2-Exp | 64 卡 CP64 | 128 卡 EP128+dense_tp4 | `cann-recipes-infer/docs/models/deepseek-v3.2-exp/` |
| GLM-5 | 64 卡 CP64 | 128 卡 EP128+dense_tp4 | — |
| Kimi-K2 | 64 卡 CP64 | 128 卡 EP128+dense_tp4 | `cann-recipes-infer/docs/models/kimi-k2-thinking/` |
| LongCat-Flash | — | 32/128 卡 dense_tp8+EP | `cann-recipes-infer/docs/models/longcat-flash/` |
| Qwen3-MoE | — | 16 卡 TP16 / 128 卡 EP128 | `cann-recipes-infer/docs/models/qwen3-moe/` |
| GPT-OSS | — | 8 卡 TP8 / 单卡 | `cann-recipes-infer/docs/models/gpt-oss/` |

---

## 设计文档匹配

| 目标模型特征 | 参考文档 | 重点关注 |
|-------------|---------|---------|
| MoE + MLA + 大规模 EP Decode | `cann-recipes-infer/docs/models/deepseek-r1/` | EP 通信调优、MTP 收益、moe_chunk 配置 |
| MoE + MLA + DSA/Sparse Attention | `cann-recipes-infer/docs/models/deepseek-v3.2-exp/` | CP vs TP 选择依据、KVCache Offload、MLA Absorb 权衡 |
| MoE + MLA + 超长序列（256K+）| `cann-recipes-infer/docs/models/longcat-flash/` | KVP 约束、多流 core limiting、AFD、权重预取 |
| MoE + 标准 GQA + 中规模 TP | `cann-recipes-infer/docs/models/qwen3-moe/` | head 切分、gate+up 合并、npu_swiglu 融合 |
| MoE + W4A16 混合精度 | `cann-recipes-infer/docs/models/kimi-k2-thinking/` | 混合精度策略、Flash Decode |
| Dense + 标准 GQA | `cann-recipes-infer/docs/models/gpt-oss/` | 纯 TP 配置、Fixed KV Cache |
