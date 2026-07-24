---
name: external-cannbot-model-model-infer-kvcache
description: 基于 PyTorch 框架的昇腾 NPU 模型推理 KVCache 优化技能。分析并改造 LLM 推理模型的 KVCache 实现，覆盖
  Legacy 连续缓存与分页注意力（Paged Attention）配 FA 融合算子、MLA 压缩缓存、SlidingWindow / 多 attn_type
  混合。触发场景：KVCache 管理实现、分页注意力接入、KV 压缩、FA 融合算子、OOM / 性能问题、block_table / slot_mapping
  构造。支持框架部署与独立部署两种模式。
original-name: model-infer-kvcache
synced-from: https://gitcode.com/cann/cannbot-skills
synced-date: '2026-07-24'
synced-commit: bff73845607ac78808e2af3e0014d7eb72094ef3
license: UNKNOWN
---

# KVCache 优化技能

按模型 attention 形态选择 KVCache 模式与算子组合，并完成代码改造与验证。本技能由 analyzer（选型分析）和 implementer（代码实施）共同调用——前者只读「第一层快速选型」与「第二层数据结构与算子」，后者按部署模式读「第三层实施流程」对应大节。前两层为两模式共通内容，第三层按部署模式分岔。

---

## 部署模式判定

**框架部署模式**：模型接入 `cann-recipes-infer/executor/core/`，调度 / 批组装 / KV 管理由框架统一负责。KV 路径由模型类自选两种模式：
- **Legacy 模式（migrator 输出的起点）**：attention 声明 `cache_unit = (num_kv_heads_per_rank * head_dim,)`，框架据此静态预分配 `k_cache` / `v_cache`，forward 内 `torch_npu.scatter_update_` 按 `kv_len` 写入；配合 HF 原版 SDPA 即可跑通。
- **Paged 模式（推荐）**：模型类实现 `get_cache_info()` 返回 `ModelCacheInfo`，每层声明 `cache_entries`；`BlockPool` 动态分配 block、`slot_mapping` / `block_table` 由 `cann-recipes-infer/executor/core/kv_cache/` 自动构造，配合 NPU FA 融合算子（v1 / v2）+ TND layout（设计文档 §1 起均以 Paged Attention 为框架核心机制）。

**本 skill 决策点**：是否把 Legacy KV 改造为 Paged 模式。默认推荐改造，仅在架构限制 / 算子不兼容场景下保留 Legacy。

**独立部署模式**：Runner 自管 KV tensor 分配、`block_table` 静态预分配、`slot_mapping` 动态计算，并自管 `forward_metadata` 构造。

### 判定优先级

| 优先级 | 判定来源 | 说明 |
|---|---|---|
| 1 | dispatch prompt 显式指定 | prompt 含 `部署模式: 框架部署` 或 `部署模式: 独立部署` → 直接采用 |
| 2 | 用户对话指定 | 仅独立调用时适用 |
| 3 | migrator 阶段产物自检 | 模型注册到 `cann-recipes-infer/executor/core/support_models.py` → **框架部署**；模型目录有 `runner_*.py` 不继承 framework 基类、自管推理循环 → **独立部署**；不确定时默认框架部署 |

判定后跳转：第三层「3.1 框架部署模式实施」或「3.2 独立部署模式实施」。第一层与第二层是两模式共通理论，无需分流。

---

## 第一层：快速选型

按模型 attention 形态决定 KVCache 模式 + attn_type + FA 算子组合，输出选型清单交由后续实施步骤接入。

### 1.1 KV 模式选型

> **改造目标：Paged 模式（FA + TND layout）**
> - **Paged**：`BlockPool` 动态分配 block，支撑 batch 内序列长度差异（设计文档 §1 / §2 的核心机制）
> - **FA**：NPU 融合算子（`npu_fused_infer_attention_score{,_v2}`），性能远优于标准 SDPA
> - **TND layout**：变长 batch 拼一维，与 `cann-recipes-infer/executor/core/` ExecutionEngine 的 `input_ids: [TotalTokens]` packed 输入协议契合；MLA 用 `TND_NTD`

| 模型类型 | KV 模式 | 框架部署接入 | 独立部署接入 | 参考模型 |
|---|---|---|---|---|
| 标准 LLM（MHA/GQA，含 MoE 变体） | **Paged 模式（FA + TND）** | `attn_type="FullAttention"` | 模式二（手动 Paged + FA） | qwen3_8b、qwen3-moe |
| 含滑窗 attention | **Paged 模式（FA + TND + 滑窗约束）** | `attn_type="SlidingWindow"`（含 `sliding_window`） | 模式二（手动 Paged + sparse_mode=4） | — （仓内 gpt-oss 是滑窗 + 全注意力混合，参考其滑窗层实现） |
| MLA 架构 | **Paged 模式（FA + TND_NTD + MLA absorb）** | `attn_type="FullAttention"`（nope + rope 两 entry，模型类设 `is_mla_backend=True`） | 模式三（MLA absorb） | deepseek_r1、deepseek-v3.2、kimi-k2 |
| migrator 输出起点（未改造） | Legacy + SDPA + BSH | `cache_unit` 静态预分配 + `scatter_update_`（不进 `ATTN_TYPE_MANAGER_MAP`） | 模式一（list of (k,v)） | — |

> **Legacy 模式仅支持 offline 推理**。要使用 online / PD 服务或框架数据集评测能力，必须升级到 Paged 模式（实现 `get_cache_info()` 接入 KVCacheManager）。

> **MLA absorb 路径补充**：
> - latent KV 共享，**不沿 `attn_tp_size` 切分**（每 attn_tp rank 持完整 latent KV；Q 按 head 切、attn_dp 按 batch 切）。
> - V 由 latent 经 `kv_b_proj` 重投影获得，自动消除 `qk_head_dim ≠ v_head_dim` 时 V 需 pad 到 qk_head_dim 的问题。

### 1.2 算子改造

| 改造项 | 触发条件 | 框架部署 | 独立部署 |
|---|---|---|---|
| SDPA → NPU FA 融合算子（FA v1 / v2） | 性能必做 | 改造时接入 FA 调用（v1 / v2 对照见 §2.7） | modeling 内手动接入 FA 调用 |
| 标准 indexing → `npu_scatter_nd_update_` 或 `npu_kv_rmsnorm_rope_cache` | 性能必做 | 标准/滑窗用 `npu_scatter_nd_update_`，MLA 用 `npu_kv_rmsnorm_rope_cache` 融合写入 | Runner 自管 cache + slot_mapping，modeling 内手动选写入算子 |
| MLA absorb（key=value=cache_nope，rope 分离传入） | 模型架构是 MLA | modeling forward 内实现，cache_entries 配 nope_cache + rope_cache | 同左，cache 由 Runner 自管 |
| 多 attn_type 混合处理 | 模型含混合层 | 各层 `__init__` 按条件设置 `attn_type`，框架按 attn_type 自动分组 | 较少见，可手动模拟 |

### 1.3 决策流程

部署模式（框架 / 独立）与 KV 模式（Legacy / Paged）正交：

```
前置：从 progress.md 读取（1）部署模式（2）架构类型（基于 config）（3）migrator 输出起点
       │
改造为 Paged 模式？─→ 否（极少：架构限制 / 算子不兼容）→ 保留 Legacy
                    │       └─ 框架部署：`cache_unit` + `scatter_update_` + SDPA
                    │       └─ 独立部署：模式一 list of (k,v) + SDPA + BSH
                    │
                    └→ 是 → 按架构选 attn_type + layout（部署模式不影响该选择）
                            ├─ 标准 LLM（MHA/GQA/MoE） → `FullAttention` + TND
                            ├─ 含滑窗层               → `SlidingWindow` + TND + sparse_mode=4 + pre_tokens
                            └─ MLA 架构              → `FullAttention`（nope + rope 两 entry）+ TND_NTD + MLA absorb
                            │
                            └─→ 多 attn_type 混合？ → 各层 `__init__` 按条件设置 `attn_type`
```

### 1.4 选型完成标志（analyzer 输出）

- [ ] 是否改造为 Paged 模式确定（不改造时给出 Legacy 保留依据）
- [ ] 各 cache 的 `attn_type` 确定（`FullAttention` / `SlidingWindow`）
- [ ] FA 版本确定（v1 / v2）
- [ ] 是否启用 MLA absorb 确定
- [ ] 是否多 attn_type 混合确定

---

## 第二层：核心数据结构与算子

无论框架部署还是独立部署，KV cache 涉及的数据结构（`block_table` / `slot_mapping` / `actual_seq_lengths` / `kv_len`）和 FA 算子参数语义都是相同的。差异仅在「谁负责构造」（框架 vs Runner）。

### 2.1 物理内存布局与逻辑映射

分页注意力把**逻辑地址**`(batch_idx, seq_pos)` 映射到**物理 block**。物理存储 shape `[total_num_blocks, block_size, num_kv_heads, head_dim]`。

**映射公式**：

```
逻辑 block 编号  = seq_pos // block_size
block 内偏移     = seq_pos % block_size
物理 block ID    = block_table[batch_idx, 逻辑 block 编号]
物理 slot 位置   = 物理 block ID × block_size + block 内偏移
```

### 2.2 block_table 构造

shape `[batch_size, num_blocks_per_seq]`，dtype `int32`，推理全程**不变**：

- **框架部署**：`prepare_block_tables()`（`cann-recipes-infer/executor/core/kv_cache/cache_utils.py`）按各请求构造，输出 `Dict[attn_type → Tensor]`，attention 内取 `block_table[self.attn_type]`。
- **独立部署**：Runner 在 `_init_kvcache(...)` 内一次性构造（`arange(total_blocks).reshape(batch_size, -1)`）并持有 `self.block_table`，传给 modeling。完整代码见 `references/standalone_kv_reference.md` §2.1。

### 2.3 slot_mapping 构造

`slot_mapping` 是缓存**写入**位置索引，给 `npu_kv_rmsnorm_rope_cache` / `npu_scatter_nd_update_` 等写入算子用。

**两种部署模式的 slot 计算逻辑不同**：

| 部署模式 | 公式 | 说明 |
|---|---|---|
| **独立部署**（block_table 静态预分配为 arange） | `slot(batch_idx, seq_pos) = batch_idx × max_seq_len + seq_pos` | 顺序分配模式下等于展平后线性索引 |
| **框架部署**（BlockPool 动态分配 block） | `slot = block_table[batch_idx, seq_pos // block_size] × block_size + seq_pos % block_size` | 框架自动构造，模型代码不感知 |

> 独立部署 Runner 自管 block_table 时通常走静态预分配，与线性公式一致；框架部署由 `BlockPool` 动态分配 block，slot 计算依赖 `block_table` 索引。两种公式在 block_table 恰为 arange 时等价。

**实施差异**：
- 框架部署：`prepare_slot_mapping()`（`cann-recipes-infer/executor/core/kv_cache/cache_utils.py`）按 `position_ids` + `block_table` 自动生成，输出 `Dict[attn_type → Tensor]`，模型代码从 `forward_metadata.slot_mapping` 取。
- 独立部署：Runner 每步推理前调 `_build_slot_mapping_prefill / decode` 自管构造（Prefill 按 batch 拼接、Decode 用 `kv_len + kv_len_offset`），完整代码见 `references/standalone_kv_reference.md` §2.2。

**slot_mapping 与 block_table 的分工**：
- `slot_mapping` → 缓存**写入**算子（`npu_kv_rmsnorm_rope_cache` / `npu_scatter_nd_update_`）
- `block_table` → FA **注意力读取**算子（`npu_fused_infer_attention_score{,_v2}`）
- 二者寻址逻辑一致，职责不同

### 2.4 actual_seq_lengths 构造

FA 算子需要每个 batch 的实际 KV/Q 长度，构造方式取决于 `input_layout`：

| | TND layout（多 batch token 拼一维） | BSH layout（各 batch 独立） |
|--|------|------|
| **Prefill KV** | `cumsum(kv_len)` → [512, 768] | `kv_len` → [512, 256] |
| **Prefill Q** | 同 KV | 同 KV |
| **Decode KV** | `kv_len` → [522, 266] | `kv_len` → [522, 266] |
| **Decode Q** | `cumsum([1,1])` → [1, 2] | `[1, 1]` |

> ForwardMetaData 字段命名（框架部署模式）：`actual_seq_lengths_kv` / `actual_seq_lengths_q` 是直接形态；`actual_seq_lengths_cu_kv` / `actual_seq_lengths_cu_q` 是 cumulative 形态。FA 调用 TND layout 时 `actual_seq_qlen` 取 cu 形态、`actual_seq_kvlen` 取直接形态。独立部署 Runner 自管构造，可参考 `references/standalone_kv_reference.md` §2.3。

### 2.5 kv_len 生命周期

`kv_len` 是驱动 `slot_mapping` / `actual_seq_lengths_kv` / `position_ids` 的核心变量。Prefill 阶段从 `attention_mask` 计算（`kv_len = max(cumsum(mask)) + 1`），每次 Decode 后递增 1。

> **kv_len 是 Runner 层变量，各层只读不写**：作为参数传入所有 Transformer 层，用同一个值驱动每层各自的 cache 写入位置。不要在 attention/cache 内部递增 `kv_len`——否则 Prefill 阶段各层写入位置逐层偏移，精度损坏。完整代码见 `references/standalone_kv_reference.md` §4。

### 2.6 数据流总览

```
初始化（一次性）：block_table、kv_cache、kv_len_offset

每步 Prefill / Decode：
  kv_len → slot_mapping            → 写 cache（npu_scatter_nd_update_ / npu_kv_rmsnorm_rope_cache）
  kv_len → actual_seq_lengths_kv   → FA 读 cache via block_table
  kv_len → position_ids            → RoPE

Decode 后：kv_len += 1
```

### 2.7 FA 算子配置

#### FA v1 / v2 对照

| 特性 | FA v1 (`npu_fused_infer_attention_score`) | FA v2 (`npu_fused_infer_attention_score_v2`) |
|-----|------------------------------------------|----------------------------------------------|
| 调用方式 | `torch.ops.npu.npu_fused_infer_attention_score(...)` | `self.fa_ops.npu_fused_infer_attention_score_v2(...)` |
| 量化 KV | `antiquant_mode` / `antiquant_scale` | `dequant_scale_key/value` + `query_quant_mode` |
| Sink token | 不支持 | `learnable_sink` 参数 |
| 典型使用 | Qwen3-MoE、LongCat-Flash、Kimi-K2 | DeepSeek-R1、GPT-OSS |

两个版本均支持：Paged 模式（传 `block_table`）/ 非 Paged（不传）、MLA rope 分离（`query_rope` / `key_rope`）。

**v1/v2 关键参数名映射**（混用不报错，会静默落到算子默认值导致精度异常）：

| 功能 | FA v1 | FA v2 | 易错点 |
|------|-------|-------|--------|
| 缩放系数 | `scale` | `softmax_scale` | 默认 1.0，传错名精度崩溃 |
| Q head 数 | `num_heads` | `num_query_heads` | 传错名走默认值 |
| Q 长度 | `actual_seq_lengths` | `actual_seq_qlen` | v1 + BSH + Q_S=1 路径下算子内部忽略，仍需传 |
| KV 长度 | `actual_seq_lengths_kv` | `actual_seq_kvlen` | 名称完全不同 |
| KV head 数 | `num_key_value_heads` | `num_key_value_heads` | **相同** |

> **模型相关约束补充**：
> - **YaRN / `mscale_all_dim` 模型的 scale**：scale = `head_dim^-0.5 × mscale²`（mscale 按 YaRN 公式计算，`mscale_all_dim` 启用时生效）。漏写 mscale² 静默偏低不报错。
> - **MLA absorb 路径**：FA 调用必须 `num_key_value_heads = 1`（latent 单 head），与 `cache_entries.num_head = 1` 严格一致；不一致触发 GQA 比例错。

#### input_layout 选择

| layout | Q 格式 | KV 格式 | 适用场景 |
|--------|--------|---------|---------|
| **`"TND"`** ★ | [T, N, D] | [T, N, D] | **Paged 模式标准 LLM 默认推荐**，变长 batch 拼一维 |
| **`"TND_NTD"`** ★ | [T, N, D] | [N, T, D] | **Paged 模式 MLA 默认推荐**，NZ 格式缓存 |
| `"BSH"` | [B, S, N*D] | [B, S, N*D] | Legacy 模式（框架 `cache_unit` / 独立模式一） |
| `"BNSD"` | [B, N, S, D] | [B, N, S, D] | 非 Paged，扩散模型 |
| `"BSND_NBSD"` | [B, S, N, D] | [N, B, S, D] | Paged + KVP 场景（长序列 KV 沿 head 切分多卡） |

> ★ TND / TND_NTD 是变长 batch packed `[TotalTokens, ...]` 形态；框架部署 Paged 模式下与 `cann-recipes-infer/executor/core/` ExecutionEngine 的 `input_ids: [TotalTokens]` packed 输入契合，独立部署模式二同样推荐。

#### Paged 模式 KV cache 实际 shape

`input_layout` 字符串与 paged KV 的实际 tensor shape 是两个概念，传 FA 前不要按字面 `[T,N,D]` 套到 cache 上：

| Attention | cache_entries 分配 | 传给 FA 的 KV shape |
|---|---|---|
| GQA / MHA | `[bn, bs, num_kv_heads_per_rank, head_dim]` | `view(*shape[:2], -1)` → `[bn, bs, H]` 3D |
| MLA（nope_cache / rope_cache） | 逻辑 `[bn, bs, 1, D]`；写入算子 `cache_mode="PA_NZ"` 实际按 NZ 排布 | view 为 5D NZ `[bn, 1, D/NZ_DIM, bs, NZ_DIM]`（`NZ_DIM=16` for bf16，int8 量化时 ×2） |

Q 在两种模式下均为 packed `[T, N, D]` 3D。`input_layout="TND"` / `"TND_NTD"` 描述的是 FA 算子对各张量的解释方式，paged KV 的实际形态如上表。MLA 路径下 `npu_kv_rmsnorm_rope_cache` / `npu_mla_prolog_v3` 写入必须配 `cache_mode="PA_NZ"`，FA 调用前 view 到 5D NZ；参考 `cann-recipes-infer/models/deepseek_r1/models/modeling_deepseek.py` 搜 `KV_CACHE_NZ_DIM`。

#### sparse_mode 与 atten_mask

FA 算子的 `sparse_mode` 决定注意力遮蔽方式，`atten_mask` 配合提供掩码。两者组合错误是精度问题的第一大来源。

| sparse_mode | 含义 | atten_mask 要求 | 适用场景 |
|:-----------:|------|----------------|---------|
| 0 | Dense | 可选，通常传 **None** | **MLA absorb Decode**（q_len=1 单 token，由 `actual_seq_lengths_kv` 控制有效长度）；非 MLA 路径不推荐 |
| 1 | allMask | **必传**完整矩阵 `(Q_S, KV_S)` | 特殊场景 |
| 2 | leftUpCausal | **不推荐**，建议改用 3 | — |
| 3 | Causal（标准因果） | **必传** `[2048, 2048]` bool 下三角 | **标准 LLM TND PA（Prefill + Decode 统一）**、MLA absorb Prefill、MTP Decode（sq>1，无滑窗） |
| 4 | Band（滑动窗口） | **必传** `[2048, 2048]` bool | 滑窗模型（gpt-oss sliding 层 Prefill + Decode 统一），需配合 `pre_tokens` |

**atten_mask 硬约束**：
- **dtype**：只允许 `torch.bool`（推荐）、`torch.int8`、`torch.uint8`。浮点类型直接报错。
- **shape**：`sparse_mode=3/4` 时固定 `[2048, 2048]`，与 `max_position_embeddings` 无关。本仓库统一用 `executor.utils.common_utils.get_init_attn_mask(2048, device)` 构造。
- **标准 full-attention 单 token Decode 不需要 mask**：`sparse_mode=0` + `atten_mask=None`。
- **滑窗 Decode 必须保留窗口约束**：`sparse_mode=4` + `pre_tokens={sliding_window}` + `next_tokens=0`，不要因短序列下与 full attention 结果等价而改成 `sparse_mode=0`。
- **FA v1 + BSH + Q_S=1（标准 Decode）算子限制**：op 内部**忽略** `sparse_mode` 与 `pre_tokens`，sparse_mode=0 与 4 在该路径下等价。sliding 模型仍应显式写 `sparse_mode=4` 表达意图（FA v2 / Prefill / Q_S>1 路径生效）；**长序列 `KV_len > sliding_window` 的正确性必须靠模型层保证**——环形 buffer 写 cache、或 `actual_seq_lengths_kv` 截断到窗口长度，不是 op 层负责。
- **MLA absorb 模式算子约束**：query D=512 时 `sparse_mode` 仅支持 0、3、4；rope D 必须为 64；MLA query D 仅支持 512 或 128；`query_rope` 与 `key_rope` 必须同时传或同时不传。

**按路径分类的 sparse_mode 写法**：

| 路径 | sparse_mode 写法 | atten_mask | 参考 |
|---|---|---|---|
| 标准 LLM TND PA（FullAttention 非滑窗） | 3（Prefill / Decode 统一） | causal `[2048,2048]` bool | `modeling_qwen.py`、`modeling_qwen3_moe.py` 搜 `sparse_mode` |
| 滑窗层 | `4 if self.sliding_window else 3`（Prefill / Decode 统一） | band `[2048,2048]` bool（配 `pre_tokens=sliding_window` / `next_tokens=0`） | `modeling_gpt_oss.py` 搜 `sparse_mode` |
| MLA absorb | Prefill `3`、Decode `0` 显式分支 | Prefill causal / Decode `None` | `modeling_deepseek.py` 搜 `forward_absorb` |

> 标准 LLM 与滑窗层 Decode（q_len=1）下，causal / band mask 上三角部分不参与计算，因此 Prefill / Decode 可共用一个 sparse_mode 写法。MLA absorb Decode（q_len=1 + 无 mask）改用 sparse_mode=0 减少不必要计算。

#### Prefill vs Decode 参数差异

| 参数 | Prefill | Decode |
|-----|---------|--------|
| `sparse_mode` | 3（因果）；滑窗层 4；MLA absorb 3 | **标准 LLM TND PA：3（与 Prefill 一致）**；滑窗层 4；**MLA absorb：0**；MTP 且无滑窗时 sq>1 用 3 |
| `actual_seq_qlen` | 输入序列长度 | 1（单 token）；MTP 时为 `next_n+1` |
| `actual_seq_kvlen` | = `actual_seq_qlen` | 累计的 KV 长度 |
| `slot_mapping` | 多 token 拼接 | 单 token 偏移 |
| `atten_mask` | 因果 / band mask（**dtype 必须为 bool/int8/uint8**）或 None | 标准 LLM TND PA：causal mask（与 Prefill 同）；滑窗层：band mask；**MLA absorb：None** |

#### NPU KVCache 算子速查

| 算子 | 功能 | 用于 |
|-----|------|------|
| `torch_npu.scatter_update_` | Legacy 形态 KV 按位置写入（接 `kv_len` 索引） | 框架部署 Legacy / 独立部署模式一 |
| `torch_npu.npu_scatter_nd_update_` | PA 形态 KV 按 slot 写入（接 `slot_mapping` 索引） | 框架部署 PA / 独立部署模式二（标准 LLM、滑窗） |
| `torch_npu.npu_kv_rmsnorm_rope_cache` | 融合 RMSNorm + RoPE + Cache 写入（接 `slot_mapping`） | MLA 模型（DeepSeek 系列、LongCat-Flash） |
| `torch_npu.npu_fused_infer_attention_score` | FA v1 融合注意力（PA 时接 `block_table`） | Qwen3-MoE、LongCat-Flash、Kimi-K2 |
| `torch_npu.npu_fused_infer_attention_score_v2` | FA v2 融合注意力（PA 时接 `block_table`） | DeepSeek-R1、GPT-OSS |

完整 FA 调用代码示例（含 GPT-OSS / Qwen3-MoE / DeepSeek-R1 / Kimi-K2 / LongCat-Flash 五种）见 `references/fa-code-examples.md`。

---

## 第三层：实施流程

### 3.1 框架部署模式实施

按 `cann-recipes-infer/docs/design/kv_cache_design.md §5` Checklist 落 6 步。完整代码骨架见 `references/framework_kv_reference.md`。

> **packed sequence 协议（modeling forward 必须遵守）**：框架部署模式下 modeling 接收 packed 一维输入：
> - `input_ids: [TotalTokens]`（Prefill 各 prompt 串接、Decode 每请求 1 token；变长 batch 不做 padding）
> - `position_ids: [TotalTokens]`（**packed 1D**，模型内 RoPE 直接用此索引取 cos/sin）
> - **hidden_states 全程保持 `[TotalTokens, hidden_size]` 二维**（不要 reshape 成 `[B, S, H]`——变长 batch 不能简单 reshape；用 `actual_seq_lengths_cu_q` 在 FA 内表达 batch 边界）
> - **Prefill / Decode 分支判断统一用 `forward_metadata.is_prefill`**
> - 输出 packed logits `[TotalTokens, vocab]`

#### 步骤 1：确定每个 cache 的 attn_type

按第一层选型结果给每个 cache 标注 `attn_type`。本仓库内置：

| attn_type | 对应 Manager | 特性 |
|---|---|---|
| `FullAttention` | `FullAttentionManager` | cache 随序列增长线性追加（GQA / MHA / MLA 均归此类） |
| `SlidingWindow` | `SlidingWindowManager` | 固定窗口大小，cache 不随序列增长（gpt-oss 滑窗层） |

如需新 attn_type，进入步骤 4 自定义；标准模型用现有两类即可。

#### 步骤 2：attention 类内定义 cache_entries

每个 attention 类的 `__init__` 内：

1. 定义 cache 占位 tensor：`self.k_cache = torch.Tensor([])`（框架在初始化时通过 `tensor_setter` 注入实际 storage）
2. 构造 `self.cache_entries` 列表，每个 cache 一个 `CacheEntry`

完整骨架（含 GQA / SlidingWindow / MLA 三种）见 `references/framework_kv_reference.md` §2。

> `num_head` 必须用 `num_key_value_heads_per_rank = max(num_kv_heads // attn_tp_size, 1)`，与 parallel-impl skill 改 `attn_tp_size` 时联动。MLA cache 通常 `num_head=1`。

> 模型自有 cache（非 KVCache 状态，如 N-gram window / hash buffer / speculative buffer 等）由 model class 内部自管，不上报 `cache_entries`，与框架 KV 解耦。

#### 步骤 3：模型类实现 get_cache_info

模型类（`XxxForCausalLM`）实现 `get_cache_info() -> ModelCacheInfo`，被 `ModelWorker._get_cache_info()` 调用：

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

模型类 `__init__` 必须保存 `self.block_size = infer_config.scheduler_config.block_size`。MTP 模型的 `self.model.layers` 是 `dict`，需用 `.values()` 遍历。含 dual / multi sublayer 的模型（每 logical layer 含多个 attention sublayer，如 LongCat 系列）：遍历应展开到 sublayer 粒度，`ModelCacheInfo.layer_infos` 数量与 sublayer 总数一致。**MLA 模型**返回 `ModelCacheInfo` 时必须显式设 `is_mla_backend=True`（默认 False；PD 传输等场景据此挑单 rank 传输 latent KV）。

完整示例见 `references/framework_kv_reference.md` §3。

#### 步骤 4：（可选）新增 SingleTypeKVCacheManager 子类

仅当现有 `ATTN_TYPE_MANAGER_MAP`（`FullAttention` / `SlidingWindow`）无法满足新 cache 类型时执行：

1. 在 `cann-recipes-infer/executor/core/kv_cache/single_type_kv_cache_manager.py` 新增子类
2. 注册到 `ATTN_TYPE_MANAGER_MAP`
3. 如新 attn_type 是「固定 block 数」类型（如类似 SlidingWindow），扩展 `calculate_block_num()` 加入 `FIXED_BLOCK_ATTN_TYPES` 列表，并实现 `calculate_fixed_block_memory_bytes()` 分支

详见 `cann-recipes-infer/docs/design/kv_cache_design.md §5.3 / §5.4` 和 `references/framework_kv_reference.md` §5。

#### 步骤 5：attention forward 接入 block_table / slot_mapping

attention 层 `forward()` 接收 `slot_mapping` / `block_table`（dict 类型），按 `self.attn_type` 取出对应 tensor：

```python
def forward(self, ..., slot_mapping=None, block_table=None, **kwargs):
    # 写入：用 slot_mapping[self.attn_type]
    self.update_cache(slot_mapping, key_states, value_states)

    # FA：传 block_table[self.attn_type]
    attn_output, _ = FA_OP(
        ..., block_table=block_table[self.attn_type], ...,
    )
```

模型顶层 forward 从 `forward_metadata.block_table` / `.slot_mapping` 取出整个 dict 透传给各层。完整示例（含 update_cache 写入算子选择）见 `references/framework_kv_reference.md` §4。

#### 步骤 6：验证三链路

按 `cann-recipes-infer/docs/design/kv_cache_design.md §5.6`：

1. **初始化链路**：`get_cache_info()` 返回的 `ModelCacheInfo` 字段正确，`KVCacheManager.allocate_cache_tensors()` 后每层 cache 已被注入 storage（不再是 `torch.Tensor([])` 占位）
2. **运行期链路**：forward 内 `block_table[self.attn_type]` 不为 None；FA 调用前后 dtype / shape 与 `cache_entries` 声明一致；Prefill 后第一个 Decode step 的 `actual_seq_lengths_kv` 等于 `prompt_length + 1`
3. **释放链路**：请求完成后 `KVCacheManager` 释放 block，下一请求复用 block 不出现脏数据；长跑无显存泄漏

### 3.2 独立部署模式实施

Runner 内自管 KV 数据结构、`block_table` 静态预分配、`slot_mapping` 动态计算、`forward_metadata` 构造。完整代码骨架见 `references/standalone_kv_reference.md`。

#### 步骤 1：选择 KV 模式

按第一层选型结果选模式：
- 模式一（连续缓存）：migrator 阶段最简骨架，无 paging，FA 直接读取整个 cache
- 模式二（手动 Paged 模式 + FA）：性能主路径，自管 `block_table` + `slot_mapping`
- 模式三（MLA absorb 压缩缓存）：MLA 模型，叠加在模式一或模式二之上，缓存 `cache_nope` + `cache_rope`

#### 步骤 2：Runner 内自管 KV tensor

在 `_init_kvcache(...)` 内一次性分配所有层的 KV cache：
- 模式一：每层 `(k, v)` BSH layout，shape `[batch, max_seq_len, num_kv_heads * head_dim]`
- 模式二：每层 `(k, v)` 物理 cache，shape `[total_blocks, block_size, num_kv_heads, head_dim]`，并构造 `self.block_table = arange().reshape(batch, -1).int().npu()`
- 模式三：每层 `(cache_nope, cache_rope)`，dim 远小于完整 KV

完整骨架见 `references/standalone_kv_reference.md` §1 / §2 / §3。

#### 步骤 3：Runner 内构造 slot_mapping（模式二/三）+ forward_metadata

每步推理前 Runner 调 `_build_slot_mapping_prefill / decode` 构造 `slot_mapping`，并组装 `ForwardMetaData`（含 `is_prefill` / `kv_len` / `slot_mapping` / `block_table` / `actual_seq_lengths_kv` / `actual_seq_lengths_q` / `attention_mask`）。

`ForwardMetaData` 直接 import 仓内 `executor.utils.forward_metadata.ForwardMetaData`，或参考字段在 runner 内 inline 一个轻量 dataclass。

完整骨架见 `references/standalone_kv_reference.md` §4。

#### 步骤 4：modeling 内 scatter_update_ + FA 调用

modeling forward 接收 `forward_metadata`（或自定义 `past_key_values + kv_len + attention_mask`），按 `is_prefill` 分支：
- 写入：`scatter_update_` 或 `npu_kv_rmsnorm_rope_cache`
- 读取：FA 算子调用（详见第二层 §2.7）

完整骨架（FA v1 / v2、TND / BSH / TND_NTD layout、MLA absorb）见 `references/standalone_kv_reference.md` 各章节代码块 + `references/fa-code-examples.md`。

#### 步骤 5：验证

1. Prefill 输出与基线对比（migrator 阶段已建立的基线）
2. Decode 单步耗时合理（与基线对比无显著回归）
3. 多卡场景：各 rank 输出形状一致，KV 切分无错位
4. 长跑无显存泄漏

---

## 高阶特性

> 以下特性仅供参考，超出本 skill 标准接入范围：

- **KV 量化（INT8 / W8A8C8）**：cache_entries.dtype 改 INT8 + FA 调用 dequant_scale 等改造由 model-infer-quantization skill 处理
- **KV 并行（KVP）**：KVCache 沿 head 维度切分到多卡，与 model-infer-parallel-impl skill 联动；参考 `cann-recipes-infer/models/longcat-flash/`
- **CPU-GPU Offload**：HBM 不足时 offload 到 CPU；`torch_npu.empty_with_swapped_memory` + 异步双流；参考 `cann-recipes-infer/models/hstu/modules/gpu_kv_cache_manager.py`
- **DiT Cache（扩散模型）**：TeaCache / FBCache / TaylorSeer；参考 `cann-recipes-infer/module/dit_cache/cache_method.py`

---

## 常见错误

| 错误模式 | 根因 | 预防 |
|------|------|------|
| Prefill 输出乱码但不报错 | `sparse_mode=0` + `mask=None`，无因果遮蔽 | 改用 `sparse_mode=3` + `[2048, 2048]` mask |
| MLA absorb Decode 输出严重偏差 | 误用 `sparse_mode=3` | 改 `sparse_mode=0`、`mask=None`（仅 MLA absorb Decode 路径切此配置；标准 LLM TND PA Prefill+Decode 统一用 sparse_mode=3） |
| 滑窗 Decode 长序列输出偏差 | 误把滑窗层改成 `sparse_mode=0`，导致看见窗口外 KV | 保留 `sparse_mode=4`，设置 `pre_tokens=sliding_window`, `next_tokens=0` |
| `atten_mask dtype` 报错 | mask 用了 float16/bfloat16 | `mask.to(torch.bool)` |
| `atten_mask shape` 报错 | mask 不是 `[2048, 2048]` | 固定用 `executor.utils.common_utils.get_init_attn_mask(2048, device)` |
| `scale` 默认 1.0 导致精度崩溃 | FA v1 用 `scale`，v2 用 `softmax_scale`，传错名静默生效 | 确认参数名与 FA 版本匹配（见 §2.7 v1/v2 映射表） |
| 框架部署：`block_table` 直接当 Tensor 传给 FA | 实际是 `Dict[attn_type → Tensor]`，需要按 `self.attn_type` 取 | `block_table=block_table[self.attn_type]` |
| 框架部署：`get_cache_info()` 漏字段 | `block_size` 未保存或 `layer_infos` 数量错 | 步骤 3 `__init__` 内保存 `self.block_size`，遍历层数与配置一致 |
| 独立部署：各层共享 `kv_len` 但内部递增 | Prefill 各层写入位置逐层偏移 | `kv_len` 是 Runner 层变量，各层只读不写（详见 §2.5） |
| 多 attn_type 混合：所有层用同一 manager 名 | gpt-oss 部分层 SlidingWindow / 部分层 FullAttention 写错 | 各层 `__init__` 内按条件设置 `attn_type` 字段 |
| Paged 模式 num_head 错配 | `attn_tp_size` 改动后 `cache_entries.num_head` 没更新 | 用 `max(num_kv_heads // attn_tp_size, 1)`，与 parallel-impl skill 联动 |

---

## 参考实现索引

| 实现模式 | 参考文件 | 搜索关键词 |
|---|---|---|
| 标准 GQA + FullAttention（含 MoE 变体） | `cann-recipes-infer/models/qwen/models/modeling_qwen.py`、`cann-recipes-infer/models/qwen3_moe/models/modeling_qwen3_moe.py` | `cache_entries`、`get_cache_info`、`num_key_value_heads_per_rank` |
| SlidingWindow + 混合 attn_type | `cann-recipes-infer/models/gpt_oss/models/modeling_gpt_oss.py` | `sliding_window`、`block_table[self.attn_type]` |
| MLA absorb + MTP | `cann-recipes-infer/models/deepseek_r1/models/modeling_deepseek.py` | `nope_cache`、`rope_cache`、`forward_absorb` |
| 长序列 + KVP（高级） | `cann-recipes-infer/models/longcat-flash/models/modeling_longcat_flash.py` | `kvp_size`、`BSND_NBSD` |
| 框架核心实现 | `cann-recipes-infer/executor/core/kv_cache/{cache_info,cache_utils,kv_cache_manager,single_type_kv_cache_manager,block_pool}.py` | `CacheEntry`、`prepare_block_tables`、`prepare_slot_mapping` |
| 设计文档 | `cann-recipes-infer/docs/design/kv_cache_design.md` | §4 辅助模块、§5 接入 Checklist |
| FA 完整代码示例 | `references/fa-code-examples.md` | — |
| 框架部署接入参考 | `references/framework_kv_reference.md` | — |
| 独立部署自管参考 | `references/standalone_kv_reference.md` | — |
