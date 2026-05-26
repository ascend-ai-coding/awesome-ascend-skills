---
name: external-cannbot-model-model-infer-kvcache
description: 基于 PyTorch 框架的昇腾 NPU 模型推理 KVCache 优化技能。分析和优化 LLM 文本推理模型中的 KVCache 实现，包括连续缓存、分页注意力（Paged
  Attention）配合 FA 融合算子、MLA 压缩缓存。触发场景包括：KVCache 管理实现、paged attention、KV 压缩、FA 融合算子、OOM/性能问题、block_table/slot_mapping
  构造。基于本仓库已有模型的 KVCache 实现经验，按模型类型和场景推荐最佳方案。
original-name: model-infer-kvcache
synced-from: https://gitcode.com/cann/cannbot-skills
synced-date: '2026-05-26'
synced-commit: ac5bbd2b4cf427d011874e11f8d1e8b1bef66eda
license: UNKNOWN
---

# KVCache 优化技能

> 本技能按渐进式披露组织：快速选型 → 实现模式 → 深入原理 → 参考附录。
> 根据需要的深度选择阅读层级。

---

## 第一层：快速选型

根据模型类型，选择 KVCache 模式和参考实现：

| 模型类型 | 选择模式 | 参考实现 | 要点 |
|---------|---------|---------|------|
| 标准 LLM（MHA/GQA） | **模式一**：连续缓存 | gpt-oss, qwen3-moe | 最简单，无需 block_table |
| 高性能 LLM | **模式二**：分页注意力 + FA | deepseek-r1, kimi-k2 | 需构造 block_table + slot_mapping |
| MLA 架构（DeepSeek 系列） | **模式三**：MLA 压缩 | deepseek-r1, deepseek-v3.2 | 叠加在模式一或模式二之上 |
| 扩散/视频模型 | FA 直接计算 | hunyuan-video, wan2.2-i2v | 无 KVCache 分页，BNSD layout |

**决策路径**：
```
前置确认：从 progress.md 读取 Phase 0 已验证的架构类型（基于实际 config 值，非代码推断）
         │
需要分页注意力？ ──→ 否 → 模式一（连续缓存，scatter_update_ + FA）
                  └→ 是 → 模式二（分页注意力，block_table + slot_mapping + FA）
                           │
使用 MLA？ ──────→ 是 → 叠加模式三（cache_nope + cache_rope 压缩存储）
```

---

## 第二层：实现模式

### 模式一：连续缓存（非分页注意力）

KV 以连续 tensor 存储，`scatter_update_` 写入，FA 直接读取整个缓存。

**三步实现**：

```python
# 1. 初始化缓存（BSH layout 示例）
past_key = torch.zeros(batch_size, max_seq_len, num_kv_heads * head_dim, dtype=dtype, device="npu")
past_value = torch.zeros_like(past_key)

# 2. 写入缓存（每步调用）
torch_npu.scatter_update_(past_key, kv_len, key_states, 1)     # axis=1 for BSH
torch_npu.scatter_update_(past_value, kv_len, value_states, 1)

# 3. FA 注意力计算（FA v1 示例，不传 block_table）
attn_output, _ = torch.ops.npu.npu_fused_infer_attention_score(
    query_states, past_key_states, past_value_states,
    num_heads=num_heads, num_key_value_heads=num_kv_heads,
    input_layout="BSH",
    scale=1.0 / math.sqrt(head_dim),
    actual_seq_lengths_kv=actual_seq_lengths_kv,    # [batch_size] 每个 batch 的 KV 长度
    atten_mask=attention_mask,
    sparse_mode=0,                                   # Decode; Prefill 用 3(causal)
)
```

**改造边界**：只改 attention 计算（→FA）和 cache 存储（→scatter_update_）。上游（QKV projection、RoPE 等）保持不变，layout 不匹配时在接缝处 transpose/reshape 适配。

**参考文件**：`cann-recipes-infer/models/qwen3_moe/models/modeling_qwen3_moe.py`（BSH）、`cann-recipes-infer/models/gpt_oss/models/modeling_gpt_oss.py`（TND）

---

### 模式二：分页注意力 + FA 融合算子

KV 按固定大小 Block 存储，FA 通过 `block_table` 索引分块缓存。**PA 必须配合 FA 使用。**

**四步实现**：

```python
# 1. 初始化缓存和 block_table（一次性，推理全程不变）
block_size = 128
num_blocks_per_seq = max_seq_len // block_size
total_blocks = batch_size * num_blocks_per_seq

kv_cache = torch.zeros(total_blocks, block_size, num_kv_heads, head_dim, dtype=dtype, device="npu")
block_table = torch.arange(0, total_blocks).reshape(batch_size, -1).to(torch.int32).npu()
# block_table[b, i] = 第 b 个 batch 的第 i 个逻辑 block 对应的物理 block ID

# 2. 计算 slot_mapping（每步重算，公式见「深入原理」）
slot_mapping = kv_len + kv_len_offset    # Decode: [batch_size, 1]

# 3. 写入缓存
torch_npu.scatter_update_(kv_cache, kv_len, key_states, -2)
# 或融合写入: torch_npu.npu_kv_rmsnorm_rope_cache(..., slot_mapping, ...)

# 4. FA 注意力计算（FA v2 示例，传 block_table）
attn_output, _ = FA_OP(
    query, key_cache, value_cache,
    block_table=block_table,                        # 逻辑 block → 物理 block 映射
    block_size=block_size,                          # 块大小
    actual_seq_kvlen=actual_seq_lengths_kv,         # 每个 batch 的实际 KV 长度
    actual_seq_qlen=actual_seq_lengths_q,           # 每个 batch 的实际 Q 长度
    input_layout="TND_NTD",                         # PA 模式常用 layout
    sparse_mode=0,                                  # Decode=0, Prefill=3
    num_query_heads=num_heads, num_key_value_heads=num_kv_heads,
    softmax_scale=1.0 / math.sqrt(head_dim),
)
```

**参考文件**：`cann-recipes-infer/models/deepseek_r1/models/modeling_deepseek.py`、`cann-recipes-infer/models/deepseek-v3.2-exp/models/modeling_deepseek.py`

---

### 模式三：MLA 压缩缓存（叠加在模式一或模式二之上）

MLA 将 KV 压缩为低维 latent，只缓存压缩后的 `cache_nope`（非位置）和 `cache_rope`（位置），维度远小于原始 KV。

**核心差异**：

```python
# 缓存存储：压缩维度（kv_lora_rank=512）远小于完整 KV（num_heads * head_dim=16384）
cache_nope = torch.zeros(..., 1, kv_lora_rank, ...)     # 非位置部分
cache_rope = torch.zeros(..., 1, qk_rope_head_dim, ...)  # 位置编码部分

# FA 调用：key 和 value 传同一个 cache_nope（absorb 技术将 V 投影吸收到 O 投影中）
attn_output, _ = FA_OP(
    q_nope, k_nope_cache, k_nope_cache,   # key = value = cache_nope
    query_rope=q_pe, key_rope=k_rope_cache,  # RoPE 单独传入
    ...
)
```

**重要约束**：
- `query_rope` 和 `key_rope` 必须同时传或同时不传
- rope D 必须为 64
- MLA query D 仅支持 512 或 128
- MLA D=512 时仅支持 `sparse_mode` 为 0、3、4

**参考文件**：`cann-recipes-infer/models/deepseek_r1/models/modeling_deepseek.py`（`forward_absorb()`）

---

## 第三层：深入原理

> 以下内容在实现分页注意力（模式二）时需要深入理解。

### 3.1 物理内存布局与逻辑映射

分页注意力的核心是将 **逻辑地址**（batch_idx, seq_pos）映射到 **物理 block**。

```
KV 缓存物理存储: [total_num_blocks, block_size, num_kv_heads, head_dim]

假设 batch_size=2, max_seq_len=2048, block_size=128:
  每个 batch 需要: 2048 / 128 = 16 个 block
  总共需要: 2 × 16 = 32 个 block

  Block 0~15:  batch_0 的 token [0, 2047]
  Block 16~31: batch_1 的 token [0, 2047]
```

**映射公式**：
```
给定: batch_idx, seq_pos, block_size, block_table

逻辑 block 编号  = seq_pos // block_size
block 内偏移     = seq_pos % block_size
物理 block ID    = block_table[batch_idx, 逻辑 block 编号]
物理 slot 位置   = 物理 block ID × block_size + block 内偏移

示例: batch_idx=1, seq_pos=300, block_size=128
  逻辑 block = 300 // 128 = 2, 偏移 = 300 % 128 = 44
  物理 block = block_table[1, 2] = 18
  物理 slot  = 18 × 128 + 44 = 2348
```

### 3.2 block_table 构造

静态预分配，推理全程**不变**：

```python
# 参考: cann-recipes-infer/models/deepseek-v3.2-exp/models/modeling_deepseek.py (line 809-820)
num_blocks_per_seq = max_seq_len // block_size
total_blocks = batch_size * num_blocks_per_seq

block_table = torch.arange(0, total_blocks).reshape(batch_size, -1).to(torch.int32).npu()
# 结果: [[0,1,...,15], [16,17,...,31]]  shape=[batch_size, num_blocks_per_seq], dtype=int32
```

### 3.3 slot_mapping 构造

`slot_mapping` 是缓存**写入**位置索引，用于 `npu_kv_rmsnorm_rope_cache` 等写入算子。

**核心公式**：`slot(batch_idx, seq_pos) = batch_idx × max_seq_len + seq_pos`

> 在本仓库的顺序分配模式下，slot_mapping 等于展平后的线性索引，无需额外查表。

**Prefill** —— 每个 batch 写入多个 token：
```python
# 参考: cann-recipes-infer/models/deepseek-v3.2-exp/models/modeling_deepseek.py (line 2275-2282)
all_tensors = []
for i, seq_len in enumerate(kv_len):
    all_tensors.append(torch.arange(max_seq_len * i, seq_len.item() + max_seq_len * i, ...))
slot_mapping = torch.cat(all_tensors)

# 示例: kv_len=[512, 256], max_seq_len=2048
#   batch 0: [0, 1, ..., 511]
#   batch 1: [2048, 2049, ..., 2303]
#   拼接为 shape=[768] 的一维 tensor
```

**Decode** —— 每个 batch 写入 1 个 token：
```python
# 参考: cann-recipes-infer/models/deepseek-v3.2-exp/models/modeling_deepseek.py (line 2283-2284)
# 预计算偏移（一次性）
kv_len_offset = torch.arange(0, batch_size * max_seq_len, max_seq_len, ...).view(-1, 1)

# 每步计算
slot_mapping = kv_len.view(batch_size, -1) + kv_len_offset

# 示例: kv_len=[522, 266], kv_len_offset=[[0], [2048]]
#   slot_mapping = [[522], [2314]]
```

**与 block_table 的分工**：
- `slot_mapping` → 缓存**写入**算子（npu_kv_rmsnorm_rope_cache 等）
- `block_table` → FA **注意力读取**算子（npu_fused_infer_attention_score 等）
- 二者寻址逻辑一致，职责不同

### 3.4 actual_seq_lengths 构造

FA 算子需要每个 batch 的实际 KV/Q 长度，构造方式取决于 `input_layout`：

| | TND layout（多 batch token 拼一维） | BSH layout（各 batch 独立） |
|--|------|------|
| **Prefill KV** | `cumsum(kv_len)` → [512, 768] | `kv_len` → [512, 256] |
| **Prefill Q** | 同 KV | 同 KV |
| **Decode KV** | `kv_len` → [522, 266] | `kv_len` → [522, 266] |
| **Decode Q** | `cumsum([1,1])` → [1, 2] | `[1, 1]` |

```python
# 参考: cann-recipes-infer/models/deepseek-v3.2-exp/models/modeling_deepseek.py (line 2286-2301)
# TND Prefill 示例:
actual_seq_lengths_kv = torch.cumsum(kv_len, dim=0)    # [512, 768]
actual_seq_lengths_q = actual_seq_lengths_kv.clone()
```

### 3.5 kv_len 生命周期

`kv_len` 是驱动所有入参计算的核心变量：

```python
# Prefill: 从 attention_mask 计算
position_ids = attention_mask.long().cumsum(-1) - 1
kv_len = torch.max(position_ids, axis=1)[0] + 1   # 如 [512, 256]（具体是否 +1 视框架约定）

# 每次 Decode 后递增（在 Runner 层，model.forward() 之外，每个 forward pass 只做一次）
kv_len = kv_len + 1   # [513, 257], [514, 258], ...

# kv_len 驱动三个计算:
# 1. slot_mapping → 决定新 token 写到哪
# 2. actual_seq_lengths_kv → 告诉 FA 读多少 KV
# 3. position_ids → RoPE 位置编码
```

**kv_len 是 Runner 层变量，各层只读不写。** `kv_len` 作为参数传入所有 Transformer 层，`scatter_update_` 使用同一个 `kv_len` 写入每层各自的缓存。不应在 attention/cache 内部递增 kv_len——否则 Prefill 阶段各层写入位置逐层偏移，精度损坏。参考 `cann-recipes-infer/models/qwen3_moe/models/modeling_qwen3_moe.py`。

### 3.6 数据流总览

```
初始化（一次性）
  block_table = arange().reshape(batch, -1)   ← 推理全程不变
  kv_cache = zeros(total_blocks, block_size, heads, dim)
  kv_len_offset = arange(0, batch*max_seq_len, max_seq_len)
      │
      ▼
Prefill
  kv_len = 从 attention_mask 计算              ← 如 [512, 256]
  slot_mapping = [0..511, 2048..2303]          ← 多 token 写入
  actual_seq_lengths = cumsum(kv_len) 或 kv_len
  → 每层: 写入 kv_cache + FA 读取 kv_cache
      │
      ▼
Decode（循环，kv_len 递增时机视具体 Runner 实现）
  slot_mapping = kv_len + kv_len_offset        ← 写入位置
  actual_seq_lengths_kv = kv_len + 1           ← FA 需读到刚写入的 token
  → 每层: 写入 kv_cache + FA 读取 kv_cache
  kv_len += 1                                  ← forward 之后递增
```

---

## 完整代码示例

> 以上模式一/二/三给出了精简的实现骨架。仓库中各模型的完整 FA 调用示例见 [`references/fa-code-examples.md`](references/fa-code-examples.md)，覆盖：
>
> - 模式一（连续缓存）：GPT-OSS（TND, FA v2, sliding window）、Qwen3-MoE（BSH, FA v1）
> - 模式二+三（PA + MLA）：DeepSeek-R1（TND_NTD, FA v2, MLA absorb）、Kimi-K2（FA v1, Prefill/Decode 分离）、LongCat-Flash（BSND_NBSD, KVP）
> - 缓存写入融合算子：`npu_kv_rmsnorm_rope_cache`

---

## 附录 A：FA 融合算子版本对照

| 特性 | FA v1 (`npu_fused_infer_attention_score`) | FA v2 (`npu_fused_infer_attention_score_v2`) |
|-----|------------------------------------------|----------------------------------------------|
| 调用方式 | `torch.ops.npu.npu_fused_infer_attention_score(...)` | `self.fa_ops.npu_fused_infer_attention_score_v2(...)` |
| 量化 KV | `antiquant_mode` / `antiquant_scale` | `dequant_scale_key/value` + `query_quant_mode` |
| Sink token | 不支持 | `learnable_sink` 参数 |
| 典型使用 | Qwen3-MoE, LongCat-Flash, Kimi-K2 | DeepSeek-R1, GPT-OSS |

两个版本均支持：PA（传 block_table）/ 非 PA（不传）、MLA rope 分离（`query_rope` / `key_rope`）。

**v1/v2 关键参数名映射（混用不会报错，静默使用默认值导致精度异常）**：

| 功能 | FA v1 | FA v2 | 易错点 |
|------|-------|-------|--------|
| 缩放系数 | `scale` | `softmax_scale` | 默认 1.0，传错名精度崩溃 |
| Q head 数 | `num_heads` | `num_query_heads` | 传错名默认 1 |
| Q 长度 | `actual_seq_lengths` | `actual_seq_qlen` | v1 Decode 时忽略但仍需传 |
| KV 长度 | `actual_seq_lengths_kv` | `actual_seq_kvlen` | 名称完全不同 |
| KV head 数 | `num_key_value_heads` | `num_key_value_heads` | **相同** |

## 附录 B：input_layout 选择

| layout | Q 格式 | KV 格式 | 适用场景 |
|--------|--------|---------|---------|
| `"TND"` | [T, N, D] | [T, N, D] | 非 PA，flattened batch |
| `"BSH"` | [B, S, N*D] | [B, S, N*D] | 非 PA，标准 batch |
| `"BNSD"` | [B, N, S, D] | [B, N, S, D] | 非 PA，扩散模型 |
| `"TND_NTD"` | [T, N, D] | [N, T, D] | PA 模式，NZ 格式缓存 |
| `"BSND_NBSD"` | [B, S, N, D] | [N, B, S, D] | PA 模式，KVP 场景 |

## 附录 C：sparse_mode 与 atten_mask

FA 算子的 `sparse_mode` 决定注意力遮蔽方式，`atten_mask` 配合提供掩码。两者组合错误是精度问题的第一大来源。

### 参数配置规则

| sparse_mode | 含义 | atten_mask 要求 | 适用场景 |
|:-----------:|------|----------------|---------|
| 0 | Dense | 可选，通常传 **None** | **Decode**（标准），由 `actual_seq_lengths_kv` 控制有效长度 |
| 1 | allMask | **必传**完整矩阵 `(Q_S, KV_S)` | 特殊场景 |
| 2 | leftUpCausal | **不推荐**，建议改用 3 | — |
| 3 | Causal（标准因果） | **必传** `[2048, 2048]` bool 下三角 | **Prefill**（Decoder-only LLM 标准选择）；MTP Decode（sq>1） |
| 4 | Band（滑动窗口） | **必传** `[2048, 2048]` bool | Prefill / Decode（滑窗模型，如 gpt-oss，需配合 `pre_tokens`） |

### atten_mask 硬约束

- **dtype**：只允许 `torch.bool`（推荐）、`torch.int8`、`torch.uint8`。浮点类型直接报错
- **shape**：`sparse_mode=3/4` 时固定 `[2048, 2048]`，与 `max_position_embeddings` 无关。本仓库统一用 `get_init_attn_mask(2048, device)` 构造（见 `cann-recipes-infer/executor/utils/common_utils.py` 的 `get_init_attn_mask` 函数）
- **Decode 不需要 mask**：`sparse_mode=0` + `atten_mask=None`

### Prefill / Decode 标准配置

```python
if q_len > 1:  # Prefill（或 MTP Decode）
    sparse_mode = 3
    atten_mask = share_mask_tril     # [2048, 2048] bool
else:           # Decode
    sparse_mode = 0
    atten_mask = None
```
参考：`cann-recipes-infer/models/deepseek_r1/models/modeling_deepseek.py` 的 Prefill/Decode 分支逻辑

### 高频错误速查

| 症状 | 根因 | 修复 |
|------|------|------|
| Prefill 输出乱码但不报错 | `sparse_mode=0` + `mask=None`，无因果遮蔽 | 改用 `sparse_mode=3` + `[2048, 2048]` mask |
| Decode 输出严重偏差 | Decode 误用 `sparse_mode=3` | Decode 改 `sparse_mode=0`, `mask=None` |
| `atten_mask dtype` 报错 | mask 用了 float16/bfloat16 | `mask.to(torch.bool)` |
| `atten_mask shape` 报错 | mask 不是 `[2048, 2048]` | 固定用 `get_init_attn_mask(2048, device)` |
| `scale` 默认 1.0 导致精度崩溃 | FA v1 用 `scale`，v2 用 `softmax_scale`，传错名静默生效 | 确认参数名与 FA 版本匹配（见附录 A 映射表） |

## 附录 D：NPU KVCache 算子速查

| 算子 | 功能 | 接收入参 |
|-----|------|---------|
| `torch_npu.scatter_update_` | KV 按位置写入缓存 | kv_len |
| `torch_npu.npu_kv_rmsnorm_rope_cache` | 融合 RMSNorm+RoPE+Cache 写入 | slot_mapping |
| `torch_npu.npu_fused_infer_attention_score` | FA v1 融合注意力 | block_table |
| `torch_npu.npu_fused_infer_attention_score_v2` | FA v2 融合注意力 | block_table |

## 附录 E：Prefill vs Decode 参数差异

| 参数 | Prefill | Decode |
|-----|---------|--------|
| `sparse_mode` | 3（因果） | 0（dense）；MTP 时 sq>1 需用 3（因果） |
| `actual_seq_qlen` | 输入序列长度 | 1（单 token）；MTP 时为 next_n+1 |
| `actual_seq_kvlen` | = actual_seq_qlen | 累计的 KV 长度 |
| `slot_mapping` | 多 token 拼接 | 单 token 偏移 |
| `atten_mask` | 因果 mask（**dtype 必须为 bool/int8/uint8**）或 None | 通常 None |
| FA 参数配置 | Prefill 配置（sparse_mode=3 + mask） | Decode 配置（sparse_mode=0 + mask=None） |

---

## 高阶特性

> 以下特性仅供参考，尚未完整文档化。

- **KV 并行（KVP）**：KVCache 沿 head 维度切分到多卡。参考 `cann-recipes-infer/models/longcat-flash/`
- **KVCache 量化**：INT8/W8A8C8 量化。参考 `cann-recipes-infer/models/deepseek-v3.2-exp/models/offload_cache.py`
- **CPU-GPU Offload**：`torch_npu.empty_with_swapped_memory` + 异步双流。参考 `cann-recipes-infer/models/hstu/modules/gpu_kv_cache_manager.py`
- **扩散模型缓存（DiT Cache）**：TeaCache/FBCache/TaylorSeer。参考 `cann-recipes-infer/module/dit_cache/cache_method.py`

---

## 注意事项

1. **PA 必须配合 FA**：分页注意力必须通过 FA 融合算子的 `block_table` 参数索引，无法使用标准 softmax
2. **block_table 静态不变**：初始化后推理全程不修改，只有 kv_len 在递增
3. **slot_mapping 每步重算**：基于 kv_len + kv_len_offset，Prefill 和 Decode 公式不同
4. **block_table dtype 必须为 int32**
5. **Prefill/Decode 参数分离**：Prefill 和 Decode 需分别配置 FA 参数（sparse_mode、atten_mask、actual_seq_lengths 等）
6. **NZ 格式**：NPU 上 PA 模式推荐 NZ 格式存储 KVCache（`cache_mode="PA_NZ"`）
7. **inner_precise 行无效**：当 mask 存在整行全被遮蔽时（如自定义 Prefill mask），设置 `inner_precise=2` 开启行无效修正
