# 并行层替换代码示例

> **API 速查**：`cann-recipes-infer/module/linear.py` 中的 `ColumnParallelLinear` / `RowParallelLinear` / `QKVParallelLinear` / `VocabParallelEmbedding` 均使用 `tp_size: int` + `tp_rank: int` 参数（不接受 `tp_group`）。通信组对象通过 `comm_manager.get_group("xxx_group")` 获取，用于显式 collective（all_gather / reduce_scatter 等）。
>
> `RowParallelLinear.forward` **不内置 AllReduce**，模型层在调用后显式 `dist.all_reduce(out, group=tp_group)`（参考 `cann-recipes-infer/models/qwen/models/modeling_qwen.py` 的 `down_proj` 后写法）。

## Attention 层（当 attn_tp_size > 1）

| 原组件 | 替换为 | 通信组 | 说明 |
|-------|-------|--------|------|
| QKV Linear | `QKVParallelLinear` | `attn_tp_group` | 列切分，Q/K/V 按头数分 |
| O Linear | `RowParallelLinear` | `attn_tp_group` | 行切分，调用方负责 AllReduce |

```python
from module.linear import QKVParallelLinear, RowParallelLinear

attn_tp_rank = comm_manager.get_rank("attn_tp_group")

# QKV 投影
self.qkv_proj = QKVParallelLinear(
    hidden_size=config.hidden_size,
    head_size=config.head_dim,
    total_num_heads=config.num_attention_heads,
    total_num_kv_heads=config.num_key_value_heads,
    tp_size=self.attn_tp_size,
    tp_rank=attn_tp_rank,
)

# O 投影
self.o_proj = RowParallelLinear(
    config.hidden_size, config.hidden_size,
    tp_size=self.attn_tp_size,
    tp_rank=attn_tp_rank,
)
```

**o_proj_tp_size 独立配置**（当 `o_proj_tp_size ≠ attn_tp_size` 时，如 MLA 模型）：

```python
oproj_tp_rank = comm_manager.get_rank("oproj_tp_group")

self.o_proj = RowParallelLinear(
    config.hidden_size, config.hidden_size,
    tp_size=self.o_proj_tp_size,
    tp_rank=oproj_tp_rank,
)
```

---

## Dense FFN 层（当 dense_tp_size > 1）

| 原组件 | 替换为 | 通信组 | 说明 |
|-------|-------|--------|------|
| Gate Linear | `ColumnParallelLinear` | `dense_tp_group` | 列切分 |
| Up Linear | `ColumnParallelLinear` | `dense_tp_group` | 列切分 |
| Down Linear | `RowParallelLinear` | `dense_tp_group` | 行切分，调用方负责 AllReduce |

```python
from module.linear import ColumnParallelLinear, RowParallelLinear

dense_tp_rank = comm_manager.get_rank("dense_tp_group")

self.gate_proj = ColumnParallelLinear(
    config.hidden_size, config.intermediate_size,
    tp_size=self.dense_tp_size,
    tp_rank=dense_tp_rank,
)
self.up_proj = ColumnParallelLinear(
    config.hidden_size, config.intermediate_size,
    tp_size=self.dense_tp_size,
    tp_rank=dense_tp_rank,
)
self.down_proj = RowParallelLinear(
    config.intermediate_size, config.hidden_size,
    tp_size=self.dense_tp_size,
    tp_rank=dense_tp_rank,
)
```

---

## Embedding / LMHead（当 embed_tp_size > 1 或 lmhead_tp_size > 1）

```python
from module.linear import VocabParallelEmbedding, ColumnParallelLinear

# Embedding
self.embed_tokens = VocabParallelEmbedding(
    config.vocab_size,
    config.hidden_size,
    self.padding_idx,
    torch.bfloat16,
    tp_size=self.embed_tp_size,
    tp_rank=comm_manager.get_rank("embed_tp_group"),
)

# LMHead
self.lm_head = ColumnParallelLinear(
    input_size=config.hidden_size,
    output_size=config.vocab_size,
    bias=False,
    tp_size=self.lmhead_tp_size,
    tp_rank=comm_manager.get_rank("lmhead_tp_group"),
)
```

---

## 模块间数据重排（当相邻模块 TP 度不同时）

```python
embed_tp_group = comm_manager.get_group("embed_tp_group")
dense_tp_group = comm_manager.get_group("dense_tp_group")

# Embed(embed_tp=16) → Attention(attn_tp=1)
dist.all_gather_into_tensor(full_input, embed_output, group=embed_tp_group)

# Dense FFN(dense_tp=8) 的输入/输出
dist.all_gather_into_tensor(x_output, x, group=dense_tp_group)  # 输入聚合
# ... FFN 计算 ...
dist.reduce_scatter_tensor(mlp_res, down_proj, group=dense_tp_group)  # 输出分散
```

参考实现：
- `cann-recipes-infer/models/qwen3_moe/models/modeling_qwen3_moe.py`、`cann-recipes-infer/models/gpt_oss/models/modeling_gpt_oss.py`（标准 TP 替换）
- `cann-recipes-infer/models/deepseek_r1/models/modeling_deepseek.py`（MLA + 模块间数据重排 + EP）
