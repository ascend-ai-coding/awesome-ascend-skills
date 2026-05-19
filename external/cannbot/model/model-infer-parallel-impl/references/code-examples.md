# 并行层替换代码示例

> **API 差异**：当前仓库的 `ColumnParallelLinear`/`RowParallelLinear`/`QKVParallelLinear` 使用 `tp_size: int` + `tp_rank: int` 参数（不接受 `tp_group`）。以下示例中的 `tp_group=` 写法对应重构版 API；当前仓库需改为 `tp_rank=dist.get_rank(self.hccl_comm_dict["xxx_group"])`。`VocabParallelEmbedding` 在两个版本中都用 `tp_size + tp_rank`。

## Attention 层（当 attn_tp_size > 1）

| 原组件 | 替换为 | 通信组 | 说明 |
|-------|-------|--------|------|
| QKV Linear | `QKVParallelLinear` | `attn_tp_group` | 列切分，Q/K/V 按头数分 |
| O Linear | `RowParallelLinear` | `attn_tp_group` | 行切分，含 AllReduce |

```python
from module.linear import QKVParallelLinear, RowParallelLinear

# QKV 投影
self.qkv_proj = QKVParallelLinear(
    hidden_size=config.hidden_size,
    num_heads=config.num_attention_heads,
    num_key_value_heads=config.num_key_value_heads,
    head_dim=config.head_dim,
    tp_group=self.attn_tp_group,
    tp_size=self.attn_tp_size,
)

# O 投影
self.o_proj = RowParallelLinear(
    config.hidden_size, config.hidden_size,
    tp_group=self.attn_tp_group,
    tp_size=self.attn_tp_size,
)
```

**o_proj_tp_size 独立配置**（当 `o_proj_tp_size ≠ attn_tp_size` 时，如 MLA 模型）：

```python
self.o_proj = RowParallelLinear(
    config.hidden_size, config.hidden_size,
    tp_group=self.oproj_tp_group,  # 独立通信组
    tp_size=self.o_proj_tp_size,
)
```

---

## Dense FFN 层（当 dense_tp_size > 1）

| 原组件 | 替换为 | 通信组 | 说明 |
|-------|-------|--------|------|
| Gate Linear | `ColumnParallelLinear` | `dense_tp_group` | 列切分 |
| Up Linear | `ColumnParallelLinear` | `dense_tp_group` | 列切分 |
| Down Linear | `RowParallelLinear` | `dense_tp_group` | 行切分，含 AllReduce |

```python
from module.linear import ColumnParallelLinear, RowParallelLinear

self.gate_proj = ColumnParallelLinear(
    config.hidden_size, config.intermediate_size,
    tp_group=self.dense_tp_group,
    tp_size=self.dense_tp_size,
)
self.up_proj = ColumnParallelLinear(
    config.hidden_size, config.intermediate_size,
    tp_group=self.dense_tp_group,
    tp_size=self.dense_tp_size,
)
self.down_proj = RowParallelLinear(
    config.intermediate_size, config.hidden_size,
    tp_group=self.dense_tp_group,
    tp_size=self.dense_tp_size,
)
```

---

## Embedding / LMHead（当 embed_tp_size > 1 或 lmhead_tp_size > 1）

```python
from module.linear import VocabParallelEmbedding, ColumnParallelLinear

# Embedding（参数为 tp_size + tp_rank，无 tp_group）
self.embed_tokens = VocabParallelEmbedding(
    config.vocab_size,
    config.hidden_size,
    self.padding_idx,
    torch.bfloat16,
    tp_size=self.embed_tp_size,
    tp_rank=dist.get_rank(self.hccl_comm_dict["embed_tp_group"]) if self.embed_tp_size > 1 else 0,
)

# LMHead（当前仓库：tp_size + tp_rank，同 Embedding）
self.lm_head = ColumnParallelLinear(
    config.hidden_size,
    config.vocab_size,
    tp_size=self.lmhead_tp_size,
    tp_rank=dist.get_rank(self.hccl_comm_dict["lmhead_tp_group"]) if self.lmhead_tp_size > 1 else 0,
)
```

---

## 模块间数据重排（当相邻模块 TP 度不同时）

```python
# Embed(embed_tp=16) → Attention(attn_tp=1)
dist.all_gather_into_tensor(full_input, embed_output, group=embed_tp_group)

# Dense FFN(dense_tp=8) 的输入/输出
dist.all_gather_into_tensor(x_output, x, group=dense_tp_group)  # 输入聚合
# ... FFN 计算 ...
dist.reduce_scatter_tensor(mlp_res, down_proj, group=dense_tp_group)  # 输出分散
```

参考实现：
- `cann-recipes-infer/models/longcat-flash/models/modeling_longcat_flash.py`（搜索 `all_gather_into_tensor` 和 `reduce_scatter_tensor`）
- `cann-recipes-infer/models/deepseek_r1/models/modeling_deepseek.py`（搜索同上）
