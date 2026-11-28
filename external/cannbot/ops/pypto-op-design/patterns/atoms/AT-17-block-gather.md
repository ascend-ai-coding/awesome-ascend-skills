---
type: pattern/atom
title: Block Table Gather (Paged KV)
description: 从分页 KV 缓存中按 block_table 索引拼装连续的 KV 块。
tags:
- gather
flow_pattern:
- V
examples:
- GLMAttention
- PageAttnFP8
- SparseCompressFA
---

## AT-17: Block Table Gather (Paged KV)

**描述**: 从分页 KV 缓存中按 block_table 索引拼装连续的 KV 块。

**CV 排布**: 纯 V

**计算流**:
```
# 在 loop 外分配拼装缓冲区
kj_assemble = tensor([s2_tile, D], dtype, "kj_assemble")

# 在 loop 内逐块拼装，这里不会真的产生搬运，因此没有时间消耗
for i in range(block_num):
    block_idx = block_table[batch_idx, logical_idx + i]
    block_idx_valid = max(block_idx, 0)           # 防止负索引
    kj_assemble[i*BS:(i+1)*BS, :] = view(k_cache, [BS, D], [block_idx_valid*BS, offset])

# 处理尾块 valid_shape
kj_assemble = view(kj_assemble, [s2_tile, D], valid_shape=[actual_len, D])
```

**⚠️ 实现警示**：本模式的结构机制是 view 拼装（零搬运），**禁止替换为 `gather_in_l1` / `gather_in_ub`**——后者是显式 GM→L1/UB 搬运指令，功能等价但机制冲突，会引入真实搬运开销。

**使用算子**: GLMAttention, PageAttnFP8



---
