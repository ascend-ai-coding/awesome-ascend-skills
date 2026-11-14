---
type: pattern/atom
title: Paged Cache Scatter/Update
description: 将计算结果写入分页 KV 缓存的指定物理位置。
tags:
- scatter
flow_pattern:
- V
examples:
- GLMAttnFusion
- MLAPrologQuant
- Compressor
---

## AT-16: Paged Cache Scatter/Update

**描述**: 将计算结果写入分页 KV 缓存的指定物理位置。

**CV 排布**: 纯 V

**计算流**:
```
# 将 cache_index 映射到物理 block 位置
physical_idx = block_table[batch_idx, logical_block]
cache_4d = reshape(cache, [total_blocks, block_size, N, D])
src_4d = reshape(src, [1, 1, N, D])

# scatter_update: 在 axis=-2 方向将 src 写入 cache 的 physical_idx 位置
scatter_update(cache_4d, axis=-2, index=physical_idx, src=src_4d)
cache_4d.move()   # 确保写回
```

**使用算子**: GLMAttnFusion, MLAPrologQuant



---
