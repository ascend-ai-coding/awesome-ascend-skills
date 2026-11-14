---
type: pattern/atom
title: Expert Gating (TopK Selection)
description: MoE 场景下的专家选择，包含组级 TopK + 专家级 TopK 两阶段。
tags:
- routing
flow_pattern:
- V
examples:
- GLMSelectExperts
- GLMMoEFusion
---

## AT-15: Expert Gating (TopK Selection)

**描述**: MoE 场景下的专家选择，包含组级 TopK + 专家级 TopK 两阶段。

**CV 排布**: 纯 V

**计算流**:
```
scores = sigmoid(logits) + bias

# 阶段1: 组级选择
group_scores = amax(reshape(scores, [batch, num_groups, group_unit]), dim=-1)
topk_group_ids = topk(group_scores, topk_group)
mask = scatter_(zeros, 1, topk_group_ids, 1.0)  # 组掩码
masked_scores = where(logical_not(mask), 0.0, scores)

# 阶段2: 专家级选择
topk_ids = topk(masked_scores, topk)
topk_weights = gather(scores, topk_ids)
[可选] topk_weights = topk_weights / sum(topk_weights)  # 重归一化
```

**使用算子**: GLMSelectExperts, GLMMoEFusion (内联)



---
