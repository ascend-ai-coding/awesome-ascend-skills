---
type: pattern/atom
title: Gated Recurrent Update
description: 循环结构中的门控状态更新，如 LSTM 的 f/i/o 门控和 Delta Rule 的门控衰减。
tags:
- recurrent-update
flow_pattern:
- V
- C
- V
- C
examples:
- GatedDeltaRule
- SumLSTM
---

## AT-14: Gated Recurrent Update

**描述**: 循环结构中的门控状态更新，如 LSTM 的 f/i/o 门控和 Delta Rule 的门控衰减。

**CV 排布**: 纯 V（LSTM）/ CVC（Delta Rule）

### LSTM 变体:
```
# 门控
f = sigmoid(pre_f)
i = sigmoid(pre_i)
o = sigmoid(pre_o)
c_act = gelu(rms_norm(pre_c))    # gelu ≈ x * sigmoid(1.702x)

# 状态更新
c_new = add(mul(prev_c, f), mul(c_act, i))
h_new = mul(gelu(rms_norm(c_new)), o)
```

### Delta Rule 变体:
```
# 门控衰减
gate_cum = cumsum(gate)          # 累积门控
decay = exp(gate_cum - gate_cum^T) * causal_mask

# 状态更新 (块级)
A = key*beta @ key^T * decay
A_inv = block_inverse(A)         # 分块求逆
state_new = state * last_decay + value_update - correction
```

**特征维度标签**:
- 跨 loop 状态更新: 状态变量 `prev_state` 跨迭代传递
- Loop 特征: 序列维度的串行依赖



---
