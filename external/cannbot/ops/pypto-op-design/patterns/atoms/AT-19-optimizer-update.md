---
type: pattern/atom
title: Optimizer Update (Adam/RMSProp/SGD 族)
description: 优化器状态更新模板。覆盖整个"动量累积 + 参数衰减 + 参数写回"算子族，最常见实例是 AdamW 和 RMSProp。
tags:
- optimizer-step
flow_pattern:
- V
examples:
- ApplyAdamWV2
- ApplyRMSProp（泛化：Lion/LAMB/AdaFactor）
---

## AT-19: Optimizer Update (Adam/RMSProp/SGD 族)

**描述**: 优化器状态更新模板。覆盖整个"动量累积 + 参数衰减 + 参数写回"算子族，最常见实例是 AdamW 和 RMSProp。

**CV 排布**: 纯 V

**计算流 (AdamW 实例)**:
```
# 一阶动量
m_new = add(mul(m, beta1), mul(grad, 1-beta1))
m_hat = div(m_new, bias_correction1)

# 二阶动量
v_new = add(mul(v, beta2), mul(mul(grad, grad), 1-beta2))
v_hat = div(v_new, bias_correction2)

# 参数更新
denom = add(sqrt(v_hat), eps)
update = add(div(m_hat, denom), mul(w, weight_decay))
w_new = sub(w, mul(update, lr))
```

**计算流 (泛化骨架)**:
```
# === 1. 动量/状态更新（按变体配置）===
state_new = state_update_fn(state_old, grad, hyperparams)

# === 2. 偏置修正（可选）===
state_hat = bias_correction_fn(state_new, step)

# === 3. 自适应学习率（可选）===
adaptive_lr = adaptive_lr_fn(state_hat)

# === 4. 权重衰减（可选）===
decay_term = mul(w, weight_decay)

# === 5. 参数更新 ===
w_new = sub(w, mul(adaptive_lr + decay_term, lr))

# === 6. 状态写回（in-place）===
state[:] = state_new
```

**泛化变体（适用算子族）**:

| 变体 | 一阶状态 | 二阶状态 | 自适应 LR | 衰减形式 | 典型实现 |
|------|---------|---------|-----------|---------|---------|
| **AdamW**（当前样本） | `m = β₁m + (1-β₁)g` | `v = β₂v + (1-β₂)g²` | `m_hat / (√v_hat + eps)` | decoupled `w*wd` | ApplyAdamWV2 |
| **RMSProp**（当前样本） | 无 | `v = β v + (1-β)g²` | `g / (√v + eps)` | 可选 `w*wd` | ApplyRMSProp |
| **SGD with Momentum** | `m = μm + g` | 无 | 无 | 可选 `w*wd` | 训练框架常见 |
| **AdaGrad** | 无 | `v = v + g²` (累加) | `g / (√v + eps)` | 可选 | 稀疏特征 |
| **Lion** | `m = β m + (1-β) g` | 无 | `sign(β₂m + (1-β₂)g)` | decoupled | Google Lion |
| **LAMB / LARS** | `m = β₁m + (1-β₁)g` | `v = β₂v + (1-β₂)g²` | `m_hat/(√v_hat+eps)` + layer-wise scale | 可选 | 大 batch 训练 |
| **AdaFactor** | 无（行/列各一） | 行/列分离 `v_r, v_c` | rank-1 重构 | decoupled | 显存优化 |
| **Muon / Sophia** | `m = β m + (1-β)g` | 二阶信息（Hessian 近似） | 矩阵预处理（牛顿步） | 可选 | 新兴优化器 |

**编程约束（PyPTO 特化）**:
- 全程 FP32 计算，参数最终 cast 回 BF16
- 必须用 `submit_before_loop=True` 实现 pipeline overlap，否则带宽利用不满
- 必须配 `set_cache_policy(NONE_CACHEABLE)`，参数过大会污染 L2 cache
- 沿参数维度（K 轴）分块，N_TILE=2048 是经验值；过小 prologue 占比高，过大 UB 装不下
- 多状态（m, v, w, grad）同时存在时，UB 容量约束更紧，TILE 需相应缩小

**使用算子**: ApplyAdamWV2, ApplyRMSProp（其他变体可参考本骨架实现）



---
